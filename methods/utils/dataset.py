import os
import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms  # 如果你有图像预处理需求
import re
import ipdb
import torch
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.train.train import LazySupervisedDataset, DataArguments,  tokenizer_image_token
import copy
from llava import conversation as conversation_lib
import transformers
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 可选
])


def clean_text(text: str) -> str:
    # 1. 移除 "RT @用户名"
    text = re.sub(r'\bRT\s*@\w+\s*', '', text)
    
    # 2. 移除所有以 http 开头的 URL
    text = re.sub(r'http\S+', '', text)
    # 3. 移除冒号（中英文）
    text = re.sub(r'[:：]', '', text)

    # 可选：去掉多余的空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class ImageTextDataset(Dataset):
    def __init__(self, task, question_path, image_path, max_samples=500):
        self.task = task
        self.data = self.load_data(question_path)
            
        if max_samples != -1:
            self.data = self.data[:max_samples]  # 截断前N条
        self.image_path = image_path
        self.transform = transform

        # 补全每条数据的 image 路径
        for d in self.data:
            if "image_path" in d:
                d["image_path"] = os.path.join(image_path, d["image_path"])
            elif "imageId" in d:
                d["image_path"] = os.path.join(image_path, f"{d['imageId']}.jpg")
            elif "image_id" in d:
                d["image_path"] = os.path.join(image_path, f"{d['image_id']}.jpg")
            
    def load_data(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            return list(load_dataset(path)['test'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # image = Image.open(item["image_path"]).convert("RGB")

        # if self.transform:
        #     image = self.transform(image)
        if self.task == "gqa":
            return {
                "image_path": item["image_path"],
                "text": item["question"],
                "answer": item["answer"],  # 防止部分缺失
                # "qid": item.get("qid", idx)
            }
        elif self.task == "textvqa":
            return {
                "image_path": item["image_path"],
                "text": item["question"],
                "answers": tuple(item["answers"]),  # 防止部分缺失
                # "qid": item.get("qid", idx)
            }
        elif self.task in ["mvsa_m", "mvsa_s"]:
            text = clean_text(item["text"])
            
            text = f"Text: '{text}'. Confirm the sentiment from the choice (positive, negative, neutral). "
            return {
                "id": item["id"],
                "image_path": item["image_path"],
                "text": text,
                # "answer": item["sentiment"],  # 防止部分缺失
                "new_answer": item["new_sentiment"],  # 防止部分缺失
                # "qid": item.get("qid", idx)
            }
            


@dataclass
class CustomDataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning with image support."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # 截断处理，确保不超过最大长度
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images  # fallback if shapes are inconsistent

        return batch
    

def preprocess_v1(sources, tokenizer, has_image: bool = False):

    
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
    
    
def preprocess_v2(sources, tokenizer, has_image: bool = False):
    conv_template = conversation_lib.default_conversation
    roles = {"user": conv_template.roles[0], "assistant": conv_template.roles[1]}

    conversations = []
    
    conv = conv_template.copy()
    conv.messages = []

    # 
    
    conv.append_message(roles["user"], sources["question"])
    conv.append_message(roles["assistant"], sources["answer"])

    conversations.append(conv.get_prompt())

    # Tokenize prompts
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(conversations, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True,).input_ids

    targets = input_ids.clone()

    # Target masking logic
    sep = conv_template.sep + conv_template.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv_template.sep2)
        cur_len = 1  # BOS

        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # prompt part (including role label)

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)"
                )

    
    
    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
    )



            
class TrainDataset(Dataset):
    def __init__(self, task, question_path, image_path, tokenizer, image_processor, max_samples=500):
        self.task = task
        self.tokenizer= tokenizer
        self.image_processor = image_processor
        self.image_path = image_path

        self.image_aspect_ratio = 'square'
        
        self.data_args = DataArguments(
            data_path=question_path,
            lazy_preprocess=True,
            is_multimodal=True,
            image_folder=image_path,
            image_aspect_ratio="pad"  # 或 "square"
        )

        with open(question_path, "r") as f:
            self.data = json.load(f)


        if max_samples != -1:
            self.data = self.data[:max_samples]

        # 补全每条数据的 image 路径
        for d in self.data:
            if "image_path" in d:
                d["image_path"] = os.path.join(image_path, d["image_path"])
            elif "imageId" in d:
                d["image_path"] = os.path.join(image_path, f"{d['imageId']}.jpg")
            elif "image_id" in d:
                d["image_path"] = os.path.join(image_path, f"{d['image_id']}.jpg")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        print(item['question'], item["image_path"])

        image = Image.open(item["image_path"]).convert("RGB")
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image = image.to(torch.bfloat16)

        source = copy.deepcopy(item)
        source["question"] = f"{DEFAULT_IMAGE_TOKEN}\nAnswer the question using a single word or phrase\n{item['question']}"

        if self.task == "gqa":
            source["answer"] = item.get("answer", "")
            task_description = ""
        elif self.task == "textvqa":
            source["answer"] = item["one_answer"]
            task_description = ""
        elif self.task in ["mvsa_m", "mvsa_s"]:
            source["answer"] = item.get("new_sentiment", "")
            task_description = "Classify the Multimodal sentiment label (negative, neutral, positive). Provide a short answer with 1 label for the Multimodal label."
        else:
            source["answer"] = item.get("answer", "")
            task_description = ""

        data_dict = preprocess_v2(source, self.tokenizer, has_image=('image_path' in source))
        data_dict['image'] = image
        # data_dict: input_ids, labels, image

        
        return data_dict
    
    @staticmethod
    def collate_fn(batch):
        max_len = max([len(b["input_ids"]) for b in batch])
        pad_token_id = 0

        def pad_to_max(seq_list, pad_val):
            return torch.stack([
                torch.cat([s, torch.full((max_len - len(s),), pad_val, dtype=s.dtype)]) if len(s) < max_len else s[:max_len]
                for s in seq_list
            ])

        input_ids = pad_to_max([b["input_ids"] for b in batch], pad_token_id)
        attention_mask = pad_to_max([b["attention_mask"] for b in batch], 0)
        labels = pad_to_max([b["labels"] for b in batch], -100)

        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        answer_labels = [b["answer_labels"] for b in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            # "answer_labels": answer_labels
        }