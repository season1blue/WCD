import os
import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms  # 如果你有图像预处理需求
import re
import ipdb
import torch


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
        if task == "textvqa":
            self.data = self.load_data(question_path)['data']
        else:
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
            
            
            
class TrainDataset(Dataset):
    def __init__(self, task, question_path, image_path, processor, max_samples=500):
        self.task = task
        self.processor = processor
        self.image_path = image_path

        self.data = self.load_data(question_path)

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
        image = Image.open(item["image_path"]).convert("RGB")

        if self.task == "gqa":
            text = item["question"]
            question = item["question"]
            answer = item.get("answer", "")
            task_description = ""
            
        elif self.task == "textvqa":
            text = item["question"]    
            question = item["question"]
            answer = item["one_answer"]
            task_description = ""
            
        elif self.task in ["mvsa_m", "mvsa_s"]:
            text = clean_text(item["text"].replace("\n", " ").strip())
            question = text
            answer = item.get("new_sentiment", "")
            task_description = "Classify the Multimodal sentiment label (negative, neutral, positive). Provide a short answer with 1 label for the Multimodal label."
        
        else:
            text = ""
            question = item.get("question", "")
            answer = item.get("answer", "")
            task_description = ""

        full_prompt = f"""
            Consider the following Image and Text: 
            Image: <image> Text: {question} <pad>
            {task_description}
            ASSISTANT: {answer}
            """
        inputs = self.processor(full_prompt, image, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        # inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        answer_labels = self.processor.tokenizer(answer, return_tensors="pt")["input_ids"]

        # labels = self.processor.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).input_ids[0]
        # prefix_len = len(self.processor.tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
        # labels[:prefix_len] = -100

        # inputs["labels"] = labels
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"][0],
            "labels": labels[0],
            "answer_labels": answer_labels
        }
    
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