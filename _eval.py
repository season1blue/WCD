
import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoConfig
import argparse
from tqdm import tqdm
import json
from datasets import load_dataset

from methods.utils.utils import *
from methods.utils.args import load_args

from peft import PeftModel
from methods.utils.dataset import ImageTextDataset
from torch.utils.data import Dataset, DataLoader
from methods.utils.get_score import *
import ipdb
from datetime import datetime


def custom_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        # 保持 answers 原始结构
        if key == "answers":
            batch_out[key] = values  # List[Tuple[str, ...]]
        else:
            batch_out[key] = values  # 或进一步处理其他字段
    return batch_out



from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def _eval(args, epoch=None, model=None, processor=None):

    if model is None:

        model_path = args.model_id
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name)


        if args.lora_name != "NONE":
            ckpt_path = os.path.join("/ai/teacher/ssz/layer_task/mllms_know/results/ckpts", args.lora_name)
            model = PeftModel.from_pretrained(model, ckpt_path, adapter_file="adapter_model.safetensors")
        

    dataset = ImageTextDataset(
        task=args.task,
        question_path=args.question_path,
        image_path=args.image_path,
        max_samples=args.max_sample
    )
    
    if args.task == "textvqa":
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
            # num_workers=1
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            # num_workers=1
        )

    new_datas = []

    
    for dd in tqdm(dataloader, desc="Processing", ncols=100):
        questions = dd["text"]  # batch_size个问题
        image_paths = dd["image_path"]  # batch_size个图片路径


        prompts = []
        input_ids = []
        for qs in questions:
            # prompts.append(f"<image>\nUSER: {qs} Answer the question using a single word or phrase.\nASSISTANT:")
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + 'Answer the question using a single word or phrase \n' + qs
            else:
                qs =  DEFAULT_IMAGE_TOKEN + '\n' + 'Answer the question using a single word or phrase \n' + qs

            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
            # input_ids.append(input_id)
            input_ids = input_id.unsqueeze(0)

        # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

        image = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        image_tensor = process_images(image, image_processor, model.config)
        
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                image_sizes=[image[0].size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                cache_position=None,
                max_new_tokens=1024,
                use_cache=True)

        multi_generations = [
            i.strip()
            for i in tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ]

        if args.task in ["mvsa_m", "mvsa_s"]:
            # 把batch里的每个样本分别补充
            for i in range(len(questions)):
                new_d = {k: v[i] for k, v in dd.items()}  # 把第i个样本单独取出
                new_d["gen_answer"] = multi_generations[i]
                new_d['id'] = new_d['id'].item() 
                new_datas.append(new_d)
        else:
            for i in range(len(questions)):
                new_d = {k: v[i] for k, v in dd.items()}  # 把第i个样本单独取出
                new_d["gen_answer"] = multi_generations[i]
                new_datas.append(new_d)

    output_path = os.path.join(args.save_path, "jsons", args.lora_name+ ".json")    
    print("evaluation output to", output_path)
    
    with open(output_path, "w") as f:
        json.dump(new_datas, f, indent=4)


    if args.task in ["mvsa_m", "mvsa_s"]:
        acc, _ = evaluate_mvsa(new_datas)
    elif args.task == "gqa":
        acc, _ = evaluate_gqa(new_datas)
    elif args.task == "textvqa":
        acc, _ = evaluate_textvqa(new_datas)

        
    result = {
        'version': args.lora_name,
        'epoch': epoch,
        'time': datetime.now().strftime("%Y-%m-%dd %H:%M:%S"),
        'acc': acc,
    }
    report_path = os.path.join(args.save_path, "result_all.json")

    print(f"Acc: {acc} write into {report_path}")
    with open(report_path, 'a') as f:
        json.dump(result, f, indent=4)
        f.write('\n')  # 添加换行符

if __name__ == "__main__":
    args = load_args()
    _eval(args)