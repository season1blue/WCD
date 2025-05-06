
import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoConfig
from methods.my_llava import MyLlava
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



def _eval(args, epoch=None, model=None, processor=None):
    """
    Main function to run the visual cropping and question answering pipeline.
    
    This function loads the specified model and processor, processes the dataset,
    applies the visual cropping and question answering to each data point,
    and saves the results to a JSON file.
    
    Args:
        args: An argparse.Namespace object containing the following attributes:
            - model: String indicating which model to use ("llava" or "blip")
            - model_id: The model identifier for loading from HuggingFace
            - device: The device to run the model on ("cuda" or "cpu")
            - question_path: Path to the question dataset
            - image_path: Path to the directory containing images
            - task: The task identifier
            - method: The attention method to use
            - output_path: Path to save the results
            - total_chunks: Total number of chunks to split the dataset into
            - chunk_id: The ID of the current chunk to process
            
    Returns:
        None: Results are saved to the specified output file
    """

    if model is None:
        config = AutoConfig.from_pretrained(args.model_id)
        model = MyLlava.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, attn_implementation="eager", config=config).to(args.device)
        if args.lora_name != "NONE":
            ckpt_path = os.path.join("/ai/teacher/ssz/layer_task/mllms_know/results/ckpts", args.lora_name)
            model = PeftModel.from_pretrained(model, ckpt_path, adapter_file="adapter_model.safetensors")
        processor = AutoProcessor.from_pretrained(args.model_id) 
        

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
        short_questions = dd["short_question"] if 'short_question' in dd else questions

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        model.eval()

        multi_prompts = [
            f"<image>\nUSER: {q} Answer the question using a single word or phrase.\nASSISTANT:"
            for q in questions
        ]

        multi_inputs = processor(multi_prompts, images, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        
        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        
        multi_generations = [
            i.split('ASSISTANT: ')[1]
            for i in processor.batch_decode(multi_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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