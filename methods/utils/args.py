
import argparse
import torch
import os

task_to_image_path = {
    "aokvqa": "./data/aokvqa/images",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/images",
    "mvsa_m": "/ai/teacher/ssz/all_data/msa/MVSA_Multiple/data",
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/data",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/train_val",
    "docvqa": "./data/docvqa/images",
    "pope": "./data/pope/images",
    "vstar": "./data/vstar/images",
    "vqav2": "./data/vqav2/images",
}

task_to_question_path = {
    "aokvqa": "./data/aokvqa/data.json",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/balanced_evalset.json",
    # "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/eval_random_5000.json",
    "mvsa_m": "/ai/teacher/ssz/all_data/msa/MVSA_Multiple/test.json",
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/new_test_mix.json",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/processed/val_mix.json",
    "docvqa": "./data/docvqa/data.json",
    "pope": "./data/pope/data.json",
    "vstar": "./data/vstar/data.json",
    "vqav2": "./data/vqav2/data.json",
}


task_to_train_path = {
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/new_train_mix.json",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/balanced_trainset.json",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/processed/train_mix.json",
}

model_to_fullname = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "qwen2_5": "Qwen/Qwen2.5-VL-3B-Instruct",
    "llava_new": "liuhaotian/llava-v1.5-7b"
}

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", choices=model_to_fullname.keys())
    parser.add_argument("--task", type=str, default="textvqa", choices=task_to_question_path.keys())
    parser.add_argument("--method", type=str, default="new", choices=["rel_att", "pure_grad", "grad_att", "grad", "rel_att_high", "pure_grad_high", "grad_att_high", "grad_high"])
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--max_sample", type=int, default=500)  # -1代表所有数据全跑
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_output_dir", type=str, default="results/ckpts")
    parser.add_argument("--lora_name", type=str, default="")
    
    parser.add_argument("--json_file", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--is_eval", type=int, default=0)
    
    
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    output_name = f'{args.model}-{args.task}-{args.method}-{args.lora_name}'
    args.model_id = model_to_fullname[args.model]

    # args.output_path = os.path.join(args.save_path, "jsons", f"{output_name}.json")
    # args.ckpt_path = os.path.join(args.save_path, "ckpts", f"{output_name}")

    args.image_path = task_to_image_path[args.task]
    args.question_path = task_to_question_path[args.task]
    args.train_path = task_to_train_path[args.task]
    
    return args
