
import argparse
import torch
import os

task_to_image_path = {
    "aokvqa": "./data/aokvqa/images",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/images",
    "mvsa_m": "/ai/teacher/ssz/all_data/msa/MVSA_Multiple/data",
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/data",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/train_val",
    "docvqa": "/ai/teacher/ssz/all_data/mqa/docvqa",
    "okvqa": "/ai/teacher/ssz/all_data/mqa/OKVQA/val2014",
    "vizwiz": "/ai/teacher/ssz/all_data/mqa/vizwiz/val",
    "vqav2": "/ai/teacher/ssz/all_data/mqa/VQAv2/val2014",
    "pope": "./data/pope/images",
    "vstar": "./data/vstar/images",
    
}

# eval
task_to_question_path = {
    "aokvqa": "./data/aokvqa/data.json",
    # "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/balanced_evalset.json",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/random_20000.json",
    "mvsa_m": "/ai/teacher/ssz/all_data/msa/MVSA_Multiple/test.json",
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/new_test_mix.json",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/processed/val_mix.json",
    "docvqa": "/ai/teacher/ssz/all_data/mqa/docvqa/val_v1.0_withQT.json",
    "vqav2": "/ai/teacher/ssz/all_data/mqa/VQAv2/processed/vqav2val.json",
    "okvqa": "/ai/teacher/ssz/all_data/mqa/OKVQA/processed_datasets/okvqa_val.json",
    "vizwiz": "/ai/teacher/ssz/all_data/mqa/vizwiz/processed/val.json",
    "pope": "./data/pope/data.json",
    "vstar": "./data/vstar/data.json",
}

# train
task_to_train_path = {
    "mvsa_s": "/ai/teacher/ssz/all_data/msa/MVSA_Single/new_train_mix.json",
    "gqa": "/ai/teacher/ssz/all_data/mqa/GQA/jsons/balanced_trainset.json",
    # "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/processed/train_mix.json",
    "textvqa": "/ai/teacher/ssz/all_data/mqa/TextVQA/processed/train_example.json",
    "docvqa": "/ai/teacher/ssz/all_data/mqa/docvqa/train_v1.0_withQT.json",
    "vqav2": "/ai/teacher/ssz/all_data/mqa/VQAv2/processed/vqav2train.json",
    "vizwiz": "/ai/teacher/ssz/all_data/mqa/vizwiz/processed/train.json",
    "okvqa": "/ai/teacher/ssz/all_data/mqa/OKVQA/processed_datasets/okvqa_train.json"
}

model_to_fullname = {
    "llava_new_7": "liuhaotian/llava-v1.5-7b",
    "llava_new_13": "liuhaotian/llava-v1.5-13b",
    "llava_new_1.6_7": "liuhaotian/llava-v1.6-vicuna-7b",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5":"Qwen/Qwen2.5-VL-7B-Instruct",
    "llavaonevision": "none",
    "insblip": "Salesforce/instructblip-vicuna-7b",
    "vipllava": "llava-hf/vip-llava-7b-hf",
    # 
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip": "Salesforce/instructblip-vicuna-7b",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "glm4v": "THUDM/glm-4v-9b",
    "phi4": "microsoft/Phi-4-multimodal-instruct",
    "llama": "none",
    "vargpt": "VARGPT-family/VARGPT_LLaVA-v1",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}



def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", choices=model_to_fullname.keys())
    parser.add_argument("--task", type=str, default="textvqa", choices=task_to_question_path.keys())
    parser.add_argument("--method", type=str, default="rel_att")
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
    
    parser.add_argument("--attn_layer_idx", type=int, default=-1)
    parser.add_argument("--target_layer_idx", type=int, default=-1)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    
    
    
    
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    output_name = f'{args.model}-{args.task}-{args.method}-{args.lora_name}'
    args.model_path = model_to_fullname[args.model]
    args.model_id = args.model

    # args.output_path = os.path.join(args.save_path, "jsons", f"{output_name}.json")
    # args.ckpt_path = os.path.join(args.save_path, "ckpts", f"{output_name}")

    args.image_path = task_to_image_path[args.task]
    args.question_path = task_to_question_path[args.task]
    args.train_path = task_to_train_path[args.task]
    
    return args
