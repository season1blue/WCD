
import argparse
import torch
import os
from .info import model_to_fullname, task_to_question_path, task_to_image_path

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
    
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    output_name = f'{args.model}-{args.task}-{args.method}.json'
    args.model_id = model_to_fullname[args.model]
    args.output_path = os.path.join(args.save_path, output_name)
    args.image_path = task_to_image_path[args.task]
    args.question_path = task_to_question_path[args.task]
    
    return args
