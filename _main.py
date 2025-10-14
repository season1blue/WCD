
import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoConfig
from tqdm import tqdm
import json
from methods.utils.utils import *
from methods.utils.args import load_args
from methods.utils.dataset import ImageTextDataset
from torch.utils.data import Dataset, DataLoader
from methods.utils.get_score import *
import ipdb
from datetime import datetime
import time
from transformers import GenerationConfig

def custom_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        batch_out[key] = values  # 或进一步处理其他字段
    return batch_out






# import

def load_model_and_processor(args, dtype=torch.bfloat16, device_map="auto"):
    model, processor, vision_start_token_id, vision_end_token_id = None, None, None, None
    if "qwen2" in args.model_id.lower():
        from models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True, attn_implementation="eager").eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    elif "llama4" in args.model_id.lower():
        from models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
        model = Llama4ForConditionalGeneration.from_pretrained(args.model_path, attn_implementation="flex_attention", device_map="auto",torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(args.model_path)
    else:
        raise ValueError("model is None")
    
    return model, processor, vision_start_token_id, vision_end_token_id


def data_prepare(model_id: str, processor, image: Image.Image, question: str, vision_start_token_id, vision_end_token_id, device):
    pos, pos_end = None, None
    if "qwen2" in model_id.lower():
        from qwen_vl_utils import process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "max_pixels": 512*28*28},  # 本地路径或 PIL.Image
                    {"type": "text",  "text": f"{question} Answer the question using a single word or phrase."},
                ],
            }
        ]
            
        texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[texts],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
        pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    

    elif "llama4" in model_id.lower():
        
        inputs = processor(images=image, text=question, return_tensors="pt")
        pos, pos_end = 0, 1


    return inputs, pos, pos_end




def _eval(args, epoch=None, model=None):

    model, processor, vision_start_token_id, vision_end_token_id = load_model_and_processor(args)
    
    
    generation_config = GenerationConfig.from_model_config(model.config)
    # generation_config.generation_mode = "dola_generation"  # 你可也用枚举或字符串标记
    generation_config.attn_layer_idx = args.attn_layer_idx
    generation_config.target_layer_idx = args.target_layer_idx
    generation_config.mask_ratio = args.mask_ratio

    generation_config.dola_layers = "low"
    # generation_config.dola_layers = [i for i in range(32)]
    generation_config.attn_diff = True

    generation_config.output_attentions = True
    # generation_config.return_dict_in_generate = False

    generation_config.attn_mask = True
    
    
    
    
    begin_time = time.time()
    
    dataset = ImageTextDataset(
        task=args.task,
        question_path=args.question_path,
        image_path=args.image_path,
        max_samples=args.max_sample
    )
    
    if args.task == "textvqa":
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn
            # num_workers=1
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn
            # num_workers=1
        )

    new_datas = []
    
    
    for dd in tqdm(dataloader, desc="Processing", ncols=100):
        question = dd["text"][0]  # batch_size个问题
        image_path = dd["image_path"][0]  # batch_size个图片路径
        image = Image.open(image_path).convert("RGB")
        
        
        inputs, pos, pos_end = data_prepare(args.model_id, processor, image, question, vision_start_token_id, vision_end_token_id, model.device)
        generation_config.pos, generation_config.pos_end = pos, pos_end
        
        
        
        generated_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=128)
        input_lens = inputs.input_ids.size(1)
        new_tokens = generated_ids[:, input_lens:]
        answers = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if args.task in ["mvsa_m", "mvsa_s"]:
            # 把batch里的每个样本分别补充
            new_d = {k: v[0] for k, v in dd.items()}  # 把第i个样本单独取出
            new_d["gen_answer"] = answers[0]
            new_d['id'] = new_d['id'].item() 
            new_datas.append(new_d)
        else:
            new_d = {k: v[0] for k, v in dd.items()}  # 把第i个样本单独取出
            new_d["gen_answer"] = answers[0]
            new_datas.append(new_d)

    output_path = os.path.join(args.save_path, "jsons", args.lora_name+ ".json")    
    print("evaluation output to", output_path)
    
    with open(output_path, "w") as f:
        json.dump(new_datas, f, indent=4)

    if args.task in ["mvsa_m", "mvsa_s"]:
        acc, _ = evaluate_mvsa(new_datas)
    elif args.task in ["gqa"]:
        acc, _ = evaluate_gqa(new_datas)
    elif args.task in ["textvqa", "docvqa", "vizwiz"]:
        acc, _ = evaluate_textvqa(new_datas)
    elif args.task in ["vqav2", "okvqa"]:
        acc, _ = evaluate_okvqa(new_datas)
        
    result = {
        'version': args.lora_name,
        'epoch': epoch,
        'run_time': time.time()-begin_time,
        'time': datetime.now().strftime("%Y-%m-%dd %H:%M:%S"),
        'acc': acc,
    }
    report_path = os.path.join(args.save_path, args.result_path)
    curr_time = datetime.now().strftime("%Y-%m-%dd %H:%M:%S")
    
    print(f"Time: {curr_time}, Version: {args.lora_name}, Acc: {acc} write into {report_path}")
    with open(report_path, 'a') as f:
        json.dump(result, f, indent=4)
        f.write('\n')  # 添加换行符

    
    
if __name__ == "__main__":
    args = load_args()
    _eval(args)