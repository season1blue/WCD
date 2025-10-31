
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
from types import SimpleNamespace

def custom_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        batch_out[key] = values  # 或进一步处理其他字段
    return batch_out


# import
from transformers import AutoModelForCausalLM, AutoTokenizer
def load_model_and_processor(args, dtype=torch.bfloat16, device_map="auto"):
    model, processor, vision_start_token_id, vision_end_token_id = None, None, None, None
    if "qwen2" == args.model_id.lower():
        from models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True, attn_implementation="eager").eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    elif "qwen2.5" == args.model_id.lower():
        from models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from transformers import AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True, attn_implementation="eager").eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    elif "glm4v" == args.model_id.lower():
        from models.glm4v.modeling_chatglm import ChatGLMForConditionalGeneration
        model = ChatGLMForConditionalGeneration.from_pretrained(
            "THUDM/glm-4v-9b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        processor = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
        vision_start_token_id = processor.convert_tokens_to_ids("<|begin_of_image|>")
        vision_end_token_id = processor.convert_tokens_to_ids("<|end_of_image|>")
    elif "insblip" == args.model_id.lower():
        from models.instructblip.modeling_instructblip import InstructBlipForConditionalGeneration
        from transformers import InstructBlipProcessor
        
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path, device_map=device_map, trust_remote_code=True).eval()
        processor = InstructBlipProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        
    elif "vipllava" == args.model_id.lower():
        from models.vipllava.modeling_vipllava import VipLlavaForConditionalGeneration
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf", device_map="auto", use_fast=True, num_additional_image_tokens=1)
        model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", device_map=device_map).eval()
        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
        
        
    elif "llavaonevision" == args.model_id.lower():
        from models.llava_onevision.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            dtype=torch.float16,
            device_map=device_map,
        ).eval()

        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    elif "vargpt" == args.model_id.lower():
        from models.vargpt.modeling_vargpt_llava import VARGPTLlavaForConditionalGeneration
        from models.vargpt.prepare_vargpt_llava import prepare_vargpt_llava 
        from models.vargpt.processing_vargpt_llava import VARGPTLlavaProcessor
        from transformers import AutoProcessor, AutoTokenizer
        # from patching_utils.patching import patching

        prepare_vargpt_llava(args.model_path)
        model = VARGPTLlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(0)
        # patching(model)

        processor = VARGPTLlavaProcessor.from_pretrained(args.model_path)

        
    
    return model, processor, vision_start_token_id, vision_end_token_id




def data_prepare(model_id: str, processor, image: Image.Image, question: str, vision_start_token_id, vision_end_token_id, device):
    pos, pos_end = None, None
    if "qwen2" == model_id.lower():
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
    elif "qwen2.5" == model_id.lower():
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
    elif "glm4v" == model_id.lower():
        inputs = processor.apply_chat_template([{"role": "user", "image": image, "content": question}], add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True).to(device)  # chat mode
        # inputs = processor(text=[question], images=[image])
        
        # pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
        # pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    elif "vipllava" == model_id.lower():
        prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
        prompt = prompt.format(question)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)

        pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
        pos_end = pos+576 #config中的设置image_seq_length是576

        # inputs = processor(text=question, images=image, return_tensors="pt")
        # pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1  # 32000
        # pos_end = pos+576
    elif "llavaonevision" == model_id.lower():
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)

        positions = (inputs['input_ids'][0] == vision_start_token_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            pos = positions[0].item()       # 第一个位置
            pos_end = positions[-1].item()        # 最后一个位置
    elif "insblip" == model_id.lower():
        image = image.convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    elif "vargpt" == model_id.lower():
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float32)
        
        
        
        
    return inputs, pos, pos_end








def _eval(args, epoch=None, model=None):

    model, processor, vision_start_token_id, vision_end_token_id = load_model_and_processor(args)
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
        
        mask_config = SimpleNamespace()
        mask_config.pos, mask_config.pos_end = pos, pos_end
        mask_config.target_layer = args.target_layer_idx
        mask_config.mask_ratio = args.mask_ratio
        
        # generated_ids = model.generate(**inputs, max_new_tokens=128, pad_token_id=processor.tokenizer.eos_token_id)
        generated_ids = model.generate(**inputs, mask_config=mask_config, max_new_tokens=128)
        # generated_ids = model.generate(**inputs, max_new_tokens=128)  # 普通生成，测试是否可以跑通
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