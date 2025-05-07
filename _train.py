import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from PIL import Image
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType  # 加入 PEFT

from methods.utils.dataset import TrainDataset
from methods.utils.token_loss import AdaLoss # type: ignore
from methods.my_llava import MyLlava
from llava.model import LlavaLlamaForCausalLM

from methods.utils.args import load_args # type: ignore

import ipdb
from tqdm import tqdm
from _eval import _eval

def train(args, model, loss_fn, dataloader, processor, device, lora_output_dir):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        for step, batch in pbar:

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            ada_loss = loss_fn(outputs["logits"], outputs["token_select"])
            loss = outputs["loss"] + 50 * ada_loss
            # loss = outputs["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "adaloss": f"{ada_loss.item():.4f}"})

        # 每个epoch结束的时候输出
        if args.is_eval > 0:
            print("Start Evaluation")
            _eval(args, epoch, model, processor)
    # 保存 LoRA adapter
    model.save_pretrained(os.path.join(lora_output_dir, args.lora_name), safe_serialization=False )
    print(f"LoRA adapter saved to {os.path.join(lora_output_dir, args.lora_name)}")

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)




from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# ========== 主入口 ==========
def main(args):
    
    model_path = args.model_id
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name)
    
    # # 加载配置
    # config = AutoConfig.from_pretrained(args.model_id)
    # config.dropout=0.1
    # # config.token_select_tau = 5
    # # config.token_select_layers = 1

    # processor = AutoProcessor.from_pretrained(args.model_id)
    # model = LlavaLlamaForCausalLM.from_pretrained(
    #     args.model_id,
    #     torch_dtype=torch.bfloat16,
    #     # config=config,
    #     # attn_implementation="eager"
    # ).to(args.device)


    # ==== 注入 LoRA ====
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    # 构建损失函数
    Ada_loss_fn = AdaLoss(base_criterion=torch.nn.CrossEntropyLoss())

    train_dataset = TrainDataset(args.task, args.train_path, args.image_path, processor=processor, max_samples=args.max_sample)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=TrainDataset.collate_fn)

    train(args, model, Ada_loss_fn, train_dataloader, processor, args.device, args.lora_output_dir)


if __name__ == "__main__":
    args = load_args()
    main(args)
