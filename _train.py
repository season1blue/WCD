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
from methods.llava_model import MyLlava

from methods.utils.args import load_args # type: ignore

import ipdb
from tqdm import tqdm

def train(args, model, loss_fn, dataloader, device, lora_output_dir):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        for step, batch in pbar:

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs["loss"] + 5 * loss_fn(outputs["logits"], outputs["token_select"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 保存 LoRA adapter
    model.save_pretrained(os.path.join(lora_output_dir, args.lora_name))
    print(f"LoRA adapter saved to {os.path.join(lora_output_dir, args.lora_name)}")


# ========== 主入口 ==========
def main(args):
    
    
    # 加载配置
    config = AutoConfig.from_pretrained(args.model_id)
    # config.token_select_tau = 5
    # config.token_select_layers = 1

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = MyLlava.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(args.device)


    # ==== 注入 LoRA ====
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # 需视模型结构调整
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    # 构建损失函数
    Ada_loss_fn = AdaLoss(base_criterion=torch.nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id))

    dataset = TrainDataset(args.task, args.question_path, args.image_path, processor=processor, max_samples=args.max_sample)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=TrainDataset.collate_fn)

    train(args, model, Ada_loss_fn, dataloader, args.device, args.lora_output_dir)


if __name__ == "__main__":
    args = load_args()
    main(args)
