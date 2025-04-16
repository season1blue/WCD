def train(model, loss_fn, dataloader, device):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        # 假设模型输出为 dict，其中 prediction、token_select 存在
        adaloss, loss_dict = loss_fn(outputs, batch["labels"])

        adaloss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step} | Loss: {adaloss.item():.4f} | Detail: {loss_dict}")


# ========== 主入口 ==========
def main(args):
    # 加载配置
    config = AutoConfig.from_pretrained(args.model_id)
    config.token_select_tau = 5
    config.token_select_layers = 1

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = MyLlava.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(args.device)

    # 构建数据
    data = [
        {"question": "What is the man doing?", "image_path": "img1.jpg", "answer": "He is cooking."},
        {"question": "What color is the bird?", "image_path": "img2.jpg", "answer": "It is red."}
    ]
    dataset = MyDataset(data, processor, args.image_root)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 构建损失函数
    loss_fn = AdaLoss(base_criterion=torch.nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id))

    train(model, loss_fn, dataloader, args.device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
