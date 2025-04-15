import os
import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms  # 如果你有图像预处理需求

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 可选
])

class ImageTextDataset(Dataset):
    def __init__(self, task, question_path, image_path, max_samples=500):
        self.task = task
        self.data = self.load_data(question_path)
        if max_samples is not None:
            self.data = self.data[:max_samples]  # 截断前N条
        self.image_path = image_path
        self.transform = transform

        # 补全每条数据的 image 路径
        for d in self.data:
            if "image_path" in d:
                d["image_path"] = os.path.join(image_path, d["image_path"])
            else:
                d["image_path"] = os.path.join(image_path, f"{d['image_id']}.jpg")

    def load_data(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            return list(load_dataset(path)['test'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # image = Image.open(item["image_path"]).convert("RGB")

        # if self.transform:
        #     image = self.transform(image)
        if self.task in ["gqa", "textvqa"]:
            return {
                "image_path": item["image_path"],
                "text": item["question"],
                "answer": item.get("answer", ""),  # 防止部分缺失
                # "qid": item.get("qid", idx)
            }
        elif self.task in ["mvsa_m", "mvsa_s"]:
            text = item["text"]
            text = f"Based on text : '{text}'. Confirm the sentiment from the choice (positive, negative, neutral). If the sentiment is unclear or ambiguous between positive and negative, please prioritize labeling it as neutral."
            return {
                "image_path": item["image_path"],
                "text": text,
                "answer": item.get("sentiment", ""),  # 防止部分缺失
                # "qid": item.get("qid", idx)
            }