# 데이터 전처리 스크립트

import json as json_json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import numpy as np

import os
import glob


"""
image: [C, H, W] FloatTensor, 0~1 범위,
target['boxes']: [N, 4] FloatTensor, 반드시 (x1, y1, x2, y2) 형식 (원본 JSON이 x,y,w,h면 변환 필요),
target['labels']: [N] Int64Tensor, 1부터 시작 (0은 background),
target['image_id']: 이미지 고유 번호,
target['area']: 각 박스의 넓이 ((x2-x1) * (y2-y1)),
target['iscrowd']: 전부 0으로 채우면 됩니다
"""


training_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


validation_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


class FasterRCNNDataset(Dataset):
    def __init__(self, img_dir, annt_dir, transforms=training_transforms):
        img_p_list, annt_list = self._sorted_img_annt_path(img_dir, annt_dir)
        self.img_p_list = img_p_list
        self.annt_list = annt_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_p_list)

    def __getitem__(self, idx):

        img_path = self.img_p_list[idx]
        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(self.annt_list[idx][0], dtype=torch.float32)
        labels = torch.tensor(self.annt_list[idx][1], dtype=torch.int64)

        if self.transforms:
            image, boxes = self.transforms(image, boxes)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": torch.abs((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image, target
    
    def _sorted_img_annt_path(self, img_dir, annt_dir, load_exts = ("*.jpg", "*.png", "*.jpeg"), d_path = r"C:\Users\User1\Downloads\ai-07-object-detection\sprint_ai_project1_data\train_annotations"):
        img_p_list = []
        for ext in load_exts:
            img_p_list.extend(
                glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
            )
        
        anntation_file_path_list = glob.glob(os.path.join(annt_dir,  "**"), recursive=True)
        annt_p_list = [p for p in anntation_file_path_list if os.path.splitext(p)[1]==".json"]
        json_list = []
        for path in annt_p_list:
            with open(path, "r", encoding="utf-8") as f:
                json_list.append(json_json.load(f))
        
        cat_id_to_yolo = {}
        with open(os.path.join(d_path, "cat_id_to_yolo.txt"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():   # 빈 줄 스킵
                    continue
                k, v = line.strip().split()
                cat_id_to_yolo[int(k)] = int(v)

        annt_list = []
        for img_p in img_p_list:
            img_annt_boxes = []
            img_annt_labels = []
            for json in json_list:
                if json["images"][0]["file_name"] == os.path.basename(img_p):
                    boxes = json["annotations"][0]["bbox"]
                    img_annt_boxes.append(boxes)
                    img_annt_labels.append(cat_id_to_yolo[json["categories"][0]["id"]])

            annt_list.append([img_annt_boxes, img_annt_labels])

        return img_p_list, annt_list