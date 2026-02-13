# Dataset 클래스 + augmentation 정의
# train/valid/test 데이터셋 로딩, Albumentations augmentation, Mosaic 등

import json as json_json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import numpy as np

import os
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import cv2


###     Faster R-CNN dataset class
"""
image: [C, H, W] FloatTensor, 0~1 범위,
target['boxes']: [N, 4] FloatTensor, 반드시 (x1, y1, x2, y2) 형식 (원본 JSON이 x,y,w,h면 변환 필요),
target['labels']: [N] Int64Tensor, 1부터 시작 (0은 background),
target['image_id']: 이미지 고유 번호,
target['area']: 각 박스의 넓이 ((x2-x1) * (y2-y1)),
target['iscrowd']: 전부 0으로 채우면 됩니다
"""

###     Mosaic Augmentation (실험 결과: 232장/57클래스에서는 class collapse 발생하여 사용 안 함)
def mosaic_augmentation(dataset, idx, img_size=(1280, 976)):
    """
    4장의 이미지를 하나로 합치는 Mosaic augmentation.
    대규모 데이터(COCO 등)에서는 효과적이나, 소규모(232장)+다클래스(57)에서는
    class collapse가 발생하여 v8 이후 use_mosaic=False로 비활성화.
    """
    w, h = img_size
    # 중심점을 랜덤으로 설정 (이미지의 25%~75% 범위)
    cx = random.randint(int(w * 0.25), int(w * 0.75))
    cy = random.randint(int(h * 0.25), int(h * 0.75))

    # 4개 이미지 인덱스 선택 (현재 + 랜덤 3개)
    indices = [idx] + random.sample(range(len(dataset)), 3)

    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    all_boxes = []
    all_labels = []

    # 4개 영역에 이미지 배치
    placements = [
        (0, 0, cx, cy),        # 좌상
        (cx, 0, w, cy),        # 우상
        (0, cy, cx, h),        # 좌하
        (cx, cy, w, h),        # 우하
    ]

    for i, (x1, y1, x2, y2) in enumerate(placements):
        data_idx = indices[i]
        img_path = dataset.img_p_list[data_idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # 영역 크기에 맞게 리사이즈
        region_w = x2 - x1
        region_h = y2 - y1
        if region_w <= 0 or region_h <= 0:
            continue

        scale_x = region_w / orig_w
        scale_y = region_h / orig_h
        resized = cv2.resize(img, (region_w, region_h))
        mosaic_img[y1:y2, x1:x2] = resized

        # 박스 좌표 변환
        boxes_list = dataset.annt_list[data_idx][0]
        labels_list = dataset.annt_list[data_idx][1]

        for box, label in zip(boxes_list, labels_list):
            bx1, by1, bx2, by2 = box
            # 원본 좌표 → 리사이즈 좌표 → mosaic 좌표
            new_x1 = bx1 * scale_x + x1
            new_y1 = by1 * scale_y + y1
            new_x2 = bx2 * scale_x + x1
            new_y2 = by2 * scale_y + y1

            # 영역 클리핑
            new_x1 = max(x1, min(new_x1, x2))
            new_y1 = max(y1, min(new_y1, y2))
            new_x2 = max(x1, min(new_x2, x2))
            new_y2 = max(y1, min(new_y2, y2))

            # 너무 작은 박스 제거
            if (new_x2 - new_x1) > 5 and (new_y2 - new_y1) > 5:
                all_boxes.append([new_x1, new_y1, new_x2, new_y2])
                all_labels.append(label)

    return mosaic_img, all_boxes, all_labels


###     Albumentations 기반 augmentation (v9 최종)
def get_training_transforms_albu():
    """학습용 augmentation. A.Normalize 사용 금지 (torchvision 내부 정규화와 충돌)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=0,
            p=0.4,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.15),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
    ))

def get_validation_transforms_albu():
    """Validation용 - augmentation 없이 [0,1] 정규화 + Tensor 변환만"""
    return A.Compose([
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

###     기본 v2.Compose transform (하위 호환용)
training_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
validation_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

###     train,valid dataset class
class FasterRCNNDataset(Dataset):
    def __init__(self, img_dir, annt_dir, cat_path, transforms=None, use_albumentations=False,
                 img_path_list_mode=False, img_path_list=[],
                 use_mosaic=False, mosaic_prob=0.5):
        self.img_dir = img_dir
        self.annt_dir = annt_dir
        img_p_list, annt_list = self._sorted_img_annt_path(img_dir, annt_dir, cat_path=cat_path, img_path_list_mode=img_path_list_mode, img_path_list=img_path_list)
        self.img_p_list = img_p_list
        self.annt_list = annt_list
        self.use_albumentations = use_albumentations
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob

        # Albumentations 사용 시 자동 설정
        if use_albumentations:
            self.transforms = transforms  # Albumentations Compose 객체
        else:
            self.transforms = transforms if transforms else training_transforms

    def __len__(self):
        return len(self.img_p_list)

    def __getitem__(self, idx):
        # Mosaic augmentation (50% 확률로 4장 합침)
        if self.use_mosaic and random.random() < self.mosaic_prob:
            mosaic_img, boxes_list, labels_list = mosaic_augmentation(self, idx)
            # Mosaic 결과에 Albumentations 적용
            if self.use_albumentations and self.transforms:
                if len(boxes_list) == 0:
                    transformed = self.transforms(image=mosaic_img, bboxes=[], labels=[])
                    image = transformed['image']
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    transformed = self.transforms(
                        image=mosaic_img, bboxes=boxes_list, labels=labels_list
                    )
                    image = transformed['image']
                    if len(transformed['bboxes']) == 0:
                        boxes = torch.zeros((0, 4), dtype=torch.float32)
                        labels = torch.zeros((0,), dtype=torch.int64)
                    else:
                        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                        labels = torch.tensor(transformed['labels'], dtype=torch.int64)
            else:
                image = torch.from_numpy(mosaic_img).permute(2, 0, 1).float() / 255.0
                boxes = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.tensor(labels_list, dtype=torch.int64) if labels_list else torch.zeros((0,), dtype=torch.int64)

            if boxes.shape[0] == 0:
                area = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            else:
                area = torch.abs((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {
                "boxes": boxes, "labels": labels,
                "image_id": torch.tensor([idx]), "area": area, "iscrowd": iscrowd,
            }
            return image, target

        # 일반 이미지 로딩
        img_path = self.img_p_list[idx]
        image = Image.open(img_path).convert("RGB")

        boxes_list = self.annt_list[idx][0]
        labels_list = self.annt_list[idx][1]

        if self.use_albumentations and self.transforms:
            # Albumentations 사용
            image_np = np.array(image)

            # 빈 박스 처리
            if len(boxes_list) == 0:
                transformed = self.transforms(image=image_np, bboxes=[], labels=[])
                image = transformed['image']
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                transformed = self.transforms(
                    image=image_np,
                    bboxes=boxes_list,
                    labels=labels_list
                )
                image = transformed['image']

                # 변환 후 박스가 사라질 수 있음 (min_visibility 때문)
                if len(transformed['bboxes']) == 0:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                    labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            # 기존 v2 transforms 사용
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)

            if self.transforms:
                image, boxes = self.transforms(image, boxes)

        # 빈 박스일 때 area, iscrowd 처리
        if boxes.shape[0] == 0:
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            area = torch.abs((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        return image, target
    
    #   _sorted_img_annt_path: pair된 idx번호를 가진 img path list와 annotation list를 반환 
    def _sorted_img_annt_path(self, img_dir, annt_dir, cat_path, load_exts = ("*.jpg", "*.png", "*.jpeg"), img_path_list_mode = False, img_path_list=[]): #   cat_path: make_classID_txt.py에서 생성한 class 매핑 txt 파일이 있는 폴더 경로
        
        if img_path_list_mode:
            img_p_list = img_path_list
        else:
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
        
        cat_id = {}
        with open(os.path.join(cat_path), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                k, v = line.strip().split()
                cat_id[int(k)] = int(v)

        annt_list = []
        for img_p in img_p_list:
            img_annt_boxes = []
            img_annt_labels = []
            for json in json_list:
                if json["images"][0]["file_name"] == os.path.basename(img_p):
                    x, y, w, h = json["annotations"][0]["bbox"]
                    img_annt_boxes.append([x, y, x + w, y + h])  # [x,y,w,h] → [x1,y1,x2,y2]
                    img_annt_labels.append(cat_id[json["categories"][0]["id"]])

            annt_list.append([img_annt_boxes, img_annt_labels])

        return img_p_list, annt_list

###     test dataset class
class TestDataset(Dataset):
    def __init__(self, img_dir, transforms=None, use_albumentations=False, load_exts = ("*.jpg", "*.png", "*.jpeg")):
        self.img_dir = img_dir
        self.use_albumentations = use_albumentations
        self.transforms = transforms if transforms else training_transforms
        img_p_list = []
        for ext in load_exts:
            img_p_list.extend(
                glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
            )
        self.img_p_list = img_p_list

    def __len__(self):
        return len(self.img_p_list)

    def __getitem__(self, idx):
        img_path = self.img_p_list[idx]
        image = Image.open(img_path).convert("RGB")

        if self.use_albumentations and self.transforms:
            image_np = np.array(image)
            transformed = self.transforms(image=image_np)
            image = transformed['image']
        elif self.transforms:
            image = self.transforms(image)

        return image


def get_test_transforms_albu():
    """Test용 - [0,1] 정규화 + Tensor 변환만 (bbox 없음)"""
    return A.Compose([
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])