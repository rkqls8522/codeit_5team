import json as json_json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
import numpy as np

import os
import glob


###     Faster R-CNN dataset class
"""
image: [C, H, W] FloatTensor, 0~1 범위,
target['boxes']: [N, 4] FloatTensor, 반드시 (x1, y1, x2, y2) 형식 (원본 JSON이 x,y,w,h면 변환 필요),
target['labels']: [N] Int64Tensor, 1부터 시작 (0은 background),
target['image_id']: 이미지 고유 번호,
target['area']: 각 박스의 넓이 ((x2-x1) * (y2-y1)),
target['iscrowd']: 전부 0으로 채우면 됩니다
"""

###     기본 v2.Compose transform
training_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
validation_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

###     train,valid dataset class
class FasterRCNNDataset(Dataset):                     # 데이터셋 클래스
    def __init__(self, img_dir, annt_dir, class_dict, transforms=training_transforms, img_path_list_mode=False, img_path_list=[]):
        self.img_dir = img_dir      # 이미지 폴더 경로
        self.annt_dir = annt_dir    # annotation 폴더 경로
        img_p_list, annt_list = self._sorted_img_annt_path(img_dir, annt_dir, class_dict, img_path_list_mode=img_path_list_mode, img_path_list=img_path_list)   #이미지 path list와 annotation list를 가져옴
        self.img_p_list = img_p_list    # 이미지 경로 리스트
        self.annt_list = annt_list      # annotation 경로 리스트
        self.transforms = transforms    #transforms

    def __len__(self):
        return len(self.img_p_list)

    def __getitem__(self, idx):         # dataset __getitem__ 부분

        img_path = self.img_p_list[idx]
        image = Image.open(img_path).convert("RGB")

        for annt in self.annt_list[idx][0]:     # X,y,W,H -> X,Y,X,Y 형식으로 변환
            annt[2] = annt[0] +annt[2]
            annt[3] = annt[1] +annt[3]
        boxes = torch.tensor(self.annt_list[idx][0], dtype=torch.float32)
        labels = torch.tensor(self.annt_list[idx][1], dtype=torch.int64)
        imgsize = self.annt_list[idx][2][0]

        target = {                      # 필요 내용 딕셔너리 저장
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": torch.abs((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms:         # transforms가 있으면 변환
            if len(boxes) > 0:
                tv_boxes = tv_tensors.BoundingBoxes(    # tv_tensor 적용
                    target["boxes"], 
                    format="XYXY", 
                    canvas_size=imgsize
                )
                image, tv_boxes = self.transforms(image, tv_boxes)
                target["boxes"] = tv_boxes

                boxes = target["boxes"].as_subclass(torch.Tensor)

                bb0 = boxes[:, 0].clamp(min=0, max=imgsize[0])
                bb1 = boxes[:, 1].clamp(min=0, max=imgsize[1])
                bb2 = boxes[:, 2].clamp(min=0, max=imgsize[0])
                bb3 = boxes[:, 3].clamp(min=0, max=imgsize[1])

                area = torch.abs((bb3 - bb1) * (bb2 - bb0))
                target["area"] = area
            else:
                image = self.transforms(image)

        return image, target
    
    #   _sorted_img_annt_path: pair된 idx번호를 가진 img path list와 annotation list를 반환 
    def _sorted_img_annt_path(self, img_dir, annt_dir, class_dict, load_exts = ("*.jpg", "*.png", "*.jpeg"), img_path_list_mode = False, img_path_list=[]): #   cat_path: make_classID_txt.py에서 생성한 class 매핑 txt 파일이 있는 폴더 경로
        
        if img_path_list_mode:
            img_p_list = img_path_list
        else:
            img_p_list = []
            for ext in load_exts:                   # 폴더에 모든 이미지 경로 저장
                img_p_list.extend(
                    glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
                )
        
        anntation_file_path_list = glob.glob(os.path.join(annt_dir,  "**"), recursive=True)     # 폴더에 모든 파일 경로 저장
        annt_p_list = [p for p in anntation_file_path_list if os.path.splitext(p)[1]==".json"]  # json 파일만 걸러냄
        json_list = []
        for path in annt_p_list:
            with open(path, "r", encoding="utf-8") as f:
                json_list.append(json_json.load(f))         # json 파일 읽어옴

        annt_list = []
        for img_p in img_p_list:
            img_annt_boxes = []
            img_annt_labels = []
            img_annt_imgsize = []
            for json in json_list:                                                  # 파일 이름, bbox, id, 이미지 높이, 이미지 넓이를 가져옴
                if json["images"][0]["file_name"] == os.path.basename(img_p):
                    boxes = json["annotations"][0]["bbox"]
                    img_annt_boxes.append(boxes)
                    img_annt_labels.append(class_dict[json["categories"][0]["id"]]["fasterrcnn_id"])
                    img_annt_imgsize.append((json["images"][0]["height"], json["images"][0]["width"]))

            annt_list.append([img_annt_boxes, img_annt_labels, img_annt_imgsize])

        return img_p_list, annt_list

###     test dataset class
class TestDataset(Dataset):                     # 테스트 데이터셋 클래스
    def __init__(self, img_dir, transforms=training_transforms, load_exts = ("*.jpg", "*.png", "*.jpeg")):
        self.img_dir = img_dir
        self.transforms = transforms
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

        if self.transforms:
            image = self.transforms(image)

        return image