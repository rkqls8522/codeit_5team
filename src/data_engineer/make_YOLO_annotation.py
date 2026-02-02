# YOLO 학습용 txt 파일 제작 프로그램
# 


import os
import glob
import json as json_json
from pathlib import Path as Path_Path


master_dir = r".\codeit_5team\codeit_5team\data\ai-07-object-detection\sprint_ai_project1_data"

anntation_dir = os.path.join(master_dir, r"train_annotations")
image_dir = os.path.join(master_dir, r"train_images")
#   json, image 경로 설정


load_exts = ("*.jpg", "*.png", "*.jpeg")
#   불러올 이미지 확장자 설정



anntation_file_path_list = glob.glob(os.path.join(anntation_dir,  "**"), recursive=True)
annt_p_list = [p for p in anntation_file_path_list if os.path.splitext(p)[1]==".json"]

img_p_list = []
for ext in load_exts:
    img_p_list.extend(
        glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
    )
#   image, json 경로 리스트 작성    img_p_list==모든 이미지 파일 경로 리스트    annt_p_list =모든 json파일 경로 리스트


json_list = []
for path in annt_p_list:
    with open(path, "r", encoding="utf-8") as f:
        json_list.append(json_json.load(f))

annt_list = [{"file_name":json["images"][0]["file_name"], "categories":json["categories"], "annotations":json["annotations"], "image_size":(json["images"][0]["width"], json["images"][0]["height"])} for json in json_list]
#   json 파일 읽어오기 --> 리스트화 --> 필요한 내용만 가져온 annt_list 생성     {file_name, categories, annotations}


creat_flag = True

YOLO_annt_dir = os.path.join(anntation_dir, "YOLO annotation")

if Path_Path(YOLO_annt_dir).is_dir():
    print("YOLO annotation 폴더가 이미 존재합니다. 폴더 삭제 후 다시 시도해주세요")
    creat_flag = False

try:
    Path_Path(YOLO_annt_dir).mkdir(parents=True)
except:
    print("폴더 생성 실패")
    creat_flag = False

if creat_flag:
    #annt_unq_yolo = set()
    #for cat in annt_list:
    #    annt_unq_yolo.add(cat["categories"][0]["id"])
    #    cat_id_to_yolo = {cat: idx for idx, cat in enumerate(annt_unq_yolo)}

    cat_id_to_yolo = {
        cat_id: idx
        for idx, cat_id in enumerate(
            sorted({annt["categories"][0]["id"] for annt in annt_list})
        )
    }

    for annt in annt_list:
        file_name = os.path.splitext(annt["file_name"])[0]
        img_w = annt["image_size"][0]
        img_h = annt["image_size"][1]

        id = cat_id_to_yolo[annt["categories"][0]["id"]]
        x, y, w, h = annt["annotations"][0]["bbox"]
        
        
        txt_path = os.path.join(YOLO_annt_dir, f"{file_name}.txt")

        if not Path_Path(txt_path).exists():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{id} {(x+(w/2))/img_w:.6f} {(y+(h/2))/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}")
        else:
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"\n{id} {(x+(w/2))/img_w:.6f} {(y+(h/2))/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}")

