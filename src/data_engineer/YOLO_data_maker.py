import glob
import os
import torch
import json as json_json
from pathlib import Path as Path_Path

from torchvision import tv_tensors
from PIL import Image


def data_maker(image_dir, annotation_dir, image_train_dir, labels_train_dir, transforms, img_name_list, how_many, class_dict, load_exts = ("*.jpg", "*.png", "*.jpeg"), all_mode = False):

    print("경로를 불러옵니다.")

    anntation_file_path_list = glob.glob(os.path.join(annotation_dir,  "**"), recursive=True)
    annt_p_list = [
        p for p in anntation_file_path_list
        if os.path.splitext(p)[1] == ".json"
    ]                                               #모든 json annotation 경로 리스트

    img_p_list = []
    for ext in load_exts:
        img_p_list.extend(
            glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
    )                                               #모든 이미지 경로 리스트
    
    image_path_list = []
    for img_p in img_p_list:
        if os.path.basename(img_p) in img_name_list:
            image_path_list.append(img_p)           # img_name_list에 이름이 올라와 있는 있는 이미지 경로만 추출


    image_nn_list = [os.path.splitext(os.path.basename(pth))[0] for pth in image_path_list] # image_nn_list 확장자 제외 이름 증강 이미지 이름 리스트

    annt_path_list = []
    for annt_p in annt_p_list:
        if os.path.splitext(os.path.basename(annt_p))[0] in image_nn_list:
            annt_path_list.append(annt_p)                                       #증강 이미지에 대한 annotation 파일 경로 추출

    if all_mode:
        image_path_list = img_p_list
        annt_path_list = annt_p_list

    print("json파일 읽는 중")

    json_list = []
    for path in annt_path_list:
        with open(path, "r", encoding="utf-8") as f:
            json_list.append(json_json.load(f))

    if len(json_list) != len(annt_path_list):
        print("json != annt_path_list")
        return None

    ### annt_path_list, image_path_list --> 증강 이미지와 해당 이미지 annotation



    print("데이터 증강 시작")
    for num in range(how_many):
        for image_path in image_path_list:      # 이미지 경로 순차적으로 불러옴

            target = {}
            boxes_list = []
            id_list = []
            img_size = []
            file_name = None
            for (annt_path, json) in zip(annt_path_list, json_list):    # anntation 경로 순차적으로 불러옴
                
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                if file_name == os.path.splitext(os.path.basename(annt_path))[0]:   # 이미지와 annotation 확장자 제외 파일 이름이 같을 시,

                    id_list.append(class_dict[json["categories"][0]["id"]]["yolo_id"])
                    x, y, w, h = json["annotations"][0]["bbox"]
                    boxes_list.append([x, y, w+x, h+y])

                    img_size.append((json['images'][0]['width'], json['images'][0]['height']))

            target["id"] = torch.tensor(id_list, dtype=torch.int64)
            target["boxes"] = torch.tensor(boxes_list, dtype=torch.float32)
            


            image = Image.open(image_path).convert("RGB")
            if len(target["boxes"]) > 0:
                tv_boxes = tv_tensors.BoundingBoxes(
                    target["boxes"], 
                    format="XYXY", 
                    canvas_size=(img_size[0][1], img_size[0][0])
                )
                image, tv_boxes = transforms(image, tv_boxes)
                target["boxes"] = tv_boxes


            img_w, img_h = image.size
            boxes_list = target["boxes"].as_subclass(torch.Tensor)
            id_list = target["id"].as_subclass(torch.Tensor)
            txt_path = os.path.join(labels_train_dir, f"{file_name}_{num}.txt")
            im_path = os.path.join(image_train_dir, f"{file_name}_{num}.png")
            if Path_Path(txt_path).is_file():
                if Path_Path(im_path).is_file():
                    print(f"{file_name}_{num} 파일 존재")
                else:
                    print(f"{file_name}_{num} txt 파일만 존재")
                continue
            elif Path_Path(im_path).is_file():
                print(f"{file_name}_{num} 이미지 파일만 존재")
                continue
                
            for id, bbox in zip(id_list, boxes_list):

                id = int(id.item())
                x, y, xx, yy = bbox.tolist()

                if not Path_Path(txt_path).exists():
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"{id} {(x+((xx-x)/2))/img_w:.6f} {(y+((yy-y)/2))/img_h:.6f} {(xx-x)/img_w:.6f} {(yy-y)/img_h:.6f}")
                else:
                    with open(txt_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{id} {(x+((xx-x)/2))/img_w:.6f} {(y+((yy-y)/2))/img_h:.6f} {(xx-x)/img_w:.6f} {(yy-y)/img_h:.6f}")

            
            save_img_path = os.path.join(image_train_dir, f"{file_name}_{num}.png")
            image.save(save_img_path)

            print(f"{file_name}_{num}.png 이미지 증강 완료")