import random
import os
import glob
import cv2
import numpy as np
from pathlib import Path as Path_Path
from PIL import Image
import torch
from torchvision import tv_tensors


def data_maker2(img_dir, yolo_annt_dir, cls_id, class_list, start_num = 0, how_many=3, file_name="yolo_augmentation", load_exts = ("*.jpg", "*.png", "*.jpeg"), transforms = None):

    yolo_annt_list = glob.glob(os.path.join(yolo_annt_dir, "**", "*.txt"), recursive=True)

    img_path_list = []
    for ext in load_exts:
        img_path_list.extend(
            glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
    )                                               #모든 이미지 경로 리스트


    cls_idid = {}
    for key, val in cls_id.items():
        cls_idid[val['yolo_id']] = key

    for num in range(start_num, how_many + start_num):

        img = cv2.imread(img_path_list[0])
        ps_img = np.full_like(img, (128, 128, 128))   #팔레트 생성
        h, w, _ = img.shape
        ps_center = (int(h/2), int(w/2))
        

        random.shuffle(yolo_annt_list)
        cmp_annt = []
        print("증강 시작-------------------:")

        bbox_list = []
        id_list = []
        i = 0
        i2 = 0

        for yolo_annt in yolo_annt_list:

            if yolo_annt not in cmp_annt:

                cmp_annt.append(yolo_annt)
                    
                with open(yolo_annt) as f:
                    for line in f:
                            
                        cls, cx, cy, bw, bh = map(float, line.split())
                        cls = int(cls)
                        if (cls_idid[cls] in class_list):
                            x1 = int((cx - bw/2) * w)      # 이미지 x1, y1, x2, y2 구하기
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)

                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)

                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            for img_path in img_path_list:
                                if os.path.splitext(os.path.basename(img_path))[0] == os.path.splitext(os.path.basename(yolo_annt))[0]: #이름 같은 이미지 불러옴
                                    img = cv2.imread(img_path)
                                    
                       
                                    if transforms:
                                        target = {}
                                        h_img, w_img = img.shape[:2]
        
                                        target["boxes"] = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32) ############################### tv_tensor 변환
                                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
                                        img = tv_tensors.Image(img_tensor)
                                        
                                        if len(target["boxes"]) > 0:
                                            tv_boxes = tv_tensors.BoundingBoxes(
                                                target["boxes"],
                                                format="XYXY",
                                                canvas_size=(h_img, w_img)
                                            )
                                            out_img, tv_boxes = transforms(img, tv_boxes)

                                            boxes_list = tv_boxes.as_subclass(torch.Tensor)

                                            x1, y1, x2, y2 = map(int, boxes_list[0].tolist())

                                            img = out_img.permute(1, 2, 0).numpy().astype('uint8').copy()

                                            h_img, w_img = img.shape[:2]

                                            x1 = np.clip(x1, 0, w_img)
                                            y1 = np.clip(y1, 0, h_img)
                                            x2 = np.clip(x2, 0, w_img)
                                            y2 = np.clip(y2, 0, h_img)                                      ############################### tv_tensor 변환
                                    
                                    mean_color = img[y2-2:y2, x2-2:x2].mean(axis=(0, 1))
                                    pill_roi = img[y1:y2, x1:x2]   # 관심 영역
        
                            hig, wth = pill_roi.shape[:2]
                            
                            if i == 0:
                                ps_img[0:ps_center[0],0:ps_center[1]] = mean_color
                                if ((ps_center[0] / 2) < (hig / 2)) or ((ps_center[1] / 2) < (wth / 2)):
                                    ps_img[0:hig,0:wth] = pill_roi
                                else:
                                    y1 = int(ps_center[0]/2) - int(hig / 2)
                                    y2 = y1+hig
                                    x1 = int(ps_center[1]/2) - int(wth / 2)
                                    x2 = x1+wth
                                    ps_img[y1:y2,x1:x2] = pill_roi
                
                            elif i == 1:
                                ps_img[0:ps_center[0],ps_center[1]:w] = mean_color
                                if ((ps_center[0] / 2) < (hig / 2)) or ((ps_center[1] / 2) < (wth / 2)):
                                    ps_img[0:hig,w-wth:w] = pill_roi
                                else:
                                    y1 = int(ps_center[0]/2) - int(hig / 2)
                                    y2 = y1+hig
                                    x1 = ps_center[1] + int(ps_center[1]/2) - int(wth / 2)
                                    x2 = x1+wth
                                    ps_img[y1:y2,x1:x2] = pill_roi
                
                            elif i == 2:
                                ps_img[ps_center[0]:h,0:ps_center[1]] = mean_color
                                if ((ps_center[0] / 2) < (hig / 2)) or ((ps_center[1] / 2) < (wth / 2)):
                                    ps_img[h-hig:h,0:wth] = pill_roi
                                else:
                                    y1 = ps_center[0] + int(ps_center[0]/2) - int(hig / 2)
                                    y2 = y1+hig
                                    x1 = int(ps_center[1]/2) - int(wth / 2)
                                    x2 = x1+wth
                                    ps_img[y1:y2,x1:x2] = pill_roi
                
                            elif i == 3:
                                ps_img[ps_center[0]:h,ps_center[1]:w] = mean_color
                                if ((ps_center[0] / 2) < (hig / 2)) or ((ps_center[1] / 2) < (wth / 2)):
                                    ps_img[h-hig:h, w-wth:w] = pill_roi
                                else:
                                    y1 = ps_center[0] + int(ps_center[0]/2) - int(hig / 2)
                                    y2 = y1+hig
                                    x1 = ps_center[1] + int(ps_center[1]/2) - int(wth / 2)
                                    x2 = x1+wth
                                    ps_img[y1:y2,x1:x2] = pill_roi
                            
                            bbox_list.append([x1,y1,x2,y2])
                            id_list.append(cls)
            
                            if (i == 3) or (len(yolo_annt_list) == len(cmp_annt)):
                                ps_img_rgb = cv2.cvtColor(ps_img, cv2.COLOR_BGR2RGB)
        
                                img = cv2.imread(img_path_list[0])
                                ps_img = np.full_like(img, (128, 128, 128))
                                h, w, _ = img.shape
                                ps_center = (int(h/2), int(w/2))
                                i = 0

                                txt_path = os.path.join(yolo_annt_dir, f"{file_name}_{num}_{i2}.txt")

                                for bbox, id in zip(bbox_list, id_list):
                                    x, y, xx, yy = bbox

                                    if not Path_Path(txt_path).exists():
                                        with open(txt_path, "w", encoding="utf-8") as f:
                                            f.write(f"{id} {(x+((xx-x)/2))/w:.6f} {(y+((yy-y)/2))/h:.6f} {(xx-x)/w:.6f} {(yy-y)/h:.6f}")
                                    else:
                                        with open(txt_path, "a", encoding="utf-8") as f:
                                            f.write(f"\n{id} {(x+((xx-x)/2))/w:.6f} {(y+((yy-y)/2))/h:.6f} {(xx-x)/w:.6f} {(yy-y)/h:.6f}")

                                save_img_path = os.path.join(img_dir, f"{file_name}_{num}_{i2}.png")
                                Image.fromarray(ps_img_rgb).save(save_img_path)

                                print(f"{file_name}_{num}_{i2}.png 증강 완료.")

                                bbox_list = []
                                id_list = []
                            else:
                                i += 1     
            i2 += 1