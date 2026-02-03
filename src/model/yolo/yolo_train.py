# 모델 학습
from ultralytics import YOLO
import yolo_config as config
import os

def train():
    model = YOLO(config.model_file)

    print("학습 시작")

    model.train(data = "data.yaml",     # 같은 파일 없을 경우 위의 yaml_path변수에 넣기
                epochs = 50,
                imgsz = 640,
                device = config.device,
                batch = 16,
                project = config.TRAIN_RESULT_DIR,
                name = 'final_model',
                exist_ok = True)
    
    print("학습 완료, 모델 저장")

if __name__ == "__main__":
    train()