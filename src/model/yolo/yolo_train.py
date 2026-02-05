# 모델 학습
from ultralytics import YOLO
import yolo_config as config
import os
import time

def train(resume=False):
    """
    YOLO 모델 학습 수행
    
    Args:
        resume: 이어서 학습할지 여부
    
    Returns:
        str: 학습된 best 모델 경로
    """
    model = YOLO(config.model_file)
    
    # 학습 시간 측정 시작
    start_time = time.time()
    print(f"학습 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")      # 학습 시작 시간 출력
    
    model.train(data=config.data_yaml_path,     # ← config에서 가져오기
                epochs=10,
                imgsz=640,
                device=config.device,
                batch=16,
                project=config.TRAIN_RESULT_DIR,
                name='yolo_final_model',
                exist_ok=True,
                resume=resume)
    
    # 학습 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 시/분/초로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"학습 종료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")  # 학습 종료 시간 출력
    print(f"총 학습 소요 시간: {hours}시간 {minutes}분 {seconds}초 (총 {elapsed_time:.2f}초)")  # 학습 종료 시간 출력 및 총 학습 소요 시간 출력
    
    # 학습된 모델 경로 반환
    best_model_path = os.path.join(config.TRAIN_RESULT_DIR, 'final_model', 'weights', 'best.pt')
    print(f"학습 완료, Best 모델 저장 위치: {best_model_path}")
    
    return best_model_path
if __name__ == "__main__":
    train()

### 학습실행 시, 터미널 명령어: " python yolo_train.py " ### 