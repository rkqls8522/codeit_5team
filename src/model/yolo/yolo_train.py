# 모델 학습
from ultralytics import YOLO
import yolo_config as config
import os
def train(resume=False):
    """
    YOLO 모델 학습 수행
    
    Args:
        resume: 이어서 학습할지 여부
    
    Returns:
        str: 학습된 best 모델 경로
    """
    model = YOLO(config.model_file)
    print("학습 시작")
    model.train(data=config.data_yaml_path,     # ← config에서 가져오기
                epochs=15,
                imgsz=640,
                device=config.device,
                batch=16,
                project=config.TRAIN_RESULT_DIR,
                name='yolo_final_model',
                exist_ok=True,
                resume=resume)
    
    # 학습된 모델 경로 반환
    best_model_path = os.path.join(config.TRAIN_RESULT_DIR, 'final_model', 'weights', 'best.pt')
    print(f"학습 완료, Best 모델 저장 위치: {best_model_path}")
    
    return best_model_path
if __name__ == "__main__":
    train()