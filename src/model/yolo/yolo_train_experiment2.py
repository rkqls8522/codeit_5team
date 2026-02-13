#======================================================================================
# YOLO 실험 4번: 학습 전용 스크립트
# 준비된 experiment4 데이터를 이용해서 YOLO 모델을 학습시키는 과정
# 
# 전제조건: yolo_experiment4_data.py를 먼저 실행하여 데이터가 준비되어 있어야 함
#
# 주요 특징:
# - 원본 + 선명도 향상 데이터(sharpness_factor=50.0)로 학습
# - data_experiment4.yaml 파일 사용
# - experiment4 폴더의 데이터 참조
# - 결과는 results/yolo_experiment4_model/에 저장
#======================================================================================
from ultralytics import YOLO
import yolo_config as config
from datetime import datetime
import os
import time
import shutil

def train_experiment4(resume=False):
    """
    YOLO 실험 4번 모델 학습 수행
    
    실험 목적:
    원본 데이터 + 선명도 향상 데이터를 함께 학습하여
    선명도 전처리가 객체 탐지 성능에 미치는 영향 측정
    
    Args:
        resume (bool): 이전 학습을 이어서 할지 여부. Default는 False.
    
    전제조건:
        yolo_experiment4_data.py를 먼저 실행하여
        data/yolo_dataset/experiment4 폴더와 data_experiment4.yaml이 생성되어 있어야 함
    
    Returns:
        str: Best 모델의 경로 (성공 시) 또는 None (실패 시)
    """
    print("\n" + "="*80)
    print("[YOLO 실험 4번] 모델 학습 시작")
    print(" - 원본 + 선명도 향상 데이터 (sharpness_factor=50.0)")
    print("="*80 + "\n")
    
    # ============================================================================
    # [1단계] 데이터 파일 경로 설정 및 확인
    # ============================================================================
    # data_experiment4.yaml 파일 경로
    data_yaml_path = os.path.join(config.CURRENT_DIR, 'data_experiment4.yaml')
    
    # experiment4 데이터 폴더 경로
    experiment4_dir = os.path.join(config.ROOT_DIR, 'data', 'yolo_dataset', 'experiment4')
    
    # 데이터 존재 여부 확인
    if not os.path.exists(data_yaml_path):
        print("오류: data_experiment4.yaml 파일을 찾을 수 없습니다.")
        print(f"   경로: {data_yaml_path}")
        print("\n먼저 다음 명령어로 데이터를 준비하세요:")
        print("   python src/model/yolo/yolo_experiment4_data.py")
        return None
    
    if not os.path.exists(experiment4_dir):
        print("오류: experiment4 폴더를 찾을 수 없습니다.")
        print(f"   경로: {experiment4_dir}")
        print("\n먼저 다음 명령어로 데이터를 준비하세요:")
        print("   python src/model/yolo/yolo_experiment4_data.py")
        return None
    
    # ============================================================================
    # [2단계] 학습 데이터 정보 출력
    # ============================================================================
    print("="*80)
    print("▶ 학습 데이터 정보")
    print(f" - data.yaml: {data_yaml_path}")
    print(f" - 데이터 경로: {experiment4_dir}")
    print(f" - 증강 방법: 선명도 향상 (sharpness_factor=50.0)")
    
    # 이미지 개수 확인
    train_images = os.path.join(experiment4_dir, 'images', 'train')
    val_images = os.path.join(experiment4_dir, 'images', 'val')
    
    if os.path.exists(train_images):
        train_count = len([f for f in os.listdir(train_images) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f" - Train 이미지: {train_count}개 (원본 + 선명도 향상 987개)")
    
    if os.path.exists(val_images):
        val_count = len([f for f in os.listdir(val_images) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f" - Val 이미지: {val_count}개 (원본만)")
    
    print("="*80 + "\n")
    
    # ============================================================================
    # [3단계] YOLO 모델 로드
    # ============================================================================
    # YOLOv8n 베이스 모델 로드
    # config.model_file은 yolo_config.py에 정의되어 있음 (기본값: yolov8n.pt)
    model = YOLO(config.model_file)
    print(f"✓ 모델 로드 완료: {config.model_file}\n")
    
    # ============================================================================
    # [4단계] 모델 학습 실행
    # ============================================================================
    # 학습 시간 측정 시작
    start_time = time.time()
    print(f"학습 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'yolo_experiment2_train_{timestamp}'
    
    # YOLO 모델 학습
    # 주요 하이퍼파라미터:
    # - epochs: 50 (최대 50번 반복 학습)
    # - imgsz: 640 (이미지 크기를 640x640으로 리사이즈)
    # - batch: 10 (한 번에 10개 이미지씩 학습)
    # - patience: 10 (10 epoch 동안 개선이 없으면 조기 종료)
    model.train(
        data=data_yaml_path,              # experiment4 전용 yaml 사용
        epochs=50,                        # 최대 학습 epoch 수
        imgsz=640,                        # 이미지 크기
        device=config.device,             # 학습 디바이스 (GPU/CPU)
        batch=10,                         # 배치 크기
        patience=10,                      # Early stopping patience
        project=config.TRAIN_RESULT_DIR,  # 결과 저장 루트 폴더
        name=run_name,    # experiment4 전용 폴더명
        exist_ok=True,                    # 기존 폴더가 있어도 덮어쓰기
        resume=resume                     # 이전 학습 이어서 하기
    )
    
    # ============================================================================
    # [5단계] 학습 완료 및 결과 출력
    # ============================================================================
    # 학습 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 시/분/초로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("="*80)
    print(f"학습 종료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 학습 소요 시간: {hours}시간 {minutes}분 {seconds}초 (총 {elapsed_time:.2f}초)")
    
    # 학습된 모델 경로 반환
    best_model_path = os.path.join(config.TRAIN_RESULT_DIR, run_name, 'weights', 'best.pt')
    print(f"학습 완료, Best 모델 저장 위치: {best_model_path}")

    # [추가] 추론(Inference) 편의를 위해 최신 모델을 기본 경로(yolo_final_model)로 복사
    final_fixed_path = config.trained_model_path  # .../results/yolo_final_model/weights/best.pt

    # 복사할 폴더가 없으면 생성
    os.makedirs(os.path.dirname(final_fixed_path), exist_ok=True)

    # 파일 복사
    shutil.copy(best_model_path, final_fixed_path)
    print(f"[자동갱신] 최신 모델이 기본 경로로 복사되었습니다: {final_fixed_path}")
    print("이제 yolo_predict.py를 실행하면 자동으로 이 모델이 사용됩니다.")

    return best_model_path

if __name__ == "__main__":
    # 학습 실행
    train_experiment4()

### 학습실행 시, 터미널 명령어: python src/model/yolo/yolo_train_experiment2.py ###
