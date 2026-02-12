#======================================================================================
# YOLO 실험 3번: 학습 전용 스크립트
# 준비된 experiment3 데이터를 이용해서 YOLO 모델을 학습시키는 과정
# 
# 전제조건: prepare_experiment3_data.py를 먼저 실행하여 데이터가 준비되어 있어야 함
#
# 주요 변경사항:
# - prepare_data() 함수 제거 (데이터는 이미 준비됨)
# - data_experiment3.yaml 파일 사용
# - experiment3 폴더의 데이터 참조
#======================================================================================
from ultralytics import YOLO
import yolo_config as config
from datetime import datetime
import os
import time
import shutil

def train_experiment3(resume=False):
    """
    YOLO 실험 3번 모델 학습 수행
    
    전제조건: prepare_experiment3_data.py를 먼저 실행하여
             data/yolo_dataset/experiment3 폴더와 data_experiment3.yaml이 생성되어 있어야 함
    """
    print("\n" + "="*80)
    print("[YOLO 실험 3번] 모델 학습 시작")
    print("="*80 + "\n")
    
    # data_experiment3.yaml 경로
    data_yaml_path = os.path.join(config.CURRENT_DIR, 'data_experiment3.yaml')
    
    # experiment3 데이터 경로
    experiment3_dir = os.path.join(config.ROOT_DIR, 'data', 'yolo_dataset', 'experiment3')
    
    # 데이터 존재 여부 확인
    if not os.path.exists(data_yaml_path):
        print("오류: data_experiment3.yaml 파일을 찾을 수 없습니다.")
        print(f"   경로: {data_yaml_path}")
        print("\n먼저 다음 명령어로 데이터를 준비하세요:")
        print("   python src/data_engineer/prepare_experiment3_data.py")
        return None
    
    if not os.path.exists(experiment3_dir):
        print("오류: experiment3 폴더를 찾을 수 없습니다.")
        print(f"   경로: {experiment3_dir}")
        print("\n먼저 다음 명령어로 데이터를 준비하세요:")
        print("   python src/data_engineer/prepare_experiment3_data.py")
        return None
    
    # 데이터 정보 출력
    print("="*80)
    print("▶ 학습 데이터 정보")
    print(f" - data.yaml: {data_yaml_path}")
    print(f" - 데이터 경로: {experiment3_dir}")
    
    # 이미지 개수 확인
    train_images = os.path.join(experiment3_dir, 'images', 'train')
    val_images = os.path.join(experiment3_dir, 'images', 'val')
    
    if os.path.exists(train_images):
        train_count = len([f for f in os.listdir(train_images) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f" - Train 이미지: {train_count}개")
    
    if os.path.exists(val_images):
        val_count = len([f for f in os.listdir(val_images) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f" - Val 이미지: {val_count}개")
    
    print("="*80 + "\n")
    
    # 모델 로드
    model = YOLO(config.model_file)
    
    # 학습 시간 측정 시작
    start_time = time.time()
    print(f"학습 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'yolo_experiment1_train_{timestamp}'

    # 학습 실행
    model.train(
        data=data_yaml_path,              # experiment3 전용 yaml 사용
        epochs=50,
        imgsz=640,
        device=config.device,
        batch=10,
        patience=10,
        project=config.TRAIN_RESULT_DIR,
        name='yolo_experiment3_model',    # experiment3 전용 폴더명
        exist_ok=True,
        resume=resume
    )
    
    # 학습 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 시/분/초로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n학습 종료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    train_experiment3()

### 학습실행 시, 터미널 명령어: python src/model/yolo/yolo_train_experiment.py ###
