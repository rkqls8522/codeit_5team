#======================================================================================
# 준비된 데이터셋(변환 이미지 + YOLO 라벨)을 이용해서 YOLO모델을 학습시키는 과정
#  < 함수 설명>
# - prepare_data(): 학습 시작 전 converted train/val 데이터셋 확인 및 data.yaml 생성 함수
# - train(): 학습 실행 함수 (학습 시간 측정 + best 모델 경로 반환)
# (모델 학습)
#======================================================================================
from ultralytics import YOLO
import yolo_config as config
from datetime import datetime
import os
import time
import sys
import yaml
import shutil

# 1. data_engineer 모듈 경로 먼저 추가
#    - class_mapping.py(클래스 매핑 로직) import를 위해 경로를 파이썬 검색 경로에 등록
DATA_ENGINEER_DIR = os.path.join(config.ROOT_DIR, "src", "data_engineer")
sys.path.append(DATA_ENGINEER_DIR)

# 2. data_engineer 모듈 import
#    - read_classID: ClassID.txt를 읽어 YOLO 클래스 정보를 딕셔너리 형태로 반환
try:
    from class_mapping import read_classID
    print("data_engineer 모듈 로드 성공")
    print(f"   [확인] 전처리 모듈 경로: {os.path.abspath(DATA_ENGINEER_DIR)}")
except ImportError as e:
    print(f"data_engineer 모듈 로드 실패: {e}")
    sys.exit(1)


# 3. 데이터 준비 함수 정의
def prepare_data():
    """YOLO 학습용 converted 데이터셋 검증 + data.yaml 생성"""
    print("\n[1/2] 데이터 준비 시작")

    # 경로 설정 (이미 생성된 converted 데이터셋 기준)
    # - split_dir 아래 구조를 학습에 그대로 사용
    #   converted/
    #     ├─ images/train
    #     ├─ images/val
    #     ├─ labels/train
    #     └─ labels/val
    split_dir = os.path.join(config.ROOT_DIR, "data", "yolo_dataset", "converted")
    image_train_dir = os.path.join(split_dir, "images", "train")
    image_val_dir = os.path.join(split_dir, "images", "val")
    label_train_dir = os.path.join(split_dir, "labels", "train")
    label_val_dir = os.path.join(split_dir, "labels", "val")

    print("=" * 60)
    print("[1단계] converted 데이터셋 경로 확인")
    print(f" - Dataset 경로: {os.path.abspath(split_dir)}")
    print(f" - Train image 경로: {os.path.abspath(image_train_dir)}")
    print(f" - Val image 경로: {os.path.abspath(image_val_dir)}")
    print(f" - Train label 경로: {os.path.abspath(label_train_dir)}")
    print(f" - Val label 경로: {os.path.abspath(label_val_dir)}")
    print("=" * 60)

    # 필수 폴더 존재 여부 점검
    # - 하나라도 없으면 학습 진행 시 파일 로딩 단계에서 실패하므로 즉시 종료
    required_dirs = [image_train_dir, image_val_dir, label_train_dir, label_val_dir]
    missing_dirs = [path for path in required_dirs if not os.path.exists(path)]
    if missing_dirs:
        print("경고: converted 데이터셋 폴더를 찾을 수 없습니다.")
        for path in missing_dirs:
            print(f" - 누락 경로: {path}")
        sys.exit(1)

    # ClassID 읽기
    # - ClassID.txt 기준으로 클래스 이름/YOLO ID 매핑을 로드
    try:
        class_dict = read_classID(DATA_ENGINEER_DIR)
    except Exception as e:
        print(f"ClassID 로드 실패: {e}")
        sys.exit(1)

    print("ClassID 정보를 이용해 data.yaml을 생성합니다..")

    # 1) YOLO용 names 딕셔너리 생성 (yolo_id: name)
    #    - Ultralytics data.yaml의 names 필드 형식에 맞춰 구성
    yolo_names = {}
    for v in class_dict.values():
        yolo_names[v["yolo_id"]] = v["name"]

    # 2) yaml에 들어갈 내용 구성
    #    - path: 데이터셋 루트
    #    - train/val: path 기준 상대 경로
    #    - nc: 클래스 개수
    #    - names: 클래스 ID-이름 매핑
    data_config = {
        "path": split_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": len(yolo_names),
        "names": yolo_names,
    }

    # 3) data.yaml 파일 덮어쓰기
    #    - sort_keys=False: 키 순서를 유지해 사람이 읽기 쉽게 저장
    #    - allow_unicode=True: 한글 클래스명 깨짐 방지
    with open(config.data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)

    print(f"data.yaml 갱신 완료 (Classes: {len(yolo_names)})")

    print("=" * 60)
    print("[2단계] converted 데이터셋 준비 확인")
    print(f" - 사용 경로: {os.path.abspath(split_dir)}")
    print(" - 참고: 기존에 생성된 converted 이미지/라벨을 그대로 사용합니다.")
    print("=" * 60)
    print("데이터 준비 완료!\n")


def train(resume=False):
    """
    YOLO 모델 학습 실행
    """
    # 4. 학습 시작 전에 converted 데이터셋 준비 실행
    #    - data.yaml 최신화 및 경로 누락 여부를 먼저 검증
    prepare_data()

    # 모델 로드
    # - config.model_file: 사전학습 가중치 또는 사용자 지정 시작 모델 경로
    model = YOLO(config.model_file)

    # 학습 시간 측정 시작
    start_time = time.time()
    print(f"학습 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'yolo_converted_train_{timestamp}'

    # 실제 학습 실행
    # - optimizer='AdamW': 가중치 감쇠 포함 옵티마이저
    # - lr0=0.001: 초기 학습률
    # - epochs=10: 총 학습 epoch 수
    # - imgsz=640: 입력 이미지 크기
    # - device=config.device: cuda/cpu 등 실행 디바이스
    # - batch=10: 배치 크기
    # - patience=0: early stopping 비활성(개선 없어도 끝까지 학습)
    # - box/cls: box loss, class loss 가중치
    # - flipud/mixup: 데이터 증강 옵션
    # - project/name: 결과 저장 위치(run 폴더)
    # - resume: 이전 학습 이어서 진행 여부
    model.train(
        data=config.data_yaml_path,
        optimizer="AdamW",
        lr0=0.001,
        epochs=10,
        imgsz=640,
        device=config.device,
        batch=10,
        patience=0,
        box=7.5,
        cls=1.0,
        flipud=0.5,
        mixup=0.1,
        project=config.TRAIN_RESULT_DIR,
        name=run_name,
        exist_ok=False,
        resume=resume,
    )

    # 학습 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 시/분/초로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

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
    train()

### 학습실행 시 터미널 명령어: python src/model/yolo/yolo_train_converted.py ###
