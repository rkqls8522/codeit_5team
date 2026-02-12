#======================================================================================
# YOLO 학습용 이미지 변환(선명도 조정) + 라벨 생성 + Train/Val 분할을 한 번에 처리하는 과정
#  < 함수 설명>
# - _collect_images(): 이미지 폴더에서 유효 확장자(.jpg/.jpeg/.png) 파일만 수집
# - _ensure_clean_dir(): 대상 폴더를 비운 뒤 다시 생성(깨끗한 출력 보장)
# - convert_images_with_sharpness(): converted 데이터셋 생성 메인 함수
# - main(): 기본 경로 기준으로 변환 파이프라인 실행
# (데이터 변환/구축)
#======================================================================================

import os
import sys
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

# 1. data_engineer 모듈 경로 설정
#    - 현재 파일 기준으로 프로젝트 루트/전처리 모듈 경로를 계산
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
DATA_ENGINEER_DIR = os.path.join(ROOT_DIR, "src", "data_engineer")
sys.path.insert(0, DATA_ENGINEER_DIR)

# 2. 데이터 전처리/라벨 생성에 필요한 모듈 import
#    - apply_v2_compose_and_save: torchvision 변환 적용 후 이미지 저장
#    - read_classID: ClassID.txt를 읽어 클래스 매핑 정보 반환
#    - make_YOLO_annotation: 원본 annotation -> YOLO txt 라벨 생성
from yolo_img_converter import apply_v2_compose_and_save
from class_mapping import read_classID
from make_YOLO_annotation import make_YOLO_annotation


# 3. 이미지 파일 수집 유틸 함수
def _collect_images(image_dir):
    """
    지정 폴더에서 이미지 파일만 수집해서 Path 리스트로 반환
    """
    return [
        p
        for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]


# 4. 폴더 초기화 유틸 함수
def _ensure_clean_dir(dir_path):
    """
    기존 폴더가 있으면 삭제 후 재생성
    - 이전 실행 결과가 섞이지 않도록 출력 폴더를 초기화할 때 사용
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


# 5. converted 데이터셋 생성 메인 함수
def convert_images_with_sharpness(
    master_dir,
    output_base_dir=None,
    sharpness_factor=50.0,
    val_ratio=0.2,
    seed=42,
    shuffle=True,
):
    """
    converted YOLO 데이터셋 생성

    처리 개요:
    1) 원본 train 이미지를 선명도 변환해서 임시 폴더에 저장
    2) 원본 annotation으로 YOLO 라벨(txt) 생성
    3) 원본 기준으로 train/val 분할
    4) train은 변환 이미지 + 라벨, val은 원본 이미지 + 라벨 복사

    Args:
        master_dir (str): `images/train`, `train_annotations`를 포함한 원본 데이터 루트
        output_base_dir (str): 출력 베이스 경로 (기본값: ROOT_DIR/data/yolo_dataset)
        sharpness_factor (float): RandomAdjustSharpness 강도
        val_ratio (float): 검증셋 비율
        seed (int): 분할 재현성을 위한 random seed
        shuffle (bool): 분할 전 셔플 여부

    Returns:
        dict: 생성된 converted 데이터셋 주요 경로
    """
    # 입력 경로 구성
    annotation_dir = os.path.join(master_dir, "train_annotations")
    image_dir = os.path.join(master_dir, "images", "train")
    classid_dir = DATA_ENGINEER_DIR

    # 출력 경로 기본값 설정
    if output_base_dir is None:
        output_base_dir = os.path.join(ROOT_DIR, "data", "yolo_dataset")
    converted_base_dir = os.path.join(output_base_dir, "converted")

    print("[1/5] 경로 설정 완료")
    print(f"  - master_dir: {master_dir}")
    print(f"  - annotation_dir: {annotation_dir}")
    print(f"  - original_image_dir: {image_dir}")
    print(f"  - output_dir: {converted_base_dir}")

    # ClassID 로드
    class_dict = read_classID(classid_dir)
    print(f"[2/5] 클래스 매핑 로드 완료: {len(class_dict)}개")

    # 1) 원본 train 전체에 선명도 변환 적용 -> 임시 폴더 저장
    temp_converted_train_dir = os.path.join(master_dir, "temp_converted_train")
    if os.path.exists(temp_converted_train_dir):
        shutil.rmtree(temp_converted_train_dir)

    transforms = v2.Compose(
        [v2.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=1.0)]
    )
    print(f"[3/5] 이미지 변환 시작 (sharpness_factor={sharpness_factor})")
    temp_converted_train_dir = apply_v2_compose_and_save(
        transforms, image_dir, master_dir, output_dir_name="temp_converted_train"
    )

    # 2) 원본 annotation 기준 YOLO 라벨 생성
    #    - 실행 전 기존 임시 라벨 폴더가 있으면 삭제
    temp_yolo_annotation_dir = os.path.join(annotation_dir, "temp_YOLO_annotation")
    if os.path.exists(temp_yolo_annotation_dir):
        shutil.rmtree(temp_yolo_annotation_dir)

    print("[4/5] YOLO 라벨 생성 + 원본 기준 train/val 분할")
    yolo_annotation_dir = make_YOLO_annotation(
        image_dir, annotation_dir, class_dict, "temp_YOLO_annotation"
    )

    # 라벨이 실제로 존재하는 이미지 페어만 추출
    original_images = _collect_images(image_dir)
    valid_images = [
        p
        for p in original_images
        if os.path.exists(os.path.join(yolo_annotation_dir, f"{p.stem}.txt"))
    ]

    if not valid_images:
        raise ValueError("유효한 image/label 쌍을 찾지 못했습니다.")

    # 원본 기준 train/val 분할
    train_imgs, val_imgs = train_test_split(
        valid_images,
        test_size=val_ratio,
        random_state=seed,
        shuffle=shuffle,
    )
    print(f"  - 분할 완료: train={len(train_imgs)}, val={len(val_imgs)}")

    # 3) 출력 폴더 구조 생성
    #    converted/
    #      ├─ images/train (변환 이미지)
    #      ├─ images/val   (원본 이미지)
    #      ├─ labels/train
    #      └─ labels/val
    _ensure_clean_dir(converted_base_dir)
    train_image_out = os.path.join(converted_base_dir, "images", "train")
    val_image_out = os.path.join(converted_base_dir, "images", "val")
    train_label_out = os.path.join(converted_base_dir, "labels", "train")
    val_label_out = os.path.join(converted_base_dir, "labels", "val")
    for path in [train_image_out, val_image_out, train_label_out, val_label_out]:
        os.makedirs(path, exist_ok=True)

    # 변환 이미지 탐색을 빠르게 하기 위한 이름->Path 매핑
    converted_name_map = {p.name: p for p in _collect_images(temp_converted_train_dir)}

    copied_train = 0
    copied_val = 0

    # train: 변환 이미지 + 라벨 복사
    for img in train_imgs:
        converted_img = converted_name_map.get(img.name)
        label_path = os.path.join(yolo_annotation_dir, f"{img.stem}.txt")
        if converted_img is None or not os.path.exists(label_path):
            continue
        shutil.copy2(converted_img, os.path.join(train_image_out, img.name))
        shutil.copy2(label_path, os.path.join(train_label_out, f"{img.stem}.txt"))
        copied_train += 1

    # val: 원본 이미지 + 라벨 복사
    for img in val_imgs:
        label_path = os.path.join(yolo_annotation_dir, f"{img.stem}.txt")
        if not os.path.exists(label_path):
            continue
        shutil.copy2(img, os.path.join(val_image_out, img.name))
        shutil.copy2(label_path, os.path.join(val_label_out, f"{img.stem}.txt"))
        copied_val += 1

    print("[5/5] 데이터셋 복사 완료")
    print(f"  - train 페어 복사 수: {copied_train}")
    print(f"  - val 페어 복사 수: {copied_val}")

    # 4) 임시 폴더 정리
    if os.path.exists(temp_converted_train_dir):
        shutil.rmtree(temp_converted_train_dir)
    if os.path.exists(yolo_annotation_dir):
        shutil.rmtree(yolo_annotation_dir)

    # 5) 결과 경로 반환
    result_paths = {
        "converted_dataset_dir": converted_base_dir,
        "train_images_dir": os.path.join(converted_base_dir, "images", "train"),
        "val_images_dir": os.path.join(converted_base_dir, "images", "val"),
        "train_labels_dir": os.path.join(converted_base_dir, "labels", "train"),
        "val_labels_dir": os.path.join(converted_base_dir, "labels", "val"),
    }

    print("=" * 60)
    print("converted 데이터셋 생성 완료")
    print(f"dataset: {result_paths['converted_dataset_dir']}")
    print(f"train images (converted): {result_paths['train_images_dir']}")
    print(f"val images (original): {result_paths['val_images_dir']}")
    print("=" * 60)
    return result_paths


# 6. 단독 실행용 메인 함수
def main():
    # 기본 입력 경로 설정
    master_dir = os.path.join(ROOT_DIR, "data", "original")
    train_images_dir = os.path.join(master_dir, "images", "train")

    print("YOLO 변환 파이프라인 시작")
    print("=" * 60)
    print("train은 변환 이미지 사용, val은 원본 이미지 유지")
    print("=" * 60)

    # 필수 경로 존재 여부 확인
    if not os.path.exists(master_dir):
        print(f"error: master_dir를 찾을 수 없습니다: {master_dir}")
        return
    if not os.path.exists(train_images_dir):
        print(f"error: train_images_dir를 찾을 수 없습니다: {train_images_dir}")
        return

    # 변환 파이프라인 실행
    try:
        result_paths = convert_images_with_sharpness(
            master_dir=master_dir,
            sharpness_factor=50.0,
            val_ratio=0.2,
        )
        print("모든 단계가 정상적으로 완료되었습니다.")
        print(f"dataset path: {result_paths['converted_dataset_dir']}")
    except Exception as exc:
        print(f"error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

### 실행 명령어: python src/model/yolo/yolo_imge_convert.py ###
