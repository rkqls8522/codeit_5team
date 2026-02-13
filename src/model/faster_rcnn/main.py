# 전체 파이프라인 실행 (데이터 로딩 → 학습 → 평가)
# 로컬에서 돌릴 때 쓰는 파일, Colab은 노트북 사용

import os
import torch
from sklearn.model_selection import train_test_split

from config import CONFIG
from model import get_model
from make_classID_txt import make_classIDtxt
from Faster_RCNN_dataset import (
    FasterRCNNDataset,
    training_transforms,
    validation_transforms,
    get_training_transforms_albu,
    get_validation_transforms_albu,
)
from Faster_RCNN_dataloader import train_valid_build_dataloaders
from train import train
from evaluate import evaluate

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'data', 'original', 'train_images')
TRAIN_ANNT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'train_annotations')
TEST_IMG_DIR = os.path.join(BASE_DIR, 'data', 'original', 'test_images')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')

VALID_RATIO = 0.2
RANDOM_SEED = 42


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 클래스 ID 매핑 생성
    print("\n[1/5] 클래스 ID 매핑 생성 중...")
    cat_path, cat_id = make_classIDtxt(TRAIN_ANNT_DIR)
    num_classes = len(cat_id) + 1  # +1 for background
    print(f"  클래스 수: {num_classes} (background 포함)")

    # 2. 전체 Dataset 생성 후 train/valid 8:2 분할
    print("\n[2/5] Dataset 생성 + train/valid 분할 중...")

    # Albumentations 사용 여부 (CONFIG에서 설정)
    use_albu = CONFIG.get('use_albumentations', False)
    if use_albu:
        print("  ✓ Albumentations 강력한 augmentation 사용 (v8)")
        train_tf = get_training_transforms_albu()
        valid_tf = get_validation_transforms_albu()
    else:
        print("  ✓ 기본 transforms 사용")
        train_tf = training_transforms
        valid_tf = validation_transforms

    full_dataset = FasterRCNNDataset(
        img_dir=TRAIN_IMG_DIR,
        annt_dir=TRAIN_ANNT_DIR,
        cat_path=cat_path,
        transforms=train_tf,
        use_albumentations=use_albu,
    )
    print(f"  전체 이미지 수: {len(full_dataset)}")

    # 8:2 split (이미지 경로 기준)
    all_img_paths = full_dataset.img_p_list
    train_paths, valid_paths = train_test_split(
        all_img_paths, test_size=VALID_RATIO, random_state=RANDOM_SEED
    )

    # Mosaic augmentation (소규모 데이터 핵심)
    use_mosaic = CONFIG.get('use_mosaic', False)
    mosaic_prob = CONFIG.get('mosaic_prob', 0.5)
    if use_mosaic:
        print(f"  ✓ Mosaic augmentation 사용 (prob={mosaic_prob})")

    train_dataset = FasterRCNNDataset(
        img_dir=TRAIN_IMG_DIR,
        annt_dir=TRAIN_ANNT_DIR,
        cat_path=cat_path,
        transforms=train_tf,
        use_albumentations=use_albu,
        img_path_list_mode=True,
        img_path_list=train_paths,
        use_mosaic=use_mosaic,
        mosaic_prob=mosaic_prob,
    )
    valid_dataset = FasterRCNNDataset(
        img_dir=TRAIN_IMG_DIR,
        annt_dir=TRAIN_ANNT_DIR,
        cat_path=cat_path,
        transforms=valid_tf,
        use_albumentations=use_albu,
        img_path_list_mode=True,
        img_path_list=valid_paths,
    )
    print(f"  Train: {len(train_dataset)}장 / Valid: {len(valid_dataset)}장")

    # 3. DataLoader 구성
    print("\n[3/5] DataLoader 구성 중...")
    train_loader, valid_loader = train_valid_build_dataloaders(
        train_dataset, valid_dataset,
        batch_size=CONFIG['batch_size'],
    )

    # 데이터 로딩 테스트 (1 batch)
    images, targets = next(iter(train_loader))
    print(f"  1 batch 테스트 통과 - 이미지 {len(images)}장, "
          f"첫 이미지 shape: {images[0].shape}, "
          f"첫 타겟 boxes: {targets[0]['boxes'].shape}")

    # 4. 모델 로드 + 학습
    print("\n[4/5] 모델 로드 + 학습 시작...")
    backbone = CONFIG.get('backbone', 'resnet50')
    print(f"  Backbone: {backbone}")
    print(f"  Optimizer: {CONFIG.get('optimizer', 'sgd').upper()}")
    print(f"  Learning Rate: {CONFIG['learning_rate']}")
    print(f"  Backbone LR ratio: {CONFIG.get('backbone_lr_ratio', 1.0)}")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Warmup: {CONFIG.get('warmup_epochs', 0)} epochs")
    print(f"  Scheduler: {CONFIG.get('lr_scheduler_type', 'step')}")
    print(f"  Grad Clip: {CONFIG.get('grad_clip_max_norm', 'None')}")
    print(f"  Score Threshold: {CONFIG['score_threshold']}")
    print(f"  Soft-NMS: {CONFIG.get('use_soft_nms', False)}")

    model = get_model(
        num_classes=num_classes,
        backbone=backbone,
        score_threshold=CONFIG['score_threshold'],
        nms_threshold=CONFIG['nms_threshold'],
    )
    model.to(device)

    train_losses = train(model, train_loader, valid_loader, device, save_dir=SAVE_DIR)

    # 5. 최종 평가
    print("\n[5/5] 최종 평가 중...")
    best_ckpt = torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    mAP, class_ap = evaluate(model, valid_loader, device)
    print(f"  mAP@0.5: {mAP:.4f}")

    print("\n학습 완료!")


if __name__ == '__main__':
    main()
