# 실험 하이퍼파라미터 모아둔 파일
# 버전별로 뭐 바꿨는지 주석으로 남겨둠

CONFIG = {
    # 데이터 정보 (EDA 기반)
    'num_classes': 57,  # background(0) + 약 56 클래스 = 57
    'image_size': (976, 1280),  # (width, height), 전체 이미지 동일
    'bbox_area_range': (23250, 272435),  # 최소~최대 bbox 면적 (px²)

    # ============================================================
    # v9_5fold_ensemble (2026.02.12) — Kaggle mAP 0.931 (BEST)
    # v8_optimized (0.915) 기반 + 5-Fold CV 앙상블
    # 핵심 변경:
    #   - 5-Fold Cross Validation (KFold, seed=42)
    #   - 5개 모델 WBF 앙상블 추론 (iou_thr=0.55)
    #   - Fold별 mAP: 1.0 / 0.99 / 0.98 / 1.0 / 1.0 (평균 0.9949)
    # v8 설정 유지:
    #   - AdamW, CosineAnnealing + Warmup 1ep, 30ep
    #   - Gradient Clipping max_norm=10.0
    #   - Albumentations 안정적 augmentation
    #   - Mosaic 제거 / 차등 LR 제거
    # ============================================================

    # 학습 하이퍼파라미터
    'batch_size': 4,
    'learning_rate': 0.0001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'num_epochs': 30,  # 30ep + CosineAnnealing으로 충분히 수렴

    # Optimizer
    'optimizer': 'adamw',  # 'sgd' | 'adamw'

    # LR 스케줄러
    'lr_scheduler_type': 'cosine',  # 'step' | 'multistep' | 'cosine'
    'lr_scheduler_step': 5,  # step 방식일 때 사용
    'lr_scheduler_gamma': 0.1,
    'lr_scheduler_milestones': [15, 30, 40],  # multistep 방식일 때 LR 감소 시점

    # LR Warmup
    'warmup_epochs': 1,  # 처음 1 epoch 동안 LR을 서서히 올림

    # Backbone 차등 LR
    'backbone_lr_ratio': 1.0,  # 동일 LR (소규모 데이터에서 차등 LR보다 효과적)

    # Gradient Clipping
    'grad_clip_max_norm': 10.0,  # gradient explosion 방지

    # 모델 설정
    'backbone': 'resnet50_v2',  # 'resnet50' | 'resnet50_v2' | 'resnet101' | 'mobilenet_v3'
    'use_albumentations': True,

    # Mosaic Augmentation
    'use_mosaic': False,  # 소규모(232장)+다클래스(57)에서 class collapse 유발 → 제거
    'mosaic_prob': 0.5,

    # 추론
    'score_threshold': 0.05,
    'nms_threshold': 0.5,
    'use_soft_nms': True,  # Soft-NMS 사용 (단일 모델 추론 시)
    'soft_nms_sigma': 0.5,  # Gaussian decay sigma

    # 5-Fold 앙상블 (v9)
    'n_folds': 5,
    'wbf_iou_threshold': 0.55,  # WBF에서 같은 객체로 판단할 IoU
    'wbf_skip_threshold': 0.001,  # WBF 후 이 score 이하 제거
}
