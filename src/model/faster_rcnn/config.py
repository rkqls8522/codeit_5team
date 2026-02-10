CONFIG = {
    # 데이터 정보 (EDA 기반)
    'num_classes': 57,  # background(0) + 약 56 클래스 = 57
    'image_size': (976, 1280),  # (width, height), 전체 이미지 동일
    'bbox_area_range': (23250, 272435),  # 최소~최대 bbox 면적 (px²)

    # ============================================================
    # v8_optimized (2026.02.10) — Kaggle mAP 0.915 (BEST)
    # 핵심 변경 (v6 mAP 0.69 → 0.915):
    #   - AdamW optimizer (SGD 대비 안정적 수렴)
    #   - CosineAnnealing + Warmup 1ep (30ep 충분)
    #   - Soft-NMS (+1~1.7 mAP, 재학습 불필요)
    #   - Gradient Clipping max_norm=10.0
    #   - Albumentations 안정적 augmentation
    #   - Mosaic 제거 (소규모+다클래스에서 class collapse 유발)
    #   - 차등 LR 제거 (동일 LR이 소규모 데이터에서 더 효과적)
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
    'use_soft_nms': True,  # Soft-NMS 사용 (+1~1.7 mAP, 공짜)
    'soft_nms_sigma': 0.5,  # Gaussian decay sigma
}
