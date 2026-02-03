CONFIG = {
    # 데이터 정보 (EDA 기반)
    'num_classes': None,  # background + 약 클래스 수 (데이터 담당 확인 후 설정)
    'image_size': (976, 1280),  # (width, height), 전체 이미지 동일
    'bbox_area_range': (23250, 272435),  # 최소~최대 bbox 면적 (px²)

    # 학습 하이퍼파라미터
    'batch_size': 4,
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'num_epochs': 10,

    # LR 스케줄러
    'lr_scheduler_step': 3,
    'lr_scheduler_gamma': 0.1,

    # 추론
    'score_threshold': 0.5,
    'nms_threshold': 0.5,
}
