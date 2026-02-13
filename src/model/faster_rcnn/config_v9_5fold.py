"""Faster R-CNN v9 5-Fold final settings."""

CONFIG_V9_5FOLD = {
    # data/model
    "backbone": "resnet50_v2",
    "num_classes": 57,  # background 포함
    "n_folds": 5,
    "random_seed": 42,

    # train
    "batch_size": 4,
    "num_workers": 2,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "optimizer": "adamw",
    "grad_clip_max_norm": 10.0,

    # scheduler
    "lr_scheduler_type": "cosine",
    "lr_scheduler_step": 5,
    "lr_scheduler_gamma": 0.1,
    "lr_scheduler_milestones": [15, 30, 40],
    "warmup_epochs": 1,

    # inference
    "score_threshold": 0.05,
    "nms_threshold": 0.5,
    "wbf_iou_threshold": 0.55,
    "wbf_skip_threshold": 0.001,
    "csv_score_threshold": 0.05,

    # paths
    "checkpoints_subdir": "5fold",
    "submission_name": "submission_5fold_wbf.csv",
    "checkpoint_name_template": "best_model_fold{fold}.pth",
}
