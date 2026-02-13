"""Train Faster R-CNN v9 final model with 5-Fold CV."""

import gc
import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_v9_5fold import CONFIG_V9_5FOLD
from evaluate import evaluate
from make_classID_txt import make_classIDtxt
from model import get_model
from Faster_RCNN_dataset import FasterRCNNDataset
from Faster_RCNN_dataloader import collate_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "data", "original", "train_images")
TRAIN_ANNT_DIR = os.path.join(BASE_DIR, "data", "processed", "train_annotations")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints", CONFIG_V9_5FOLD["checkpoints_subdir"])


def get_train_transforms_v9():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ],
                p=0.15,
            ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_valid_transforms_v9():
    return A.Compose(
        [A.ToFloat(max_value=255.0), ToTensorV2()],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )


def train_one_epoch_v9(model, loader, optimizer, device, grad_clip_max_norm):
    model.train()
    running = 0.0
    for images, targets in tqdm(loader, leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()

        running += loss.item()
    return running / max(len(loader), 1)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cat_path, cat_id_map = make_classIDtxt(TRAIN_ANNT_DIR)
    num_classes = len(cat_id_map) + 1
    print(f"Classes: {num_classes}")

    full_dataset = FasterRCNNDataset(
        img_dir=TRAIN_IMG_DIR,
        annt_dir=TRAIN_ANNT_DIR,
        cat_path=cat_path,
        transforms=get_train_transforms_v9(),
        use_albumentations=True,
    )
    all_img_paths = full_dataset.img_p_list
    print(f"Images: {len(all_img_paths)}")

    cfg = CONFIG_V9_5FOLD
    kfold = KFold(
        n_splits=cfg["n_folds"],
        shuffle=True,
        random_state=cfg["random_seed"],
    )
    fold_results = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(all_img_paths), start=1):
        print(f"\n===== Fold {fold_idx}/{cfg['n_folds']} =====")
        train_paths = [all_img_paths[i] for i in train_idx]
        valid_paths = [all_img_paths[i] for i in valid_idx]
        print(f"Train {len(train_paths)} / Valid {len(valid_paths)}")

        train_dataset = FasterRCNNDataset(
            img_dir=TRAIN_IMG_DIR,
            annt_dir=TRAIN_ANNT_DIR,
            cat_path=cat_path,
            transforms=get_train_transforms_v9(),
            use_albumentations=True,
            img_path_list_mode=True,
            img_path_list=train_paths,
            use_mosaic=False,
        )
        valid_dataset = FasterRCNNDataset(
            img_dir=TRAIN_IMG_DIR,
            annt_dir=TRAIN_ANNT_DIR,
            cat_path=cat_path,
            transforms=get_valid_transforms_v9(),
            use_albumentations=True,
            img_path_list_mode=True,
            img_path_list=valid_paths,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=collate_fn,
        )

        model = get_model(
            num_classes=num_classes,
            backbone=cfg["backbone"],
            score_threshold=cfg["score_threshold"],
            nms_threshold=cfg["nms_threshold"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["num_epochs"] - cfg["warmup_epochs"],
            eta_min=1e-6,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=cfg["warmup_epochs"],
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[cfg["warmup_epochs"]],
        )
        best_loss = float("inf")

        for epoch in range(cfg["num_epochs"]):
            avg_loss = train_one_epoch_v9(
                model,
                train_loader,
                optimizer,
                device,
                cfg["grad_clip_max_norm"],
            )
            scheduler.step()
            if (epoch + 1) % 5 == 0 or avg_loss < best_loss:
                print(
                    f"[F{fold_idx} E{epoch+1}/{cfg['num_epochs']}] "
                    f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "loss": avg_loss,
                        "fold": fold_idx,
                    },
                    os.path.join(
                        SAVE_DIR,
                        cfg["checkpoint_name_template"].format(fold=fold_idx),
                    ),
                )

        ckpt = torch.load(
            os.path.join(SAVE_DIR, cfg["checkpoint_name_template"].format(fold=fold_idx)),
            map_location=device,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        m_ap, _ = evaluate(model, valid_loader, device)
        fold_results.append({"fold": fold_idx, "best_loss": best_loss, "mAP": float(m_ap)})
        print(f"Fold {fold_idx} done | Best loss {best_loss:.4f} | mAP {float(m_ap):.4f}")

        del model, optimizer, scheduler, train_loader, valid_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n===== v9 5-Fold Summary =====")
    for r in fold_results:
        print(f"Fold {r['fold']}: loss={r['best_loss']:.4f}, mAP={r['mAP']:.4f}")
    print(f"Average mAP: {np.mean([r['mAP'] for r in fold_results]):.4f}")
    print(f"Checkpoint dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()
