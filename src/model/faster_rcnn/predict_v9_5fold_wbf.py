"""Inference for Faster R-CNN v9 5-Fold ensemble with WBF."""

import csv
import os

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

from config_v9_5fold import CONFIG_V9_5FOLD
from make_classID_txt import make_classIDtxt
from model import get_model
from Faster_RCNN_dataset import TestDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_ANNT_DIR = os.path.join(BASE_DIR, "data", "processed", "train_annotations")
TEST_IMG_DIR = os.path.join(BASE_DIR, "data", "original", "test_images")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", CONFIG_V9_5FOLD["checkpoints_subdir"])
OUTPUT_PATH = os.path.join(BASE_DIR, CONFIG_V9_5FOLD["submission_name"])


def get_test_transforms_v9():
    return A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])


def load_models_v9_5fold(device, num_classes):
    cfg = CONFIG_V9_5FOLD
    models = []
    for fold_idx in range(1, cfg["n_folds"] + 1):
        model = get_model(
            num_classes=num_classes,
            backbone=cfg["backbone"],
            score_threshold=cfg["score_threshold"],
            nms_threshold=cfg["nms_threshold"],
        )
        ckpt_path = os.path.join(
            CKPT_DIR, cfg["checkpoint_name_template"].format(fold=fold_idx)
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
        print(
            f"Loaded fold {fold_idx}: epoch={ckpt.get('epoch', '?')} "
            f"loss={ckpt.get('loss', 0):.4f}"
        )
    return models


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cat_path, cat_id_map = make_classIDtxt(TRAIN_ANNT_DIR)
    reverse_map = {v: k for k, v in cat_id_map.items()}
    num_classes = len(cat_id_map) + 1

    test_dataset = TestDataset(
        img_dir=TEST_IMG_DIR,
        transforms=get_test_transforms_v9(),
        use_albumentations=True,
    )
    print(f"Test images: {len(test_dataset)}")

    models = load_models_v9_5fold(device, num_classes)
    cfg = CONFIG_V9_5FOLD
    rows = []
    annotation_id = 1

    for img_idx in tqdm(range(len(test_dataset)), desc="v9 5fold WBF"):
        image = test_dataset[img_idx]
        image_gpu = image.to(device)
        img_path = test_dataset.img_p_list[img_idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        _, img_h, img_w = image.shape

        all_boxes = []
        all_scores = []
        all_labels = []

        for model in models:
            output = model([image_gpu])[0]
            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            keep = scores >= cfg["csv_score_threshold"]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(boxes) > 0:
                norm_boxes = boxes.clone()
                norm_boxes[:, 0] /= img_w
                norm_boxes[:, 1] /= img_h
                norm_boxes[:, 2] /= img_w
                norm_boxes[:, 3] /= img_h
                norm_boxes = norm_boxes.clamp(0, 1)
                all_boxes.append(norm_boxes.numpy())
                all_scores.append(scores.numpy())
                all_labels.append(labels.numpy())
            else:
                all_boxes.append(np.empty((0, 4)))
                all_scores.append(np.empty((0,)))
                all_labels.append(np.empty((0,)))

        if not any(len(b) > 0 for b in all_boxes):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=[1] * cfg["n_folds"],
            iou_thr=cfg["wbf_iou_threshold"],
            skip_box_thr=cfg["wbf_skip_threshold"],
        )

        fused_boxes[:, 0] *= img_w
        fused_boxes[:, 1] *= img_h
        fused_boxes[:, 2] *= img_w
        fused_boxes[:, 3] *= img_h

        keep = fused_scores >= cfg["csv_score_threshold"]
        fused_boxes = fused_boxes[keep]
        fused_scores = fused_scores[keep]
        fused_labels = fused_labels[keep]

        for box, label, score in zip(fused_boxes, fused_labels, fused_scores):
            original_id = reverse_map.get(int(label), int(label))
            x1, y1, x2, y2 = box.tolist()
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": original_id,
                    "bbox_x": round(x1, 1),
                    "bbox_y": round(y1, 1),
                    "bbox_w": round(x2 - x1, 1),
                    "bbox_h": round(y2 - y1, 1),
                    "score": round(float(score), 4),
                }
            )
            annotation_id += 1

    fieldnames = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {OUTPUT_PATH}")
    print(f"Objects: {len(rows)}")


if __name__ == "__main__":
    pattern = os.path.join(
        CKPT_DIR, CONFIG_V9_5FOLD["checkpoint_name_template"].format(fold=1)
    )
    if not os.path.exists(pattern):
        print(f"No v9 5fold checkpoint found in {CKPT_DIR}")
    else:
        main()
