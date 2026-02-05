import os
import csv
import torch
import torchvision.transforms.v2.functional as F
from Faster_RCNN_dataset import TestDataset, validation_transforms
from Faster_RCNN_dataloader import test_build_dataloaders

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    HAS_WBF = False
    print("[WARNING] ensemble_boxes 미설치. WBF 대신 NMS 사용. 설치: pip install ensemble_boxes")


@torch.no_grad()
def predict(model, image, device, score_threshold=0.5):
    """
    단일 이미지에 대해 예측 수행.

    Returns:
        boxes, labels, scores (confidence threshold 이상만)
    """
    model.eval()
    image = image.to(device)
    output = model([image])[0]

    keep = output['scores'] >= score_threshold
    return {
        'boxes': output['boxes'][keep].cpu(),
        'labels': output['labels'][keep].cpu(),
        'scores': output['scores'][keep].cpu(),
    }


@torch.no_grad()
def predict_tta(model, image, device):
    """
    TTA: 원본 + 좌우반전으로 2회 추론 후 결과를 합친다.
    WBF가 설치되어 있으면 WBF로 박스를 합치고, 없으면 단순 concat 후 NMS를 적용한다.

    Returns:
        boxes [N,4] (x1,y1,x2,y2), labels [N], scores [N]
    """
    model.eval()
    img_h, img_w = image.shape[-2], image.shape[-1]

    # --- 원본 추론 ---
    img_orig = image.to(device)
    out_orig = model([img_orig])[0]

    # --- 좌우반전 추론 ---
    img_flip = F.horizontal_flip(image).to(device)
    out_flip = model([img_flip])[0]

    # 좌우반전 박스를 원본 좌표로 되돌리기: x1' = w - x2, x2' = w - x1
    flip_boxes = out_flip['boxes'].cpu().clone()
    x1 = flip_boxes[:, 0].clone()
    x2 = flip_boxes[:, 2].clone()
    flip_boxes[:, 0] = img_w - x2
    flip_boxes[:, 2] = img_w - x1

    orig_boxes = out_orig['boxes'].cpu()
    orig_scores = out_orig['scores'].cpu()
    orig_labels = out_orig['labels'].cpu()
    flip_scores = out_flip['scores'].cpu()
    flip_labels = out_flip['labels'].cpu()

    if HAS_WBF:
        # WBF: 박스를 0~1로 정규화해야 함
        def normalize_boxes(boxes, w, h):
            normed = boxes.clone().float()
            normed[:, 0] /= w
            normed[:, 1] /= h
            normed[:, 2] /= w
            normed[:, 3] /= h
            return normed.clamp(0, 1)

        boxes_list = [
            normalize_boxes(orig_boxes, img_w, img_h).numpy(),
            normalize_boxes(flip_boxes, img_w, img_h).numpy(),
        ]
        scores_list = [orig_scores.numpy(), flip_scores.numpy()]
        labels_list = [orig_labels.numpy(), flip_labels.numpy()]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=0.5, skip_box_thr=0.01
        )

        # 0~1 → 원본 좌표로 복원
        fused_boxes[:, 0] *= img_w
        fused_boxes[:, 1] *= img_h
        fused_boxes[:, 2] *= img_w
        fused_boxes[:, 3] *= img_h

        return (
            torch.tensor(fused_boxes, dtype=torch.float32),
            torch.tensor(fused_labels, dtype=torch.int64),
            torch.tensor(fused_scores, dtype=torch.float32),
        )
    else:
        # WBF 없으면 단순 concat + torchvision NMS
        all_boxes = torch.cat([orig_boxes, flip_boxes], dim=0)
        all_scores = torch.cat([orig_scores, flip_scores], dim=0)
        all_labels = torch.cat([orig_labels, flip_labels], dim=0)

        keep = torchvision.ops.nms(all_boxes, all_scores, iou_threshold=0.5)
        return all_boxes[keep], all_labels[keep], all_scores[keep]


@torch.no_grad()
def generate_csv(model, test_img_dir, device, cat_id_map, output_path='submission.csv',
                 score_threshold=0.3, batch_size=4, use_tta=False):
    """
    test 이미지 전체를 추론하여 Kaggle 제출용 CSV를 생성한다.

    Args:
        use_tta: True면 TTA+WBF 적용, False면 기존 방식

    Kaggle 형식: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    """
    # mapped_id → original_id 역매핑
    reverse_map = {v: k for k, v in cat_id_map.items()}

    # 테스트 데이터셋 로드
    test_dataset = TestDataset(img_dir=test_img_dir, transforms=validation_transforms)

    model.eval()
    rows = []
    annotation_id = 1

    if use_tta:
        print(f"[TTA+{'WBF' if HAS_WBF else 'NMS'}] 추론 시작 (이미지 {len(test_dataset)}장)")
        # TTA는 이미지 1장씩 처리
        for img_idx in range(len(test_dataset)):
            image = test_dataset[img_idx]
            img_path = test_dataset.img_p_list[img_idx]
            image_id = os.path.splitext(os.path.basename(img_path))[0]

            boxes, labels, scores = predict_tta(model, image, device)

            keep = scores >= score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            for box, label, score in zip(boxes, labels, scores):
                original_id = reverse_map.get(int(label.item()), int(label.item()))
                x1, y1, x2, y2 = box.tolist()
                rows.append({
                    'annotation_id': annotation_id,
                    'image_id': image_id,
                    'category_id': original_id,
                    'bbox_x': round(x1, 1),
                    'bbox_y': round(y1, 1),
                    'bbox_w': round(x2 - x1, 1),
                    'bbox_h': round(y2 - y1, 1),
                    'score': round(float(score), 4),
                })
                annotation_id += 1

            if (img_idx + 1) % 100 == 0:
                print(f"  {img_idx + 1}/{len(test_dataset)}장 완료")
    else:
        # 기존 방식 (배치 처리)
        test_loader = test_build_dataloaders(test_dataset, batch_size=batch_size)
        img_idx = 0
        for images in test_loader:
            images_gpu = [img.to(device) for img in images]
            outputs = model(images_gpu)

            for output in outputs:
                img_path = test_dataset.img_p_list[img_idx]
                image_id = os.path.splitext(os.path.basename(img_path))[0]

                boxes = output['boxes'].cpu()
                labels = output['labels'].cpu()
                scores = output['scores'].cpu()

                keep = scores >= score_threshold
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                for box, label, score in zip(boxes, labels, scores):
                    original_id = reverse_map.get(label.item(), label.item())
                    x1, y1, x2, y2 = box.tolist()
                    rows.append({
                        'annotation_id': annotation_id,
                        'image_id': image_id,
                        'category_id': original_id,
                        'bbox_x': round(x1, 1),
                        'bbox_y': round(y1, 1),
                        'bbox_w': round(x2 - x1, 1),
                        'bbox_h': round(y2 - y1, 1),
                        'score': round(score.item(), 4),
                    })
                    annotation_id += 1

                img_idx += 1

    # CSV 저장
    fieldnames = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    img_count = len(test_dataset) if use_tta else img_idx
    print(f"CSV 저장 완료: {output_path} (이미지 {img_count}장, 객체 {len(rows)}개)")
    return output_path
