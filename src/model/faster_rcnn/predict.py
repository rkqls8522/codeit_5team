# 추론 + Kaggle CSV 생성
# Soft-NMS 적용해서 겹치는 박스 처리하고 제출용 csv 만듦

import os
import csv
import torch
import torchvision
from Faster_RCNN_dataset import TestDataset, validation_transforms
from Faster_RCNN_dataloader import test_build_dataloaders
from config import CONFIG


def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS (Gaussian decay) — 논문: Bodla et al., ICCV 2017
    Hard NMS 대비 +1.0~1.7 mAP 향상. 재학습 없이 추론만 바꾸면 됨.

    겹치는 박스를 제거하지 않고, IoU에 따라 score를 부드럽게 감소시킴.
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = boxes.float()
    scores = scores.float()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort(descending=True)
    keep_boxes = []
    keep_scores = []

    while len(order) > 0:
        i = order[0]
        keep_boxes.append(boxes[i])
        keep_scores.append(scores[i])

        if len(order) == 1:
            break

        remaining = order[1:]

        # IoU 계산
        xx1 = torch.clamp(x1[remaining], min=x1[i].item())
        yy1 = torch.clamp(y1[remaining], min=y1[i].item())
        xx2 = torch.clamp(x2[remaining], max=x2[i].item())
        yy2 = torch.clamp(y2[remaining], max=y2[i].item())

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        union = areas[i] + areas[remaining] - inter
        iou = inter / (union + 1e-6)

        # Gaussian decay: score *= exp(-iou^2 / sigma)
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[remaining] = scores[remaining] * decay

        # threshold 이하 제거
        mask = scores[remaining] >= score_threshold
        order = remaining[mask]
        # 다시 score 순으로 정렬
        if len(order) > 0:
            reorder = scores[order].argsort(descending=True)
            order = order[reorder]

    if len(keep_boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))

    return torch.stack(keep_boxes), torch.stack(keep_scores)


@torch.no_grad()
def predict(model, image, device, score_threshold=0.5):
    """단일 이미지에 대해 예측 수행."""
    model.eval()
    image = image.to(device)
    output = model([image])[0]

    keep = output['scores'] >= score_threshold
    return {
        'boxes': output['boxes'][keep].cpu(),
        'labels': output['labels'][keep].cpu(),
        'scores': output['scores'][keep].cpu(),
    }


def apply_nms(boxes, labels, scores, config):
    """NMS 또는 Soft-NMS 적용. 클래스별로 처리."""
    use_soft = config.get('use_soft_nms', False)
    sigma = config.get('soft_nms_sigma', 0.5)
    nms_threshold = config.get('nms_threshold', 0.5)

    if len(boxes) == 0:
        return boxes, labels, scores

    unique_labels = labels.unique()
    final_boxes = []
    final_labels = []
    final_scores = []

    for label in unique_labels:
        mask = labels == label
        label_boxes = boxes[mask]
        label_scores = scores[mask]

        if use_soft:
            kept_boxes, kept_scores = soft_nms(label_boxes, label_scores, sigma=sigma)
        else:
            keep = torchvision.ops.nms(label_boxes, label_scores, iou_threshold=nms_threshold)
            kept_boxes = label_boxes[keep]
            kept_scores = label_scores[keep]

        final_boxes.append(kept_boxes)
        final_labels.append(torch.full((len(kept_boxes),), label.item(), dtype=torch.int64))
        final_scores.append(kept_scores)

    if len(final_boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64), torch.zeros((0,))

    return torch.cat(final_boxes), torch.cat(final_labels), torch.cat(final_scores)


@torch.no_grad()
def generate_csv(model, test_img_dir, device, cat_id_map, output_path='submission.csv',
                 score_threshold=0.3, batch_size=4):
    """
    test 이미지 전체를 추론하여 Kaggle 제출용 CSV를 생성한다.
    Soft-NMS가 config에서 켜져 있으면 자동 적용.

    Kaggle 형식: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    """
    # mapped_id → original_id 역매핑
    reverse_map = {v: k for k, v in cat_id_map.items()}

    # 테스트 데이터셋 로드
    test_dataset = TestDataset(img_dir=test_img_dir, transforms=validation_transforms)
    test_loader = test_build_dataloaders(test_dataset, batch_size=batch_size)

    use_soft = CONFIG.get('use_soft_nms', False)
    print(f"[추론] NMS: {'Soft-NMS (sigma={})'.format(CONFIG.get('soft_nms_sigma', 0.5)) if use_soft else 'Hard NMS'}")

    model.eval()
    rows = []
    annotation_id = 1
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

            # Score threshold 적용
            keep = scores >= score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            # Soft-NMS 또는 Hard NMS 적용
            if use_soft and len(boxes) > 0:
                boxes, labels, scores = apply_nms(boxes, labels, scores, CONFIG)

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

    print(f"CSV 저장 완료: {output_path} (이미지 {img_idx}장, 객체 {len(rows)}개)")
    return output_path
