import torch
from torchvision.ops import box_iou


@torch.no_grad()
def evaluate(model, data_loader, device, iou_threshold=0.5):
    """Validation 데이터에 대해 예측 수행 + mAP 계산."""
    model.eval()
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_gt_boxes = []
    all_gt_labels = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            all_pred_boxes.append(output['boxes'].cpu())
            all_pred_labels.append(output['labels'].cpu())
            all_pred_scores.append(output['scores'].cpu())
            all_gt_boxes.append(target['boxes'].cpu())
            all_gt_labels.append(target['labels'].cpu())

    # mAP 계산
    mAP, class_ap = compute_mAP(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_gt_boxes, all_gt_labels,
        iou_threshold=iou_threshold
    )

    return mAP, class_ap


def compute_ap(precision, recall):
    """Precision-Recall 커브에서 AP(Average Precision)를 계산한다."""
    # recall 양 끝에 0과 1 추가
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    # precision을 오른쪽에서 왼쪽으로 max 누적 (monotone decreasing으로 만들기)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # recall이 변하는 지점에서의 넓이 합산
    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            ap += (recall[i] - recall[i - 1]) * precision[i]

    return ap


def compute_mAP(pred_boxes_list, pred_labels_list, pred_scores_list,
                gt_boxes_list, gt_labels_list, iou_threshold=0.5):
    """
    전체 이미지에 대한 mAP를 계산한다.

    Returns:
        mAP: 전체 클래스 평균 AP
        class_ap: {class_id: AP} 딕셔너리
    """
    # 전체 클래스 수집
    all_classes = set()
    for labels in gt_labels_list:
        all_classes.update(labels.tolist())

    class_ap = {}

    for cls in sorted(all_classes):
        # 이 클래스에 해당하는 예측과 정답 수집
        scores = []
        matched = []
        n_gt = 0

        for img_idx in range(len(gt_boxes_list)):
            gt_boxes = gt_boxes_list[img_idx]
            gt_labels = gt_labels_list[img_idx]
            pred_boxes = pred_boxes_list[img_idx]
            pred_labels = pred_labels_list[img_idx]
            pred_scores = pred_scores_list[img_idx]

            # 이 클래스의 GT 박스
            gt_mask = gt_labels == cls
            gt_cls_boxes = gt_boxes[gt_mask]
            n_gt += len(gt_cls_boxes)

            # 이 클래스의 예측 박스
            pred_mask = pred_labels == cls
            pred_cls_boxes = pred_boxes[pred_mask]
            pred_cls_scores = pred_scores[pred_mask]

            if len(pred_cls_boxes) == 0:
                continue

            if len(gt_cls_boxes) == 0:
                # GT가 없는데 예측이 있으면 전부 False Positive
                for s in pred_cls_scores:
                    scores.append(s.item())
                    matched.append(False)
                continue

            # IoU 계산
            ious = box_iou(pred_cls_boxes, gt_cls_boxes)
            gt_matched = set()

            # score 높은 순으로 정렬
            sorted_idx = torch.argsort(pred_cls_scores, descending=True)
            for idx in sorted_idx:
                scores.append(pred_cls_scores[idx].item())
                max_iou, max_gt_idx = ious[idx].max(dim=0)

                if max_iou >= iou_threshold and max_gt_idx.item() not in gt_matched:
                    matched.append(True)
                    gt_matched.add(max_gt_idx.item())
                else:
                    matched.append(False)

        # AP 계산
        if n_gt == 0:
            class_ap[cls] = 0.0
            continue

        # score 기준 정렬
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        matched_sorted = [matched[i] for i in sorted_indices]

        tp = torch.cumsum(torch.tensor(matched_sorted, dtype=torch.float32), dim=0)
        fp = torch.cumsum(torch.tensor([not m for m in matched_sorted], dtype=torch.float32), dim=0)

        precision = tp / (tp + fp)
        recall = tp / n_gt

        class_ap[cls] = compute_ap(precision, recall)

    # mAP = 전체 클래스 AP의 평균
    if len(class_ap) == 0:
        return 0.0, {}

    mAP = sum(class_ap.values()) / len(class_ap)
    return mAP, class_ap
