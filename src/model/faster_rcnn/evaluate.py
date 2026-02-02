import torch


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Validation 데이터에 대한 예측 수행. mAP 계산은 별도 구현 필요."""
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            results.append({
                'pred_boxes': output['boxes'].cpu(),
                'pred_labels': output['labels'].cpu(),
                'pred_scores': output['scores'].cpu(),
                'gt_boxes': target['boxes'],
                'gt_labels': target['labels'],
            })

    return results
