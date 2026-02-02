import torch


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
