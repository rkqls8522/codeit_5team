import os
import csv
import torch
from Faster_RCNN_dataset import TestDataset, validation_transforms
from Faster_RCNN_dataloader import test_build_dataloaders


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
def generate_csv(model, test_img_dir, device, cat_id_map, output_path='submission.csv',
                 score_threshold=0.3, batch_size=4):
    """
    test 이미지 전체를 추론하여 Kaggle 제출용 CSV를 생성한다.

    Kaggle 형식: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    - 각 행 = 객체 1개
    - image_id = 파일명 숫자
    - bbox = [x, y, w, h] (모델 출력 [x1,y1,x2,y2]에서 변환)
    - category_id = 원본 클래스 ID
    """
    # mapped_id → original_id 역매핑
    reverse_map = {v: k for k, v in cat_id_map.items()}

    # 테스트 데이터셋 로드
    test_dataset = TestDataset(img_dir=test_img_dir, transforms=validation_transforms)
    test_loader = test_build_dataloaders(test_dataset, batch_size=batch_size)

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

            keep = scores >= score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            for box, label, score in zip(boxes, labels, scores):
                original_id = reverse_map.get(label.item(), label.item())
                x1, y1, x2, y2 = box.tolist()
                # [x1, y1, x2, y2] → [x, y, w, h]
                bbox_x = x1
                bbox_y = y1
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                rows.append({
                    'annotation_id': annotation_id,
                    'image_id': image_id,
                    'category_id': original_id,
                    'bbox_x': round(bbox_x, 1),
                    'bbox_y': round(bbox_y, 1),
                    'bbox_w': round(bbox_w, 1),
                    'bbox_h': round(bbox_h, 1),
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
