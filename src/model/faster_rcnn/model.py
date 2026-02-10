import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_model(num_classes, backbone='resnet50', score_threshold=0.5, nms_threshold=0.5):
    """
    Faster R-CNN 모델을 로드하고, classifier head를 우리 클래스 수에 맞게 교체한다.

    Args:
        num_classes: background 포함 전체 클래스 수
        backbone: 'resnet50' | 'resnet50_v2' | 'resnet101' | 'mobilenet_v3'
        score_threshold: 추론 시 이 confidence 이하인 예측 필터링
        nms_threshold: NMS에서 겹치는 박스 제거 기준 IoU
    """
    # backbone 선택
    if backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    elif backbone == 'resnet50_v2':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    elif backbone == 'resnet101':
        backbone_net = resnet_fpn_backbone('resnet101', weights='DEFAULT')
        model = FasterRCNN(backbone_net, num_classes=num_classes)
    elif backbone == 'mobilenet_v3':
        model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
    else:
        raise ValueError(f"지원하지 않는 backbone: {backbone}")

    # classifier head 교체 (resnet101은 FasterRCNN 생성 시 num_classes 지정 완료)
    if backbone != 'resnet101':
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 추론 threshold 설정
    model.roi_heads.score_thresh = score_threshold
    model.roi_heads.nms_thresh = nms_threshold

    return model
