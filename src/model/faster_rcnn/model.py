import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    """
    COCO pretrained Faster R-CNN을 로드하고,
    classifier head를 우리 클래스 수에 맞게 교체한다.

    Args:
        num_classes: background 포함 전체 클래스 수
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
