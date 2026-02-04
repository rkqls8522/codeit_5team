# YOLO 모델 로드 및 추론 클래스
from ultralytics import YOLO
import yolo_config as config
import os

## YOLO모델 로드 후 입력 이미지에 대해 객체 탐지 수행 후 결과 후처리진행하여 반환하는 클래스
class DrugDetector:     # 모델 로드 및 추론
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            print(f"학습 모델 불러오는 중: {config.model_file}")    # config.model_file: 학습파일 경로
            self.model = YOLO(model_path)
        else:
            print(f"기본 모델 불러오는 중: {config.model_file}")
            self.model = YOLO(config.model_file)
        
        self.conf = config.conf_threshold       # 설정 값 지정 / 신뢰도 기준
        self.device = config.device             # 설정 값 지정 / CPU, GPU 설정

    # 입력 이미지 1장을 가져와서 YOLO 모델 추론 결과를 가공해서 반환하는 과정 -> predict를 이용해 이미지 객체 탐지 수행
    def detect(self, image, conf = None, iou = 0.5):
        if conf is None:
            conf = self.conf
        
        results = self.model.predict(source = image,        # 추론 대상 이미지
                                     conf = conf,      # self.conf값보다 낮은 신뢰도는 무시
                                     iou = iou,        # IoU 임계값
                                     device = self.device,  # CPU or GPU 사용
                                     verbose = False)       # 로그 출력 최소화
        
        final_list = []     # 최종 결과를 저장할 리스트

        for r in results:
            boxes = r.boxes 
            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # xyxy = (x1, y1): 왼쪽 위, (x2, y2): 오른쪽 아래
                x1, y1, x2, y2 = map(int, xyxy) # 픽셀좌표로 사용하기 위해 정수로 변환(YOLO모델 tensor형태)
                cls_id = int(box.cls[0])    # 예측된 클래스 번호
                class_name = self.model.names[cls_id]   # 위의 클래스번호에서 실제 이름으로 변환
                score = float(box.conf[0])   # 해당 약에 대한 예측 신뢰도
                final_list.append({
                    "class_name": class_name,
                    "category_id": cls_id,
                    "confidence": score,
                    "bbox": [x1, y1, x2, y2]})
        return final_list