#======================================================================================
# 이미지를 예측하기 전에 학습된 YOLO 모델을 불러온 후 실제로 객체를 탐지하는 기능을 수행하는 클래스
#   < 함수설명 >
# - __init__(): YOLO 모델파일을 읽어온 후 사용준비를 하는 함수
# - detect(): 준비된 YOLO모델한테 사진을 보여주고 어디에 알약이 있는지 찾아주라고 시키는 함
# 
# (YOLO 모델 로드 및 추론 클래스)
#======================================================================================

from ultralytics import YOLO
import yolo_config as config
import os

## YOLO모델 로드 후 입력 이미지에 대해 객체 탐지 수행 후 결과 후처리진행하여 반환하는 클래스
class DrugDetector:     # 모델 로드 및 추론 // 외부에서도 사용할 수 있도록 클래스 함수로 진행
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            print(f"학습 모델 불러오는 중: {model_path}")    # 학습파일 경로(실제 사용하는 model_path출력)
            self.model = YOLO(model_path)
        else:
            print(f"기본 모델 불러오는 중: {config.model_file}")    # 기본 모델 경로이기때문에 config.model_file사용
            self.model = YOLO(config.model_file)
        
        self.conf = config.conf_threshold       # 설정 값 지정 / 신뢰도 기준
        self.device = config.device             # 설정 값 지정 / CPU, GPU 설정

    # 입력 이미지 1장을 가져와서 YOLO 모델 추론 결과를 가공해서 반환하는 과정 -> predict를 이용해 이미지 객체 탐지 수행

    ## => 실제 이미지를 YOLO모델에게 입력 후 찾은 결과를 보고 분석하는 과정
    def detect(self, image, conf = None, iou = config.iou_threshold):
        if conf is None:
            conf = self.conf
        
        results = self.model.predict(source = image,        # 추론 대상 이미지          => 모델에게 사진을 찾아보라고 시킨 후 찾은 정보들을 results 변수에 저장
                                     conf = conf,      # self.conf값보다 낮은 신뢰도는 무시
                                     iou = iou,        # IoU 임계값
                                     device = self.device,  # CPU or GPU 사용
                                     verbose = False)       # 로그 출력 최소화
        
        final_list = []     # 최종 결과를 저장할 리스트

        for r in results:           # 모델이 이미지에서 찾은 결과(results에서)를 하나씩 꺼내서 정리
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
