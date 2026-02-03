# 모델 경로, 클래스 이름, 임계값 등의 설정파일
import os

# 1. 현재 파일(yolo_config.py)의 위치: src/model/yolo/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 프로젝트 루트 경로 찾기 (부모의 부모의 부모 폴더)
# src/model/yolo/ -> src/model/ -> src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

# 3. 결과 저장 경로 설정
TRAIN_RESULT_DIR = os.path.join(ROOT_DIR, 'results') 
INFERENCE_RESULT_DIR = os.path.join(ROOT_DIR, 'results', 'figures')

# 초기 모델 파일이름
model_file = 'yolov8n.pt'

# 임시, 테스트보기위해 하나만 지정
test_image_path = os.path.join(ROOT_DIR, 'data', 'test_images', '54.png')

# 민감도 설정
# 50% 이상 확실할 때 판단 => 나중 조정 예정(임시조치)
conf_threshold = 0.5

# 실행 시, "CUDA not available"에러 시 "cpu"로 변경
device = "cpu"