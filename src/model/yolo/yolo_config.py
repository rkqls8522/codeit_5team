#======================================================================================
# YOLO 모델을 학습 및 예측을 수행하기 전에 필요한 파일 경로와 하이퍼파라미터를 정의하는 과정
# (모델 경로, 클래스 이름, 임계값 등의 설정파일)
#======================================================================================
import os

# 1. 현재 파일의 위치
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

# 3. 결과 저장 경로 설정
TRAIN_RESULT_DIR = os.path.join(ROOT_DIR, 'results') 
INFERENCE_RESULT_DIR = os.path.join(ROOT_DIR, 'results', 'submission')
# 4. 데이터셋 경로 설정
data_yaml_path = os.path.join(CURRENT_DIR, 'data.yaml')

# 5. 학습 데이터셋 경로 설정
trained_model_path = os.path.join(TRAIN_RESULT_DIR, 'yolo_final_model', 'weights', 'best.pt')

# 6. 테스트 이미지 폴더 (배치 추론용)
test_images_dir = os.path.join(ROOT_DIR, 'data', 'original', 'images', 'test')

# 7. 제출 CSV 저장 경로
submission_csv_path = os.path.join(INFERENCE_RESULT_DIR, 'submission.csv')

# 8. mAP평가용 설정 
conf_threshold_submission = 0.001   # 모델평가점수를 높이기 위해 기준을 최대한 낮춤

# 9. IoU 임계값 설정 (같은 알약에 네모칸이 여러 개 겹칠 때, 50% 이상 겹치면 중복으로 보고 하나만 남김)
iou_threshold = 0.5 

# 초기 모델 파일이름 (현재 폴더에 위치하도록 설정)
model_file = os.path.join(CURRENT_DIR, 'yolov8s.pt')

# 임시, 테스트보기위해 하나만 지정
test_image_path = os.path.join(ROOT_DIR, 'data', 'original', 'images', 'test', '54.png')

# 민감도 설정
# 50% 이상 확실할 때 판단
conf_threshold = 0.3

# 디바이스 자동 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device 설정] {device}")