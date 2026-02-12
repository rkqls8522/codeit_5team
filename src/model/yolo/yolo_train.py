#======================================================================================
# 준비된 데이터(알약사진과 정답사진)를 이용해서 YOLO모델을 학습시키는 과정
#  < 함수설명>
# - prepare_data(): 공부 시작 전, train과 test 데이터를 나누는 준비하는 함수
# - train(): 학습을 위한 함수
# (모델 학습)
#======================================================================================
from ultralytics import YOLO
import yolo_config as config
from datetime import datetime
import os
import time
import sys
import shutil
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. data_engineer 모듈 경로 먼저 추가 (그래야 함수를 쓸 수 있음)
DATA_ENGINEER_DIR = os.path.join(config.ROOT_DIR, 'src', 'data_engineer')
sys.path.append(DATA_ENGINEER_DIR)
# 2. data_engineer 모듈 임포트
try:
    from make_YOLO_annotation import make_YOLO_annotation
    from yolo_data_split import split_yolo_dataset
    from class_mapping import read_classID
    print("data_engineer 모듈 로드 성공")

    print(f"   └─ [확인] 전처리 모듈 경로: {os.path.abspath(DATA_ENGINEER_DIR)}")       

except ImportError as e:
    print(f"data_engineer 모듈 로드 실패: {e}")
    sys.exit(1)
# 3. 데이터 준비 함수 정의
def prepare_data():
    """YOLO 학습을 위한 데이터 준비 (어노테이션 변환 및 분할)"""
    print("\n[1/2] 데이터 준비 시작")
    
    # 경로 설정 (data/original 기준)
    master_dir = os.path.join(config.ROOT_DIR, 'data', 'original')
    image_dir = os.path.join(master_dir, "images", "train")
    annotation_dir = os.path.join(master_dir, "train_annotations")

    print("="*60)                                                       
    print(f"▶ [1단계] 원본 데이터 경로 확인")                           
    print(f" - Image 경로: {os.path.abspath(image_dir)}")               
    print(f" - Annotation 경로: {os.path.abspath(annotation_dir)}")       
    print(" (만약 이 경로가 틀리면 학습 시작 전에 바로 멈춥니다!)")      
    print("="*60)                                                      

    # 분할된 데이터셋 저장할 위치 (새로 만들 곳)
    split_dir = os.path.join(config.ROOT_DIR, 'data', 'yolo_dataset')
    
    if not os.path.exists(annotation_dir):
        print(f"경고: 어노테이션 폴더를 찾을 수 없습니다: {annotation_dir}")
        return
        
    # ClassID 읽기
    try:
        class_dict = read_classID(DATA_ENGINEER_DIR)        # 경로 수정
    except Exception as e:
        print(f"ClassID 로드 실패: {e}")
        return

    print("ClassID 정보를 이용해 data.yaml을 생성합니다...")

    # 1. YOLO용 names 딕셔너리 생성 (yolo_id: name)
    yolo_names = {}
    for v in class_dict.values():
        yolo_names[v['yolo_id']] = v['name']
    # 2. yaml에 들어갈 내용 구성 (절대 경로 사용 권장)
    data_config = {
        'path': split_dir,  
        'train': 'images/train',     
        'val': 'images/val',         
        'test': 'images/test',       
        'nc': len(yolo_names),       
        'names': yolo_names          
    }
    # 3. data.yaml 파일 덮어쓰기
    with open(config.data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
        
    print(f"data.yaml 갱신 완료 (Classes: {len(yolo_names)})")
        
    # YOLO 어노테이션 생성
    print("YOLO 어노테이션 생성 중")
    yolo_annt_dir = make_YOLO_annotation(image_dir, annotation_dir, class_dict, "YOLO_annotation")
    
    # Train/Val 분할
    print("Train/Val 데이터 분할 중")
    split_yolo_dataset(image_dir=image_dir, anntation_dir=yolo_annt_dir, output_dir=split_dir, val_ratio=0.2)

    print("="*60)                                                                
    print(f"▶ [2단계] 변환된 데이터셋 저장 위치 확인")                          
    print(f" - 저장 경로: {os.path.abspath(split_dir)}")                       
    print(f" - 이곳에 Train/Val 폴더가 8:2로 정확하게 나뉘어 저장됩니다.")          
    print("="*60)                                                             

    print("데이터 준비 완료!\n")
    # 주의: 여기서 자기 자신(prepare_data)을 절대 호출하면 안 됨!
def train(resume=False):
    """
    YOLO 모델 학습 수행
    """
    # 4. 학습 시작 전에 데이터 준비 실행 (여기서 호출!)
    prepare_data()
    model = YOLO(config.model_file)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'yolo_train_{timestamp}'
    
    # 학습 시간 측정 시작
    start_time = time.time()
    print(f"학습 시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model.train(data=config.data_yaml_path,     # ← config에서 가져오기
                epochs=50,
                imgsz=640,
                device=config.device,
                batch=10,
                patience=10,
                project=config.TRAIN_RESULT_DIR,
                name=run_name,
                exist_ok=True,                  
                resume=resume)
    
    # 학습 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 시/분/초로 변환
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"학습 종료 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}") 
    print(f"총 학습 소요 시간: {hours}시간 {minutes}분 {seconds}초 (총 {elapsed_time:.2f}초)") 
    
    # 학습된 모델 경로 반환
    best_model_path = os.path.join(config.TRAIN_RESULT_DIR, run_name, 'weights', 'best.pt')
    print(f"학습 완료, Best 모델 저장 위치: {best_model_path}")

    # [추가] 추론(Inference) 편의를 위해 최신 모델을 기본 경로(yolo_final_model)로 복사
    final_fixed_path = config.trained_model_path  # .../results/yolo_final_model/weights/best.pt

    # 복사할 폴더가 없으면 생성
    os.makedirs(os.path.dirname(final_fixed_path), exist_ok=True)

    # 파일 복사
    shutil.copy(best_model_path, final_fixed_path)
    print(f"[자동갱신] 최신 모델이 기본 경로로 복사되었습니다: {final_fixed_path}")
    print("이제 yolo_predict.py를 실행하면 자동으로 이 모델이 사용됩니다.")

    return best_model_path
if __name__ == "__main__":
    train()

### 학습실행 시, 터미널 명령어: python src/model/yolo/yolo_train.py ###