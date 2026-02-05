### 서브미션 생성 터미널 명령어: " python src/model/yolo/yolo_predict.py --mode submission " ###
# 추론 및 결과 저장 (이미지 + CSV)
import os
import csv
import re
import cv2
from datetime import datetime

import yolo_config as config
from yolo_detector import DrugDetector
from yolo_utils import load_image, draw_box

### 단일 이미지 예측 및 저장하는 함수(사진 한 장 넣고 -> 탐지 후 -> 결과를 저장까지 하는 과정)
def predict_and_save(image_path, output_dir=None, save_image=True, save_csv=True, model_path=None): # 이미지경로(image_path)를 받아서 모델로 객체를 찾고 결과를 파일과 csv파일로 저장하는 함수
    """
    단일 이미지에 대해 YOLO 모델 추론을 수행하고 결과를 저장합니다.
    
    Args:
        image_path: 입력 이미지 경로
        output_dir: 결과 저장 디렉토리 (기본값: config.INFERENCE_RESULT_DIR)
        save_image: 결과 이미지 저장 여부
        save_csv: 결과 CSV 저장 여부
        model_path: 사용할 모델 경로 (기본값: 학습된 모델 또는 기본 모델)
    
    Returns:
        dict: 추론 결과 정보
    """
    # 출력 디렉토리 설정(저장 폴더 설정)
    if output_dir is None:                      # output_dir에 None이면 -> config 파일에 지정한 기본 위(INFERENCE_RESULT_DIR)로 사용
        output_dir = config.INFERENCE_RESULT_DIR
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)      # 관련 폴더가 있는지 확인 후 없으면 생성(exist_ok=True: 폴더가 이미 있어도 오류 안남)
    
    # 모델 로드 (모델 경로 설정)
    if model_path is None and os.path.exists(config.trained_model_path):    # 만약 학습된 모델 파일이 있을 경우
        model_path = config.trained_model_path                              # config.trained_model_path(학습시킨 best.pt)를 사용한다는 설정
    
    detector = DrugDetector(model_path)                # DrugDetector 클래스의 객체를 생성
    image = load_image(image_path)                     # load_image 함수를 이용해서 이미지 불러오기
    
    # 실제 추론 수행
    results = detector.detect(image)                    # 이미지 속 객체를 찾아 results에 좌표, 이름, 점수를 리스트 형태로 넣음
    
    # 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]       # 원본 파일 경로에서 파일명만 떼고(os.path.basename) -> splitext: 확장자(.jpg를 뗌)
                                                     # => 파일 명 중간 확장자가 들어가는 걸 막은 후 "이름+결과표시+화장자"조합으로 하기 위해 하는 작업(파일명을 깔끔하게 하기 위한 작업 및 결과 이미지의 화질 보호)
    # 결과 이미지 저장(save_image가 True일때만 저장)
    if save_image:
        result_image = draw_box(image, results)         # 원본 이미지 위에 박스 그리기
        image_output_path = os.path.join(output_dir, f"{base_name}_result_{timestamp}.png")     # 저장할 전체 경로 생성: 폴더/파일명_result_시간.png
        cv2.imwrite(image_output_path, result_image)    # OpenCV를 이용해서 실제 파일로 저장
        print(f"결과 이미지 저장 완료: {image_output_path}")
    
    # CSV 파일 저장(save_csv가 True일때만 저장)
    if save_csv:
        csv_output_path = os.path.join(output_dir, f"{base_name}_result_{timestamp}.csv")       # 저장할 CSV파일 경로 생
        save_results_to_csv(results, csv_output_path, image_path)
        print(f"CSV 파일 저장 완료: {csv_output_path}")
    
    return {
        "image_path": image_path,           # 분석한 파일 경로
        "detections": results,              # 분석 결과(찾은 객체 정보)
        "num_detections": len(results),       # 찾은 객체 수
        "output_dir": output_dir            # 결과 저장 경로(위)
    }

### 여러 장을 한꺼번에 처리하기 위한 배치 추론부분
def predict_batch(image_dir, output_dir=None, extensions=None, model_path=None):
    """
    디렉토리 내 모든 이미지에 대해 배치 추론을 수행합니다.
    
    Args:
        image_dir: 이미지가 있는 디렉토리 경로
        output_dir: 결과 저장 디렉토리 (기본값: config.INFERENCE_RESULT_DIR)
        extensions: 처리할 이미지 확장자 목록 (기본값: ['.jpg', '.jpeg', '.png', '.bmp'])
        model_path: 사용할 모델 경로
    
    Returns:
        list: 각 이미지별 추론 결과
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if output_dir is None:          # 출력 폴더 없을 경우 기본 설정값 사용
        output_dir = config.INFERENCE_RESULT_DIR        
    
    # 저장폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드 (학습된 모델 우선)
    if model_path is None and os.path.exists(config.trained_model_path):
        model_path = config.trained_model_path
    
    detector = DrugDetector(model_path)
    
    # 이미지 파일 목록 가져오기(처리할 파일 리스트 만드는 작업)
    image_files = []
    for f in os.listdir(image_dir):         # 폴더 내 파일 목록 가져오기
        ext = os.path.splitext(f)[1].lower()    # 파일 확장자만 떼어내서 소문자로 변경(.PNG -> .png)
        if ext in extensions:                   # 이미지 파일이 맞으면
            image_files.append(os.path.join(image_dir, f))      # 리스트에 추가해달라고 하는 부분
    
    if not image_files:                       # 만약 이미지가 하나도 없을 경우, 경고 메시지 출력
        print(f"경고: {image_dir}에서 이미지를 찾을 수 없습니다.")
        return []
    
    print(f"총 {len(image_files)}개 이미지 추론 시작...")
    
    ### 결과 저장 준비 구간(CSV 파일 생성)
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 전체 결과를 저장할 CSV
    batch_csv_path = os.path.join(output_dir, f"batch_results_{timestamp}.csv")
    
    # 파일을 쓰기(w)모드로 열고, CSV 파일의 헤더를 추가
    with open(batch_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['이미지파일', '클래스명', '신뢰도', 'x1', 'y1', 'x2', 'y2'])
        
        ### 반복문을 통해서 이미지 하나씩 처리
        for idx, img_path in enumerate(image_files, 1):             # enumerate: 인덱스와 파일경로를 동시에 가져옴
            print(f"[{idx}/{len(image_files)}] 처리 중: {os.path.basename(img_path)}")
            
            try:
                # 이미지 로드 및 추론(실제 추론 수행)
                image = load_image(img_path)
                results = detector.detect(image)
                
                # 결과 이미지 저장(박스 그려서 저장)
                result_image = draw_box(image, results)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                image_output_path = os.path.join(output_dir, f"{base_name}_result.png")
                cv2.imwrite(image_output_path, result_image)
                
                # 객체들을 하나씩 꺼내서 CSV에 결과 추가
                for det in results:
                    x1, y1, x2, y2 = det['bbox']
                    writer.writerow([
                        os.path.basename(img_path),         # 어떤 이미지에서 나왔는지
                        det['class_name'],                  # 약 이름
                        f"{det['confidence']:.4f}",        # 신뢰도
                        x1, y1, x2, y2                      # 위치(좌표)
                    ])
                
                # 결과 모으는 구간(함수가 끝날 때 반환하기 위해서)
                all_results.append({
                    "image_path": img_path,
                    "detections": results,
                    "num_detections": len(results)
                })
                
            except Exception as e:              # 파일오류(깨지거나, 에러)가 있어도 멈추지 않고 계속 진행 
                print(f"오류 발생 ({img_path}): {e}")
                continue
    
    print(f"\n배치 추론 완료!")
    print(f"- 총 처리 이미지: {len(all_results)}개")
    print(f"- 결과 이미지 저장 위치: {output_dir}")
    print(f"- 배치 CSV 저장 위치: {batch_csv_path}")
    
    return all_results

# 결과 csv로 저장
def save_results_to_csv(results, csv_path, image_path=None):
    """
    단일 이미지 추론 결과를 CSV 파일 한 개로 저장해 줍니다.
    
    Args:
        results: 추론 결과 리스트
        csv_path: 저장할 CSV 파일 경로
        image_path: 원본 이미지 경로 (선택사항)
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 첫 줄 제목 작성
        writer.writerow(['이미지파일', '클래스명', '신뢰도', 'x1', 'y1', 'x2', 'y2'])
        
        # 결과 작성
        img_name = os.path.basename(image_path) if image_path else "unknown"        # 이미지 파일 이름 가져오기(파일 경로가 없을 경우, unknown으로 표시)
        for det in results:                                                         # 찾은 객체만큼 반복해서 작성
            x1, y1, x2, y2 = det['bbox']
            writer.writerow([
                img_name,
                det['class_name'],
                f"{det['confidence']:.4f}",
                x1, y1, x2, y2
            ])

### 제출용 CSV 생성
def create_submission_csv(image_dir, output_path=None, model_path=None, conf = 0.001):
    """
    제출용 CSV 파일을 생성합니다.
    
    형식: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    
    Args:
        image_dir: 테스트 이미지가 있는 디렉토리
        output_path: 출력 CSV 경로 (기본값: config.submission_csv_path)
        model_path: 사용할 모델 경로 (기본값: 학습된 모델)
    
    Returns:
        str: 생성된 CSV 파일 경로
    """
    # 저장 경로 설정
    if output_path is None:
        output_path = config.submission_csv_path
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)        # 파일이 저장될 폴더가 있는지 확인 후 없으면 생성
    
    # 모델 로드 (학습된 모델 우선 사용)
    if model_path is None and os.path.exists(config.trained_model_path):
        model_path = config.trained_model_path
    
    detector = DrugDetector(model_path)
    
    # 이미지 파일 찾기
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for f in os.listdir(image_dir):         # 폴더를 확인해서 이미지 파일만 골라옴
        ext = os.path.splitext(f)[1].lower()
        if ext in extensions:
            image_files.append(os.path.join(image_dir, f))
    
    if not image_files:
        print(f"경고: {image_dir}에서 이미지를 찾을 수 없습니다.")
        return None
    
    print(f"총 {len(image_files)}개 이미지에 대해 제출용 CSV 생성 중...")
    
    ### 제출용 파일 작성
    annotation_id = 1  # 1번부터 시작해서 1씩 증가
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 제출 형식 제목 작성
        writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] 처리 중: {os.path.basename(img_path)}")
            
            try:
                # 파일명에서 숫자ID만 추출
                filename = os.path.basename(img_path)
                image_id = re.sub(r'\D', '', os.path.splitext(filename)[0])  # 숫자만 추출(ex: 123.png → 123)
                
                if not image_id:  # 숫자가 없으면 파일명 그대로 사용(ex: test.png → test)
                    image_id = os.path.splitext(filename)[0]
                
                # 추론 진행
                image = load_image(img_path)
                results = detector.detect(image, conf = conf)   # 신뢰도 기준을 낮출 수 있게 설정하였으나 변경가능예정
                
                for det in results:
                    if det['class_name'] == 'background':       # background(0번 클래스)는 저장하지 않고 건너뜀
                        continue
                    x1, y1, x2, y2 = det['bbox']
                    # x1,y1,x2,y2 → bbox_x, bbox_y, bbox_w, bbox_h 변환
                    bbox_x = x1
                    bbox_y = y1
                    bbox_w = x2 - x1    # 폭 = 오른쪽 - 왼쪽
                    bbox_h = y2 - y1    # 높이 = 아래쪽 - 위쪽
                    
                    writer.writerow([
                        annotation_id,      # 고유번호
                        image_id,           # 이미지 번호
                        det['class_name'], # 약 종류 번호(class_id)
                        bbox_x,             # 왼쪽 x좌표
                        bbox_y,             # 위쪽 y좌표
                        bbox_w,             # 폭
                        bbox_h,             # 높이
                        f"{det['confidence']:.4f}"      # 점수는 소수점 4자리까지 표시
                    ])
                    annotation_id += 1          # 다음 줄을 위한 번호 하나 증
                    
            except Exception as e:
                print(f"오류 발생 ({img_path}): {e}")
                continue
    
    print(f"\n제출용 CSV 생성 완료!")
    print(f"- 총 annotation 수: {annotation_id - 1}개")
    print(f"- 저장 위치: {output_path}")
    
    return output_path

### 결과 요약 함수(통계)
def get_detection_summary(results):
    """
    추론 결과 요약 정보를 반환합니다.
    
    Args:
        results: 추론 결과 리스트
    
    Returns:
        dict: 클래스별 탐지 개수 및 평균 신뢰도
    """
    summary = {}  # 빈 딕셔너리 생성
    
    for det in results:
        class_name = det['class_name']
        confidence = det['confidence']
        
        # 만약 처음 보는 약 이름일 경우 목록에 추가
        if class_name not in summary:
            summary[class_name] = {'count': 0, 'total_conf': 0}
        
        # 점수 합산을 위해 개수 1개 추가(평균 내기 위해서)
        summary[class_name]['count'] += 1
        summary[class_name]['total_conf'] += confidence
    
    # 평균 신뢰도 계산
    for class_name in summary:
        count = summary[class_name]['count']        # 총점 / 개수 = 평균
        summary[class_name]['avg_confidence'] = summary[class_name]['total_conf'] / count
        del summary[class_name]['total_conf']       # total_conf는 평균 계산에 필요없어서 삭제
    
    return summary


### 메인 실행 부분
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO 모델 추론')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'batch', 'submission'],          # mode 옵션: single(한장), batch(여러장), submission(제출용) 중 하나 선택
                        help='실행 모드: single(단일), batch(배치), submission(제출용)')
    parser.add_argument('--image', type=str, default=None,                  # single 모드일 때 사용할 이미지 경로
                        help='단일 이미지 경로 (single 모드용)')
    parser.add_argument('--image_dir', type=str, default=None,              # batch/submission 모드일 때 사용할 이미지 폴더 경로
                        help='이미지 디렉토리 경로 (batch/submission 모드용)')
    parser.add_argument('--output', type=str, default=None,                  # 출력 경로
                        help='출력 경로')
    parser.add_argument('--model', type=str, default=None,                  # 사용할 모델 경로
                        help='사용할 모델 경로')
    
    args = parser.parse_args()          # 입력받은 인자들을 args에 저장
    
    print("=" * 50)
    print("YOLO 모델 추론")
    print("=" * 50)
    
    ### 모드에 따른 기능 실행
    if args.mode == 'single':
        # 단일 이미지 추론
        image_path = args.image if args.image else config.test_image_path       # 입력받은 이미지가 있을 경우 사용, 없을 경우 기본 테스트 이미지 사용
        
        if os.path.exists(image_path):
            result = predict_and_save(image_path, model_path=args.model)
            print(f"\n탐지된 객체 수: {result['num_detections']}")
            
            if result['detections']:
                summary = get_detection_summary(result['detections'])
                print("\n[탐지 결과 요약]")
                for class_name, info in summary.items():
                    print(f"  - {class_name}: {info['count']}개 (평균 신뢰도: {info['avg_confidence']:.2%})")
        else:
            print(f"이미지를 찾을 수 없습니다: {image_path}")
        
    elif args.mode == 'batch':
        # 배치 추론(폴더 자체로)
        image_dir = args.image_dir if args.image_dir else config.test_images_dir
        
        if os.path.exists(image_dir):
            results = predict_batch(image_dir, model_path=args.model)
            print(f"\n총 {len(results)}개 이미지 처리 완료")
        else:
            print(f"디렉토리를 찾을 수 없습니다: {image_dir}")
        
    elif args.mode == 'submission':
        # 제출용 CSV 파일 생성
        image_dir = args.image_dir if args.image_dir else config.test_images_dir
        output_path = args.output if args.output else config.submission_csv_path
        
        if os.path.exists(image_dir):
            create_submission_csv(image_dir, output_path, model_path=args.model)
        else:
            print(f"디렉토리를 찾을 수 없습니다: {image_dir}")