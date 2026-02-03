# 실행 파일
import cv2
import os
import yolo_config as config
from yolo_detector import DrugDetector
from yolo_utils import load_image, draw_box

def main():
    print("=== 약 분류 실행===")

    # [수정 1] 결과 저장할 폴더가 없으면 미리 만들기
    # yolo_config.py에서 정의한 INFERENCE_RESULT_DIR 경로를 사용
    if not os.path.exists(config.INFERENCE_RESULT_DIR):
        os.makedirs(config.INFERENCE_RESULT_DIR, exist_ok=True)
        print(f"폴더 확인: {config.INFERENCE_RESULT_DIR}")

    detector = DrugDetector()   # yolo_detector에서 모델 로드

    try:
        print(f"이미지 로드 중: {config.test_image_path}")  # 이미지 불러오기
        image = load_image(config.test_image_path)      # yolo_config에서 테스트 이미지 경로불러와서 -> load_image: Open CV형태로 읽은 후 반환

        print("약을 찾고 있는 중")
        detections = detector.detect(image)         # YOLO 모델 사용하여 객체 탐지 수행

        if len(detections) > 0:             # 탐지 결과를 바로 볼 수 있도록 콘솔 형태로 정리(클래스 이름, 신뢰도)
            print(f"\n성공: 총 {len(detections)}ea의 약 발견")
            for i, d in enumerate(detections):
                print(f"{i+1}, 이름: {d['class_name']} | 확신: {d['confidence'] * 100:.1f}%")
        else:
            print("\n결과: 약을 못 찾음")

        result_img = draw_box(image, detections)        # 박스그리기
        save_path = os.path.join(config.INFERENCE_RESULT_DIR, "result.png") # 결과 이미지 저장
        extension = os.path.splitext(save_path)[1]
        result, encoded_img = cv2.imencode(extension, result_img)

        # 인코딩된 데이터를 파일로 쓰기
        if result:
            with open(save_path, mode='w+b') as f:
                encoded_img.tofile(f)
            print(f"\n완료: 결과 이미지가 저장:\n -> {save_path}")
        else:
            print("\n실패: 이미지 인코딩 실패")

    except Exception as e:      # 이미지 로드 및 추론 과정 중 오류 처리부분
        print(f"\n에러 발생: {e}")

if __name__ == "__main__":      # 자동 실행 방지
    main()                      # YOLO 모델을 이용해 테스트 이미지에 대한 객체 탐지를 실행하고 결과를 저장하는 메인 함수