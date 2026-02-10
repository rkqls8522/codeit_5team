#======================================================================================
# 모델이 예측을 하기 전에 이미지를 불러오거나 박스를 그리게하는 보조 도구 모음 파일
#   <함수설명>
# -  load_image(): 모델에 입력할 이미지를 불러오기 위한 함수화
# -  draw_box(): 모델이 불러온 이미지에 bbox를 그리는 함수화(파일 저장X)
# 
# (이미지 그림 및 파일 불러오기를 위한 보조도구)
#======================================================================================
import cv2  # 이미지 불러오기

def load_image(path):
    img = cv2.imread(path)      # path경로에 있는 사진읽음
    if img is None:             # 만약 없을 경우, 밑의 이미지지없음 표시
        raise FileNotFoundError(f"이미지를 못 찾았습니다:{path}")
    return img

## 이미지에 bbox진행(네모칸)
def draw_box(image, results):       # (원본 이미지, 모델 예측 결과 목록(알약위치, 이름, 확률))
    img_copy = image.copy()         # 원본사용 시 망가질 수도 있어서 복사본 생성
    
    for item in results:
        x1, y1, x2, y2 = item['bbox']   # bbox: (왼쪽, 위, 오른쪽, 아래)
        name = item['class_name']       # 모델 예측 클래스 이름
        conf = item['confidence']       # 예측이 맞을 확률 같은 값(신뢰도)

        # 박스 그리는 부분
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # (x1, y1)에서 (x2, y2)까지 초록색으로 (0, 255, 0) 두께는 2로 그림
                    # img_copy: 그림 그릴 이미지 / (x1, y1): 시작점 / (x2, y2): 끝점
        text = f"{name} {conf:.2f}"     # 객체 확인 및 신뢰값이 얼마나되는지 확인
        cv2.putText(img_copy, text, (x1, y1 - 10),              # (복사본 이미지/ 문자열(클래스이름 + 신뢰도)/ 텍스트표시 위치좌표(x1, y1: 왼쪽 위, -10: 글자가 보일 수 있도록 약간 위로 지정)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # cv2.FONT_HERSHEY_SIMPLEX: 글꼴 지정/ 0.5: 글자 크기)
    return img_copy