# 프로젝트 제목

헬스잇5팀 - 약 이미지에서 실시간 클래스 검출 모델 개발

---

# 개요 & 문제 정의

스마트폰으로 촬영한 알약 이미지에서 최대 4개의 알약을 탐지하고,
각 알약의 클래스(종류)와 위치(바운딩 박스)를 예측하는 객체 탐지 모델을 개발합니다.

주요 요구사항:

- 다양한 조명·배경에 강건할 것
- 한 장에 최대 4개 객체를 안정적으로 탐지할 것
- 실무 적용을 고려한 추론 속도와 정확도의 균형

---

# 핵심 접근 방식

- 데이터 전처리: 원본 COCO-style JSON 어노테이션을 정제하고 YOLO 포맷으로 변환한 뒤 Train/Val/Test로 분할합니다 (`src/data_engineer`).
- 모델 실험: 여러 YOLO 버전(YOLOv8, YOLO11)을 테스트해 성능·속도 균형을 검토했습니다.
- 하이퍼파라미터 튜닝: Optuna를 사용해 학습률, 배치 크기, 에폭 수 등을 최적화했습니다.
- 학습: Ultralytics API로 진행(예: `imgsz=640, epochs=50, batch=16`)
- 평가: Ultralytics `val()` 메서드로 mAP@0.5, mAP@0.5:0.95, precision, recall 계산하고, 추론 로그로 latency/FPS 측정합니다.

---

# 기술 스택

- 언어: Python 3.11
- 프레임워크: PyTorch, torchvision, Ultralytics YOLOv8
- 하이퍼파라미터 튜닝: Optuna
- 이미지 처리: OpenCV, Pillow
- 데이터: NumPy, Pandas, YAML
- 환경: Conda, CUDA (권장), Jupyter Notebook, Git

---

# 역할 담당자

- **프로젝트 매니저**: 노가빈
- **데이터 엔지니어**: 노가빈, 박수성
- **모델 설계 및 실험**:
  - YOLO 모델: 정주희
  - Faster R-CNN 모델: 강하은

---

# 개발 일정

| 기간        | 작업 내용                                                  |
| ----------- | ---------------------------------------------------------- |
| 1/29 ~ 1/30 | 요구사항 분석, 프로젝트 계획 수립, 환경 셋팅, 팀 규칙 논의 |
| 2/2 ~ 2/4   | 데이터 이해 및 전처리, 모델 초기 실험                      |
| 2/5 ~ 2/10  | 모델 튜닝 및 개선                                          |
| 2/11        | 최종 실험 및 결과 정리                                     |
| 2/12 ~ 2/13 | 제출물 정리 및 발표 자료 제작                              |

---

# 프로젝트 구조

```
📦코드잇-5팀-초급프로젝트-헬스잇
 ┣ 📂data
 ┃ ┣ 📂original                 # 원본 데이터 (이미지, 어노테이션)
 ┃ ┣ 📂processed                # 전처리 완료된 데이터
 ┃ ┗ 📜README.md                # 데이터 설명과 출처
 ┣ 📂docs                       # 보고서 및 발표 자료
 ┣ 📂env                        # 가상환경 설정 (requirements.txt, environment.yml)
 ┣ ┗ 📜README.md
 ┣ 📂notebooks                  # Jupyter Notebook (EDA, 테스트 등)
 ┣ 📂results                    # 학습 결과, 모델 가중치, 제출 파일
 ┃ ┣ 📂submission               # 제출용 CSV 파일들
 ┃ ┗ 📂yolo_train_*             # YOLO 각 실험별 결과 (weights, args.yaml, results.csv)
 ┣ 📂src                        # 메인 코드
 ┃ ┣ 📂data_engineer            # 데이터 전처리 및 파이프라인
 ┃ ┗ 📂model                    # 모델 개발
 ┃   ┣ 📂yolo                   # YOLO 학습/추론 코드
 ┃   ┗ 📂faster_rcnn            # Faster R-CNN 학습/평가/예측 코드
 ┣ 📜.gitignore
 ┣ 📜README.md                  # 프로젝트 설명
```

---

# 디렉터리 구조

- `data/` : 원본 및 전처리된 데이터 파일을 저장합니다.
- `docs/` : 보고서 등 문서 자료가 있습니다.
- `env/` : 가상환경 설정 파일(`requirements.txt` 또는 `environment.yml`)을 포함합니다.
- `notebooks/` : 데이터 분석 기록을 위한 Jupyter Notebook을 모아둡니다.
- `results/` : 실험 결과, 로그, 제출용 파일 등을 저장하는 폴더입니다.
- `src/` : 프로젝트의 코드가 위치하는 주요 폴더입니다.
  - `data_engineer/` : 데이터 전처리 및 파이프라인 관련 코드
  - `model/` : 모델 설계, 학습, 튜닝, 평가 등의 코드가 포함됩니다.
- `.gitignore` : Git 추적 제외 파일 목록
- `README.md` : 프로젝트 개요 및 사용법 안내 문서

---

# 사용한 모델 및 실험

- YOLO 실험
  - 프레임워크: `ultralytics` (YOLOv8)
  - 사용 모델 파일: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11x.pt` (사전학습 가중치에서 파인튜닝)
  - 학습 스크립트: [src/model/yolo/yolo_train.py](src/model/yolo/yolo_train.py#L1)
  - 특징: 빠른 학습/추론, 데이터 전처리 모듈(`src/data_engineer`)과 연동

- Faster R-CNN 실험
  - 프레임워크: `torch` + `torchvision` (fasterrcnn_resnet50_fpn 등)
  - 학습 스크립트: [src/model/faster_rcnn/train.py](src/model/faster_rcnn/train.py#L1)
  - 특징: 높은 정확도(특히 작은 객체에서 유리), 학습 안정성 확보를 위해 scheduler/SGD 사용

---

# 데이터 및 전처리

- 원본: `data/original` 폴더에 이미지와 COCO-style JSON 어노테이션이 위치합니다.
- 전처리: 어노테이션 정제, YOLO 포맷 변환, Train/Val/Test 분할을 `src/data_engineer`의 스크립트로 수행합니다.
- 파일 형식: 이미지(JPG/PNG), 어노테이션(JSON, YOLO txt), 설정(YAML)

---

# 최종 선택 모델: YOLO

본 프로젝트의 최종 제출 모델은 **YOLO**입니다.

- 학습/추론 스크립트: `src/model/yolo/yolo_train.py`, `src/model/yolo/yolo_predict.py`
- 평가/분석: `src/model/yolo/` 하의 utility 스크립트들

(참고) Faster R-CNN은 비교 모델로 개발되었습니다.

---

# YOLO 실험 결과 요약

본 프로젝트에서 50 에포크(epoch)로 학습한 YOLO 모델들의 성능 비교 결과입니다.

**학습 설정**: imgsz=640, batch=16, epochs=50

| 모델    | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 전체 학습시간 | 추론시간 | fps |
| ------- | ------- | ------------ | --------- | ------ | ------------- | -------- | --- |
| YOLOv8n | 0.9089  | 0.8748       | 0.8612    | 0.8546 | 186초         | 0.0414s  | 24  |
| YOLO11n | 0.8677  | 0.8499       | 0.8592    | 0.7333 | 196초         | 0.0330s  | 30  |
| YOLO11s | 0.9760  | 0.9565       | 0.9305    | 0.9627 | 267초         | 0.0277s  | 36  |
| YOLO11m | 0.9786  | 0.9645       | 0.9290    | 0.9445 | 789초         | 0.0307s  | 33  |
| YOLO11x | 0.9900  | 0.9716       | 0.9409    | 0.9440 | 7305초        | 0.0380s  | 26  |

**주요 발견:**

- **최고 정확도 (mAP@0.5)**: YOLO11x (0.9900)
- **최고 정확도 (mAP@0.5:0.95)**: YOLO11x (0.9716)
- **빠른 추론**: YOLO11s (fps = 36, 추론시간 = 0.0277초)

**평가 방법 및 수치 출처:**

1. **mAP@0.5, mAP@0.5:0.95, Precision, Recall** (로컬 검증 지표)
   - Ultralytics API의 `model.val(data='data.yaml', imgsz=640)` 실행 후 출력
   - 각 모델 폴더의 `results.csv` (50번째 epoch 행)에서 추출
   - 예시 경로: `results/yolo_final_model11/results.csv`
   - **주의**: mAP@0.5:0.95는 로컬 검증 지표입니다.

2. **추론 시간**
   - `src/model/yolo/yolo_predict.py` → `create_submission_csv()` 실행 후 로그 파일 기록
   - 로그 파일: `results/submission/submission_{timestamp}_data.log`

3. **FPS (참고용)**
   - 계산: `fps ≈ 1 / 평균 추론 시간`
   - ⚠️ **주의**: 이 값은 참고용입니다. 실제 배포 환경에서는 배치 처리, GPU 워밍업 등을 고려하여 재측정 필요합니다.

---

# 재현(간단) — 환경 및 빠른 실행 예시

## 요구사항

- Conda (Anaconda 또는 Miniconda)

## 사용법 및 환경설정

본 프로젝트는 Conda 가상환경을 기반으로 실행됩니다.

### 환경 설정 및 종속성 설치

상세한 환경 구축(GPU 설정, DLL 에러 해결)은 [환경 설정 가이드(env/README.md)](./env/README.md)를 반드시 참고하세요.

## 1. 데이터 준비

https://drive.google.com/drive/folders/16wokuICTlGxwMbB-XxW5qGtbvnROHfs3?usp=drive_link 에 있는 `data.zip` 파일을 다운로드하여 `data/original/`위치에 압축 해제합니다.

## 2. Conda 가상환경 생성 및 활성화

```bash
conda env create -f env/environment.yml
```

```bash
conda activate healthit_5team
```

```bash
pip install torch torchvision
```

## 3. 학습 실행 및 학습 중 시각화

아래 명령어로 실행합니다.

```bash
python src/model/yolo/yolo_train.py

```

- 학습이 되는 동안 학습 시간을 측정하여, 학습이 완료된 후 터미널에서 학습 소요 시간을 확인하실 수 있습니다.

- 학습이 완료되면 `yolo_train_{학습시작 시각}`이라는 폴더명으로 `results/` 하위에 생성되며,
  모델 가중치는 `src/model/yolo/`에 저장됩니다.

- 학습 중에 가장 성능이 좋았던 모델은 weight/best.pt에 저장됩니다.

  ex) results/yolo_train_20260213_012013/weight/best.pt

## 4. 추론 실행 및 추론 결과 파일 생성

학습된 모델을 사용하여 이미지 추론을 수행합니다.

```bash
python src/model/yolo/yolo_predict.py --mode submission
```

- 예측 결과 이미지는 `results/submission/`에 저장됩니다.

  ex) results/submission/submission_20260213_0120_csv

---

# 결과 아티팩트 위치

- 모델 가중치: `results/.../weights/best.pt`
- 제출용 CSV: `results/submission/`
- 실험별 설정/로그: 각 `results/yolo_final_model*` 폴더의 `args.yaml`, `results.csv`

---

# 협업 일지

- 노가빈 : https://sordid-daffodil-392.notion.site/2f7989ffa6ed802a80e0d45f31beea96?source=copy_link
- 정주희 : https://www.notion.so/2f7f3d707ce880fb91a2c7f4ae455074?source=copy_link
- 박수성 : https://docs.google.com/document/d/1Hf6h7Vu_1eF07uzN-lPkGGjWrmwsw9-ftvhKOtmLd8o/edit?usp=sharing
- 강하은 : https://www.notion.so/2f7ba7f4895a80e589ebcf4898290296?source=copy_link
