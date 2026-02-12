# 프로젝트 제목

헬스잇

<br>

# 프로젝트 개요

본 프로젝트는 사용자가 촬영한 경구약제 이미지에서

최대 4개의 알약을 탐지하고 각 알약의 클래스와 위치를 예측하는

객체 탐지 모델을 개발합니다

<br>

# 문제 정의

본 프로젝트의 목적은 스마트폰으로 촬영한 복용 약 사진에서 알약을 정확히 탐지하고(최대 4개),
각 알약의 클래스(종류)와 위치(바운딩 박스)를 예측하는 것입니다. 주요 요구사항은 다음과 같습니다:

- 실세계(다양한 조명, 배경 등)에 강건한 탐지
- 각 이미지에서 최대 4개 객체 탐지 및 클래스 예측
- 경량/실무 적용을 고려한 추론 속도와 정확도의 균형

<br>

# 기술 스택

### Language

- Python 3.11

### Deep Learning Framework

- PyTorch
- Ultralytics YOLOv8
- torchvision

### Hyperparameter Optimization

- Optuna

### Computer Vision

- OpenCV
- Pillow

### Data Handling & Visualization

- NumPy
- Pandas

### Development & Environment

- Jupyter Notebook
- CUDA (11.8 / 12.1)
- Conda
- Git

<br>

# 역할 담당자

- 프로젝트 매니저 : 노가빈
- 데이터 엔지니어 : 노가빈, 박수성
- 모델 설계, 실험, 평가
  - YOLO 모델 : 정주희
  - Faster R-CNN 모델 : 강하은

<br>

# 개발 일정

| 기간        | 작업 내용                                                  |
| ----------- | ---------------------------------------------------------- |
| 1/29 ~ 1/30 | 요구사항 분석, 프로젝트 계획 수립, 환경 셋팅, 팀 규칙 논의 |
| 2/2 ~ 2/4   | 데이터 이해 및 전처리, 모델 초기 실험                      |
| 2/5 ~ 2/10  | 모델 튜닝 및 개선                                          |
| 2/11        | 최종 실험 및 결과 정리                                     |
| 2/12 ~ 2/13 | 제출물 정리 및 발표 자료 제작                              |

<br>

# 프로젝트 구조

```
📦코드잇-5팀-초급프로젝트-헬스잇
 ┣ 📂data
 ┃ ┣ 📂original                 # 원본 데이터
 ┃ ┣ 📂processed                # 전처리 완료된 데이터
 ┃ ┗ 📜README.md                # 데이터 설명과 출처
 ┣ 📂docs                       # 보고서 및 발표 자료
 ┃ ┗ 📜report.md
 ┣ 📂env                        # 설정 파일 모음
 ┣ ┗ 📜README.md
 ┣ 📂notebooks                  # 탐색적 분석, 테스트 등 수행
 ┃ ┣ 📜eda.ipynb
 ┣ 📂results                    # 결과 모음
 ┃ ┣ 📂submission               # 추론결과 파일 모음
 ┃ ┗ 📂yolo_final_model         # 그래프, 차트 등 시각화 이미지
 ┣ 📂src                        # 주요 코드 모음
 ┃ ┣ 📂data_engineer            # 데이터 관련 코드
 ┃ ┣ 📂model                    # 모델 개발
 ┃ ┃ ┣ 📂faster_rcnn
 ┃ ┃ ┗ 📂yolo
 ┣ 📜.gitignore
 ┗ 📜README.md
```

<br>

# 디렉터리 구조 요약

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

<br>

# 접근 방식 요약

모델 학습과 평가 파이프라인은 다음 흐름으로 구성됩니다:

- 데이터 전처리: 원본 COCO-style JSON 어노테이션을 읽어 YOLO 형식 텍스트 라벨로 변환하고, 학습/검증/테스트로 분할합니다.
- 모델 실험: Ultralytics YOLOv8 계열 모델과 torchvision 제공 Faster R-CNN 계열 모델을 사용해 여러 실험을 수행했습니다.
- 하이퍼파라미터 튜닝: 학습률, 배치 크기, 에폭 수 등 주요 하이퍼파라미터를 실험적으로 조정했습니다 (Optuna 검색 가능).
- 평가: 검증셋에서의 mAP(특히 mAP@0.5), precision/recall, confusion matrix 등을 중심으로 성능을 비교했습니다.

<br>

# 사용한 모델 및 실험 요약

- YOLO 실험
  - 프레임워크: `ultralytics` (YOLOv8)
  - 사용 모델 파일: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11x.pt`, `yolo26n.pt`, `yolo26x.pt` (사전학습 가중치에서 파인튜닝)
  - 학습 스크립트: [src/model/yolo/yolo_train.py](src/model/yolo/yolo_train.py#L1)
  - 특징: 빠른 학습/추론, 데이터 전처리 모듈(`src/data_engineer`)과 연동

- Faster R-CNN 실험
  - 프레임워크: `torch` + `torchvision` (fasterrcnn_resnet50_fpn 등)
  - 학습 스크립트: [src/model/faster_rcnn/train.py](src/model/faster_rcnn/train.py#L1)
  - 특징: 높은 정확도(특히 작은 객체에서 유리), 학습 안정성 확보를 위해 scheduler/SGD 사용

<br>

# 데이터 및 전처리

- 원본: `data/original` 폴더에 이미지와 COCO-style JSON 어노테이션이 위치합니다.
- 전처리: 어노테이션 정제, YOLO 포맷 변환, Train/Val/Test 분할을 `src/data_engineer`의 스크립트로 수행합니다.
- 파일 형식: 이미지(JPG/PNG), 어노테이션(JSON, YOLO txt), 설정(YAML)

<br>

# 학습 설정(대표값)

- Python 3.11 기반 Conda 환경
- 주요 라이브러리: `torch`, `torchvision`, `ultralytics`, `opencv`, `pillow`, `numpy`, `pandas`, `scikit-learn`, `tqdm`
- YOLO 대표 하이퍼파라미터: `imgsz=640`, `epochs=50`, `batch=16`
- Faster R-CNN 대표 하이퍼파라미터: SGD, 학습률 및 scheduler는 `src/model/faster_rcnn/config.py` 또는 `CONFIG`에서 설정

<br>

# 평가 및 최종 결과

- 실험 아티팩트는 `results/` 폴더에 저장됩니다: 모델 가중치(`.pt`, `.pth`), `submission.csv`, 제출 파일 등.
- 현재 저장된 주요 아티팩트 예시:
  - 학습된 YOLO 가중치: `results/.../weights/best.pt` (학습 스크립트가 자동 복사하도록 구성)
  - 제출용 CSV: `results/submission/submission.csv`

※ README에 수치(예: mAP, Precision, Recall, inference FPS)를 추가하려면, 각 실험의 `results.csv` 또는 로그에서 정확한 수치를 가져와 아래 섹션에 기록해 주세요. 현재 저장된 결과 파일을 기반으로 요약을 추가해 드릴 수 있습니다.

<br>

# 시작 가이드

## 요구사항

- Conda (Anaconda 또는 Miniconda)

## 사용법 및 환경설정

본 프로젝트는 Conda 가상환경을 기반으로 실행됩니다.

### 환경 설정 및 종속성 설치

상세한 환경 구축(GPU 설정, DLL 에러 해결)은 [환경 설정 가이드(env/README.md)](./env/README.md)를 반드시 참고하세요.

## 1. 데이터 준비

프로젝트 루트의 `data/data.zip` 파일을 `data/`위치에 압축 해제합니다.

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

<br>

# 향후 개선/권장 작업

- 결과 표준화: 각 실험의 mAP, precision, inference latency를 한 표로 정리
- 앙상블 실험: YOLO + Faster R-CNN 앙상블로 정확도 향상 검토
- 경량화: ONNX 변환 및 TensorRT/ONNX Runtime을 통한 배포 최적화
- 데이터 확장: 추가 증강(조명/블러/랜덤 배치)으로 일반화 성능 향상
