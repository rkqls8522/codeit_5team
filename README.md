# 프로젝트 제목

5팀의 헬스-잇

<br>

# 프로젝트 개요

본 프로젝트는 사용자가 촬영한 경구약제 이미지에서

최대 4개의 알약을 탐지하고 각 알약의 클래스와 위치를 예측하는

객체 탐지(Object Detection) 문제를 다룹니다.

<br>

# 기술 스택

- Python
- PyTorch
- OpenCV, NumPy, Matplotlib
- Jupyter Notebook
- Conda

<br>

# 역할 담당자

- 프로젝트 매니저 : 노가빈
- 데이터 엔지니어 : 노가빈, 박수성
- 모델 설계, 실험, 평가
  - YOLO 모델 : 정주희
  - Faster R-CNN 모델 : 강하은

<br>

# 개발 일정

| 기간        | 작업 내용                                                   |
| ----------- | ----------------------------------------------------------- |
| 1/29 ~ 1/30 | 요구사항 분석, 프로젝트 계획 수립, 환경 셋팅, 팀 규칙 논의  |
| 2/2 ~ 2/4   | 데이터 이해 및 전처리, 모델 초기 실험                       |
| 2/5 ~ 2/9   | (추가 데이터 수집 및 전처리, ) 모델 튜닝 및 개선            |
| 2/10 ~ 2/12 | (추가 데이터로 비즈니스 로직 통합, ) 최종 튜닝 및 결과 정리 |
| 2/13        | 제출물 정리 및 발표 자료 제작                               |

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
 ┣ 📂notebooks                  # 탐색적 분석, 테스트 등 수행
 ┃ ┣ 📜eda.ipynb
 ┃ ┗ 📜model_experiment.ipynb
 ┣ 📂results                    # 결과 모음
 ┃ ┣ 📂figures                  # 그래프, 차트 등 시각화 이미지
 ┃ ┃ ┗ 📜.gitkeep
 ┃ ┗ 📜experiment_log.txt
 ┣ 📂src                        # 주요 코드 모음
 ┃ ┣ 📂data_engineer            # 데이터 관련 코드
 ┃ ┃ ┣ 📜augmentation.py
 ┃ ┃ ┣ 📜pipeline.py
 ┃ ┃ ┗ 📜preprocessing.py
 ┃ ┣ 📂experiments              # 모델 학습, 평가, 튜닝
 ┃ ┃ ┣ 📜evaluate.py
 ┃ ┃ ┣ 📜hyperparameter_tuning.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┃ ┗ 📜result.md
 ┃ ┣ 📂model                    # 모델 개발
 ┃ ┃ ┣ 📂resnet
 ┃ ┃ ┃ ┗ 📜base_model.py
 ┃ ┃ ┗ 📂yolo
 ┃ ┃ ┃ ┗ 📜base_model.py
 ┃ ┗ 📂utils                    # 공통, 보조 함수 모음
 ┃ ┃ ┣ 📜utils.py
 ┃ ┃ ┗ 📜visualization.py
 ┣ 📜.gitignore
 ┗ 📜README.md
```

<br>

# 디렉터리 구조 요약

- `data/` : 원본 및 전처리된 데이터 파일을 저장합니다.
- `docs/` : 프로젝트 기획서, 회의록, 보고서 등 문서 자료가 있습니다.
- `env/` : 가상환경 설정 파일(`requirements.txt` 또는 `environment.yml`)을 포함합니다.
- `notebooks/` : 데이터 분석, 실험 내용 기록을 위한 Jupyter Notebook을 모아둡니다.
- `results/` : 실험 결과, 로그, 제출용 파일 등을 저장하는 폴더입니다.
- `src/` : 프로젝트의 코드가 위치하는 주요 폴더입니다.
  - `data_engineer/` : 데이터 전처리 및 파이프라인 관련 코드
  - `experiments/` : 학습, 평가, 하이퍼파라미터 튜닝 등의 실험 코드
  - `model/` : 다양한 모델 설계 코드
  - `utils/` : 공통으로 사용하는 유틸리티 함수 및 스크립트
- `.gitignore` : Git 추적 제외 파일 목록
- `README.md` : 프로젝트 개요 및 사용법 안내 문서

<br>

# 시작 가이드

## 요구사항

- Python 3.9 이상
- Conda (Anaconda 또는 Miniconda)

## 사용법 및 환경설정

본 프로젝트는 Conda 가상환경을 기반으로 실행됩니다.

### 1. 데이터 준비

프로젝트 루트의 `data/data.zip` 파일을 압축 해제합니다.

```bash
python scripts/prepare_data.py
```

### 2. Conda 가상환경 생성 및 활성화

```bash
conda env create -f env/environment.yml
conda activate healthit_project
```

### 3. 학습 실행

아래 명령어로 실행합니다.

```bash
python src/train.py

```

학습이 완료되면 모델 가중치는 `results/final_model/`에 저장됩니다.

### 4. 추론 실행 및 결과 시각화

학습된 모델을 사용하여 이미지 추론을 수행합니다.

```bash
python src/inference.py \
  --weights results/final_model/best.pt \
  --image data/sample.jpg
```

예측 결과 이미지는 `results/figures/`에 저장됩니다.
