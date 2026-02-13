# Faster R-CNN 알약 객체 탐지 — 재현 가이드

## 최종 성과
- **Kaggle mAP: 0.931** (v9, 5-Fold CV + WBF 앙상블)
- 모델: Faster R-CNN (ResNet50v2 + FPN, torchvision 사전학습)

---

## 1. 환경 설정

### 필수 라이브러리
```
torch >= 2.0
torchvision >= 0.15
albumentations >= 1.3
opencv-python
numpy
scikit-learn
ensemble-boxes >= 1.0   # WBF 앙상블용
```

### 설치
```bash
pip install torch torchvision albumentations opencv-python numpy scikit-learn ensemble-boxes
```

---

## 2. 데이터 구조

```
data/
├── original/
│   ├── train_images/       # 학습 이미지 232장
│   └── test_images/        # 테스트 이미지 58장
└── processed/
    └── train_annotations/  # JSON 어노테이션 파일
```

- 이미지 크기: 976×1280 (원본 유지, 리사이즈 없음)
- 클래스 수: 57개 (background 포함)
- 어노테이션 형식: COCO JSON (`[x, y, w, h]` → 코드 내부에서 `[x1, y1, x2, y2]` 변환)

---

## 3. 파일 구조

```
src/model/faster_rcnn/
├── config.py                    # 하이퍼파라미터 설정
├── model.py                     # Faster R-CNN 모델 로드 + head 교체
├── Faster_RCNN_dataset.py       # Dataset 클래스 + Augmentation
├── Faster_RCNN_dataloader.py    # DataLoader 구성
├── make_classID_txt.py          # 클래스 ID 매핑 생성
├── train.py                     # 학습 루프 (optimizer, scheduler)
├── evaluate.py                  # mAP 평가
├── predict.py                   # 추론 + Soft-NMS + CSV 생성
├── main.py                      # 전체 파이프라인 (로컬 실행용)
├── train_v9_5fold.py            # 5-Fold CV 학습 (v9 최종)
├── predict_v9_5fold_wbf.py      # 5-Fold WBF 앙상블 추론 (v9 최종)
├── train_colab.ipynb            # Colab 학습 노트북 (단일 모델)
└── train_5fold_colab.ipynb      # Colab 5-Fold 학습 노트북 (v9 최종)
```

---

## 4. 실행 방법

### 방법 A: 로컬 실행 (단일 모델)

```bash
cd src/model/faster_rcnn
python main.py
```

`config.py`에서 하이퍼파라미터를 수정할 수 있습니다.

### 방법 B: Google Colab 실행 (권장)

1. `train_5fold_colab.ipynb`를 Colab에 업로드
2. 데이터를 Google Drive에 업로드 (노트북 내 경로를 본인 Drive 경로에 맞게 수정)
3. GPU 런타임 선택 (T4 또는 A100)
4. 노트북 셀 순서대로 실행

---

## 5. 최종 모델 (v9) 재현

### 5.1 학습 (5-Fold CV)
```bash
cd src/model/faster_rcnn
python train_v9_5fold.py
```

- 232장을 5개 Fold로 분할하여 각각 30 epoch 학습
- 체크포인트 5개 생성: `checkpoints/fold_1_best.pth` ~ `checkpoints/fold_5_best.pth`

### 5.2 추론 (WBF 앙상블)
```bash
python predict_v9_5fold_wbf.py
```

- `checkpoints/` 폴더에서 5개 체크포인트를 불러와 각각 추론
- 5개 모델의 예측을 WBF (iou_threshold=0.55)로 가중 평균
- `submission.csv` 생성 (Kaggle 제출 형식)

```
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,test_001,12,150.3,200.1,80.5,60.2,0.9521
```

---

## 6. 주요 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| Backbone | ResNet50v2 + FPN (COCO 사전학습) |
| Optimizer | AdamW (lr=0.0001, weight_decay=0.0005) |
| Scheduler | Warmup(1ep) + CosineAnnealing(29ep) |
| Batch Size | 4 |
| Epochs | 30 |
| Gradient Clipping | max_norm=10.0 |
| Score Threshold | 0.05 |
| Soft-NMS | sigma=0.5 |
| 5-Fold CV | K=5, seed=42 |
| WBF iou_threshold | 0.55 |

---

## 7. 주의사항

- **A.Normalize 사용 금지**: torchvision Faster R-CNN 내부에서 ImageNet 정규화를 자동 수행합니다. 외부에서 `A.Normalize`를 추가하면 이중 정규화로 성능이 폭락합니다. `A.ToFloat(max_value=255.0)` + `ToTensorV2()`만 사용하세요.
- **Mosaic Augmentation 사용 금지**: 232장/57클래스 소규모 데이터에서 class collapse가 발생합니다.
- **차등 학습률 사용 금지**: backbone_lr_ratio는 1.0 (동일)으로 설정하세요. 소규모 데이터에서 차등 LR은 역효과입니다.

---

## 8. 실험 히스토리

| 버전 | Kaggle mAP | 주요 변경 |
|------|-----------|----------|
| v1 | 0.360 | 초기 모델, score threshold 0.5 |
| v2 | 0.691 | score threshold 0.5 → 0.05 |
| v3 | 0.580 | 학습률 변경 실험 (하락) |
| v4 | 0.692 | ResNet50v2 + Albumentations |
| v5 | 0.620 | 하이퍼파라미터 탐색 (하락) |
| v6 | 0.690 | CosineAnnealing 스케줄러 |
| v7 | 0.360 | Custom Anchor 시도 (폭락) |
| v8 | 0.090 | Mosaic + 차등 LR (최악) |
| v8_opt | 0.915 | 잘못된 것 제거 + AdamW + Soft-NMS |
| **v9** | **0.931** | **5-Fold CV + WBF 앙상블 (최종)** |
| v10 | 0.900 | TTA 시도 (하락) |
