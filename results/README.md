# 실험 결과 및 아티팩트

## 개요

`results/` 폴더는 YOLO 모델의 모든 학습 실험, 평가 결과, 제출 파일을 저장하는 중앙 저장소입니다.

---

## 폴더 구조

### `submission/` 폴더

**최종 제출 파일들이 저장되는 위치**

**CSV 포맷:**

- 각 행은 한 이미지에 대한 예측 결과
- 형식: `annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score`

---

### `yolo_train_*` 폴더 (학습 실험)

각 폴더는 특정 모델의 한 번의 학습 실험 결과를 저장합니다.

**각 폴더 내용 예시 :**

```
yolo_train_20260209_12_24/
├── weights/
│   ├── best.pt              # 최고 성능 모델 가중치
│   └── last.pt              # 마지막 에폭 모델 가중치
├── args.yaml                # 학습 설정 (imgsz, batch, epochs, optimizer 등)
├── results.csv              # 50행 × 14열: 각 에폭별 메트릭 기록
│                            # 열: epoch, mAP@0.5, mAP@0.5:0.95, precision, recall, loss 등
├── confusion_matrix.png     # 혼동 행렬 시각화
├── F1_curve.png             # F1-score 곡선
├── PR_curve.png             # Precision-Recall 곡선
├── P_curve.png              # Precision 곡선
├── R_curve.png              # Recall 곡선
├── events.out.tfevents.*    # TensorBoard 로그 (선택사항)
└── ... (기타 시각화 파일들)
```

---

## 주요 파일 설명

### `results.csv` (가장 중요한 메트릭)

**구조:**

```
epoch,
time,
train/box_loss,
train/cls_loss,
train/dfl_loss,
metrics/precision(B),
metrics/recall(B),
metrics/mAP50(B),
metrics/mAP50-95(B),
val/box_loss,
val/cls_loss,
val/dfl_loss,
lr/pg0,
lr/pg1,
lr/pg2
```

**50행 데이터 중 마지막 행(epoch 50)은 최종 성능 지표:**

- `metrics/precision(B)`: 정밀도 (Precision)
- `metrics/recall(B)`: 재현율 (Recall)
- `metrics/mAP50(B)`: mAP@0.5 (IoU 임계값 0.5)
- `metrics/mAP50-95(B)`: mAP@0.5:0.95 (IoU 임계값 0.5~0.95 평균)

**사용 예:**

```python
import pandas as pd
df = pd.read_csv('results/yolo_train_20260213_031449/results.csv')
final_row = df.iloc[-1]  # 50번째 에폭
print(f"mAP@0.5: {final_row['metrics/mAP50(B)']}")
print(f"mAP@0.5:0.95: {final_row['metrics/mAP50-95(B)']}")
```

### `args.yaml`

학습 시 사용된 모든 하이퍼파라미터:

```yaml
model: yolov8m.pt
data: data.yaml
epochs: 100
imgsz: 1024
batch: 10
patience=20
device: config.device
exist_ok=False,
resume=resume
...
```

### `best.pt` & `last.pt`

- **best.pt**: 검증 성능(mAP)이 최고였던 에폭의 가중치 (추론에 주로 사용)
- **last.pt**: 마지막 에폭(50번째)의 가중치 (필요시 재개 학습용)

---

## 최종 모델 및 제출 파일

### 최종 선택: YOLOv8m

- **학습 폴더**: `yolo_train_20260209_12_24/`
- **모델 가중치**: `yolo_train_20260209_12_24/weights/best.pt`
- **제출 파일**: `submission/submission_yolov8m_epoch100.csv`
- **Kaggle 점수**: mAP@[0.75:0.95] = **0.96975**

---
