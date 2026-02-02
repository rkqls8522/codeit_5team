# Faster R-CNN 모델 개발 계획서

> 담당자: 강하은 (5팀)
> 역할: **Faster R-CNN 모델 설계 / 학습 / 검증** + **Experimentation Lead** (다양한 실험 주도, 하이퍼파라미터 튜닝 및 모델 성능 평가)
> 기간: 2026.02.02 (월) ~ 2026.02.13 (금), 평일 10일 (주말 제외)
> 중간발표: 02.09 (월) / 최종발표: 02.23 (월)

---

## 전체 일정 요약

| 일차 | 날짜 | 단계 | 핵심 목표 |
|------|------|------|-----------|
| Day 1 | 02.02 (월) | 기초 학습 | Faster R-CNN 구조 이해 + 개발환경 세팅 |
| Day 2 | 02.03 (화) | 데이터 파악 | 데이터 EDA + Object Detection용 데이터 포맷 이해 |
| Day 3 | 02.04 (수) | 파이프라인 구축 | Dataset/DataLoader 구현 + 전처리 파이프라인 |
| Day 4 | 02.05 (목) | 모델 구현 | Pretrained Faster R-CNN 로딩 + 학습 루프 작성 |
| Day 5 | 02.06 (금) | 첫 학습 + 평가 | 첫 학습 실행 + 평가 코드(mAP) + 중간발표 준비 |
| | 02.07~08 | **주말 OFF** | |
| Day 6 | 02.09 (월) | **중간발표** | 중간발표 + 피드백 정리 + 실험 계획 수립 |
| Day 7 | 02.10 (화) | 실험 1 | 하이퍼파라미터 튜닝 (LR, optimizer, scheduler) |
| Day 8 | 02.11 (수) | 실험 2 | 데이터 증강 실험 + Kaggle 제출 |
| Day 9 | 02.12 (목) | 실험 3 | Backbone 교체 + Anchor/NMS 튜닝 |
| Day 10 | 02.13 (금) | 마무리 | 최종 모델 확정 + Kaggle 최종 제출 + 코드 정리 |
| | 02.14~15 | **주말 OFF** | |
| | 02.16~20 | 발표 준비 | 발표자료 작성 + 보고서 작성 + 리허설 |
| | **02.23 (월)** | **최종발표** | 팀 발표 (20분) + 질의응답 (5분) |

---

## Phase 1: 기반 구축 (Day 1~5)

### Day 1 (02.02 월) - 기초 학습 + 환경 세팅

**목표**: Faster R-CNN이 무엇인지 이해하고, 개발 환경을 준비한다.

**할 일**
- [ ] Faster R-CNN 핵심 개념 학습
  - **이해해야 할 키워드**: Region Proposal Network (RPN), Anchor Box, RoI Pooling, Non-Maximum Suppression (NMS)
  - 추천 자료:
    - [Faster R-CNN 논문 리뷰 (한글)](https://herbwood.tistory.com/10)
    - PyTorch 공식 문서: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- [ ] 개발 환경 확인 및 세팅
  ```bash
  pip install torch torchvision
  pip install pycocotools  # mAP 평가용
  pip install albumentations  # 데이터 증강용 (나중에 사용)
  pip install matplotlib pandas numpy opencv-python
  ```
- [ ] GPU 환경 확인 (Colab / 로컬 / 서버)
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
  ```
- [ ] 프로젝트 폴더 구조 정리
  ```
  src/model/faster_rcnn/
  ├── dataset.py          # Dataset 클래스
  ├── model.py            # 모델 정의
  ├── train.py            # 학습 루프
  ├── evaluate.py         # 평가 (mAP 계산)
  ├── predict.py          # 추론 + Kaggle 제출 파일 생성
  ├── config.py           # 하이퍼파라미터 설정
  └── utils.py            # 시각화 등 유틸리티
  ```

**협업일지 포인트**: 환경 세팅 과정, Faster R-CNN 구조 학습 내용

---

### Day 2 (02.03 화) - 데이터 EDA

**목표**: 경구약제 이미지 데이터의 구조, 라벨 형식, 클래스 분포를 파악한다.

**할 일**
- [ ] 데이터 파일 구조 확인 (이미지 경로, 라벨 파일 형식)
- [ ] 라벨 데이터 분석
  - 총 이미지 수, 클래스 수, 클래스별 분포
  - 바운딩 박스 크기 분포 (작은 약? 큰 약?)
  - 이미지당 약 개수 분포 (1~4개)
- [ ] 샘플 이미지 시각화 (바운딩 박스 오버레이)
  ```python
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches

  fig, ax = plt.subplots(1)
  ax.imshow(image)
  for box, label in zip(boxes, labels):
      rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
      ax.text(box[0], box[1]-5, label, color='red', fontsize=10)
  plt.show()
  ```
- [ ] EDA 결과를 팀에 공유 + 전처리 방향 논의
- [ ] **(Experimentation Lead)** EDA 결과 기반으로 실험 방향 초안 메모
  - 클래스 불균형이 심하면 → 오버샘플링/class weight 고려
  - 바운딩 박스가 작으면 → anchor size 조정 필요
  - 이미지 해상도가 다양하면 → 리사이즈 전략 필요

**협업일지 포인트**: 클래스 불균형 여부, 이미지 해상도 패턴, 특이 사항

---

### Day 3 (02.04 수) - Dataset + DataLoader 구현

**목표**: PyTorch Faster R-CNN에 맞는 데이터 파이프라인을 구축한다.

**할 일**
- [ ] `dataset.py` 작성 - Custom Dataset 클래스
  ```python
  class PillDataset(torch.utils.data.Dataset):
      """
      Faster R-CNN이 요구하는 포맷:
      - image: [C, H, W] 텐서
      - target: dict {
          'boxes': [N, 4] FloatTensor (x1, y1, x2, y2),
          'labels': [N] Int64Tensor (클래스 인덱스),
          'image_id': Int64Tensor,
          'area': [N] FloatTensor,
          'iscrowd': [N] Int64Tensor
        }
      """
      def __getitem__(self, idx):
          # 이미지 로드 + 라벨 파싱 + 텐서 변환
          ...
          return image, target
  ```
- [ ] DataLoader 구성 (collate_fn 커스텀 필요)
  ```python
  def collate_fn(batch):
      return tuple(zip(*batch))

  train_loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)
  ```
- [ ] Train/Validation 분할 (8:2 또는 9:1)
- [ ] 데이터 로딩 테스트 (1 batch 확인)

**핵심 주의사항**
- boxes는 반드시 `(x1, y1, x2, y2)` 형식, `x2 > x1`, `y2 > y1`
- labels는 0이 아닌 **1부터** 시작 (0은 background)
- collate_fn을 반드시 커스텀해야 함 (기본 collate는 동작 안 함)

---

### Day 4 (02.05 목) - 모델 구현 + 학습 코드

**목표**: Pretrained Faster R-CNN을 불러와서 우리 데이터에 맞게 수정하고, 학습 루프를 작성한다.

**할 일**
- [ ] `model.py` 작성
  ```python
  import torchvision
  from torchvision.models.detection import fasterrcnn_resnet50_fpn
  from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

  def get_model(num_classes):
      model = fasterrcnn_resnet50_fpn(pretrained=True)
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      return model
  ```
- [ ] `config.py` 작성 - 초기 하이퍼파라미터 설정
  ```python
  CONFIG = {
      'num_classes': ?,        # background + 약 클래스 수 (EDA에서 확인)
      'batch_size': 4,
      'learning_rate': 0.005,
      'momentum': 0.9,
      'weight_decay': 0.0005,
      'num_epochs': 10,
      'lr_scheduler_step': 3,
      'lr_scheduler_gamma': 0.1,
  }
  ```
- [ ] `train.py` 작성 - 학습 루프
  ```python
  optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

  for epoch in range(num_epochs):
      model.train()
      for images, targets in train_loader:
          loss_dict = model(images, targets)
          losses = sum(loss for loss in loss_dict.values())
          optimizer.zero_grad()
          losses.backward()
          optimizer.step()
      lr_scheduler.step()
  ```
- [ ] 학습 중 loss 로깅 + 체크포인트 저장 기능 추가
  ```python
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
  }, f'checkpoint_epoch_{epoch}.pth')
  ```

---

### Day 5 (02.06 금) - 첫 학습 + 평가 체계 + 중간발표 준비

**목표**: 첫 학습을 돌리고, 평가 코드를 작성하고, 중간발표 자료를 준비한다.

**할 일**
- [ ] 소규모 데이터(50~100장)로 **오버피팅 테스트**
  - Loss가 꾸준히 감소하는지 확인
  - 학습 이미지에 대해 예측 결과 시각화
- [ ] 버그 수정 및 디버깅
- [ ] 전체 데이터로 첫 학습 시작 (10 epoch)
- [ ] `evaluate.py` 작성 - mAP 계산
  ```python
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  ```
- [ ] `predict.py` 작성 - Kaggle 제출 파일 생성
- [ ] 첫 Kaggle 제출 시도 (팀원과 협의 후)
- [ ] **중간발표 자료 준비**
  - 모델 구조 설명 다이어그램
  - 학습 Loss 그래프
  - 예측 결과 시각화 이미지 (잘 된 것 + 못 된 것)
  - baseline mAP 수치
  - 왜 Faster R-CNN을 선택했는지
  - 다음 주 실험 계획

**트러블슈팅 가이드**
| 증상 | 원인 | 해결 |
|------|------|------|
| CUDA out of memory | batch_size 너무 큼 | batch_size를 2 또는 1로 줄이기 |
| Loss가 안 줄어듦 | learning rate 문제 | lr을 0.001로 낮춰보기 |
| NaN loss | 잘못된 박스 좌표 | x2>x1, y2>y1 확인 |
| 예측 결과 없음 | confidence threshold | score_threshold 낮춰보기 |

---

## Phase 2: 중간발표 + 실험 (Day 6~10)

### Day 6 (02.09 월) - 중간발표 + 실험 계획 수립

**목표**: 중간발표를 진행하고, Experimentation Lead로서 후반부 실험 전략을 체계적으로 설계한다.

**할 일**
- [ ] **중간발표 진행**
- [ ] 강사/멘토 피드백 기록
- [ ] 다른 팀의 접근법에서 배울 점 기록
- [ ] **(Experimentation Lead) 실험 계획표 작성**
  - baseline 성능을 기준으로 어떤 변수를 바꿀지 정리
  - 한 번에 하나의 변수만 바꾸는 것이 원칙 (controlled experiment)
  ```
  [실험 계획]
  실험 A: Learning Rate 비교      → 0.001 vs 0.005 vs 0.01
  실험 B: Optimizer 비교           → SGD vs Adam vs AdamW
  실험 C: LR Scheduler 비교       → StepLR vs CosineAnnealing vs Warm-up
  실험 D: 데이터 증강 ON/OFF       → 증강 없음 vs Flip+Color vs 강한 증강
  실험 E: Backbone 비교            → ResNet50 vs ResNet50-v2 vs MobileNet-v3
  실험 F: Anchor/NMS 조정          → anchor size, NMS threshold
  ```
- [ ] 팀원별 모델 성능 비교 기준 통일 (mAP@0.5 기준 등)

---

### Day 7 (02.10 화) - 실험 라운드 1: 하이퍼파라미터 튜닝

**목표**: 가장 영향이 큰 하이퍼파라미터부터 체계적으로 실험한다.

**할 일**
- [ ] **실험 A: Learning Rate 탐색**
  - 0.001, 0.005, 0.01 각각 학습 후 mAP 비교
  - 최적 LR 확정
- [ ] **실험 B: Optimizer 변경**
  - SGD (momentum=0.9) vs Adam (lr=0.0001) vs AdamW (lr=0.0001)
  ```python
  # Adam은 보통 SGD보다 낮은 lr 사용
  optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
  ```
- [ ] **실험 C: LR Scheduler 비교**
  ```python
  # CosineAnnealing
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

  # Warm-up + Cosine
  from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
  warmup = LinearLR(optimizer, start_factor=0.01, total_iters=3)
  cosine = CosineAnnealingLR(optimizer, T_max=num_epochs-3)
  scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[3])
  ```
- [ ] 모든 실험 결과를 실험 기록 표에 기록
- [ ] Kaggle 제출 (현시점 최고 성능 모델)

**하이퍼파라미터 튜닝 순서 (영향력 큰 순)**
1. Learning Rate (가장 중요)
2. Optimizer + Scheduler
3. Epoch 수 + Early Stopping
4. Batch Size
5. Weight Decay

---

### Day 8 (02.11 수) - 실험 라운드 2: 데이터 증강

**목표**: 데이터 증강을 통해 모델의 일반화 성능을 높인다.

**할 일**
- [ ] **실험 D-1: 기본 증강**
  ```python
  import albumentations as A
  from albumentations.pytorch import ToTensorV2

  train_transform = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
      ToTensorV2(),
  ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
  ```
- [ ] **실험 D-2: 강한 증강**
  ```python
  strong_transform = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.3),
      A.RandomBrightnessContrast(p=0.3),
      A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
      A.GaussNoise(var_limit=(10, 50), p=0.2),
      A.Blur(blur_limit=3, p=0.1),
      A.RandomRotate90(p=0.2),
      ToTensorV2(),
  ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
  ```
  - 주의: Object Detection 증강은 **바운딩 박스도 함께 변환**해야 함
- [ ] 증강 없음 vs 기본 증강 vs 강한 증강 mAP 비교
- [ ] **(Experimentation Lead)** 팀 전체 모델(ResNet, YOLO 등) 성능 비교표 업데이트
- [ ] Kaggle 제출

---

### Day 9 (02.12 목) - 실험 라운드 3: 모델 구조 실험

**목표**: Backbone 교체, Anchor/NMS 등 모델 구조 레벨의 실험을 수행한다.

**할 일**
- [ ] **실험 E: Backbone 교체**
  ```python
  # v2 (개선된 버전)
  from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

  # MobileNet (가벼움, 빠른 추론)
  from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
  ```
- [ ] **실험 F-1: Anchor size 조정** (EDA에서 파악한 약 크기에 맞게)
  ```python
  from torchvision.models.detection.rpn import AnchorGenerator
  anchor_generator = AnchorGenerator(
      sizes=((32, 64, 128, 256, 512),),
      aspect_ratios=((0.5, 1.0, 2.0),)
  )
  ```
- [ ] **실험 F-2: NMS threshold 조정** (0.3, 0.5, 0.7 비교)
- [ ] **실험 F-3: Score threshold 조정** (confidence가 낮은 예측 필터링)
- [ ] **(Experimentation Lead)** 전체 실험 결과 종합 정리
  - 어떤 실험이 가장 효과적이었는지 분석
  - 최적 조합 도출
- [ ] Kaggle 제출

---

### Day 10 (02.13 금) - 최종 모델 확정 + 정리

**목표**: 최고 성능 모델을 확정하고, 코드를 정리하고, Kaggle 최종 제출한다.

**할 일**
- [ ] 지금까지 실험 중 **최적 조합으로 최종 학습** (epoch 늘려서)
- [ ] Validation 데이터에 대한 최종 성능 측정
  - mAP@0.5, mAP@0.5:0.95
  - 클래스별 AP (어떤 약을 잘/못 맞추는지)
- [ ] **Kaggle 최종 제출** (팀원과 협의)
- [ ] 코드 정리 및 주석 추가
- [ ] Git 정리 (불필요한 파일 제거, 코드 정돈, commit)
- [ ] **(Experimentation Lead)** 전체 실험 결과 보고서 초안 작성
  - 실험별 성능 비교표
  - 최종 모델 선정 근거
  - 실패한 시도와 이유

---

## Phase 3: 발표 준비 (02.16 ~ 02.23)

### 02.16 (월) ~ 02.20 (금) - 발표자료 + 보고서 작성

**할 일**
- [ ] 최종 발표자료에 포함할 Faster R-CNN 파트 작성
  - 모델 선택 이유 + 아키텍처 설명
  - 실험 과정 요약 (어떤 시도를 했고, 무엇이 효과적이었는지)
  - 성능 비교표 + 그래프
  - 예측 시각화 (성공 사례 + 실패 사례 + 분석)
  - 회고 및 개선 방향
- [ ] 보고서 PDF 작성 (팀 공동)
- [ ] Github Repository README 정리
- [ ] 발표 리허설
- [ ] 협업일지 최종 정리

### 02.23 (월) - 최종 발표

- [ ] **최종 발표 진행** (20분 발표 + 5분 질의응답)
- [ ] 질의응답 대비 예상 질문 정리
- [ ] 협업일지 최종 제출 (23:50까지)
- [ ] 보고서 PDF 제출

---

## 실험 기록 템플릿

아래 표를 복사하여 실험할 때마다 행을 추가한다.

| 날짜 | 실험명 | 변경 변수 | backbone | optimizer | LR | scheduler | epoch | batch | augmentation | mAP@0.5 | Kaggle Score | 비고 |
|------|--------|-----------|----------|-----------|-----|-----------|-------|-------|-------------|---------|-------------|------|
| 02.06 | baseline | - | ResNet50-FPN | SGD | 0.005 | StepLR | 10 | 4 | 없음 | - | - | 첫 실험 |
| 02.10 | exp_A1 | LR | ResNet50-FPN | SGD | 0.001 | StepLR | 10 | 4 | 없음 | - | - | |
| 02.10 | exp_A2 | LR | ResNet50-FPN | SGD | 0.01 | StepLR | 10 | 4 | 없음 | - | - | |
| 02.10 | exp_B1 | optimizer | ResNet50-FPN | Adam | 0.0001 | StepLR | 10 | 4 | 없음 | - | - | |
| 02.10 | exp_B2 | optimizer | ResNet50-FPN | AdamW | 0.0001 | StepLR | 10 | 4 | 없음 | - | - | |
| 02.11 | exp_D1 | augmentation | ResNet50-FPN | best | best | best | 10 | 4 | 기본 증강 | - | - | |
| 02.11 | exp_D2 | augmentation | ResNet50-FPN | best | best | best | 10 | 4 | 강한 증강 | - | - | |
| 02.12 | exp_E1 | backbone | ResNet50-v2 | best | best | best | 10 | 4 | best | - | - | |
| 02.12 | exp_E2 | backbone | MobileNet-v3 | best | best | best | 10 | 4 | best | - | - | |
| 02.13 | final | 최적 조합 | best | best | best | best | 20+ | best | best | - | - | 최종 모델 |

> **실험 원칙**: 한 번에 하나의 변수만 바꾸고, 나머지는 고정한다. 이전 실험에서 best인 값을 다음 실험의 기본값으로 사용한다.

---

## 핵심 체크리스트

### 매일 반드시 할 것
- [ ] 협업일지 작성 (당일 작업 내용, 인사이트, 어려운 점)
- [ ] Git commit & push (작업 단위별)
- [ ] 실험 결과 기록 (위 표 형태)

### 자주 하는 실수 방지
| 번호 | 주의사항 |
|------|----------|
| 1 | 바운딩 박스 형식은 `(x1, y1, x2, y2)`, `x2 > x1`, `y2 > y1` 반드시 확인 |
| 2 | 클래스 라벨은 1부터 시작 (0은 background) |
| 3 | `model.train()` / `model.eval()` 모드 전환 잊지 않기 |
| 4 | GPU 메모리 부족 시 batch_size부터 줄이기 |
| 5 | 학습 결과 저장 안 하고 날리지 않기 (체크포인트 필수) |
| 6 | Kaggle 제출은 팀 단위로만 (개인 제출 금지) |
| 7 | 실험할 때 한 번에 여러 변수 바꾸지 않기 |

### 성능 개선 시 시도할 순서
1. **Learning Rate 조정** (가장 효과 큼)
2. **Optimizer + Scheduler 변경**
3. **데이터 증강** (Flip, Color Jitter, Noise 등)
4. **Epoch 늘리기** (과적합 주시하면서)
5. **Backbone 교체** (ResNet50 → ResNet50-v2 → ResNet101)
6. **Anchor 크기 조정** (데이터에 맞게)
7. **NMS threshold 조정**
8. **이미지 해상도 변경**
9. **데이터 품질 개선** (잘못된 라벨 수정)

---

## 참고 자료

- [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [torchvision Faster R-CNN 공식 문서](https://pytorch.org/vision/stable/models/faster_rcnn.html)
- [Albumentations Object Detection](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)
- [mAP 계산 이해하기](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
