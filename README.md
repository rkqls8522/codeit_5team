# Faster R-CNN v9 5-Fold Repro Guide

## Scope
- Final reproduction target: v9 5-Fold + WBF submission flow
- Core scripts:
  - `src/model/faster_rcnn/config_v9_5fold.py`
  - `src/model/faster_rcnn/train_v9_5fold.py`
  - `src/model/faster_rcnn/predict_v9_5fold_wbf.py`

## Required paths
- Train images: `data/original/train_images`
- Train annotations: `data/processed/train_annotations`
- Test images: `data/original/test_images`

## Environment
```bash
pip install torch torchvision albumentations ensemble-boxes scikit-learn tqdm
```

## Run order
1. Train 5 folds and save checkpoints
```bash
cd src/model/faster_rcnn
python train_v9_5fold.py
```

2. Check checkpoints
```bash
ls ../../../checkpoints/5fold/best_model_fold*.pth
```

3. Run 5-Fold WBF inference
```bash
python predict_v9_5fold_wbf.py
```

4. Check final CSV
```bash
ls ../../../submission_5fold_wbf.csv
```

## Output
- Checkpoints: `checkpoints/5fold/best_model_fold{1..5}.pth`
- Submission CSV: `submission_5fold_wbf.csv`

## PR policy
- Include only core code and settings.
- Exclude large outputs (images, temporary notebooks, intermediate artifacts).
