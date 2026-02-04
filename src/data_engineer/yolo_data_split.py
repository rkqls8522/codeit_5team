import shutil
from pathlib import Path as Path_Path
from sklearn.model_selection import train_test_split


def split_yolo_dataset(image_dir, anntation_dir, output_dir, val_ratio=0.2, seed=42, shuffle=True, image_exts=(".jpg", ".jpeg", ".png")):
    image_dir = Path_Path(image_dir)
    anntation_dir = Path_Path(anntation_dir)
    output_dir = Path_Path(output_dir)

    # output 폴더 구조 생성
    for p in [output_dir / "images/train", output_dir / "images/val", output_dir / "labels/train", output_dir / "labels/val"]:
        p.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 수집
    images = [img for img in image_dir.iterdir() if img.suffix.lower() in image_exts]

    if len(images) == 0:
        raise ValueError("이미지 파일이 없습니다.")

    # train / val split
    train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=seed, shuffle=True)

    def copy_pair(img_path, split):
        label_path = anntation_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"annotation 없음: {img_path.name}")
            return

        shutil.copy2(img_path, output_dir / f"images/{split}" / img_path.name)
        shutil.copy2(label_path, output_dir / f"labels/{split}" / label_path.name)

    for img in train_imgs:
        copy_pair(img, "train")

    for img in val_imgs:
        copy_pair(img, "val")

    print("=== Split 완료 ===")
    print(f"Train: {len(train_imgs)}")
    print(f"Val: {len(val_imgs)}")