from pathlib import Path as Path_Path
from PIL import Image
from torchvision.transforms import v2
import torch
import glob
import os


# transform:            적용할 transform
# image_dir:            이미지 폴더 경로
# master_dir:           저장할 폴더 경로
# output_dir_name:      저장할 폴더 이름

def apply_v2_compose_and_save(transform, image_dir, master_dir, output_dir_name="converted img", load_exts = ("*.jpg", "*.png", "*.jpeg")):
    master_dir = Path_Path(master_dir)
    save_dir = master_dir / output_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in load_exts:
        image_paths.extend(
            glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
        )

    for img_path in image_paths:
        img_path = Path_Path(img_path)

        # 이미지 로드 (PIL)
        image = Image.open(img_path).convert("RGB")

        transformed = transform(image)

        # Tensor -> PIL 변환 (필요한 경우)
        # v2는 PIL도 지원하지만 Tensor 변환이 섞일 수 있으므로 안전하게 처리
        if isinstance(transformed, torch.Tensor):
            transformed = v2.ToPILImage()(transformed)

        # 저장 파일명
        save_path = save_dir / img_path.name
        transformed.save(save_path)

    print(f"{len(image_paths)}개 이미지 저장됨")
    return save_dir