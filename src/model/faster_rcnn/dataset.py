import torch
from torch.utils.data import Dataset


class PillDataset(Dataset):
    """
    경구약제 이미지 Object Detection용 Dataset.

    Faster R-CNN이 요구하는 포맷:
    - image: [C, H, W] FloatTensor (0~1 범위)
    - target: dict {
        'boxes': [N, 4] FloatTensor (x1, y1, x2, y2),
        'labels': [N] Int64Tensor (1부터 시작, 0은 background),
        'image_id': Int64Tensor,
        'area': [N] FloatTensor,
        'iscrowd': [N] Int64Tensor
      }
    """

    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = []  # TODO: 데이터 경로 로딩

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # TODO: 데이터 포맷에 맞게 구현
        # 1. 이미지 로드
        # 2. 라벨(바운딩 박스 + 클래스) 파싱
        # 3. target dict 구성
        # 4. transforms 적용
        raise NotImplementedError("데이터 포맷 확인 후 구현")


def collate_fn(batch):
    """Faster R-CNN용 collate 함수. 기본 collate는 동작하지 않으므로 필수."""
    return tuple(zip(*batch))
