from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

def train_valid_build_dataloaders(train_dataset, valid_dataset, batch_size, num_workers=0, shuffle=True, valid_shuffle=False):  #데이터 로더 1번, train dataset과 valid dataset을 함께 받음

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=valid_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader

def full_train_build_dataloaders(train_dataset, batch_size, num_workers=0, shuffle=True):  #데이터 로더 2번, 하나의 데이터 로더만 받음

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader

def test_build_dataloaders(test_dataset, batch_size, num_workers=0, shuffle=False):         #테스트 데이터 로더, collate_fn 없음

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return test_loader