from torchvision.transforms import v2
import torch

                                        ###     padding과 Resize가 작용된 transform 입니다.
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomRotation(360,fill=255),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
    v2.Resize((600,600)),
])
test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
    v2.Resize((600,600)),
])
v1_transform = (train_transforms, test_transforms)


train_transforms2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomRotation(360,fill=255),
    v2.Resize((600,600)),
])
test_transforms2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.Resize((600,600)),
])
v2_transform = (train_transforms2, test_transforms2)


train_transforms3 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
    v2.Resize((600,600)),
])
test_transforms3 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
    v2.Resize((600,600)),
])
v3_transform = (train_transforms3, test_transforms3)



###---------------- 리사이즈, padding 하지 않음



n_train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomRotation(360,fill=255),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
])
n_test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
])
n_v1_transform = (n_train_transforms, n_test_transforms)


n_train_transforms2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomRotation(360,fill=255),
])
n_test_transforms2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
n_v2_transform = (n_train_transforms2, n_test_transforms2)


n_train_transforms3 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
])
n_test_transforms3 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
])
n_v3_transform = (n_train_transforms3, n_test_transforms3)


