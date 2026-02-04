from torchvision.transforms import v2

                                        ###     padding과 Resize가 작용된 transform 입니다.
train_transforms = v2.Compose([
        v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomRotation(360,fill=255),
    #v2.RandomPosterize(bits=6,p=1.0),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
    v2.Resize((600,600)),
])

test_transforms = v2.Compose([
        v2.Pad(
        padding=(152,0),
        fill=255,
        padding_mode="constant"
    ),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
    v2.Resize((600,600)),
])


                                        ###     padding과 Resize가 적용되지 않은 transform 입니다. 원본 사이즈를 입력 해야할 시 사용.
no_Resize_train_transforms = v2.Compose([
    v2.RandomRotation(360,fill=255),
    #v2.RandomPosterize(bits=6,p=1.0),
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
])

no_Resize_test_transforms = v2.Compose([
    v2.RandomAdjustSharpness(sharpness_factor=100.0, p=1.0),
])


