from typing import Tuple

import torchvision.transforms as T
import albumentations as A


def get_train_transforms(dataset_name: str,
                         size: Tuple[int, int] = (224, 224)) -> Tuple[A.Compose, T.Compose, T.Compose]:
    resizer = A.Resize(*size)
    if dataset_name == "figaro1k":
        resizer = A.RandomCrop(*size)

    joint_transforms = A.Compose([
        resizer,
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5),
    ])

    image_transforms = T.Compose([
        T.ToPILImage(mode="RGB"),
        T.ColorJitter(0.3, 0.05, 0.05, 0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transforms = T.Compose([
        T.ToTensor()
    ])

    return joint_transforms, image_transforms, mask_transforms


def get_test_transforms(dataset_name: str,
                        size: Tuple[int, int] = (512, 512)) -> Tuple[A.Compose, T.Compose, T.Compose]:
    joint_transforms = A.Compose([
        A.Resize(*size)
    ])

    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transforms = T.Compose([
        T.ToTensor()
    ])

    return joint_transforms, image_transforms, mask_transforms
