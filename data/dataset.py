from typing import Tuple
from pathlib import Path

import torch
from torch.utils import data
from torchvision.transforms import Compose as T_Compose
from albumentations.core.composition import Compose as A_Compose
import cv2


class HairDataset(data.Dataset):
    def __init__(self,
                 dataset_path: Path,
                 mode: str,
                 joint_transforms: A_Compose,
                 image_transforms: T_Compose,
                 mask_transforms: T_Compose) -> None:
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        self.image_path_list = sorted(list((dataset_path / mode / "images").iterdir()))
        self.mask_path_list = sorted(list((dataset_path / mode / "masks").iterdir()))

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = str(self.image_path_list[idx])
        mask_path = str(self.mask_path_list[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        augmented = self.joint_transforms(image=image, mask=mask)

        image = self.image_transforms(augmented["image"])
        mask = self.mask_transforms(augmented["mask"])

        return image, mask
