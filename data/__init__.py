from typing import Tuple
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import HairDataset
from .transforms import get_test_transforms, get_train_transforms


def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    data_path = Path("/home/alshev/Projects/hair_segmentation_test_assignment/datasets/processed/")
    dataset_name = "celeba_hq"
    dataset_path = data_path / dataset_name

    train_transforms = get_train_transforms(dataset_name)
    train_dataset = HairDataset(
        dataset_path=dataset_path,
        mode="train",
        joint_transforms=train_transforms[0],
        image_transforms=train_transforms[1],
        mask_transforms=train_transforms[2]
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=5,
    )

    val_transforms = get_test_transforms(dataset_name)
    val_dataset = HairDataset(
        dataset_path=dataset_path,
        mode="val",
        joint_transforms=val_transforms[0],
        image_transforms=val_transforms[1],
        mask_transforms=val_transforms[2]
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=5
    )

    return train_dataloader, val_dataloader
