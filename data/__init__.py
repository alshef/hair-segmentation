from typing import Tuple
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import HairDataset
from .transforms import get_test_transforms, get_train_transforms


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    data_path = Path(config["data_path"])
    dataset_name = config["data"]["train"]["dataset_name"]
    dataset_path = data_path / dataset_name

    train_transforms = get_train_transforms(dataset_name, size=tuple(config["data"]["train"]["size"]))
    train_dataset = HairDataset(
        dataset_path=dataset_path,
        mode="train",
        joint_transforms=train_transforms[0],
        image_transforms=train_transforms[1],
        mask_transforms=train_transforms[2]
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["data"]["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["train"]["num_workers"],
    )

    val_transforms = get_test_transforms(dataset_name, size=tuple(config["data"]["val"]["size"]))
    val_dataset = HairDataset(
        dataset_path=dataset_path,
        mode="val",
        joint_transforms=val_transforms[0],
        image_transforms=val_transforms[1],
        mask_transforms=val_transforms[2]
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config["data"]["val"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["val"]["num_workers"]
    )

    return train_dataloader, val_dataloader
