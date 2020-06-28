from pathlib import Path
from typing import Tuple
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

import models
from utils.image_funcs import prepare_image, make_mask_from_logits


def get_model(model_path: str, model_type: str = "UNet") -> torch.nn.Module:
    model = models.__dict__[model_type]()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


def prepare_batch(image: Image, image_transforms: transforms.Compose) -> torch.Tensor:
    image_tensor = image_transforms(image)
    image_batch = torch.unsqueeze(image_tensor, 0)

    return image_batch


def define_argparser():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
