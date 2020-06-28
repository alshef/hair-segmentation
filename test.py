from pathlib import Path
from typing import Tuple
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

import models
from utils.image_funcs import prepare_image, make_mask_from_logits


model_types = [name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name])]


def get_model(model_path: str, model_type: str = "UNet") -> torch.nn.Module:
    model = models.__dict__[model_type]()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


def prepare_batch(image: Image.Image, image_transforms: transforms.Compose) -> torch.Tensor:
    image_tensor = image_transforms(image)
    image_batch = torch.unsqueeze(image_tensor, 0)

    return image_batch


def define_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Test hair segmentator on dir with images')
    parser.add_argument("--experiment-name", type=str,
                        help='Experiment name. Used to load model')
    parser.add_argument("--model-type", type=str, default="UNet", choices=model_types,
                        help="Model type. Used to load model state_dict")
    parser.add_argument("--path-to-images", type=str, help="Path to the folder with images to segment")
    parser.add_argument("--path-to-masks", type=str,
                        help="Path to dir where masks should be saved. Dir 'masks' will be created there")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold of segmentation")

    return parser


def main():
    cli_parser = define_argparser()
    args = cli_parser.parse_args()


if __name__ == "__main__":
    main()
