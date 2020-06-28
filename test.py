from pathlib import Path
from typing import Tuple
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from tqdm import tqdm

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

    experiments_path = Path("outputs")

    path_to_masks = Path(args.path_to_masks) / "masks"
    path_to_masks.mkdir(parents=True)

    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = get_model(experiments_path / args.experiment_name / "best_checkpoint.pth")

    path_to_images = Path(args.path_to_images)
    for image_path in tqdm(path_to_images.glob("*.jpg")):
        image_name = image_path.name
        image, raw_size, size_after_resizing, raw_image = prepare_image(image_path)
        batch = prepare_batch(image, image_transforms)

        with torch.no_grad():
            logits = model(batch)

        mask = make_mask_from_logits(logits, raw_size, size_after_resizing, threshold=args.threshold)
        mask.save(path_to_masks / image_name)


if __name__ == "__main__":
    main()
