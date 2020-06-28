from typing import Tuple
from pathlib import Path

from PIL import Image, ImageOps


def pad_image_for_model(image, model_depth: int = 5, fill_const: int = 0) -> Image:
    width, height = image.size
    scale = 2 ** model_depth

    left_padding = 0
    top_padding = 0
    right_padding = 0
    bottom_padding = 0

    if width % scale != 0:
        delta = scale - width % scale
        left_padding = delta // 2
        right_padding = delta - left_padding

    if height % scale != 0:
        delta = scale - height % scale
        top_padding = delta // 2
        bottom_padding = delta - top_padding

    image = ImageOps.expand(
        image,
        border=(
            left_padding,
            top_padding,
            right_padding,
            bottom_padding
        ),
        fill=fill_const
    )
    return image


def prepare_image(image_path: Path) -> Tuple[Image, Tuple[int, int], Tuple[int, int]. Image]:
    raw_image = Image.open(image_path).convert("RGB")
    raw_size = raw_image.size
    image = raw_image.copy()

    image.thumbnail((256, 256))
    new_size = image.size

    out_image = pad_image_for_model(image)
    return out_image, raw_size, new_size, raw_image
