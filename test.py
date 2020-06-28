import torch
from PIL import Image, ImageOps

import models


def get_model(model_path: str, model_type: str = "UNet") -> torch.nn.Module:
    model = models.__dict__[model_type]()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


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
