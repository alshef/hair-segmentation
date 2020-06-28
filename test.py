import torch

import models


def get_model(model_path: str, model_type: str = "UNet") -> torch.nn.Module:
    model = models.__dict__[model_type]()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model
