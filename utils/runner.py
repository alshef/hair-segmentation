import sys
sys.path.append("..")
from pathlib import Path
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Trainer:
    def __init__(self, config: dict, experiment_path: Path) -> None:
        self.exp_path = experiment_path
        self.config = config
        self.writer = SummaryWriter(experiment_path)
        self._fix_all_seeds(config["seed"])

    def run(self) -> None:
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def _fix_all_seeds(self, seed: int = 42) -> None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
