import sys
sys.path.append("..")
from pathlib import Path
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import models
from data import get_data_loaders


class Trainer:
    def __init__(self, config: dict, experiment_path: Path) -> None:
        self.exp_path = experiment_path
        self.config = config
        self.writer = SummaryWriter(experiment_path)
        self._fix_all_seeds(config["seed"])
        self.device = f"cuda:{config['gpu']}"

    def _get_model(self) -> None:
        self.model = models.__dict__[self.config["model"]["type"]]()

    def _get_dataloaders(self) -> None:
        self.train_dataloader, self.val_dataloader = get_data_loaders(self.config)
        self.n_train_examples = len(self.train_dataloader)
        self.n_val_examples = len(self.val_dataloader)

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
