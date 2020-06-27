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
        self.model = models.__dict__[self.config["model"]["type"]]()

    def _get_dataloaders(self):
        train_dataloader, val_dataloader = get_data_loaders(self.config)
        return train_dataloader, val_dataloader

    def _get_loss(self):
        return torch.nn.BCEWithLogitsLoss()

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=float(self.config["optimizer"]["lr"]))

    def run(self) -> None:
        self.

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
