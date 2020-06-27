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
        loss = self._get_loss()
        optimizer = self._get_optimizer()
        train_loader, val_loader = self._get_dataloaders()
        n_epochs = self.config["train"]["epochs"]

        self.model.to(self.device)
        loss.to(self.device)

        for epoch in range(n_epochs):
            train_epoch_loss = self.train(train_loader, loss, optimizer)
            val_epoch_loss = self.validate(val_loader, loss)

            self.writer.add_scalars('Losses', {"train": train_epoch_loss, "val": val_epoch_loss}, epoch)
        self.writer.close()

    def train(self, train_loader, loss, optimizer) -> float:
        running_loss = 0.0
        self.model.train()

        for i, (images, target) in enumerate(train_loader):
            images.to(self.device)
            target.to(self.device)

            output = self.model(images)
            batch_loss = loss(output, target)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Train loss: {epoch_loss}')

        return epoch_loss

    def validate(self, val_loader, loss) -> float:
        running_loss = 0.0
        self.model.eval()

        for images, masks in val_loader:
            images.to(self.device)
            masks.to(self.device)

            with torch.no_grad():
                output = self.model(images)
                batch_loss = loss(output, masks)

            running_loss += batch_loss.item()

        epoch_loss = running_loss / len(val_loader)
        print(f'Val loss: {epoch_loss}')

        return epoch_loss

    def _fix_all_seeds(self, seed: int = 42) -> None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
