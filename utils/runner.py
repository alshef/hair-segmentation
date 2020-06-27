import sys
sys.path.append("..")
from pathlib import Path
from typing import Tuple
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import models
from data import get_data_loaders
from .meters import AverageMeter
from .metrics import IoUMetric
from .losses import HairMattingLoss


class Trainer:
    def __init__(self, config: dict, experiment_path: Path) -> None:
        self.exp_path = experiment_path
        self.config = config
        self.writer = SummaryWriter(experiment_path)
        self._fix_all_seeds(config["seed"])
        self.device = torch.device(f"cuda:{config['gpu']}")
        self.model = models.__dict__[self.config["model"]["type"]]()

    def _get_dataloaders(self):
        train_dataloader, val_dataloader = get_data_loaders(self.config)
        return train_dataloader, val_dataloader

    def _get_loss(self):
        loss_name = self.config["model"]["loss"]
        if loss_name == "BCEWithLogits":
            return torch.nn.BCEWithLogitsLoss()
        elif loss_name == "HairMatting":
            return HairMattingLoss(self.device, 0.5)
        else:
            raise ValueError("Incorrect loss name in config")

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=float(self.config["optimizer"]["lr"]))

    def run(self) -> None:
        loss = self._get_loss()
        optimizer = self._get_optimizer()
        train_loader, val_loader = self._get_dataloaders()
        n_epochs = self.config["train"]["epochs"]

        self.model = self.model.to(self.device)
        loss = loss.to(self.device)

        best_iou = 0.0
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}/{n_epochs}")
            train_epoch_loss = self.train(train_loader, loss, optimizer)
            val_epoch_loss, val_epoch_iou = self.validate(val_loader, loss)

            self.writer.add_scalars('Losses', {"train": train_epoch_loss, "val": val_epoch_loss}, epoch)
            self.writer.add_scalar("Metrics/IoU/val", val_epoch_iou, epoch)

            if val_epoch_iou > best_iou:
                best_iou = val_epoch_iou
                torch.save(self.model.state_dict(), self.exp_path / "best_checkpoint.pth")
                print(f"Best epoch: {epoch}")
        self.writer.close()
        torch.save(self.model.state_dict(), self.exp_path / "model.pth")

    def train(self, train_loader, loss, optimizer) -> float:
        bce_losses = AverageMeter()
        self.model.train()

        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            output = self.model(images)
            batch_loss = loss(output, masks)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            bce_losses.update(batch_loss.item(), images.size()[0])

        epoch_loss = bce_losses.value()
        print(f'Train loss: {epoch_loss}')

        return epoch_loss

    def validate(self, val_loader, loss) -> Tuple[float, float]:
        iou_metric = IoUMetric()
        iou_results = AverageMeter()
        bce_losses = AverageMeter()
        self.model.eval()

        for images, masks in val_loader:
            n = images.size()[0]
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.no_grad():
                output = self.model(images)
                batch_loss = loss(output, masks)
                iou = iou_metric.compute(output, masks)

            bce_losses.update(batch_loss.item(), n)
            iou_results.update(iou.item(), n)

        epoch_loss = bce_losses.value()
        epoch_iou = iou_results.value()
        print(f'Val loss: {epoch_loss}')
        print(f"Val IoU: {epoch_iou}")

        return epoch_loss, epoch_iou

    def _fix_all_seeds(self, seed: int = 42) -> None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
