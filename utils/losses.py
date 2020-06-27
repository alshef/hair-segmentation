import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn


class ImageGradientLoss(_Loss):
    def __init__(self, device, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = device

    def forward(self, logits: torch.Tensor, gray_images: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))
        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_images, gradient_tensor_x)
        M_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_images, gradient_tensor_y)
        M_y = F.conv2d(pred, gradient_tensor_y)

        G = torch.sqrt(torch.pow(M_x, 2) + torch.pow(M_y, 2))

        gradient = 1 - torch.pow(I_x * M_x + I_y * M_y, 2)

        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / (torch.sum(G) + 1e-6)

        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0

        return image_gradient_loss
