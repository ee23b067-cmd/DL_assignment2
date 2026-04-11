"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, batchnorm: bool = True):
        super().__init__()
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            batchnorm: Whether to use batch normalization.
        """
        self.encoder = VGG11Encoder(in_channels=in_channels, batchnorm=batchnorm)
        self.image_size = 224  # Assuming input images are 224x224
        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4)  # (cx, cy, w, h)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            [B, 4] bounding boxes in pixel space
        """
        _, _, h, w = x.shape

        x = self.encoder(x)
        out = self.localization_head(x)

        # Normalize to [0,1]
        centers = torch.sigmoid(out[:, :2]) * self.image_size
        wh = torch.sigmoid(out[:, 2:]) * self.image_size
        boxes = torch.cat([centers, wh], dim=1)
        return boxes
