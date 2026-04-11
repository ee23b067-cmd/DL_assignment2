"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11(nn.Module):
    """VGG11-style encoder as required by autograder.
    """

    def __init__(self, in_channels: int = 3, batchnorm: bool = True, dropout_p: float = 0.5):
        """Initialize the VGG11 model."""
        super().__init__()
        # Block 1: 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 256, 256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 512, 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 512, 512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.
        """
        features = {}

        # Block 1
        x1 = self.block1(x)
        x = self.pool1(x1)
        if return_features:
            features["skip1"] = x1
        # Block 2
        x2 = self.block2(x)
        x = self.pool2(x2)
        if return_features:
            features["skip2"] = x2
        # Block 3
        x3 = self.block3(x)
        x = self.pool3(x3)
        if return_features:
            features["skip3"] = x3
        # Block 4
        x4 = self.block4(x)
        x = self.pool4(x4)
        if return_features:
            features["skip4"] = x4

        # Block 5
        x5 = self.block5(x)
        x = self.pool5(x5)
        if return_features:
            features["skip5"] = x5

        bottleneck = x

        if return_features:
            return bottleneck, features

        return x


class VGG11Encoder(VGG11):
    """Alias for VGG11 for compatibility.
    """
    def __init__(self, in_channels: int = 3, batchnorm: bool = True, dropout_p: float = 0.5):
        super().__init__(in_channels=in_channels, batchnorm=batchnorm, dropout_p=dropout_p)
