"""Segmentation model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch (VERY IMPORTANT)
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5,batchnorm: bool = True):
        super().__init__()
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
            batchnorm: Whether to use batch normalization.
        """
        self.encoder = VGG11Encoder(in_channels=in_channels, batchnorm=batchnorm)

        # Decoder
        
        # bottleneck (skip5) is 512 channels, size 14x14
        self.decode4 = DecoderBlock(512, 512, dropout_p)  # 14 → 28
        self.decode3 = DecoderBlock(512, 256, dropout_p)  # 28 → 56
        self.decode2 = DecoderBlock(256, 128, dropout_p)  # 56 → 112
        self.decode1 = DecoderBlock(128, 64, dropout_p)   # 112 → 224

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, skips = self.encoder(x, return_features=True)

        # bottleneck is already at the skip5 level (14x14)
        x = bottleneck
        x = self.decode4(x, skips["skip4"])
        x = self.decode3(x, skips["skip3"])
        x = self.decode2(x, skips["skip2"])
        x = self.decode1(x, skips["skip1"])

        return self.final_conv(x)