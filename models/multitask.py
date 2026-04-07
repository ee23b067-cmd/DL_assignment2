import os
import torch
import torch.nn as nn

try:
    from numpy._core.multiarray import scalar as numpy_scalar
except ImportError:
    from numpy.core.multiarray import scalar as numpy_scalar

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet



class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        drive: bool = True,
    ):
        super().__init__()

        # Initialize models
        classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
            batchnorm=True,
            head_batchnorm=True,
        )
        localizer_model = VGG11Localizer(in_channels=in_channels, batchnorm=True)
        unet_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, batchnorm=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def load_weights(model, path):
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=device, weights_only=False)
                except TypeError:
                    with torch.serialization.safe_globals([numpy_scalar]):
                        checkpoint = torch.load(path, map_location=device)
                except Exception:
                    with torch.serialization.safe_globals([numpy_scalar]):
                        checkpoint = torch.load(path, map_location=device, weights_only=False)

                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict)
                print(f"Loaded weights from {path}")
            else:
                print(f"Warning: {path} not found.")
        if drive:
            try:
                import importlib
                gdown = importlib.import_module("gdown")
                gdown.download(id="1LE9vRm3EHsWLHy58WDyWkjmiSgDllMCG", output=classifier_path, quiet=False)
                gdown.download(id="1QFDa-O74KMjanvWc1_oCpxANtzn4N7vJ", output=localizer_path, quiet=False)
                gdown.download(id="14rA4MTIgPRg9dj51oUj1ErUKsRHr862a", output=unet_path, quiet=False)
            except ImportError:
                print("Warning: gdown is not installed, skipping remote checkpoint download.")
        load_weights(classifier_model, classifier_path)
        load_weights(localizer_model, localizer_path)
        load_weights(unet_model, unet_path)

        # Shared backbone
        self.backbone = classifier_model.encoder


        # Heads
        self.classification_head = classifier_model.classifier
        self.localization_head = localizer_model.localization_head
        self.localization_activation = nn.Sigmoid()

        # Segmentation decoder
        self.decode4 = unet_model.decode4
        self.decode3 = unet_model.decode3
        self.decode2 = unet_model.decode2
        self.decode1 = unet_model.decode1
        self.segmentation_final = unet_model.final_conv

    def forward(self, x: torch.Tensor):
        bottleneck, skips = self.backbone(x, return_features=True)

        # ✅ Classification
        class_out = self.classification_head(bottleneck)

        # ✅ Localization
        loc_out = self.localization_head(bottleneck)

        _, _, h, w = x.shape
        loc_out = self.localization_activation(loc_out)

        loc_out = loc_out.clone()
        loc_out[:, 0] *= w
        loc_out[:, 1] *= h
        loc_out[:, 2] *= w
        loc_out[:, 3] *= h

        # ✅ Segmentation
        s = bottleneck
        s = self.decode4(s, skips["skip4"])
        s = self.decode3(s, skips["skip3"])
        s = self.decode2(s, skips["skip2"])
        s = self.decode1(s, skips["skip1"])
        seg_out = self.segmentation_final(s)

        return {
            "classification": class_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }