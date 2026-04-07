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

        # Initialize task-specific models
        self.classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels,
            batchnorm=True,
            head_batchnorm=True,
        )
        self.localizer_model = VGG11Localizer(in_channels=in_channels, batchnorm=True)
        self.unet_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, batchnorm=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def resolve_checkpoint_path(path):
            if os.path.exists(path):
                return path

            checkpoint_path = os.path.join("checkpoints", os.path.basename(path))
            if os.path.exists(checkpoint_path):
                return checkpoint_path

            return path

        def load_weights(model, path):
            path = resolve_checkpoint_path(path)

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
        load_weights(self.classifier_model, classifier_path)
        load_weights(self.localizer_model, localizer_path)
        load_weights(self.unet_model, unet_path)

    def forward(self, x: torch.Tensor):
        class_out = self.classifier_model(x)
        loc_out = self.localizer_model(x)
        seg_out = self.unet_model(x)

        return {
            "classification": class_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }