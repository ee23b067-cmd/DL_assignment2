# test_multitask.py

import torch
from multitask import MultiTaskPerceptionModel

def test_multitask_forward():

    model = MultiTaskPerceptionModel()
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(x)

    print("Checking outputs...")

    assert "classification" in outputs
    assert "localization" in outputs
    assert "segmentation" in outputs

    assert outputs["classification"].shape[0] == 2
    assert outputs["classification"].shape[1] == 37

    assert outputs["localization"].shape == (2, 4)

    assert outputs["segmentation"].shape == (2, 3, 224, 224)

    print("✅ Multi-task forward pass correct")


def test_value_ranges():
    model = MultiTaskPerceptionModel()
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    loc = out["localization"]

    print("Localization output:", loc)

    assert (loc >= 0).all(), "Negative bbox values detected"
    assert (loc <= 224).all(), "BBox exceeds image size"

    print("✅ Localization scaling correct")


if __name__ == "__main__":
    test_multitask_forward()
    test_value_ranges()