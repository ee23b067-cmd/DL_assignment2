# Oxford-IIIT Pet Multi-Task Perception

PyTorch implementation for the Oxford-IIIT Pet dataset covering breed classification, bounding-box localization, and trimap segmentation. The repository also includes a combined multi-task model that reuses a shared VGG11 encoder.

## Features

- `VGG11Classifier` for 37-breed classification
- `VGG11Localizer` for bounding-box regression in `(cx, cy, w, h)` format
- `VGG11UNet` for 3-class segmentation
- `MultiTaskPerceptionModel` for joint inference across all three tasks
- `IoULoss` for localization training

## Repository Layout

```text
train.py              # Training entry point for the single-task models
inference.py          # Simple inference wrapper for the 
data/pets_dataset.py  # Oxford-IIIT Pet dataset loader
losses/iou_loss.py    # IoU loss for bounding-box regression
models/               # Classifier, localizer, segmenter, and shared backbone code
```

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The project uses PyTorch, NumPy, Pillow, scikit-learn, matplotlib, albumentations, and Weights & Biases.

## Dataset Setup

The loader expects the Oxford-IIIT Pet dataset under `data/oxford-iiit-pet/` with this structure:

```text
data/oxford-iiit-pet/
	images/
	annotations/
		xmls/
		trimaps/
		trainval.txt
		test.txt
```

If the folder contains `images.tar.gz` and `annotations.tar.gz`, the dataset loader can extract them automatically the first time it runs. The default dataset root in `train.py` is `data/oxford-iiit-pet`; update that path if your data lives elsewhere.

## Training

Run the default training script with:

```bash
python train.py
```

This script trains the classifier, localizer, and segmenter sequentially. Best checkpoints are written to `checkpoints/classifier.pth`, `checkpoints/localizer.pth`, and `checkpoints/unet.pth`.

The file also defines `train_multitask(...)` if you want to run joint training manually from Python, but it is not wired into the default `main()` entry point.

Training uses CUDA automatically when available and falls back to CPU otherwise.


## Inference

The `inference.py` script loads the multi-task model and runs a forward pass on one image. The current file imports `MultiTaskPerceptionModel` from a top-level `multitask` module; if you keep the current repository layout, change that import to `from models.multitask import MultiTaskPerceptionModel` before running it.

Example usage after that import is aligned:

```bash
python inference.py path/to/image.jpg --classifier checkpoints/classifier.pth --localizer checkpoints/localizer.pth --unet checkpoints/unet.pth
```

The script prints the output tensor shapes for each task.

## Notes

- Images and masks are resized to 224 x 224.
- Classification and segmentation loaders do not require bounding boxes.
- Localization and multi-task training expect bounding boxes to be available in the annotations.


git hub : https://github.com/ee23b067-cmd/DL_assignment2/tree/main

