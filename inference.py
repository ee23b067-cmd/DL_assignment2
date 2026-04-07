"""Simple inference entry point for the multi-task model.
"""

import argparse

import numpy as np
import torch
from PIL import Image

from multitask import MultiTaskPerceptionModel


def load_image(image_path: str, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
	image = Image.open(image_path).convert("RGB")
	image = image.resize(size, Image.BILINEAR)
	tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
	mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
	return (tensor - mean) / std


def run_inference(image_path: str, classifier_path: str, localizer_path: str, unet_path: str) -> dict:
	model = MultiTaskPerceptionModel(
		classifier_path=classifier_path,
		localizer_path=localizer_path,
		unet_path=unet_path,
		drive=False,
	)
	model.eval()

	image = load_image(image_path).unsqueeze(0)

	with torch.no_grad():
		outputs = model(image)

	return outputs


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("image", type=str)
	parser.add_argument("--classifier", type=str, default="checkpoints/classifier.pth")
	parser.add_argument("--localizer", type=str, default="checkpoints/localizer.pth")
	parser.add_argument("--unet", type=str, default="checkpoints/unet.pth")
	args = parser.parse_args()

	outputs = run_inference(args.image, args.classifier, args.localizer, args.unet)
	print({key: value.shape for key, value in outputs.items()})


if __name__ == "__main__":
	main()
