"""Oxford-IIIT Pet dataset loader for classification, localization, and segmentation."""

import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from xml.etree import ElementTree as ET
import torch
from torchvision.transforms import functional as F


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, root, split="train", transform=None, bbox=False, load_mask=False):
        self.root = root
        self.transform = transform
        self.bbox = bbox
        self.load_mask = load_mask
        self.target_size = (224, 224)

        self.image_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.trimap_dir = os.path.join(self.ann_dir, "trimaps")

        if split == "train":
            split_file = os.path.join(self.ann_dir, "trainval.txt")
        else:
            split_file = os.path.join(self.ann_dir, "test.txt")

        
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                label = int(parts[1]) - 1
            
                if self.bbox:
                    xml_path = os.path.join(self.ann_dir, "xmls", img_name + ".xml")
                    if not os.path.exists(xml_path):
                        continue  
            
                self.samples.append((img_name, label))

    def _get_sample(self, idx):
        if self._cached_idx != idx or self._cached_sample is None:
            self._cached_idx = idx
            self._cached_sample = self.dataset[idx]
        return self._cached_sample

    def _load_image(self, idx):
        img_name, _ = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name + ".jpg")

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.target_size)

        tensor = F.to_tensor(image)
        tensor = F.normalize(tensor, mean=(0.485,0.456,0.406),
                                    std=(0.229,0.224,0.225))
        return tensor

    def _load_label(self, idx):
        _, label = self.samples[idx]
        return torch.tensor(label, dtype=torch.long)

    def _load_mask(self, idx):
        img_name, _ = self.samples[idx]
        mask_path = os.path.join(self.trimap_dir, img_name + ".png")

        mask = Image.open(mask_path)
        mask = mask.resize(self.target_size)

        mask = F.pil_to_tensor(mask).squeeze(0).long() - 1
        return mask

    def _load_bbox(self, idx):
        img_name, _ = self.samples[idx]
        xml_path = os.path.join(self.ann_dir, "xmls", img_name + ".xml")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # resize scaling
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        w, h = Image.open(img_path).size

        sx = self.target_size[0] / w
        sy = self.target_size[1] / h

        xmin, xmax = xmin * sx, xmax * sx
        ymin, ymax = ymin * sy, ymax * sy

        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        return (x, y, w, h)
    def _apply_transform(self, item):
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        out = {}
        out["image"] = self._load_image(idx)
        out["label"] = self._load_label(idx)

        if self.load_mask:
            out["mask"] = self._load_mask(idx)

        if self.bbox:
            bbox = self._load_bbox(idx)
            if bbox is not None:
                out["bbox"] = torch.tensor(bbox, dtype=torch.float32)

        return out
