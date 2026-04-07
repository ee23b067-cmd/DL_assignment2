"""Dataset for Oxford-IIIT Pet (Classification + Localization + Segmentation)"""

import os
import xml.etree.ElementTree as ET
import tarfile
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(self, root: str, split: str = "trainval", transform=None, bbox = True,target_size=(224, 224)):
        """
        Args:
            root: Root directory of dataset.
            split: "train" or "test".
            transform: Transformations for the image.
            bbox: Whether to include bounding box information.
            target_size: Resize images and masks to this size.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.bbox = bbox
        self.target_size = target_size
        self.base_dir = root
        self.images_dir = os.path.join(self.base_dir, "images")
        self.anns_dir = os.path.join(self.base_dir, "annotations")
        self.xml_dir = os.path.join(self.anns_dir, "xmls")
        self.mask_dir = os.path.join(self.anns_dir, "trimaps")
        split_name = {
            "train": "trainval",
            "val": "test",
            "valid": "test",
            "test": "test",
            "trainval": "trainval",
        }.get(split, split)
        self.split_file = os.path.join(self.anns_dir, f"{split_name}.txt")
        # File list
        self._ensure_extracted()

        self.class2idx = self._build_class2idx()
        self.samples = self._build_samples()

    def _ensure_extracted(self):
        if not os.path.exists(self.images_dir):
            with tarfile.open(os.path.join(self.base_dir, "images.tar.gz"), "r:gz") as tar:
                tar.extractall(path=self.base_dir)
        if not os.path.exists(self.anns_dir):
            with tarfile.open(os.path.join(self.base_dir, "annotations.tar.gz"), "r:gz") as tar:
                tar.extractall(path=self.base_dir)

    def _build_class2idx(self):

        class_names = set()
        with open(self.split_file, "r") as f:
            for line in f:
                img_id, class_id, _, _ = line.strip().split()
                label = img_id.rsplit("_",1)[0]
                class_names.add((label, int(class_id) - 1))

        class_names = sorted(list(class_names))
        class2idx = {name: idx for name, idx in class_names}
        idx2class = {idx: name for name, idx in class2idx.items()}
        self.idxtoclass = idx2class
        return class2idx 
    def _build_samples(self):
        samples = []
        skipped = 0
        with open(self.split_file, "r") as f:
            for line in f:
                img_id, class_id, _, _ = line.strip().split()
                img_name = f"{img_id}.jpg"
                mask_name = f"{img_id}.png"
                xml_name = f"{img_id}.xml"

                label_name = img_id.rsplit("_",1)[0]

                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.mask_dir, mask_name)
                xml_path = os.path.join(self.xml_dir, xml_name)
                if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                    skipped += 1
                    continue
                if self.bbox and not os.path.isfile(xml_path):
                    skipped += 1
                    continue

                samples.append({
                    "img_path": img_path,
                    "label_name": label_name,
                    "label_id": self.class2idx[label_name],
                    "mask_path": mask_path,
                    "xml_path": xml_path
                })
        if skipped:
            print(f"Skipped {skipped} incomplete samples for split {self.split}.")
        return samples
    
    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return 0
    def _load_image(self, idx):
        if self.samples is None:
            raise ValueError("Samples not built yet.")
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.target_size is not None:
            img = img.resize(self.target_size, Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img

    def _load_label(self, idx):
        if self.samples is None:
            raise ValueError("Samples not built yet.")
        sample = self.samples[idx]
        return torch.tensor(sample["label_id"], dtype=torch.long)

    def _load_mask(self, idx):
        if self.samples is None:
            raise ValueError("Samples not built yet.")
        sample = self.samples[idx]
        mask = Image.open(sample["mask_path"])
        if self.target_size is not None:
            mask = mask.resize(self.target_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long() - 1
        return mask

    def _load_bbox(self, idx):
        if self.samples is None:
            raise ValueError("Samples not built yet.")
        sample = self.samples[idx]
        tree = ET.parse(sample["xml_path"])
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return None
        xmin = bndbox.find("xmin") 
        ymin = bndbox.find("ymin")
        xmax = bndbox.find("xmax")
        ymax = bndbox.find("ymax")
        xmin = float(xmin.text) if xmin is not None and xmin.text is not None else 0.0
        ymin = float(ymin.text) if ymin is not None and ymin.text is not None else 0.0
        xmax = float(xmax.text) if xmax is not None and xmax.text is not None else 0.0
        ymax = float(ymax.text) if ymax is not None and ymax.text is not None else 0.0
        if self.target_size is not None:
            size = root.find("size")
            if size is not None:
                width_node = size.find("width")
                height_node = size.find("height")
                original_w = float(width_node.text) if width_node is not None and width_node.text is not None else None
                original_h = float(height_node.text) if height_node is not None and height_node.text is not None else None
                if original_w and original_h:
                    scale_x = self.target_size[0] / original_w
                    scale_y = self.target_size[1] / original_h
                    xmin *= scale_x
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y
        w = xmax - xmin
        h = ymax - ymin
        if w < 0 or h < 0:
            return None
        x = 0.5 * (xmin + xmax)
        y = 0.5 * (ymin + ymax)
        if x < 0 or y < 0:
            return None
        return [x, y, w, h]
            
    def __getitem__(self, idx):
        '''output normalised item'''
        out = {}
        out['image'] = self._load_image(idx)
        out['label'] = self._load_label(idx)
        out['mask'] = self._load_mask(idx)
        if self.bbox:
            bbox = self._load_bbox(idx)
            if bbox is not None:
                out['bbox'] = torch.tensor(bbox, dtype=torch.float32)  
        return out
