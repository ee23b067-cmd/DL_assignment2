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

try:
    _BILINEAR = Image.Resampling.BILINEAR
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST

_SPLIT_ALIASES = {
    "train": "trainval",
    "trainval": "trainval",
    "val": "test",
    "valid": "test",
    "test": "test",
}


@dataclass(frozen=True)
class _PetSample:
    image_path: Path
    mask_path: Path
    label: int
    bbox: Optional[Tuple[float, float, float, float]] = None


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet samples for classification, localization, and segmentation."""

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform=None,
        bbox: bool = True,
        target_size: Optional[Tuple[int, int]] = (224, 224),
    ) -> None:
        self.root = self._resolve_root(Path(root).expanduser())
        self.split = split
        self._split_name = self._normalize_split(split)
        self.transform = transform
        self.bbox = bbox
        self.target_size = target_size

        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.xml_dir = self.annotations_dir / "xmls"
        self.mask_dir = self.annotations_dir / "trimaps"
        self.split_file = self.annotations_dir / f"{self._split_name}.txt"

        self._ensure_extracted()

        raw_rows = self._read_split_rows()
        self.class2idx, self.idx2class, self.classes = self._build_class_index(raw_rows)
        self.idxtoclass = self.idx2class
        self.num_classes = len(self.classes)
        self.samples = self._build_samples(raw_rows)

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        candidates = (root, root / "oxford-iiit-pet", root / "oxford_pet")
        for candidate in candidates:
            if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
                return candidate
            if (candidate / "images.tar.gz").is_file() and (candidate / "annotations.tar.gz").is_file():
                return candidate
        return root

    @staticmethod
    def _normalize_split(split: str) -> str:
        try:
            return _SPLIT_ALIASES[split]
        except KeyError as exc:
            valid = ", ".join(sorted(_SPLIT_ALIASES))
            raise ValueError(f"Unsupported split '{split}'. Expected one of: {valid}.") from exc

    @staticmethod
    def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
        destination = destination.resolve()
        for member in archive.getmembers():
            target_path = (destination / member.name).resolve()
            try:
                target_path.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe archive member path: {member.name}") from exc
        archive.extractall(destination)

    def _ensure_extracted(self) -> None:
        if self.images_dir.is_dir() and self.annotations_dir.is_dir():
            return

        archives = (
            (self.root / "images.tar.gz", self.images_dir),
            (self.root / "annotations.tar.gz", self.annotations_dir),
        )

        for archive_path, expected_dir in archives:
            if expected_dir.is_dir():
                continue
            if not archive_path.is_file():
                raise FileNotFoundError(
                    f"Missing dataset files. Expected {expected_dir} or archive {archive_path}."
                )
            with tarfile.open(archive_path, "r:gz") as archive:
                self._safe_extract(archive, self.root)

        if not self.images_dir.is_dir() or not self.annotations_dir.is_dir():
            raise FileNotFoundError(
                "Oxford-IIIT Pet dataset is incomplete. Expected images/ and annotations/ under the dataset root."
            )

    def _read_split_rows(self) -> list[tuple[str, int, int, str]]:
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        rows: list[tuple[str, int, int, str]] = []
        with self.split_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Malformed split file line {line_number} in {self.split_file}: expected 4 columns, got {len(parts)}."
                    )

                image_id, class_id, species_id, breed_id = parts
                rows.append((image_id, int(class_id), int(species_id), breed_id))

        if not rows:
            raise ValueError(f"Split file {self.split_file} does not contain any samples.")

        return rows

    def _build_class_index(self, rows: list[tuple[str, int, int, str]]) -> tuple[dict[str, int], dict[int, str], list[str]]:
        class_to_idx: dict[str, int] = {}
        for image_id, class_id, _, _ in rows:
            class_name = image_id.rsplit("_", 1)[0]
            label = class_id - 1

            existing = class_to_idx.get(class_name)
            if existing is not None and existing != label:
                raise ValueError(
                    f"Conflicting labels found for class '{class_name}': {existing} and {label}."
                )
            class_to_idx[class_name] = label

        ordered = sorted(class_to_idx.items(), key=lambda item: item[1])
        idx2class = {idx: name for name, idx in ordered}
        classes = [name for name, _ in ordered]
        return class_to_idx, idx2class, classes

    def _build_samples(self, rows: list[tuple[str, int, int, str]]) -> list[_PetSample]:
        samples: list[_PetSample] = []
        skipped = 0

        for image_id, class_id, _, _ in rows:
            image_path = self.images_dir / f"{image_id}.jpg"
            mask_path = self.mask_dir / f"{image_id}.png"

            if not image_path.is_file() or not mask_path.is_file():
                skipped += 1
                continue

            bbox = None
            if self.bbox:
                xml_path = self.xml_dir / f"{image_id}.xml"
                if not xml_path.is_file():
                    skipped += 1
                    continue
                bbox = self._read_bbox(xml_path)
                if bbox is None:
                    skipped += 1
                    continue

            samples.append(
                _PetSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=class_id - 1,
                    bbox=bbox,
                )
            )

        if skipped:
            print(f"Skipped {skipped} incomplete samples for split {self._split_name}.")

        if not samples:
            raise RuntimeError(f"No valid samples found for split {self._split_name}.")

        return samples

    def _read_bbox(self, xml_path: Path) -> Optional[Tuple[float, float, float, float]]:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            return None

        root = tree.getroot()
        object_node = root.find("object")
        if object_node is None:
            return None

        box_node = object_node.find("bndbox")
        if box_node is None:
            return None

        try:
            xmin = float(box_node.findtext("xmin", default="0"))
            ymin = float(box_node.findtext("ymin", default="0"))
            xmax = float(box_node.findtext("xmax", default="0"))
            ymax = float(box_node.findtext("ymax", default="0"))
        except (TypeError, ValueError):
            return None

        if self.target_size is not None:
            size_node = root.find("size")
            if size_node is not None:
                try:
                    original_w = float(size_node.findtext("width", default="0"))
                    original_h = float(size_node.findtext("height", default="0"))
                except (TypeError, ValueError):
                    original_w = 0.0
                    original_h = 0.0

                if original_w > 0 and original_h > 0:
                    scale_x = self.target_size[0] / original_w
                    scale_y = self.target_size[1] / original_h
                    xmin *= scale_x
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y

        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            return None

        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        if center_x < 0 or center_y < 0:
            return None

        return center_x, center_y, width, height

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: _PetSample) -> torch.Tensor:
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if self.target_size is not None:
                image = image.resize(self.target_size, _BILINEAR)
            image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).div(255.0)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor

    def _load_mask(self, sample: _PetSample) -> torch.Tensor:
        with Image.open(sample.mask_path) as mask:
            if self.target_size is not None:
                mask = mask.resize(self.target_size, _NEAREST)
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long() - 1

        return mask_tensor

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        output: dict[str, torch.Tensor] = {
            "image": self._load_image(sample),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "mask": self._load_mask(sample),
        }

        if sample.bbox is not None:
            output["bbox"] = torch.tensor(sample.bbox, dtype=torch.float32)

        return output
