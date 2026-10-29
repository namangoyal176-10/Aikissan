"""
PyTorch Dataset and Data Augmentation Pipeline for PlantVillage Leaf Disease Classification.

Includes strong in-the-wild augmentations (rotation, lighting/contrast jitter,
blur, random crops) to bridge the domain gap between clean lab photos and real farm conditions.
Supports class-weight computation to mitigate class imbalance.
"""

from collections import Counter
import csv
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ALL_DISEASE_CLASSES, CLASS_TO_IDX, DATA_PROCESSED_DIR


def get_training_transforms() -> transforms.Compose:
    """
    Strong data augmentation for training.
    Simulates real-world farm conditions:
    - RandomResizedCrop: simulates varying distance of mobile camera from leaf
    - RandomHorizontalFlip & VerticalFlip: leaves appear in various orientations
    - RandomRotation (35 deg): handheld phones are often tilted
    - ColorJitter: simulates direct sunlight vs overcast skies vs shadows under canopy
    - GaussianBlur: simulates motion blur from wind or shaky hands
    - Normalization: standard ImageNet mean and std
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=35),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.25,
                hue=0.08,
            ),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """
    Deterministic transforms for Validation, Testing, and Inference.
    Resizes image to 256x256 and center crops to 224x224.
    """
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compute_class_weights(
    manifest_csv: Union[str, Path],
    num_classes: int = len(ALL_DISEASE_CLASSES),
) -> torch.Tensor:
    """
    Computes balanced class weights from the training manifest:
    weight[c] = total_samples / (num_classes * count[c])
    Helps prevent the model from ignoring minority classes (e.g., Potato Late Blight).
    """
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        # Fallback to uniform weights
        return torch.ones(num_classes, dtype=torch.float32)

    class_indices = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["class_idx"])
            class_indices.append(idx)

    if not class_indices:
        return torch.ones(num_classes, dtype=torch.float32)

    counts = Counter(class_indices)
    total_samples = len(class_indices)

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        count = counts.get(c, 0)
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            weights[c] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset loading images and disease labels from manifest CSV files.
    Supports on-the-fly sanity-checking (limiting samples per class).
    """

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Args:
            manifest_csv: Path to disease_train.csv, disease_val.csv, or disease_test.csv
            transform: PyTorch torchvision transform pipeline
            max_samples_per_class: If set, limits samples per class for quick sanity tests
        """
        self.manifest_path = Path(manifest_csv)
        self.transform = transform
        self.samples: List[Tuple[Path, int, str]] = []

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path}")

        per_class_counter: Dict[int, int] = {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_idx = int(row["class_idx"])
                class_name = row["class_name"]

                if max_samples_per_class is not None:
                    curr_count = per_class_counter.get(class_idx, 0)
                    if curr_count >= max_samples_per_class:
                        continue
                    per_class_counter[class_idx] = curr_count + 1

                # Resolve path: first try absolute, then relative to project root
                raw_path = Path(row["image_path"].strip('"'))
                if raw_path.exists():
                    img_path = raw_path
                else:
                    rel_path = Path(row.get("relative_path", "").strip('"'))
                    img_path = PROJECT_ROOT / rel_path

                self.samples.append((img_path, class_idx, class_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, class_idx, class_name = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error reading image {img_path}: {e}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, class_idx, class_name
