"""
EcoSort AI - Dataset Loader & Augmentation Pipeline
Loads preprocessed images from processed_dataset/, applies data augmentation strictly
to training data, standardizes normalization, and manages PyTorch DataLoaders.
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from typing import Tuple, Dict, Optional, List
import numpy as np

DATASET_ROOT = "processed_dataset"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Returns train, validation, and test transforms.
    Augmentation is strictly restricted to training images.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


def load_class_weights(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Loads class weights calculated during preprocessing to handle class imbalance."""
    weights_path = os.path.join(DATASET_ROOT, "class_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            weights_dict = json.load(f)
        weight_list = [weights_dict.get(c, 1.0) for c in CLASSES]
    else:
        weight_list = [1.0] * len(CLASSES)
    return torch.tensor(weight_list, dtype=torch.float32, device=device)


def create_stratified_subset(dataset: datasets.ImageFolder, max_samples_per_class: int, seed: int = 42) -> Subset:
    """
    Creates a balanced stratified subset of an ImageFolder dataset.
    """
    np.random.seed(seed)
    class_indices = {i: [] for i in range(len(CLASSES))}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    selected_indices = []
    for cls_idx, indices in class_indices.items():
        if len(indices) <= max_samples_per_class:
            selected_indices.extend(indices)
        else:
            chosen = np.random.choice(indices, size=max_samples_per_class, replace=False)
            selected_indices.extend(chosen.tolist())

    return Subset(dataset, selected_indices)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    """
    tf = get_transforms()

    train_dir = os.path.join(DATASET_ROOT, "train")
    val_dir = os.path.join(DATASET_ROOT, "val")
    test_dir = os.path.join(DATASET_ROOT, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Processed dataset directory not found at: {DATASET_ROOT}")

    train_dataset = datasets.ImageFolder(train_dir, transform=tf["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform=tf["val"])
    test_dataset = datasets.ImageFolder(test_dir, transform=tf["test"])

    # Optional stratified subsetting for computational feasibility
    if max_train_samples is not None:
        per_class = max(1, max_train_samples // len(CLASSES))
        train_dataset = create_stratified_subset(train_dataset, per_class)

    if max_val_samples is not None:
        per_class = max(1, max_val_samples // len(CLASSES))
        val_dataset = create_stratified_subset(val_dataset, per_class)

    if max_test_samples is not None:
        per_class = max(1, max_test_samples // len(CLASSES))
        test_dataset = create_stratified_subset(test_dataset, per_class)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, CLASSES


if __name__ == "__main__":
    weights = load_class_weights()
    print("Class weights:", weights)
    print("Transforms defined successfully.")
