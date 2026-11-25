"""
Custom PyTorch Dataset for Tyre Defect Detection.
Handles loading images from CSV labels and applying transforms.
Includes GPU-optimized data loading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import config


def gpu_collate_fn(batch):
    """
    Default collate function that stacks data.
    Data will be moved to device by training loop.

    Args:
        batch: List of (image, label) tuples from dataset

    Returns:
        Tuple of (images, labels) on CPU (for pin_memory to work)
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])

    return images, labels



class TyreDefectDataset(Dataset):
    """
    Custom Dataset for Tyre Defect Detection.

    Reads CSV file with columns: filename, Bulge, Cracks, Non-defective
    Each row has exactly one column set to 1 (single-label classification).
    Loads grayscale images and converts to 3-channel RGB.
    """

    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str or Path): Path to CSV file
            image_dir (str or Path): Directory containing images
            transform (torchvision.transforms.Compose): Image transforms
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Read CSV
        self.df = pd.read_csv(self.csv_path)

        # Derive labels from one-hot columns
        self.labels = []
        for idx, row in self.df.iterrows():
            # Find which column is 1 (skip 'filename' column)
            label_cols = [col for col in config.LABEL_COLUMNS]
            for label_idx, col in enumerate(label_cols):
                if row[col] == 1:
                    self.labels.append(label_idx)
                    break

        assert len(self.labels) == len(self.df), "Label extraction failed"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Image tensor
            label (int): Class label (0, 1, or 2)
        """
        # Get filename and label
        filename = self.df.iloc[idx]["filename"]
        label = self.labels[idx]

        # Load image
        image_path = self.image_dir / filename

        try:
            # Load grayscale image
            image = Image.open(image_path).convert("L")  # Grayscale
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image {image_path}: {e}")

        # Convert grayscale to RGB by duplicating the channel
        image = np.array(image)  # Shape: (H, W)
        image_rgb = np.stack([image, image, image], axis=-1)  # Shape: (H, W, 3)
        image = Image.fromarray(image_rgb.astype(np.uint8))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_transforms(split="train", rotation_degrees=None):
    """
    Get appropriate transform pipeline for train/val/test splits.

    Args:
        split (str): "train", "val", or "test"
        rotation_degrees (int or None): Custom rotation degrees for train split

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if rotation_degrees is None:
        rotation_degrees = config.TRAIN_AUG_ROTATION

    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((config.TARGET_IMG_SIZE, config.TARGET_IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ColorJitter(
                brightness=config.TRAIN_AUG_BRIGHTNESS,
                contrast=config.TRAIN_AUG_CONTRAST
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD
            )
        ])
    else:  # val or test
        transform = transforms.Compose([
            transforms.Resize((config.TARGET_IMG_SIZE, config.TARGET_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD
            )
        ])

    return transform


def get_dataloaders(batch_size=None, rotation_degrees=None):
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size (int or None): Batch size for training
        rotation_degrees (int or None): Custom rotation degrees for augmentation

    Returns:
        dict: Dictionary with 'train', 'val', 'test' dataloaders
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Transforms
    train_transform = get_transforms("train", rotation_degrees=rotation_degrees)
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")

    # Datasets
    train_dataset = TyreDefectDataset(
        config.TRAIN_LABELS,
        config.TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = TyreDefectDataset(
        config.VALID_LABELS,
        config.VALID_DIR,
        transform=val_transform
    )

    test_dataset = TyreDefectDataset(
        config.TEST_LABELS,
        config.TEST_DIR,
        transform=test_transform
    )

    # DataLoaders (optimized for GPU)
    # Note: num_workers=0 on Windows (required for GPU training)
    # On Linux/Mac with GPU, can use num_workers=4 for parallel loading
    num_workers = 0  # Windows GPU fix
    # pin_memory=True for faster GPU transfer
    # collate_fn=gpu_collate_fn to move data to GPU immediately
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=gpu_collate_fn if torch.cuda.is_available() else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE_VAL_TEST,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=gpu_collate_fn if torch.cuda.is_available() else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_VAL_TEST,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=gpu_collate_fn if torch.cuda.is_available() else None
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset
    }


def compute_class_weights(train_loader):
    """
    Compute class weights from training set distribution.
    Handles class imbalance.

    Args:
        train_loader (DataLoader): Training dataloader

    Returns:
        torch.Tensor: Class weights tensor
    """
    # Collect all labels
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)

    # Compute class counts
    class_counts = np.bincount(all_labels, minlength=config.NUM_CLASSES)
    total = len(all_labels)

    # Compute weights (inverse frequency)
    class_weights = total / (config.NUM_CLASSES * (class_counts + 1))

    # Normalize to sum to NUM_CLASSES
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES

    print(f"\nClass Distribution in Training Set:")
    for idx, (name, count) in enumerate(zip(config.CLASS_NAMES, class_counts)):
        print(f"  {name}: {count} ({100*count/total:.2f}%)")

    print(f"\nComputed Class Weights:")
    for idx, (name, weight) in enumerate(zip(config.CLASS_NAMES, class_weights)):
        print(f"  {name}: {weight:.4f}")

    return torch.tensor(class_weights, dtype=torch.float32, device=config.DEVICE)
