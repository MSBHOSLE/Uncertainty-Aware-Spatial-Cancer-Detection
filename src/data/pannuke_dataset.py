"""
PanNuke-compatible Dataset Loader and PyTorch Data Pipeline
Supports both real PanNuke numpy partitions and synthetic benchmark partitions
with organ stratification, held-out cross-tissue splits, and histopathology corruptions.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class PanNukeDataset(Dataset):
    """
    PanNuke Histopathology Dataset.
    Labels:
      0: Background
      1: Neoplastic
      2: Inflammatory
      3: Connective/Stroma
      4: Dead/Necrotic
      5: Non-neoplastic Epithelial
    """

    def __init__(self, images, masks, instance_masks=None, organ_types=None, transform=None):
        """
        Args:
            images: np.ndarray (N, H, W, 3) uint8
            masks: np.ndarray (N, H, W) int64 (0 to 5)
            instance_masks: np.ndarray (N, H, W) int32 (instance IDs)
            organ_types: list of str (length N)
            transform: boolean or callable for augmentations
        """
        self.images = images
        self.masks = masks
        self.instance_masks = instance_masks
        self.organ_types = organ_types if organ_types is not None else ["unknown"] * len(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3) uint8
        mask = self.masks[idx]  # (H, W) int64
        organ = self.organ_types[idx]

        # Convert to torch tensor: [C, H, W] normalized to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        if self.transform:
            # Random horizontal & vertical flips
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)

        inst_tensor = (
            torch.from_numpy(self.instance_masks[idx]).int()
            if self.instance_masks is not None
            else torch.zeros_like(mask_tensor, dtype=torch.int32)
        )

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "instance": inst_tensor,
            "organ": organ,
            "idx": idx
        }


def save_dataset_split(filepath, images, masks, instance_masks, organ_types):
    """Saves a split into a compressed npz file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(
        filepath,
        images=images,
        masks=masks,
        instance_masks=instance_masks,
        organ_types=np.array(organ_types)
    )


def load_dataset_split(filepath):
    """Loads a split from a compressed npz file."""
    data = np.load(filepath, allow_pickle=True)
    return {
        "images": data["images"],
        "masks": data["masks"],
        "instance_masks": data["instance_masks"],
        "organ_types": list(data["organ_types"])
    }
