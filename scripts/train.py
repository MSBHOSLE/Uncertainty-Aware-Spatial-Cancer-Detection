"""
Model Training Script for Uncertainty-Aware Attention U-Net and Deep Ensembles
Trains networks using a combined Multi-Class Cross-Entropy and Soft-Dice Loss
with Spatial Dropout active during training.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pannuke_dataset import PanNukeDataset, load_dataset_split
from src.models.unet_dropout import UncertaintyAttentionUNet


class CombinedCrossEntropyDiceLoss(nn.Module):
    """
    Combined Cross-Entropy and Multi-Class Soft-Dice Loss.
    Addresses severe foreground-background pixel imbalance in histopathology.
    """
    def __init__(self, num_classes=6, dice_weight=0.5, ce_weight=1.0):
        super(CombinedCrossEntropyDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        # 1. Standard Cross-Entropy
        ce_loss = F.cross_entropy(logits, targets)

        # 2. Soft Dice Loss
        probs = F.softmax(logits, dim=1)
        # Convert targets to one-hot: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute dice for foreground classes 1 to C-1
        probs_fg = probs[:, 1:, :, :]
        targets_fg = targets_one_hot[:, 1:, :, :]

        dims = (0, 2, 3)
        intersection = torch.sum(probs_fg * targets_fg, dim=dims)
        cardinality = torch.sum(probs_fg + targets_fg, dim=dims)

        dice_score = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        dice_loss = 1.0 - torch.mean(dice_score)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def train_single_model(
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=5e-4,
    device="cpu",
    save_path="experiments/checkpoints/best_model.pt"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CombinedCrossEntropyDiceLoss(num_classes=6)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Training on device: {device} | Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_fg_correct = 0
        val_fg_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * len(images)

                preds = torch.argmax(logits, dim=1)
                fg_mask = masks > 0
                val_fg_correct += (preds[fg_mask] == masks[fg_mask]).sum().item()
                val_fg_total += fg_mask.sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = (val_fg_correct / max(1, val_fg_total)) * 100

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            saved_str = "[Checkpoint Saved]"
        else:
            saved_str = ""

        if epoch % 3 == 0 or epoch == epochs or saved_str != "":
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val FG Acc: {val_acc:.1f}% {saved_str}")

    print(f"Finished training. Best model saved to: {save_path}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Uncertainty-Aware U-Net Models")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train_ensemble", action="store_true", help="Train ensemble members")
    parser.add_argument("--num_ensemble", type=int, default=3)
    args = parser.parse_args()

    # Load data
    train_data = load_dataset_split("data/pannuke/train/pannuke_train.npz")
    val_data = load_dataset_split("data/pannuke/val/pannuke_val.npz")

    train_ds = PanNukeDataset(
        train_data["images"],
        train_data["masks"],
        instance_masks=train_data["instance_masks"],
        organ_types=train_data["organ_types"],
        transform=True
    )
    val_ds = PanNukeDataset(
        val_data["images"],
        val_data["masks"],
        instance_masks=val_data["instance_masks"],
        organ_types=val_data["organ_types"],
        transform=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"=== Starting Training Pipeline (Device: {device}) ===")

    # 1. Train MC-Dropout U-Net Model (Primary Backbone)
    print("\n--- Training UncertaintyAttentionUNet (MC-Dropout Backbone) ---")
    torch.manual_seed(42)
    mc_model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3).to(device)
    train_single_model(
        mc_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path="experiments/checkpoints/unet_mcdropout_best.pt"
    )

    # 2. Train Deep Ensemble Members if requested
    if args.train_ensemble:
        print(f"\n--- Training Deep Ensemble ({args.num_ensemble} Members) ---")
        for m in range(args.num_ensemble):
            seed = 100 + m * 42
            torch.manual_seed(seed)
            print(f"\nTraining Ensemble Member {m+1}/{args.num_ensemble} (Seed={seed})...")
            ens_model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.0).to(device)
            train_single_model(
                ens_model,
                train_loader,
                val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                save_path=f"experiments/checkpoints/ensemble_member_{m}.pt"
            )

    print("\n=== All Training Completed Successfully! ===")


if __name__ == "__main__":
    main()
