"""
Deep Ensemble Architecture for Histopathology Nuclei Segmentation
Coordinates independently initialized and trained models for epistemic uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.unet_dropout import UncertaintyAttentionUNet


class DeepEnsemble(nn.Module):
    """
    Ensemble container holding M distinct neural network models.
    """
    def __init__(self, models):
        super(DeepEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    @classmethod
    def create(cls, num_models=3, in_channels=3, num_classes=6, dropout_p=0.0):
        """Creates an ensemble of independently initialized U-Net models."""
        models = [
            UncertaintyAttentionUNet(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout_p=dropout_p
            )
            for _ in range(num_models)
        ]
        return cls(models)

    def forward(self, x):
        """
        Returns stacked logits from all M ensemble members:
        Shape: [M, Batch, Classes, Height, Width]
        """
        all_logits = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                all_logits.append(logits)
        return torch.stack(all_logits, dim=0)

    def predict_with_uncertainty(self, x):
        """
        Computes ensemble mean prediction and predictive variance/entropy.
        """
        stacked_logits = self.forward(x)  # [M, B, C, H, W]
        stacked_probs = F.softmax(stacked_logits, dim=2)  # [M, B, C, H, W]

        mean_probs = torch.mean(stacked_probs, dim=0)  # [B, C, H, W]
        variance = torch.var(stacked_probs, dim=0)     # [B, C, H, W]
        
        # Total variance across foreground classes (1 to C-1)
        foreground_variance = torch.sum(variance[:, 1:, :, :], dim=1, keepdim=True)

        # Jensen-Shannon Divergence / Ensemble Disagreement
        # JSD = H(mean_p) - mean(H(p_m))
        eps = 1e-8
        mean_entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1, keepdim=True)
        member_entropies = -torch.sum(stacked_probs * torch.log(stacked_probs + eps), dim=2)  # [M, B, H, W]
        avg_member_entropy = torch.mean(member_entropies, dim=0).unsqueeze(1)  # [B, 1, H, W]
        disagreement = torch.clamp(mean_entropy - avg_member_entropy, min=0.0)

        return {
            "mean_probs": mean_probs,
            "variance": variance,
            "foreground_variance": foreground_variance,
            "disagreement": disagreement,
            "stacked_probs": stacked_probs
        }
