"""
Baseline Confidence and Entropy Metrics
Calculates Maximum Softmax Probability (MSP) and single-pass Predictive Entropy.
"""

import torch
import torch.nn.functional as F


def compute_baseline_uncertainty(model, image_tensor):
    """
    Computes standard deterministic single-pass uncertainty baselines:
    - Maximum Softmax Probability (MSP): 1.0 - max_c p_c (Inverted confidence)
    - Predictive Entropy: -sum_c p_c * log(p_c)
    
    Args:
        model: PyTorch segmentation model
        image_tensor: (B, C, H, W)
    
    Returns:
        dict containing:
            - probabilities: (B, C, H, W)
            - predicted_classes: (B, H, W)
            - max_softmax_prob: (B, 1, H, W) (max confidence)
            - msp_uncertainty: (B, 1, H, W) (1 - confidence)
            - predictive_entropy: (B, 1, H, W)
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)

    max_prob, pred_classes = torch.max(probs, dim=1, keepdim=True)
    msp_uncertainty = 1.0 - max_prob

    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1, keepdim=True)

    return {
        "probabilities": probs,
        "predicted_classes": pred_classes.squeeze(1),
        "max_softmax_prob": max_prob,
        "msp_uncertainty": msp_uncertainty,
        "predictive_entropy": entropy
    }
