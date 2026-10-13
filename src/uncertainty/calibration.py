"""
Uncertainty Calibration Metrics and Reliability Diagrams
Implements Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Brier Score for multi-class histopathology segmentation.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_calibration_metrics(probs, labels, n_bins=15, foreground_only=True):
    """
    Computes ECE, MCE, and Brier Score.
    
    Args:
        probs: np.ndarray or torch.Tensor (N, C, H, W) or (Total_Pixels, C)
        labels: np.ndarray or torch.Tensor (N, H, W) or (Total_Pixels,)
        n_bins: number of confidence bins
        foreground_only: if True, evaluates calibration on non-background pixels
    
    Returns:
        dict containing:
            - ece: float (Expected Calibration Error)
            - mce: float (Maximum Calibration Error)
            - brier_score: float
            - bin_accuracies: list
            - bin_confidences: list
            - bin_counts: list
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Reshape if spatial tensors
    if probs.ndim == 4:
        # [N, C, H, W] -> [Total_Pixels, C]
        probs = np.transpose(probs, (0, 2, 3, 1)).reshape(-1, probs.shape[1])
        labels = labels.reshape(-1)

    if foreground_only:
        fg_mask = labels > 0
        if np.sum(fg_mask) > 100:
            probs = probs[fg_mask]
            labels = labels[fg_mask]

    # Predictions and confidences
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    # Brier Score: Mean squared error of probability vector
    # One-hot encode targets
    num_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    np.put_along_axis(one_hot, labels[:, None], 1.0, axis=1)
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Binning for ECE
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    mce = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        bin_counts.append(int(count))

        if count > 0:
            acc = float(np.mean(accuracies[in_bin]))
            conf = float(np.mean(confidences[in_bin]))
            bin_accuracies.append(acc)
            bin_confidences.append(conf)

            gap = abs(acc - conf)
            ece += (count / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2.0)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }
