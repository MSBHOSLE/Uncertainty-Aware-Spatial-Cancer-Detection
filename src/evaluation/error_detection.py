"""
Uncertainty-Guided Error Detection Evaluation Module
Measures whether uncertainty estimates reliably detect actual segmentation errors.
Calculates AUROC, AUPRC, and FPR95 for both pixel-level and object-level errors.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve


def compute_fpr95(y_true, y_scores):
    """
    Computes False Positive Rate (FPR) when True Positive Rate (TPR) is at 95%.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find smallest FPR where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return float(fpr[idx[0]])
    return 1.0


def evaluate_pixel_error_detection(pred_masks, gt_masks, uncertainty_maps, foreground_only=True):
    """
    Evaluates pixel-level error detection performance using uncertainty maps.
    
    Args:
        pred_masks: (N, H, W) np.ndarray
        gt_masks: (N, H, W) np.ndarray
        uncertainty_maps: (N, H, W) np.ndarray
        foreground_only: if True, focuses on nuclei regions
    
    Returns:
        dict containing:
            - auroc: float
            - auprc: float
            - fpr95: float
            - error_rate: float
    """
    errors = (pred_masks != gt_masks).astype(np.int32).reshape(-1)
    scores = uncertainty_maps.reshape(-1)

    if foreground_only:
        # Include pixels where either GT or Pred is foreground (nuclei)
        fg_mask = ((gt_masks > 0) | (pred_masks > 0)).reshape(-1)
        if np.sum(fg_mask) > 500:
            errors = errors[fg_mask]
            scores = scores[fg_mask]

    num_errors = np.sum(errors)
    total_pix = len(errors)

    if num_errors == 0 or num_errors == total_pix:
        return {"auroc": 0.5, "auprc": 0.0, "fpr95": 1.0, "error_rate": float(num_errors / max(1, total_pix))}

    # Subsample if large array for fast computation
    if total_pix > 200000:
        idx = np.random.choice(total_pix, size=200000, replace=False)
        errors = errors[idx]
        scores = scores[idx]

    try:
        auroc = float(roc_auc_score(errors, scores))
        prec, rec, _ = precision_recall_curve(errors, scores)
        auprc = float(auc(rec, prec))
        fpr95 = compute_fpr95(errors, scores)
    except Exception:
        auroc, auprc, fpr95 = 0.5, 0.0, 1.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr95": fpr95,
        "error_rate": float(num_errors / total_pix)
    }
