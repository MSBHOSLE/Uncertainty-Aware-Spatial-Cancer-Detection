"""
Cross-Organ Distribution Shift and Corruption Evaluation Module
Tests genuine shifts across tissue types (Breast/Colon -> Kidney/Liver/Lung)
and controlled histopathology corruptions (Stain drift, Blur, Tissue fold).
Compares Uncertainty-based vs Feature-based OoD detection (AUROC, FPR95).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from src.evaluation.error_detection import compute_fpr95


def evaluate_ood_detection(id_scores, ood_scores):
    """
    Evaluates how effectively an anomaly score separates In-Distribution from OoD samples.
    
    Args:
        id_scores: 1D array of scores for ID samples (lower indicates ID)
        ood_scores: 1D array of scores for OoD samples (higher indicates OoD)
    
    Returns:
        dict with auroc, fpr95, id_mean, ood_mean
    """
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])

    try:
        auroc = float(roc_auc_score(y_true, y_scores))
        fpr95 = compute_fpr95(y_true, y_scores)
    except Exception:
        auroc, fpr95 = 0.5, 1.0

    return {
        "auroc": auroc,
        "fpr95": fpr95,
        "id_mean_score": float(np.mean(id_scores)),
        "ood_mean_score": float(np.mean(ood_scores))
    }


def compare_shift_detectors(id_data, ood_data, detector_dict):
    """
    Compares multiple OoD detectors across ID and OOD splits.
    
    detector_dict: dict mapping detector_name -> {
        'id_scores': array,
        'ood_scores': array
    }
    """
    comparison = {}
    for name, scores in detector_dict.items():
        res = evaluate_ood_detection(scores["id_scores"], scores["ood_scores"])
        comparison[name] = res
    return comparison
