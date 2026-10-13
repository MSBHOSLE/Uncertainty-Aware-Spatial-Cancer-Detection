"""
Segmentation and Cell Classification Metrics
Implements Multi-Class Dice Similarity Coefficient (DSC), Mean IoU,
and Instance-Level Nuclei Detection Precision, Recall, and F1 Score.
"""

import numpy as np
import torch


def compute_segmentation_metrics(preds, targets, num_classes=6):
    """
    Computes per-class and mean Dice and IoU metrics.
    
    Args:
        preds: (N, H, W) np.ndarray or torch.Tensor
        targets: (N, H, W) np.ndarray or torch.Tensor
        num_classes: int
    
    Returns:
        dict containing:
            - mean_dice: float
            - mean_iou: float
            - per_class_dice: dict
            - per_class_iou: dict
            - foreground_dice: float (classes 1..5)
            - foreground_iou: float (classes 1..5)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    per_class_dice = {}
    per_class_iou = {}

    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)

        intersection = np.sum(p_c & t_c)
        union = np.sum(p_c | t_c)
        total_pixels = np.sum(p_c) + np.sum(t_c)

        if total_pixels == 0:
            dice_c = 1.0
            iou_c = 1.0
        else:
            dice_c = (2.0 * intersection) / (total_pixels + 1e-8)
            iou_c = intersection / (union + 1e-8)

        per_class_dice[c] = float(dice_c)
        per_class_iou[c] = float(iou_c)

    mean_dice = float(np.mean(list(per_class_dice.values())))
    mean_iou = float(np.mean(list(per_class_iou.values())))

    # Foreground only (classes 1 to 5)
    fg_dices = [per_class_dice[c] for c in range(1, num_classes)]
    fg_ious = [per_class_iou[c] for c in range(1, num_classes)]
    foreground_dice = float(np.mean(fg_dices))
    foreground_iou = float(np.mean(fg_ious))

    return {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "foreground_dice": foreground_dice,
        "foreground_iou": foreground_iou,
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou
    }


def compute_instance_detection_metrics(gt_cells, pred_cells, dist_threshold=12.0):
    """
    Matches predicted cell centroids with ground truth cell centroids within dist_threshold.
    
    Returns:
        dict containing:
            - precision: float
            - recall: float
            - f1: float
            - classification_accuracy: float (for matched true positive cells)
            - true_positives: int
            - false_positives: int (hallucinated cells)
            - false_negatives: int (missed cells)
    """
    if len(gt_cells) == 0 and len(pred_cells) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if len(gt_cells) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": len(pred_cells), "fn": 0}
    if len(pred_cells) == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gt_cells)}

    gt_coords = np.array([[g["centroid_x"], g["centroid_y"]] for g in gt_cells])
    pred_coords = np.array([[p["centroid_x"], p["centroid_y"]] for p in pred_cells])
    gt_classes = np.array([g["class_id"] for g in gt_cells])
    pred_classes = np.array([p["class_id"] for p in pred_cells])

    # Distance matrix (M, N)
    dists = np.linalg.norm(pred_coords[:, None, :] - gt_coords[None, :, :], axis=2)

    matched_gt = set()
    matched_pred = set()
    correct_classes = 0

    # Greedy matching by minimum distance
    flat_indices = np.argsort(dists, axis=None)
    for idx in flat_indices:
        p_idx, g_idx = np.unravel_index(idx, dists.shape)
        if dists[p_idx, g_idx] > dist_threshold:
            break
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            if pred_classes[p_idx] == gt_classes[g_idx]:
                correct_classes += 1

    tp = len(matched_pred)
    fp = len(pred_cells) - tp
    fn = len(gt_cells) - tp

    precision = float(tp / max(1, len(pred_cells)))
    recall = float(tp / max(1, len(gt_cells)))
    f1 = float(2 * precision * recall / max(1e-8, precision + recall))
    cls_acc = float(correct_classes / max(1, tp))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_accuracy": cls_acc,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }
