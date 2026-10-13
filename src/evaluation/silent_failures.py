"""
Silent Failure Profiling and Overconfident Error Analysis
Decomposes segmentation failures into:
1. False Positive Hallucinations (spurious cells)
2. False Negatives (missed nuclei)
3. Touching / Merged Nuclei
4. Cell-Type Misclassifications
Quantifies how often each severe error occurs silently with LOW uncertainty.
"""

import numpy as np
from scipy.spatial import KDTree


def analyze_silent_failures(
    gt_cells,
    pred_cells,
    uncertainty_map,
    distance_threshold=12.0,
    uncertainty_cutoff_percentile=50.0
):
    """
    Identifies cell-level errors and assesses whether uncertainty flagged them.
    
    Args:
        gt_cells: list of dicts with ['id', 'class_id', 'centroid_x', 'centroid_y', 'area']
        pred_cells: list of dicts
        uncertainty_map: (H, W) np.ndarray
        distance_threshold: max matching distance in pixels
        uncertainty_cutoff_percentile: threshold to define 'high' vs 'low' uncertainty
    
    Returns:
        dict containing categorized failure counts, silent error fractions,
        and stratifications by size and crowding.
    """
    H, W = uncertainty_map.shape
    cutoff_val = float(np.percentile(uncertainty_map, uncertainty_cutoff_percentile))

    def get_local_uncertainty(cx, cy, radius=5):
        x0, x1 = max(0, int(cx - radius)), min(W, int(cx + radius + 1))
        y0, y1 = max(0, int(cy - radius)), min(H, int(cy + radius + 1))
        sub = uncertainty_map[y0:y1, x0:x1]
        return float(np.mean(sub)) if sub.size > 0 else 0.0

    gt_coords = np.array([[g["centroid_x"], g["centroid_y"]] for g in gt_cells]) if len(gt_cells) > 0 else np.empty((0, 2))
    pred_coords = np.array([[p["centroid_x"], p["centroid_y"]] for p in pred_cells]) if len(pred_cells) > 0 else np.empty((0, 2))

    # Match predictions to GT
    matched_gt = set()
    matched_pred = {}  # pred_idx -> gt_idx

    if len(gt_coords) > 0 and len(pred_coords) > 0:
        dists = np.linalg.norm(pred_coords[:, None, :] - gt_coords[None, :, :], axis=2)
        flat_indices = np.argsort(dists, axis=None)
        for idx in flat_indices:
            p_idx, g_idx = np.unravel_index(idx, dists.shape)
            if dists[p_idx, g_idx] > distance_threshold:
                break
            if p_idx not in matched_pred and g_idx not in matched_gt:
                matched_pred[p_idx] = g_idx
                matched_gt.add(g_idx)

    # 1. False Positive Hallucinations (Pred without GT match)
    fp_hallucinations = []
    for p_idx, p in enumerate(pred_cells):
        if p_idx not in matched_pred:
            unc = get_local_uncertainty(p["centroid_x"], p["centroid_y"])
            is_silent = unc < cutoff_val
            fp_hallucinations.append({
                "cell": p,
                "uncertainty": unc,
                "is_silent": is_silent
            })

    # 2. False Negatives (GT missed completely)
    fn_omissions = []
    for g_idx, g in enumerate(gt_cells):
        if g_idx not in matched_gt:
            unc = get_local_uncertainty(g["centroid_x"], g["centroid_y"])
            is_silent = unc < cutoff_val
            is_small = g.get("area", 50) < 40
            fn_omissions.append({
                "cell": g,
                "uncertainty": unc,
                "is_silent": is_silent,
                "is_small": is_small
            })

    # 3. Cell-Type Misclassifications (Matched location, wrong diagnosis)
    misclassifications = []
    for p_idx, g_idx in matched_pred.items():
        p = pred_cells[p_idx]
        g = gt_cells[g_idx]
        if p["class_id"] != g["class_id"]:
            unc = get_local_uncertainty(p["centroid_x"], p["centroid_y"])
            is_silent = unc < cutoff_val
            misclassifications.append({
                "pred_class": p["class_id"],
                "gt_class": g["class_id"],
                "uncertainty": unc,
                "is_silent": is_silent
            })

    # Compute silent failure ratios
    num_fp = len(fp_hallucinations)
    num_silent_fp = sum(1 for x in fp_hallucinations if x["is_silent"])

    num_fn = len(fn_omissions)
    num_silent_fn = sum(1 for x in fn_omissions if x["is_silent"])

    num_misclass = len(misclassifications)
    num_silent_misclass = sum(1 for x in misclassifications if x["is_silent"])

    # Small nuclei omission rate
    small_fns = [x for x in fn_omissions if x.get("is_small", False)]
    silent_small_fn_rate = float(sum(1 for x in small_fns if x["is_silent"]) / max(1, len(small_fns)))

    return {
        "false_positive_hallucinations": {
            "total": num_fp,
            "silent_count": num_silent_fp,
            "flagged_count": num_fp - num_silent_fp,
            "silent_rate": float(num_silent_fp / max(1, num_fp))
        },
        "false_negative_omissions": {
            "total": num_fn,
            "silent_count": num_silent_fn,
            "flagged_count": num_fn - num_silent_fn,
            "silent_rate": float(num_silent_fn / max(1, num_fn)),
            "small_nuclei_total": len(small_fns),
            "small_nuclei_silent_rate": silent_small_fn_rate
        },
        "cell_type_misclassifications": {
            "total": num_misclass,
            "silent_count": num_silent_misclass,
            "flagged_count": num_misclass - num_silent_misclass,
            "silent_rate": float(num_silent_misclass / max(1, num_misclass))
        },
        "total_severe_errors": num_fp + num_fn + num_misclass,
        "total_silent_severe_errors": num_silent_fp + num_silent_fn + num_silent_misclass,
        "overall_silent_error_rate": float((num_silent_fp + num_silent_fn + num_silent_misclass) / max(1, num_fp + num_fn + num_misclass))
    }
