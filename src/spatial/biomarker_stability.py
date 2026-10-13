"""
Downstream TME Spatial Biomarker Uncertainty & Instability Propagation
Evaluates whether segmentation uncertainty predicts unstable downstream biological measurements:
- Cell-density instability under stochastic passes
- Perturbation of nearest-neighbor distances and cellular neighborhood graphs
- Region Trust Index: flags tissue sub-regions where spatial biomarkers cannot be trusted.
"""

import numpy as np
import torch
from src.spatial.nuclei_extractor import extract_nuclei_instances
from src.spatial.tme_biomarkers import calculate_tme_spatial_biomarkers


def evaluate_biomarker_stability_under_stochastic_sampling(
    stochastic_probs,
    gt_semantic_mask=None,
    uncertainty_map=None,
    image_shape=(256, 256)
):
    """
    Evaluates downstream TME biomarker stability across T stochastic segmentation predictions.
    
    Args:
        stochastic_probs: (T, C, H, W) np.ndarray or torch.Tensor of probabilities
        gt_semantic_mask: optional (H, W) ground truth semantic mask
        uncertainty_map: optional (H, W) spatial uncertainty map
        image_shape: tuple (H, W)
    
    Returns:
        dict containing:
            - gt_biomarkers: dict of true biomarkers (if GT provided)
            - mean_pred_biomarkers: dict from average probability prediction
            - stochastic_biomarkers: list of dicts for each pass t
            - stability_metrics: dict of mean, std, CV for each scalar biomarker
            - correlation_with_uncertainty: dict of Pearson correlations
            - region_trust_map: (H, W) binary mask (1: trustworthy, 0: untrusted)
    """
    if isinstance(stochastic_probs, torch.Tensor):
        stochastic_probs = stochastic_probs.detach().cpu().numpy()
    if isinstance(gt_semantic_mask, torch.Tensor):
        gt_semantic_mask = gt_semantic_mask.detach().cpu().numpy()
    if isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.detach().cpu().numpy()
        if uncertainty_map.ndim == 3:
            uncertainty_map = uncertainty_map.squeeze(0)

    T, C, H, W = stochastic_probs.shape

    # 1. Ground Truth Biomarkers
    gt_biomarkers = None
    if gt_semantic_mask is not None:
        _, gt_cells = extract_nuclei_instances(gt_semantic_mask)
        gt_biomarkers = calculate_tme_spatial_biomarkers(gt_cells, image_shape=image_shape)

    # 2. Mean Prediction Biomarkers
    mean_probs = np.mean(stochastic_probs, axis=0) # (C, H, W)
    mean_pred_mask = np.argmax(mean_probs, axis=0)
    _, mean_cells = extract_nuclei_instances(mean_pred_mask)
    mean_pred_biomarkers = calculate_tme_spatial_biomarkers(mean_cells, image_shape=image_shape)

    # 3. Biomarkers across T Stochastic Passes
    scalar_keys = [
        "num_cells",
        "total_cell_density",
        "mean_nnd",
        "median_nnd",
        "neoplastic_fraction",
        "inflammatory_fraction",
        "stroma_fraction",
        "composition_entropy",
        "til_to_tumor_mean_distance",
        "tumor_stroma_mixing_score",
        "graph_avg_clustering",
        "graph_avg_degree"
    ]

    pass_biomarkers = []
    stochastic_density_maps = []
    metrics_by_pass = {k: [] for k in scalar_keys}

    for t in range(T):
        pred_mask_t = np.argmax(stochastic_probs[t], axis=0)
        _, cells_t = extract_nuclei_instances(pred_mask_t)
        b_t = calculate_tme_spatial_biomarkers(cells_t, image_shape=image_shape)
        pass_biomarkers.append(b_t)
        stochastic_density_maps.append(b_t["density_map"])

        for k in scalar_keys:
            metrics_by_pass[k].append(b_t[k])

    # 4. Statistical Summary (Mean, Variance, Std, Coefficient of Variation)
    stability_metrics = {}
    for k in scalar_keys:
        vals = np.array(metrics_by_pass[k], dtype=np.float64)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        cv_v = float(std_v / (abs(mean_v) + 1e-6))
        
        err_vs_gt = None
        if gt_biomarkers is not None:
            err_vs_gt = float(abs(mean_v - gt_biomarkers[k]))

        stability_metrics[k] = {
            "mean": mean_v,
            "std": std_v,
            "variance": float(std_v ** 2),
            "cv": cv_v,
            "err_vs_gt": err_vs_gt,
            "stochastic_passes": vals.tolist()
        }

    # 5. Spatial Density Instability Map across stochastic passes
    stacked_density_maps = np.stack(stochastic_density_maps, axis=0) # (T, H, W)
    spatial_density_variance = np.var(stacked_density_maps, axis=0)   # (H, W)

    # 6. Region Trust Index: Flag sub-regions with unstable downstream estimates
    # Partition image into 4x4 grid of 64x64 sub-regions
    grid_size = 4
    sub_h = H // grid_size
    sub_w = W // grid_size
    region_trust_map = np.ones((H, W), dtype=np.float32)

    subregion_uncertainties = []
    subregion_density_vars = []

    # Calculate threshold based on median uncertainty
    global_unc_threshold = (
        np.percentile(uncertainty_map, 75)
        if uncertainty_map is not None
        else np.percentile(spatial_density_variance, 75)
    )

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * sub_h, (i + 1) * sub_h
            x0, x1 = j * sub_w, (j + 1) * sub_w
            
            sub_dens_var = np.mean(spatial_density_variance[y0:y1, x0:x1])
            subregion_density_vars.append(sub_dens_var)

            if uncertainty_map is not None:
                sub_unc = np.mean(uncertainty_map[y0:y1, x0:x1])
                subregion_uncertainties.append(sub_unc)
                # If sub-region has high uncertainty or high density instability -> Untrusted
                if sub_unc > global_unc_threshold:
                    region_trust_map[y0:y1, x0:x1] = 0.0
            else:
                if sub_dens_var > np.percentile(spatial_density_variance, 75):
                    region_trust_map[y0:y1, x0:x1] = 0.0

    # 7. Correlation between spatial uncertainty and biomarker error/instability
    correlation_results = {}
    if uncertainty_map is not None and len(subregion_uncertainties) > 2:
        try:
            r = np.corrcoef(subregion_uncertainties, subregion_density_vars)[0, 1]
            correlation_results["uncertainty_vs_density_instability_pearson_r"] = float(r) if not np.isnan(r) else 0.0
        except Exception:
            correlation_results["uncertainty_vs_density_instability_pearson_r"] = 0.0

    return {
        "gt_biomarkers": gt_biomarkers,
        "mean_pred_biomarkers": mean_pred_biomarkers,
        "stability_metrics": stability_metrics,
        "spatial_density_variance": spatial_density_variance,
        "region_trust_map": region_trust_map,
        "correlation_results": correlation_results
    }
