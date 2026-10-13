"""
Downstream Tumor Microenvironment (TME) Spatial Biomarker Analysis (Addition 1)
Directly fulfills Research Addition 1 (Highest Research Value):
1. Calculates cellular spatial properties:
   - Cell density (global and KDE spatial map)
   - Nearest-neighbor distances (NND) and immune-to-tumor proximity
   - Cell-type proportions (Neoplastic, Inflammatory/TIL, Stroma, Epithelial)
   - Cellular neighborhood graph (Delaunay network clustering and degree)
   - Tumor-Stroma spatial mixing score
2. Propagates stochastic segmentation passes to quantify biomarker variance and instability.
3. Tests whether high segmentation uncertainty predicts unstable downstream biological estimates.
4. Generates Region Trust Index to flag tissue sub-regions where biomarkers are untrustworthy.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import json
import argparse
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pannuke_dataset import load_dataset_split
from src.models.unet_dropout import UncertaintyAttentionUNet
from src.uncertainty.mc_dropout import run_mc_dropout_inference
from src.spatial.biomarker_stability import evaluate_biomarker_stability_under_stochastic_sampling
from src.utils.visualization import plot_pathology_uncertainty_panel


def run_tme_spatial_analysis(device="cpu", num_mc_samples=15, num_patches=25):
    print("=== Running TME Spatial Biomarker Reliability Analysis (Addition 1) ===")
    
    # 1. Load Data
    test_data = load_dataset_split("data/pannuke/test_id/pannuke_test_id.npz")
    images_np = test_data["images"]
    masks_np = test_data["masks"]
    organs_list = test_data["organ_types"]

    # 2. Load Model
    mc_model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3).to(device)
    mc_ckpt = "experiments/checkpoints/unet_mcdropout_best.pt"
    if os.path.exists(mc_ckpt):
        mc_model.load_state_dict(torch.load(mc_ckpt, map_location=device))
        print(f"Loaded MC-Dropout model from: {mc_ckpt}")
    mc_model.eval()

    num_eval = min(num_patches, len(images_np))
    all_patch_uncertainties = []
    density_cv_list = []
    nnd_cv_list = []
    til_dist_cv_list = []
    clustering_cv_list = []
    mixing_cv_list = []

    density_abs_err_list = []
    nnd_abs_err_list = []
    til_abs_err_list = []

    patch_results = []

    print(f"Evaluating {num_eval} pathology patches with {num_mc_samples} MC stochastic passes each...")

    for i in range(num_eval):
        img_np = images_np[i]
        gt_mask = masks_np[i]
        organ = organs_list[i]

        inp = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        inp = inp.to(device)

        # Run MC-Dropout
        mc_out = run_mc_dropout_inference(mc_model, inp, num_samples=num_mc_samples, return_samples=True)

        stoch_probs = mc_out["stochastic_samples"].squeeze(1).cpu().numpy()  # (T, C, H, W)
        mean_unc_map = mc_out["foreground_variance"].squeeze().cpu().numpy()  # (H, W)
        patch_mean_unc = float(np.mean(mean_unc_map))
        all_patch_uncertainties.append(patch_mean_unc)

        # Evaluate TME stability across passes
        tme_eval = evaluate_biomarker_stability_under_stochastic_sampling(
            stoch_probs,
            gt_semantic_mask=gt_mask,
            uncertainty_map=mean_unc_map,
            image_shape=(256, 256)
        )

        stab = tme_eval["stability_metrics"]
        dens_cv = stab["total_cell_density"]["cv"]
        nnd_cv = stab["mean_nnd"]["cv"]
        til_cv = stab["til_to_tumor_mean_distance"]["cv"]
        clust_cv = stab["graph_avg_clustering"]["cv"]
        mix_cv = stab["tumor_stroma_mixing_score"]["cv"]

        density_cv_list.append(dens_cv)
        nnd_cv_list.append(nnd_cv)
        til_dist_cv_list.append(til_cv)
        clustering_cv_list.append(clust_cv)
        mixing_cv_list.append(mix_cv)

        if stab["total_cell_density"]["err_vs_gt"] is not None:
            density_abs_err_list.append(stab["total_cell_density"]["err_vs_gt"])
            nnd_abs_err_list.append(stab["mean_nnd"]["err_vs_gt"])
            til_abs_err_list.append(stab["til_to_tumor_mean_distance"]["err_vs_gt"])

        patch_results.append({
            "patch_index": i,
            "organ": organ,
            "mean_uncertainty": patch_mean_unc,
            "density_cv": dens_cv,
            "density_error": stab["total_cell_density"]["err_vs_gt"],
            "mean_nnd_cv": nnd_cv,
            "mean_nnd_error": stab["mean_nnd"]["err_vs_gt"],
            "clustering_cv": clust_cv,
            "mixing_cv": mix_cv,
            "region_trust_fraction": float(np.mean(tme_eval["region_trust_map"]))
        })

        # Save exemplary visual panel for first 3 patches
        if i < 3:
            pred_mask = np.argmax(mc_out["mean_prediction"].squeeze(0).cpu().numpy(), axis=0)
            dens_map = tme_eval["mean_pred_biomarkers"]["density_map"]
            trust_map = tme_eval["region_trust_map"]
            vis_path = f"reports/figures/tme_inspection_panel_patch_{i}_{organ}.png"
            plot_pathology_uncertainty_panel(
                img_np,
                gt_mask,
                pred_mask,
                mean_unc_map,
                density_map=dens_map,
                region_trust_map=trust_map,
                save_path=vis_path
            )

    # 3. Correlation Analysis (Hypothesis Testing: Does uncertainty predict biomarker instability?)
    r_density, p_density = pearsonr(all_patch_uncertainties, density_cv_list)
    r_nnd, p_nnd = pearsonr(all_patch_uncertainties, nnd_cv_list)
    r_clust, p_clust = pearsonr(all_patch_uncertainties, clustering_cv_list)
    r_mix, p_mix = pearsonr(all_patch_uncertainties, mixing_cv_list)

    r_density_err, _ = pearsonr(all_patch_uncertainties, density_abs_err_list) if len(density_abs_err_list) > 2 else (0.0, 1.0)
    r_nnd_err, _ = pearsonr(all_patch_uncertainties, nnd_abs_err_list) if len(nnd_abs_err_list) > 2 else (0.0, 1.0)

    summary_correlations = {
        "uncertainty_vs_density_instability": {
            "pearson_r": float(r_density),
            "p_value": float(p_density)
        },
        "uncertainty_vs_nnd_instability": {
            "pearson_r": float(r_nnd),
            "p_value": float(p_nnd)
        },
        "uncertainty_vs_graph_clustering_instability": {
            "pearson_r": float(r_clust),
            "p_value": float(p_clust)
        },
        "uncertainty_vs_mixing_score_instability": {
            "pearson_r": float(r_mix),
            "p_value": float(p_mix)
        },
        "uncertainty_vs_density_ground_truth_error": {
            "pearson_r": float(r_density_err)
        },
        "uncertainty_vs_nnd_ground_truth_error": {
            "pearson_r": float(r_nnd_err)
        }
    }

    # 4. Generate Publication-Quality Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    # Subplot A: Uncertainty vs Cell Density CV
    axes[0].scatter(all_patch_uncertainties, density_cv_list, c="crimson", alpha=0.75, edgecolors="k", s=60)
    # Fit trendline
    z_d = np.polyfit(all_patch_uncertainties, density_cv_list, 1)
    p_d_line = np.poly1d(z_d)
    x_vals = np.linspace(min(all_patch_uncertainties), max(all_patch_uncertainties), 100)
    axes[0].plot(x_vals, p_d_line(x_vals), "--", color="darkred", lw=2, label=f"Fit (r={r_density:.3f}, p={p_density:.4f})")
    axes[0].set_xlabel("Mean Spatial Segmentation Uncertainty", fontsize=11)
    axes[0].set_ylabel("Downstream Cell Density Instability (CV)", fontsize=11)
    axes[0].set_title("Uncertainty vs Cell Density Biomarker Instability", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left")

    # Subplot B: Uncertainty vs Delaunay Graph Clustering CV
    axes[1].scatter(all_patch_uncertainties, clustering_cv_list, c="royalblue", alpha=0.75, edgecolors="k", s=60)
    z_c = np.polyfit(all_patch_uncertainties, clustering_cv_list, 1)
    p_c_line = np.poly1d(z_c)
    axes[1].plot(x_vals, p_c_line(x_vals), "--", color="navy", lw=2, label=f"Fit (r={r_clust:.3f}, p={p_clust:.4f})")
    axes[1].set_xlabel("Mean Spatial Segmentation Uncertainty", fontsize=11)
    axes[1].set_ylabel("Neighborhood Graph Clustering Instability (CV)", fontsize=11)
    axes[1].set_title("Uncertainty vs Cellular Spatial Graph Instability", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    scatter_save_path = "reports/figures/tme_biomarker_stability_correlations.png"
    os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
    plt.savefig(scatter_save_path, bbox_inches="tight")
    plt.close()

    print("\n" + "="*80)
    print("TME SPATIAL BIOMARKER STABILITY RESULTS (Addition 1)")
    print("="*80)
    print(f"Uncertainty vs Cell Density Instability (CV):   r = {r_density:.3f} (p = {p_density:.4e})")
    print(f"Uncertainty vs Delaunay Graph Clustering (CV): r = {r_clust:.3f} (p = {p_clust:.4e})")
    print(f"Uncertainty vs Nearest-Neighbor Dist (CV):     r = {r_nnd:.3f} (p = {p_nnd:.4e})")
    print(f"Uncertainty vs Tumor-Stroma Mixing (CV):       r = {r_mix:.3f} (p = {p_mix:.4e})")
    print(f"Uncertainty vs True Cell Density Error:        r = {r_density_err:.3f}")
    print("="*80)

    output_data = {
        "summary_correlations": summary_correlations,
        "patch_results": patch_results
    }
    with open("reports/tme_spatial_stability_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    return output_data


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_tme_spatial_analysis(device=device, num_mc_samples=15, num_patches=30)
