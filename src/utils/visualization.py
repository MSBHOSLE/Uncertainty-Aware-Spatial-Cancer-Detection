"""
Publication-Quality Histopathology and Spatial Uncertainty Visualization
Produces high-resolution figures for cell segmentation, uncertainty heatmaps,
TME cellular neighborhood graphs, and calibration/risk-coverage curves.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_COLORS = {
    0: np.array([245, 245, 245]), # Background
    1: np.array([220, 20, 60]),   # Neoplastic (Crimson)
    2: np.array([46, 139, 87]),   # Inflammatory (Sea Green)
    3: np.array([30, 144, 255]),  # Connective / Stroma (Dodger Blue)
    4: np.array([148, 0, 211]),   # Dead / Necrotic (Dark Violet)
    5: np.array([255, 140, 0]),   # Epithelial (Dark Orange)
}


def create_color_mask(semantic_mask):
    """Converts (H, W) class index mask to (H, W, 3) RGB image."""
    H, W = semantic_mask.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        rgb_mask[semantic_mask == c] = color
    return rgb_mask


def plot_pathology_uncertainty_panel(
    rgb_image,
    gt_mask,
    pred_mask,
    uncertainty_map,
    density_map=None,
    region_trust_map=None,
    save_path=None
):
    """
    Creates a multi-panel figure displaying:
    1. Original H&E Slide Patch
    2. Ground Truth Cell Masks
    3. Model Predicted Masks
    4. Spatial Uncertainty Heatmap
    5. Downstream Cell Density Map
    6. Region Trust Index (Biomarker Reliability Map)
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=200)

    # 1. H&E Image
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("H&E Pathology Patch", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Ground Truth Mask
    gt_rgb = create_color_mask(gt_mask)
    axes[0, 1].imshow(rgb_image)
    axes[0, 1].imshow(gt_rgb, alpha=0.55)
    axes[0, 1].set_title("Ground Truth Annotations", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Model Prediction
    pred_rgb = create_color_mask(pred_mask)
    axes[0, 2].imshow(rgb_image)
    axes[0, 2].imshow(pred_rgb, alpha=0.55)
    axes[0, 2].set_title("Segmentation & Cell Types", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # 4. Spatial Uncertainty Map
    im_u = axes[1, 0].imshow(uncertainty_map, cmap="inferno")
    axes[1, 0].set_title("Spatial Uncertainty Map (MC-Dropout)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im_u, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Downstream Cell Density
    if density_map is not None:
        im_d = axes[1, 1].imshow(density_map, cmap="viridis")
        axes[1, 1].set_title("Downstream Cell Density (KDE)", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")
        plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        axes[1, 1].axis("off")

    # 6. Region Trust Map
    if region_trust_map is not None:
        im_t = axes[1, 2].imshow(region_trust_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[1, 2].set_title("Region Trust Index (TME Validity)", fontsize=12, fontweight="bold")
        axes[1, 2].axis("off")
        plt.colorbar(im_t, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(calib_results, title="Reliability Diagram", save_path=None):
    """Plots ECE reliability diagram."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    conf = calib_results["bin_confidences"]
    acc = calib_results["bin_accuracies"]

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    ax.bar(conf, acc, width=0.05, alpha=0.6, color="royalblue", edgecolor="black", label=f"Accuracy (ECE={calib_results['ece']:.3f})")
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_risk_coverage(rc_results, title="Selective Prediction Risk-Coverage", save_path=None):
    """Plots selective prediction risk vs coverage curve."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    cov = rc_results["coverages"]
    risk = rc_results["risks"]

    ax.plot(cov, risk, "-o", color="darkred", lw=2, label=f"Uncertainty Referral (AURC={rc_results['aurc']:.3f})")
    if "oracle_aurc" in rc_results:
        ax.plot([1.0, 0.0], [risk[0], 0.0], "--", color="black", label=f"Oracle Optimal (AURC={rc_results['oracle_aurc']:.3f})")

    ax.set_xlabel("Coverage (Fraction of Autonomous Predictions)", fontsize=11)
    ax.set_ylabel("Residual Risk (Error Rate)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_review_simulation(review_results, title="Human-in-the-Loop Pathologist Review", save_path=None):
    """Plots errors caught vs pathologist review budget."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

    budgets = [review_results["budgets"][k]["budget_fraction"] * 100 for k in review_results["budgets"]]
    unc_caught = [review_results["budgets"][k]["uncertainty_guided"]["error_catch_rate"] * 100 for k in review_results["budgets"]]
    rand_caught = [review_results["budgets"][k]["random"]["error_catch_rate"] * 100 for k in review_results["budgets"]]
    oracle_caught = [review_results["budgets"][k]["oracle"]["error_catch_rate"] * 100 for k in review_results["budgets"]]

    ax.plot(budgets, unc_caught, "-o", color="crimson", lw=2.5, label="Uncertainty-Guided Review")
    ax.plot(budgets, rand_caught, "--s", color="gray", lw=2, label="Random Inspection Baseline")
    ax.plot(budgets, oracle_caught, ":^", color="darkgreen", lw=1.5, label="Oracle Theoretical Maximum")

    ax.set_xlabel("Pathologist Review Budget (% of Patches)", fontsize=11)
    ax.set_ylabel("Errors Discovered & Corrected (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
