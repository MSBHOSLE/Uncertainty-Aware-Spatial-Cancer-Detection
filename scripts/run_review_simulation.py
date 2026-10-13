"""
Human-in-the-Loop Active Pathologist Review Experiment (Addition 4)
Simulates clinical pathologist review across fixed inspection budgets:
[5%, 10%, 20%, 30%, 50%] of pathology slide patches.
Compares:
1. Random Review Strategy
2. Uncertainty-Guided Review Strategy (ranked by MC-Dropout uncertainty)
3. Heuristic Stain Anomaly Strategy
4. Oracle Upper Bound

Evaluates:
- Cumulative genuine segmentation errors discovered and resolved
- Severe cell failure (missed neoplastic cells and hallucinations) reduction
- Recovery of downstream TME spatial biomarkers (cell density MAE reduction)
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pannuke_dataset import load_dataset_split
from src.models.unet_dropout import UncertaintyAttentionUNet
from src.uncertainty.mc_dropout import run_mc_dropout_inference
from src.human_in_the_loop.review_simulation import simulate_pathologist_review
from src.utils.visualization import plot_review_simulation


def run_pathologist_review_simulation(device="cpu", num_mc=10):
    print("=== Running Human-in-the-Loop Pathologist Review Simulation (Addition 4) ===")

    # 1. Load In-Distribution Test Cohort
    test_data = load_dataset_split("data/pannuke/test_id/pannuke_test_id.npz")
    images_np = test_data["images"]
    masks_np = test_data["masks"]
    N = len(images_np)

    # 2. Load Model
    model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3).to(device)
    ckpt = "experiments/checkpoints/unet_mcdropout_best.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded model from: {ckpt}")
    model.eval()

    # 3. Extract Predictions and Uncertainty
    print(f"Generating inference predictions and uncertainty maps across {N} patches...")
    tensors = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0

    preds_list = []
    unc_list = []

    for i in range(N):
        inp = tensors[i:i+1].to(device)
        mc_out = run_mc_dropout_inference(model, inp, num_samples=num_mc, return_samples=False)
        preds_list.append(mc_out["predicted_classes"].squeeze(0).cpu().numpy())
        unc_list.append(mc_out["foreground_variance"].squeeze().cpu().numpy())

    preds_np = np.stack(preds_list, axis=0)
    unc_np = np.stack(unc_list, axis=0)

    # 4. Run Review Simulation across budgets
    budgets = [0.05, 0.10, 0.20, 0.30, 0.50]
    print(f"\nSimulating review strategies across budgets: {[f'{int(b*100)}%' for b in budgets]}...")
    review_results = simulate_pathologist_review(
        preds_np,
        masks_np,
        unc_np,
        budgets=budgets,
        seed=42
    )

    # 5. Display Comparative Table
    print("\n" + "="*80)
    print("HUMAN-IN-THE-LOOP PATHOLOGIST REVIEW BUDGET EFFICIENCY (Addition 4)")
    print("="*80)
    header = f"{'Budget':<8} | {'Patches':<8} | {'Uncertainty Catch %':<21} | {'Random Catch %':<16} | {'Efficiency Gain':<15}"
    print(header)
    print("-" * 80)
    for b_str, b_data in review_results["budgets"].items():
        unc_c = b_data["uncertainty_guided"]["error_catch_rate"] * 100
        rand_c = b_data["random"]["error_catch_rate"] * 100
        gain = b_data["efficiency_gain_ratio"]
        k = b_data["patches_reviewed"]
        print(f"{b_str:<8} | {k:<8} | {unc_c:>6.1f}%                 | {rand_c:>6.1f}%          | {gain:>5.2f}x faster")
    print("="*80)

    # 6. Save Plot
    save_fig_path = "reports/figures/human_review_simulation_curve.png"
    plot_review_simulation(
        review_results,
        title="Pathologist Active Review: Uncertainty vs Random Prioritization",
        save_path=save_fig_path
    )
    print(f"\nSaved review simulation curve to: {save_fig_path}")

    with open("reports/human_review_simulation_results.json", "w") as f:
        json.dump(review_results, f, indent=2)

    return review_results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_pathologist_review_simulation(device=device)
