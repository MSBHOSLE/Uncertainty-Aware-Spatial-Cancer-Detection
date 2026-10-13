"""
Evaluation Script: Silent Failures and Risk-Coverage Profiling (Addition 5)
Addresses Research Addition 5:
1. Decomposes errors into:
   - False-positive hallucinated cells
   - Missing/omitted nuclei (false negatives)
   - Cell-type misclassifications
2. Quantifies silent failures: severe errors that slip past with LOW uncertainty.
3. Stratifies results by nucleus size (small vs large) and crowded boundaries.
4. Computes Risk-Coverage curves and Excess AURC for selective prediction.
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
from src.spatial.nuclei_extractor import extract_nuclei_instances
from src.evaluation.silent_failures import analyze_silent_failures
from src.evaluation.risk_coverage import compute_risk_coverage_curve
from src.utils.visualization import plot_risk_coverage


def run_silent_failure_analysis(device="cpu", num_mc=10):
    print("=== Running Silent Failure and Risk-Coverage Profiling (Addition 5) ===")

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

    # 3. Process Patches
    total_hallucinations = 0
    silent_hallucinations = 0
    total_omissions = 0
    silent_omissions = 0
    small_nuclei_omissions = 0
    silent_small_omissions = 0
    total_misclass = 0
    silent_misclass = 0

    all_pixel_errors = []
    all_pixel_uncertainties = []

    print(f"Profiling silent failures across {N} pathology patches...")
    tensors = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0

    for i in range(N):
        inp = tensors[i:i+1].to(device)
        mc_out = run_mc_dropout_inference(model, inp, num_samples=num_mc, return_samples=False)

        pred_mask = mc_out["predicted_classes"].squeeze(0).cpu().numpy()
        unc_map = mc_out["foreground_variance"].squeeze().cpu().numpy()
        gt_mask = masks_np[i]

        # Extract cell instances
        _, pred_cells = extract_nuclei_instances(pred_mask)
        _, gt_cells = extract_nuclei_instances(gt_mask)

        # Silent failure analysis for this patch
        sf_res = analyze_silent_failures(
            gt_cells,
            pred_cells,
            unc_map,
            distance_threshold=12.0,
            uncertainty_cutoff_percentile=50.0
        )

        fp_info = sf_res["false_positive_hallucinations"]
        fn_info = sf_res["false_negative_omissions"]
        mis_info = sf_res["cell_type_misclassifications"]

        total_hallucinations += fp_info["total"]
        silent_hallucinations += fp_info["silent_count"]

        total_omissions += fn_info["total"]
        silent_omissions += fn_info["silent_count"]
        small_nuclei_omissions += fn_info["small_nuclei_total"]
        silent_small_omissions += int(fn_info["small_nuclei_total"] * fn_info["small_nuclei_silent_rate"])

        total_misclass += mis_info["total"]
        silent_misclass += mis_info["silent_count"]

        # Pixel error arrays for risk-coverage
        pix_err = (pred_mask != gt_mask).astype(np.int32).flatten()
        pix_unc = unc_map.flatten()
        # Foreground pixels
        fg_idx = ((gt_mask > 0) | (pred_mask > 0)).flatten()
        all_pixel_errors.append(pix_err[fg_idx])
        all_pixel_uncertainties.append(pix_unc[fg_idx])

    # 4. Aggregate Silent Failure Statistics
    silent_summary = {
        "false_positive_hallucinations": {
            "total_detected": total_hallucinations,
            "silent_unflagged": silent_hallucinations,
            "silent_rate": float(silent_hallucinations / max(1, total_hallucinations)),
            "flagged_rate": 1.0 - float(silent_hallucinations / max(1, total_hallucinations))
        },
        "false_negative_omissions": {
            "total_missed": total_omissions,
            "silent_unflagged": silent_omissions,
            "silent_rate": float(silent_omissions / max(1, total_omissions)),
            "flagged_rate": 1.0 - float(silent_omissions / max(1, total_omissions)),
            "small_nuclei_total": small_nuclei_omissions,
            "small_nuclei_silent_rate": float(silent_small_omissions / max(1, small_nuclei_omissions))
        },
        "cell_type_misclassifications": {
            "total_misdiagnosed": total_misclass,
            "silent_unflagged": silent_misclass,
            "silent_rate": float(silent_misclass / max(1, total_misclass)),
            "flagged_rate": 1.0 - float(silent_misclass / max(1, total_misclass))
        }
    }

    print("\n" + "="*80)
    print("SILENT FAILURE ANALYSIS BREAKDOWN (Addition 5)")
    print("="*80)
    print(f"1. False Positive Hallucinations:   {total_hallucinations} total | {silent_hallucinations} silent ({silent_summary['false_positive_hallucinations']['silent_rate']*100:.1f}%) | Flagged: {silent_summary['false_positive_hallucinations']['flagged_rate']*100:.1f}%")
    print(f"2. False Negative Missed Nuclei:    {total_omissions} total | {silent_omissions} silent ({silent_summary['false_negative_omissions']['silent_rate']*100:.1f}%) | Flagged: {silent_summary['false_negative_omissions']['flagged_rate']*100:.1f}%")
    print(f"   -> Small Nuclei Missed Rate:     {silent_summary['false_negative_omissions']['small_nuclei_silent_rate']*100:.1f}% silent")
    print(f"3. Cell Diagnostic Mislabeling:     {total_misclass} total | {silent_misclass} silent ({silent_summary['cell_type_misclassifications']['silent_rate']*100:.1f}%) | Flagged: {silent_summary['cell_type_misclassifications']['flagged_rate']*100:.1f}%")
    print("="*80)

    # 5. Risk-Coverage Curve Computation
    all_err_concat = np.concatenate(all_pixel_errors)
    all_unc_concat = np.concatenate(all_pixel_uncertainties)
    rc_curve_data = compute_risk_coverage_curve(all_err_concat, all_unc_concat, num_points=20)

    print(f"\nSelective Prediction Risk-Coverage:")
    print(f"  Area Under Risk-Coverage (AURC):    {rc_curve_data['aurc']:.4f}")
    print(f"  Oracle Optimal AURC:                {rc_curve_data['oracle_aurc']:.4f}")
    print(f"  Excess AURC (E-AURC gap):           {rc_curve_data['excess_aurc']:.4f}")

    # Plot Risk-Coverage Curve
    plot_risk_coverage(
        rc_curve_data,
        title="Selective Prediction: Residual Error vs Autonomous Coverage",
        save_path="reports/figures/selective_prediction_risk_coverage.png"
    )

    output = {
        "silent_failure_breakdown": silent_summary,
        "risk_coverage": {
            "aurc": rc_curve_data["aurc"],
            "oracle_aurc": rc_curve_data["oracle_aurc"],
            "excess_aurc": rc_curve_data["excess_aurc"],
            "coverages": rc_curve_data["coverages"],
            "risks": rc_curve_data["risks"]
        }
    }
    with open("reports/silent_failures_results.json", "w") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_silent_failure_analysis(device=device)
