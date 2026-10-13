"""
Evaluation Script: Comparison of Uncertainty Estimation Methods (Addition 2)
Directly addresses Research Addition 2:
Compares:
1. Baseline Confidence: Maximum Softmax Probability (MSP) & Predictive Entropy
2. Monte Carlo Spatial Dropout (Variance & Mutual Information)
3. Deep Ensembles (Variance & Disagreement)

Measures:
- Pixel- and Object-level Error Detection (AUROC, AUPRC, FPR95)
- Calibration: Expected Calibration Error (ECE) and Brier Score
- Computational Cost: Inference Latency per Patch (ms) and Forward Passes
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pannuke_dataset import PanNukeDataset, load_dataset_split
from src.models.unet_dropout import UncertaintyAttentionUNet
from src.models.ensemble import DeepEnsemble
from src.uncertainty.mc_dropout import run_mc_dropout_inference
from src.uncertainty.entropy import compute_baseline_uncertainty
from src.uncertainty.calibration import compute_calibration_metrics
from src.evaluation.metrics import compute_segmentation_metrics
from src.evaluation.error_detection import evaluate_pixel_error_detection
from src.utils.visualization import plot_calibration_curve


def benchmark_uncertainty_methods(device="cpu", num_mc_samples=15):
    print("=== Benchmarking Uncertainty Estimation Methods (Addition 2) ===")
    
    # 1. Load In-Distribution Test Data
    test_data = load_dataset_split("data/pannuke/test_id/pannuke_test_id.npz")
    images_np = test_data["images"]
    masks_np = test_data["masks"]
    N = len(images_np)

    # 2. Load Models
    mc_model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3).to(device)
    mc_ckpt = "experiments/checkpoints/unet_mcdropout_best.pt"
    if os.path.exists(mc_ckpt):
        mc_model.load_state_dict(torch.load(mc_ckpt, map_location=device))
        print(f"Loaded MC-Dropout model from: {mc_ckpt}")
    else:
        print("Warning: Checkpoint not found, using initialized model.")
    mc_model.eval()

    # Load Ensemble Members
    ens_models = []
    for m in range(3):
        ens_ckpt = f"experiments/checkpoints/ensemble_member_{m}.pt"
        m_model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.0).to(device)
        if os.path.exists(ens_ckpt):
            m_model.load_state_dict(torch.load(ens_ckpt, map_location=device))
        ens_models.append(m_model)
    ensemble = DeepEnsemble(ens_models).to(device)

    # Convert test images to tensors
    test_tensors = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0

    results = {}

    # ----------------------------------------------------
    # Method 1: Baseline Maximum Softmax Probability (MSP)
    # ----------------------------------------------------
    print("\nEvaluating Method 1: Baseline Maximum Softmax Probability (MSP)...")
    start_t = time.time()
    msp_probs_list, msp_unc_list, msp_preds_list = [], [], []

    for i in range(N):
        inp = test_tensors[i:i+1].to(device)
        base_out = compute_baseline_uncertainty(mc_model, inp)
        msp_probs_list.append(base_out["probabilities"].cpu().numpy())
        msp_unc_list.append(base_out["msp_uncertainty"].squeeze(1).cpu().numpy())
        msp_preds_list.append(base_out["predicted_classes"].cpu().numpy())

    msp_latency = ((time.time() - start_t) / N) * 1000.0  # ms per patch
    msp_probs = np.concatenate(msp_probs_list, axis=0)
    msp_unc = np.concatenate(msp_unc_list, axis=0)
    msp_preds = np.concatenate(msp_preds_list, axis=0)

    msp_seg = compute_segmentation_metrics(msp_preds, masks_np)
    msp_calib = compute_calibration_metrics(msp_probs, masks_np)
    msp_err_det = evaluate_pixel_error_detection(msp_preds, masks_np, msp_unc)

    results["Maximum_Softmax_Probability"] = {
        "mean_dice": msp_seg["mean_dice"],
        "foreground_dice": msp_seg["foreground_dice"],
        "ece": msp_calib["ece"],
        "brier_score": msp_calib["brier_score"],
        "error_auroc": msp_err_det["auroc"],
        "error_auprc": msp_err_det["auprc"],
        "error_fpr95": msp_err_det["fpr95"],
        "latency_ms_per_patch": msp_latency,
        "forward_passes": 1
    }

    # ----------------------------------------------------
    # Method 2: Predictive Entropy (Single-Pass)
    # ----------------------------------------------------
    print("Evaluating Method 2: Baseline Single-Pass Predictive Entropy...")
    eps = 1e-8
    entropy_unc = -np.sum(msp_probs * np.log(msp_probs + eps), axis=1)
    ent_err_det = evaluate_pixel_error_detection(msp_preds, masks_np, entropy_unc)

    results["Predictive_Entropy"] = {
        "mean_dice": msp_seg["mean_dice"],
        "foreground_dice": msp_seg["foreground_dice"],
        "ece": msp_calib["ece"],
        "brier_score": msp_calib["brier_score"],
        "error_auroc": ent_err_det["auroc"],
        "error_auprc": ent_err_det["auprc"],
        "error_fpr95": ent_err_det["fpr95"],
        "latency_ms_per_patch": msp_latency,
        "forward_passes": 1
    }

    # ----------------------------------------------------
    # Method 3: Monte Carlo Spatial Dropout (T Samples)
    # ----------------------------------------------------
    print(f"Evaluating Method 3: Monte Carlo Spatial Dropout (T={num_mc_samples})...")
    start_t = time.time()
    mc_probs_list, mc_var_list, mc_mi_list, mc_preds_list = [], [], [], []

    for i in range(N):
        inp = test_tensors[i:i+1].to(device)
        mc_out = run_mc_dropout_inference(mc_model, inp, num_samples=num_mc_samples, return_samples=False)
        mc_probs_list.append(mc_out["mean_prediction"].cpu().numpy())
        mc_var_list.append(mc_out["foreground_variance"].squeeze(1).cpu().numpy())
        mc_mi_list.append(mc_out["mutual_information"].squeeze(1).cpu().numpy())
        mc_preds_list.append(mc_out["predicted_classes"].cpu().numpy())

    mc_latency = ((time.time() - start_t) / N) * 1000.0
    mc_probs = np.concatenate(mc_probs_list, axis=0)
    mc_var = np.concatenate(mc_var_list, axis=0)
    mc_mi = np.concatenate(mc_mi_list, axis=0)
    mc_preds = np.concatenate(mc_preds_list, axis=0)

    mc_seg = compute_segmentation_metrics(mc_preds, masks_np)
    mc_calib = compute_calibration_metrics(mc_probs, masks_np)
    mc_var_err_det = evaluate_pixel_error_detection(mc_preds, masks_np, mc_var)
    mc_mi_err_det = evaluate_pixel_error_detection(mc_preds, masks_np, mc_mi)

    results["Monte_Carlo_Dropout_Variance"] = {
        "mean_dice": mc_seg["mean_dice"],
        "foreground_dice": mc_seg["foreground_dice"],
        "ece": mc_calib["ece"],
        "brier_score": mc_calib["brier_score"],
        "error_auroc": mc_var_err_det["auroc"],
        "error_auprc": mc_var_err_det["auprc"],
        "error_fpr95": mc_var_err_det["fpr95"],
        "latency_ms_per_patch": mc_latency,
        "forward_passes": num_mc_samples
    }

    results["Monte_Carlo_Dropout_Mutual_Info"] = {
        "mean_dice": mc_seg["mean_dice"],
        "foreground_dice": mc_seg["foreground_dice"],
        "ece": mc_calib["ece"],
        "brier_score": mc_calib["brier_score"],
        "error_auroc": mc_mi_err_det["auroc"],
        "error_auprc": mc_mi_err_det["auprc"],
        "error_fpr95": mc_mi_err_det["fpr95"],
        "latency_ms_per_patch": mc_latency,
        "forward_passes": num_mc_samples
    }

    # ----------------------------------------------------
    # Method 4: Deep Ensembles (M=3 Members)
    # ----------------------------------------------------
    print("Evaluating Method 4: Deep Ensembles (M=3 Members)...")
    start_t = time.time()
    ens_probs_list, ens_var_list, ens_dis_list, ens_preds_list = [], [], [], []

    for i in range(N):
        inp = test_tensors[i:i+1].to(device)
        ens_out = ensemble.predict_with_uncertainty(inp)
        ens_probs_list.append(ens_out["mean_probs"].cpu().numpy())
        ens_var_list.append(ens_out["foreground_variance"].squeeze(1).cpu().numpy())
        ens_dis_list.append(ens_out["disagreement"].squeeze(1).cpu().numpy())
        ens_preds_list.append(torch.argmax(ens_out["mean_probs"], dim=1).cpu().numpy())

    ens_latency = ((time.time() - start_t) / N) * 1000.0
    ens_probs = np.concatenate(ens_probs_list, axis=0)
    ens_var = np.concatenate(ens_var_list, axis=0)
    ens_dis = np.concatenate(ens_dis_list, axis=0)
    ens_preds = np.concatenate(ens_preds_list, axis=0)

    ens_seg = compute_segmentation_metrics(ens_preds, masks_np)
    ens_calib = compute_calibration_metrics(ens_probs, masks_np)
    ens_err_det = evaluate_pixel_error_detection(ens_preds, masks_np, ens_var)

    results["Deep_Ensembles"] = {
        "mean_dice": ens_seg["mean_dice"],
        "foreground_dice": ens_seg["foreground_dice"],
        "ece": ens_calib["ece"],
        "brier_score": ens_calib["brier_score"],
        "error_auroc": ens_err_det["auroc"],
        "error_auprc": ens_err_det["auprc"],
        "error_fpr95": ens_err_det["fpr95"],
        "latency_ms_per_patch": ens_latency,
        "forward_passes": 3
    }

    # Plot reliability diagrams
    plot_calibration_curve(
        msp_calib,
        title="Reliability Diagram: Baseline MSP",
        save_path="reports/figures/calibration_baseline_msp.png"
    )
    plot_calibration_curve(
        mc_calib,
        title="Reliability Diagram: MC-Dropout",
        save_path="reports/figures/calibration_mc_dropout.png"
    )
    plot_calibration_curve(
        ens_calib,
        title="Reliability Diagram: Deep Ensemble",
        save_path="reports/figures/calibration_deep_ensemble.png"
    )

    # Print markdown table
    print("\n" + "="*80)
    print("COMPARATIVE UNCERTAINTY BENCHMARK RESULTS")
    print("="*80)
    header = f"{'Method':<32} | {'Dice (FG)':<9} | {'ECE':<7} | {'Brier':<7} | {'Err AUROC':<9} | {'FPR95':<7} | {'Latency':<8}"
    print(header)
    print("-" * len(header))
    for m_name, m_res in results.items():
        print(f"{m_name:<32} | {m_res['foreground_dice']:.3f}     | {m_res['ece']:.3f}   | {m_res['brier_score']:.3f}   | {m_res['error_auroc']:.3f}     | {m_res['error_fpr95']:.3f}   | {m_res['latency_ms_per_patch']:.1f} ms")
    print("="*80)

    os.makedirs("reports", exist_ok=True)
    with open("reports/uncertainty_methods_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    benchmark_uncertainty_methods(device=device)
