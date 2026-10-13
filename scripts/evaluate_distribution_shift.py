"""
Evaluation Script: Genuine Cross-Organ Shift & Corruptions (Addition 3)
Addresses Research Addition 3:
1. Evaluates on genuine held-out tissue types (Kidney, Liver, Lung) vs In-Distribution (Breast, Colon).
2. Evaluates on controlled pathology corruptions (Stain drift, Out-of-focus blur, Tissue folding).
3. Compares Uncertainty-based Detectors (Entropy, MC-Dropout Variance, Mutual Info)
   against Feature-based Baselines (Bottleneck Mahalanobis Distance, Free Energy Score).
4. Quantifies AUROC, FPR95, and Segmentation Dice degradation under distribution shift.
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
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pannuke_dataset import load_dataset_split
from src.models.unet_dropout import UncertaintyAttentionUNet
from src.uncertainty.mc_dropout import run_mc_dropout_inference
from src.ood.feature_ood import FeatureMahalanobisOOD, compute_energy_score
from src.ood.cross_organ_shift import evaluate_ood_detection
from src.evaluation.metrics import compute_segmentation_metrics


def extract_cohort_scores(model, images_np, device="cpu", num_mc=10):
    N = len(images_np)
    tensors = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0

    entropy_scores = []
    mc_var_scores = []
    mc_mi_scores = []
    energy_scores = []
    features_list = []
    preds_list = []

    model.eval()
    for i in range(N):
        inp = tensors[i:i+1].to(device)

        # Single pass with features & energy
        with torch.no_grad():
            logits, feats = model(inp, return_features=True)
            probs = F.softmax(logits, dim=1)
            eps = 1e-8
            ent = -torch.sum(probs * torch.log(probs + eps), dim=1)
            entropy_scores.append(float(torch.mean(ent).cpu().item()))
            energy_scores.append(float(compute_energy_score(logits)[0]))
            features_list.append(feats.squeeze(0).cpu().numpy())

        # MC forward passes
        mc_out = run_mc_dropout_inference(model, inp, num_samples=num_mc, return_samples=False)
        mc_var_scores.append(float(torch.mean(mc_out["foreground_variance"]).cpu().item()))
        mc_mi_scores.append(float(torch.mean(mc_out["mutual_information"]).cpu().item()))
        preds_list.append(mc_out["predicted_classes"].squeeze(0).cpu().numpy())

    return {
        "entropy": np.array(entropy_scores),
        "mc_variance": np.array(mc_var_scores),
        "mc_mutual_info": np.array(mc_mi_scores),
        "energy": np.array(energy_scores),
        "features": np.array(features_list),
        "predictions": np.array(preds_list)
    }


def run_distribution_shift_experiments(device="cpu"):
    print("=== Running Genuine Cross-Organ Distribution Shift Evaluation (Addition 3) ===")

    # 1. Load Data Splits
    id_test = load_dataset_split("data/pannuke/test_id/pannuke_test_id.npz")
    id_train = load_dataset_split("data/pannuke/train/pannuke_train.npz")
    ood_organs = load_dataset_split("data/pannuke/test_ood_organs/pannuke_test_ood_organs.npz")
    ood_corrupted = load_dataset_split("data/pannuke/test_ood_corrupted/pannuke_test_ood_corrupted.npz")

    # 2. Load Model
    model = UncertaintyAttentionUNet(in_channels=3, num_classes=6, dropout_p=0.3).to(device)
    ckpt = "experiments/checkpoints/unet_mcdropout_best.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded model from: {ckpt}")
    model.eval()

    # 3. Fit Feature Mahalanobis detector on In-Distribution Training Set
    print("\nFitting Feature Mahalanobis baseline on In-Distribution training features...")
    tr_tensors = torch.from_numpy(id_train["images"][:40]).permute(0, 3, 1, 2).float() / 255.0
    tr_feats = []
    with torch.no_grad():
        for i in range(len(tr_tensors)):
            _, f = model(tr_tensors[i:i+1].to(device), return_features=True)
            tr_feats.append(f.squeeze(0).cpu().numpy())
    tr_feats = np.array(tr_feats)

    mahalanobis_detector = FeatureMahalanobisOOD()
    mahalanobis_detector.fit(tr_feats)

    # 4. Extract scores for cohorts
    print("Extracting representations and uncertainty for ID Test Cohort...")
    id_res = extract_cohort_scores(model, id_test["images"], device=device)
    id_mahal = mahalanobis_detector.score(id_res["features"])

    print("Extracting representations and uncertainty for Held-Out Organ Cohort (Kidney, Liver, Lung)...")
    organ_res = extract_cohort_scores(model, ood_organs["images"], device=device)
    organ_mahal = mahalanobis_detector.score(organ_res["features"])

    print("Extracting representations and uncertainty for Corrupted Cohort (Stain drift, Blur, Fold)...")
    corr_res = extract_cohort_scores(model, ood_corrupted["images"][:50], device=device)
    corr_mahal = mahalanobis_detector.score(corr_res["features"])

    # 5. Measure Segmentation Quality Degradation
    id_seg = compute_segmentation_metrics(id_res["predictions"], id_test["masks"])
    organ_seg = compute_segmentation_metrics(organ_res["predictions"], ood_organs["masks"])
    corr_seg = compute_segmentation_metrics(corr_res["predictions"], ood_corrupted["masks"][:50])

    # 6. Benchmark Anomaly / OoD Detection Performance
    # Task A: In-Distribution vs Cross-Organ Shift
    print("\n--- Task A: Detecting Cross-Organ Morphological Shift ---")
    organ_detectors = {
        "Predictive_Entropy": evaluate_ood_detection(id_res["entropy"], organ_res["entropy"]),
        "MC_Dropout_Variance": evaluate_ood_detection(id_res["mc_variance"], organ_res["mc_variance"]),
        "MC_Mutual_Information": evaluate_ood_detection(id_res["mc_mutual_info"], organ_res["mc_mutual_info"]),
        "Energy_Score": evaluate_ood_detection(id_res["energy"], organ_res["energy"]),
        "Feature_Mahalanobis": evaluate_ood_detection(id_mahal, organ_mahal)
    }

    # Task B: In-Distribution vs Histopathology Corruptions
    print("--- Task B: Detecting Histopathology Corruptions ---")
    corr_detectors = {
        "Predictive_Entropy": evaluate_ood_detection(id_res["entropy"], corr_res["entropy"]),
        "MC_Dropout_Variance": evaluate_ood_detection(id_res["mc_variance"], corr_res["mc_variance"]),
        "MC_Mutual_Information": evaluate_ood_detection(id_res["mc_mutual_info"], corr_res["mc_mutual_info"]),
        "Energy_Score": evaluate_ood_detection(id_res["energy"], corr_res["energy"]),
        "Feature_Mahalanobis": evaluate_ood_detection(id_mahal, corr_mahal)
    }

    # 7. Print Results Table
    print("\n" + "="*85)
    print("DISTRIBUTION SHIFT DETECTION PERFORMANCE (Addition 3)")
    print("="*85)
    print(f"{'Detector / Method':<26} | {'Cross-Organ AUROC':<18} | {'FPR95':<7} | {'Corruption AUROC':<17} | {'FPR95':<7}")
    print("-" * 85)
    for det_name in organ_detectors:
        org_r = organ_detectors[det_name]
        cor_r = corr_detectors[det_name]
        print(f"{det_name:<26} | {org_r['auroc']:.3f}              | {org_r['fpr95']:.3f}   | {cor_r['auroc']:.3f}             | {cor_r['fpr95']:.3f}")
    print("="*85)
    print(f"\nSegmentation Quality Degradation:")
    print(f"  In-Distribution Foreground Dice:      {id_seg['foreground_dice']:.3f} (mIoU: {id_seg['foreground_iou']:.3f})")
    print(f"  Cross-Organ Shift Foreground Dice:    {organ_seg['foreground_dice']:.3f} (Degradation: -{(id_seg['foreground_dice']-organ_seg['foreground_dice']):.3f})")
    print(f"  Corrupted Slides Foreground Dice:     {corr_seg['foreground_dice']:.3f} (Degradation: -{(id_seg['foreground_dice']-corr_seg['foreground_dice']):.3f})")

    # 8. Generate Degradation & Detection Barcharts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Plot A: Segmentation Dice Degradation
    shifts = ["In-Distribution\n(Breast/Colon)", "Cross-Organ Shift\n(Kidney/Liver/Lung)", "Corrupted Slides\n(Stain/Blur/Fold)"]
    dices = [id_seg["foreground_dice"], organ_seg["foreground_dice"], corr_seg["foreground_dice"]]
    colors = ["mediumseagreen", "coral", "indianred"]

    axes[0].bar(shifts, dices, color=colors, width=0.55, edgecolor="black")
    axes[0].set_ylabel("Foreground Dice Similarity Coefficient", fontsize=11)
    axes[0].set_title("Segmentation Degradation Under Distribution Shift", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1.0])
    for j, v in enumerate(dices):
        axes[0].text(j, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    # Plot B: OoD Detection AUROC Comparison
    method_names = list(organ_detectors.keys())
    org_aurocs = [organ_detectors[m]["auroc"] for m in method_names]
    corr_aurocs = [corr_detectors[m]["auroc"] for m in method_names]

    x = np.arange(len(method_names))
    width = 0.35
    axes[1].bar(x - width/2, org_aurocs, width, label="Cross-Organ Shift", color="royalblue", edgecolor="black")
    axes[1].bar(x + width/2, corr_aurocs, width, label="Stain/Blur Corruptions", color="crimson", edgecolor="black")
    axes[1].set_ylabel("OoD Detection AUROC", fontsize=11)
    axes[1].set_title("Uncertainty vs Feature-Based Shift Detection", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    clean_labels = [m.replace("_", "\n") for m in method_names]
    axes[1].set_xticklabels(clean_labels, fontsize=8)
    axes[1].set_ylim([0.4, 1.0])
    axes[1].axhline(0.5, color="gray", linestyle="--")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/distribution_shift_comparison.png", bbox_inches="tight")
    plt.close()

    output = {
        "segmentation_degradation": {
            "id": id_seg,
            "cross_organ": organ_seg,
            "corrupted": corr_seg
        },
        "cross_organ_detection": organ_detectors,
        "corruption_detection": corr_detectors
    }
    with open("reports/distribution_shift_results.json", "w") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_distribution_shift_experiments(device=device)
