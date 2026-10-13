"""
Master Experiment Runner
Executes the full suite of research experiments:
1. Addition 2: Uncertainty Estimation Methods Benchmark (MSP, Entropy, MC-Dropout, Ensemble)
2. Addition 1: Downstream TME Spatial Biomarker Reliability (The Core Contribution)
3. Addition 3: Genuine Cross-Organ Distribution Shift & Corruption Evaluation
4. Addition 4: Human-in-the-Loop Pathologist Active Review Simulation
5. Addition 5: Silent Failure and Risk-Coverage Profiling
Generates reports/summary_results.json and calls reporting scripts.
"""

import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.evaluate_uncertainty_methods import benchmark_uncertainty_methods
from scripts.run_tme_spatial_analysis import run_tme_spatial_analysis
from scripts.evaluate_distribution_shift import run_distribution_shift_experiments
from scripts.run_review_simulation import run_pathologist_review_simulation
from scripts.analyze_silent_failures import run_silent_failure_analysis


def main():
    print("="*85)
    print("MASTER RESEARCH EVALUATION PIPELINE")
    print("Reliable Tumor Microenvironment Mapping: Spatial Cell Segmentation with OoD Detection")
    print("Research Title: Uncertainty-Guided Reliability of Cellular Spatial Biomarkers Under Cross-Tissue Shift")
    print("="*85)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Executing all benchmark experiments on: {device.upper()}\n")

    t0 = time.time()

    # Step 1: Benchmark Uncertainty Estimation Methods (Addition 2)
    print("\n>>> STEP 1: Benchmarking Uncertainty Methods (Addition 2)...")
    res_uncertainty = benchmark_uncertainty_methods(device=device, num_mc_samples=15)

    # Step 2: Downstream TME Spatial Biomarkers Analysis (Addition 1 - Primary Contribution)
    print("\n>>> STEP 2: Downstream TME Spatial Biomarker Reliability (Addition 1)...")
    res_tme = run_tme_spatial_analysis(device=device, num_mc_samples=15, num_patches=30)

    # Step 3: Genuine Cross-Organ Distribution Shifts & Corruptions (Addition 3)
    print("\n>>> STEP 3: Evaluating Genuine Cross-Organ Shift & Corruptions (Addition 3)...")
    res_shifts = run_distribution_shift_experiments(device=device)

    # Step 4: Human-in-the-Loop Pathologist Review Simulation (Addition 4)
    print("\n>>> STEP 4: Pathologist Active Review Budget Simulation (Addition 4)...")
    res_review = run_pathologist_review_simulation(device=device, num_mc=10)

    # Step 5: Silent Failure & Risk-Coverage Profiling (Addition 5)
    print("\n>>> STEP 5: Silent Failure & Selective Prediction Risk-Coverage (Addition 5)...")
    res_silent = run_silent_failure_analysis(device=device, num_mc=10)

    total_time = time.time() - t0

    # Save comprehensive master summary
    master_summary = {
        "metadata": {
            "title": "Uncertainty-Guided Reliability of Cellular Spatial Biomarkers Under Cross-Tissue Distribution Shift",
            "device": device,
            "total_runtime_seconds": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "addition_1_tme_spatial_stability": res_tme["summary_correlations"],
        "addition_2_uncertainty_methods": res_uncertainty,
        "addition_3_distribution_shifts": res_shifts,
        "addition_4_human_review_simulation": res_review["budgets"],
        "addition_5_silent_failures": res_silent["silent_failure_breakdown"],
        "risk_coverage": {
            "aurc": res_silent["risk_coverage"]["aurc"],
            "oracle_aurc": res_silent["risk_coverage"]["oracle_aurc"],
            "excess_aurc": res_silent["risk_coverage"]["excess_aurc"]
        }
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary_results.json", "w") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "="*85)
    print(f"ALL RESEARCH EXPERIMENTS SUCCESSFULLY EXECUTED! Total Time: {total_time:.1f}s")
    print("Master benchmark results saved to: reports/summary_results.json")
    print("="*85)


if __name__ == "__main__":
    main()
