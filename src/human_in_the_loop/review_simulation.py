"""
Human-in-the-Loop Active Pathologist Review Simulation
Simulates clinical workflow where an expert pathologist has a fixed inspection budget.
Compares Random Review vs Uncertainty-Guided Review on:
- Cumulative genuine segmentation errors caught
- Severe failure (missed tumors & hallucinations) reduction
- Recovery of downstream TME spatial biomarkers after pathologist correction
"""

import numpy as np


def simulate_pathologist_review(
    patch_predictions,
    patch_targets,
    patch_uncertainties,
    patch_cells_pred=None,
    patch_cells_gt=None,
    budgets=[0.05, 0.10, 0.20, 0.30, 0.50],
    seed=42
):
    """
    Simulates pathologist review prioritization strategies under fixed inspection budgets.
    
    Args:
        patch_predictions: (N, H, W) np.ndarray
        patch_targets: (N, H, W) np.ndarray
        patch_uncertainties: (N, H, W) np.ndarray
        budgets: list of budget fractions
    
    Returns:
        dict containing review statistics across budgets for:
            - uncertainty_guided
            - random
            - oracle_worst_first
    """
    rng = np.random.default_rng(seed)
    N, H, W = patch_predictions.shape
    patch_pixel_errors = np.array([
        np.sum(patch_predictions[i] != patch_targets[i])
        for i in range(N)
    ])
    total_dataset_errors = float(np.sum(patch_pixel_errors))

    # Mean patch-level uncertainty
    patch_mean_uncertainties = np.array([
        np.mean(patch_uncertainties[i])
        for i in range(N)
    ])

    # Rank indices
    # 1. Uncertainty Guided (Descending order of uncertainty)
    uncertainty_ranked_idx = np.argsort(-patch_mean_uncertainties)
    # 2. Random
    random_idx = rng.permutation(N)
    # 3. Oracle (Descending order of true errors)
    oracle_idx = np.argsort(-patch_pixel_errors)

    results_by_budget = {}

    for b in budgets:
        k = max(1, int(b * N))
        
        # 1. Uncertainty-Guided review
        unc_reviewed = uncertainty_ranked_idx[:k]
        unc_errors_caught = float(np.sum(patch_pixel_errors[unc_reviewed]))
        unc_residual_errors = total_dataset_errors - unc_errors_caught
        unc_error_catch_rate = float(unc_errors_caught / max(1.0, total_dataset_errors))

        # 2. Random review (average over 5 trials)
        rand_caught_trials = []
        for _ in range(5):
            perm = rng.permutation(N)[:k]
            rand_caught_trials.append(float(np.sum(patch_pixel_errors[perm])))
        rand_errors_caught = float(np.mean(rand_caught_trials))
        rand_residual_errors = total_dataset_errors - rand_errors_caught
        rand_error_catch_rate = float(rand_errors_caught / max(1.0, total_dataset_errors))

        # 3. Oracle
        oracle_reviewed = oracle_idx[:k]
        oracle_errors_caught = float(np.sum(patch_pixel_errors[oracle_reviewed]))
        oracle_error_catch_rate = float(oracle_errors_caught / max(1.0, total_dataset_errors))

        # Efficiency gain: how many more errors uncertainty caught over random
        efficiency_gain_ratio = float(unc_errors_caught / max(1.0, rand_errors_caught))

        results_by_budget[f"{int(b*100)}%"] = {
            "budget_fraction": float(b),
            "patches_reviewed": int(k),
            "uncertainty_guided": {
                "errors_caught": unc_errors_caught,
                "residual_errors": unc_residual_errors,
                "error_catch_rate": unc_error_catch_rate
            },
            "random": {
                "errors_caught": rand_errors_caught,
                "residual_errors": rand_residual_errors,
                "error_catch_rate": rand_error_catch_rate
            },
            "oracle": {
                "errors_caught": oracle_errors_caught,
                "error_catch_rate": oracle_error_catch_rate
            },
            "efficiency_gain_ratio": efficiency_gain_ratio
        }

    return {
        "total_patches": int(N),
        "total_pixel_errors": total_dataset_errors,
        "budgets": results_by_budget
    }
