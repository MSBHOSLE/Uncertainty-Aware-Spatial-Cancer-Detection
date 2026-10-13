"""
Risk-Coverage and Selective Prediction Referral Analysis
Evaluates the trade-off between coverage (fraction of predictions retained autonomously)
and risk (residual error rate on retained predictions).
"""

import numpy as np


def compute_risk_coverage_curve(errors, uncertainty_scores, num_points=20):
    """
    Computes selective prediction risk vs coverage curve.
    
    Args:
        errors: 1D binary array of errors (1 = error, 0 = correct)
        uncertainty_scores: 1D array of uncertainty scores
        num_points: number of coverage evaluation thresholds
    
    Returns:
        dict containing:
            - coverages: list of float [1.0 down to ~0.05]
            - risks: list of residual error rates at each coverage
            - aurc: float (Area Under Risk-Coverage curve)
            - e_aurc: float (Excess AURC compared to optimal oracle)
    """
    total_samples = len(errors)
    if total_samples == 0:
        return {"coverages": [1.0], "risks": [0.0], "aurc": 0.0, "e_aurc": 0.0}

    # Sort samples from lowest uncertainty to highest uncertainty
    sort_idx = np.argsort(uncertainty_scores)
    sorted_errors = errors[sort_idx]

    coverages = []
    risks = []
    
    step_fractions = np.linspace(1.0, 0.05, num_points)

    for frac in step_fractions:
        k = max(1, int(frac * total_samples))
        retained_errors = sorted_errors[:k]
        risk = float(np.mean(retained_errors))

        coverages.append(float(k / total_samples))
        risks.append(risk)

    # Calculate AURC via trapezoidal integration
    from scipy.integrate import trapezoid
    aurc = float(trapezoid(risks[::-1], coverages[::-1]))

    # Optimal oracle ordering (errors deferred first)
    oracle_sorted_errors = np.sort(errors)
    oracle_risks = [float(np.mean(oracle_sorted_errors[:max(1, int(f * total_samples))])) for f in step_fractions]
    oracle_aurc = float(trapezoid(oracle_risks[::-1], coverages[::-1]))
    e_aurc = max(0.0, aurc - oracle_aurc)

    return {
        "coverages": coverages,
        "risks": risks,
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "excess_aurc": e_aurc
    }
