"""
Unit Tests for Uncertainty Metrics, Calibration, and OoD Detectors
"""

import pytest
import numpy as np
import torch
from src.uncertainty.calibration import compute_calibration_metrics
from src.evaluation.error_detection import evaluate_pixel_error_detection
from src.evaluation.silent_failures import analyze_silent_failures
from src.evaluation.risk_coverage import compute_risk_coverage_curve
from src.ood.feature_ood import FeatureMahalanobisOOD, compute_energy_score


def test_calibration_metrics():
    # Perfectly calibrated case
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4]
    ])
    labels = np.array([0, 0, 0, 0])
    res = compute_calibration_metrics(probs, labels, n_bins=5, foreground_only=False)
    assert res["ece"] >= 0.0
    assert res["brier_score"] >= 0.0


def test_pixel_error_detection():
    pred_masks = np.zeros((2, 64, 64), dtype=np.int64)
    gt_masks = np.zeros((2, 64, 64), dtype=np.int64)
    gt_masks[0, 10:20, 10:20] = 1  # 100 error pixels
    unc_maps = np.random.uniform(0.0, 1.0, (2, 64, 64)).astype(np.float32)
    unc_maps[0, 10:20, 10:20] += 2.0  # High uncertainty where error exists

    res = evaluate_pixel_error_detection(pred_masks, gt_masks, unc_maps, foreground_only=False)
    assert 0.0 <= res["auroc"] <= 1.0
    assert 0.0 <= res["fpr95"] <= 1.0


def test_silent_failures():
    gt_cells = [{"id": 1, "class_id": 1, "centroid_x": 30.0, "centroid_y": 30.0, "area": 80}]
    pred_cells = [{"id": 1, "class_id": 1, "centroid_x": 100.0, "centroid_y": 100.0, "area": 80}] # FP hallucination
    unc_map = np.ones((128, 128), dtype=np.float32) * 0.1

    res = analyze_silent_failures(gt_cells, pred_cells, unc_map)
    assert res["false_positive_hallucinations"]["total"] == 1
    assert res["false_negative_omissions"]["total"] == 1


def test_risk_coverage_curve():
    errors = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    unc = np.array([0.1, 0.2, 0.3, 0.25, 0.9, 0.85, 0.7, 0.15, 0.2, 0.95])
    rc = compute_risk_coverage_curve(errors, unc, num_points=5)
    assert rc["aurc"] >= 0.0
    assert len(rc["coverages"]) == 5


def test_feature_mahalanobis_ood():
    id_feats = np.random.randn(20, 16).astype(np.float32)
    ood_feats = np.random.randn(10, 16).astype(np.float32) + 5.0 # Shifted

    det = FeatureMahalanobisOOD()
    det.fit(id_feats)
    id_scores = det.score(id_feats)
    ood_scores = det.score(ood_feats)

    assert np.mean(ood_scores) > np.mean(id_scores), "OOD scores should exceed ID scores!"
