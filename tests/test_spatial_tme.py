"""
Unit Tests for Downstream TME Spatial Biomarkers (Addition 1)
"""

import pytest
import numpy as np
from src.spatial.nuclei_extractor import extract_nuclei_instances
from src.spatial.tme_biomarkers import calculate_tme_spatial_biomarkers
from src.spatial.biomarker_stability import evaluate_biomarker_stability_under_stochastic_sampling


def test_nuclei_extraction():
    # Create synthetic mask with 2 distinct circular nuclei
    mask = np.zeros((256, 256), dtype=np.int64)
    # Circle 1 (Neoplastic, class 1)
    y, x = np.ogrid[:256, :256]
    c1 = (x - 60)**2 + (y - 60)**2 <= 10**2
    # Circle 2 (Inflammatory, class 2)
    c2 = (x - 140)**2 + (y - 140)**2 <= 6**2

    mask[c1] = 1
    mask[c2] = 2

    inst_mask, cells = extract_nuclei_instances(mask, min_size=10)
    assert len(cells) == 2, f"Expected 2 cells, found {len(cells)}"
    assert cells[0]["class_id"] in [1, 2]
    assert cells[1]["class_id"] in [1, 2]


def test_tme_spatial_biomarkers():
    cells = [
        {"id": 1, "class_id": 1, "centroid_x": 50.0, "centroid_y": 50.0, "area": 100},
        {"id": 2, "class_id": 1, "centroid_x": 70.0, "centroid_y": 50.0, "area": 110},
        {"id": 3, "class_id": 2, "centroid_x": 60.0, "centroid_y": 80.0, "area": 40},
        {"id": 4, "class_id": 3, "centroid_x": 120.0, "centroid_y": 120.0, "area": 80},
    ]
    biomarkers = calculate_tme_spatial_biomarkers(cells, image_shape=(256, 256))

    assert biomarkers["num_cells"] == 4
    assert biomarkers["total_cell_density"] > 0.0
    assert biomarkers["mean_nnd"] > 0.0
    assert biomarkers["neoplastic_fraction"] == 0.5
    assert biomarkers["inflammatory_fraction"] == 0.25
    assert biomarkers["density_map"].shape == (256, 256)


def test_biomarker_stability_evaluation():
    T, C, H, W = 5, 6, 256, 256
    rng = np.random.default_rng(42)
    # Generate random stochastic probabilities
    probs = rng.uniform(0.1, 1.0, size=(T, C, H, W)).astype(np.float32)
    probs /= np.sum(probs, axis=1, keepdims=True)

    gt_mask = np.zeros((H, W), dtype=np.int64)
    gt_mask[40:55, 40:55] = 1

    unc_map = np.var(probs, axis=0)[1]  # variance of class 1

    stab_results = evaluate_biomarker_stability_under_stochastic_sampling(
        probs,
        gt_semantic_mask=gt_mask,
        uncertainty_map=unc_map,
        image_shape=(H, W)
    )

    assert "stability_metrics" in stab_results
    assert "total_cell_density" in stab_results["stability_metrics"]
    assert stab_results["region_trust_map"].shape == (H, W)
