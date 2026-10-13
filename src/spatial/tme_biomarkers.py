"""
Tumor Microenvironment (TME) Spatial Biomarker Extraction Module
Calculates biological spatial biomarkers derived from segmented cell nuclei:
- Cell density and spatial 2D Kernel Density Estimation (KDE)
- Nearest-neighbor distance (NND) distributions & immune-to-tumor proximity
- Cell-type proportions and composition Shannon entropy
- Cellular neighborhood graph (Delaunay triangulation network)
- Tumor-Stroma spatial mixing score (heterotypic interaction index)
"""

import math
import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.ndimage import gaussian_filter
import networkx as nx


def calculate_tme_spatial_biomarkers(cells, image_shape=(256, 256), max_delaunay_edge=60.0):
    """
    Computes a comprehensive dictionary of quantitative TME spatial biomarkers.
    
    Args:
        cells: list of dicts with keys: ['id', 'class_id', 'centroid_x', 'centroid_y', 'area']
        image_shape: tuple (H, W)
        max_delaunay_edge: maximum Euclidean distance to retain Delaunay graph edge
    
    Returns:
        dict of scalar biomarkers and spatial maps
    """
    H, W = image_shape
    total_area_pixels = float(H * W)
    num_cells = len(cells)

    # Default empty profile if no cells detected
    if num_cells < 2:
        return {
            "num_cells": num_cells,
            "total_cell_density": float(num_cells / (total_area_pixels / 1e4)),
            "mean_nnd": 0.0,
            "median_nnd": 0.0,
            "std_nnd": 0.0,
            "neoplastic_fraction": 0.0,
            "inflammatory_fraction": 0.0,
            "stroma_fraction": 0.0,
            "epithelial_fraction": 0.0,
            "composition_entropy": 0.0,
            "til_to_tumor_mean_distance": 0.0,
            "tumor_stroma_mixing_score": 0.0,
            "graph_avg_clustering": 0.0,
            "graph_avg_degree": 0.0,
            "density_map": np.zeros(image_shape, dtype=np.float32)
        }

    # Extract coordinates and class IDs
    coords = np.array([[c["centroid_x"], c["centroid_y"]] for c in cells]) # (N, 2)
    classes = np.array([c["class_id"] for c in cells])                     # (N,)

    # 1. Total Cell Density (cells per 10^4 pixels)
    density_per_10k_px = float(num_cells / (total_area_pixels / 1e4))

    # 2. 2D Spatial Density Map via Gaussian KDE
    density_map = np.zeros(image_shape, dtype=np.float32)
    for cx, cy in coords:
        ix, iy = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
        density_map[iy, ix] += 1.0
    density_map = gaussian_filter(density_map, sigma=15.0) * 1e4

    # 3. Nearest-Neighbor Distances (NND)
    kdtree = KDTree(coords)
    # k=2 because nearest neighbor of a point is itself at dist 0
    dists, _ = kdtree.query(coords, k=2)
    nnd_values = dists[:, 1]
    mean_nnd = float(np.mean(nnd_values))
    median_nnd = float(np.median(nnd_values))
    std_nnd = float(np.std(nnd_values))

    # 4. Cell-Type Proportions and Composition Entropy
    # Classes: 1: Neoplastic, 2: Inflammatory, 3: Connective, 4: Dead, 5: Epithelial
    counts = {cls_id: int(np.sum(classes == cls_id)) for cls_id in [1, 2, 3, 4, 5]}
    neoplastic_count = counts[1]
    inflammatory_count = counts[2]
    stroma_count = counts[3]
    epithelial_count = counts[5]

    neoplastic_fraction = float(neoplastic_count / num_cells)
    inflammatory_fraction = float(inflammatory_count / num_cells)
    stroma_fraction = float(stroma_count / num_cells)
    epithelial_fraction = float(epithelial_count / num_cells)

    # Shannon diversity entropy of cell types
    probs = np.array([counts[c] for c in [1, 2, 3, 4, 5]]) / num_cells
    probs = probs[probs > 0]
    composition_entropy = float(-np.sum(probs * np.log(probs)))

    # 5. Tumor-Infiltrating Lymphocyte (TIL) Proximity to Tumor
    # Distance from each inflammatory cell to the nearest neoplastic cell
    til_to_tumor_mean_dist = 0.0
    if neoplastic_count > 0 and inflammatory_count > 0:
        tumor_coords = coords[classes == 1]
        immune_coords = coords[classes == 2]
        tumor_tree = KDTree(tumor_coords)
        immune_to_tumor_dists, _ = tumor_tree.query(immune_coords, k=1)
        til_to_tumor_mean_dist = float(np.mean(immune_to_tumor_dists))

    # 6. Cellular Spatial Neighborhood Graph via Delaunay Triangulation
    G = nx.Graph()
    for i in range(num_cells):
        G.add_node(i, pos=coords[i], class_id=classes[i])

    heterotypic_edges = 0
    total_edges = 0

    if num_cells >= 4:
        try:
            tri = Delaunay(coords)
            # Find unique edges
            edges = set()
            for simplex in tri.simplices:
                for u, v in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                    edge = tuple(sorted([u, v]))
                    edges.add(edge)

            for u, v in edges:
                dist = float(np.linalg.norm(coords[u] - coords[v]))
                if dist <= max_delaunay_edge:
                    G.add_edge(u, v, weight=dist)
                    total_edges += 1
                    # Check if heterotypic (e.g. Tumor <-> Stroma or Tumor <-> Immune)
                    if classes[u] != classes[v]:
                        heterotypic_edges += 1
        except Exception:
            pass

    # Graph network metrics
    if G.number_of_edges() > 0:
        avg_clustering = float(nx.average_clustering(G))
        degrees = [d for _, d in G.degree()]
        avg_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
    else:
        avg_clustering = 0.0
        avg_degree = 0.0

    # Tumor-Stroma Mixing Score (Fraction of heterotypic interactions)
    mixing_score = float(heterotypic_edges / max(1, total_edges))

    return {
        "num_cells": int(num_cells),
        "total_cell_density": density_per_10k_px,
        "mean_nnd": mean_nnd,
        "median_nnd": median_nnd,
        "std_nnd": std_nnd,
        "neoplastic_fraction": neoplastic_fraction,
        "inflammatory_fraction": inflammatory_fraction,
        "stroma_fraction": stroma_fraction,
        "epithelial_fraction": epithelial_fraction,
        "composition_entropy": composition_entropy,
        "til_to_tumor_mean_distance": til_to_tumor_mean_dist,
        "tumor_stroma_mixing_score": mixing_score,
        "graph_avg_clustering": avg_clustering,
        "graph_avg_degree": avg_degree,
        "density_map": density_map
    }
