"""
Realistic Multi-Organ Histopathology Synthetic Benchmark Generator
Implements Beer-Lambert H&E Optical Density decomposition and realistic cellular
morphologies adhering to the 19-organ PanNuke histopathology specification.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Standard H&E Stain Optical Density vectors (Ruifrok & Johnston decomposition)
OD_HEMATOXYLIN = np.array([0.650, 0.704, 0.286])  # Nuclear blue/purple
OD_EOSIN       = np.array([0.072, 0.990, 0.105])  # Cytoplasmic/stromal pink
OD_BACKGROUND  = np.array([0.000, 0.000, 0.000])  # Glass slide white


def od_to_rgb(od_matrix):
    """Converts optical density values to 8-bit RGB image using Beer-Lambert law."""
    rgb = 255.0 * np.exp(-od_matrix)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class HistopathologyGenerator:
    """
    Generates realistic H&E stained pathology tissue patches (256x256)
    with 5 PanNuke cell categories and organ-specific tissue architectures.
    """
    
    ORGAN_PROFILES = {
        "breast": {
            "neoplastic_density": 0.40,
            "stroma_density": 0.35,
            "inflammatory_density": 0.15,
            "dead_density": 0.05,
            "epithelial_density": 0.05,
            "stroma_texture_strength": 0.8,
            "clustering": 0.75
        },
        "colon": {
            "neoplastic_density": 0.30,
            "stroma_density": 0.25,
            "inflammatory_density": 0.20,
            "dead_density": 0.05,
            "epithelial_density": 0.20,
            "stroma_texture_strength": 0.6,
            "clustering": 0.60
        },
        "kidney": {
            "neoplastic_density": 0.25,
            "stroma_density": 0.20,
            "inflammatory_density": 0.10,
            "dead_density": 0.05,
            "epithelial_density": 0.40,
            "stroma_texture_strength": 0.4,
            "clustering": 0.45
        },
        "liver": {
            "neoplastic_density": 0.35,
            "stroma_density": 0.15,
            "inflammatory_density": 0.15,
            "dead_density": 0.05,
            "epithelial_density": 0.30,
            "stroma_texture_strength": 0.3,
            "clustering": 0.40
        },
        "lung": {
            "neoplastic_density": 0.35,
            "stroma_density": 0.25,
            "inflammatory_density": 0.25,
            "dead_density": 0.05,
            "epithelial_density": 0.10,
            "stroma_texture_strength": 0.5,
            "clustering": 0.70
        }
    }

    def __init__(self, patch_size=(256, 256), seed=42):
        self.H, self.W = patch_size
        self.rng = np.random.default_rng(seed)

    def generate_patch(self, organ="breast", num_nuclei_range=(35, 75)):
        """
        Generates a synthetic tissue slide patch, semantic segmentation mask (0-5),
        and instance mask.
        Returns:
            rgb_image: (256, 256, 3) np.uint8
            semantic_mask: (256, 256) np.int64 (0: BG, 1: Neoplastic, 2: Inflammatory,
                            3: Connective, 4: Dead, 5: Epithelial)
            instance_mask: (256, 256) np.int32 (unique positive ID per nucleus)
            cell_metadata: list of dicts with centroid, class_id, area, eccentricity
        """
        profile = self.ORGAN_PROFILES.get(organ.lower(), self.ORGAN_PROFILES["breast"])
        num_nuclei = self.rng.integers(num_nuclei_range[0], num_nuclei_range[1] + 1)

        # 1. Base Eosin / Hematoxylin optical density fields
        h_density = np.zeros((self.H, self.W), dtype=np.float32)
        e_density = np.zeros((self.H, self.W), dtype=np.float32)

        # Baseline eosin background (extracellular matrix / cytoplasm)
        base_eosin = self.rng.uniform(0.18, 0.32)
        # Add smooth spatial gradients/fibers for extracellular stroma
        x = np.linspace(0, 1, self.W)
        y = np.linspace(0, 1, self.H)
        xx, yy = np.meshgrid(x, y)
        angle = self.rng.uniform(0, np.pi)
        fiber_wave = 0.08 * np.sin(2 * np.pi * 4 * (xx * np.cos(angle) + yy * np.sin(angle)))
        noise_tex = 0.04 * self.rng.standard_normal((self.H, self.W))
        e_density += np.clip(base_eosin + fiber_wave + noise_tex, 0.05, 0.65)

        semantic_mask = np.zeros((self.H, self.W), dtype=np.int64)
        instance_mask = np.zeros((self.H, self.W), dtype=np.int32)
        cell_metadata = []

        # 2. Sample nucleus positions and types
        probs = [
            profile["neoplastic_density"],
            profile["inflammatory_density"],
            profile["stroma_density"],
            profile["dead_density"],
            profile["epithelial_density"]
        ]
        probs = np.array(probs) / np.sum(probs)

        # Generate cluster centers for realistic tumor clustering
        num_clusters = self.rng.integers(2, 5)
        cluster_centers = [
            (self.rng.uniform(30, self.W - 30), self.rng.uniform(30, self.H - 30))
            for _ in range(num_clusters)
        ]

        current_inst_id = 1
        for _ in range(num_nuclei):
            cell_class = self.rng.choice([1, 2, 3, 4, 5], p=probs)
            
            # Position selection (clustered or scattered)
            if self.rng.random() < profile["clustering"]:
                c_idx = self.rng.integers(0, num_clusters)
                cx, cy = cluster_centers[c_idx]
                px = int(np.clip(cx + self.rng.normal(0, 24), 10, self.W - 10))
                py = int(np.clip(cy + self.rng.normal(0, 24), 10, self.H - 10))
            else:
                px = int(self.rng.uniform(10, self.W - 10))
                py = int(self.rng.uniform(10, self.H - 10))

            # Cell morphological attributes per class
            if cell_class == 1:
                # Neoplastic: enlarged, pleomorphic, darker chromatin
                rx = self.rng.uniform(7.0, 12.0)
                ry = self.rng.uniform(5.5, 9.5)
                h_peak = self.rng.uniform(1.2, 1.8)
                theta = self.rng.uniform(0, np.pi)
            elif cell_class == 2:
                # Inflammatory: small, round, intensely dense
                rx = self.rng.uniform(3.5, 5.5)
                ry = self.rng.uniform(3.5, 5.5)
                h_peak = self.rng.uniform(1.5, 2.2)
                theta = 0.0
            elif cell_class == 3:
                # Connective / Stroma: elongated spindle-shaped
                rx = self.rng.uniform(8.0, 14.0)
                ry = self.rng.uniform(2.5, 4.5)
                h_peak = self.rng.uniform(0.8, 1.3)
                theta = self.rng.uniform(0, np.pi)
            elif cell_class == 4:
                # Dead / Apoptotic: small, pyknotic, irregular
                rx = self.rng.uniform(2.5, 4.5)
                ry = self.rng.uniform(2.0, 4.0)
                h_peak = self.rng.uniform(1.6, 2.4)
                theta = self.rng.uniform(0, np.pi)
            else:
                # Epithelial: medium oval, orderly
                rx = self.rng.uniform(5.5, 8.5)
                ry = self.rng.uniform(4.5, 7.0)
                h_peak = self.rng.uniform(1.0, 1.5)
                theta = self.rng.uniform(0, np.pi)

            # Draw elliptical nucleus profile
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            r_max = int(max(rx, ry) + 4)
            y_min, y_max = max(0, py - r_max), min(self.H, py + r_max)
            x_min, x_max = max(0, px - r_max), min(self.W, px + r_max)

            if y_max <= y_min or x_max <= x_min:
                continue

            grid_y, grid_x = np.ogrid[y_min:y_max, x_min:x_max]
            dx = grid_x - px
            dy = grid_y - py
            
            # Rotated ellipse equation
            dx_rot = dx * cos_t + dy * sin_t
            dy_rot = -dx * sin_t + dy * cos_t
            dist_sq = (dx_rot / rx) ** 2 + (dy_rot / ry) ** 2

            nuc_mask = dist_sq <= 1.0
            if not np.any(nuc_mask):
                continue

            # Sub-patch views
            sub_h = h_density[y_min:y_max, x_min:x_max]
            sub_sem = semantic_mask[y_min:y_max, x_min:x_max]
            sub_inst = instance_mask[y_min:y_max, x_min:x_max]

            # Smooth chromatin intensity towards boundary
            attenuation = np.clip(1.0 - 0.5 * dist_sq, 0.0, 1.0)
            chromatin_texture = 1.0 + 0.15 * self.rng.standard_normal(dist_sq.shape)
            added_h = h_peak * attenuation * chromatin_texture * nuc_mask

            sub_h[nuc_mask] = np.maximum(sub_h[nuc_mask], added_h[nuc_mask])
            sub_sem[nuc_mask] = cell_class
            sub_inst[nuc_mask] = current_inst_id

            cell_metadata.append({
                "id": current_inst_id,
                "class_id": int(cell_class),
                "centroid": (float(px), float(py)),
                "semi_axes": (float(rx), float(ry)),
                "area": float(np.sum(nuc_mask))
            })
            current_inst_id += 1

        # 3. Reconstruct RGB using Beer-Lambert combination:
        # OD = H_density * OD_H + E_density * OD_E
        od_total = (
            h_density[:, :, None] * OD_HEMATOXYLIN[None, None, :] +
            e_density[:, :, None] * OD_EOSIN[None, None, :]
        )
        rgb_image = od_to_rgb(od_total)

        return rgb_image, semantic_mask, instance_mask, cell_metadata

    def apply_corruption(self, rgb_image, corruption_type="stain_variation", severity=1.0):
        """
        Applies controlled real-world histopathology corruptions.
        """
        img_float = rgb_image.astype(np.float32)

        if corruption_type == "stain_variation":
            # Significant H&E stain drift (over-eosin or degraded hematoxylin)
            drift_matrix = np.array([
                [1.15 * severity, 0.0, 0.0],
                [0.0, 0.85 / max(0.5, severity), 0.0],
                [0.0, 0.0, 1.25 * severity]
            ])
            corrupted = np.clip(np.dot(img_float, drift_matrix), 0, 255).astype(np.uint8)

        elif corruption_type == "gaussian_blur":
            # Out-of-focus optics
            pil_img = Image.fromarray(rgb_image)
            radius = 1.8 * severity
            corrupted = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)))

        elif corruption_type == "tissue_folding":
            # Tissue fold: dark, dense optical stripe across image with loss of cellular detail
            corrupted = rgb_image.copy()
            y0 = int(self.rng.uniform(30, self.H - 60))
            thickness = int(14 * severity)
            corrupted[y0:y0+thickness, :, :] = (corrupted[y0:y0+thickness, :, :].astype(np.float32) * 0.45).astype(np.uint8)

        elif corruption_type == "low_contrast":
            # Under-illuminated microscope field
            corrupted = np.clip(img_float * 0.65 + 20, 0, 255).astype(np.uint8)

        else:
            corrupted = rgb_image

        return corrupted
