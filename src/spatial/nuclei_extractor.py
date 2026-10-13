"""
Nuclei Instance Extraction and Morphometry Module
Extracts discrete cellular instances and attributes from semantic segmentation masks
using morphological filtering and connected component labeling.
"""

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects


def extract_nuclei_instances(semantic_mask, min_size=12):
    """
    Extracts individual cell nuclei from a semantic segmentation mask.
    
    Args:
        semantic_mask: (H, W) np.ndarray with class indices (0: BG, 1..5: cells)
        min_size: minimum pixel area to be considered a valid nucleus
    
    Returns:
        instance_mask: (H, W) np.ndarray with unique positive integer IDs
        cells_df: list of dicts with:
            - id: instance ID
            - class_id: majority class of the nucleus (1..5)
            - centroid_x: float
            - centroid_y: float
            - area: int (pixel count)
            - eccentricity: float
    """
    foreground_binary = semantic_mask > 0
    # Clean tiny noise specks using max_size
    try:
        cleaned_binary = remove_small_objects(foreground_binary, max_size=max(0, min_size - 1))
    except TypeError:
        cleaned_binary = remove_small_objects(foreground_binary, min_size=min_size)

    # Label connected components
    instance_mask, num_features = ndimage.label(cleaned_binary)

    cells_list = []
    if num_features == 0:
        return instance_mask, cells_list

    props = regionprops(instance_mask)

    for prop in props:
        inst_id = prop.label
        cy, cx = prop.centroid
        area = prop.area

        # Determine majority cell class inside this nucleus instance
        nucleus_pixels = semantic_mask[instance_mask == inst_id]
        if len(nucleus_pixels) == 0:
            continue
        # Bincount ignoring 0
        counts = np.bincount(nucleus_pixels[nucleus_pixels > 0])
        if len(counts) == 0:
            continue
        majority_class = int(np.argmax(counts))

        cells_list.append({
            "id": int(inst_id),
            "class_id": majority_class,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area": int(area),
            "eccentricity": float(prop.eccentricity) if hasattr(prop, "eccentricity") else 0.0
        })

    return instance_mask, cells_list
