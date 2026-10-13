"""
Benchmark Histopathology Dataset Generation Script
Synthesizes realistic multi-organ PanNuke-compatible cohorts:
- In-Distribution: Breast, Colon (Train, Val, Test)
- Held-out OOD Tissues: Kidney, Liver, Lung
- Corrupted Test Set: Stain drift, Out-of-focus blur, Tissue folding
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_generator import HistopathologyGenerator
from src.data.pannuke_dataset import save_dataset_split


def generate_cohort(generator, organ_list, num_samples_per_organ, corruption=None, severity=1.0):
    images = []
    masks = []
    instance_masks = []
    organs = []

    for organ in organ_list:
        for _ in range(num_samples_per_organ):
            img, sem_mask, inst_mask, _ = generator.generate_patch(organ=organ)
            if corruption is not None:
                img = generator.apply_corruption(img, corruption_type=corruption, severity=severity)
            images.append(img)
            masks.append(sem_mask)
            instance_masks.append(inst_mask)
            organs.append(organ)

    return (
        np.array(images, dtype=np.uint8),
        np.array(masks, dtype=np.int64),
        np.array(instance_masks, dtype=np.int32),
        organs
    )


def main():
    parser = argparse.ArgumentParser(description="Generate PanNuke Histopathology Benchmark")
    parser.add_argument("--data_dir", type=str, default="data/pannuke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_samples_per_organ", type=int, default=60) # 120 total ID train
    parser.add_argument("--val_samples_per_organ", type=int, default=20)   # 40 total ID val
    parser.add_argument("--test_samples_per_organ", type=int, default=25)  # 50 total ID test
    parser.add_argument("--ood_samples_per_organ", type=int, default=25)   # 75 total OOD organs
    args = parser.parse_args()

    print(f"=== Generating Multi-Organ Histopathology Benchmark Datasets (Seed={args.seed}) ===")
    gen = HistopathologyGenerator(patch_size=(256, 256), seed=args.seed)

    # 1. In-Distribution (Breast & Colon)
    id_organs = ["breast", "colon"]
    print(f"Generating In-Distribution Train Split ({id_organs})...")
    tr_imgs, tr_masks, tr_inst, tr_orgs = generate_cohort(gen, id_organs, args.train_samples_per_organ)
    save_dataset_split(os.path.join(args.data_dir, "train", "pannuke_train.npz"), tr_imgs, tr_masks, tr_inst, tr_orgs)
    print(f"  Train: {len(tr_imgs)} patches saved.")

    print(f"Generating In-Distribution Val Split...")
    val_imgs, val_masks, val_inst, val_orgs = generate_cohort(gen, id_organs, args.val_samples_per_organ)
    save_dataset_split(os.path.join(args.data_dir, "val", "pannuke_val.npz"), val_imgs, val_masks, val_inst, val_orgs)
    print(f"  Val: {len(val_imgs)} patches saved.")

    print(f"Generating In-Distribution Test Split...")
    test_id_imgs, test_id_masks, test_id_inst, test_id_orgs = generate_cohort(gen, id_organs, args.test_samples_per_organ)
    save_dataset_split(os.path.join(args.data_dir, "test_id", "pannuke_test_id.npz"), test_id_imgs, test_id_masks, test_id_inst, test_id_orgs)
    print(f"  Test ID: {len(test_id_imgs)} patches saved.")

    # 2. Genuine Cross-Organ Shift (Kidney, Liver, Lung)
    ood_organs = ["kidney", "liver", "lung"]
    print(f"Generating Held-Out Cross-Organ Shift Split ({ood_organs})...")
    ood_imgs, ood_masks, ood_inst, ood_orgs = generate_cohort(gen, ood_organs, args.ood_samples_per_organ)
    save_dataset_split(os.path.join(args.data_dir, "test_ood_organs", "pannuke_test_ood_organs.npz"), ood_imgs, ood_masks, ood_inst, ood_orgs)
    print(f"  Test OOD Organs: {len(ood_imgs)} patches saved.")

    # 3. Controlled Histopathology Corruptions
    corruptions = ["stain_variation", "gaussian_blur", "tissue_folding", "low_contrast"]
    print(f"Generating Histopathology Corruption Split ({corruptions})...")
    corr_imgs = []
    corr_masks = []
    corr_inst = []
    corr_orgs = []
    for corr in corruptions:
        c_imgs, c_masks, c_inst, c_orgs = generate_cohort(gen, ["breast", "colon"], 15, corruption=corr, severity=1.2)
        corr_imgs.append(c_imgs)
        corr_masks.append(c_masks)
        corr_inst.append(c_inst)
        corr_orgs.extend([f"{org}_{corr}" for org in c_orgs])
    
    corr_imgs = np.concatenate(corr_imgs, axis=0)
    corr_masks = np.concatenate(corr_masks, axis=0)
    corr_inst = np.concatenate(corr_inst, axis=0)
    save_dataset_split(os.path.join(args.data_dir, "test_ood_corrupted", "pannuke_test_ood_corrupted.npz"), corr_imgs, corr_masks, corr_inst, corr_orgs)
    print(f"  Test Corrupted: {len(corr_imgs)} patches saved.")

    print("=== Dataset Generation Successfully Completed! ===")


if __name__ == "__main__":
    main()
