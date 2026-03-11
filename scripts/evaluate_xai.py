#!/usr/bin/env python3
"""
evaluate_xai.py — Evaluate XAI saliency maps against ground truth segmentations.

Measures how well the model's attention aligns with actual tumor locations
using three metrics:
    1. Pointing Game — peak saliency inside tumor? (binary hit/miss)
    2. Saliency Coverage — % of attention on the tumor (0–1)
    3. Saliency IoU — overlap of thresholded saliency with GT mask (0–1)

Reads previously exported Grad-CAM NIfTI volumes and the original BraTS
ground truth segmentations. Outputs a per-patient CSV and summary stats.

Usage:
    python scripts/evaluate_xai.py \
        --gradcam_dir slicer_export/XAI/Grad_CAM \
        --data_dir data/BraTS2023-Training \
        --output results/CSVs/xai_evaluation.csv
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.xai.metrics import evaluate_saliency

# BraTS output channels → region names and GT label mappings
REGIONS = {
    "wt": {"name": "Whole Tumor",     "labels": [1, 2, 3]},
    "tc": {"name": "Tumor Core",      "labels": [1, 3]},
    "et": {"name": "Enhancing Tumor",  "labels": [3]},
}


def load_ground_truth_region(seg_path: Path, labels: list) -> np.ndarray:
    """
    Load a BraTS segmentation and create a binary mask for the given labels.

    Args:
        seg_path: Path to the *-seg.nii.gz ground truth file.
        labels:   List of BraTS label values to include (e.g. [1, 2, 3] for WT).

    Returns:
        Binary numpy array (same shape as seg), 1 = region, 0 = background.
    """
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    mask = np.zeros_like(seg, dtype=np.uint8)
    for label in labels:
        mask[seg == label] = 1
    return mask


def load_gradcam_normalised(gradcam_path: Path) -> np.ndarray:
    """
    Load a Grad-CAM NIfTI (uint8 0–255) and normalise to [0, 1].
    """
    cam = nib.load(str(gradcam_path)).get_fdata().astype(np.float32)
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    return cam


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM saliency maps against ground truth."
    )
    parser.add_argument("--gradcam_dir", type=str, required=True,
                        help="Directory containing per-patient Grad-CAM exports "
                             "(e.g. slicer_export/XAI/Grad_CAM)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="BraTS data directory containing patient folders "
                             "with *-seg.nii.gz ground truth files")
    parser.add_argument("--output", type=str,
                        default="results/CSVs/xai_evaluation.csv",
                        help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Saliency threshold for IoU (default: 0.5)")
    args = parser.parse_args()

    gradcam_dir = Path(args.gradcam_dir)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all patient folders with Grad-CAM exports
    patient_dirs = sorted([
        d for d in gradcam_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS")
    ])

    if not patient_dirs:
        print(f"No patient directories found in {gradcam_dir}")
        return

    print(f"Found {len(patient_dirs)} patients with Grad-CAM exports")
    print(f"Saliency IoU threshold: {args.threshold}")
    print(f"Output: {output_path}\n")

    # ── CSV setup ───────────────────────────────────────────────────────
    fieldnames = [
        "Patient", "Region",
        "Pointing_Game", "Saliency_Coverage", "Saliency_IoU",
    ]

    rows = []

    for patient_dir in tqdm(patient_dirs, desc="Evaluating"):
        patient_id = patient_dir.name

        # Find ground truth segmentation
        gt_dir = data_dir / patient_id
        seg_files = list(gt_dir.glob("*-seg.nii.gz"))
        if not seg_files:
            tqdm.write(f"  ⚠ {patient_id}: no ground truth found, skipping")
            continue
        seg_path = seg_files[0]

        for tag, region_info in REGIONS.items():
            gradcam_path = patient_dir / f"{patient_id}_gradcam_{tag}.nii.gz"
            if not gradcam_path.exists():
                tqdm.write(f"  ⚠ {patient_id}: missing gradcam_{tag}, skipping")
                continue

            # Load and prepare data
            gt_mask = load_ground_truth_region(seg_path, region_info["labels"])
            saliency = load_gradcam_normalised(gradcam_path)

            # Handle shape mismatch (Grad-CAM may be in preprocessed space)
            if saliency.shape != gt_mask.shape:
                # Resize saliency to match GT using scipy
                from scipy.ndimage import zoom
                zoom_factors = [
                    gt / sal for gt, sal in zip(gt_mask.shape, saliency.shape)
                ]
                saliency = zoom(saliency, zoom_factors, order=1)

            # Check if the GT region has any positive voxels
            if gt_mask.sum() == 0:
                # No tumor present for this region → skip
                rows.append({
                    "Patient": patient_id,
                    "Region": region_info["name"],
                    "Pointing_Game": "N/A",
                    "Saliency_Coverage": "N/A",
                    "Saliency_IoU": "N/A",
                })
                continue

            # Compute metrics
            metrics = evaluate_saliency(saliency, gt_mask, args.threshold)

            rows.append({
                "Patient": patient_id,
                "Region": region_info["name"],
                "Pointing_Game": metrics["pointing_game"],
                "Saliency_Coverage": round(metrics["coverage"], 4),
                "Saliency_IoU": round(metrics["iou"], 4),
            })

    # ── Write CSV ───────────────────────────────────────────────────────
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Per-patient results saved to: {output_path}")

    # ── Summary statistics ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY: Saliency vs Ground Truth Alignment")
    print("=" * 65)

    for tag, region_info in REGIONS.items():
        region_rows = [
            r for r in rows
            if r["Region"] == region_info["name"]
            and r["Pointing_Game"] != "N/A"
        ]

        if not region_rows:
            print(f"\n  {region_info['name']}: No valid samples")
            continue

        pg_vals = [r["Pointing_Game"] for r in region_rows]
        cov_vals = [r["Saliency_Coverage"] for r in region_rows]
        iou_vals = [r["Saliency_IoU"] for r in region_rows]

        n = len(region_rows)
        pg_acc = np.mean(pg_vals) * 100
        cov_mean = np.mean(cov_vals)
        cov_std = np.std(cov_vals)
        iou_mean = np.mean(iou_vals)
        iou_std = np.std(iou_vals)

        print(f"\n  {region_info['name']} (n={n})")
        print(f"    Pointing Game Accuracy : {pg_acc:6.1f}%")
        print(f"    Saliency Coverage      : {cov_mean:.4f} ± {cov_std:.4f}")
        print(f"    Saliency IoU (τ={args.threshold})   : {iou_mean:.4f} ± {iou_std:.4f}")

    # Count patients with N/A (no tumor in that region)
    na_count = sum(1 for r in rows if r["Pointing_Game"] == "N/A")
    if na_count > 0:
        print(f"\n  Note: {na_count} region evaluations skipped (no tumor in GT)")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
