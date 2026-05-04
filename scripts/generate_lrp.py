#!/usr/bin/env python3
"""
generate_lrp.py — Generate Layer-wise Relevance Propagation (LRP) volumes.

Produces three NIfTI relevance maps (WT, TC, ET) per patient using the
Input x Gradient proxy method (equivalent to epsilon-LRP in ReLU networks).
Exports maps for 3D Slicer / SlicerVR visualisation.

Also computes XAI evaluation metrics (Pointing Game, Saliency Coverage,
Saliency IoU, Weighted Dice) inline against ground truth labels — both
in the same MONAI-preprocessed coordinate space.

Usage:
    python scripts/generate_lrp.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 10
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

import monai
from monai.utils import set_determinism

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.dataset import create_data_loaders
from src.models.factory import create_model
from src.xai.lrp import LRP3D
from src.xai.metrics import evaluate_saliency

# BraTS regions
REGIONS = {0: "wt", 1: "tc", 2: "et"}
REGION_NAMES = {0: "Whole Tumor", 1: "Tumor Core", 2: "Enhancing Tumor"}

def main():
    parser = argparse.ArgumentParser(
        description="Generate LRP relevance volumes for test patients."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N patients (0 = all)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N patients (for incremental runs)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to process")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Saliency threshold for IoU metric (default: 0.5)")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(cfg["project"].get("project_root", os.getcwd()))
    lrp_export_dir = project_root / "slicer_export" / "XAI" / "LRP"
    lrp_export_dir.mkdir(parents=True, exist_ok=True)

    results_dir = project_root / "results" / "CSVs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV setup ───────────────────────────────────────────────────────
    csv_path = results_dir / "xai_lrp_metrics.csv"
    csv_fieldnames = [
        "Patient", "Region",
        "Pointing_Game", "Saliency_Coverage", "Saliency_IoU", "Weighted_Dice",
    ]
    metric_rows = []

    # ── Data ────────────────────────────────────────────────────────────
    print(f"\nLoading data (split: {args.split})...")
    loaders = create_data_loaders(cfg)
    loader = loaders[args.split]

    if loader.batch_size != 1:
        loader = monai.data.DataLoader(
            loader.dataset, batch_size=1, shuffle=False,
            num_workers=cfg["data"]["num_workers"]
        )

    # ── Model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = create_model(cfg, device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    lrp_engine = LRP3D(model)

    processed = 0

    print("\nGenerating LRP saliency volumes")
    print(f"  Export directory → {lrp_export_dir}")
    print(f"  Metrics CSV      → {csv_path}\n")

    for data in tqdm(loader, desc="LRP"):

        if args.limit > 0 and processed >= args.limit:
            break

        # Skip already-tested patients (for incremental runs)
        if processed < args.skip:
            processed += 1
            continue

        # ── Extract patient ID ──────────────────────────────────────────
        if "image_meta_dict" in data and "filename_or_obj" in data["image_meta_dict"]:
            img_paths = data["image_meta_dict"]["filename_or_obj"]
        elif isinstance(data["image"], monai.data.MetaTensor) and \
                "filename_or_obj" in data["image"].meta:
            img_paths = data["image"].meta["filename_or_obj"]
        else:
            continue

        if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0 \
                and isinstance(img_paths[0], (list, tuple)):
            patient_modality_paths = img_paths[0]
        elif isinstance(img_paths, (list, tuple)):
            patient_modality_paths = img_paths
        else:
            patient_modality_paths = [img_paths]

        patient_id = Path(patient_modality_paths[0]).parent.name
        patient_export_dir = lrp_export_dir / patient_id
        patient_export_dir.mkdir(parents=True, exist_ok=True)

        # ── Affine ──────────────────────────────────────────────────────
        if isinstance(data["image"], monai.data.MetaTensor):
            affine = data["image"].meta.get("affine", None)
            if affine is not None:
                affine = np.array(affine)
                if affine.ndim == 3:
                    affine = affine[0]
            else:
                affine = np.eye(4)
        elif "image_meta_dict" in data and "affine" in data["image_meta_dict"]:
            affine = np.array(data["image_meta_dict"]["affine"])
            if affine.ndim == 3:
                affine = affine[0]
        else:
            affine = np.eye(4)

        # ── Get the ground truth label ──────────────────────────────────
        gt_label = data["label"].cpu().numpy()  # (1, 3, D, H, W)

        # ── Divisibility padding ────────────────────────────────────────
        raw_input = data["image"].to(device)
        orig_shape = raw_input.shape[2:]
        divisor = 16
        pad_d = (divisor - orig_shape[0] % divisor) % divisor
        pad_h = (divisor - orig_shape[1] % divisor) % divisor
        pad_w = (divisor - orig_shape[2] % divisor) % divisor

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            inputs = F.pad(raw_input, (0, pad_w, 0, pad_h, 0, pad_d),
                           mode="constant", value=0)
        else:
            inputs = raw_input

        # ── Generate LRP maps ───────────────────────────────────────────
        for ch_idx, tag in REGIONS.items():
            relevance = lrp_engine.generate(inputs, target_class=ch_idx)
            relevance = relevance[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

            # Convert to uint8 and save NIfTI
            rel_uint8 = (relevance * 255).astype(np.uint8)
            nifti_img = nib.Nifti1Image(rel_uint8, affine)
            out_path = patient_export_dir / f"{patient_id}_lrp_{tag}.nii.gz"
            nib.save(nifti_img, out_path)

            # ── LRP metrics ─────────────────────────────────────────────
            gt_channel = gt_label[0, ch_idx]  # (D, H, W)

            if gt_channel.sum() == 0:
                metric_rows.append({
                    "Patient": patient_id,
                    "Region": REGION_NAMES[ch_idx],
                    "Pointing_Game": "N/A",
                    "Saliency_Coverage": "N/A",
                    "Saliency_IoU": "N/A",
                    "Weighted_Dice": "N/A",
                })
            else:
                metrics = evaluate_saliency(
                    relevance, gt_channel, threshold=args.iou_threshold
                )
                metric_rows.append({
                    "Patient": patient_id,
                    "Region": REGION_NAMES[ch_idx],
                    "Pointing_Game": metrics["pointing_game"],
                    "Saliency_Coverage": round(metrics["coverage"], 4),
                    "Saliency_IoU": round(metrics["iou"], 4),
                    "Weighted_Dice": round(metrics["weighted_dice"], 4),
                })

        processed += 1
        tqdm.write(f"  ✓ {patient_id} — LRP relevance saved")

        del inputs, raw_input, gt_label
        torch.cuda.empty_cache()

    # ── Write LRP metrics CSV ───────────────────────────────────────────
    if metric_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    # ── Summary stats ───────────────────────────────────────────────────
    print(f"\nCompleted: {processed} patients")
    print(f"   LRP volumes:  {lrp_export_dir}")
    print(f"   Metrics CSV:  {csv_path}")

    if metric_rows:
        print(f"\n{'=' * 65}")
        print(f"  SUMMARY: LRP — Saliency vs Ground Truth Alignment")
        print(f"{'=' * 65}")

        for ch_idx, region_name in REGION_NAMES.items():
            region_rows = [
                r for r in metric_rows
                if r["Region"] == region_name and r["Pointing_Game"] != "N/A"
            ]
            if not region_rows:
                print(f"\n  {region_name}: No valid samples")
                continue

            n = len(region_rows)
            pg_acc = np.mean([r["Pointing_Game"] for r in region_rows]) * 100
            cov_mean = np.mean([r["Saliency_Coverage"] for r in region_rows])
            cov_std = np.std([r["Saliency_Coverage"] for r in region_rows])
            iou_mean = np.mean([r["Saliency_IoU"] for r in region_rows])
            iou_std = np.std([r["Saliency_IoU"] for r in region_rows])
            dice_mean = np.mean([r["Weighted_Dice"] for r in region_rows])
            dice_std = np.std([r["Weighted_Dice"] for r in region_rows])

            print(f"\n  {region_name} (n={n})")
            print(f"    Pointing Game Accuracy : {pg_acc:6.1f}%")
            print(f"    Saliency Coverage      : {cov_mean:.4f} ± {cov_std:.4f}")
            print(f"    Saliency IoU (τ={args.iou_threshold})   : {iou_mean:.4f} ± {iou_std:.4f}")
            print(f"    Weighted Dice          : {dice_mean:.4f} ± {dice_std:.4f}")

        na_count = sum(1 for r in metric_rows if r["Pointing_Game"] == "N/A")
        if na_count > 0:
            print(f"\n  Note: {na_count} region evaluations skipped (no tumor in GT)")

        print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
