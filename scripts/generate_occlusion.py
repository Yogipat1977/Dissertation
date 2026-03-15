#!/usr/bin/env python3
"""
generate_occlusion.py — Generate 3D Occlusion Sensitivity relevance maps.

Uses a sliding window to occlude portions of the test set MRI scans,
recording the drop in the model's confidence for the tumor classes.
Exports NIfTI relevance maps to `slicer_export/XAI/Occlusion` and computes
quantitative metrics (including MSR Accuracy & Weighted Dice).
"""

import argparse
import os
import csv
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
from src.xai.occlusion import OcclusionSensitivity3D
from src.xai.metrics import evaluate_saliency

# BraTS regions
REGION_NAMES = {0: "wt", 1: "tc", 2: "et"}

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D Occlusion Sensitivity maps and evaluate metrics."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N patients (0 = all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Threshold for Saliency IoU metric")
    # Occlusion-specific parameters
    parser.add_argument("--window_size", type=int, default=16,
                        help="Size of 3D occluder cube (e.g. 16 means 16x16x16)")
    parser.add_argument("--stride", type=int, default=16,
                        help="Stride (step size) of the occluder (e.g. 16 for faster testing)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of occluded volumes to pass simultaneously")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(cfg["project"].get("project_root", os.getcwd()))
    
    # Export paths
    occ_export_dir = project_root / "slicer_export" / "XAI" / "Occlusion"
    occ_export_dir.mkdir(parents=True, exist_ok=True)

    results_dir = project_root / "results" / "CSVs"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "xai_occlusion_metrics.csv"
    
    csv_fieldnames = [
        "Patient", "Region", "Pointing_Game", "MSR_Accuracy",
        "Saliency_Coverage", "Saliency_IoU", "Weighted_Dice"
    ]
    metric_rows = []

    # ── Data ────────────────────────────────────────────────────────────
    print(f"\nLoading data (split: {args.split})...")
    loaders = create_data_loaders(cfg)
    loader = loaders[args.split]

    # Force batch_size=1 and num_workers=0 to prevent multiprocessing 
    # crashes on rented instances (often due to limited shared memory /dev/shm)
    loader = monai.data.DataLoader(
        loader.dataset, batch_size=1, shuffle=False,
        num_workers=0
    )

    # ── Model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = create_model(cfg, device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    win_sz = (args.window_size, args.window_size, args.window_size)
    strides = (args.stride, args.stride, args.stride)
    occ_engine = OcclusionSensitivity3D(model, window_size=win_sz, stride=strides)

    processed = 0

    print(f"\nGenerating Occlusion maps (Window {win_sz}, Stride {strides})")
    print(f"  Export directory → {occ_export_dir}")
    print(f"  Metrics CSV      → {csv_path}\n")

    for data in tqdm(loader, desc="Occlusion"):

        if args.limit > 0 and processed >= args.limit:
            break

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
        patient_export_dir = occ_export_dir / patient_id
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

        # ── Setup Inputs ────────────────────────────────────────────────
        raw_input = data["image"].to(device)
        gt_label = data["label"].numpy()[0]  # (3, 160, 160, 160)
        
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

        # ── Generate and Evaluate Maps ──────────────────────────────────
        # Generate all 3 classes at once -> (3, D, H, W)
        heatmaps_3d = occ_engine.generate(inputs, batch_size=args.batch_size)
        
        for ch_idx, tag in REGION_NAMES.items():
            heatmap = heatmaps_3d[ch_idx]
            heatmap = heatmap[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

            # Convert to uint8 and save NIfTI
            hm_uint8 = (heatmap * 255).astype(np.uint8)
            nifti_img = nib.Nifti1Image(hm_uint8, affine)
            out_path = patient_export_dir / f"{patient_id}_occlusion_{tag}.nii.gz"
            nib.save(nifti_img, out_path)

            # Evaluate Metrics
            gt_channel = gt_label[ch_idx]
            if gt_channel.sum() == 0:
                metric_rows.append({
                    "Patient": patient_id,
                    "Region": tag,
                    "Pointing_Game": "N/A",
                    "MSR_Accuracy": "N/A",
                    "Saliency_Coverage": "N/A",
                    "Saliency_IoU": "N/A",
                    "Weighted_Dice": "N/A",
                })
            else:
                metrics = evaluate_saliency(heatmap, gt_channel, threshold=args.iou_threshold)
                metric_rows.append({
                    "Patient": patient_id,
                    "Region": tag,
                    "Pointing_Game": metrics["pointing_game"],
                    "MSR_Accuracy": metrics["msr_accuracy"],
                    "Saliency_Coverage": round(metrics["coverage"], 4),
                    "Saliency_IoU": round(metrics["iou"], 4),
                    "Weighted_Dice": round(metrics["weighted_dice"], 4),
                })

        processed += 1
        tqdm.write(f"  ✓ {patient_id} — Occlusion map saved")

        del inputs, raw_input, gt_label
        torch.cuda.empty_cache()

    # ── Write CSV ───────────────────────────────────────────────────────
    if metric_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    # ── Summary stats ───────────────────────────────────────────────────
    print(f"\nCompleted: {processed} patients")
    print(f"   Occlusion volumes: {occ_export_dir}")
    print(f"   Metrics CSV:       {csv_path}")

    if metric_rows:
        print(f"\n{'=' * 65}")
        print(f"  SUMMARY: Occlusion Sensitivity Metrics")
        print(f"{'=' * 65}")

        for tag in REGION_NAMES.values():
            region_rows = [r for r in metric_rows if r["Region"] == tag and r["Pointing_Game"] != "N/A"]
            if not region_rows:
                print(f"\n  {tag}: No valid samples")
                continue

            n = len(region_rows)
            pg_acc = np.mean([r["Pointing_Game"] for r in region_rows]) * 100
            msr_acc = np.mean([r["MSR_Accuracy"] for r in region_rows]) * 100
            cov_mean = np.mean([r["Saliency_Coverage"] for r in region_rows])
            cov_std = np.std([r["Saliency_Coverage"] for r in region_rows])
            iou_mean = np.mean([r["Saliency_IoU"] for r in region_rows])
            iou_std = np.std([r["Saliency_IoU"] for r in region_rows])
            dice_mean = np.mean([r["Weighted_Dice"] for r in region_rows])
            dice_std = np.std([r["Weighted_Dice"] for r in region_rows])

            print(f"\n  {tag.upper()} (n={n})")
            print(f"    Pointing Game          : {pg_acc:5.1f}%")
            print(f"    MSR Accuracy           : {msr_acc:5.1f}%")
            print(f"    Saliency Coverage      : {cov_mean:.4f} ± {cov_std:.4f}")
            print(f"    Saliency IoU (τ={args.iou_threshold})   : {iou_mean:.4f} ± {iou_std:.4f}")
            print(f"    Weighted Dice          : {dice_mean:.4f} ± {dice_std:.4f}")

        na_count = sum(1 for r in metric_rows if r["Pointing_Game"] == "N/A")
        if na_count > 0:
            print(f"\n  Note: {na_count} region evaluations skipped (no tumor in GT)")

        print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
