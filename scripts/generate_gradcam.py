#!/usr/bin/env python3
"""
generate_gradcam.py — Generate 3D Grad-CAM saliency volumes for test patients.

For each patient, produces three NIfTI heatmaps (WT, TC, ET) that can be
overlaid on MRI scans in 3D Slicer / SlicerVR.  Also computes XAI evaluation
metrics (Pointing Game, Saliency Coverage, Saliency IoU, Weighted Dice)
by comparing the saliency maps against the ground truth labels — both in
the same MONAI-preprocessed coordinate space to ensure alignment.

Usage:
    python scripts/generate_gradcam.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 1
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
from src.xai.grad_cam import GradCAM3D
from src.xai.metrics import evaluate_saliency

# BraTS region names matching our 3 output channels
REGIONS = {0: "wt", 1: "tc", 2: "et"}
REGION_NAMES = {0: "Whole Tumor", 1: "Tumor Core", 2: "Enhancing Tumor"}


def _get_target_layer(model, layer_name: str):
    """
    Resolve a target layer from the SegResNet model.

    MONAI SegResNet architecture (with blocks_down=(1,2,2,4), init_filters=32):
        model.convInit        — initial convolution (4 → 32 filters)
        model.down_layers[0]  — encoder level 1:  32 filters, 160³
        model.down_layers[1]  — encoder level 2:  64 filters,  80³
        model.down_layers[2]  — encoder level 3: 128 filters,  40³
        model.down_layers[3]  — encoder level 4: 256 filters,  20³  (deepest/"bottleneck")
        model.up_layers[0..2] — decoder blocks
        model.conv_final      — final 1×1×1 conv → out_channels

    Args:
        model:      The loaded SegResNet model.
        layer_name: One of 'bottleneck', 'encoder3', 'encoder2', 'encoder1',
                    'decoder1'.

    Returns:
        The nn.Module to hook into.
    """
    if layer_name == "bottleneck":
        # Deepest encoder block (256 filters at 20³ for init_filters=32)
        return model.down_layers[-1]
    elif layer_name == "encoder3":
        # Third encoder block (128 filters at 40³)
        return model.down_layers[-2]
    elif layer_name == "encoder2":
        # Second encoder block (64 filters at 80³)
        return model.down_layers[-3]
    elif layer_name == "encoder1":
        # First encoder block (32 filters at 160³)
        return model.down_layers[0]
    elif layer_name == "decoder1":
        return model.up_layers[0]
    else:
        valid = ["bottleneck", "encoder3", "encoder2", "encoder1", "decoder1"]
        raise ValueError(f"Unknown layer '{layer_name}'. Choose from: {valid}")


def _patient_gradcam_done(patient_dir: Path, patient_id: str) -> bool:
    """Return True if all 3 Grad-CAM NIfTIs already exist for this patient."""
    return all(
        (patient_dir / f"{patient_id}_gradcam_{tag}.nii.gz").exists()
        for tag in REGIONS.values()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D Grad-CAM saliency volumes for test patients."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N patients (0 = all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to process")
    parser.add_argument("--layer", type=str, default="bottleneck",
                        choices=["bottleneck", "encoder3", "encoder2",
                                 "encoder1", "decoder1"],
                        help="Which layer to hook for Grad-CAM (default: bottleneck)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Saliency threshold for IoU metric (default: 0.5)")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    export_dir = Path(cfg["project"].get("project_root", os.getcwd())) / "slicer_export" / "XAI" / "Grad_CAM"
    export_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(cfg["project"].get("project_root", os.getcwd())) / "results" / "CSVs"
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Resolve target layer & create Grad-CAM ──────────────────────────
    target_layer = _get_target_layer(model, args.layer)
    print(f"Grad-CAM target layer: {args.layer} → {target_layer.__class__.__name__}")

    gcam = GradCAM3D(model, target_layer)
    roi_size = cfg["data"]["roi_size"]

    processed = 0
    skipped = 0

    # ── CSV for XAI evaluation metrics ──────────────────────────────────
    csv_path = results_dir / "xai_gradcam_metrics.csv"
    csv_fieldnames = [
        "Patient", "Region",
        "Pointing_Game", "Saliency_Coverage", "Saliency_IoU", "Weighted_Dice",
    ]
    metric_rows = []

    print(f"\nGenerating Grad-CAM volumes → {export_dir}")
    print(f"XAI metrics will be saved to → {csv_path}")
    print("Checking for already-exported patients to resume...\n")

    for data in tqdm(loader, desc="Grad-CAM"):

        if args.limit > 0 and processed >= args.limit:
            break

        # ── Extract patient ID ──────────────────────────────────────────
        if "image_meta_dict" in data and "filename_or_obj" in data["image_meta_dict"]:
            img_paths = data["image_meta_dict"]["filename_or_obj"]
        elif isinstance(data["image"], monai.data.MetaTensor) and \
                "filename_or_obj" in data["image"].meta:
            img_paths = data["image"].meta["filename_or_obj"]
        else:
            print("Warning: Could not find original filename. Skipping.")
            continue

        if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0 \
                and isinstance(img_paths[0], (list, tuple)):
            patient_modality_paths = img_paths[0]
        elif isinstance(img_paths, (list, tuple)):
            patient_modality_paths = img_paths
        else:
            patient_modality_paths = [img_paths]

        first_img_path = Path(patient_modality_paths[0])
        patient_id = first_img_path.parent.name

        # ── Per-patient export directory ────────────────────────────────
        patient_export_dir = export_dir / patient_id
        patient_export_dir.mkdir(parents=True, exist_ok=True)

        # Resume check
        if _patient_gradcam_done(patient_export_dir, patient_id):
            skipped += 1
            continue

        # ── Get the transform-aware affine for spatial alignment ────────
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

        # ── Get the ground truth label (already in preprocessed space) ──
        # The label has 3 channels: [WT, TC, ET] after
        # ConvertToMultiChannelBraTS2023d, same spatial dims as the image.
        gt_label = data["label"].cpu().numpy()  # (1, 3, D, H, W)

        # ── Prepare input ───────────────────────────────────────────────
        # Pad spatial dims to be divisible by 16 (2^4 for 4 encoder levels)
        # so the encoder downsampling/upsampling doesn't cause size mismatches.
        raw_input = data["image"].to(device)
        orig_shape = raw_input.shape[2:]  # (D, H, W) before padding
        divisor = 16
        pad_d = (divisor - orig_shape[0] % divisor) % divisor
        pad_h = (divisor - orig_shape[1] % divisor) % divisor
        pad_w = (divisor - orig_shape[2] % divisor) % divisor

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # F.pad expects (w_left, w_right, h_left, h_right, d_left, d_right)
            inputs = F.pad(raw_input, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0)
        else:
            inputs = raw_input

        inputs.requires_grad_(True)

        # ── Generate Grad-CAM for each region ───────────────────────────
        for ch_idx, tag in REGIONS.items():
            heatmap = gcam.generate(inputs, target_class=ch_idx, upsample=True)

            # Crop back to original preprocessed size (remove padding)
            heatmap = heatmap[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

            # Save NIfTI heatmap
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            nifti_img = nib.Nifti1Image(heatmap_uint8, affine)
            out_path = patient_export_dir / f"{patient_id}_gradcam_{tag}.nii.gz"
            nib.save(nifti_img, out_path)

            # ── Compute XAI evaluation metrics ──────────────────────────
            # Both heatmap and gt_channel are in the same preprocessed
            # coordinate space — no alignment issues!
            gt_channel = gt_label[0, ch_idx]  # (D, H, W), binary

            if gt_channel.sum() == 0:
                # No tumor present for this region
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
                    heatmap, gt_channel, threshold=args.iou_threshold
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
        tqdm.write(f"  ✓ {patient_id} — 3 Grad-CAM volumes + metrics saved")

        # Clean up memory
        del inputs, gt_label
        torch.cuda.empty_cache()

    gcam.remove_hooks()

    # ── Write metrics CSV ───────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\nCompleted: {processed} new + {skipped} skipped (already exported)")
    print(f"   Grad-CAM volumes: {export_dir}")
    print(f"   XAI metrics CSV:  {csv_path}")

    if metric_rows:
        print("\n" + "=" * 65)
        print("  SUMMARY: Saliency vs Ground Truth Alignment (Grad-CAM)")
        print("=" * 65)

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

        print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
