#!/usr/bin/env python3
"""
evaluate_gradcam_coarse.py — Evaluate Grad-CAM at native feature map resolution.

Instead of upsampling the Grad-CAM heatmap to full input resolution (160³),
this script downsamples the ground truth to match the native feature map
resolution (e.g. 20³ for bottleneck, 40³ for encoder3). This eliminates
the upsampling blur penalty and measures whether the model's coarse
spatial attention genuinely aligns with the tumor region.

This is a separate, complementary evaluation to the standard full-resolution
metrics computed in generate_gradcam.py.

Usage:
    python scripts/evaluate_gradcam_coarse.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 10 --layer bottleneck
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
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

REGIONS = {0: "wt", 1: "tc", 2: "et"}
REGION_NAMES = {0: "Whole Tumor", 1: "Tumor Core", 2: "Enhancing Tumor"}


def _get_target_layer(model, layer_name: str):
    """Resolve a target layer from the SegResNet model."""
    if layer_name == "bottleneck":
        return model.down_layers[-1]
    elif layer_name == "encoder3":
        return model.down_layers[-2]
    elif layer_name == "encoder2":
        return model.down_layers[-3]
    elif layer_name == "encoder1":
        return model.down_layers[0]
    elif layer_name == "decoder1":
        return model.up_layers[0]
    else:
        valid = ["bottleneck", "encoder3", "encoder2", "encoder1", "decoder1"]
        raise ValueError(f"Unknown layer '{layer_name}'. Choose from: {valid}")


def downsample_gt(gt_channel: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Downsample a binary ground truth mask to a coarser resolution.

    Uses average pooling to get the proportion of tumor voxels in each
    coarse block, then thresholds at 0.5 to get a binary coarse mask.
    A block is counted as 'tumor' if >50% of its voxels are tumor.

    Args:
        gt_channel: Binary 3D mask (D, H, W) with values 0 or 1.
        target_shape: Desired output shape (D', H', W').

    Returns:
        Downsampled binary mask (D', H', W') with values 0 or 1.
    """
    # Convert to tensor for F.interpolate: (1, 1, D, H, W)
    gt_tensor = torch.from_numpy(gt_channel.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Trilinear interpolation (acts like average pooling for downsampling)
    gt_coarse = F.interpolate(
        gt_tensor, size=target_shape, mode="trilinear", align_corners=False
    )

    # Threshold: if >50% of the original block was tumor → 1, else → 0
    gt_coarse_binary = (gt_coarse.squeeze().numpy() >= 0.5).astype(np.float32)

    return gt_coarse_binary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM at native feature map resolution "
                    "(downsampled GT comparison)."
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
                        help="Grad-CAM target layer (default: bottleneck)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Saliency threshold for IoU metric (default: 0.5)")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(cfg["project"].get("project_root", os.getcwd()))
    results_dir = project_root / "results" / "CSVs"
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

    target_layer = _get_target_layer(model, args.layer)

    # ── CSV setup ───────────────────────────────────────────────────────
    csv_path = results_dir / f"xai_gradcam_coarse_{args.layer}_metrics.csv"
    csv_fieldnames = [
        "Patient", "Region", "Native_Resolution",
        "Pointing_Game", "Saliency_Coverage", "Saliency_IoU", "Weighted_Dice",
    ]
    metric_rows = []

    print(f"\nEvaluating Grad-CAM at native resolution (layer: {args.layer})")
    print(f"  Strategy: downsample GT to match feature map size")
    print(f"  Output:   {csv_path}\n")

    processed = 0

    for data in tqdm(loader, desc=f"Coarse Grad-CAM ({args.layer})"):

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

        # ── Get the ground truth label ──────────────────────────────────
        gt_label = data["label"].cpu().numpy()  # (1, 3, D, H, W)

        # ── Pad input for divisibility ──────────────────────────────────
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

        # ── Generate Grad-CAM at NATIVE resolution (no upsample) ────────
        gcam = GradCAM3D(model, target_layer)

        for ch_idx, tag in REGIONS.items():
            # Get heatmap at native feature map resolution
            cam_native = gcam.generate(inputs, target_class=ch_idx, upsample=False)
            native_shape = cam_native.shape  # e.g. (20, 20, 20) for bottleneck

            # Downsample GT to match native resolution
            gt_channel = gt_label[0, ch_idx]  # (D, H, W) at full res
            gt_coarse = downsample_gt(gt_channel, native_shape)

            resolution_str = f"{native_shape[0]}x{native_shape[1]}x{native_shape[2]}"

            if gt_coarse.sum() == 0:
                metric_rows.append({
                    "Patient": patient_id,
                    "Region": REGION_NAMES[ch_idx],
                    "Native_Resolution": resolution_str,
                    "Pointing_Game": "N/A",
                    "Saliency_Coverage": "N/A",
                    "Saliency_IoU": "N/A",
                    "Weighted_Dice": "N/A",
                })
            else:
                metrics = evaluate_saliency(
                    cam_native, gt_coarse, threshold=args.iou_threshold
                )
                metric_rows.append({
                    "Patient": patient_id,
                    "Region": REGION_NAMES[ch_idx],
                    "Native_Resolution": resolution_str,
                    "Pointing_Game": metrics["pointing_game"],
                    "Saliency_Coverage": round(metrics["coverage"], 4),
                    "Saliency_IoU": round(metrics["iou"], 4),
                    "Weighted_Dice": round(metrics["weighted_dice"], 4),
                })

        gcam.remove_hooks()
        del gcam

        processed += 1
        tqdm.write(f"  ✓ {patient_id} — evaluated at {resolution_str}")

        del inputs, raw_input, gt_label
        torch.cuda.empty_cache()

    # ── Write CSV ───────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\nCompleted: {processed} patients")
    print(f"  Metrics: {csv_path}")

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY: Grad-CAM Coarse ({args.layer}) — Downsampled GT")
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
        print(f"    Native resolution      : {region_rows[0]['Native_Resolution']}")
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
