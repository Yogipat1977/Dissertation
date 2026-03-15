#!/usr/bin/env python3
"""
generate_gbp.py — Generate Guided Backpropagation and Guided Grad-CAM saliency
volumes for test patients.

For each patient, produces three NIfTI saliency maps (WT, TC, ET) using
Guided Backpropagation. Optionally combines with Grad-CAM to produce
Guided Grad-CAM maps (--guided_gradcam flag).

Also computes XAI evaluation metrics (Pointing Game, Saliency Coverage,
Saliency IoU, Weighted Dice) inline against ground truth labels — both
in the same MONAI-preprocessed coordinate space.

Usage:
    # GBP only
    python scripts/generate_gbp.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 1

    # GBP + Guided Grad-CAM
    python scripts/generate_gbp.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 1 --guided_gradcam

    # GBP + Guided Grad-CAM with Top-K thresholding
    python scripts/generate_gbp.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 1 --guided_gradcam --topk 15
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
from src.xai.guided_backprop import GuidedBackprop3D
from src.xai.metrics import evaluate_saliency

# BraTS region names matching our 3 output channels
REGIONS = {0: "wt", 1: "tc", 2: "et"}
REGION_NAMES = {0: "Whole Tumor", 1: "Tumor Core", 2: "Enhancing Tumor"}


def _get_target_layer(model, layer_name: str):
    """Resolve a target layer from the SegResNet model (for Guided Grad-CAM)."""
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


def topk_threshold(heatmap: np.ndarray, k_percent: float) -> np.ndarray:
    """
    Apply Top-K% thresholding to a saliency heatmap.

    Keeps only the top K% of voxels by intensity, setting the rest to zero.
    The retained voxels keep their original values (not binarised).

    Args:
        heatmap:   3D numpy array with values in [0, 1].
        k_percent: Percentage of voxels to retain (e.g. 15 = top 15%).

    Returns:
        Thresholded heatmap with bottom (100-K)% voxels set to zero.
    """
    threshold = np.percentile(heatmap, 100 - k_percent)
    thresholded = heatmap.copy()
    thresholded[heatmap < threshold] = 0.0
    return thresholded


def main():
    parser = argparse.ArgumentParser(
        description="Generate Guided Backpropagation (and Guided Grad-CAM) "
                    "saliency volumes for test patients."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N patients (0 = all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to process")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Saliency threshold for IoU metric (default: 0.5)")
    parser.add_argument("--guided_gradcam", action="store_true",
                        help="Also generate Guided Grad-CAM (GBP × Grad-CAM)")
    parser.add_argument("--layer", type=str, default="bottleneck",
                        choices=["bottleneck", "encoder3", "encoder2",
                                 "encoder1", "decoder1"],
                        help="Grad-CAM target layer for Guided Grad-CAM "
                              "(default: bottleneck)")
    parser.add_argument("--topk", type=float, default=0,
                        help="Top-K%% thresholding for Guided Grad-CAM: keep "
                             "only the top K%% of saliency voxels "
                             "(e.g. --topk 15). Saves separate exports and CSV. "
                             "Set to 0 to disable (default: 0).")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(cfg["project"].get("project_root", os.getcwd()))
    gbp_export_dir = project_root / "slicer_export" / "XAI" / "GBP"
    gbp_export_dir.mkdir(parents=True, exist_ok=True)

    results_dir = project_root / "results" / "CSVs"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.guided_gradcam:
        ggcam_export_dir = project_root / "slicer_export" / "XAI" / "Guided_Grad_CAM"
        ggcam_export_dir.mkdir(parents=True, exist_ok=True)

        do_topk = args.topk > 0
        if do_topk:
            topk_pct = args.topk
            topk_tag = f"topk{int(topk_pct)}"
            topk_export_dir = project_root / "slicer_export" / "XAI" / "Guided_Grad_CAM_TopK"
            topk_export_dir.mkdir(parents=True, exist_ok=True)
    else:
        do_topk = False

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

    # NOTE: We do NOT create GBP or Grad-CAM engines here.
    # When running --guided_gradcam, both hook systems interfere if
    # active simultaneously. Instead, we create them per-patient in
    # sequence: Grad-CAM first (clean model) → GBP second (GBP hooks only).

    if args.guided_gradcam:
        from src.xai.grad_cam import GradCAM3D
        target_layer = _get_target_layer(model, args.layer)
        print(f"Guided Grad-CAM mode (Grad-CAM layer: {args.layer})")

    processed = 0
    # ── CSV setup ───────────────────────────────────────────────────────
    gbp_csv_path = results_dir / "xai_gbp_metrics.csv"
    csv_fieldnames = [
        "Patient", "Region",
        "Pointing_Game", "Saliency_Coverage", "Saliency_IoU", "Weighted_Dice",
    ]
    gbp_metric_rows = []

    ggcam_csv_path = results_dir / "xai_guided_gradcam_metrics.csv"
    ggcam_metric_rows = []

    if do_topk:
        topk_csv_path = results_dir / f"xai_guided_gradcam_{topk_tag}_weighted_dice.csv"
        topk_fieldnames = [
            "Patient", "Region", "TopK_Percent", "Weighted_Dice",
        ]
        topk_rows = []

    methods_str = "GBP" + (" + Guided Grad-CAM" if args.guided_gradcam else "")
    print(f"\nGenerating {methods_str} saliency volumes")
    print(f"  GBP output    → {gbp_export_dir}")
    if args.guided_gradcam:
        print(f"  G-GradCAM out → {ggcam_export_dir}")
    print(f"  GBP metrics   → {gbp_csv_path}")
    if args.guided_gradcam:
        print(f"  G-GradCAM met → {ggcam_csv_path}")
    if do_topk:
        print(f"Top-K threshold enabled for Guided Grad-CAM (K={topk_pct}%)")
        print(f"  Top-K exports → {topk_export_dir}")
        print(f"  Top-K metrics → {topk_csv_path}")
    print()

    for data in tqdm(loader, desc=methods_str):

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

        # ── Per-patient export directories ──────────────────────────────
        gbp_patient_dir = gbp_export_dir / patient_id
        gbp_patient_dir.mkdir(parents=True, exist_ok=True)

        if args.guided_gradcam:
            ggcam_patient_dir = ggcam_export_dir / patient_id
            ggcam_patient_dir.mkdir(parents=True, exist_ok=True)

        # ── Get the transform-aware affine ──────────────────────────────
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

        # ── Pad input for divisibility ──────────────────────────────────
        raw_input = data["image"].to(device)
        orig_shape = raw_input.shape[2:]  # (D, H, W)
        divisor = 16
        pad_d = (divisor - orig_shape[0] % divisor) % divisor
        pad_h = (divisor - orig_shape[1] % divisor) % divisor
        pad_w = (divisor - orig_shape[2] % divisor) % divisor

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            inputs = F.pad(raw_input, (0, pad_w, 0, pad_h, 0, pad_d),
                           mode="constant", value=0)
        else:
            inputs = raw_input

        # ════════════════════════════════════════════════════════════════
        # STEP 1: Grad-CAM (clean model, NO GBP hooks)
        # ════════════════════════════════════════════════════════════════
        gradcam_heatmaps = {}
        if args.guided_gradcam:
            gcam = GradCAM3D(model, target_layer)
            for ch_idx, tag in REGIONS.items():
                heatmap = gcam.generate(
                    inputs, target_class=ch_idx, upsample=True
                )
                gradcam_heatmaps[ch_idx] = heatmap[
                    :orig_shape[0], :orig_shape[1], :orig_shape[2]
                ]
            gcam.remove_hooks()
            del gcam

        # ════════════════════════════════════════════════════════════════
        # STEP 2: GBP (install GBP hooks, NO Grad-CAM hooks)
        # ════════════════════════════════════════════════════════════════
        gbp = GuidedBackprop3D(model)

        for ch_idx, tag in REGIONS.items():
            gbp_saliency = gbp.generate(inputs, target_class=ch_idx)
            gbp_saliency = gbp_saliency[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

            # Save GBP NIfTI
            gbp_uint8 = (gbp_saliency * 255).astype(np.uint8)
            nifti_img = nib.Nifti1Image(gbp_uint8, affine)
            out_path = gbp_patient_dir / f"{patient_id}_gbp_{tag}.nii.gz"
            nib.save(nifti_img, out_path)

            # ── GBP metrics ─────────────────────────────────────────────
            gt_channel = gt_label[0, ch_idx]  # (D, H, W)

            if gt_channel.sum() == 0:
                gbp_metric_rows.append({
                    "Patient": patient_id,
                    "Region": REGION_NAMES[ch_idx],
                    "Pointing_Game": "N/A",
                    "Saliency_Coverage": "N/A",
                    "Saliency_IoU": "N/A",
                    "Weighted_Dice": "N/A",
                })
            else:
                metrics = evaluate_saliency(
                    gbp_saliency, gt_channel, threshold=args.iou_threshold
                )
                gbp_metric_rows.append({
                    "Patient": patient_id,
                    "Region": REGION_NAMES[ch_idx],
                    "Pointing_Game": metrics["pointing_game"],
                    "Saliency_Coverage": round(metrics["coverage"], 4),
                    "Saliency_IoU": round(metrics["iou"], 4),
                    "Weighted_Dice": round(metrics["weighted_dice"], 4),
                })

            # ════════════════════════════════════════════════════════════
            # STEP 3: Guided Grad-CAM = GBP × Grad-CAM
            # ════════════════════════════════════════════════════════════
            if args.guided_gradcam and ch_idx in gradcam_heatmaps:
                guided_gradcam = gbp_saliency * gradcam_heatmaps[ch_idx]

                # Re-normalise to [0, 1]
                gg_min = guided_gradcam.min()
                gg_max = guided_gradcam.max()
                if gg_max - gg_min > 1e-8:
                    guided_gradcam = (guided_gradcam - gg_min) / (gg_max - gg_min)

                # Save Guided Grad-CAM NIfTI
                gg_uint8 = (guided_gradcam * 255).astype(np.uint8)
                gg_nifti = nib.Nifti1Image(gg_uint8, affine)
                gg_path = ggcam_patient_dir / \
                    f"{patient_id}_guided_gradcam_{tag}.nii.gz"
                nib.save(gg_nifti, gg_path)

                # ── Guided Grad-CAM metrics ─────────────────────────────
                if gt_channel.sum() == 0:
                    ggcam_metric_rows.append({
                        "Patient": patient_id,
                        "Region": REGION_NAMES[ch_idx],
                        "Pointing_Game": "N/A",
                        "Saliency_Coverage": "N/A",
                        "Saliency_IoU": "N/A",
                        "Weighted_Dice": "N/A",
                    })
                else:
                    gg_metrics = evaluate_saliency(
                        guided_gradcam, gt_channel,
                        threshold=args.iou_threshold
                    )
                    ggcam_metric_rows.append({
                        "Patient": patient_id,
                        "Region": REGION_NAMES[ch_idx],
                        "Pointing_Game": gg_metrics["pointing_game"],
                        "Saliency_Coverage": round(gg_metrics["coverage"], 4),
                        "Saliency_IoU": round(gg_metrics["iou"], 4),
                        "Weighted_Dice": round(gg_metrics["weighted_dice"], 4),
                    })

                # ── Top-K thresholding on Guided Grad-CAM (if enabled) ──
                if do_topk:
                    gg_topk = topk_threshold(guided_gradcam, topk_pct)

                    # Save thresholded NIfTI
                    topk_patient_dir = topk_export_dir / patient_id
                    topk_patient_dir.mkdir(parents=True, exist_ok=True)
                    topk_uint8 = (gg_topk * 255).astype(np.uint8)
                    topk_nifti = nib.Nifti1Image(topk_uint8, affine)
                    topk_path = topk_patient_dir / \
                        f"{patient_id}_guided_gradcam_{topk_tag}_{tag}.nii.gz"
                    nib.save(topk_nifti, topk_path)

                    # Compute Weighted Dice on thresholded heatmap
                    if gt_channel.sum() == 0:
                        topk_rows.append({
                            "Patient": patient_id,
                            "Region": REGION_NAMES[ch_idx],
                            "TopK_Percent": topk_pct,
                            "Weighted_Dice": "N/A",
                        })
                    else:
                        topk_metrics = evaluate_saliency(
                            gg_topk, gt_channel,
                            threshold=args.iou_threshold
                        )
                        topk_rows.append({
                            "Patient": patient_id,
                            "Region": REGION_NAMES[ch_idx],
                            "TopK_Percent": topk_pct,
                            "Weighted_Dice": round(
                                topk_metrics["weighted_dice"], 4
                            ),
                        })

        # Remove GBP hooks before next patient
        gbp.restore_relus()
        del gbp

        processed += 1
        msg = f"  ✓ {patient_id} — {methods_str} saliency saved"
        if do_topk:
            msg += f" + Top-{int(topk_pct)}% exports"
        tqdm.write(msg)

        # Clean up memory
        del inputs, raw_input, gt_label, gradcam_heatmaps
        torch.cuda.empty_cache()

    # ── Write GBP metrics CSV ───────────────────────────────────────────
    with open(gbp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(gbp_metric_rows)

    # ── Write Guided Grad-CAM metrics CSV (if applicable) ───────────────
    if args.guided_gradcam and ggcam_metric_rows:
        with open(ggcam_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(ggcam_metric_rows)

    # ── Write Top-K Guided Grad-CAM Weighted Dice CSV ───────────────────
    if do_topk and topk_rows:
        with open(topk_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=topk_fieldnames)
            writer.writeheader()
            writer.writerows(topk_rows)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\nCompleted: {processed} patients")
    print(f"   GBP volumes:  {gbp_export_dir}")
    print(f"   GBP metrics:  {gbp_csv_path}")
    if args.guided_gradcam:
        print(f"   G-GradCAM:    {ggcam_export_dir}")
        print(f"   G-GradCAM met:{ggcam_csv_path}")
    if do_topk:
        print(f"   Top-K exports:{topk_export_dir}")
        print(f"   Top-K metrics:{topk_csv_path}")

    # ── Summary stats ───────────────────────────────────────────────────
    for method_name, rows in [("GBP", gbp_metric_rows),
                               ("Guided Grad-CAM", ggcam_metric_rows)]:
        if not rows:
            continue

        print(f"\n{'=' * 65}")
        print(f"  SUMMARY: {method_name} — Saliency vs Ground Truth Alignment")
        print(f"{'=' * 65}")

        for ch_idx, region_name in REGION_NAMES.items():
            region_rows = [
                r for r in rows
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

        na_count = sum(1 for r in rows if r["Pointing_Game"] == "N/A")
        if na_count > 0:
            print(f"\n  Note: {na_count} region evaluations skipped (no tumor in GT)")

        print(f"\n{'=' * 65}")

    # ── Top-K summary ───────────────────────────────────────────────────
    if do_topk and topk_rows:
        print(f"\n{'=' * 65}")
        print(f"  SUMMARY: Guided Grad-CAM Top-{int(topk_pct)}% — Weighted Dice")
        print(f"{'=' * 65}")

        for ch_idx, region_name in REGION_NAMES.items():
            region_rows = [
                r for r in topk_rows
                if r["Region"] == region_name and r["Weighted_Dice"] != "N/A"
            ]
            if not region_rows:
                print(f"\n  {region_name}: No valid samples")
                continue

            n = len(region_rows)
            dice_mean = np.mean([r["Weighted_Dice"] for r in region_rows])
            dice_std = np.std([r["Weighted_Dice"] for r in region_rows])

            print(f"\n  {region_name} (n={n})")
            print(f"    Weighted Dice          : {dice_mean:.4f} ± {dice_std:.4f}")

        na_count = sum(1 for r in topk_rows if r["Weighted_Dice"] == "N/A")
        if na_count > 0:
            print(f"\n  Note: {na_count} region evaluations skipped (no tumor in GT)")

        print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
