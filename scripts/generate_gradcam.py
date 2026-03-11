#!/usr/bin/env python3
"""
generate_gradcam.py — Generate 3D Grad-CAM saliency volumes for test patients.

For each patient, produces three NIfTI heatmaps (WT, TC, ET) that can be
overlaid on MRI scans in 3D Slicer / SlicerVR.

Usage:
    python scripts/generate_gradcam.py \
        --config configs/full_training_segresnet.yaml \
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
        --limit 1
"""

import argparse
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
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    export_dir = Path(cfg["project"].get("project_root", os.getcwd())) / "slicer_export" / "XAI" / "Grad_CAM"
    export_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"\nGenerating Grad-CAM volumes → {export_dir}")
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

        # ── Get original NIfTI affine & shape for spatial alignment ──────
        # The dataloader applies transforms (CropForeground, SpatialPad)
        # that change the spatial dimensions. We need to save the Grad-CAM
        # in the ORIGINAL image space so it aligns with raw MRI in Slicer.
        original_nifti = nib.load(str(first_img_path))
        original_affine = original_nifti.affine
        original_shape = original_nifti.shape[:3]  # (D, H, W) or (H, W, D)

        # ── Prepare input (requires grad for Grad-CAM backward) ────────
        inputs = data["image"].to(device)
        inputs.requires_grad_(True)

        # ── Generate Grad-CAM for each region ───────────────────────────
        for ch_idx, tag in REGIONS.items():
            heatmap = gcam.generate(inputs, target_class=ch_idx, upsample=True)

            # Resample from preprocessed space → original image space
            # heatmap is (D', H', W') from preprocessed; resize to original
            heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32)
            heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,D',H',W')
            heatmap_resampled = F.interpolate(
                heatmap_tensor,
                size=original_shape,
                mode="trilinear",
                align_corners=False,
            )
            heatmap_np = heatmap_resampled.squeeze().numpy()

            # Scale to [0, 255] uint8 for compact NIfTI storage
            heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)

            nifti_img = nib.Nifti1Image(heatmap_uint8, original_affine)
            out_path = patient_export_dir / f"{patient_id}_gradcam_{tag}.nii.gz"
            nib.save(nifti_img, out_path)

        processed += 1
        tqdm.write(f"  ✓ {patient_id} — 3 Grad-CAM volumes saved")

        # Clean up memory
        del inputs
        torch.cuda.empty_cache()

    gcam.remove_hooks()

    print(f"\n✅ Completed: {processed} new + {skipped} skipped (already exported)")
    print(f"   Output: {export_dir}")
    print("   Each patient folder contains:")
    print("     <patient>_gradcam_wt.nii.gz  (Whole Tumor heatmap)")
    print("     <patient>_gradcam_tc.nii.gz  (Tumor Core heatmap)")
    print("     <patient>_gradcam_et.nii.gz  (Enhancing Tumor heatmap)")
    print("\n   Load the patient's MRI from data/BraTS2023-Training/ in 3D Slicer to overlay.")


if __name__ == "__main__":
    main()
