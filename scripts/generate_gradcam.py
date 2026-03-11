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


def get_affine_from_meta(meta_dict):
    """Extract the affine matrix from a MONAI meta dictionary."""
    if "affine" in meta_dict:
        return np.array(meta_dict["affine"])
    return np.eye(4)


def _get_target_layer(model, layer_name: str):
    """
    Resolve a target layer from the SegResNet model.

    SegResNet architecture (MONAI):
        model.down_layers  — list of encoder blocks [level1, level2, level3]
        model.down_samples — list of downsampling ops
        model.bottleneck   — the deepest encoder block (256 filters)
        model.up_layers    — list of decoder blocks
        model.up_samples   — list of upsampling ops

    Args:
        model:      The loaded SegResNet model.
        layer_name: One of 'bottleneck', 'encoder3', 'encoder2', 'encoder1',
                    'decoder1'.

    Returns:
        The nn.Module to hook into.
    """
    layer_map = {
        "bottleneck": "bottleneck",
        "encoder3": None,   # down_layers[-1]
        "encoder2": None,   # down_layers[-2]
        "encoder1": None,   # down_layers[0]
        "decoder1": None,   # up_layers[0]
    }

    if layer_name == "bottleneck":
        return model.bottleneck
    elif layer_name == "encoder3":
        return model.down_layers[-1]
    elif layer_name == "encoder2":
        return model.down_layers[-2]
    elif layer_name == "encoder1":
        return model.down_layers[0]
    elif layer_name == "decoder1":
        return model.up_layers[0]
    else:
        raise ValueError(
            f"Unknown layer '{layer_name}'. "
            f"Choose from: {list(layer_map.keys())}"
        )


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

    export_dir = Path(cfg["project"].get("project_root", os.getcwd())) / "slicer_export"
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

        # ── Get affine for NIfTI spatial alignment ──────────────────────
        if "image_meta_dict" in data:
            affine = get_affine_from_meta(data["image_meta_dict"])
        elif isinstance(data["image"], monai.data.MetaTensor):
            affine = get_affine_from_meta(data["image"].meta)
        else:
            affine = np.eye(4)
        if len(affine.shape) == 3 and affine.shape[0] == 1:
            affine = affine[0]

        # ── Prepare input (requires grad for Grad-CAM backward) ────────
        inputs = data["image"].to(device)
        inputs.requires_grad_(True)

        # ── Generate Grad-CAM for each region ───────────────────────────
        for ch_idx, tag in REGIONS.items():
            heatmap = gcam.generate(inputs, target_class=ch_idx, upsample=True)

            # Scale to [0, 255] uint8 for compact NIfTI storage
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)

            nifti_img = nib.Nifti1Image(heatmap_uint8, affine)
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
    print("   Each patient folder now contains:")
    print("     <patient>_gradcam_wt.nii.gz  (Whole Tumor)")
    print("     <patient>_gradcam_tc.nii.gz  (Tumor Core)")
    print("     <patient>_gradcam_et.nii.gz  (Enhancing Tumor)")
    print("\n   Load these in 3D Slicer as scalar overlay volumes.")


if __name__ == "__main__":
    main()
