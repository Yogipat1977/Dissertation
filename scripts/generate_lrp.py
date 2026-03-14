#!/usr/bin/env python3
"""
generate_lrp.py — Generate Layer-wise Relevance Propagation (LRP) volumes.

Produces three NIfTI relevance maps (WT, TC, ET) per patient using the
Input x Gradient proxy method (equivalent to epsilon-LRP in ReLU networks).
Exports maps for 3D Slicer / SlicerVR visualisation.

By request, this script ONLY generates NIfTIs and omits metric generation.

Usage:
    python scripts/generate_lrp.py \\
        --config configs/full_training_segresnet.yaml \\
        --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \\
        --limit 10
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
from src.xai.lrp import LRP3D

# BraTS regions
REGIONS = {0: "wt", 1: "tc", 2: "et"}

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
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to process")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(cfg["project"].get("project_root", os.getcwd()))
    lrp_export_dir = project_root / "slicer_export" / "XAI" / "LRP"
    lrp_export_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  Export directory → {lrp_export_dir}\n")

    for data in tqdm(loader, desc="LRP"):

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

        processed += 1
        tqdm.write(f"  ✓ {patient_id} — LRP relevance saved")

        del inputs, raw_input
        torch.cuda.empty_cache()

    print(f"\nCompleted: {processed} patients")
    print(f"LRP volumes exported to: {lrp_export_dir}")


if __name__ == "__main__":
    main()
