#!/usr/bin/env python3
"""
generate_mc_dropout.py — Generate MC Dropout uncertainty maps for test patients.

Runs N stochastic forward passes with dropout enabled to produce:
  - Mean prediction maps (stabilised ensemble-like segmentation)
  - Uncertainty maps (per-voxel variance)
  - Evaluation metrics CSV

Usage:
    python scripts/generate_mc_dropout.py \
      --config configs/full_training_segresnet.yaml \
      --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
      --num_iters 20 \
      --limit 5
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Add project root to path so we can import from src
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import monai
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import create_data_loaders
from src.models.factory import create_model
from src.xai.uncertainty import MCDropout3D
from src.xai.lrp import LRP3D
from src.xai.metrics import weighted_dice


def pad_to_multiple(tensor, multiple=16):
    """Pad spatial dims to next multiple of `multiple`."""
    _, _, d, h, w = tensor.shape
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if pd + ph + pw == 0:
        return tensor, (d, h, w)
    padded = F.pad(tensor, (0, pw, 0, ph, 0, pd))
    return padded, (d, h, w)


def compute_boundary_mask(gt_binary, dilation=3):
    """Create a boundary band mask by dilating the GT and subtracting the eroded GT."""
    gt_tensor = torch.from_numpy(gt_binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel_size = 2 * dilation + 1

    # Dilate (max pool)
    dilated = F.max_pool3d(gt_tensor, kernel_size=kernel_size, stride=1,
                           padding=dilation)
    # Erode (min via negation)
    eroded = -F.max_pool3d(-gt_tensor, kernel_size=kernel_size, stride=1,
                           padding=dilation)

    boundary = (dilated - eroded).squeeze().numpy()
    return (boundary > 0.5).astype(np.uint8)


def saliency_uncertainty_correlation(saliency: np.ndarray, uncertainty: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation between saliency (e.g., LRP) and uncertainty inside the target mask."""
    if mask.sum() < 2:
        return 0.0
    s_vals = saliency[mask == 1]
    u_vals = uncertainty[mask == 1]
    
    std_s = np.std(s_vals)
    std_u = np.std(u_vals)
    
    if std_s < 1e-8 or std_u < 1e-8:
        return 0.0
        
    corr = np.corrcoef(s_vals, u_vals)[0, 1]
    return float(corr)


def compute_mc_metrics(uncertainty, ground_truth, lrp_map=None, boundary_dilation=3):
    """
    Compute MC Dropout-specific evaluation metrics, incorporating LRP comparison.

    Returns dict with:
        - uar: Uncertainty Area Ratio (fraction of unc mass inside tumor)
        - boundary_uncertainty_ratio: fraction of uncertainty mass at GT boundary
        - mean_unc_inside: mean variance inside tumor
        - mean_unc_outside: mean variance outside tumor
        - weighted_dice_lrp: weighted dice of LRP map vs GT (if LRP is provided)
        - saliency_unc_correlation: correlation between LRP and Unc inside tumor
    """
    gt_binary = (ground_truth >= 0.5).astype(np.uint8)
    boundary = compute_boundary_mask(gt_binary, dilation=boundary_dilation)

    total_unc = uncertainty.sum()
    if total_unc < 1e-12:
        return {
            "uar": 0.0,
            "boundary_uncertainty_ratio": 0.0,
            "mean_unc_inside": 0.0,
            "mean_unc_outside": 0.0,
            "weighted_dice_lrp": 0.0,
            "saliency_unc_correlation": 0.0,
        }

    # Uncertainty Area Ratio (UAR)
    inside_mask = gt_binary == 1
    outside_mask = gt_binary == 0
    
    unc_inside = uncertainty[inside_mask].sum()
    uar = float(unc_inside / total_unc)

    # Boundary Uncertainty Ratio
    boundary_unc = uncertainty[boundary == 1].sum()
    boundary_ratio = float(boundary_unc / total_unc)

    # Mean uncertainty inside / outside tumor
    mean_inside = float(uncertainty[inside_mask].mean()) if inside_mask.sum() > 0 else 0.0
    mean_outside = float(uncertainty[outside_mask].mean()) if outside_mask.sum() > 0 else 0.0

    # LRP-based metrics
    wd_lrp = 0.0
    sal_unc_corr = 0.0
    
    if lrp_map is not None:
        wd_lrp = weighted_dice(lrp_map, gt_binary)
        sal_unc_corr = saliency_uncertainty_correlation(lrp_map, uncertainty, inside_mask)

    return {
        "uar": uar,
        "boundary_uncertainty_ratio": boundary_ratio,
        "mean_unc_inside": mean_inside,
        "mean_unc_outside": mean_outside,
        "weighted_dice_lrp": wd_lrp,
        "saliency_unc_correlation": sal_unc_corr,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MC Dropout uncertainty maps.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_iters", type=int, default=20, help="Number of MC forward passes")
    parser.add_argument("--limit", type=int, default=0, help="Limit patients (0=all)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--patient_ids", type=str, default=None, 
                        help="Comma-separated list of specific patient IDs to process "
                             "(e.g. 'BraTS-GLI-00291-000,BraTS-GLI-00002-000')")
    args = parser.parse_args()

    # ---------- Setup ----------
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"MC iterations: {args.num_iters}")

    # ---------- Model ----------
    model = create_model(cfg, device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---------- Data ----------
    cfg["data"]["num_workers"] = 0
    loaders = create_data_loaders(cfg)
    loader = loaders[args.split]
    dataset = loader.dataset

    # Filter by specific patient IDs if provided
    if args.patient_ids:
        target_ids = [pid.strip() for pid in args.patient_ids.split(",")]
        filtered_dataset = []
        for sample in dataset:
            # Reusing the robust extraction logic
            if "image_meta_dict" in sample and "filename_or_obj" in sample["image_meta_dict"]:
                img_paths = sample["image_meta_dict"]["filename_or_obj"]
            elif isinstance(sample["image"], monai.data.MetaTensor) and \
                    "filename_or_obj" in sample["image"].meta:
                img_paths = sample["image"].meta["filename_or_obj"]
            else:
                continue

            if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0 and isinstance(img_paths[0], (list, tuple)):
                first_img_path = Path(img_paths[0][0])
            elif isinstance(img_paths, (list, tuple)):
                first_img_path = Path(img_paths[0])
            else:
                first_img_path = Path(img_paths)
            
            patient_name = first_img_path.parent.name
            if patient_name in target_ids:
                filtered_dataset.append(sample)
                
        dataset = filtered_dataset
        print(f"\nFiltered to {len(dataset)} specific requested patients.")
        if len(dataset) < len(target_ids):
            print("⚠️ Warning: Some requested patient IDs were not found in the split!")

    n_patients = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    print(f"\nProcessing {n_patients} patients from '{args.split}' split\n")

    # ---------- Output dirs ----------
    project_root = Path(cfg.get("_project_root", "."))
    export_base = project_root / "slicer_export" / "XAI" / "MC_Dropout"
    csv_path = project_root / "results" / "CSVs" / "xai_mc_dropout_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    region_names = ["wt", "tc", "et"]
    region_labels = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]

    # ---------- Generators ----------
    mcd = MCDropout3D(model, num_iters=args.num_iters)
    lrp_gen = LRP3D(model)

    csv_rows = []

    for idx in tqdm(range(n_patients), desc="MC Dropout + LRP"):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)  # (1, 4, D, H, W)
        label = sample["label"]  # (3, D, H, W)

        # Extract patient name
        if "image_meta_dict" in sample and "filename_or_obj" in sample["image_meta_dict"]:
            img_paths = sample["image_meta_dict"]["filename_or_obj"]
        elif isinstance(sample["image"], monai.data.MetaTensor) and \
                "filename_or_obj" in sample["image"].meta:
            img_paths = sample["image"].meta["filename_or_obj"]
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
        patient_name = first_img_path.parent.name

        # Pad input to multiple of 16
        image_padded, orig_shape = pad_to_multiple(image, 16)

        # Generate MC Dropout outputs
        mean_pred, uncertainty = mcd.generate(image_padded)

        # Crop back to original shape
        d, h, w = orig_shape
        mean_pred = mean_pred[:, :d, :h, :w]
        uncertainty = uncertainty[:, :d, :h, :w]

        # ── Get the transform-aware affine for spatial alignment ────────
        if isinstance(sample["image"], monai.data.MetaTensor):
            affine = sample["image"].meta.get("affine", None)
            if affine is not None:
                affine = np.array(affine)
                if affine.ndim == 3:
                    affine = affine[0]
            else:
                affine = np.eye(4)
        elif "image_meta_dict" in sample and "affine" in sample["image_meta_dict"]:
            affine = np.array(sample["image_meta_dict"]["affine"])
            if affine.ndim == 3:
                affine = affine[0]
        else:
            affine = np.eye(4)

        # --- Export NIfTI ---
        patient_dir = export_base / patient_name
        patient_dir.mkdir(parents=True, exist_ok=True)
        for ch, rname in enumerate(region_names):
            # Mean prediction
            mean_nii = nib.Nifti1Image(mean_pred[ch], affine)
            nib.save(mean_nii, str(patient_dir / f"{patient_name}_mc_mean_{rname}.nii.gz"))

            # Uncertainty map
            unc_nii = nib.Nifti1Image(uncertainty[ch], affine)
            nib.save(unc_nii, str(patient_dir / f"{patient_name}_mc_uncertainty_{rname}.nii.gz"))

        # --- Compute metrics per region ---
        label_np = label.numpy()
        for ch, (rname, rlabel) in enumerate(zip(region_names, region_labels)):
            gt_ch = label_np[ch]
            unc_ch = uncertainty[ch]

            # Generate LRP for this region to compute correlation and LRP Weighted Dice
            lrp_padded = lrp_gen.generate(image_padded, target_class=ch)
            lrp_ch = lrp_padded[:d, :h, :w]

            metrics = compute_mc_metrics(unc_ch, gt_ch, lrp_map=lrp_ch)
            
            csv_rows.append({
                "Patient": patient_name,
                "Region": rlabel,
                "UAR": f"{metrics['uar']:.4f}",
                "Boundary_Uncertainty_Ratio": f"{metrics['boundary_uncertainty_ratio']:.4f}",
                "Mean_Unc_Inside": f"{metrics['mean_unc_inside']:.6f}",
                "Mean_Unc_Outside": f"{metrics['mean_unc_outside']:.6f}",
                "Weighted_Dice_LRP": f"{metrics['weighted_dice_lrp']:.4f}",
                "Saliency_Unc_Correlation": f"{metrics['saliency_unc_correlation']:.4f}",
            })

        tqdm.write(f"  {patient_name} ✓")

    # --- Save CSV ---
    fieldnames = [
        "Patient", "Region", "UAR", "Boundary_Uncertainty_Ratio",
        "Mean_Unc_Inside", "Mean_Unc_Outside", 
        "Weighted_Dice_LRP", "Saliency_Unc_Correlation"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✅ Metrics saved to: {csv_path}")
    print(f"✅ NIfTI exports in: {export_base}/")


if __name__ == "__main__":
    main()
