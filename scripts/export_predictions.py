#!/usr/bin/env python3
"""
export_predictions.py — Generate and save segmentation predictions for Slicer/VR.

Reads the test set, runs inference with sliding window, reconstructs the
discrete segmentation mask, and exports the prediction alongside the
ground truth and structural MRI to a flat folder structure for 3D Slicer.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

import monai
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

import sys
# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.dataset import create_data_loaders
from src.models.factory import create_model

def get_affine_from_meta(meta_dict):
    """
    Attempt to extract the affine matrix from a MONAI meta dictionary.
    Returns a 4x4 identity matrix if it cannot be found.
    """
    if "affine" in meta_dict:
        return np.array(meta_dict["affine"])
    return np.eye(4)

def convert_multichannel_to_discrete(pred_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a multi-channel boolean prediction (WT, TC, ET) back into a 
    single volume with BraTS discrete labels (1=NCR/NET, 2=ED, 3=ET).
    
    Channels assumed:
    0: WT (labels 1, 2, 3)
    1: TC (labels 1, 3)
    2: ET (label 3)
    
    Inverse Logic:
    If ET==1 -> label 3
    Else If TC==1 -> label 1
    Else If WT==1 -> label 2
    Else -> 0 (Background)
    
    Input: (3, H, W, D) Boolean or 0/1 tensor
    Output: (H, W, D) numpy array with values 0, 1, 2, 3
    """
    wt = pred_tensor[0] > 0.5
    tc = pred_tensor[1] > 0.5
    et = pred_tensor[2] > 0.5
    
    # Initialize background
    discrete_mask = torch.zeros_like(wt, dtype=torch.uint8)
    
    # Apply conditions in reverse order of precedence
    # If it's part of the Whole Tumor, it's Edema (2) by default
    discrete_mask[wt] = 2
    
    # If it's part of the Tumor Core, it's Necrotic Core (1)
    discrete_mask[tc] = 1
    
    # If it's Enhancing Tumor, it's (3)
    discrete_mask[et] = 3
    
    return discrete_mask.cpu().numpy()

def _patient_fully_exported(patient_export_dir: Path, patient_id: str) -> bool:
    """Return True if the prediction NIfTI already exists."""
    return (patient_export_dir / f"{patient_id}_pred.nii.gz").exists()

def main():
    parser = argparse.ArgumentParser(description="Export test predictions to NIfTI for 3D Slicer.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N patients (0 for all)")
    parser.add_argument("--split", type=str, default="test", help="Which subset to run on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create root export directory
    export_dir = Path(cfg["project"].get("project_root", os.getcwd())) / "slicer_export" / "Model_Test_Results"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Loading Data
    print(f"\nLoading data (split: {args.split})...")
    loaders = create_data_loaders(cfg)
    loader = loaders[args.split]
    
    # Ensure test batch size is 1 to track individual files
    if loader.batch_size != 1:
        loader = monai.data.DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    # Loading Model
    print("\nLoading model...")
    model = create_model(cfg, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    roi_size = cfg["data"]["roi_size"]
    post_trans = AsDiscrete(threshold=0.5)
    use_amp = device.type == "cuda"
    
    processed = 0
    skipped = 0

    print(f"\nExporting test predictions to: {export_dir}")
    print(f"Checking for already-exported patients to resume...")
    
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for data in tqdm(loader, desc="Generating NIfTIs"):
            
            if args.limit > 0 and processed >= args.limit:
                break
                
            inputs = data["image"].to(device)
            
            # Extract patient ID and original file paths
            if "image_meta_dict" in data and "filename_or_obj" in data["image_meta_dict"]:
                img_paths = data["image_meta_dict"]["filename_or_obj"]
            elif isinstance(data["image"], monai.data.MetaTensor) and "filename_or_obj" in data["image"].meta:
                img_paths = data["image"].meta["filename_or_obj"]
            else:
                print("Warning: Could not find original filename in dataloader. Skipping.")
                continue
            
            if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0 and isinstance(img_paths[0], (list, tuple)):
                patient_modality_paths = img_paths[0]
            elif isinstance(img_paths, (list, tuple)):
                patient_modality_paths = img_paths
            else:
                patient_modality_paths = [img_paths]
                
            first_img_path = Path(patient_modality_paths[0])
            patient_id = first_img_path.parent.name
            
            # Create a dedicated export folder for this patient
            patient_export_dir = export_dir / patient_id
            patient_export_dir.mkdir(parents=True, exist_ok=True)
            
            # Resume check: skip if prediction already exists
            if _patient_fully_exported(patient_export_dir, patient_id):
                skipped += 1
                continue
            
            # 1. Inference
            outputs = monai.inferers.sliding_window_inference(inputs, roi_size, 4, model)
            
            # Binarize outputs: (B, C, H, W, D) → take first item → (3, H, W, D)
            pred_batch = torch.stack([post_trans(torch.sigmoid(i)) for i in outputs])
            pred_single = pred_batch[0]

            # 2. Convert multi-channel prediction to discrete labels (0, 1, 2, 3)
            #    Same format as seg.nii.gz so Slicer shows both identically
            #    Label 1 = NCR/NET, Label 2 = Edema, Label 3 = Enhancing Tumor
            discrete_pred = convert_multichannel_to_discrete(pred_single)
            
            # 3. Get affine from original image for spatial alignment
            if "image_meta_dict" in data:
                affine = get_affine_from_meta(data["image_meta_dict"])
            elif isinstance(data["image"], monai.data.MetaTensor):
                affine = get_affine_from_meta(data["image"].meta)
            else:
                affine = np.eye(4)
            if len(affine.shape) == 3 and affine.shape[0] == 1:
                affine = affine[0]

            # 4. Save prediction as discrete label NIfTI (same format as seg.nii.gz)
            pred_nifti = nib.Nifti1Image(discrete_pred, affine)
            nib.save(pred_nifti, patient_export_dir / f"{patient_id}_pred.nii.gz")
            
            # 5. Copy source structural scans and ground truth
            source_dir = first_img_path.parent
            for file in source_dir.glob("*.nii.gz"):
                dst_file = patient_export_dir / file.name
                if not dst_file.exists():
                    shutil.copy2(file, dst_file)
            
            processed += 1
            
            # Clean up memory
            del inputs, outputs, pred_batch, pred_single, discrete_pred
            torch.cuda.empty_cache()

    print(f"\n Completed: {processed} new + {skipped} skipped (already exported)")
    print(f" Output: {export_dir}")
    print("These folders can now be dragged directly into 3D Slicer / SlicerVR.")

if __name__ == "__main__":
    main()
