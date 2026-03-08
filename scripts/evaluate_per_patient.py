#!/usr/bin/env python3
"""
evaluate_per_patient.py — Evaluate a saved BraTS model on the test set,
outputting metrics for *each* patient individually rather than just the split aggregate.

Outputs:
  1. A detailed CSV with (Patient, Region, Dice, HD95, IoU, Sensitivity, Specificity)
  2. A summary JSON/txt file for inclusion in the final report.

Usage:
    python scripts/evaluate_per_patient.py --config configs/full_training_segresnet.yaml --checkpoint models/<run_name>/best_model.pth --split test
"""

import argparse
import csv
import os
import json
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

import monai
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.utils import set_determinism

import sys
# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.dataset import create_data_loaders
from src.models.factory import create_model

REGIONS = ["Whole Tumor", "Tumor Core", "Enhancing"]

def _compute_sensitivity_specificity(y_pred: torch.Tensor, y: torch.Tensor):
    """
    Compute per-channel sensitivity and specificity.
    y_pred: (1, C, ...)
    y:      (1, C, ...)
    Returns: sens (C,), spec (C,)
    """
    eps = 1e-8
    pred_flat = y_pred.view(1, y_pred.shape[1], -1)  # (1, C, N)
    true_flat = y.view(1, y.shape[1], -1)            # (1, C, N)

    tp = (pred_flat * true_flat).sum(-1).squeeze(0)        # (C,)
    fn = ((1 - pred_flat) * true_flat).sum(-1).squeeze(0)  # (C,)
    tn = ((1 - pred_flat) * (1 - true_flat)).sum(-1).squeeze(0)  # (C,)
    fp = (pred_flat * (1 - true_flat)).sum(-1).squeeze(0)  # (C,)

    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)

    return sens, spec

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model per-patient on the test set.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"], help="Which dataset split to evaluate (default: test)")
    parser.add_argument("--output_name", type=str, default="per_patient_metrics.csv", help="Name of output CSV in the results directory.")
    args = parser.parse_args()

    # Setup
    cfg = load_config(args.config)
    set_determinism(seed=cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    
    # Data
    print(f"\nLoading data (evaluating split: {args.split})...")
    loaders = create_data_loaders(cfg)
    loader = loaders[args.split]
    
    # Must enforce batch_size=1 so we process exactly one patient at a time
    if loader.batch_size != 1:
        print("Warning: Forced batch_size to 1 for per-patient evaluation.")
        # Just manually recreating a 1-batch loader based on the dataset
        loader = monai.data.DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    # Model
    print("\nLoading model...")
    model = create_model(cfg, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded weights from: {args.checkpoint}")

    # Metrics
    roi_size = cfg["data"]["roi_size"]
    post_trans = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=True, reduction="mean_batch")

    use_amp = device.type == "cuda"
    
    # Store results: list of dicts
    all_results = []
    
    print("\nStarting per-patient evaluation...")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for data in tqdm(loader, desc=f"Evaluating {args.split}"):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            
            # Extract patient ID from the file path
            # Depending on MONAI version and Dataset type, metadata varies.
            if "label_meta_dict" in data and "filename_or_obj" in data["label_meta_dict"]:
                label_path = data["label_meta_dict"]["filename_or_obj"][0]
            elif isinstance(data["label"], monai.data.MetaTensor) and "filename_or_obj" in data["label"].meta:
                label_path = data["label"].meta["filename_or_obj"]
            else:
                # Fallback: using a counter for patient ID if metadata is entirely lost
                label_path = f"UnknownPatient_{len(all_results) // 3}"
                
            if isinstance(label_path, list):
                label_path = label_path[0]
                
            patient_id = Path(label_path).parent.name

            outputs = monai.inferers.sliding_window_inference(inputs, roi_size, 4, model)

            # Binarize predictions
            preds = torch.stack([post_trans(torch.sigmoid(i)) for i in outputs])  # (B, C, ...)

            # Calculate standard metrics for this single patient
            dice_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)
            iou_metric(y_pred=preds, y=labels)

            # Extract the raw tensors
            d_tensor = dice_metric.get_buffer()[-1] # shape (C)
            h_tensor = hd95_metric.get_buffer()[-1]
            i_tensor = iou_metric.get_buffer()[-1]
            
            # Calculate sens/spec manually for this patient
            s_tensor, sp_tensor = _compute_sensitivity_specificity(preds, labels)

            for c, region in enumerate(REGIONS):
                all_results.append({
                    "Patient": patient_id,
                    "Region": region,
                    "Dice": round(d_tensor[c].item(), 4),
                    "HD95": round(h_tensor[c].item(), 4),
                    "IoU": round(i_tensor[c].item(), 4),
                    "Sensitivity": round(s_tensor[c].item(), 4),
                    "Specificity": round(sp_tensor[c].item(), 4)
                })
                
            # Free memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

    # Aggregate summaries before resetting metrics, just to verify it matches
    print("\n--- Final Aggregate Consistency Check ---")
    d_agg = dice_metric.aggregate()
    h_agg = hd95_metric.aggregate()
    for c, region in enumerate(REGIONS):
        print(f"  {region:<15}: Dice {d_agg[c].item():.4f} | HD95 {h_agg[c].item():.4f}")

    # Output to CSV
    results_dir = Path(cfg["paths"]["results_dir"]) / "CSVs"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / args.output_name
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed per-patient metrics saved to: {csv_path}")
    
    # Calculate Summary Statistics for the Quarto document
    summary_stats = []
    grouped = df.groupby("Region")
    for region in REGIONS:
        group = grouped.get_group(region)
        summary_stats.append({
            "Region": region,
            "Dice_Mean": group["Dice"].mean(),
            "Dice_Std": group["Dice"].std(),
            "HD95_Mean": group["HD95"].mean(),
            "HD95_Std": group["HD95"].std(),
            "IoU_Mean": group["IoU"].mean(),
            "IoU_Std": group["IoU"].std(),
            "Sensitivity_Mean": group["Sensitivity"].mean(),
            "Sensitivity_Std": group["Sensitivity"].std(),
            "Specificity_Mean": group["Specificity"].mean(),
            "Specificity_Std": group["Specificity"].std(),
        })
        
    summary_df = pd.DataFrame(summary_stats)
    summary_path = results_dir / f"summary_{args.output_name}"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to: {summary_path}")

if __name__ == "__main__":
    main()
