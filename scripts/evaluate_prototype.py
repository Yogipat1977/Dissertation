#!/usr/bin/env python3
"""
evaluate_prototype.py — Evaluate the prototype_32_v1 model with all metrics.

Reproduces the exact data pipeline from prototype.py (sorted patients,
no shuffle, seed=0) and computes:
  - Dice Score
  - Hausdorff Distance 95 (HD95)
  - IoU (Jaccard)
  - Sensitivity (Recall / True Positive Rate)
  - Specificity (True Negative Rate)

Usage:
    python scripts/evaluate_prototype.py
"""

import csv
import os
from pathlib import Path

import torch
from tqdm import tqdm

import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    EnsureTyped, CropForegroundd, SpatialPadd, MapTransform, AsDiscrete,
)
from monai.data import PersistentDataset, DataLoader
from monai.utils import set_determinism
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
import argparse

# ── Settings ─────────────────────────
SEED = 0
DATA_DIR = Path("/home/yogipatel/Desktop/Dissertation/data/Prototype_Data")
RESULTS_DIR = Path("/home/yogipatel/Desktop/Dissertation/results/CSVs")
ROI_SIZE = (160, 160, 160)
REGIONS = ["Whole Tumor", "Tumor Core", "Enhancing"]


# ── Label transform (identical to prototype.py) ────────────────────────
class ConvertToMultiChannelBraTS2023d(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float()
        return d


# ── Data listing (sorted, NO shuffle — matches prototype.py) ──────────
def get_brats_data_list(data_path):
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    data_list = []
    for p_dir in patient_dirs:
        p_name = p_dir.name
        data_list.append({
            "image": [
                str(p_dir / f"{p_name}-t1n.nii.gz"),
                str(p_dir / f"{p_name}-t1c.nii.gz"),
                str(p_dir / f"{p_name}-t2w.nii.gz"),
                str(p_dir / f"{p_name}-t2f.nii.gz"),
            ],
            "label": str(p_dir / f"{p_name}-seg.nii.gz"),
        })
    return data_list


# ── Custom metrics: sensitivity & specificity ──────────────────────────
def compute_sensitivity_specificity(y_pred: torch.Tensor, y: torch.Tensor):
    """Per-channel sensitivity and specificity."""
    eps = 1e-8
    pred_flat = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)
    true_flat = y.view(y.shape[0], y.shape[1], -1)

    tp = (pred_flat * true_flat).sum(-1)
    fn = ((1 - pred_flat) * true_flat).sum(-1)
    tn = ((1 - pred_flat) * (1 - true_flat)).sum(-1)
    fp = (pred_flat * (1 - true_flat)).sum(-1)

    sens = (tp / (tp + fn + eps)).mean(0)
    spec = (tn / (tn + fp + eps)).mean(0)
    return sens, spec


# ── Evaluation function ───────────────────────────────────────────────
def evaluate_set(model, loader, device, name="Test"):
    """Evaluate model on a data split with all metrics."""
    model.eval()
    post_trans = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="mean_batch"
    )
    iou_metric = MeanIoU(include_background=True, reduction="mean_batch")

    n_channels = 3
    sens_acc = torch.zeros(n_channels)
    spec_acc = torch.zeros(n_channels)
    n_batches = 0

    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for data in tqdm(loader, desc=f"Evaluating {name}"):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)

            outputs = monai.inferers.sliding_window_inference(
                inputs, ROI_SIZE, 1, model
            )

            preds = torch.stack(
                [post_trans(torch.sigmoid(i)) for i in outputs]
            )

            dice_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)
            iou_metric(y_pred=preds, y=labels)

            s, sp = compute_sensitivity_specificity(preds, labels)
            sens_acc += s.cpu()
            spec_acc += sp.cpu()
            n_batches += 1

            del inputs, labels, outputs, preds
            if device.type == "cuda":
                torch.cuda.empty_cache()

    dice_results = dice_metric.aggregate()
    hd95_results = hd95_metric.aggregate()
    iou_results = iou_metric.aggregate()
    dice_metric.reset()
    hd95_metric.reset()
    iou_metric.reset()

    return {
        "dice":        [dice_results[i].item() for i in range(3)],
        "hd95":        [hd95_results[i].item() for i in range(3)],
        "iou":         [iou_results[i].item()  for i in range(3)],
        "sensitivity": [(sens_acc[i] / n_batches).item() for i in range(3)],
        "specificity": [(spec_acc[i] / n_batches).item() for i in range(3)],
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--out-csv", type=str, required=True, help="Name of the output CSV file")
    parser.add_argument("--init-filters", type=int, default=32, help="Features for SegResNet")
    args = parser.parse_args()

    set_determinism(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # --- Data (exact same pipeline as prototype.py) ---
    datalist = get_brats_data_list(DATA_DIR)

    val_test_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ConvertToMultiChannelBraTS2023d(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160]),
    ])

    # Same split as prototype.py: 32 train / 8 val / 5 test (sorted, no shuffle)
    train_files = datalist[:32]
    val_files = datalist[32:40]
    test_files = datalist[40:]

    print(f"\n  Train : {len(train_files)} patients")
    print(f"  Val   : {len(val_files)} patients")
    print(f"  Test  : {len(test_files)} patients")

    cache_dir = Path("/home/yogipatel/Desktop/Dissertation/prototype/persistent_cache")

    val_ds = PersistentDataset(data=val_files, transform=val_test_transform, cache_dir=cache_dir)
    test_ds = PersistentDataset(data=test_files, transform=val_test_transform, cache_dir=cache_dir)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # --- Model (try both upsample modes for MONAI version compatibility) ---
    print("\nLoading model...")
    model = None
    for upsample_mode in ["nontrainable", "deconv"]:
        try:
            m = SegResNet(
                spatial_dims=3,
                init_filters=args.init_filters,
                in_channels=4,
                out_channels=3,
                dropout_prob=0.1,
                upsample_mode=upsample_mode,
            ).to(device)
            m.load_state_dict(
                torch.load(args.checkpoint, map_location=device, weights_only=True)
            )
            model = m
            print(f"  ✓ Loaded with init_filters={args.init_filters}, upsample_mode='{upsample_mode}'")
            break
        except RuntimeError as e:
            print(f"  ✗ init_filters={args.init_filters}, upsample_mode='{upsample_mode}' failed")
            continue

    if model is None:
        raise RuntimeError("Could not load checkpoint with any upsample mode.")

    # --- Evaluate ---
    print("\n--- Evaluation ---")
    val_res = evaluate_set(model, val_loader, device, "Val")
    test_res = evaluate_set(model, test_loader, device, "Test")

    # --- Print results ---
    metrics = ["dice", "hd95", "iou", "sensitivity", "specificity"]
    metric_labels = {
        "dice":        "Dice ↑",
        "hd95":        "HD95 ↓ (mm)",
        "iou":         "IoU (Jaccard) ↑",
        "sensitivity": "Sensitivity ↑",
        "specificity": "Specificity ↑",
    }

    for metric in metrics:
        print(f"\n  {metric_labels[metric]}")
        print(f"  {'Region':<17} | {'Val':<8} | {'Test':<8}")
        print(f"  {'-'*40}")
        for i, reg in enumerate(REGIONS):
            print(
                f"  {reg:<17} | "
                f"{val_res[metric][i]:<8.4f} | "
                f"{test_res[metric][i]:<8.4f}"
            )

    # --- Save CSV ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / args.out_csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Region", "Val", "Test"])
        for metric in metrics:
            for i, reg in enumerate(REGIONS):
                writer.writerow([
                    metric,
                    reg,
                    val_res[metric][i],
                    test_res[metric][i],
                ])

    print(f"\n✓ Results saved to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
