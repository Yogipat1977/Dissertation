"""
evaluator.py — Model evaluation and results reporting.
"""

import csv
import os
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

import monai
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

REGIONS = ["Whole Tumor", "Tumor Core", "Enhancing"]


def evaluate_set(
    model: torch.nn.Module,
    loader,
    cfg: dict,
    device: torch.device,
    name: str = "Test",
) -> list[float]:
    """Evaluate a model on a single data split.

    Returns:
        List of per-region Dice scores [WT, TC, ET].
    """
    model.eval()
    roi_size = cfg["data"]["roi_size"]
    post_trans = AsDiscrete(threshold=0.5)
    metric = DiceMetric(include_background=True, reduction="mean_batch")

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {name}"):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            outputs = monai.inferers.sliding_window_inference(
                inputs, roi_size, 4, model
            )
            outputs = [post_trans(torch.sigmoid(i)) for i in outputs]
            metric(y_pred=outputs, y=labels)

    results = metric.aggregate()
    metric.reset()
    return [results[i].item() for i in range(len(REGIONS))]


def run_evaluation(
    model: torch.nn.Module,
    loaders: dict,
    cfg: dict,
    device: torch.device,
):
    """Evaluate on all splits, print results, log to W&B, and save CSV."""
    print("\n--- Evaluation ---")

    train_res = evaluate_set(model, loaders["train"], cfg, device, "Train")
    val_res = evaluate_set(model, loaders["val"], cfg, device, "Val")
    test_res = evaluate_set(model, loaders["test"], cfg, device, "Test")

    # Console table
    print(f"\n{'='*55}")
    print(f"  {'Region':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print(f"  {'-'*51}")
    for i, reg in enumerate(REGIONS):
        print(
            f"  {reg:<15} | {train_res[i]:.4f}   | {val_res[i]:.4f}   | {test_res[i]:.4f}"
        )
    print(f"{'='*55}")

    # W&B table
    table = wandb.Table(columns=["Region", "Train", "Val", "Test"])
    for i, reg in enumerate(REGIONS):
        table.add_data(reg, train_res[i], val_res[i], test_res[i])
    wandb.log({"Evaluation_Summary": table})

    # Save CSV
    results_dir = cfg["paths"]["results_dir"]
    run_name = cfg["_run_name"]
    csv_path = os.path.join(results_dir, f"{run_name}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Region", "Train", "Val", "Test"])
        for i, reg in enumerate(REGIONS):
            writer.writerow([reg, train_res[i], val_res[i], test_res[i]])

    print(f"\nResults saved to: {csv_path}")
    return {"train": train_res, "val": val_res, "test": test_res}
