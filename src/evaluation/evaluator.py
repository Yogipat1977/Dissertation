"""
evaluator.py — Model evaluation and results reporting.

Metrics computed per BraTS region (WT, TC, ET):
  - Dice Score          : overlap-based similarity
  - Hausdorff95 (HD95)  : 95th-percentile surface distance (mm)
  - Sensitivity         : true positive rate (recall)
  - Specificity         : true negative rate
"""

import csv
import os

import torch
import wandb
from tqdm import tqdm

import monai
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

REGIONS = ["Whole Tumor", "Tumor Core", "Enhancing"]


def _compute_sensitivity_specificity(y_pred: torch.Tensor, y: torch.Tensor):
    """
    Compute per-channel sensitivity and specificity.

    Args:
        y_pred: binary predictions, shape (B, C, ...)
        y:      binary ground truth, shape (B, C, ...)

    Returns:
        sens: (C,) tensor — sensitivity per channel
        spec: (C,) tensor — specificity per channel
    """
    eps = 1e-8
    # Flatten spatial dims
    pred_flat = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)  # (B, C, N)
    true_flat = y.view(y.shape[0], y.shape[1], -1)                 # (B, C, N)

    tp = (pred_flat * true_flat).sum(-1)        # (B, C)
    fn = ((1 - pred_flat) * true_flat).sum(-1)  # (B, C)
    tn = ((1 - pred_flat) * (1 - true_flat)).sum(-1)  # (B, C)
    fp = (pred_flat * (1 - true_flat)).sum(-1)  # (B, C)

    sens = (tp / (tp + fn + eps)).mean(0)  # mean over batch -> (C,)
    spec = (tn / (tn + fp + eps)).mean(0)  # mean over batch -> (C,)

    return sens, spec


def evaluate_set(
    model: torch.nn.Module,
    loader,
    cfg: dict,
    device: torch.device,
    name: str = "Test",
) -> dict:
    """Evaluate a model on a single data split.

    Returns:
        dict with keys 'dice', 'hd95', 'sensitivity', 'specificity',
        each a list of per-region floats [WT, TC, ET].
    """
    model.eval()
    roi_size = cfg["data"]["roi_size"]
    post_trans = AsDiscrete(threshold=0.5)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="mean_batch"
    )
    iou_metric = MeanIoU(include_background=True, reduction="mean_batch")

    # Accumulators for sensitivity / specificity (averaged across batches)
    n_channels = cfg["model"]["out_channels"]
    sens_acc = torch.zeros(n_channels)
    spec_acc = torch.zeros(n_channels)
    n_batches = 0

    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for data in tqdm(loader, desc=f"Evaluating {name}"):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)

            outputs = monai.inferers.sliding_window_inference(
                inputs, roi_size, 1, model        # sw_batch_size=1 to avoid OOM
            )

            # Binarise predictions
            preds = torch.stack(
                [post_trans(torch.sigmoid(i)) for i in outputs]
            )  # (B, C, ...)

            dice_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)
            iou_metric(y_pred=preds, y=labels)

            s, sp = _compute_sensitivity_specificity(preds, labels)
            sens_acc += s.cpu()
            spec_acc += sp.cpu()
            n_batches += 1

            # Free GPU memory between patients to prevent accumulation
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

    dice_results = dice_metric.aggregate()
    hd95_results = hd95_metric.aggregate()
    iou_results  = iou_metric.aggregate()
    dice_metric.reset()
    hd95_metric.reset()
    iou_metric.reset()

    dice_list = [dice_results[i].item() for i in range(len(REGIONS))]
    hd95_list = [hd95_results[i].item() for i in range(len(REGIONS))]
    iou_list  = [iou_results[i].item()  for i in range(len(REGIONS))]
    sens_list = [(sens_acc[i] / n_batches).item() for i in range(len(REGIONS))]
    spec_list = [(spec_acc[i] / n_batches).item() for i in range(len(REGIONS))]

    return {
        "dice": dice_list,
        "hd95": hd95_list,
        "iou": iou_list,
        "sensitivity": sens_list,
        "specificity": spec_list,
    }


def run_evaluation(
    model: torch.nn.Module,
    loaders: dict,
    cfg: dict,
    device: torch.device,
):
    """Evaluate on val and test splits, print results, log to W&B, and save CSV."""
    print("\n--- Evaluation ---")

    val_res   = evaluate_set(model, loaders["val"],   cfg, device, "Val")
    test_res  = evaluate_set(model, loaders["test"],  cfg, device, "Test")

    metrics = ["dice", "hd95", "iou", "sensitivity", "specificity"]
    metric_labels = {
        "dice":        "Dice \u2191",
        "hd95":        "HD95 \u2193 (mm)",
        "iou":         "IoU (Jaccard) \u2191",
        "sensitivity": "Sensitivity \u2191",
        "specificity": "Specificity \u2191",
    }

    # \u2500\u2500 Console table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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

    # \u2500\u2500 W&B tables \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    log_payload = {}
    for metric in metrics:
        table = wandb.Table(columns=["Region", "Val", "Test"])
        for i, reg in enumerate(REGIONS):
            table.add_data(
                reg,
                val_res[metric][i],
                test_res[metric][i],
            )
        log_payload[f"Evaluation/{metric_labels[metric]}"] = table

    wandb.log(log_payload)

    # \u2500\u2500 Save CSV \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    results_dir = cfg["paths"]["results_dir"]
    run_name    = cfg["_run_name"]
    csv_path    = os.path.join(results_dir, f"{run_name}.csv")

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

    print(f"\nResults saved to: {csv_path}")
    return {"val": val_res, "test": test_res}
