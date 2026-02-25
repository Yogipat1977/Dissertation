#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation of a saved checkpoint.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint models/<run_name>/best_model.pth
"""

import argparse
import os

import torch
import wandb
from monai.utils import set_determinism

from src.config import load_config
from src.data.dataset import create_data_loaders
from src.models.factory import create_model
from src.evaluation.evaluator import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved BraTS model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)

    # --- Reproducibility ---
    set_determinism(seed=cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # --- Data ---
    print("\nLoading data...")
    loaders = create_data_loaders(cfg)

    # --- Model ---
    print("\nLoading model...")
    model = create_model(cfg, device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    print(f"  Loaded weights from: {args.checkpoint}")

    # --- W&B (disabled for standalone eval) ---
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=cfg["wandb"]["project"], name=f"eval_{cfg['_run_name']}")

    # --- Evaluate ---
    run_evaluation(model, loaders, cfg, device)

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
