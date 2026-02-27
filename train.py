#!/usr/bin/env python3
"""
train.py — Main entry point for training.

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

import torch
import wandb
from monai.utils import set_determinism

from src.config import load_config
from src.data.dataset import create_data_loaders
from src.models.factory import create_model, create_loss, create_optimizer
from src.training.trainer import Trainer
from src.evaluation.evaluator import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train a BraTS segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    # --- Load .env (WANDB_API_KEY, etc.) ---
    load_dotenv(Path(__file__).resolve().parent / ".env")

    # --- Load config ---
    cfg = load_config(args.config)
    print(f"Run name : {cfg['_run_name']}")
    print(f"Run dir  : {cfg['_run_dir']}")

    # --- Reproducibility ---
    set_determinism(seed=cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Device   : {device} ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")

    # --- Data ---
    print("\nLoading data...")
    loaders = create_data_loaders(cfg)

    # --- Model ---
    print("\nCreating model...")
    model = create_model(cfg, device)
    loss_fn = create_loss(cfg)
    optimizer, scheduler = create_optimizer(model, cfg)

    print(f"  Architecture : {cfg['model']['architecture']}")
    print(f"  Init filters : {cfg['model']['init_filters']}")
    print(f"  Dropout      : {cfg['model']['dropout_prob']}")
    print(f"  Loss         : {cfg['training']['loss']}")
    print(f"  LR           : {cfg['training']['learning_rate']}")
    print(f"  Epochs       : {cfg['training']['epochs']}")

    # --- Multi-GPU ---
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"  Multi-GPU    : DataParallel across {num_gpus} GPUs")

    # --- W&B ---
    wandb_cfg = cfg["wandb"]
    os.environ["WANDB_MODE"] = wandb_cfg["mode"]

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    elif wandb_cfg["mode"] == "online":
        print("[WARNING] WANDB_API_KEY not found in .env — "
              "online logging will fail. Add it to your .env file.")

    wandb.init(
        project=wandb_cfg["project"],
        name=cfg["_run_name"],
        config={k: v for k, v in cfg.items() if not k.startswith("_")},
    )
    wandb.watch(model, log_freq=100)

    # --- Train ---
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    trainer = Trainer(model, loaders, loss_fn, optimizer, scheduler, cfg, device)
    trainer.fit()

    # --- Evaluate ---
    print(f"\n{'='*60}")
    print("Running evaluation...")
    print(f"{'='*60}")

    # Load best checkpoint (unwrap DataParallel if needed)
    state_dict = torch.load(trainer.best_model_path, map_location=device, weights_only=True)
    eval_model = create_model(cfg, device)
    eval_model.load_state_dict(state_dict)
    run_evaluation(eval_model, loaders, cfg, device)

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
