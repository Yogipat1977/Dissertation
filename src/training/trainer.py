"""
trainer.py — Training loop with validation, checkpointing, and W&B logging.

Supports:
  - Automatic Mixed Precision (AMP) for ~2× speedup and ~50% less VRAM
  - Best-model checkpointing (saves when val Dice improves)
  - Last-checkpoint saving (saves every epoch for crash recovery)
  - Resume from last checkpoint (restore model, optimizer, scheduler, epoch)
"""

import os
import torch
import wandb
from tqdm import tqdm

import monai
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from src.config import save_config


class Trainer:
    """Handles the full training lifecycle: train, validate, checkpoint."""

    def __init__(self, model, loaders, loss_fn, optimizer, scheduler, cfg, device):
        self.model = model
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

        self.epochs = cfg["training"]["epochs"]
        self.roi_size = cfg["data"]["roi_size"]
        self.run_dir = cfg["_run_dir"]

        self.post_trans = AsDiscrete(threshold=0.5)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.dice_regions = ["WT", "TC", "ET"]
        self.best_metric = -1
        self.start_epoch = 0
        self.best_model_path = os.path.join(self.run_dir, "best_model.pth")
        self.last_checkpoint_path = os.path.join(self.run_dir, "last_checkpoint.pth")

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda")
        self.use_amp = device.type == "cuda"

    def save_last_checkpoint(self, epoch: int):
        """Save a full training snapshot for resume capability."""
        # Unwrap DataParallel so checkpoints are portable
        state_dict = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
        }
        torch.save(checkpoint, self.last_checkpoint_path)

    def resume(self):
        """Load training state from last checkpoint. Call before fit().

        Returns True if checkpoint was loaded, False otherwise.
        """
        if not os.path.exists(self.last_checkpoint_path):
            print("[Resume] No checkpoint found — training from scratch.")
            return False

        print(f"[Resume] Loading checkpoint: {self.last_checkpoint_path}")
        checkpoint = torch.load(
            self.last_checkpoint_path, map_location=self.device, weights_only=True
        )

        # Restore model weights (handle DataParallel)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_metric = checkpoint["best_metric"]
        self.start_epoch = checkpoint["epoch"] + 1  # resume from next epoch

        print(f"[Resume] Resuming from epoch {self.start_epoch + 1}/{self.epochs}")
        print(f"[Resume] Best Dice so far: {self.best_metric:.4f}")
        return True

    def fit(self):
        """Run the full training loop."""
        # Save config alongside checkpoints
        save_config(self.cfg, self.run_dir)

        remaining = self.epochs - self.start_epoch
        total_pbar = tqdm(
            total=remaining,
            desc="Training",
            position=0,
            initial=0,
        )

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_epoch(epoch)
            val_results = self._validate_epoch(epoch)
            val_dice = val_results["mean"]

            self.scheduler.step()

            # Log everything in one call so charts share the same x-axis
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_dice_WT": val_results["WT"],
                "val_dice_TC": val_results["TC"],
                "val_dice_ET": val_results["ET"],
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            # Checkpoint best model
            if val_dice > self.best_metric:
                self.best_metric = val_dice
                # Unwrap DataParallel so checkpoints are portable
                state_dict = self.model.module.state_dict() \
                    if hasattr(self.model, "module") else self.model.state_dict()
                torch.save(state_dict, self.best_model_path)
                tqdm.write(f"  → New Best Dice: {self.best_metric:.4f}")

            # Save last checkpoint every epoch (for crash recovery)
            self.save_last_checkpoint(epoch)

            total_pbar.update(1)

        total_pbar.close()

        print(f"\nTraining complete. Best Dice: {self.best_metric:.4f}")
        print(f"Best model saved to: {self.best_model_path}")

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        epoch_loss = 0
        loader = self.loaders["train"]
        step_pbar = tqdm(
            loader, desc=f"↳ Epoch {epoch + 1}", position=1, leave=False
        )

        for batch_data in step_pbar:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            step_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            wandb.log({"batch_loss": loss.item()})

        return epoch_loss / len(loader)

    def _validate_epoch(self, epoch: int) -> dict:
        """Run validation with sliding window inference. Returns dict of Dice scores."""
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            for val_data in self.loaders["val"]:
                v_in = val_data["image"].to(self.device)
                v_lab = val_data["label"].to(self.device)
                v_out = monai.inferers.sliding_window_inference(
                    v_in, self.roi_size, 4, self.model
                )
                v_out = [self.post_trans(torch.sigmoid(i)) for i in v_out]
                self.dice_metric(y_pred=v_out, y=v_lab)

            results = self.dice_metric.aggregate()
            self.dice_metric.reset()

        return {
            region: results[i].item()
            for i, region in enumerate(self.dice_regions)
        } | {"mean": results.mean().item()}
