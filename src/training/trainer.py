"""
trainer.py — Training loop with validation, checkpointing, and W&B logging.
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
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.best_metric = -1
        self.best_model_path = os.path.join(self.run_dir, "best_model.pth")

    def fit(self):
        """Run the full training loop."""
        # Save config alongside checkpoints
        save_config(self.cfg, self.run_dir)

        total_pbar = tqdm(total=self.epochs, desc="Training", position=0)

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            val_dice = self._validate_epoch(epoch)

            self.scheduler.step()

            # Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_dice": val_dice,
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
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            step_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / len(loader)

    def _validate_epoch(self, epoch: int) -> float:
        """Run validation with sliding window inference. Returns mean Dice."""
        self.model.eval()
        with torch.no_grad():
            for val_data in self.loaders["val"]:
                v_in = val_data["image"].to(self.device)
                v_lab = val_data["label"].to(self.device)
                v_out = monai.inferers.sliding_window_inference(
                    v_in, self.roi_size, 4, self.model
                )
                v_out = [self.post_trans(torch.sigmoid(i)) for i in v_out]
                self.dice_metric(y_pred=v_out, y=v_lab)

            dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset()

        return dice
