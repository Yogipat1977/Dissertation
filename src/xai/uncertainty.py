"""
uncertainty.py — MC Dropout for uncertainty quantification in 3D segmentation.

Implements Monte Carlo (MC) Dropout to estimate model uncertainty. By running
multiple forward passes with dropout layers enabled at inference time, we can
compute the mean prediction (a more stable segmentation) and the voxel-wise
variance (a proxy for predictive uncertainty).

Key insight from notes: No model retraining needed — SegResNet already has
dropout_prob=0.1 in its architecture. We just keep those layers active
during inference.

Reference:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", ICML 2016.

Usage:
    from src.xai.uncertainty import MCDropout3D

    mcd = MCDropout3D(model, num_iters=20)
    mean_pred, uncertainty = mcd.generate(input_tensor)
    mcd.restore()  # re-disable dropout
"""

import torch
import numpy as np


class MCDropout3D:
    """
    Monte Carlo (MC) Dropout for 3D segmentation uncertainty estimation.

    Enables dropout at inference time and runs N stochastic forward passes
    on the same input to obtain:
        - Mean prediction: a stabilised, ensemble-like segmentation mask
        - Uncertainty map: per-voxel variance across passes

    Args:
        model:      A trained 3D segmentation model with Dropout layers.
        num_iters:  Number of stochastic forward passes (default: 20).
    """

    def __init__(self, model: torch.nn.Module, num_iters: int = 20):
        self.model = model
        self.num_iters = num_iters

    def _enable_dropout(self):
        """
        Enable dropout layers during inference while keeping all other
        layers (BatchNorm, etc.) in eval mode.

        This is the core of MC Dropout: model.eval() disables dropout,
        so we selectively re-enable only Dropout modules.
        """
        self.model.eval()
        for module in self.model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    def _restore_eval(self):
        """Restore full eval mode (disable dropout again)."""
        self.model.eval()

    def generate(self, input_tensor: torch.Tensor):
        """
        Generate mean prediction and uncertainty map using MC Dropout.

        Process:
            1. Enable dropout at inference time
            2. Run N forward passes on the same input
            3. Apply sigmoid → probabilities ∈ [0, 1]
            4. Stack all N outputs
            5. Compute mean (stable prediction) and variance (uncertainty)

        Args:
            input_tensor: Model input, shape (1, C, D, H, W).

        Returns:
            Tuple of:
                mean_pred:       numpy array (3, D, H, W) — mean probabilities
                uncertainty_map: numpy array (3, D, H, W) — per-voxel variance
        """
        self._enable_dropout()

        outputs = []
        with torch.no_grad():
            for _ in range(self.num_iters):
                output = self.model(input_tensor)
                probs = torch.sigmoid(output)
                outputs.append(probs)

        # Stack → (N, 1, 3, D, H, W)
        stacked = torch.stack(outputs)

        # Mean prediction → (1, 3, D, H, W)
        mean_pred = torch.mean(stacked, dim=0)

        # Variance (uncertainty) → (1, 3, D, H, W)
        uncertainty = torch.var(stacked, dim=0)

        # Restore eval mode
        self._restore_eval()

        # Remove batch dim → (3, D, H, W)
        mean_np = mean_pred.squeeze(0).cpu().numpy()
        unc_np = uncertainty.squeeze(0).cpu().numpy()

        return mean_np, unc_np
