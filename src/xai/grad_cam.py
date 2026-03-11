"""
grad_cam.py — 3D Grad-CAM for volumetric segmentation models.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for 3D CNNs.
Produces a class-discriminative saliency volume by weighting feature-map
activations at a chosen convolutional layer with gradient-derived importance
weights.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", IJCV 2020.

Usage:
    from src.xai.grad_cam import GradCAM3D

    gcam = GradCAM3D(model, target_layer=model.down_layers[-1])
    heatmap = gcam.generate(input_tensor, target_class=2)  # ET channel
    gcam.remove_hooks()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM3D:
    """
    3D Grad-CAM for volumetric segmentation models.

    Hooks into a target convolutional layer, captures activations during the
    forward pass and gradients during the backward pass, then computes a
    class-discriminative 3D heatmap.

    Args:
        model:        A trained 3D segmentation model (e.g. SegResNet).
        target_layer: The nn.Module (conv layer/block) to hook into.
                      Typically the last encoder block (bottleneck).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Storage for hooked tensors
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ──────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        """Forward hook: store the layer's output activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: store the gradients flowing into this layer."""
        self._gradients = grad_output[0].detach()

    # ── Core Grad-CAM computation ───────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        upsample: bool = True,
    ) -> np.ndarray:
        """
        Generate a 3D Grad-CAM heatmap for a given class channel.

        Args:
            input_tensor:  Model input, shape (1, C, D, H, W).
            target_class:  Output channel index (0=WT, 1=TC, 2=ET).
            upsample:      If True, trilinearly upsample the heatmap to the
                           input spatial resolution. Otherwise return at
                           feature-map resolution.

        Returns:
            Numpy array (D, H, W) with values in [0, 1].
        """
        self.model.eval()

        # ── 1. Forward pass (activations are captured by the hook) ──────
        output = self.model(input_tensor)

        # ── 2. Compute scalar score y^c ─────────────────────────────────
        # For segmentation: spatially average the pre-sigmoid logits for the
        # target class channel to obtain a single scalar.
        score = output[0, target_class].mean()

        # ── 3. Backward pass (gradients are captured by the hook) ───────
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # ── 4. Importance weights: global-average-pool the gradients ────
        # _gradients shape: (1, K, D', H', W')
        # → alpha shape:    (1, K, 1, 1, 1)  for broadcasting
        alpha = self._gradients.mean(dim=[2, 3, 4], keepdim=True)

        # ── 5. Weighted combination of feature maps ─────────────────────
        # _activations shape: (1, K, D', H', W')
        cam = (alpha * self._activations).sum(dim=1, keepdim=True)  # (1, 1, D', H', W')

        # ── 6. ReLU — keep only positive contributions ──────────────────
        cam = F.relu(cam)

        # ── 7. Normalise to [0, 1] ──────────────────────────────────────
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # ── 8. Upsample to input spatial resolution ─────────────────────
        if upsample:
            spatial_size = input_tensor.shape[2:]  # (D, H, W)
            cam = F.interpolate(
                cam, size=spatial_size, mode="trilinear", align_corners=False
            )

        # Return as numpy: squeeze batch and channel dims → (D, H, W)
        return cam.squeeze().cpu().numpy()

    # ── Cleanup ─────────────────────────────────────────────────────────

    def remove_hooks(self):
        """Remove all registered hooks. Call this when done."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        """Safety net: remove hooks on garbage collection."""
        try:
            self.remove_hooks()
        except Exception:
            pass
