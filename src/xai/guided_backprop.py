"""
guided_backprop.py — 3D Guided Backpropagation for volumetric segmentation models.

Implements Guided Backpropagation (GBP), which modifies the standard gradient
backpropagation by additionally gating negative gradients at every ReLU layer.
This produces a full-resolution, pixel-level saliency map that highlights
fine-grained features the model uses.

Unlike Grad-CAM (which is coarse but class-discriminative), GBP produces
sharp, high-resolution maps but is NOT class-discriminative on its own.
Combining GBP with Grad-CAM yields Guided Grad-CAM (sharp + class-specific).

Reference:
    Springenberg et al., "Striving for Simplicity: The All Convolutional Net",
    ICLR Workshop 2015.

Usage:
    from src.xai.guided_backprop import GuidedBackprop3D

    gbp = GuidedBackprop3D(model)
    saliency = gbp.generate(input_tensor, target_class=0)  # WT channel
    gbp.restore_relus()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class GuidedBackprop3D:
    """
    3D Guided Backpropagation for volumetric segmentation models.

    Replaces all ReLU activations in the model with "Guided ReLUs" that
    gate both negative activations (standard ReLU) and negative gradients
    during the backward pass. This produces a full-resolution saliency map.

    Args:
        model: A trained 3D segmentation model (e.g. SegResNet).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._original_relus: List[Tuple[torch.nn.Module, str, torch.nn.Module]] = []

        # Replace all ReLU layers with guided versions
        self._override_relus(model)

    # ── ReLU Override ───────────────────────────────────────────────────

    def _guided_relu_hook(self, module, grad_input, grad_output):
        """
        Backward hook for Guided Backpropagation.

        Gates the gradient: only pass through gradients that are positive
        AND where the forward activation was also positive.
        The forward ReLU already zeroes negative activations, so this hook
        additionally zeroes negative incoming gradients.
        """
        # grad_output[0] is the gradient flowing backward through this ReLU
        # Clamp: only keep positive gradients
        return (F.relu(grad_output[0]),)

    def _override_relus(self, module: torch.nn.Module):
        """
        Recursively find all ReLU/PReLU layers and register backward hooks
        that implement the guided backprop gradient gating.

        Also disables inplace=True on ReLU layers to prevent conflicts
        when both GBP and Grad-CAM hooks are active on the same model
        (even though we isolate them per-patient, inplace ops can still
        cause issues with autograd).

        For MONAI SegResNet, ReLU activations are inside residual blocks
        as nn.ReLU or act layers.
        """
        for name, child in module.named_children():
            if isinstance(child, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.LeakyReLU)):
                # Disable inplace to avoid potential autograd conflicts
                if hasattr(child, 'inplace') and child.inplace:
                    self._original_relus.append((child, 'inplace', True))
                    child.inplace = False
                # Register backward hook on this ReLU
                hook = child.register_full_backward_hook(self._guided_relu_hook)
                self._hooks.append(hook)
            else:
                # Recurse into sub-modules
                self._override_relus(child)

    # ── Core GBP Computation ────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a full-resolution Guided Backpropagation saliency map.

        Args:
            input_tensor: Model input, shape (1, C, D, H, W).
            target_class: Output channel index (0=WT, 1=TC, 2=ET).

        Returns:
            Numpy array (D, H, W) with values in [0, 1].
            Full input-space resolution saliency map.
        """
        self.model.eval()

        # Ensure input requires gradient for backprop to reach it
        input_tensor = input_tensor.detach().clone()
        input_tensor.requires_grad_(True)

        # ── 1. Forward pass ─────────────────────────────────────────────
        output = self.model(input_tensor)

        # ── 2. Compute scalar score (same as Grad-CAM) ──────────────────
        score = output[0, target_class].mean()

        # ── 3. Backward pass with guided ReLU hooks active ──────────────
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        score.backward(retain_graph=False)

        # ── 4. Extract the input gradient ───────────────────────────────
        # The gradient at the input tells us how much each input voxel
        # contributed to the target class score.
        grad = input_tensor.grad.detach()  # (1, C, D, H, W)

        # ── 5. Aggregate across input channels ──────────────────────────
        # Take the max absolute gradient across all 4 MRI modalities
        # to get a single saliency map showing the most important voxels
        saliency = grad[0].abs().max(dim=0).values  # (D, H, W)

        # ── 6. Normalise to [0, 1] ──────────────────────────────────────
        s_min = saliency.min()
        s_max = saliency.max()
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)

        return saliency.cpu().numpy()

    # ── Cleanup ─────────────────────────────────────────────────────────

    def restore_relus(self):
        """Remove all registered backward hooks and restore inplace state."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        # Restore original inplace=True on ReLUs
        for module, attr, original_value in self._original_relus:
            setattr(module, attr, original_value)
        self._original_relus.clear()

    def __del__(self):
        """Safety net: remove hooks on garbage collection."""
        try:
            self.restore_relus()
        except Exception:
            pass
