"""
lrp.py — 3D Layer-wise Relevance Propagation (LRP) proxy for volumetric models.

Implements the Input ✕ Gradient method, which is a mathematically solid
approximation for epsilon-LRP in networks composed primarily of ReLU 
activations (like SegResNet).

Unlike Guided Backpropagation, which filters gradients purely to find
"what could change the output", LRP distributes the actual prediction score
back to the input voxels to answer "what explicitly contributed to the output".
"""

import torch
import numpy as np


class LRP3D:
    """
    3D Layer-wise Relevance Propagation (LRP) via Input x Gradient proxy.

    Computes relevance maps by multiplying the raw input image with the
    gradient of the target class score with respect to the input image.
    This provides a mathematically sound approximation of epsilon-LRP
    for networks primarily using ReLU activations.

    Args:
        model: A trained 3D segmentation model (e.g. SegResNet).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a full-resolution LRP relevance map.

        Args:
            input_tensor: Model input, shape (1, C, D, H, W).
            target_class: Output channel index (0=WT, 1=TC, 2=ET).

        Returns:
            Numpy array (D, H, W) with values in [0, 1].
            Full input-space resolution relevance map.
        """
        self.model.eval()

        # Ensure input requires gradient for backprop to reach it
        input_tensor = input_tensor.detach().clone()
        input_tensor.requires_grad_(True)

        # ── 1. Forward pass ─────────────────────────────────────────────
        output = self.model(input_tensor)

        # ── 2. Compute scalar score ──────────────────────────────────────
        score = output[0, target_class].mean()

        # ── 3. Backward pass ────────────────────────────────────────────
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        score.backward(retain_graph=False)

        # ── 4. Extract the input gradient ───────────────────────────────
        grad = input_tensor.grad.detach()  # (1, C, D, H, W)

        # ── 5. Input x Gradient ─────────────────────────────────────────
        # Element-wise multiplication of gradient with original input
        relevance = grad * input_tensor.detach()

        # ── 6. Aggregate across input channels ──────────────────────────
        # Sum relevance across the 4 MRI modalities (T1, T1ce, T2, FLAIR)
        # to get total relevance per spatial voxel.
        relevance = relevance[0].sum(dim=0)  # (D, H, W)

        # We typically care about positive relevance (evidence FOR the class).
        # We clamp negative relevance to 0.
        relevance = torch.clamp(relevance, min=0)

        # ── 7. Normalise to [0, 1] ──────────────────────────────────────
        r_min = relevance.min()
        r_max = relevance.max()
        if r_max - r_min > 1e-8:
            relevance = (relevance - r_min) / (r_max - r_min)
        else:
            relevance = torch.zeros_like(relevance)

        return relevance.cpu().numpy()
