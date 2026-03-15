"""
occlusion.py — 3D Occlusion Sensitivity for volumetric models.

Implements a sliding-window perturbation method to test model reliance
on specific spatial regions. Unlike gradient methods, this physically hides
portions of the input and records the drop in target prediction confidence.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class OcclusionSensitivity3D:
    """
    3D Occlusion Sensitivity (Sliding Window).

    Args:
        model:        A trained 3D segmentation model.
        window_size:  Size of the occluding patch (D, H, W).
        stride:       Step size for the sliding window (D, H, W).
        baseline:     Value to replace occluded pixels with (e.g. 0).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window_size: tuple = (16, 16, 16),
        stride: tuple = (8, 8, 8),
        baseline: float = 0.0,
    ):
        self.model = model
        self.window_sizes = window_size
        self.strides = stride
        self.baseline = baseline

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Generate a 3D occlusion sensitivity relevance map.

        Args:
            input_tensor: Model input, shape (1, C, D, H, W).
            target_class: Output channel index (0=WT, 1=TC, 2=ET).
            batch_size:   Number of occluded volumes to pass simultaneously.

        Returns:
            Numpy array (D, H, W) with values in [0, 1]. Size matches input.
        """
        self.model.eval()
        device = input_tensor.device

        # Get spatial dimensions (1, C, D, H, W) -> (D, H, W)
        _, _, D, H, W = input_tensor.shape

        # 1. Base Unoccluded Score
        base_output = self.model(input_tensor)
        base_score = base_output[0, target_class].mean().item()

        # 2. Setup sliding window grid
        d_stride, h_stride, w_stride = self.strides
        d_win, h_win, w_win = self.window_sizes

        d_coords = list(range(0, D - d_win + 1, d_stride))
        if d_coords[-1] + d_win < D: d_coords.append(D - d_win)
            
        h_coords = list(range(0, H - h_win + 1, h_stride))
        if h_coords[-1] + h_win < H: h_coords.append(H - h_win)
            
        w_coords = list(range(0, W - w_win + 1, w_stride))
        if w_coords[-1] + w_win < W: w_coords.append(W - w_win)

        # Output heatmap corresponding to stride grid centers
        occlusion_map = torch.zeros(
            (len(d_coords), len(h_coords), len(w_coords)),
            device=device,
            dtype=torch.float32
        )

        # 3. Process occlusions sequentially to save RAM (no batching)
        # We modify the input_tensor in-place, forward pass, and then restore it.
        for i_d, d in enumerate(d_coords):
            for i_h, h in enumerate(h_coords):
                for i_w, w in enumerate(w_coords):
                    
                    # Save the original patch
                    orig_patch = input_tensor[
                        0, :, d : d + d_win, h : h + h_win, w : w + w_win
                    ].clone()
                    
                    # Apply occlusion
                    input_tensor[
                        0, :, d : d + d_win, h : h + h_win, w : w + w_win
                    ] = self.baseline
                    
                    # Forward pass (batch size 1)
                    occ_output = self.model(input_tensor)
                    occ_score = occ_output[0, target_class].mean().item()
                    
                    # Record drop
                    drop = base_score - occ_score
                    occlusion_map[i_d, i_h, i_w] = drop
                    
                    # Restore original patch
                    input_tensor[
                        0, :, d : d + d_win, h : h + h_win, w : w + w_win
                    ] = orig_patch

        # 4. Upsample strided map to full original input resolution
        # occlusion_map is currently (D', H', W'). Add dummy batch+channel for F.interpolate
        occlusion_map = occlusion_map.unsqueeze(0).unsqueeze(0)
        
        # Upsample via trilinear interpolation
        full_map = F.interpolate(
            occlusion_map,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze()  # Remove dummy dims back to (D, H, W)

        # 5. Only care about POSITIVE drops (occlusion hurt performance)
        full_map = torch.clamp(full_map, min=0)

        # 6. Normalize to [0, 1]
        m_min = full_map.min()
        m_max = full_map.max()
        if m_max - m_min > 1e-8:
            full_map = (full_map - m_min) / (m_max - m_min)
        else:
            full_map = torch.zeros_like(full_map)

        return full_map.cpu().numpy()
