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
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Generate a 3D occlusion sensitivity relevance map for all 3 classes simultaneously.

        Args:
            input_tensor: Model input, shape (1, C, D, H, W).
            batch_size:   (Deprecated, unused in sequential to save RAM)

        Returns:
            Numpy array (3, D, H, W) with values in [0, 1]. Size matches input spatial dims.
        """
        self.model.eval()
        device = input_tensor.device

        # Get spatial dimensions (1, C, D, H, W) -> (D, H, W)
        _, _, D, H, W = input_tensor.shape

        # 1. Base Unoccluded Score for all 3 classes [0, 1, 2]
        base_output = self.model(input_tensor)
        base_score_0 = base_output[0, 0].mean().item()
        base_score_1 = base_output[0, 1].mean().item()
        base_score_2 = base_output[0, 2].mean().item()

        # 2. Setup sliding window grid
        d_stride, h_stride, w_stride = self.strides
        d_win, h_win, w_win = self.window_sizes

        d_coords = list(range(0, D - d_win + 1, d_stride))
        if d_coords[-1] + d_win < D: d_coords.append(D - d_win)
            
        h_coords = list(range(0, H - h_win + 1, h_stride))
        if h_coords[-1] + h_win < H: h_coords.append(H - h_win)
            
        w_coords = list(range(0, W - w_win + 1, w_stride))
        if w_coords[-1] + w_win < W: w_coords.append(W - w_win)

        # Output heatmap corresponding to stride grid centers for 3 classes
        occlusion_map = torch.zeros(
            (3, len(d_coords), len(h_coords), len(w_coords)),
            device=device,
            dtype=torch.float32
        )

        # 3. Process occlusions sequentially to save RAM (no batching)
        # We modify the input_tensor in-place, forward pass, and then restore it.
        total_steps = len(d_coords) * len(h_coords) * len(w_coords)
        
        with tqdm(total=total_steps, desc="  Sliding", leave=False) as pbar:
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
                        
                        # Forward pass (batch size 1) evaluates all 3 classes at once
                        occ_output = self.model(input_tensor)
                        occ_0 = occ_output[0, 0].mean().item()
                        occ_1 = occ_output[0, 1].mean().item()
                        occ_2 = occ_output[0, 2].mean().item()
                        
                        # Record drop
                        occlusion_map[0, i_d, i_h, i_w] = base_score_0 - occ_0
                        occlusion_map[1, i_d, i_h, i_w] = base_score_1 - occ_1
                        occlusion_map[2, i_d, i_h, i_w] = base_score_2 - occ_2
                        
                        # Restore original patch
                        input_tensor[
                            0, :, d : d + d_win, h : h + h_win, w : w + w_win
                        ] = orig_patch
                        
                        pbar.update(1)

        # 4. Upsample strided map to full original input resolution
        # occlusion_map is currently (3, D', H', W'). Add dummy batch for F.interpolate -> (1, 3, D', H', W')
        occlusion_map = occlusion_map.unsqueeze(0)
        
        # Upsample via trilinear interpolation
        full_map = F.interpolate(
            occlusion_map,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze(0)  # Remove dummy batch dim -> (3, D, H, W)

        # 5. Only care about POSITIVE drops (occlusion hurt performance)
        full_map = torch.clamp(full_map, min=0)

        # 6. Normalize to [0, 1] per channel
        for c in range(3):
            m_min = full_map[c].min()
            m_max = full_map[c].max()
            if m_max - m_min > 1e-8:
                full_map[c] = (full_map[c] - m_min) / (m_max - m_min)
            else:
                full_map[c] = torch.zeros_like(full_map[c])

        return full_map.cpu().numpy()
