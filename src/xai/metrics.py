"""
metrics.py — Quantitative evaluation of XAI saliency maps against ground truth.

Implements metrics that measure how well a model's attention (saliency)
aligns with actual tumor locations, answering:
    "Is the model looking at the right region for the right reasons?"

Metrics:
    - Pointing Game:     Does the peak saliency voxel fall inside the tumor?
    - Saliency Coverage: What fraction of total saliency mass is inside the tumor?
    - Saliency IoU:      Overlap between thresholded saliency and ground truth mask.

Reference:
    Zhang et al., "Top-Down Neural Attention by Excitation Backprop", IJCV 2018
    (Pointing Game metric)
"""

import numpy as np
from typing import Dict


def pointing_game(saliency: np.ndarray, ground_truth: np.ndarray) -> bool:
    """
    Pointing Game: does the peak saliency voxel fall inside the GT region?

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumor, 0 = background.

    Returns:
        True if the voxel with maximum saliency is inside the GT mask.
    """
    peak_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    return bool(ground_truth[peak_idx] == 1)


def msr_accuracy(saliency: np.ndarray, ground_truth: np.ndarray) -> bool:
    """
    Most Salient Region (MSR) Accuracy.

    Similar to Pointing Game, but conceptually used for perturbation methods
    like Occlusion Sensitivity, testing if the single local patch that caused
    the largest confidence drop actually falls inside the anomaly.

    Formula:
        MSR = 1 if G(argmax(S)) == 1 else 0

    Args:
        saliency:     3D array (D, H, W) of occlusion sensitivity drops.
        ground_truth: 3D binary array (D, H, W), 1 = tumor, 0 = background.

    Returns:
        True if the maximum sensitivity drop voxel is within the tumor.
    """
    return pointing_game(saliency, ground_truth)


def saliency_coverage(saliency: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Saliency Coverage: fraction of total saliency mass inside the GT region.

    High coverage (→ 1.0) means the model focuses its attention on the
    actual tumor.  Low coverage means the model is distracted by
    irrelevant areas.

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumor, 0 = background.

    Returns:
        Float in [0, 1].  1.0 = all saliency is inside the tumor.
    """
    total = saliency.sum()
    if total < 1e-8:
        return 0.0
    inside = saliency[ground_truth == 1].sum()
    return float(inside / total)


def saliency_iou(
    saliency: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Saliency IoU: intersection-over-union of thresholded saliency and GT mask.

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumor, 0 = background.
        threshold:    Saliency values above this are considered "active".

    Returns:
        Float in [0, 1].  1.0 = perfect overlap.
    """
    s_binary = (saliency >= threshold).astype(np.uint8)
    gt_binary = (ground_truth >= 1).astype(np.uint8)

    intersection = (s_binary & gt_binary).sum()
    union = (s_binary | gt_binary).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def weighted_dice(
    saliency: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Weighted (Soft) Dice between continuous saliency and binary GT mask.

    Unlike standard Dice (which compares two binary masks), this treats
    the saliency as soft membership values [0, 1] and computes overlap
    with the binary ground truth.  This rewards saliency maps that not
    only cover the tumor but also match its shape.

    Formula:
        Weighted_Dice = 2 * Σ(S · G) / (Σ S + Σ G)

    where S ∈ [0,1] is the normalised saliency and G ∈ {0,1} is the GT.

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumor, 0 = background.

    Returns:
        Float in [0, 1].  1.0 = perfect soft overlap.
    """
    numerator = 2.0 * (saliency * ground_truth).sum()
    denominator = saliency.sum() + ground_truth.sum()
    if denominator < 1e-8:
        return 0.0
    return float(numerator / denominator)


def evaluate_saliency(
    saliency: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all saliency-vs-GT metrics in one call.

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumor, 0 = background.
        threshold:    Threshold for binarising saliency (for IoU).

    Returns:
        Dict with keys: 'pointing_game', 'msr_accuracy', 'coverage', 'iou', 'weighted_dice'.
    """
    return {
        "pointing_game": float(pointing_game(saliency, ground_truth)),
        "msr_accuracy": float(msr_accuracy(saliency, ground_truth)),
        "coverage": saliency_coverage(saliency, ground_truth),
        "iou": saliency_iou(saliency, ground_truth, threshold),
        "weighted_dice": weighted_dice(saliency, ground_truth),
    }
