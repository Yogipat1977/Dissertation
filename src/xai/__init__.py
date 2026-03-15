"""
XAI (Explainable AI) module for brain tumor segmentation.

Provides post-hoc explanation techniques for 3D CNN models.
"""

from src.xai.grad_cam import GradCAM3D
from src.xai.guided_backprop import GuidedBackprop3D
from src.xai.lrp import LRP3D
from src.xai.occlusion import OcclusionSensitivity3D
from src.xai.metrics import evaluate_saliency, msr_accuracy, pointing_game, saliency_coverage, saliency_iou, weighted_dice

__all__ = [
    "GradCAM3D",
    "GuidedBackprop3D",
    "LRP3D",
    "OcclusionSensitivity3D",
    "evaluate_saliency",
    "msr_accuracy",
    "pointing_game",
    "saliency_coverage",
    "saliency_iou",
    "weighted_dice",
]
