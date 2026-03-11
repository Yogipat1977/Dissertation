"""
XAI (Explainable AI) module for brain tumor segmentation.

Provides post-hoc explanation techniques for 3D CNN models.
"""

from src.xai.grad_cam import GradCAM3D

__all__ = ["GradCAM3D"]
