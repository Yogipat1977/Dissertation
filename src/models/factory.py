"""
factory.py — Create model, loss function, optimizer, and scheduler from config.
"""

import torch
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Instantiate the segmentation model from config."""
    model_cfg = cfg["model"]
    arch = model_cfg["architecture"]

    if arch == "SegResNet":
        model = SegResNet(
            spatial_dims=model_cfg["spatial_dims"],
            init_filters=model_cfg["init_filters"],
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            dropout_prob=model_cfg["dropout_prob"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)


def create_loss(cfg: dict):
    """Instantiate the loss function from config."""
    loss_name = cfg["training"]["loss"]
    params = cfg["training"].get("loss_params", {})

    if loss_name == "DiceCELoss":
        return DiceCELoss(to_onehot_y=False, **params)
    elif loss_name == "DiceLoss":
        return DiceLoss(to_onehot_y=False, **params)
    elif loss_name == "DiceFocalLoss":
        return DiceFocalLoss(to_onehot_y=False, **params)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def create_optimizer(model: torch.nn.Module, cfg: dict):
    """Create optimizer and learning rate scheduler from config.

    Returns:
        (optimizer, scheduler) tuple
    """
    train_cfg = cfg["training"]

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler_name = train_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return optimizer, scheduler
