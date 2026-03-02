"""
transforms.py — BraTS-specific transforms and preprocessing pipelines.
"""

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    NormalizeIntensityd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
)


class ConvertToMultiChannelBraTS2023d(MapTransform):
    """
    Groups raw BraTS labels (1, 2, 3) into clinical sub-regions:
        - Channel 0: Whole Tumor (WT) — labels 1, 2, 3
        - Channel 1: Tumor Core (TC) — labels 1, 3
        - Channel 2: Enhancing Tumor (ET) — label 3
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3
                )
            )
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float()
        return d


def get_train_transforms(cfg: dict) -> Compose:
    """Build the training transform pipeline from config."""
    roi_size = list(cfg["data"]["roi_size"])
    num_samples = cfg["data"]["num_samples"]

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ConvertToMultiChannelBraTS2023d(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
    ])


def get_val_transforms(cfg: dict) -> Compose:
    """Build the validation/test transform pipeline from config."""
    roi_size = list(cfg["data"]["roi_size"])

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ConvertToMultiChannelBraTS2023d(keys="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])
