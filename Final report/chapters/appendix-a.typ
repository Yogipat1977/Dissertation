#show raw.where(block: true): set text(size: 9pt)
= Appendices and Supplementary Materials

== Appendix A: Project Administration Forms

=== A.1 Initial Project Proposal
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../Figures/Proposal_page-0001.jpg", width: 100%),
  image("../Figures/Proposal_page-0002.jpg", width: 100%)
)

=== A.2 Final Project Proposal
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  image("../Figures/Final_Proposal_page-0001.jpg", width: 100%),
  image("../Figures/Final_Proposal_page-0002.jpg", width: 100%),
  image("../Figures/Final_Proposal_page-0003.jpg", width: 100%)
)

=== A.3 Internal Ethical Approval Process
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../Figures/Ethics_page-00.jpg", width: 100%),
  image("../Figures/Ethics_page-01.jpg", width: 100%),
  image("../Figures/Ethics_page-02.jpg", width: 100%),
  image("../Figures/Ethics_page-03.jpg", width: 100%),
  image("../Figures/Ethics_page-04.jpg", width: 100%)
)

== Appendix B: Model Configuration and Hyperparameters
This section details the complete configuration YAML used for training the SegResNet model on the BraTS 2023 dataset, ensuring reproducibility of the hyperparameters, optimiser settings, and loss functions.

#strong[Listing:] Full Training Configuration (`full_training_segresnet.yaml`)

```yaml
project:
  name: "BraTS-Dissertation-Full-SegResNet"
  seed: 42

data:
  data_dir: "data/BraTS2023-Training"       # full training dataset
  train_split: 1000
  val_split: 125                             # remaining ~125 go to test
  roi_size: [160, 160, 160]                  # 160³ - larger spatial context
  batch_size: 1                              # ~38 GB on single GPU with AMP
  num_workers: 8                             # keeps data pipeline saturated
  num_samples: 4                             # RandCropByPosNegLabeld samples

model:
  architecture: "SegResNet"
  spatial_dims: 3
  in_channels: 4
  out_channels: 3
  init_filters: 32
  dropout_prob: 0.1

training:
  epochs: 35                                 # ~50 epochs typical for full BraTS dataset
  learning_rate: 0.00005                      
  weight_decay: 0.00001
  scheduler: "cosine"
  loss: "DiceFocalLoss"
  loss_params:
    sigmoid: true
    gamma: 2.0                               # Focal factor for hard examples (ET)
```

== Appendix C: Volumetric Data Transformations
This snippet demonstrates the data preprocessing pipeline implemented in MONAI, including the custom `ConvertToMultiChannelBraTS2023d` transform that maps raw BraTS labels into the required clinical sub-regions.

#strong[Listing:] Data Transforms Pipeline (`transforms.py`)

```python
"""
transforms.py - BraTS-specific transforms and preprocessing pipelines.
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
        - Channel 0: Whole Tumour (WT) - labels 1, 2, 3
        - Channel 1: Tumour Core (TC) - labels 1, 3
        - Channel 2: Enhancing Tumour (ET) - label 3
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
```

== Appendix D: Training Framework
This section details the custom training loop implemented in PyTorch and MONAI, featuring Automatic Mixed Precision (AMP) for efficiency, Weights & Biases (W&B) integration for metric tracking, and robust checkpointing capabilities.

#strong[Listing:] Training Loop Implementation (`trainer.py`)

```python
"""
trainer.py - Training loop with validation, checkpointing, and W&B logging.

Supports:
  - Automatic Mixed Precision (AMP) for ~2× speedup and ~50% less VRAM
  - Best-model checkpointing (saves when val Dice improves)
  - Last-checkpoint saving (saves every epoch for crash recovery)
  - Resume from last checkpoint (restore model, optimiser, scheduler, epoch)
"""

import os
import torch
import wandb
from tqdm import tqdm

import monai
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from src.config import save_config


class Trainer:
    """Handles the full training lifecycle: train, validate, checkpoint."""

    def __init__(self, model, loaders, loss_fn, optimiser, scheduler, cfg, device):
        self.model = model
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

        self.epochs = cfg["training"]["epochs"]
        self.roi_size = cfg["data"]["roi_size"]
        self.run_dir = cfg["_run_dir"]

        self.post_trans = AsDiscrete(threshold=0.5)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.dice_regions = ["WT", "TC", "ET"]
        self.best_metric = -1
        self.start_epoch = 0
        self.best_model_path = os.path.join(self.run_dir, "best_model.pth")
        self.last_checkpoint_path = os.path.join(self.run_dir, "last_checkpoint.pth")

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda")
        self.use_amp = device.type == "cuda"

    def save_last_checkpoint(self, epoch: int):
        """Save a full training snapshot for resume capability."""
        # Unwrap DataParallel so checkpoints are portable
        state_dict = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
        }
        torch.save(checkpoint, self.last_checkpoint_path)

    def resume(self):
        """Load training state from last checkpoint. Call before fit().

        Returns True if checkpoint was loaded, False otherwise.
        """
        if not os.path.exists(self.last_checkpoint_path):
            print("[Resume] No checkpoint found - training from scratch.")
            return False

        print(f"[Resume] Loading checkpoint: {self.last_checkpoint_path}")
        checkpoint = torch.load(
            self.last_checkpoint_path, map_location=self.device, weights_only=True
        )

        # Restore model weights (handle DataParallel)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_metric = checkpoint["best_metric"]
        self.start_epoch = checkpoint["epoch"] + 1  # resume from next epoch

        print(f"[Resume] Resuming from epoch {self.start_epoch + 1}/{self.epochs}")
        print(f"[Resume] Best Dice so far: {self.best_metric:.4f}")
        return True

    def fit(self):
        """Run the full training loop."""
        # Save config alongside checkpoints
        save_config(self.cfg, self.run_dir)

        remaining = self.epochs - self.start_epoch
        total_pbar = tqdm(
            total=remaining,
            desc="Training",
            position=0,
            initial=0,
        )

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_epoch(epoch)
            val_results = self._validate_epoch(epoch)
            val_dice = val_results["mean"]

            self.scheduler.step()

            # Log everything in one call so charts share the same x-axis
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_dice_WT": val_results["WT"],
                "val_dice_TC": val_results["TC"],
                "val_dice_ET": val_results["ET"],
                "lr": self.optimiser.param_groups[0]["lr"],
            })

            # Checkpoint best model
            if val_dice > self.best_metric:
                self.best_metric = val_dice
                # Unwrap DataParallel so checkpoints are portable
                state_dict = self.model.module.state_dict() \
                    if hasattr(self.model, "module") else self.model.state_dict()
                torch.save(state_dict, self.best_model_path)
                tqdm.write(f"  → New Best Dice: {self.best_metric:.4f}")

            # Save last checkpoint every epoch (for crash recovery)
            self.save_last_checkpoint(epoch)

            total_pbar.update(1)

        total_pbar.close()

        print(f"\nTraining complete. Best Dice: {self.best_metric:.4f}")
        print(f"Best model saved to: {self.best_model_path}")

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        epoch_loss = 0
        loader = self.loaders["train"]
        step_pbar = tqdm(
            loader, desc=f"↳ Epoch {epoch + 1}", position=1, leave=False
        )

        for batch_data in step_pbar:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimiser.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()

            epoch_loss += loss.item()
            step_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            wandb.log({"batch_loss": loss.item()})

        return epoch_loss / len(loader)

    def _validate_epoch(self, epoch: int) -> dict:
        """Run validation with sliding window inference. Returns dict of Dice scores."""
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            for val_data in self.loaders["val"]:
                v_in = val_data["image"].to(self.device)
                v_lab = val_data["label"].to(self.device)
                v_out = monai.inferers.sliding_window_inference(
                    v_in, self.roi_size, 4, self.model
                )
                v_out = [self.post_trans(torch.sigmoid(i)) for i in v_out]
                self.dice_metric(y_pred=v_out, y=v_lab)

            results = self.dice_metric.aggregate()
            self.dice_metric.reset()

        return {
            region: results[i].item()
            for i, region in enumerate(self.dice_regions)
        } | {"mean": results.mean().item()}
```

== Appendix E: Explainable AI (XAI) Suite
The following sections document the complete source code for the post-hoc Explainable AI framework, including gradient-based methods, perturbation methods, uncertainty quantification, and quantitative validation metrics.

=== E.1 3D Gradient-weighted Class Activation Mapping (Grad-CAM)
#strong[Listing:] 3D Grad-CAM Implementation (`grad_cam.py`)

```python
"""
grad_cam.py - 3D Grad-CAM for volumetric segmentation models.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for 3D CNNs.
Produces a class-discriminative saliency volume by weighting feature-map
activations at a chosen convolutional layer with gradient-derived importance
weights.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localisation", IJCV 2020.

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

        # ── 6. ReLU - keep only positive contributions ──────────────────
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
```

=== E.2 3D Guided Backpropagation
#strong[Listing:] 3D Guided Backpropagation Implementation (`guided_backprop.py`)

```python
"""
guided_backprop.py - 3D Guided Backpropagation for volumetric segmentation models.

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
```

=== E.3 Input × Gradient (LRP Proxy)
#strong[Listing:] Input × Gradient Implementation (`lrp.py`)

```python
"""
lrp.py - Input × Gradient attribution (LRP proxy) for volumetric models.

Implements the Input × Gradient method, a first-order Taylor decomposition
that serves as a tractable proxy for epsilon-LRP in architectures where
true layer-wise propagation is infeasible (e.g., SegResNet with GroupNorm
and residual skip connections).

Unlike Guided Backpropagation, which filters gradients purely to find
"what could change the output", Input × Gradient distributes the actual
prediction score back to the input voxels to answer "what explicitly
contributed to the output".
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
```

=== E.4 3D Occlusion Sensitivity
#strong[Listing:] 3D Occlusion Sensitivity Implementation (`occlusion.py`)

```python
"""
occlusion.py - 3D Occlusion Sensitivity for volumetric models.

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
```

=== E.5 Monte Carlo (MC) Dropout
#strong[Listing:] MC Dropout Uncertainty Implementation (`uncertainty.py`)

```python
"""
uncertainty.py - MC Dropout for uncertainty quantification in 3D segmentation.

Implements Monte Carlo (MC) Dropout to estimate model uncertainty. By running
multiple forward passes with dropout layers enabled at inference time, we can
compute the mean prediction (a more stable segmentation) and the voxel-wise
variance (a proxy for predictive uncertainty).

Key insight from notes: No model retraining needed - SegResNet already has
dropout_prob=0.1 in its architecture. We just keep those layers active
during inference.

Reference:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", ICML 2016.

Usage:
    from src.xai.uncertainty import MCDropout3D

    mcd = MCDropout3D(model, num_iters=20)
    mean_pred, uncertainty = mcd.generate(input_tensor)
    mcd.restore()  # re-disable dropout
"""

import torch
import numpy as np


class MCDropout3D:
    """
    Monte Carlo (MC) Dropout for 3D segmentation uncertainty estimation.

    Enables dropout at inference time and runs N stochastic forward passes
    on the same input to obtain:
        - Mean prediction: a stabilised, ensemble-like segmentation mask
        - Uncertainty map: per-voxel variance across passes

    Args:
        model:      A trained 3D segmentation model with Dropout layers.
        num_iters:  Number of stochastic forward passes (default: 20).
    """

    def __init__(self, model: torch.nn.Module, num_iters: int = 20):
        self.model = model
        self.num_iters = num_iters

    def _enable_dropout(self):
        """
        Enable dropout layers during inference while keeping all other
        layers (BatchNorm, etc.) in eval mode.

        This is the core of MC Dropout: model.eval() disables dropout,
        so we selectively re-enable only Dropout modules.
        """
        self.model.eval()
        for module in self.model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    def _restore_eval(self):
        """Restore full eval mode (disable dropout again)."""
        self.model.eval()

    def generate(self, input_tensor: torch.Tensor):
        """
        Generate mean prediction and uncertainty map using MC Dropout.

        Process:
            1. Enable dropout at inference time
            2. Run N forward passes on the same input
            3. Apply sigmoid → probabilities ∈ [0, 1]
            4. Stack all N outputs
            5. Compute mean (stable prediction) and variance (uncertainty)

        Args:
            input_tensor: Model input, shape (1, C, D, H, W).

        Returns:
            Tuple of:
                mean_pred:       numpy array (3, D, H, W) - mean probabilities
                uncertainty_map: numpy array (3, D, H, W) - per-voxel variance
        """
        self._enable_dropout()

        outputs = []
        with torch.no_grad():
            for _ in range(self.num_iters):
                output = self.model(input_tensor)
                probs = torch.sigmoid(output)
                outputs.append(probs)

        # Stack → (N, 1, 3, D, H, W)
        stacked = torch.stack(outputs)

        # Mean prediction → (1, 3, D, H, W)
        mean_pred = torch.mean(stacked, dim=0)

        # Variance (uncertainty) → (1, 3, D, H, W)
        uncertainty = torch.var(stacked, dim=0)

        # Restore eval mode
        self._restore_eval()

        # Remove batch dim → (3, D, H, W)
        mean_np = mean_pred.squeeze(0).cpu().numpy()
        unc_np = uncertainty.squeeze(0).cpu().numpy()

        return mean_np, unc_np
```

=== E.6 XAI Evaluation Metrics
#strong[Listing:] XAI Evaluation Metrics Implementation (`metrics.py`)

```python
"""
metrics.py - Quantitative evaluation of XAI saliency maps against ground truth.

Implements metrics that measure how well a model's attention (saliency)
aligns with actual tumour locations, answering:
    "Is the model looking at the right region for the right reasons?"

Metrics:
    - Pointing Game:     Does the peak saliency voxel fall inside the tumour?
    - Saliency Coverage: What fraction of total saliency mass is inside the tumour?
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
        ground_truth: 3D binary array (D, H, W), 1 = tumour, 0 = background.

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
        ground_truth: 3D binary array (D, H, W), 1 = tumour, 0 = background.

    Returns:
        True if the maximum sensitivity drop voxel is within the tumour.
    """
    return pointing_game(saliency, ground_truth)


def saliency_coverage(saliency: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Saliency Coverage: fraction of total saliency mass inside the GT region.

    High coverage (→ 1.0) means the model focuses its attention on the
    actual tumour.  Low coverage means the model is distracted by
    irrelevant areas.

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumour, 0 = background.

    Returns:
        Float in [0, 1].  1.0 = all saliency is inside the tumour.
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
        ground_truth: 3D binary array (D, H, W), 1 = tumour, 0 = background.
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
    only cover the tumour but also match its shape.

    Formula:
        Weighted_Dice = 2 * Σ(S · G) / (Σ S + Σ G)

    where S ∈ [0,1] is the normalised saliency and G ∈ {0,1} is the GT.

    Args:
        saliency:     3D array (D, H, W) with values in [0, 1].
        ground_truth: 3D binary array (D, H, W), 1 = tumour, 0 = background.

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
        ground_truth: 3D binary array (D, H, W), 1 = tumour, 0 = background.
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
```
