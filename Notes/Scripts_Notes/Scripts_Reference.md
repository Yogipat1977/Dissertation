# Scripts & Source Code Reference

**Project:** Illuminating the Black Box — XAI for Brain Tumor Segmentation  
**Last Updated:** 13 March 2026

This document describes every script and source module in the project.

---

## 1. Entry Points

### `train.py`

**Purpose:** Main training entry point.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file path |
| `--resume` | — | Resume from last checkpoint |
| `--run-dir` | — | Existing run directory (for resume) |

**Flow:** Load config → Create DataLoaders → Create model/loss/optimizer → Init W&B → Train → Evaluate → Save results.

---

### `evaluate.py`

**Purpose:** Standalone evaluation of a trained model.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights `.pth` |

**Output:** Console metrics + W&B log + `results/CSVs/<run_name>.csv`

---

## 2. Scripts (`scripts/`)

### `scripts/download_data.py`

**Purpose:** Download BraTS 2023 Training data from Synapse.  
**Requires:** `SYNAPSE_AUTH_TOKEN` in `.env`.  
**Output:** `data/BraTS2023-Training/` (~1,250 patient directories)

---

### `scripts/extract_prototype.py`

**Purpose:** Sample a reproducible 45-patient subset for local prototyping.  
**Output:** `data/Prototype_Data/` (45 patient directories)

---

### `scripts/export_predictions.py`

**Purpose:** Run sliding-window inference on test patients and export predicted segmentations as NIfTI files for 3D Slicer.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights |
| `--limit` | `0` (all) | Process N patients only |
| `--split` | `test` | Dataset split |

**Output:** `slicer_export/Model_Test_Results/<patient>/<patient>_pred.nii.gz` + original MRI scans + GT seg

---

### `scripts/evaluate_per_patient.py`

**Purpose:** Compute per-patient metrics (Dice, HD95, IoU, Sensitivity, Specificity) for every test patient individually.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights |
| `--split` | `test` | Dataset split |
| `--output_name` | `per_patient_metrics.csv` | Output CSV filename |

**Output:** `results/CSVs/per_patient_metrics.csv` + `results/CSVs/summary_per_patient_metrics.csv`

---

### `scripts/generate_gradcam.py`

**Purpose:** Generate 3D Grad-CAM saliency heatmaps for test patients with inline XAI metric computation.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights |
| `--limit` | `0` (all) | Process N patients only |
| `--layer` | `bottleneck` | Target layer for Grad-CAM |
| `--iou_threshold` | `0.5` | Threshold for Saliency IoU metric |

**Output:**
- `slicer_export/XAI/Grad_CAM/<patient>/<patient>_gradcam_{wt,tc,et}.nii.gz`
- `results/CSVs/xai_gradcam_metrics.csv`

---

### `scripts/generate_gbp.py`

**Purpose:** Generate Guided Backpropagation saliency maps and optionally Guided Grad-CAM maps for test patients.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights |
| `--limit` | `0` (all) | Process N patients only |
| `--guided_gradcam` | — | Also generate Guided Grad-CAM (GBP × Grad-CAM) |
| `--layer` | `bottleneck` | Target Grad-CAM layer (for Guided Grad-CAM) |
| `--iou_threshold` | `0.5` | Threshold for Saliency IoU metric |

**Output:**
- GBP: `slicer_export/XAI/GBP/<patient>/<patient>_gbp_{wt,tc,et}.nii.gz` + `results/CSVs/xai_gbp_metrics.csv`
- Guided Grad-CAM: `slicer_export/XAI/Guided_Grad_CAM/<patient>/<patient>_guided_gradcam_{wt,tc,et}.nii.gz` + `results/CSVs/xai_guided_gradcam_metrics.csv`

---

### `scripts/generate_occlusion.py`

**Purpose:** Generate 3D Occlusion Sensitivity maps via a sliding window and evaluate XAI metrics (including MSR Accuracy).

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights |
| `--limit` | `0` (all) | Process N patients only |
| `--window_size`| `16` | Size of occluding 3D cube |
| `--stride`| `8` | Sliding window step size |

**Output:** `slicer_export/XAI/Occlusion/<patient>/...` + `results/CSVs/xai_occlusion_metrics.csv`

---

### `scripts/generate_lrp.py`

**Purpose:** Generate high-resolution Layer-wise Relevance Propagation (LRP) relevance maps for test patients via the Input ✕ Gradient proxy rule. Strict visual export (skips metrics).

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config file |
| `--checkpoint` | required | Model weights |
| `--limit` | `0` (all) | Process N patients only |

**Output:** `slicer_export/XAI/LRP/<patient>/<patient>_lrp_{wt,tc,et}.nii.gz`

---

### `scripts/generate_manifest.py`

**Purpose:** Generate a JSON manifest showing train/val/test patient split.  
**Output:** `models/<run_name>/manifest.json`

---

### `scripts/prototype_data.py` (Legacy)

**Purpose:** Legacy data sampling script.  
**Status:** Superseded by `extract_prototype.py`.

---

### `scripts/srcipt.py` (Legacy)

**Purpose:** Legacy Synapse download script.  
**Status:** Superseded by `download_data.py`.

---

## 3. Source Library (`src/`)

### `src/config.py`

**Purpose:** Load YAML config files and resolve relative paths. Generates unique run names. Saves config copies for reproducibility.

**Functions:**
| Function | Description |
|----------|-------------|
| `load_config(path)` | Load and resolve YAML config |
| `save_config(cfg, dir)` | Save clean config copy |

---

### `src/data/dataset.py`

**Purpose:** Scan BraTS data directories, split into train/val/test, create MONAI DataLoaders.

**Functions:**
| Function | Description |
|----------|-------------|
| `get_brats_data_list(dir)` | Scan directory for patient data |
| `create_data_loaders(cfg)` | Create train/val/test DataLoaders |

---

### `src/data/transforms.py`

**Purpose:** BraTS-specific preprocessing and augmentation pipelines.

**Classes/Functions:**
| Name | Description |
|------|-------------|
| `ConvertToMultiChannelBraTS2023d` | Convert labels 1,2,3 → WT,TC,ET channels |
| `get_train_transforms(cfg)` | Training pipeline with augmentations |
| `get_val_transforms(cfg)` | Validation/test pipeline (no augmentation) |

**Training Augmentations:** RandFlip (×3 axes), RandRotate90, RandScaleIntensity, RandShiftIntensity, RandGaussianNoise.

---

### `src/models/factory.py`

**Purpose:** Create model, loss function, optimizer, and scheduler from config.

**Supported Models:** SegResNet, AttentionUnet, SwinUNETR  
**Supported Losses:** DiceCELoss, DiceLoss, DiceFocalLoss  
**Optimiser:** AdamW with CosineAnnealingLR

---

### `src/training/trainer.py`

**Purpose:** Full training lifecycle with AMP, best-model checkpointing, crash recovery, and W&B logging.

**Class: `Trainer`**

| Method | Description |
|--------|-------------|
| `fit()` | Run full training loop |
| `resume()` | Load from last checkpoint |
| `_train_epoch()` | Single training epoch with AMP |
| `_validate_epoch()` | Sliding-window validation |

---

### `src/evaluation/evaluator.py`

**Purpose:** Test-set evaluation computing Dice, HD95, IoU, Sensitivity, Specificity.

**Functions:**
| Function | Description |
|----------|-------------|
| `evaluate_set(model, loader, cfg, device)` | Evaluate one split |
| `run_evaluation(model, loaders, cfg, device)` | Run val + test, print, log to W&B, save CSV |

---

### `src/xai/grad_cam.py`

**Purpose:** 3D Grad-CAM implementation for volumetric segmentation models.

**Class: `GradCAM3D`**

| Method | Description |
|--------|-------------|
| `__init__(model, target_layer)` | Register forward/backward hooks |
| `generate(input, target_class)` | Produce (D,H,W) heatmap in [0,1] |
| `remove_hooks()` | Clean up hooks |

---

### `src/xai/guided_backprop.py`

**Purpose:** 3D Guided Backpropagation with ReLU backward hook overrides.

**Class: `GuidedBackprop3D`**

| Method | Description |
|--------|-------------|
| `__init__(model)` | Override all ReLU backward hooks |
| `generate(input, target_class)` | Full-resolution saliency map |
| `restore_relus()` | Remove hooks, restore normal backprop |

---

### `src/xai/lrp.py`

**Purpose:** 3D Layer-wise Relevance Propagation proxy using Input ✕ Gradient calculation.

**Class: `LRP3D`**

| Method | Description |
|--------|-------------|
| `__init__(model)` | Store Model |
| `generate(input, target_class)` | Returns full-resolution positive relevance map |

---

### `src/xai/occlusion.py`

**Purpose:** 3D Occlusion Sensitivity perturbation testing via sliding window.

**Class: `OcclusionSensitivity3D`**

| Method | Description |
|--------|-------------|
| `__init__(model, window_size, stride, baseline)` | Sets up hyperparams |
| `generate(input, target_class, batch_size)` | Slides occluder block over input, returns upsampled score drop heatmap |

---

### `src/xai/metrics.py`

**Purpose:** Quantitative XAI evaluation metrics (saliency vs ground truth).

**Functions:**
| Function | Description |
|----------|-------------|
| `pointing_game(S, G)` | Peak voxel inside tumor? |
| `saliency_coverage(S, G)` | Fraction of saliency inside tumor |
| `saliency_iou(S, G, τ)` | IoU of thresholded saliency and GT |
| `weighted_dice(S, G)` | Soft Dice between continuous S and binary G |
| `evaluate_saliency(S, G, τ)` | All 4 metrics in one call |

---

### `src/xai/__init__.py`

**Purpose:** Package exports for the XAI module.

**Exports:** `GradCAM3D`, `GuidedBackprop3D`, `LRP3D`, `evaluate_saliency`, `pointing_game`, `saliency_coverage`, `saliency_iou`, `weighted_dice`
