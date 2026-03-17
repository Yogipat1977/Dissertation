# Illuminating the Black Box

**An Explainable AI Framework for Brain Tumor Segmentation Using Volumetric Data Interpretation in 3D CNNs**

*Yogi Amitkumar Patel — CN6000 Final Year Project*

## Overview

This project develops a 3D Explainable AI (XAI) framework to interpret 3D CNN models used for brain tumor segmentation on the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/622351) dataset. Explanations are visualised in an immersive VR environment to aid clinical decision-making.

## Project Structure

```
├── configs/
│   ├── default.yaml                 # Default prototype config (SegResNet)
│   ├── segresnet.yaml               # SegResNet-specific config
│   ├── attention_unet.yaml          # Attention U-Net config
│   ├── swinunetr.yaml               # Swin UNETR config
│   └── full_training_segresnet.yaml # Full dataset config (~1,250 patients)
├── src/                             # Core library
│   ├── config.py                    # YAML config loading & path resolution
│   ├── data/
│   │   ├── dataset.py               # Data listing, splitting, DataLoaders
│   │   └── transforms.py           # BraTS label conversion & pipelines
│   ├── models/
│   │   └── factory.py               # Model, loss, optimizer factory
│   ├── training/
│   │   └── trainer.py               # Training loop, validation, checkpointing
│   ├── evaluation/
│   │   └── evaluator.py             # Test-set evaluation & results export
│   └── xai/                         # Explainable AI module
│       ├── grad_cam.py              # GradCAM3D — 3D Grad-CAM class
│       ├── guided_backprop.py       # GuidedBackprop3D — Guided Backpropagation
│       ├── lrp.py                   # LRP3D — Layer-wise Relevance Propagation
│       ├── occlusion.py             # OcclusionSensitivity — Sliding window ablation
│       ├── uncertainty.py           # MCDropout3D — Monte Carlo Dropout uncertainty
│       ├── metrics.py               # XAI evaluation metrics (4 metrics)
│       └── __init__.py              # Package exports
├── train.py                         # Training entry point
├── evaluate.py                      # Standalone evaluation entry point
├── scripts/
│   ├── download_data.py             # Download training data from Synapse
│   ├── extract_prototype.py         # Create prototype subset for local testing
│   ├── generate_manifest.py         # Save dataset splits to manifest.json
│   ├── export_predictions.py        # Export NIfTI predictions for 3D Slicer / VR
│   ├── evaluate_per_patient.py      # Per-patient metrics (CSV + summary stats)
│   ├── generate_gradcam.py          # Grad-CAM saliency generation + metrics
│   ├── evaluate_gradcam_coarse.py   # Grad-CAM native resolution evaluation + metrics
│   ├── generate_gbp.py              # GBP + Guided Grad-CAM generation + metrics
│   ├── generate_lrp.py              # LRP generation + metrics
│   ├── generate_occlusion.py        # Occlusion sensitivity generation + metrics
│   └── generate_mc_dropout.py       # MC Dropout uncertainty maps + evaluation metrics
├── models/                          # Saved checkpoints (git-ignored)
├── data/                            # Datasets (git-ignored)
│   ├── BraTS2023-Training/          # Full training data (~1,250 patients)
│   └── Prototype_Data/              # 45-patient subset for prototyping
├── slicer_export/                   # Exported NIfTI volumes for 3D Slicer
│   ├── Model_Test_Results/          # Predicted segmentations + MRIs + GT
│   └── XAI/                         # XAI saliency maps
│       ├── Grad_CAM/                # Grad-CAM heatmaps per patient
│       ├── GBP/                     # Guided Backpropagation maps per patient
│       ├── Guided_Grad_CAM/         # Guided Grad-CAM maps per patient
│       ├── LRP/                     # Layer-wise Relevance Propagation maps per patient
│       ├── Occlusion/               # Occlusion sensitivity drops per patient
│       └── MC_Dropout/              # MC Dropout uncertainty maps per patient
├── results/                         # Evaluation CSVs
│   └── CSVs/                        # All metric outputs
├── Notes/                           # Research notes & documentation
│   ├── XAI/                         # XAI implementation & metrics docs
│   ├── Formulas/                    # Project-wide formula reference
│   └── Scripts/                     # Scripts & source code reference
├── Final report/                    # Dissertation (Typst)
├── Lit Review/                      # Literature review (Quarto)
├── Research papers/                 # Reference papers
└── prototype/                       # Legacy prototyping (reference)
```


## Quick Start

### 1. Environment Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. API Keys & Data Access

Before downloading data or training you need accounts on **Synapse** (data host) and **Weights & Biases** (experiment tracking).

#### Synapse — BraTS 2023 Dataset

1. **Create a free account** at [synapse.org](https://www.synapse.org/#!RegisterAccount:0).
2. Navigate to the **BraTS 2023 GLI Challenge** page:
   [synapse.org/Synapse:syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351).
3. **Accept the dataset Terms & Conditions** — you will be prompted when you first access the data tab. Access is not granted until you accept.
4. Once approved, go to **Account Settings → Personal Access Tokens**
   ([direct link](https://www.synapse.org/#!PersonalAccessTokens:)) and generate a new token.

#### Weights & Biases — Experiment Tracking

1. **Create a free account** at [wandb.ai](https://wandb.ai/site).
2. Go to [wandb.ai/authorize](https://wandb.ai/authorize) and copy your API key.

#### Add keys to `.env`

```bash
cp .env.example .env
```

Then edit `.env` and paste in your keys:

| Variable | Purpose |
|----------|---------|
| `SYNAPSE_AUTH_TOKEN` | Your Synapse personal access token |
| `WANDB_API_KEY` | Your Weights & Biases API key |

> **Note:** The `.env` file is git-ignored and will not be committed. Each user must create their own.

### 3a. Prototype (Local Machine)

Download the data, extract a subset, then train:

```bash
# Download full training data from Synapse (one-time):
python scripts/download_data.py

# Extract a 45-patient prototype subset:
python scripts/extract_prototype.py

# Train on the prototype:
python train.py
```

### 3b. Full Training (Cloud / GPU Server)

```bash
# Download full training data:
python scripts/download_data.py

# Train on the entire dataset (~1,250 patients):
python train.py --config configs/full_training_segresnet.yaml
```

### 3c. Saving the Dataset Manifest

To save the exact `train/val/test` split used during training (determined by the config seed) to a JSON file for reproducibility:

```bash
python scripts/generate_manifest.py --config configs/full_training_segresnet.yaml 
```
**Output:** `results/manifest.json`

### 4. Training Different Models

Each model has its own config file. Select a model by passing the appropriate config:

```bash
# SegResNet (default)
python train.py --config configs/segresnet.yaml

# Attention U-Net
python train.py --config configs/attention_unet.yaml

# Swin UNETR
python train.py --config configs/swinunetr.yaml
```

Each run saves a checkpoint and config copy to `models/<run_name>/` for reproducibility.

### 5. Custom Experiments

```bash
# Copy any model config and tweak hyperparameters:
cp configs/segresnet.yaml configs/experiment.yaml
# edit: dropout, learning_rate, epochs, etc.
python train.py --config configs/experiment.yaml
```

### 6. Evaluation

```bash
python evaluate.py --config configs/segresnet.yaml --checkpoint models/<run_name>/best_model.pth
```

### 7. Export Predictions for 3D Slicer / VR

`scripts/export_predictions.py` runs sliding-window inference on the test set and saves each patient's predicted segmentation as a discrete-label NIfTI file (labels 0–3, same format as `seg.nii.gz`). It also copies the original structural MRI scans and ground-truth segmentation into a per-patient folder, ready to be dragged directly into **3D Slicer** or **SlicerVR**.

```bash
python scripts/export_predictions.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to YAML config file |
| `--checkpoint` | Yes | — | Path to saved model weights (`.pth`) |
| `--limit` | No | `0` (all) | Process only the first *N* patients |
| `--split` | No | `test` | Dataset split to export (`train`, `val`, `test`) |

**Output (`slicer_export/`):**

```
slicer_export/
└── BraTS-GLI-00123-000/
    ├── BraTS-GLI-00123-000_pred.nii.gz   # Model prediction (discrete labels)
    ├── BraTS-GLI-00123-000-seg.nii.gz    # Ground-truth segmentation
    ├── BraTS-GLI-00123-000-t1c.nii.gz    # T1-contrast MRI
    ├── BraTS-GLI-00123-000-t1n.nii.gz    # T1-native MRI
    ├── BraTS-GLI-00123-000-t2f.nii.gz    # T2-FLAIR MRI
    └── BraTS-GLI-00123-000-t2w.nii.gz    # T2-weighted MRI
```

> The script supports **resuming** — patients that already have a `_pred.nii.gz` file are automatically skipped.

### 8. Per-Patient Evaluation

`scripts/evaluate_per_patient.py` evaluates the model on each test patient individually and exports granular metrics. It computes **Dice, HD95, IoU, Sensitivity, and Specificity** for every patient across the three BraTS regions (Whole Tumor, Tumor Core, Enhancing Tumor).

```bash
python scripts/evaluate_per_patient.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to YAML config file |
| `--checkpoint` | Yes | — | Path to saved model weights (`.pth`) |
| `--split` | No | `test` | Split to evaluate (`train`, `val`, `test`) |
| `--output_name` | No | `per_patient_metrics.csv` | Filename for the detailed CSV |

**Output (`results/CSVs/`):**

| File | Contents |
|------|----------|
| `per_patient_SegResNet_metrics.csv` | One row per patient × region with Dice, HD95, IoU, Sensitivity, Specificity |
| `summary_SegResNet_metrics.csv` | Mean ± Std aggregated by region — ready for the dissertation report |

### 9. Explainable AI (XAI) — Saliency Map Generation

The XAI pipeline generates saliency maps that show *where* the model looks when making segmentation decisions. Three XAI methods are implemented:

#### 9a. Grad-CAM

Produces coarse, class-discriminative heatmaps from the bottleneck layer.

```bash
python scripts/generate_gradcam.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth \
  --limit 5
```

You can also evaluate Grad-CAM at **native feature map resolution** (without upsampling blur penalties) against a downsampled Ground Truth:

```bash
python scripts/evaluate_gradcam_coarse.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth \
  --layer bottleneck --topk 15
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to YAML config file |
| `--checkpoint` | Yes | — | Path to saved model weights (`.pth`) |
| `--limit` | No | `0` (all) | Process only the first *N* patients |
| `--layer` | No | `bottleneck` | Target layer (`bottleneck`, `encoder3`, `encoder2`, `encoder1`, `decoder1`) |
| `--topk` | No | `0` | Optional Top-K% thresholding for native map evaluation |

**Output:** `slicer_export/XAI/Grad_CAM/<patient>/` + `results/CSVs/xai_gradcam_metrics.csv` (or coarse evaluation metrics)

#### 9b. Guided Backpropagation (GBP) & Guided Grad-CAM

GBP produces full-resolution saliency maps. Guided Grad-CAM combines GBP × Grad-CAM for sharp, class-specific maps.

```bash
# GBP only
python scripts/generate_gbp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth \
  --limit 5

# GBP + Guided Grad-CAM
python scripts/generate_gbp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth \
  --limit 5 --guided_gradcam
```

**Output:**

| Method | NIfTI Heatmaps | Metrics CSV |
|--------|----------------|-------------|
| GBP | `slicer_export/XAI/GBP/<patient>/` | `results/CSVs/xai_gbp_metrics.csv` |
| Guided Grad-CAM | `slicer_export/XAI/Guided_Grad_CAM/<patient>/` | `results/CSVs/xai_guided_gradcam_metrics.csv` |

#### 9c. Layer-wise Relevance Propagation (LRP)

Mathematical projection of exactly what inputs contributed to the output score.

```bash
python scripts/generate_lrp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth --limit 5
```

**Output:** `slicer_export/XAI/LRP/<patient>/` + `results/CSVs/xai_lrp_metrics.csv`

#### 9d. Occlusion Sensitivity

Tests model reliance on specific regions by sliding a black box across the input and recording confidence drops.

```bash
python scripts/generate_occlusion.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth --limit 5
```

**Output:** `slicer_export/XAI/Occlusion/<patient>/` + `results/CSVs/xai_occlusion_metrics.csv`

#### 9e. Monte Carlo Dropout (Uncertainty)

Quantifies model confidence by sampling multiple stochastic forward passes with dropout enabled during inference. Output provides both a mean prediction map and a variance (uncertainty) map.

```bash
python scripts/generate_mc_dropout.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/<run_name>/best_model.pth \
  --num_iters 20 \
  --patient_ids "BraTS-GLI-01497-000,BraTS-GLI-00291-000"
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--num_iters` | No | `20` | Number of stochastic forward passes per patient |
| `--patient_ids` | No | — | Optional comma-separated list of specific patient IDs to process |

**Output:** `slicer_export/XAI/MC_Dropout/<patient>/` + `results/CSVs/xai_mc_dropout_metrics.csv`

#### XAI Evaluation Metrics

All XAI methods are evaluated against ground truth using quantitative metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| **Pointing Game** / **MSR** | 0 or 1 | Does the peak saliency voxel fall inside the tumor? |
| **Saliency Coverage** | [0, 1] | Fraction of total saliency mass inside the tumor |
| **Saliency IoU** | [0, 1] | Overlap between thresholded saliency and GT mask |
| **Weighted Dice / LRP Dice** | [0, 1] | Soft Dice between continuous saliency and binary GT |
| **Uncertainty Area Ratio (UAR)** | [0, 1] | Fraction of uncertainty mask inside tumor vs total uncertainty mass |
| **Boundary Ratio** | [0, 1] | Fraction of uncertainty explicitly located at GT borders |
| **Saliency-Unc Correlation** |-1 to 1| Pearson correlation between uncertainty and XAI saliency (e.g., LRP) |

---

## Model Architectures

Three 3D segmentation architectures are supported, all from [MONAI](https://monai.io/):

### SegResNet

Encoder–decoder with residual blocks. Lightweight and fast to train — good baseline for 3D medical segmentation.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `init_filters` | Number of filters in the first layer (doubled at each level) | `32` |
| `dropout_prob` | Dropout probability applied after each residual block | `0.1` |

```bash
python train.py --config configs/segresnet.yaml
```

### Attention U-Net

U-Net variant with **attention gates** that learn to suppress irrelevant regions and focus on salient features — helpful for small structures like Enhancing Tumor (ET).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `channels` | Feature maps per encoder level | `[32, 64, 128, 256]` |
| `strides` | Down-sampling strides (length = `len(channels) - 1`) | `[2, 2, 2]` |
| `dropout` | Dropout probability | `0.1` |

```bash
python train.py --config configs/attention_unet.yaml
```

### Swin UNETR

Transformer-based architecture using **Swin Transformer** as the encoder. Captures long-range dependencies better than pure CNNs but is more memory-intensive.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `feature_size` | Embedding dimension (must be divisible by 12) | `48` |
| `drop_rate` | Dropout rate for transformer layers | `0.1` |

> **Note:** SwinUNETR uses `roi_size: [128, 128, 128]` by default (vs. 160³ for the others) due to higher memory requirements.

```bash
python train.py --config configs/swinunetr.yaml
```

### Model Comparison Summary

| Model | Type | Key Strength | Memory | Config |
|-------|------|-------------|--------|--------|
| SegResNet | CNN (residual) | Fast, lightweight | Low | `segresnet.yaml` |
| Attention U-Net | CNN (attention gates) | Focuses on small tumors | Medium | `attention_unet.yaml` |
| Swin UNETR | Transformer | Long-range context | High | `swinunetr.yaml` |

---

## Evaluation Metrics

The evaluator computes **5 metrics** per BraTS region (Whole Tumor, Tumor Core, Enhancing Tumor):

| Metric | Direction | Description |
|--------|-----------|-------------|
| **Dice Score** | ↑ higher = better | Overlap similarity (primary BraTS benchmark metric) |
| **HD95** | ↓ lower = better | 95th-percentile Hausdorff surface distance in mm (secondary BraTS metric) |
| **IoU (Jaccard)** | ↑ higher = better | Intersection over Union — alternative overlap metric |
| **Sensitivity** | ↑ higher = better | True positive rate (recall) — how much tumor is detected |
| **Specificity** | ↑ higher = better | True negative rate — how well background is preserved |

Results are printed to console, logged to W&B, and saved as CSV in `results/`.

---

## Configuration

All tuneable parameters live in `configs/*.yaml`:

| Section | Key parameters |
|---------|---------------|
| `model` | `architecture`, model-specific params (see above) |
| `training` | `epochs`, `learning_rate`, `weight_decay`, `loss`, `scheduler` |
| `data` | `data_dir`, `train_split`, `val_split`, `roi_size`, `batch_size` |
| `wandb` | `project`, `mode` (online/offline/disabled) |

## Tech Stack

| Area | Tools |
|------|-------|
| Deep Learning | PyTorch, MONAI |
| Models | SegResNet, Attention U-Net, Swin UNETR |
| Explainability | 3D Grad-CAM, GBP, Guided Grad-CAM, LRP, Occlusion, MC Dropout |
| XAI Metrics | PG, Coverage, IoU, Weighted Dice, Uncertainty Metrics |
| Visualisation | 3D Slicer / SlicerVR |
| Experiment Tracking | Weights & Biases |
| Documentation | Typst, Quarto |

