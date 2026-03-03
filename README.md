# Illuminating the Black Box

**An Explainable AI Framework for Brain Tumor Segmentation Using Volumetric Data Interpretation in 3D CNNs**

*Yogi Amitkumar Patel — CN6000 Final Year Project*

## Overview

This project develops a 3D Explainable AI (XAI) framework to interpret 3D CNN models used for brain tumor segmentation on the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/622351) dataset. Explanations are visualised in an immersive VR environment to aid clinical decision-making.

## Project Structure

```
├── configs/
│   ├── default.yaml             # Default prototype config (SegResNet)
│   ├── segresnet.yaml           # SegResNet-specific config
│   ├── attention_unet.yaml      # Attention U-Net config
│   ├── swinunetr.yaml           # Swin UNETR config
│   └── full_training.yaml       # Full dataset config (~1,250 patients)
├── src/                         # Core library
│   ├── config.py                # YAML config loading & path resolution
│   ├── data/
│   │   ├── dataset.py           # Data listing, splitting, DataLoaders
│   │   └── transforms.py       # BraTS label conversion & pipelines
│   ├── models/
│   │   └── factory.py           # Model, loss, optimizer factory
│   ├── training/
│   │   └── trainer.py           # Training loop, validation, checkpointing
│   └── evaluation/
│       └── evaluator.py         # Test-set evaluation & results export
├── train.py                     # Training entry point
├── evaluate.py                  # Standalone evaluation entry point
├── scripts/
│   ├── download_data.py         # Download training data from Synapse
│   └── extract_prototype.py    # Create prototype subset for local testing
├── models/                      # Saved checkpoints (git-ignored)
│   ├── prototype_v1/            # SegResNet, init_filters=8
│   ├── prototype_32_v1/         # SegResNet, init_filters=32
│   └── prototype_filters32_v1/  # SegResNet, init_filters=32 (v2)
├── data/                        # Datasets (git-ignored)
│   ├── BraTS2023-Training/      # Full training data (~1,250 patients)
│   └── Prototype_Data/          # 45-patient subset for prototyping
├── prototype/                   # Legacy prototyping (reference)
│   └── prototype.py             # Original monolithic training script
├── slicer_export/               # Exported NIfTI volumes for 3D Slicer viz
├── results/                     # Evaluation CSVs
├── Final report/                # Dissertation (Typst)
├── Lit Review/                  # Literature review (Quarto)
└── Research papers/             # Reference papers
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
python train.py --config configs/full_training.yaml
```

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
| Explainability | 3D Grad-CAM (planned) |
| Visualisation | 3D Slicer / SlicerVR |
| Experiment Tracking | Weights & Biases |
| Documentation | Typst, Quarto |
