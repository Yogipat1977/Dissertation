# Illuminating the Black Box

**An Explainable AI Framework for Brain Tumor Segmentation Using Volumetric Data Interpretation in 3D CNNs**

*Yogi Amitkumar Patel — CN6000 Final Year Project*

## Overview

This project develops a 3D Explainable AI (XAI) framework to interpret 3D CNN models used for brain tumor segmentation on the [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/622351) dataset. Explanations are visualised in an immersive VR environment to aid clinical decision-making.

## Project Structure

```
├── configs/
│   ├── default.yaml             # Prototype config (45 patients)
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

### 2a. Prototype (Local Machine)

Download the data, extract a subset, then train:

```bash
# Download full training data from Synapse (one-time):
python scripts/download_data.py

# Extract a 45-patient prototype subset:
python scripts/extract_prototype.py

# Train on the prototype:
python train.py
```

### 2b. Full Training (Cloud / GPU Server)

```bash
# Download full training data:
python scripts/download_data.py

# Train on the entire dataset (~1,250 patients):
python train.py --config configs/full_training.yaml
```

### 3. Custom Experiments

```bash
# Copy a config and tweak hyperparameters:
cp configs/default.yaml configs/experiment.yaml
# edit: init_filters, dropout, lr, epochs, etc.
python train.py --config configs/experiment.yaml
```

Each run saves a checkpoint and config copy to `models/<run_name>/` for reproducibility.

### 4. Evaluation

```bash
python evaluate.py --config configs/default.yaml --checkpoint models/<run_name>/best_model.pth
```

## Configuration

All tuneable parameters live in `configs/default.yaml`:

| Section | Key parameters |
|---------|---------------|
| `model` | `architecture`, `init_filters`, `dropout_prob`, `in_channels`, `out_channels` |
| `training` | `epochs`, `learning_rate`, `weight_decay`, `loss`, `scheduler` |
| `data` | `data_dir`, `train_split`, `val_split`, `roi_size`, `batch_size` |
| `wandb` | `project`, `mode` (online/offline/disabled) |

## Tech Stack

| Area | Tools |
|------|-------|
| Deep Learning | PyTorch, MONAI |
| Model | SegResNet (3D) |
| Explainability | 3D Grad-CAM (planned) |
| Visualisation | 3D Slicer / SlicerVR |
| Experiment Tracking | Weights & Biases |
| Documentation | Typst, Quarto |
