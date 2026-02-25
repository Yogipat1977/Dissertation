# Project: Explainable AI (XAI) Framework for Brain Tumor Segmentation

**Author:** Yogi Amitkumar Patel  
**Course:** CN6000 - Final Year Project  
**Title:** "Illuminating the Black Box: An Explainable AI Framework for Brain Tumor Segmentation Using Volumetric Data Interpretation in 3D CNNs"

## Overview
This project focuses on developing a 3D Explainable AI (XAI) framework to interpret 3D Convolutional Neural Network (CNN) models used for brain tumor segmentation. The key innovation is visualising these explanations in an immersive Virtual Reality (VR) environment to aid clinical decision-making.

## Key Objectives
1.  **3D XAI Framework:** Quantify 3D CNN segmentation predictions using voxel-based XAI (e.g., 3D Grad-CAM).
2.  **Model Training:** Train 3D CNNs (SegResNet) using **PyTorch** and **MONAI** on the BraTS 2023 dataset.
3.  **VR Visualization:** Create a pipeline using **3D Slicer** (SlicerVR) or **Unity3D** to view 3D MRI, segmentation masks, and XAI saliency volumes.
4.  **Evaluation:** Assess the trade-off between model transparency and segmentation accuracy, and validate with clinical feedback.

## Directory Structure

### Code & Training
*   **`src/`**: Core modular library
    *   `config.py`: YAML config loading & path resolution
    *   `data/`: BraTS data listing, transforms, DataLoaders
    *   `models/`: Model, loss, optimizer factory
    *   `training/`: Training loop, validation, checkpointing
    *   `evaluation/`: Metrics computation & results export
*   **`configs/`**: YAML configuration files (hyperparameters, paths)
    *   `default.yaml`: Default prototype config
*   **`train.py`**: Main training entry point
*   **`evaluate.py`**: Standalone evaluation entry point
*   **`scripts/`**: Utility scripts
    *   `download_data.py`: Download BraTS2023 training data from Synapse
    *   `extract_prototype.py`: Create reproducible patient subset for local testing
    *   `prototype_data.py`: Legacy data sampling script
    *   `srcipt.py`: Legacy Synapse download script

### Data & Models
*   **`data/`**: Datasets (git-ignored)
    *   `BraTS2023-Training/`: Full training dataset (~1,250 patients)
    *   `Prototype_Data/`: 45-patient subset for prototyping
*   **`models/`**: Saved model checkpoints (git-ignored)
*   **`results/`**: Evaluation CSV files
*   **`slicer_export/`**: Exported NIfTI volumes for 3D Slicer visualization

### Prototyping & Reference
*   **`prototype/`**: Legacy prototyping code (kept as reference)
    *   `prototype.py`: Original monolithic training script
    *   `01_data_engineering.qmd`: Data pipeline notebook (Quarto)

### Documentation & Writing
*   **`Final report/`**: Dissertation (Typst format)
*   **`Lit Review/`**: Literature review materials (Quarto)
*   **`methodology/`**: Methodology drafts
*   **`project proposal/`**: Formal project proposal
*   **`dissertation Specific Materials/`**: Admin forms, templates, ethics docs
*   **`Research papers/`**: Reference papers (PDF)
*   **`Notes/`**: Research notes and Obsidian vault

## Technical Stack
*   **Deep Learning:** PyTorch, MONAI
*   **Model:** SegResNet (3D)
*   **Explainability:** 3D Grad-CAM (planned)
*   **Visualization:** 3D Slicer (SlicerVR extension)
*   **Experiment Tracking:** Weights & Biases
*   **Documentation:** Typst, Quarto, Markdown
*   **Languages:** Python

## Usage Notes
*   **Config-driven:** All hyperparameters live in `configs/default.yaml`. Copy and edit for experiments.
*   **Entry points:** `python train.py --config configs/default.yaml` to train, `python evaluate.py` to evaluate.
*   **Research Focus:** The primary goal is the dissertation; code generates results for the report.
