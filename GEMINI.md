# Project: Explainable AI (XAI) Framework for Brain Tumor Segmentation

**Author:** Yogi Amitkumar Patel  
**Course:** CN6000 - Final Year Project  
**Title:** "Illuminating the Black Box: An Explainable AI Framework for Brain Tumor Segmentation Using Volumetric Data Interpretation in 3D CNNs"

## Overview
This project focuses on developing a 3D Explainable AI (XAI) framework to interpret 3D Convolutional Neural Network (CNN) models used for brain tumor segmentation. The key innovation is visualising these explanations in an immersive Virtual Reality (VR) environment to aid clinical decision-making.

## Key Objectives
1.  **3D XAI Framework:** Quantify 3D CNN segmentation predictions using voxel-based XAI (e.g., 3D Grad-CAM).
2.  **Model Training:** Train 3D CNNs (U-Net) using **PyTorch** and **MONAI** on the BraTS 2021 dataset.
3.  **VR Visualization:** Create a pipeline using **3D Slicer** (SlicerVR) or **Unity3D** to view 3D MRI, segmentation masks, and XAI saliency volumes.
4.  **Evaluation:** Assess the trade-off between model transparency and segmentation accuracy, and validate with clinical feedback.

## Directory Structure

### Documentation & Writing
*   **`project proposal/`**: Contains the formal project proposal.
    *   `Final_Proposal.qmd`: Source file for the proposal (Quarto).
    *   `Final_Proposal.pdf`: Rendered proposal.
*   **`Lit Review/`**: Literature review materials.
    *   `Lit.md` / `litReview.qmd`: Drafts and source text.
    *   `references.bib`: Bibliography data.
*   **`methodology/`**: Methodology drafts.
    *   `main_structure.md`: Detailed outline of the methodology (Agile, System Arch, Research Approach).
    *   `methodology.qmd`: Quarto source.
*   **`dissertation Specific Materials /`**: Administrative forms, templates, and ethical approval docs.

### Research & Resources
*   **`Research papers /`**: Categorized PDF collection of academic papers.
    *   `Exceptional Research papers/`: Key references.
    *   `model specific papers/`: Technical details on architectures (U-Net, V-Net).
    *   `Methods spacific papers/`: Papers on specific techniques + `extract_methods.py` script.
*   **`Notes/`**: Research notes, diagrams, and Obsidian vault (`.obsidian/`).

### Code & Data
*   **`data/`**: Directory for datasets (BraTS 2021 subset).
*   **`pdf_env/`**: Python environment (likely for processing PDFs or analysis).

## Technical Stack
*   **Deep Learning:** PyTorch, MONAI
*   **Explainability:** SHAP, LIME, TorchCAM
*   **Visualization:** 3D Slicer (SlicerVR extension), ITK-SNAP, potentially Unity3D (C#)
*   **Documentation:** Quarto (`.qmd`), Markdown (`.md`), LaTeX (via Quarto)
*   **Languages:** Python, C# (for Unity)

## Usage Notes
*   **Quarto:** The project uses Quarto for document generation. `.qmd` files are the source of truth for reports.
*   **Research Focus:** The primary goal is the dissertation; code is the means to generate results for the report.
*   **Environment:** A `pdf_env` exists, suggesting Python dependence for specific tasks (likely PDF processing or initial data scripts).

## Recent Activity
*   **Date:** January 21, 2026
*   **Status:** Working on Methodology and Literature Review chapters. Proposal is finalized.
