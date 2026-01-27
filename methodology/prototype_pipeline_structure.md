# Prototype Implementation Pipeline: 3D Volumetric Segmentation
**Phase 1 of "Illuminating the Black Box" Framework**

This document outlines the technical structure for the first evolutionary prototype of the dissertation project. It focuses on establishing a robust **3D Native** deep learning pipeline for brain tumor segmentation, serving as the foundation for subsequent XAI and VR integration.

## 1. Pipeline Overview
The implementation follows a modular **Monai-Core** pattern, prioritizing intuitiveness and reproducibility. The workflow transforms raw 3D MRI data into precise segmentation masks using a state-of-the-art **3D Residual U-Net**.

```mermaid
graph TD
    A[Raw BraTS 2021 Data] -->|Load NIfTI| B(Data Preprocessing)
    B -->|Crop & Normalize| C[3D Tensors (4, 192, 192, 128)]
    C -->|Augmentation| D[Training Batch]
    D -->|Feed Forward| E[3D Residual U-Net]
    E -->|Logits| F[Optimization (Dice + Focal Loss)]
    F -->|Backpropagation| E
    E -->|Inference| G[Validation Metrics]
```

---

## 2. Phase 1: Data Engineering Pipeline

### 2.1 Data Ingestion
*   **Source:** BraTS 2021 Dataset.
*   **Modalities (Input Channels):**
    1.  **T1**: Anatomical structure.
    2.  **T1ce (Contrast Enhanced)**: Highlights **Enhancing Tumor (ET)**.
    3.  **T2**: Anatomical structure + edema visibility.
    4.  **FLAIR**: Highlights **Whole Tumor (WT)** including edema.
*   **Target (Labels):**
    *   Converted from BraTS classes (1, 2, 4) to semantic regions:
        1.  **Enhancing Tumor (ET)**: Active tumor region.
        2.  **Tumor Core (TC)**: ET + Necrotic core.
        3.  **Whole Tumor (WT)**: TC + Edema.

### 2.2 Volumetric Preprocessing (The "3D-Native" Approach)
To solve the "Cognitive Friction" of 2D slice analysis, all operations preserve the Z-depth.

1.  **Volume Cropping (`CropForegroundd`)**:
    *   **Action:** Removes the large margins of zero-valued background pixels.
    *   **Target Size:** **(192, 192, 128)**.
    *   **Rationale:** Fits within standard GPU memory (16GB) while retaining the complete brain volume, preventing the information loss associated with 2D slicing.

2.  **Intensity Normalization**:
    *   **Action:** Z-score normalization (Mean = 0, Std = 1) per modality.
    *   **Rationale:** MRI signal intensities vary across scanners; this standardizes the input distribution for stable training.

3.  **Data Augmentation (Training Only)**:
    *   **Spatial:** Random 3D Flips (Axial/Sagittal/Coronal), Random Rotation.
    *   **Intensity:** Random Gibbs Noise (simulating MRI artifacts), Random Intensity Shift.
    *   **Rationale:** Prevents overfitting on the "background" class and improves robustness.

---

## 3. Phase 2: Model Architecture

### 3.1 The Engine: 3D Residual U-Net (SegResNet)
Selected based on the **BrainAR** study which demonstrated superior convergence speed (7 epochs vs 21) and accuracy compared to standard U-Nets.

*   **Structure:** Encoder-Decoder with Residual Connections.
*   **Input Shape:** `(Batch_Size, 4, 192, 192, 128)`
*   **Key Features:**
    *   **Residual Blocks:** Allow for a deeper network without the vanishing gradient problem.
    *   **Group Normalization:** More stable than Batch Norm for small batch sizes (typical in 3D medical imaging).
    *   **Dropout (0.2):** Standard regularization to prevent overfitting.

### 3.2 Configuration
*   **Spatial Dimensions:** 3
*   **Input Channels:** 4
*   **Output Channels:** 3 (Multi-label segmentation)
*   **Initial Filters:** 16 (Scalable to 32/64 for final training)

---

## 4. Phase 3: Training & Optimization Strategy

### 4.1 Loss Function
*   **Selection:** **DiceFocalLoss**
*   **Equation:** $Loss = 0.5 \times DiceLoss + 0.5 \times FocalLoss$
*   **Rationale:**
    *   **Dice Loss:** Maximizes the overlap (IoU) for precise boundary delineation.
    *   **Focal Loss:** Addresses the extreme **Class Imbalance** (Tumor is <2% of the brain volume) by assigning higher weights to hard-to-classify voxels.

### 4.2 Optimization
*   **Optimizer:** **AdamW** (Adam with Weight Decay).
*   **Learning Rate:** `1e-4` with **Cosine Annealing Scheduler**.
*   **Rationale:** Ensures smooth convergence and prevents getting stuck in local minima.

---

## 5. Phase 4: Evaluation Protocols

### 5.1 Quantitative Metrics
The prototype will be evaluated on the Validation Set (20% split) using:

1.  **Dice Similarity Coefficient (DSC):**
    *   Measures voxel-wise overlap accuracy.
    *   *Target:* > 0.85 for Whole Tumor.
2.  **Hausdorff Distance (HD95):**
    *   Measures the maximum distance between prediction and ground truth boundaries (95th percentile).
    *   *Rationale:* Critical for surgical planning (VR phase) where boundary precision determines safety.

### 5.2 Validation Strategy
*   **Sliding Window Inference:**
    *   The model predicts on overlapping 3D patches during inference to handle full-resolution volumes, averaged to reduce edge artifacts.

---

## 6. Future Integration (Phase XAI & VR)
*This prototype lays the groundwork for:*
*   **XAI:** The trained weights from this pipeline will be frozen and fed into **3D Grad-CAM** to generate saliency volumes.
*   **VR:** The resulting NIfTI files (Input + Prediction + XAI Map) will be exported for the **SlicerVR** pipeline.

```