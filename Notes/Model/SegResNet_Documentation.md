# SegResNet Model & Implementation Documentation

## 1. Overview
This document compiles the architectural details, configurations, and evaluation strategies of the **SegResNet** model utilized for the 3D brain tumor segmentation framework. The information herein is synthesized from theoretical handwritten notes, PyTorch/MONAI implementation logic, and the exact hyperparameter configurations used for the full BraTS 2023 dataset training.

---

## 2. Core Model Architecture: SegResNet

The model operates entirely in a 3D volumetric space (`spatial_dims: 3`), taking 4 input MRI modalities and producing 3 distinct segmentation channels. 

### A. Network Components
- **Encoder (Compression Path):** Shrinks the spatial dimensions of the input volume to extract deep semantic features. In SegResNet, the encoder progressively downsamples the volume across 4 levels using strided convolutions (`stride=2`), where each step takes two voxels instead of one.
- **Decoder (Expansion Path):** Takes the compressed bottleneck representation and progressively restores it to the original input size (`160x160x160`) via upsampling.
- **Skip Connections:** Bridges matching levels between the encoder and decoder. This ensures that fine-grained spatial and structural information lost during compression is passed directly to the expansion path for precise boundary reconstruction.
- **Filters (3D Kernels):** A 3D matrix (e.g., $3 \times 3 \times 3$) of learnable weights that slides across the volume detecting specific volumetric patterns (edges, textures, spatial shapes). 

### B. The Residual Block (ResBlock)
The core repeating unit of the entire SegResNet architecture. Rather than learning direct mappings, ResBlocks learn the *differences* (residuals) from what the network already has.
*   **Flow:** `Input (x) -> Group Norm -> ReLU -> Conv3D(3x3x3) -> Group Norm -> ReLU -> Conv3D(3x3x3) -> + Original Input (x) -> Output`
*   **Purpose:** By adding the original input back to the output of the convolutional layers, ResBlocks solve the **vanishing gradient problem**, allowing deep networks to train stably.

---

## 3. Implementation Configurations (`full_training_segresnet.yaml`)

The model was configured and trained on the full BraTS 2023 dataset using the following specific parameters to maximize performance on a high-memory compute node.

### A. Data & Hardware
- **Hardware Target:** 1× RTX 5880 Ada (48 GB VRAM) with Automatic Mixed Precision (AMP) enabled.
- **Dataset Split:** ~1,250 total patients -> 1,000 Train, 125 Validation, ~125 Test.
- **ROI Size:** `[160, 160, 160]` (Large spatial context for volumetric understanding).
- **Batch Size:** 1 (Consumes ~38 GB VRAM with AMP enabled).
- **Data Channels:**
  - **Inputs (4):** T1-native, T1-Contrast, T2-Weighted, T2-FLAIR.
  - **Outputs (3):** Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET).

### B. SegResNet Hyperparameters
- **`init_filters`: 32** (The starting width of the network. The first layer produces 32 different filtered versions of the input, doubling at each subsequent encoder level).
- **`dropout_prob`: 0.1** (Randomly switches off 10% of neurons during each training step. Helps the model not rely on any single neuron, acting as a regularizer to prevent overfitting).

### C. Training Hyperparameters
- **Epochs:** 35
- **Optimizer:** AdamW
- **Learning Rate:** `0.00005` (Scaled for small batch size, governed by a Cosine Annealing Scheduler).
- **Weight Decay:** `0.00001`
- **Loss Function:** `DiceFocalLoss`
  - *Focal Gamma (`2.0`):* Forces the network to heavily penalize errors on hard-to-segment examples (crucial for detecting the small Enhancing Tumor regions).
  - *Smoothing:* `1e-5` to prevent division by zero errors in Dice computation.

---

## 4. Data Processing & Transformations

Before feeding into the network, volumetric data undergoes strict preparation:
1.  **Channel Creation:** 
    - `WT`: Labels 1, 2, 3 (Necrotic/Non-enhancing, Edema, Enhancing).
    - `TC`: Labels 1, 3.
    - `ET`: Label 3.
2.  **Transformations:**
    - Load NIfTI data and ensure channel-first formatting.
    - Crop foreground image (remove empty space around the brain).
    - Normalize only non-zero data values (z-score normalization).
    - Random spatial cropping (to `160³`).
    - Random flips and intensity noise (Applied *only* during training as data augmentation to improve generalization).

---

## 5. Evaluation Metrics

The system's performance on the test split is quantitatively assessed using the following metrics:
- **Dice Score:** Measures how much the model's prediction overlaps with the Ground Truth labels. (Primary measure of segmentation quality).
- **IoU (Intersection over Union):** Similar to Dice, but stricter penalization for false positives/negatives.
- **HD95 (Hausdorff Distance 95th Percentile):** Evaluates surface boundary distances. Calculates the worst-case surface distance (in mm) between the prediction boundary and the ground truth boundary. Lower is better.
- **Sensitivity (Recall):** "Of all the actual tumor voxels, how many did the model detect?" High sensitivity means the model rarely misses the tumor.
- **Specificity:** "How effectively does the model secure true negatives (healthy tissue) and avoid false alarms?" Evaluates the model's ability to not predict healthy regions as tumors.
