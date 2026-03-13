# Project Formulas Reference

**Project:** Illuminating the Black Box — XAI for Brain Tumor Segmentation  
**Last Updated:** 13 March 2026

This document collects every mathematical formula used across the project.

---

## 1. Loss Functions (Training)

### 1.1 Dice Loss

Used in: `src/models/factory.py` → `DiceLoss`

```
Dice_Loss = 1 - (2 × Σ(P × G) + ε) / (Σ(P²) + Σ(G²) + ε)
```

Where P = predicted probability, G = ground truth, ε = smoothing constant.

### 1.2 Focal Loss

Used in: `src/models/factory.py` → `DiceFocalLoss`

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Where p_t = model confidence for the correct class, γ = focusing parameter (default: 2.0).

### 1.3 DiceFocalLoss (Combined)

Used by our trained model (`full_training_segresnet.yaml`).

```
DiceFocalLoss = Dice_Loss + λ × Focal_Loss
```

Default: λ = 1.0, γ = 2.0.

---

## 2. Model Evaluation Metrics

### 2.1 Dice Score (Sørensen–Dice Coefficient)

Used in: `src/evaluation/evaluator.py`, `src/training/trainer.py`

```
Dice(P, G) = (2 × |P ∩ G|) / (|P| + |G|)
```

Where P = binary prediction, G = binary ground truth, |·| = volume (voxel count).

### 2.2 Hausdorff Distance 95th Percentile (HD95)

Used in: `src/evaluation/evaluator.py`

```
HD95(P, G) = max( d_95(∂P, ∂G),  d_95(∂G, ∂P) )
```

Where:
- `∂P`, `∂G` = surface voxels of prediction and ground truth
- `d_95(A, B)` = 95th percentile of minimum distances from each point in A to the closest point in B

### 2.3 IoU (Jaccard Index)

Used in: `src/evaluation/evaluator.py`

```
IoU(P, G) = |P ∩ G| / |P ∪ G|
```

Relationship to Dice: `IoU = Dice / (2 - Dice)`

### 2.4 Sensitivity (True Positive Rate / Recall)

Used in: `src/evaluation/evaluator.py`

```
Sensitivity = TP / (TP + FN)
```

Where TP = true positives, FN = false negatives.

### 2.5 Specificity (True Negative Rate)

Used in: `src/evaluation/evaluator.py`

```
Specificity = TN / (TN + FP)
```

Where TN = true negatives, FP = false positives.

---

## 3. XAI Method Formulas

### 3.1 Grad-CAM

Used in: `src/xai/grad_cam.py`

**Importance weights:**
```
α_k^c = (1/Z) × Σ_d Σ_h Σ_w  ∂y^c / ∂A^k_{d,h,w}
```

**Heatmap:**
```
L^c = ReLU( Σ_k  α_k^c × A^k )
```

**Segmentation score:**
```
y^c = mean(output[0, c])    (spatial average of logits for class c)
```

### 3.2 Guided Backpropagation

Used in: `src/xai/guided_backprop.py`

**Modified ReLU backward pass:**
```
∂R/∂x = ∂R/∂f × 𝟙(f > 0) × 𝟙(∂R/∂f > 0)
```

**Saliency:**
```
S(d, h, w) = max_c |∂Score / ∂x_{c,d,h,w}|
```

### 3.3 Guided Grad-CAM

Used in: `scripts/generate_gbp.py` (with `--guided_gradcam` flag)

```
Guided_Grad-CAM = GBP ⊙ upsample(Grad-CAM)
```

---

## 4. XAI Evaluation Metrics

### 4.1 Pointing Game

Used in: `src/xai/metrics.py`

```
Pointing_Game(S, G) = 𝟙(G[argmax(S)] = 1)
```

### 4.2 Saliency Coverage

Used in: `src/xai/metrics.py`

```
Coverage(S, G) = Σ(S × G) / Σ(S)
```

### 4.3 Saliency IoU

Used in: `src/xai/metrics.py`

```
IoU(S, G, τ) = |𝟙(S ≥ τ) ∩ G| / |𝟙(S ≥ τ) ∪ G|
```

### 4.4 Weighted Dice

Used in: `src/xai/metrics.py`

```
Weighted_Dice(S, G) = (2 × Σ(S × G)) / (Σ(S) + Σ(G))
```

---

## 5. Data Preprocessing

### 5.1 BraTS Label Conversion

Used in: `src/data/transforms.py` → `ConvertToMultiChannelBraTS2023d`

```
Channel 0 (WT) = (label == 1) ∨ (label == 2) ∨ (label == 3)
Channel 1 (TC) = (label == 1) ∨ (label == 3)
Channel 2 (ET) = (label == 3)
```

### 5.2 Intensity Normalisation

Used in: `src/data/transforms.py` → `NormalizeIntensityd`

```
x_norm = (x - μ_nonzero) / σ_nonzero
```

Per-channel, computed only over non-zero voxels.

---

## 6. Training Optimisation

### 6.1 AdamW Optimiser

Used in: `src/models/factory.py`

```
θ_{t+1} = θ_t - lr × (m̂_t / (√v̂_t + ε) + λ × θ_t)
```

Where `m̂_t`, `v̂_t` = bias-corrected first/second moment estimates, λ = weight decay.

### 6.2 Cosine Annealing Learning Rate Schedule

Used in: `src/models/factory.py`

```
lr_t = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T_max))
```

Where t = current epoch, T_max = total epochs.
