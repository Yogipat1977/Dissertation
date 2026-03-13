# XAI Implementation — Detailed Documentation

**Author:** Yogi Amitkumar Patel  
**Project:** Illuminating the Black Box — Explainable AI Framework for Brain Tumor Segmentation  
**Last Updated:** 13 March 2026

---

## 1. Overview

This document describes the complete XAI (Explainable AI) pipeline implemented for interpreting the 3D SegResNet brain tumor segmentation model. The goal is to answer: **"Where is the model looking when it makes a segmentation decision, and does that match the actual tumor?"**

Three XAI methods are implemented:

| Method | Resolution | Class-Specific? | Sharpness |
|--------|-----------|-----------------|-----------|
| **Grad-CAM** | Coarse (20³ → 160³) |  Yes | Low (blurry blob) |
| **Guided Backpropagation (GBP)** | Full (160³) | No | High (sharp edges) |
| **Guided Grad-CAM** | Full (160³) |  Yes | High (best of both) |

All methods are evaluated against ground truth using four quantitative metrics (see `Notes/XAI/XAI_Evaluation_Metrics.md`).

---

## 2. Method 1: Grad-CAM (Gradient-weighted Class Activation Mapping)

### 2.1 Core Concept

Grad-CAM uses gradients flowing into a target convolutional layer to compute per-feature-map importance weights. These weights are used to create a **coarse, class-discriminative heatmap** showing which spatial regions contributed most to the model's prediction for a specific class.

### 2.2 Mathematical Derivation

**Step 1 — Compute importance weights (α):**

For target class `c` and feature map `k` of the target layer with activations `A^k`:

```
α_k^c = (1 / Z) × Σ_d Σ_h Σ_w  ∂y^c / ∂A^k_{d,h,w}
```

Where:
- `y^c` = spatially-averaged output logit for class `c` (for segmentation: `output[0, c].mean()`)
- `A^k` = activation of the k-th feature map at the target layer
- `Z = D' × H' × W'` = number of spatial elements in the feature map
- The partial derivative is computed via backpropagation

This is a **global average pooling of the gradients** — it produces one scalar weight per feature map channel.

**Step 2 — Weighted combination of feature maps:**

```
L^c = Σ_k  α_k^c × A^k
```

The heatmap is a linear combination of all feature maps, weighted by their importance for class `c`.

**Step 3 — ReLU:**

```
L^c_GradCAM = ReLU(L^c)
```

ReLU discards negative values, keeping only features that **positively contribute** to the target class. Negative values would correspond to features that belong to other classes.

**Step 4 — Upsample:**

Trilinear interpolation from feature-map resolution (e.g. 20 × 20 × 20) back to input resolution (160 × 160 × 160).

**Step 5 — Normalise:**

```
L^c_norm = (L^c - min(L^c)) / (max(L^c) - min(L^c))
```

Scales the heatmap to [0, 1] range.

### 2.3 Application to SegResNet (our model)

| Parameter | Value |
|-----------|-------|
| Target layer | `model.down_layers[-1]` (bottleneck, Level 4) |
| Feature map resolution | 20 × 20 × 20 |
| Number of channels | 256 |
| Output resolution | 160 × 160 × 160 (after upsampling) |
| Classes | 0 = Whole Tumor, 1 = Tumor Core, 2 = Enhancing Tumor |

For segmentation (vs classification), we spatially average the output logits:
```python
score = output[0, target_class].mean()   # scalar
score.backward()                          # standard backprop
```

### 2.4 Script Usage & Run Commands

The `generate_gradcam.py` script computes Grad-CAM heatmaps and inline metrics for each patient.

**Command:**
```bash
python scripts/generate_gradcam.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--limit` | `0` (all) | Number of patients to process (useful for testing) |
| `--layer` | `bottleneck` | Target layer for gradients |
| `--split` | `test` | Dataset split to evaluate |

### 2.5 Output

| Output | Location |
|--------|----------|
| NIfTI heatmaps | `slicer_export/XAI/Grad_CAM/<patient>/<patient>_gradcam_{wt,tc,et}.nii.gz` |
| Metrics CSV | `results/CSVs/xai_gradcam_metrics.csv` |

### 2.6 Reference

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", IJCV 2020.

---

## 3. Method 2: Guided Backpropagation (GBP)

### 3.1 Core Concept

Standard backpropagation computes the gradient of the output with respect to the input. GBP modifies this by adding an **additional gate at every ReLU layer** during the backward pass: it zeros out not only gradients where the forward activation was negative (standard ReLU gate), but also where the incoming gradient itself is negative.

This keeps only the purely positive signal paths, producing a **full-resolution, pixel-level saliency map** with sharp edges and fine detail.

### 3.2 Mathematical Derivation

At each ReLU layer during backpropagation, standard backprop applies:

```
Standard:  ∂R/∂x = ∂R/∂f × 𝟙(f > 0)
```

Where `f = ReLU(x)` is the forward activation and `𝟙(f > 0)` is the indicator function.

GBP adds a second gate on the gradient itself:

```
Guided:    ∂R/∂x = ∂R/∂f × 𝟙(f > 0) × 𝟙(∂R/∂f > 0)
```

In code, this is implemented as a backward hook on ReLU:
```python
def guided_relu_hook(module, grad_input, grad_output):
    return (F.relu(grad_output[0]),)   # clamp negative gradients
```

After backprop reaches the input, the saliency is:

```
Saliency = |∇_x Score|    (absolute value of input gradient)
```

For multi-channel input (4 MRI modalities), we take the **max absolute gradient across channels**:
```
S(d, h, w) = max_c |∂Score / ∂x_{c,d,h,w}|
```

Then normalise to [0, 1].

### 3.3 Key Properties

- **Full resolution** — output matches input size (160 × 160 × 160)
- **NOT class-discriminative** — highlights all visually salient features, regardless of which class
- **Sharp edges** — shows fine-grained detail at individual voxel level

### 3.4 Script Usage & Run Commands

The `generate_gbp.py` script computes Guided Backpropagation saliency maps and inline metrics.

**Command (GBP only):**
```bash
python scripts/generate_gbp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--limit` | `0` (all) | Number of patients to process |
| `--split` | `test` | Dataset split to evaluate |

### 3.5 Output

| Output | Location |
|--------|----------|
| NIfTI heatmaps | `slicer_export/XAI/GBP/<patient>/<patient>_gbp_{wt,tc,et}.nii.gz` |
| Metrics CSV | `results/CSVs/xai_gbp_metrics.csv` |

### 3.6 Reference

Springenberg et al., "Striving for Simplicity: The All Convolutional Net", ICLR Workshop 2015.

---

## 4. Method 3: Guided Grad-CAM

### 4.1 Core Concept

Guided Grad-CAM combines the strengths of both Grad-CAM and GBP through element-wise multiplication:

| Component | Provides | Lacks |
|-----------|----------|-------|
| Grad-CAM | Class discrimination, spatial localisation | Fine detail (blurry) |
| GBP | Fine-grained pixel detail | Class discrimination |
| **Guided Grad-CAM** | **Both** | — |

### 4.2 Mathematical Formula

```
Guided_Grad-CAM = GBP ⊙ upsample(Grad-CAM)
```

Where `⊙` is element-wise (Hadamard) multiplication.

**Why this works:** The Grad-CAM heatmap acts as a soft spatial mask. The GBP map has sharp detail everywhere in the brain, but multiplying by Grad-CAM zeros out all GBP activations outside the class-specific region. The result is **sharp detail, restricted to the tumor area** relevant to the target class.

After multiplication, the result is re-normalised to [0, 1]:
```
S_norm = (S - min(S)) / (max(S) - min(S))
```

### 4.3 Expected Improvement over Grad-CAM

Because Guided Grad-CAM produces tight, sharp saliency that hugs tumor boundaries (instead of Grad-CAM's blurry blob that leaks into healthy tissue), the following metric improvements are expected:

| Metric | Grad-CAM (expected) | Guided Grad-CAM (expected) |
|--------|---------------------|---------------------------|
| Pointing Game | High (~90%+) | Similar or higher |
| Coverage | High (~75%+) | Similar |
| Saliency IoU | Low (~0.05) | Significantly higher |
| Weighted Dice | Low (~0.15) | Significantly higher |

### 4.4 Script Usage & Run Commands

Guided Grad-CAM uses the same `generate_gbp.py` script, but with the `--guided_gradcam` flag. This flag tells the script to generate *both* GBP and Grad-CAM, and then multiply them together.

**Command:**
```bash
python scripts/generate_gbp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 5 \
  --guided_gradcam
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--guided_gradcam` | `False` | Enables computation of Guided Grad-CAM |
| `--layer` | `bottleneck` | Target layer for the Grad-CAM component |

### 4.5 Output

| Output | Location |
|--------|----------|
| NIfTI heatmaps | `slicer_export/XAI/Guided_Grad_CAM/<patient>/<patient>_guided_gradcam_{wt,tc,et}.nii.gz` |
| Metrics CSV | `results/CSVs/xai_guided_gradcam_metrics.csv` |

### 4.6 Reference

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", IJCV 2020 (Section 4.1: Guided Grad-CAM).

---

## 5. Quantitative XAI Validation

All saliency maps are evaluated against ground truth using four metrics implemented in `src/xai/metrics.py`. See `Notes/XAI/XAI_Evaluation_Metrics.md` for the detailed mathematical formulas and derivations.

| Metric | What It Measures | Formula Reference |
|--------|-----------------|-------------------|
| Pointing Game | Is the peak saliency inside the tumor? | Zhang et al., IJCV 2018 |
| Saliency Coverage | What fraction of attention is on the tumor? | Custom implementation |
| Saliency IoU | Shape overlap (binary threshold) | Standard IoU adapted for saliency |
| Weighted Dice | Soft shape overlap (continuous) | Adaptation of Dice for continuous saliency |

---

## 6. Spatial Alignment Design Decision

A critical design choice: **XAI metrics are computed inline during saliency generation**, not as a separate post-processing step. This ensures that both the saliency map and the ground truth label exist in the **same MONAI-preprocessed coordinate space** (after `CropForeground`, `NormalizeIntensity`, `SpatialPad`).

If computed separately (e.g. loading saved NIfTI files), the affine transformations may not perfectly align, leading to incorrect metric values. The inline approach guarantees spatial correspondence.

---

## 7. Input Padding Design Decision

SegResNet uses 4 encoder levels with stride-2 downsampling. The input spatial dimensions must be divisible by `2^4 = 16` for the encoder-decoder path to produce matching dimensions. After `CropForeground`, some patients have odd dimensions (e.g. 157 × 193 × 141).

To handle this, both `generate_gradcam.py` and `generate_gbp.py` pad the input to the next multiple of 16 before inference, then crop the saliency map back to the original preprocessed size. This prevents `RuntimeError`s from dimension mismatches in the skip connections.
