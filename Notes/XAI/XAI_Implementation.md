# XAI Implementation — Detailed Documentation

**Author:** Yogi Amitkumar Patel  
**Project:** Illuminating the Black Box — Explainable AI Framework for Brain Tumor Segmentation  
**Last Updated:** 14 March 2026

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
| `--topk` | `0` (disabled) | Top-K% thresholding (e.g. `15` for top 15%) |
| `--split` | `test` | Dataset split to evaluate |

### 2.5 Output

| Output | Location |
|--------|----------|
| NIfTI heatmaps | `slicer_export/XAI/Grad_CAM/<patient>/<patient>_gradcam_{wt,tc,et}.nii.gz` |
| Metrics CSV | `results/CSVs/xai_gradcam_metrics.csv` |

### 2.6 Related Files

| File | Purpose |
|------|---------|
| `src/xai/grad_cam.py` | `GradCAM3D` class — hooks, forward/backward, heatmap generation |
| `scripts/generate_gradcam.py` | Batch generation + inline metrics computation |
| `scripts/evaluate_gradcam_coarse.py` | Coarse-resolution evaluation (GT downsampled to feature map size) |

### 2.7 Reference

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

### 3.6 Related Files

| File | Purpose |
|------|---------|
| `src/xai/guided_backprop.py` | `GuidedBackprop3D` class — ReLU hook overrides, gradient extraction |
| `scripts/generate_gbp.py` | Batch generation + inline metrics (also supports Guided Grad-CAM) |

### 3.7 Reference

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

### 4.3 Actual Results (10 test patients, encoder3 layer)

| Metric (WT) | Grad-CAM (bottleneck) | GBP | Guided Grad-CAM (encoder3) |
|-------------|:---:|:---:|:---:|
| Pointing Game | 93% | **100%** | **100%** |
| Coverage | 0.75 | 0.22 | **0.93** |
| IoU | 0.05 | 0.008 | 0.004 |
| Weighted Dice | 0.15 | 0.13 | 0.13 |

| Metric (ET) | Grad-CAM (bottleneck) | GBP | Guided Grad-CAM (encoder3) |
|-------------|:---:|:---:|:---:|
| Pointing Game | 93% | **100%** | **100%** |
| Coverage | 0.51 | 0.15 | **0.57** |
| IoU | 0.05 | 0.07 | **0.06** |
| Weighted Dice | 0.15 | 0.14 | **0.22** |

**Key findings:**
- Guided Grad-CAM achieves **93% Coverage** for WT — meaning 93% of model attention is inside the tumor
- Enhancing Tumor Weighted Dice improves from 0.15 (Grad-CAM) to **0.22** (Guided Grad-CAM)
- GBP alone has low Coverage (0.22) confirming it is NOT class-discriminative
- All methods achieve 100% Pointing Game except standalone Grad-CAM (93%)

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

### 4.6 Related Files

| File | Purpose |
|------|---------|
| `src/xai/grad_cam.py` | Grad-CAM component |
| `src/xai/guided_backprop.py` | GBP component |
| `scripts/generate_gbp.py` | Combined generation (use `--guided_gradcam` flag) |

### 4.7 Reference

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

---

## 8. Hook Isolation Design Decision (Guided Grad-CAM)

When computing Guided Grad-CAM, both GBP backward hooks (on every ReLU) and Grad-CAM hooks (on the target layer) must coexist on the same model. If both hook systems are active simultaneously, they **corrupt each other's gradients**:

- GBP hooks gate negative gradients at every ReLU, altering the gradient signal that Grad-CAM captures
- Grad-CAM hooks intercept gradients at the bottleneck, modifying the flow that GBP relies on

**Observed impact:** GBP Pointing Game dropped from 100% to 80%, and Weighted Dice halved when hooks were active simultaneously.

**Solution:** Per-patient sequential hook isolation:
1. **STEP 1:** Create Grad-CAM hooks on clean model → compute heatmaps → remove hooks
2. **STEP 2:** Create GBP hooks (no Grad-CAM hooks active) → compute GBP saliency → remove hooks
3. **STEP 3:** Multiply stored results: `Guided_Grad-CAM = GBP × Grad-CAM`

Additionally, MONAI SegResNet uses `inplace=True` ReLU operations. These conflict with PyTorch backward hooks, so GBP disables `inplace` before registering hooks and restores it on cleanup.

---

## 9. Coarse-Resolution Evaluation (Downsampled GT)

Grad-CAM produces heatmaps at native feature map resolution (e.g. 20³ for bottleneck). Upsampling to 160³ introduces blur which penalises IoU and Weighted Dice. To test whether the model's **coarse spatial attention** genuinely aligns with the tumor:

**Approach:** Instead of upsampling saliency, **downsample the ground truth** to match the native resolution.

```
Standard:  Saliency 20³ → upsample → 160³  vs  GT 160³     (blur penalty)
Coarse:    Saliency 20³              vs  GT 160³ → downsample → 20³  (fair comparison)
```

GT downsampling uses trilinear interpolation followed by thresholding at ≥0.5 (a block is tumor if >50% of its voxels are tumor).

### 9.1 Run Command

```bash
python scripts/evaluate_gradcam_coarse.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 10 --layer bottleneck
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | `bottleneck` | Target layer (bottleneck=20³, encoder3=40³) |
| `--limit` | `0` (all) | Number of patients to process |

### 9.2 Output

| Output | Location |
|--------|----------|
| Metrics CSV | `results/CSVs/xai_gradcam_coarse_<layer>_metrics.csv` |

### 9.3 Related Files

| File | Purpose |
|------|---------|
| `scripts/evaluate_gradcam_coarse.py` | Coarse-resolution Grad-CAM evaluation script |

> **Note:** This evaluation applies only to Grad-CAM. GBP and Guided Grad-CAM already operate at full input resolution (160³), so downsampling GT would lose information unnecessarily.

---

## 10. Top-K% Threshold Technique

### 10.1 Concept

Instead of using a fixed threshold (τ=0.5) to binarise the saliency heatmap, Top-K% thresholding **adaptively** retains only the top K% of saliency voxels by intensity and zeros out the rest. This is more robust because it adjusts to each heatmap's unique intensity distribution.

### 10.2 Formula

```
threshold = percentile(S, 100 - K)
S_topk(i) = S(i)  if S(i) ≥ threshold
             0     otherwise
```

The retained voxels keep their original continuous values (not binarised). Only Weighted Dice is evaluated on the thresholded map, as it measures soft overlap.

### 10.3 Run Commands

**On full-resolution Grad-CAM (`generate_gradcam.py`):**
```bash
python scripts/generate_gradcam.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 10 --topk 15
```

**On coarse-resolution Grad-CAM (`evaluate_gradcam_coarse.py`):**
```bash
python scripts/evaluate_gradcam_coarse.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 10 --layer bottleneck --topk 15
```

**On Guided Grad-CAM (`generate_gbp.py`):**
```bash
python scripts/generate_gbp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 10 --guided_gradcam --topk 15
```

### 10.4 Output

| Script | Output File |
|--------|-----------|
| `generate_gradcam.py --topk 15` | `results/CSVs/xai_gradcam_topk15_weighted_dice.csv` |
| `generate_gradcam.py --topk 15` | `slicer_export/XAI/Grad_CAM_TopK/<patient>/` (NIfTI) |
| `evaluate_gradcam_coarse.py --topk 15` | `results/CSVs/xai_gradcam_coarse_<layer>_topk15_weighted_dice.csv` |
| `generate_gbp.py --guided_gradcam --topk 15` | `results/CSVs/xai_guided_gradcam_topk15_weighted_dice.csv` |
| `generate_gbp.py --guided_gradcam --topk 15` | `slicer_export/XAI/Guided_Grad_CAM_TopK/<patient>/` (NIfTI) |

> **Note:** Standard (non-Top-K) exports and CSVs are always generated alongside. Top-K outputs go to separate files/folders.

---

## 11. Layer-wise Relevance Propagation (LRP)

### 11.1 Concept
Unlike gradient-based methods which measure sensitivity, LRP attributes exactly which input voxels contributed to the final output prediction score. By redistributing the output score backward using the **Input ✕ Gradient** proxy (mathematically equivalent to $\epsilon$-LRP in ReLU networks), LRP provides an ultra-high resolution positive relevance map identifying explicit structural evidence for the tumor.

### 11.2 Run Command
```bash
python scripts/generate_lrp.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 10
```

### 11.3 Output
| Output | Location |
|--------|----------|
| NIfTI Volumes | `slicer_export/XAI/LRP/<patient>/` |

> **Note:** LRP export intentionally skips the quantitative metric calculation to focus solely on high-resolution clinical visualisations.

---

## 12. Occlusion Sensitivity (Testing Regional Importance via Perturbation)

Occlusion Sensitivity is a completely model-agnostic XAI strategy. Instead of relying on gradients (which can be noisy) or internal layer feature maps, it systematically hides (occludes) 3D blocks of the input volume and measures the drop in the model's confidence for the tumor.

- **How it works:** A 16x16x16 window slides across the input. At every step, the pixels inside the window are set to 0. A forward pass is made. The drop in confidence (`baseline_score - occluded_score`) is recorded.
- **Why it matters:** It directly answers the question: "If the doctor could not see this specific brain region, would they still diagnose a tumor here?"
- **MSR Accuracy (Most Salient Region):** We evaluate this using MSR Accuracy. This metric finds the *single 3D window* that caused the largest drop in model confidence, and checks if that window was physically inside the true tumor mask. 

### 12.1 Run Command

This method takes longer than gradient-based methods because it requires a forward pass for every window position. It processes windows in batches.

```bash
python scripts/generate_occlusion.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --limit 20 \
  --window_size 16 \
  --stride 8 \
  --batch_size 16
```

### 12.2 Output
- **Metrics:** `results/CSVs/xai_occlusion_metrics.csv`
- **NIfTI Heatmaps:** `slicer_export/XAI/Occlusion/<patient>/<patient>_occlusion_{wt,tc,et}.nii.gz`

---

## 13. Summary of All Output Files

| File | Method | Resolution | Contents |
|------|--------|-----------|----------|
| `xai_gradcam_metrics.csv` | Grad-CAM | Full (160³) | Per-patient metrics |
| `xai_gradcam_topk<K>_weighted_dice.csv` | Grad-CAM + Top-K | Full (160³) | Weighted Dice only |
| `xai_gradcam_coarse_bottleneck_metrics.csv` | Grad-CAM | Native (20³) | Coarse metrics (downsampled GT) |
| `xai_gradcam_coarse_encoder3_metrics.csv` | Grad-CAM | Native (40³) | Coarse metrics (downsampled GT) |
| `xai_gradcam_coarse_<layer>_topk<K>_weighted_dice.csv` | Grad-CAM + Top-K | Native | Coarse Weighted Dice only |
| `xai_gbp_metrics.csv` | GBP | Full (160³) | Per-patient metrics |
| `xai_guided_gradcam_metrics.csv` | Guided Grad-CAM | Full (160³) | Per-patient metrics |
| `xai_guided_gradcam_topk<K>_weighted_dice.csv` | Guided Grad-CAM + Top-K | Full (160³) | Weighted Dice only |
| `slicer_export/XAI/LRP/<patient>/` | LRP | Full (160³) | NIfTI Export Only |
| `xai_summary_comparison.csv` | All methods | Full (160³) | Aggregated comparison table |
