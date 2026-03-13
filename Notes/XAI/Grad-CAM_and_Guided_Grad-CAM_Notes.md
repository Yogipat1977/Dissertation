# Grad-CAM & Guided Grad-CAM — Research Notes

**Date:** 11 March 2026  
**Context:** Understanding XAI techniques before implementation for the dissertation.

---

## 1. XAI Techniques Planned

The following techniques are planned for implementation:

1. **Grad-CAM** ✅ researched
2. **Guided Grad-CAM** ✅ researched
3. Integrated Gradients (IG)
4. Occlusion Sensitivity
5. Guided Backpropagation (GBP)
6. Layer-Wise Relevance Propagation (LRP)
7. Saliency Maps

---

## 2. Grad-CAM — How It Works

**Source:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, IJCV 2020

### Core Idea

Uses the gradients flowing into the **last convolutional layer** to compute per-feature-map importance weights, producing a **coarse, class-discriminative heatmap**.

### Formula (4 steps)

**Step 1 — Importance weights (α):**

```
α_k^c = (1/Z) × Σ_x Σ_y Σ_z  ∂y^c / ∂A^k_{x,y,z}
```

- Backprop the gradient of class score y^c w.r.t. feature map A^k
- Global-average-pool across all spatial dimensions → one scalar per feature map
- Large α_k → feature map k is important for class c

**Step 2 — Weighted sum:**

```
heatmap = α₁×Map₁ + α₂×Map₂ + ... + α_K×Map_K
```

**Step 3 — ReLU:**

```
heatmap = ReLU(heatmap)
```

Keeps only positive contributions (features that *support* the target class). Negative values belong to other classes.

**Step 4 — Upsample:**

Trilinear interpolation from feature map resolution back to input resolution.

### Key Properties

- **Post-hoc** — no retraining or architecture changes needed
- **Class-discriminative** — different heatmap for each class
- **Generalises CAM** — works on any CNN, not just GAP→softmax architectures

---

## 3. Guided Backpropagation (GBP)

### Modification to Standard Backprop

At each ReLU during backprop, GBP applies **two gates**:

1. Standard gate: zero gradient if forward activation ≤ 0
2. Additional gate: zero gradient if incoming gradient ≤ 0

### Result

Produces a **full-resolution, pixel-level** saliency map — crisp and detailed.

### Limitation

**NOT class-discriminative** — highlights all visually salient features regardless of which class you're interested in.

---

## 4. Guided Grad-CAM

### Formula

```
Guided_Grad-CAM = GBP ⊙ upsample(Grad-CAM)
```

(⊙ = element-wise multiplication)

### Why This Works

| Component | Provides | Lacks |
|-----------|----------|-------|
| Grad-CAM | Class discrimination, spatial localisation | Fine detail (coarse resolution) |
| GBP | Fine-grained pixel detail | Class discrimination |
| **Guided Grad-CAM** | **Both** | — |

Multiplying masks out GBP activations outside the Grad-CAM region → **high-resolution + class-specific** saliency.

---

## 5. Application to Our SegResNet Model

### Model Config (`full_training_segresnet.yaml`)

| Parameter | Value |
|-----------|-------|
| Architecture | SegResNet |
| init_filters | 32 |
| in_channels | 4 (T1c, T1n, T2f, T2w) |
| out_channels | 3 (WT, TC, ET) |
| roi_size | [160, 160, 160] |
| dropout_prob | 0.1 |
| Loss | DiceFocalLoss (gamma=2.0) |

### Encoder Filter Progression

| Encoder Level | Filters | Spatial Resolution |
|:---:|:---:|:---|
| Level 1 | 32 | 160 × 160 × 160 |
| Level 2 | 64 | 80 × 80 × 80 |
| Level 3 | 128 | 40 × 40 × 40 |
| Level 4 (bottleneck) | **256** | **20 × 20 × 20** |

Doubling rule: `init_filters × 2^(level-1)` → Level 4 = 32 × 8 = 256

### Grad-CAM at the Bottleneck

- **256 feature maps** of size **20 × 20 × 20**
- Backprop → 256 gradient maps → global-average-pool → **256 alpha values**
- Weighted sum → one **20 × 20 × 20** heatmap
- ReLU → trilinear upsample → **160 × 160 × 160** saliency volume
- Export as NIfTI for 3D Slicer / SlicerVR

### All 3 Output Channels Can Be Explained

Grad-CAM is **not limited to ET**. You generate a separate heatmap per class:

| Channel | Class | What the heatmap reveals |
|:---:|:---|:---|
| 0 | **Whole Tumor (WT)** | Where the model detects the overall tumor boundary |
| 1 | **Tumor Core (TC)** | What drives the necrotic core + enhancing region identification |
| 2 | **Enhancing Tumor (ET)** | Why the model marks actively enhancing areas |

Each produces a **different** saliency volume because the network uses different features for each sub-region. This means per patient you would export 3 Grad-CAM NIfTIs (e.g. `patient_gradcam_wt.nii.gz`, `_tc.nii.gz`, `_et.nii.gz`).

### For Segmentation (vs. Classification)

Since segmentation outputs a per-voxel score map (not a single scalar), we spatially average the output logits for the target class:

```python
score = output[0, channel].mean()   # collapse spatial dims → scalar
score.backward()                     # then standard Grad-CAM flow
```

---

## 6. Layer Choice Considerations

| Target Layer | Resolution | Trade-off |
|---|---|---|
| **Bottleneck (Level 4)** | 20³ | Best class discrimination, coarsest spatial detail |
| **Level 3** | 40³ | Good balance between semantics and spatial detail |
| **Last decoder conv** | 160³ | Best spatial detail, weakest class discrimination |

The bottleneck is the standard choice. Trying multiple layers and comparing is good practice for the dissertation.

---

## 7. Key Papers Referenced

1. **Selvaraju et al., 2017/2020** — Original Grad-CAM paper (Grad_CAM.pdf)
2. **Zeineldin et al., 2022** — NeuroXAI framework, applied Grad-CAM in 3D for BraTS (NeuroXAI.pdf)
3. **Jeya Mala et al., 2025** — Visualizing UNet Decisions, XAI for brain MRI segmentation (Visualizing_UNet_Decisions.pdf)
