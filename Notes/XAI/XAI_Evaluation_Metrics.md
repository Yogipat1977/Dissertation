# XAI Evaluation Metrics — Detailed Documentation

**Author:** Yogi Amitkumar Patel  
**Implementation:** `src/xai/metrics.py`  
**Last Updated:** 13 March 2026

---

## Purpose

These metrics quantitatively answer: **"Is the model looking at the right region for the right reasons?"**

Unlike model evaluation metrics (Dice, HD95) which compare the model's *prediction* to ground truth, XAI evaluation metrics compare the model's *attention* (saliency map) to ground truth.

| Model Evaluation | XAI Evaluation |
|-----------------|----------------|
| Prediction vs Ground Truth | Saliency vs Ground Truth |
| "Is the output correct?" | "Is the reasoning correct?" |
| Binary mask vs Binary mask | Continuous map vs Binary mask |

---

## Metric 1: Pointing Game

### What It Measures

Does the model's **single most important voxel** fall inside the actual tumor?

This is the simplest possible check — a binary "hit or miss" test.

### Formula

```
Pointing_Game(S, G) = 𝟙(G[argmax(S)] = 1)
```

Where:
- `S` = 3D saliency map with values in [0, 1]
- `G` = 3D binary ground truth mask (1 = tumor, 0 = background)
- `argmax(S)` = (d, h, w) coordinates of the voxel with the highest saliency value
- `𝟙(·)` = indicator function (returns 1 if true, 0 if false)

### How It Works

1. Find the single voxel with the maximum saliency value across the entire 3D volume
2. Check if that voxel falls inside the ground truth tumor mask
3. If yes → Hit (1). If no → Miss (0)

### Aggregation Across Patients

```
Pointing_Game_Accuracy = (Number of Hits / Total Patients) × 100%
```

### Implementation

```python
def pointing_game(saliency, ground_truth):
    peak_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    return bool(ground_truth[peak_idx] == 1)
```

### Interpretation

| Value | Meaning |
|-------|---------|
| 100% accuracy | Peak attention is always inside tumor — model focuses correctly |
| 50% accuracy | Random chance — model may be confused |
| 0% accuracy | Peak attention is always outside tumor — model looks at wrong areas |

### Reference

Zhang et al., "Top-Down Neural Attention by Excitation Backprop", IJCV 2018.

---

## Metric 2: Saliency Coverage

### What It Measures

What **fraction of the model's total attention** is focused on the actual tumor? This captures how well the model concentrates its "energy" on the relevant region.

### Formula

```
Coverage(S, G) = Σ(S ⊙ G) / Σ(S)
```

Where:
- `Σ(S ⊙ G)` = sum of saliency values **inside** the tumor mask
- `Σ(S)` = sum of **all** saliency values (inside + outside tumor)

Equivalently:
```
Coverage = (Saliency mass inside tumor) / (Total saliency mass)
```

### How It Works

1. Multiply the saliency map element-wise by the GT mask → keeps only saliency inside the tumor
2. Sum those values → this is the "attention on tumor"
3. Sum all saliency values → this is the "total attention"
4. Divide: coverage = attention on tumor / total attention

### Implementation

```python
def saliency_coverage(saliency, ground_truth):
    total = saliency.sum()
    if total < 1e-8:
        return 0.0
    inside = saliency[ground_truth == 1].sum()
    return float(inside / total)
```

### Interpretation

| Value | Meaning |
|-------|---------|
| 1.0 | 100% of attention is inside the tumor — perfect focus |
| 0.75 | 75% of attention on tumor, 25% leaked to background |
| 0.5 | Equal attention on tumor and background |
| 0.0 | No attention on tumor at all |

### Strengths & Limitations

- **Strength:** Insensitive to saliency shape — only cares about the distribution of attention mass
- **Limitation:** A coverage of 0.9 doesn't mean the saliency *shape* matches the tumor — it could be concentrated in a small part of a large tumor

---

## Metric 3: Saliency IoU (Intersection over Union)

### What It Measures

How well does the **shape** of the high-attention region match the shape of the tumor? This is the strictest shape-matching metric.

### Formula

First, threshold the saliency to create a binary mask:
```
S_binary(i) = 𝟙(S(i) ≥ τ)     where τ = 0.5 (default)
```

Then compute standard IoU:
```
IoU(S_binary, G) = |S_binary ∩ G| / |S_binary ∪ G|
```

Where:
- `|S_binary ∩ G|` = number of voxels that are both high-saliency AND in the tumor
- `|S_binary ∪ G|` = number of voxels that are either high-saliency OR in the tumor (or both)

### How It Works

1. Threshold the saliency map at `τ = 0.5` → binary "attention mask"
2. Create binary GT mask
3. Compute intersection (AND) and union (OR) of both masks
4. IoU = intersection / union

### Implementation

```python
def saliency_iou(saliency, ground_truth, threshold=0.5):
    s_binary = (saliency >= threshold).astype(np.uint8)
    gt_binary = (ground_truth >= 1).astype(np.uint8)
    intersection = (s_binary & gt_binary).sum()
    union = (s_binary | gt_binary).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)
```

### Interpretation

| Value | Meaning |
|-------|---------|
| 1.0 | Perfect overlap — attention shape exactly matches tumor shape |
| 0.5 | Moderate overlap |
| 0.0 | No overlap at all |

### Why Grad-CAM Scores Low

Grad-CAM produces a blurry blob that spreads far beyond the tumor boundary. After thresholding at 0.5, the remaining high-saliency region is typically **much smaller** than the actual tumor (because the blob's peak is sharp but the edges are diffuse). This leads to a very small intersection relative to the union, resulting in low IoU scores (typically 0.01–0.10 for Grad-CAM).

---

## Metric 4: Weighted Dice (Our Adapted Metric)

### What It Measures

How well does the **continuous saliency distribution** match the tumor? This is our adaptation of the standard Dice coefficient for evaluating continuous saliency against binary ground truth.

### Formula

```
Weighted_Dice(S, G) = (2 × Σ(S × G)) / (Σ(S) + Σ(G))
```

Where:
- `S` = continuous saliency map with values in [0, 1]
- `G` = binary ground truth mask (0 or 1)
- `Σ(S × G)` = sum of saliency values **inside** the tumor (only voxels where G = 1 contribute)
- `Σ(S)` = total saliency mass across the entire volume
- `Σ(G)` = total number of tumor voxels (volume of the tumor)

### Derivation from Standard Dice

**Standard Dice** (for two binary masks P and G):
```
Dice(P, G) = (2 × |P ∩ G|) / (|P| + |G|)
```

**Weighted Dice** replaces the binary prediction `P` with continuous saliency `S`:
```
S ∈ [0, 1]  instead of  P ∈ {0, 1}
```

This turns the hard intersection `|P ∩ G|` into a soft intersection `Σ(S × G)`, where voxels with high saliency inside the tumor contribute more than voxels with low saliency.

### Walk-Through Example

```
Voxel:          [  1   2   3   4   5   6   7   8  ]
Saliency (S):   [0.9, 0.8, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0]
Ground Truth:   [  1,   1,   1,   0,   0,   0,   0,   0 ]

Numerator:   2 × (0.9×1 + 0.8×1 + 0.5×1 + 0.3×0 + ...)
           = 2 × (0.9 + 0.8 + 0.5)
           = 2 × 2.2 = 4.4

Denominator: (0.9 + 0.8 + 0.5 + 0.3 + 0.1 + 0 + 0 + 0) + (1 + 1 + 1 + 0 + ...)
           = 2.6 + 3.0 = 5.6

Weighted Dice = 4.4 / 5.6 = 0.786  ← Good! Attention matches tumor
```

Now if the model looked at the **wrong area**:
```
Saliency (S):   [0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 0.9]
Ground Truth:   [  1,   1,   1,   0,   0,   0,   0,   0 ]

Numerator:   2 × (0.1×1 + 0 + 0) = 2 × 0.1 = 0.2
Denominator: (0.1 + 0 + 0 + 0 + 0 + 0.3 + 0.8 + 0.9) + (1 + 1 + 1)
           = 2.1 + 3.0 = 5.1

Weighted Dice = 0.2 / 5.1 = 0.039  ← Bad! Attention is on the wrong area
```

### Implementation

```python
def weighted_dice(saliency, ground_truth):
    numerator = 2.0 * (saliency * ground_truth).sum()
    denominator = saliency.sum() + ground_truth.sum()
    if denominator < 1e-8:
        return 0.0
    return float(numerator / denominator)
```

### Why Weighted Dice Is Better Than Binary IoU for XAI

| Aspect | Saliency IoU | Weighted Dice |
|--------|-------------|---------------|
| Saliency treatment | Binary (threshold at τ) | Continuous [0, 1] |
| Information loss | High — discards all intensity info | Low — uses full gradient |
| Sensitivity | Harsh — small threshold change drastically alters score | Smooth — gradual changes |
| Use case | "Is the shape right?" | "Is the distribution right?" |

### Why Grad-CAM Scores Low on Weighted Dice

The denominator `Σ(S)` includes all saliency everywhere — including the "leaked" saliency outside the tumor from Grad-CAM's blurry upsampling. This inflates the denominator without contributing to the numerator, dragging the score down.

**Guided Grad-CAM** fixes this by removing leaked saliency (via multiplication with GBP), making `Σ(S)` much closer to `Σ(G)`.

---

## Summary Table

| Metric | Input Types | Range | Penalises |
|--------|------------|-------|-----------|
| Pointing Game | Peak voxel + binary GT | {0, 1} | Wrong peak location |
| Saliency Coverage | Continuous S + binary GT | [0, 1] | Saliency leaking to background |
| Saliency IoU | Binary S (thresholded) + binary GT | [0, 1] | Shape mismatch |
| Weighted Dice | Continuous S + binary GT | [0, 1] | Both shape mismatch and saliency leakage |

---

## Combined Evaluation Function

All four metrics are computed in one call via `evaluate_saliency()`:

```python
from src.xai.metrics import evaluate_saliency

results = evaluate_saliency(saliency_map, ground_truth, threshold=0.5)
# Returns: {'pointing_game': 1.0, 'coverage': 0.75, 'iou': 0.05, 'weighted_dice': 0.26}
```
