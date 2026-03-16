# MC Dropout (Monte Carlo Dropout) — Uncertainty Quantification for 3D CNNs

**Date:** 16 March 2026

---

## Handwritten Notes

![MC Dropout handwritten notes — 16 Mar 2026](mc_dropout_handwritten_notes.jpg)

---

## 1. What is MC Dropout?

MC Dropout (Monte Carlo Dropout) is an uncertainty quantification technique. Standard deep learning models are **deterministic** — they will give the same output when you give the same image twice. MC Dropout introduces **stochasticity** at inference time by keeping dropout layers active.

### Dropout — Standard Use
- Dropout is a network layer that "turns off" certain neurons randomly during training to prevent overfitting.
- During normal inference (`model.eval()`), dropout is disabled — all neurons contribute.

### MC Dropout — The Key Insight
- **We keep those dropped neurons turned on while testing** to check the uncertainty.
- Process the **same** test image **multiple times** with dropout layers active.
- Each forward pass will produce a **different** prediction because **different neurons will be deactivated** each time.
- This provides uncertainty estimation from the **multiple outputs**.

---

## 2. Variance as Uncertainty

- **Low Variance** (Uncertainty maps): All versions of prediction agree — "this voxel **is** tumor." → The model is **confident**.
- **High Variance**: Some passes say "tumor", other passes say "healthy" → The model is **uncertain** about this region.

---

## 3. No Retraining Needed

A critical advantage: **No model retraining is needed.** We already have the dropout layer enabled in the SegResNet configuration (`dropout_prob: 0.1`). MC Dropout uses the model's saved weights — we are just doing inference twice (or more) after keeping the dropout layer on during testing.

---

## 4. "The Boundary Glow"

MC Dropout uncertainty typically concentrates at **tumor boundaries** — the regions where tumor meets healthy tissue. This is expected and clinically meaningful:

- **Good results (well-segmented tumors):** High uncertainty appears **only at the boundary** of the tumor — a "glow" around the edges. The interior of the tumor and the clear healthy tissue have low uncertainty.
- **Bad results (poorly segmented regions):** Uncertainty is **concentrated throughout** the tumor region — the model is unsure about the entire structure.

---

## 5. The Critical Comparison: LRP vs Uncertainty

This is a novel analysis comparing per-voxel **feature importance** (from LRP) against per-voxel **uncertainty** (from MC Dropout):

| LRP Importance | Uncertainty | Interpretation |
|:-:|:-:|---|
| ↑ High | ↓ Low | **✅ Ideal** — Model relies heavily on this region AND is confident about it |
| ↑ High | ↑ High | **⚠️ Problem / Brittle** — Model relies on this region BUT is uncertain → prediction could flip |

This comparison reveals **brittle decision regions** where the model's most important features are also its most uncertain — a clinical red flag.

---

## 6. Exception Case: Patient 00291

Patient `BraTS-GLI-00291-000` is flagged as an exception case. Use the uncertainty map to investigate this patient specifically — the XAI metrics showed unusual behaviour (TC Pointing Game = 0%, ET Pointing Game = 0%) suggesting the model's attention was misaligned for this patient.

---

## 7. Implementation Design

### Class: `MCDropout3D` in `src/xai/uncertainty.py`

1. Set model to eval mode, then re-enable only the Dropout layers to `train()` mode.
2. Run $N$ forward passes (default $N = 20$) with the same input.
3. Apply sigmoid to get probabilities $\in [0, 1]$.
4. Stack all $N$ outputs.
5. Compute:
   - **Mean Prediction:** $\bar{p} = \frac{1}{N} \sum_{n=1}^{N} p_n$ → A stabilised, ensemble-like mask.
   - **Uncertainty Map:** $\sigma^2 = \frac{1}{N} \sum_{n=1}^{N} (p_n - \bar{p})^2$ → Per-voxel variance.

### Export Outputs

| Output | Format | Location |
|--------|--------|----------|
| Mean prediction (WT/TC/ET) | `.nii.gz` | `slicer_export/XAI/MC_Dropout/<patient>/<patient>_mc_mean_{wt,tc,et}.nii.gz` |
| Uncertainty map (WT/TC/ET) | `.nii.gz` | `slicer_export/XAI/MC_Dropout/<patient>/<patient>_mc_uncertainty_{wt,tc,et}.nii.gz` |
| Metrics CSV | `.csv` | `results/CSVs/xai_mc_dropout_metrics.csv` |

### Evaluation Metrics

The following metrics are computed for MC Dropout:

| Metric | What It Measures |
|--------|-----------------|
| **Uncertainty Area Ratio (UAR)** | Fraction of total uncertainty mass that falls inside the GT mask |
| **Boundary Uncertainty Ratio** | Fraction of total uncertainty that falls within a dilated boundary band of the GT mask |
| **Mean Uncertainty (Inside)** | Average variance inside the GT tumor — lower = more confident |
| **Mean Uncertainty (Outside)** | Average variance outside the GT tumor — should also be low |
| **Weighted Dice (LRP)** | Soft overlap of the LRP relevance map vs the GT tumor mask |
| **Saliency-Uncertainty Correlation** | Pearson correlation between LRP importance and Uncertainty inside the tumor |

### Script: `scripts/generate_mc_dropout.py`

```bash
python scripts/generate_mc_dropout.py \
  --config configs/full_training_segresnet.yaml \
  --checkpoint models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth \
  --num_iters 20 \
  --limit 5
```
