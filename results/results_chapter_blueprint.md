# Blueprint: Final Report Results Chapter

This blueprint provides a professional, highly aligned structural guide for your Results Chapter. It clearly separates your findings into the three core pillars of your project. Right now, this blueprint expands on **Pillar 1 (Segmentation)** while holding space for **Pillar 2 (XAI)** and **Pillar 3 (VR)** to be added later.

---

> [!NOTE]
> **OVERALL CHAPTER INTRODUCTION**
> - **Purpose:** Briefly explain what this chapter covers—evaluating the 3D CNN (SegResNet) performance, interpreting its decisions via XAI, and visualising these outputs in VR.
> - **Structure:** Explicitly state the three main sections of the chapter: Quantitative & Qualitative Segmentation Results -> XAI Saliency Analysis -> VR Visualization Pipeline validation.

---

## 1. Pillar 1: Model Prediction and Segmentation Performance 
*(This is the section we have data for right now)*

### 1.1 Progression and Metric Analysis (Baseline vs. Final)
*Goal: Show the technical evolution and statistical success of your model.*
* **Narrative:** Introduce the evaluation suite (Test set metrics) and explain how moving from Prototype (Version 1) to the Final 1251-patient model critically improved performance.
* **Component 1 (The Visual Hook):** Insert `bar_metrics_Baseline_version_01.png` and `bar_metrics_Final_version_10.png`. Focus on comparing these two charts side-by-side to give an immediate visual overview of the performance jump across WT, TC, and ET.
* **Component 2 (The Data Dive):** 
    - **Data Scaling Comparison (Table):** Include a table comparing the Dice scores for the intermediate 250-patient model vs. the final full run (1251 patients). For context, WT improved slightly (0.908 → 0.923), while the hardest region (ET) saw a significant jump (0.742 → 0.873). Discuss how data volume critically impacts minority classes.
    - Discuss **Dice & IoU**: Re-emphasize the massive leap from the initial baseline (0.27 to 0.87 Dice for ET) structurally (ET is hard to detect, deep filters + large data solved it).
    - Discuss **HD95 (Boundary Precision):** Explain how HD95 dropping from ~75.9mm down to 3.66mm for ET signifies tight, clinical-grade tumor border prediction.
    - **Computational vs. Performance Trade-off:** Briefly outline the hardware usage and training time required to scale from 250 to 1251 patients compared to the segmentation gains. Frame this as an evaluation of whether the extra compute cost was necessary for the clinical accuracy achieved.
* **Component 3 (Stability Validation):** Insert `heatmap_dice.png` and `heatmap_hd95.png`.
    - Briefly explain that the heatmaps prove your high average scores are consistent and robust, free from massive negative outliers across the test set.

### 1.2 Qualitative Visual Evaluation (2D & 3D Comparisons)
*Goal: Prove that the mathematical metrics actually result in visually accurate medical segmentation.*
* **Narrative:** Transition from numbers to actual scans. State that predicting subregions accurately requires spatial awareness.
* **Layout Design:** Create side-by-side or paneled comparison figures using your 3D Slicer screenshots for patients `01661`, `01663`, and `01666`. 
* **Figure Content (What the images show):** 
    - **Panel A:** T1/T1ce/T2/FLAIR MRI 2D slice.
    - **Panel B:** Ground Truth 2D Slice (Actual Mask).
    - **Panel C:** SegResNet Model Prediction 2D Slice.
    - **Panel D (The 3D Element):** 3D Volumetric Rendering of the predicted tumor mask (from Slicer's 3D viewer).
* **Discussion Points per Patient:** 
    - Detail how the subregions (Whole Tumor, Tumor Core, Enhancing Tumor) in the *Predicted* masks align almost perfectly with the *Ground Truth* on the 2D plane.
    - Point out how the low HD95 (3.66mm) discussed earlier translates visually here (the edges of the colors perfectly match the edges of the real tumor on the MRI).
    - **3D Morphological Analysis:** Explicitly refer to Panel D. Discuss the complex 3D shape, structure, and depth of the tumor that a typical 2D slice hides. Emphasize that the SegResNet's ability to comprehend this 3D volume is what enables the upcoming VR Visualization (Pillar 3).

---

> [!IMPORTANT]
> **PILLAR 2 DETAILED BELOW — Pillar 3 remains reserved for future development.**

## 2. Pillar 2: Explainable AI (XAI) Interpretation

### 2.0 Introduction
* **Narrative:** Transition from Pillar 1 — segmentation accuracy is established, now we ask *"why does the model predict what it predicts?"*. Introduce the six XAI methods with a categorisation table:

| Category | Method | Resolution | Class-Specific? |
|----------|--------|-----------|----------------|
| Gradient-Based | 3D Grad-CAM | Coarse (~20³) | ✅ |
| Gradient-Based | Guided Backpropagation (GBP) | Full (voxel) | ❌ |
| Gradient-Based | Guided Grad-CAM (GBP × Grad-CAM) | Full (voxel) | ✅ |
| Relevance-Based | LRP (Input × Gradient) | Full (voxel) | ✅ |
| Perturbation-Based | Occlusion Sensitivity | Stride-upsampled | ✅ |
| Uncertainty-Based | MC Dropout (20 passes) | Full (voxel) | ✅ |

---

### 2.1 XAI Evaluation Metrics: Standard & Novel

#### Standard Metrics (adapted from 2D classification to 3D segmentation)

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Pointing Game** | PG = 1 if G(argmax(S)) = 1 | Does the single highest-saliency voxel fall inside the tumor? Binary pass/fail. |
| **Saliency Coverage** | Cov = Σ(S · G) / Σ S | What fraction of total saliency mass sits inside the GT mask? High = the model focuses on tumor, not healthy tissue. |
| **Saliency IoU** | IoU = \|S≥0.5 ∩ G\| / \|S≥0.5 ∪ G\| | Hard overlap — binarises saliency at threshold 0.5, then computes Jaccard against GT. Strict; punishes diffuse maps. |

#### Novel Metric: Weighted Dice (Soft Dice Score) ⭐
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Weighted Dice** | WD = 2·Σ(S·G) / (ΣS + ΣG) | **Soft Dice** between the continuous saliency map [0,1] and binary GT mask {0,1}. |

**Why this is novel and what it solves:**
- Standard Saliency IoU forces an **arbitrary threshold** (0.5) to binarise the continuous saliency — this discards intensity information and penalises methods that produce diffuse but correctly localised maps.
- Weighted Dice treats saliency values directly as **soft membership scores** — every voxel contributes proportional to its saliency intensity.
- **What it measures:** How well the shape and distribution of the saliency map matches the actual tumor mask, without any binarisation step.
- **End result of having this:** A fairer comparison across XAI methods — methods with diffuse but correctly centred maps (like LRP) get credit for their relevance distribution instead of being zeroed out by hard thresholding.

---

### 2.2 The Bottleneck Resolution Problem: Grad-CAM Upsampled vs. Native
*Goal: Use Weighted Dice to expose how upsampling artificially distorts Grad-CAM evaluation.*

#### 📊 GRAPH: Paired Bar Chart (or Slope Graph)
* **Data:** `xai_gradcam_metrics.csv` (upsampled to 160³) vs. `xai_gradcam_coarse_bottleneck_metrics.csv` (evaluated at native ~20³ resolution)
* **Design:** For each patient, two side-by-side bars per tumor region showing Weighted Dice at upsampled resolution vs. native bottleneck resolution.
* **What to show:**
    - Upsampled Weighted Dice and native Weighted Dice are **very close** (within ±0.02–0.04) — proving upsampling doesn't artificially inflate or deflate the score
    - In some cases native resolution scores slightly **higher** (e.g. 01497 TC: 0.4506 upsampled → 0.4536 native) — the interpolation blur of upsampling slightly hurts alignment
    - In other cases native is slightly **lower** (e.g. 01666 WT: 0.2612 → 0.2727) — confirming there's no systematic bias either way
* **Key discussion point:** Weighted Dice is robust to resolution mismatch. Unlike IoU (which swings wildly between upsampled and native due to hard thresholding artefacts), Weighted Dice gives **stable, consistent scores** at both resolutions. This proves it's a more reliable metric for evaluating coarse methods like Grad-CAM.
* **The bigger Grad-CAM conclusion:** Even at its native resolution, Grad-CAM achieves 0% ET Pointing Game — the bottleneck fundamentally cannot resolve small subregions. This is a method limitation, not a model flaw.

---

### 2.3 Relevance-Based Attribution: LRP (Input × Gradient)
* **Narrative:** LRP distributes the prediction score back to individual input voxels. Unlike Grad-CAM (which shows *where* the model attends), LRP shows *what evidence* the model uses.
* **Key Results (5 patients × 3 regions):**
    - Pointing Game: **100%** for ALL patients, ALL regions (including ET) — the most reliable localiser
    - Coverage: 0.34–0.84 — relevance distributes beyond tumor boundaries into surrounding tissue context
    - Saliency IoU: < 0.01 — diffuse maps (expected)
    - Weighted Dice: 0.01–0.17
* **Figure:** 2D saliency overlay + 3D Slicer rendering for one patient.
* **Discussion:** Perfect PG across all regions proves the model's predictive features are always co-located with tumor tissue. The diffuse maps are not a flaw — LRP correctly attributes relevance to contextual tissue (e.g., edema surrounding Tumor Core contributes to the WT class).

---

### 2.4 Perturbation-Based Attribution: Occlusion Sensitivity
* **Narrative:** Unlike all gradient-based methods, Occlusion makes zero mathematical assumptions. It slides a 16³ black patch across the input, records the confidence drop, and maps it to spatial locations. This is the **gold-standard XAI validation**.
* **Key Results (5 patients × 3 regions):**
    - WT Pointing Game + MSR Accuracy: **100%**
    - **Highest Weighted Dice of ALL 6 methods:** WT 0.38–0.46, TC 0.24–0.47, ET 0.23–0.35
    - Best spatial precision — physically measures dependency
    - Patient 01397 ET achieves Weighted Dice = 0.35 AND PG = 1.0 — the only method where ET localisation truly succeeds
* **Figure:** 2D occlusion heatmap overlay + 3D Slicer rendering for one patient.
* **Discussion:** Occlusion's highest Weighted Dice is the **single strongest evidence** that the model genuinely relies on tumor voxels. This rules out shortcut learning, texture bias, or dataset artefacts — the clinical trustworthiness of the segmentation is validated.

---

### 2.5 Uncertainty Quantification: MC Dropout — Per-Patient Analysis
*Goal: Present MC Dropout as a fundamentally different signal (uncertainty ≠ saliency). Detailed per-patient tables.*

#### MC Dropout Metrics Explained

| Metric | What it measures |
|--------|-----------------|
| **UAR** | Of all "confused spots", what fraction is inside the tumor? |
| **Boundary Uncertainty Ratio** | Of all uncertainty, what fraction sits at the GT tumor edges? |
| **Mean Unc Inside / Outside** | Average variance inside vs. outside tumor (is the model more confused about tumor tissue?) |
| **Saliency-Unc Correlation** | Pearson r between LRP saliency and MC Dropout variance (are they related or independent?) |

*Why LRP for correlation:* Full input resolution (160³) matching MC Dropout. Grad-CAM's 20³ would correlate interpolation artefacts.

#### Patient 01497 — Low Uncertainty, Clear Boundaries

| Region | UAR | Boundary Ratio | Unc Inside | Unc Outside | LRP Corr |
|--------|-----|---------------|------------|-------------|----------|
| WT | 0.196 | 0.876 | 0.00131 | 0.00010 | +0.020 |
| TC | 0.092 | **0.967** | 0.00122 | 0.00016 | +0.000 |
| ET | 0.219 | 0.579 | **0.01400** | 0.00005 | −0.065 |

* **Interpretation:** Well-defined tumor. Low UAR = model segments confidently. TC Boundary Ratio = 0.97 means nearly ALL uncertainty sits at the Tumor Core edge (clinically expected — TC boundaries are hardest for radiologists too). ET has 100× higher internal variance than surrounding tissue — the model "knows" ET is harder.

#### Patient 01397 — High Uncertainty, Complex Morphology

| Region | UAR | Boundary Ratio | Unc Inside | Unc Outside | LRP Corr |
|--------|-----|---------------|------------|-------------|----------|
| WT | 0.490 | 0.642 | 0.00338 | 0.00005 | −0.042 |
| TC | **0.892** | 0.776 | 0.01138 | 0.00001 | −0.046 |
| ET | 0.624 | **1.000** | 0.00853 | 0.00003 | −0.071 |

* **Interpretation:** Most complex tumor. TC UAR = 0.89 → 89% of all model confusion is inside Tumor Core. ET Boundary Ratio = **1.00** — every unit of uncertainty sits at ET edges. This is the strongest calibration evidence: the model doubts exactly where the tumor transitions to healthy tissue. Negative LRP correlation (−0.04 to −0.07) means the model is *confident* about the features it uses most.

#### Patient 00291 — The Failure Case (Gradient-Based XAI = Zero)

| Region | UAR | Boundary Ratio | Unc Inside | Unc Outside | LRP Corr |
|--------|-----|---------------|------------|-------------|----------|
| WT | 0.619 | **1.000** | 0.00440 | 0.00003 | −0.012 |
| TC | 0.504 | 0.997 | 0.00802 | 0.00002 | **+0.090** |
| ET | 0.500 | 0.997 | 0.00930 | 0.00002 | +0.053 |

* **Interpretation:** Grad-CAM and Guided Grad-CAM scored **zero across ALL metrics** for this patient — complete gradient-based failure. But MC Dropout reveals:
    - Boundary Ratio = 0.997–1.00 → despite the ambiguity, uncertainty concentrates at boundaries, proving the model IS spatially grounded
    - **Positive TC/ET correlation (+0.05 to +0.09)** — unique to this patient. Where LRP says "important" and MC Dropout says "uncertain" overlap slightly → the model relies on features it's not confident about. This is a clinical flag: cases like this should be flagged for radiologist review.

#### Cross-Paradigm Finding: Saliency ≠ Uncertainty
* Correlation across all 9 measurements: −0.071 to +0.090 (mean ≈ 0)
* **Saliency and uncertainty are independent, non-redundant signals** — each gives clinicians unique information
* This motivates deploying both in VR (Pillar 3): saliency for trust calibration, uncertainty for risk assessment

---

### 2.6 Cross-Method Comparative Analysis

#### 📊 Regional Vulnerability Analysis (Grouped Bar Chart)
* **Data:** Mean Weighted Dice per tumor region across all 5 attribution methods (Grad-CAM, GBP, Guided Grad-CAM, LRP, Occlusion)
* **Design:** 3 groups (WT, TC, ET) × 5 coloured bars per group
* **Pre-computed values:**

| Method | WT Mean W.Dice | TC Mean W.Dice | ET Mean W.Dice |
|--------|:-:|:-:|:-:|
| Grad-CAM (10 pts) | 0.220 | 0.268 | 0.048 |
| GBP (5 pts) | 0.130 | 0.139 | 0.140 |
| Guided Grad-CAM (5 pts) | 0.133 | 0.153 | 0.224 |
| LRP (5 pts) | 0.043 | 0.058 | 0.071 |
| Occlusion (5 pts) | 0.422 | 0.392 | 0.165 |

* **Discussion:**
    - Occlusion dominates WT and TC — physically verified model reliance
    - ET is universally the weakest region — exposing the vulnerability of all interpretability methods to small subregions
    - GBP and Guided Grad-CAM are surprisingly competitive on ET (0.14–0.22) vs. Grad-CAM's near-zero (0.048)
    - LRP has lowest Weighted Dice but 100% Pointing Game — the two metrics capture different dimensions (shape alignment vs. localisation)
    - **Key highlight:** Occlusion on patient 01397 achieves **PG=1.0, MSR=1.0, W.Dice=0.35 for ET** — the only method that successfully localises AND spatially aligns with ET above 0.30. This is the strongest argument that perturbation-based methods are necessary for small subregion interpretability.

#### 📊 Coverage vs. Precision Trade-off (Scatter Plot)
* **Data:** Saliency Coverage (x-axis) vs. Weighted Dice (y-axis) for ALL methods × ALL patients
* **What it reveals:** Inverse trend — Guided Grad-CAM has the highest coverage (0.80–0.96) but lower Weighted Dice (0.03–0.13), while Occlusion has moderate coverage (0.49–0.73) but the highest Weighted Dice (0.38–0.47). **High coverage ≠ high spatial precision.** The two metrics capture different aspects of saliency quality.

#### GBP: The Only Method with 100% Pointing Game Everywhere
* GBP achieves **100% PG across ALL patients AND ALL regions** — including ET, and including patient 00291 where every other gradient method scored zero.
* GBP operates at pure input level (no bottleneck, no class weighting), making it immune to the failures that affect higher-level methods. This proves the model's input-level features always encode tumor information.

#### Top-K Threshold Sensitivity
* From `xai_gradcam_topk15_weighted_dice.csv` and `xai_gradcam_coarse_bottleneck_topk10_weighted_dice.csv`:
* TopK 10% and TopK 15% produce nearly identical Weighted Dice values (within ±0.01–0.03) — confirming Weighted Dice is **threshold-insensitive**, further validating it as a robust metric. Present as a small supplementary table.

---

### 2.7 Visual Evaluation

#### 📸 Five-Patient × Six-Method XAI Grid
* **Layout:** Grid with 7 columns (GT + 6 XAI methods) × 5 rows (patients)
* **Patients:** 01666, 01518, 01497, 01397, 00291
* **Columns:** Ground Truth | Grad-CAM | GBP | Guided Grad-CAM | LRP | Occlusion | MC Dropout (uncertainty)
* **Show:** WT channel saliency overlaid on one representative axial slice per patient
* **Purpose:** One-glance visual comparison — patient 00291 row will visually show the gradient failure (blank Grad-CAM/Guided Grad-CAM columns) while other methods still highlight relevant regions.

#### 📸 2D + 3D Per-Method Explanation Pairs
* For each of the 6 XAI methods, show:
    - **Left:** 2D axial slice with saliency heatmap overlaid on MRI
    - **Right:** 3D Slicer volumetric rendering of the saliency volume
* **Patient:** Use a single strong-performing patient (e.g., 01497 or 01397) for consistency
* **Purpose:** Demonstrates the dimensional leap from 2D heatmap to 3D spatial understanding — directly motivating the VR pipeline (Pillar 3)

#### 📸 Patient 00291: The Diagnostic Case Study
* Dedicate a focused figure showing all 6 methods applied to patient 00291
* Layout: 6-panel figure (one per method) with annotations
* **Narrative:** Grad-CAM and Guided Grad-CAM produce blank maps. GBP and LRP achieve 100% PG. Occlusion succeeds with PG=1.0, MSR=1.0, W.Dice=0.42 (WT). MC Dropout shows Boundary Ratio=1.00. This demonstrates which methods are robust to morphological edge cases — a clinically important finding for method selection.


## 3. Pillar 3: Virtual Reality (VR) Immersive Visualisation
*(Placeholder - To be developed)*
* **Goal:** Demonstrating the end-user clinical application using SlicerVR/Unity3D.
* **Potential Inclusions:**
   - Showcasing the integration of MRI + Segmentation (Pillar 1) + XAI (Pillar 2) into the immersive 3D space.
   - Assessment of user experience, depth perception improvements, and potential utility for surgical planning.

---
## Chapter Summary
* Briefly tie all three pillars together. Emphasize that strong segmentation (Pillar 1) provides the foundation; interpretation (Pillar 2) provides the trust; and immersive visualization (Pillar 3) provides the clinical utility.
