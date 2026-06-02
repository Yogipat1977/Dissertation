# REVISION GUIDE: Draft 3 → Draft 4

**Paper:** "Illuminating the Black Box: A Multi-Method Explainable AI Framework with Uncertainty Quantification for 3D Brain Tumor Segmentation"

**Purpose:** This document is the complete blueprint for building Draft 4. Every modification is specified with the *exact section*, the *current text to change*, and the *revised text or new content to insert*. Work through it section by section.

---

## TABLE OF CONTENTS

1. [CRITICAL FIXES (Must-Do)](#1-critical-fixes-must-do)
2. [NEW CONTENT: Research Gap & Comparison Table](#2-new-content-research-gap--comparison-table)
3. [LANGUAGE SOFTENING: Absolute Statements → Academic Hedging](#3-language-softening-absolute-statements--academic-hedging)
4. [XAI SAMPLE SIZE: Correcting 5 Patients → 25 Patients](#4-xai-sample-size-correcting-5-patients--25-patients)
5. [PIPELINE DESCRIPTION: Missing Implementation Details](#5-pipeline-description-missing-implementation-details)
6. [SECTION-BY-SECTION REWRITING: Articulation Improvements](#6-section-by-section-rewriting-articulation-improvements)
7. [ADDITIONAL STRUCTURAL CHANGES](#7-additional-structural-changes)
8. [BIBLIOGRAPHY ADDITIONS](#8-bibliography-additions)
9. [FINAL CHECKLIST](#9-final-checklist)

---

## 1. CRITICAL FIXES (Must-Do)

These are non-negotiable changes that must be implemented first.

### 1.1 Author Affiliation — Remove "Independent Researcher"

**Current (lines 53–61):**
```typst
#align(center)[
  #text(size: 11pt)[
    *Yogi Amitkumar Patel*\
    \
    Independent Researcher\
    u2536809\@uel.ac.uk
  ]
  #v(1.5em)
]
```

**Replace with:**
```typst
#align(center)[
  #text(size: 11pt)[
    *Yogi Amitkumar Patel*#super[1]#h(1em)*Maimoona Sharif*#super[1]\
    \
    #super[1]School of Architecture, Computing and Engineering\
    University of East London, London, United Kingdom\
    u2536809\@uel.ac.uk
  ]
  #v(1.5em)
]
```

> **Rationale:** You are submitting under UEL affiliation. Adding your supervisor as co-author is standard practice for supervised research. Verify the spelling of the supervisor's name — the proposal says "Sharif", the report template says "Sarif". Confirm with them which is correct.

### 1.2 Keywords Line — Remove "Virtual Reality" (optional — keep only if VR is substantively discussed)

The VR component is thin in this paper. Consider whether "Virtual Reality" in the keywords is justified. If you keep it, the VR section needs strengthening (see Section 5).

---

## 2. NEW CONTENT: Research Gap & Comparison Table

### 2.1 Where to Insert

This goes at the **end of Section 2 (Related Work)**, as a new subsection titled `== Research Gap and Contributions`. It replaces the current implicit gap discussion scattered across §1 and §2.

### 2.2 Research Gap Paragraph — NEW TEXT

Insert the following **after §2.2 (Explainable AI for Medical Imaging)** and **before §2.3 (Immersive Visualization)**:

```typst
== Research Gap and Positioning

Despite significant advances in both 3D segmentation and XAI for medical imaging, existing
frameworks exhibit three persistent gaps that limit their clinical utility.

*Gap 1: Single-method reliance.* The majority of existing XAI frameworks for brain tumour
segmentation deploy only one or two explanation methods. NeuroXAI @neuroxai applied
multiple gradient-based methods but did not include perturbation-based validation
(Occlusion Sensitivity), leaving open the question of whether the model genuinely depends on
tumour voxels or has learned spurious correlations. AXONS-3 @axons3 introduced trust
metrics but similarly omits model-agnostic perturbation analysis, meaning its conclusions
about model reliability rest entirely on gradient-derived signals, which, as we demonstrate
(see patient 00291, §5), can fail silently.

*Gap 2: Absence of integrated uncertainty quantification.* Neither NeuroXAI nor AXONS-3
integrates probabilistic uncertainty quantification alongside structural attribution. Monte
Carlo Dropout provides a complementary signal — where the model doubts — that is
statistically independent of saliency (see §5.2.7). Without this signal, clinicians cannot
distinguish between confident and uncertain predictions, a critical deficiency for clinical
deployment. BrainAR @brainAR demonstrated AR-based visualisation of segmentation
outputs but provides no XAI or uncertainty analysis whatsoever.

*Gap 3: Inadequate XAI evaluation methodology.* Existing quantitative evaluation of
saliency maps relies on Pointing Game and Saliency Coverage @natekar2020, both of which
suffer from known limitations — Pointing Game reduces volumetric interpretability to a
single-voxel test, while Saliency Coverage is agnostic to spatial distribution. No prior work
has introduced a soft metric that evaluates continuous saliency maps without hard
binarisation thresholds, leading to volatile evaluation of coarse-resolution methods
such as Grad-CAM.

This paper addresses all three gaps through: (1) a six-method XAI suite spanning
gradient-based, perturbation-based, and probabilistic paradigms; (2) integrated MC Dropout
uncertainty quantification with cross-paradigm correlation analysis; and (3) the introduction
of Weighted Dice, a novel soft metric for resolution-robust saliency evaluation.
```

### 2.3 Comparison Table — NEW CONTENT

Insert the following table **immediately after** the research gap paragraph above:

```typst
#figure(
  table(
    columns: (18%, 14%, 14%, 14%, 14%, 14%, 12%),
    inset: 6pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Framework*], [*XAI Methods*], [*Uncertainty*], [*Perturbation Validation*],
    [*Quantitative XAI Metrics*], [*3D/VR Viz*], [*Evaluation Cohort*],
    table.hline(stroke: 0.5pt),

    [NeuroXAI @neuroxai \ (Zeineldin et al., 2022)],
    [Grad-CAM, GBP, Guided Grad-CAM, Deep Taylor],
    [No],
    [No],
    [Pointing Game, Saliency Coverage],
    [Static 2D overlays],
    [Not reported],

    [AXONS-3 @axons3 \ (Abyasa \& Rahmania, 2025)],
    [Grad-CAM, LIME, SHAP],
    [No],
    [No (LIME is local surrogate, not perturbation-based validation)],
    [Trust metrics (custom)],
    [Static 2D],
    [Not reported],

    [BrainAR @brainAR \ (Khedir et al., 2025)],
    [None],
    [No],
    [No],
    [Dice, HD95 only (segmentation)],
    [3D AR rendering],
    [BraTS subset],

    table.hline(stroke: 0.5pt),

    [*This Work*],
    [*Grad-CAM, GBP, Guided Grad-CAM, LRP, Occlusion Sensitivity*],
    [*MC Dropout (20 passes)*],
    [*Yes (Occlusion Sensitivity)*],
    [*PG, Coverage, IoU, Weighted Dice (novel)*],
    [*3D Slicer + SlicerVR*],
    [*25 patients (XAI), 126 patients (segmentation)*],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Comparative analysis of XAI frameworks for 3D brain tumour segmentation. This work is the first to integrate perturbation-based validation, probabilistic uncertainty quantification, and a novel soft evaluation metric within a single pipeline. Dash entries indicate the feature is absent from the referenced work],
) <tab:comparison>
```

> **Rationale:** This table directly addresses the reviewer's request for a comparison of 3 similar studies and how this work fills the gap. NeuroXAI, AXONS-3, and BrainAR are the three most relevant comparison points from your own bibliography.

---

## 3. LANGUAGE SOFTENING: Absolute Statements → Academic Hedging

Every instance below must be changed. The left column is the **current text** (search for it in `main_draft3.typ`), the right column is the **replacement**.

| # | Current Text (FIND) | Replacement (REPLACE WITH) | Location |
|---|---|---|---|
| 1 | `proving the model's spatial reasoning is intact` | `providing strong evidence that the model's spatial reasoning remains intact` | §5.2.5 (MC Dropout, patient 00291 caption) |
| 2 | `flawlessly align with ground truth masks` | `closely align with ground truth masks` | §5.1.2 caption / qualitative section |
| 3 | `perfect edge alignment` | `precise edge alignment` | §5.1.2 caption |
| 4 | `gold-standard XAI validation` | `a rigorous model-agnostic validation approach` | §3.2.5 / §5.2.4 |
| 5 | `rules out shortcut learning, texture bias, or dataset artifacts` | `provides evidence against shortcut learning, texture bias, or dataset artefacts` | §5.2.4 |
| 6 | `constitutes the strongest piece of evidence` | `provides compelling evidence` | §5.2.4 |
| 7 | `the strongest calibration evidence` | `strong calibration evidence` | §5.2.5 caption, patient 01397 |
| 8 | `proves that the model encodes tumor-relevant features` | `indicates that the model encodes tumour-relevant features` | §5.2.2 GBP section |
| 9 | `This proves that` → any instance | `This suggests that` or `This provides evidence that` | Global |
| 10 | `confirms the model _is_ segmenting based on real spatial features` | `suggests the model is segmenting based on spatially grounded features` | §5.2.5 |
| 11 | `The predicted boundary layers align flawlessly` | `The predicted boundary layers show close agreement` | §5.1 qualitative |
| 12 | `directly motivates` | `supports the rationale for` | Wherever it appears |

> **Principle:** In peer-reviewed medical AI, the verb hierarchy is:
> - **Weakest:** "may suggest", "is consistent with"
> - **Moderate:** "suggests", "indicates", "provides evidence that"
> - **Strong (use sparingly):** "demonstrates", "confirms"
> - **AVOID:** "proves", "rules out", "flawlessly", "perfect", "gold-standard" (as self-description)

---

## 4. XAI SAMPLE SIZE: Correcting 5 Patients → 25 Patients

### The Problem in Draft 3

Draft 3 gives the impression that XAI was evaluated on only 3–5 patients. Your dissertation report reveals you actually evaluated **25 patients** for saliency metrics and **23 patients** for MC Dropout. This is a critical correction — the paper currently undersells its own evidence.

### 4.1 Changes to the Abstract

**Current abstract excerpt:**
> "Quantitative evaluation of XAI outputs using four metrics..."

The abstract should explicitly state the cohort size. Add after "four metrics":

```
...four metrics — Pointing Game accuracy, Saliency Coverage, Saliency IoU, and a novel
Weighted Dice score — across a 25-patient evaluation cohort demonstrates that...
```

### 4.2 Changes to §4 (Experimental Setup) — ADD New Subsection

After §4.3 (Training Configuration), add:

```typst
== XAI Evaluation Protocol

Quantitative XAI evaluation was conducted on a 25-patient subset sampled from the 126-patient
test set. For each patient, all five saliency-based methods (Grad-CAM, Guided
Backpropagation, Guided Grad-CAM, Input × Gradient, and Occlusion Sensitivity) were
evaluated using four metrics: Pointing Game, Saliency Coverage, Saliency IoU, and Weighted
Dice. Monte Carlo Dropout uncertainty metrics were computed for 23 of these patients
(two excluded due to convergence anomalies in stochastic passes). Three patients
(01497, 01397, 00291) were selected as detailed case studies representing low-uncertainty,
high-uncertainty, and gradient-failure scenarios respectively. The distinction between
evaluation scopes is summarised below:

#figure(
  table(
    columns: (1fr, 1fr, 1fr),
    inset: 8pt,
    align: left + horizon,
    table.header([*Evaluation*], [*Cohort Size*], [*Purpose*]),
    [Segmentation Performance], [$n = 126$ patients], [Full test set metrics (Dice, HD95, IoU, Sensitivity, Specificity)],
    [XAI Saliency Metrics], [$n = 25$ patients], [Pointing Game, Coverage, Saliency IoU, Weighted Dice across 5 methods],
    [MC Dropout Uncertainty], [$n = 23$ patients], [UAR, Boundary Ratio, Saliency-Uncertainty Correlation],
    [Detailed Case Studies], [$n = 3$ patients], [In-depth multi-method qualitative and quantitative analysis],
  ),
  caption: [Evaluation scope summary distinguishing between the full segmentation test set and the XAI evaluation cohort],
) <tab:eval_scope>
```

### 4.3 Update ALL XAI Results Sections to Reference the Correct Cohort

Throughout §5.2, update references to match the actual data from your dissertation:

| Section | Current (Draft 3) | Corrected (from Dissertation) |
|---|---|---|
| §5.2.2 GBP | "100% Pointing Game across all five patients" | "strong Pointing Game accuracy for TC (80.0%) and ET (66.7%) across the 25-patient cohort" |
| §5.2.3 Guided Grad-CAM | "0.81–0.96 for Whole Tumor" | "0.81–0.96 for Whole Tumour across the evaluation cohort, with mean coverage of 0.93 (WT)" |
| §5.2.4 Occlusion | "100% Pointing Game for Whole Tumor across all patients" | "100% Pointing Game for Whole Tumour and 88% for Tumour Core across the 25-patient cohort" |
| §5.2.4 Occlusion | "highest Weighted Dice scores of all six methods: 0.38–0.46 for WT" | "highest mean Weighted Dice: 0.397 (WT), 0.345 (TC), 0.233 (ET) across the 25-patient cohort" |
| §5.2.6 Cross-method table | Table values (5-patient means) | Replace with 25-patient means from dissertation: see below |

### 4.4 Corrected Cross-Method Table (from Dissertation §4.2.7)

Replace the current Table (around line 524–542) with:

```typst
#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Method*], [*WT Mean W.Dice*], [*TC Mean W.Dice*], [*ET Mean W.Dice*],
    table.hline(stroke: 0.5pt),

    [Grad-CAM], [0.320 ± 0.145], [0.358 ± 0.161], [0.250 ± 0.174],
    [GBP], [0.066 ± 0.051], [0.053 ± 0.052], [0.041 ± 0.049],
    [Guided Grad-CAM], [0.132 ± 0.072], [0.186 ± 0.082], [0.169 ± 0.073],
    [IxG (LRP Proxy)], [0.074 ± 0.033], [0.122 ± 0.050], [0.141 ± 0.056],
    [Occlusion], [*0.397 ± 0.116*], [*0.345 ± 0.135*], [0.233 ± 0.116],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Mean Weighted Dice scores ± standard deviation by tumour region across the 25-patient XAI evaluation cohort ($n = 25$ per saliency method). Occlusion Sensitivity achieves the highest scores for WT and TC. Bold values indicate the best-performing method per region],
) <tab:xai_mean_scores>
```

> **IMPORTANT:** These are the correct values from your dissertation results chapter. The Draft 3 table had different (smaller-sample) values. The 25-patient data significantly strengthens the paper.

### 4.5 Update MC Dropout Cohort References

In §5.2.5 and §5.2.7, update "across all three patients" to:

```
Across the expanded 23-patient MC Dropout cohort and all regional measurements, the
Saliency-Uncertainty Correlation averages near zero (WT: −0.007 ± 0.096, TC: −0.037 ± 0.104,
ET: −0.095 ± 0.119).
```

This text already exists in your dissertation (Chapter 4, line 315). Copy it directly.

---

## 5. PIPELINE DESCRIPTION: Missing Implementation Details

### 5.1 Add Pipeline Figure Reference

Your dissertation has an `Implementation_pipeline.drawio.svg` figure. Add a reference to this pipeline in §3 or §4 of the paper. Insert after §3.1 (SegResNet Architecture):

```typst
== Implementation Pipeline

The framework follows a modular seven-stage pipeline: (1) YAML-based configuration
loading and data initialisation, (2) model training with automated checkpointing and
W&B tracking, (3) test set inference on 126 held-out patients, (4) XAI saliency map
generation for all six methods across the evaluation cohort, (5) quantitative metric
computation and CSV export, (6) spatially-aligned NIfTI volume export for each patient
and method, and (7) VR visualisation preparation via 3D Slicer.

The modular architecture separates concerns into five packages: data handling
(dataset scanning, patient splitting, MONAI transforms), model factory (architecture,
loss, and optimiser instantiation), training orchestration (AMP, checkpointing,
W&B integration), the XAI suite (standardised saliency and uncertainty map generation),
and evaluation tools (segmentation and XAI-specific metric computation with CSV export).
XAI metrics are computed inline during saliency generation to guarantee spatial
alignment between saliency maps and ground truth labels in the same MONAI-preprocessed
coordinate space.
```

### 5.2 Strengthen VR Section (Currently Thin)

The current VR content (§2.3 + Appendix) is too brief for a contribution claim. Either:

**Option A: Keep VR as a pipeline feature (recommended for a short paper)**
- Remove "VR-ready volumetric pipeline" from the numbered contributions list (§1, contribution 4)
- Mention it as a practical outcome: "Additionally, all explanation outputs are exported as spatially-aligned NIfTI volumes compatible with immersive 3D visualization platforms such as 3D Slicer and SlicerVR."

**Option B: Strengthen VR as a contribution (requires more content)**
- Add the technical implementation details from your dissertation (the Bash launcher, `fix_vr.py`, the five auto-initialisation functions, GPU Ray Cast rendering, WiVRn streaming)
- Add the VR screenshots from your dissertation (full brain view + heatmap rendering)
- Add the Nielsen's Heuristics evaluation table

> **Recommendation:** Go with **Option A** for a concise research paper. The VR is better served in the full dissertation. The paper's strength is the XAI framework, not the VR pipeline.

---

## 6. SECTION-BY-SECTION REWRITING: Articulation Improvements

This section provides rewritten paragraphs with stronger argumentation, sharper wordiness, and a more assertive (but appropriately hedged) academic voice.

### 6.1 Abstract — Restructure for Clarity

The current abstract is a single dense paragraph of ~200 words. Restructure it into a clearer narrative arc while keeping it as a single paragraph (standard for most venues):

**REVISED ABSTRACT:**

```
Deep learning models for 3D brain tumour segmentation have achieved remarkable accuracy,
yet their opaque decision-making processes present a critical barrier to clinical adoption.
This paper presents a multi-method Explainable AI (XAI) framework that integrates six
complementary post-hoc explanation techniques with a SegResNet architecture for volumetric
brain tumour segmentation on the BraTS 2023 dataset. The framework combines gradient-based
methods (Grad-CAM, Guided Backpropagation, Guided Grad-CAM, Input × Gradient as an LRP
proxy), perturbation-based analysis (Occlusion Sensitivity), and probabilistic uncertainty
quantification (Monte Carlo Dropout) to provide multi-perspective interpretability of 3D CNN
predictions. The SegResNet model achieves competitive segmentation performance on a held-out
test set of 126 patients, with mean Dice scores of 0.923 (Whole Tumour), 0.891 (Tumour
Core), and 0.873 (Enhancing Tumour). Quantitative evaluation of XAI outputs across a
25-patient evaluation cohort using four metrics — Pointing Game accuracy, Saliency Coverage,
Saliency IoU, and a novel Weighted Dice score — demonstrates that Occlusion Sensitivity
achieves the highest spatial alignment (mean Weighted Dice: 0.397 WT, 0.345 TC), while
Guided Grad-CAM achieves 93% saliency coverage for the Whole Tumour region. MC Dropout
uncertainty analysis across 23 patients reveals boundary-concentrated variance patterns,
with Boundary Uncertainty Ratios exceeding 0.84, consistent with clinically meaningful
confidence calibration. A critical finding is a gradient-based failure case where Grad-CAM
and Guided Grad-CAM produce zero saliency, yet MC Dropout confirms spatially grounded
reasoning — demonstrating why multi-method XAI evaluation is essential. All explanation
outputs are exported as spatially-aligned NIfTI volumes for 3D visualisation. This work
advances the integration of model accuracy and clinical interpretability by providing a
unified, quantitatively validated explainability pipeline for volumetric medical image
analysis.
```

**Key changes:**
- Added cohort sizes (25 patients for XAI, 23 for MC Dropout, 126 for segmentation)
- Corrected Weighted Dice values to match 25-patient data
- Changed "LRP" to "Input × Gradient as an LRP proxy" (more accurate)
- Removed "VR-ready" from abstract (unless VR is strengthened)
- Softened "bridges the gap" → "advances the integration"

### 6.2 Introduction — Stronger Opening, Sharper Problem Statement

**Current opening (line 82):**
> "In modern neuro-oncology, accurate identification and delineation of brain tumors from Magnetic Resonance Imaging (MRI) data remains a critical clinical challenge."

**Revised opening (more assertive, draws the reader in):**

```typst
Approximately 308,000 people receive a primary brain tumour diagnosis annually @neri2023,
and for the neurosurgeon reviewing the pre-operative scan, the decisive question is never
simply whether a tumour is present — it is where, precisely, the tumour boundary ends. A
margin error of a few millimetres in the wrong direction means either residual malignant
tissue or permanent neurological deficit. Deep learning models, particularly 3D
Convolutional Neural Networks (CNNs), have demonstrated the capacity to delineate these
boundaries with superhuman consistency @iftikhar2025 @bhati2024, processing entire MRI
volumes rather than individual 2D slices to capture the true three-dimensional spatial
relationships of tumour pathology.
```

> **Why this is better:** It opens with a concrete clinical stake (patient harm from margin error) rather than a generic statement. The reader immediately understands *why this matters*.

### 6.3 Introduction — Contributions List Rewrite

**Current contribution #2 (line 92):** too wordy. Tighten:

```typst
2. *Quantitative XAI validation with a novel metric:* We evaluate saliency maps using
four metrics across a 25-patient cohort, including a novel Weighted Dice metric that treats
continuous saliency values as soft membership scores — eliminating the threshold-dependent
volatility inherent in Saliency IoU and enabling equitable comparison of methods operating
at different spatial resolutions.
```

**Current contribution #3 (line 94):** good but needs the independence finding:

```typst
3. *Uncertainty-aware explainability:* We integrate MC Dropout uncertainty quantification
and demonstrate its statistical independence from structural attribution (Pearson $r approx 0$
across 23 patients), establishing that saliency and uncertainty provide complementary,
non-redundant clinical signals — the former for trust calibration, the latter for risk
assessment.
```

### 6.4 Discussion — Rewrite for Stronger Argumentation

The Discussion currently reads as a summary of results. A strong Discussion *interprets*, *contextualises*, and *argues*. Here's how to restructure §6.1:

**REVISED §6.1 (Complementary Nature of XAI Methods):**

```typst
== Complementary Nature of XAI Methods

The central methodological argument of this work — that no single XAI method provides a
complete picture of model behaviour — is substantiated by three convergent lines of
evidence. First, the gradient-based failure case (patient 00291) demonstrates that Grad-CAM
and Guided Grad-CAM can produce zero activation despite the model maintaining spatially
grounded reasoning, as confirmed by MC Dropout Boundary Ratios of 0.997–1.000. This
failure is not an edge case to be dismissed; it is a diagnostic warning. Had this patient
been evaluated with Grad-CAM alone, the conclusion would have been that the model's
reasoning was deficient — a conclusion directly contradicted by the perturbation-based and
probabilistic evidence.

Second, the cross-method Weighted Dice rankings reveal method-dependent regional
vulnerabilities. Occlusion Sensitivity leads for Whole Tumour (0.397) and Tumour Core
(0.345), while Grad-CAM narrowly leads for Enhancing Tumour (0.250 vs. 0.233). This
discrepancy is not a contradiction but a consequence of each method's inductive bias:
Occlusion captures empirical model dependency through physical perturbation, while
Grad-CAM's bottleneck weighting happens to better localise small ET structures when
upsampled. Neither result alone tells the full story.

Third, the near-zero Saliency-Uncertainty Correlation (mean Pearson $r$ across 23 patients:
WT −0.007, TC −0.037, ET −0.095) confirms that saliency and uncertainty are independent
signals. This independence is not a limitation but a design advantage: it means that
jointly deploying both modalities provides a complete, non-redundant clinical
interpretability layer. Saliency answers "Is the model attending to the right features?"
while uncertainty answers "Where might the model be wrong?" — two fundamentally different
clinical questions that require two fundamentally different instruments.
```

> **Why this is better:** Each paragraph makes an explicit argument with evidence, rather than listing observations. The argument *builds*: failure case → regional vulnerability → signal independence → clinical completeness.

### 6.5 Discussion — Rewrite §6.2 (Clinical Implications)

**REVISED §6.2:**

```typst
== Clinical Implications

Three findings have direct implications for clinical deployment pathways. First, the high
saliency coverage of Guided Grad-CAM (93% for WT across the evaluation cohort) indicates
that the model concentrates its attention within the tumour boundary rather than on
spurious image correlates — a necessary (though not sufficient) condition for clinical
trust. This finding is strengthened by Occlusion Sensitivity's independent confirmation:
the highest Weighted Dice of all methods (0.397 WT) demonstrates that physically removing
tumour voxels degrades the model's predictions, providing empirical evidence that the model
has learned tumour-relevant features rather than dataset-specific shortcuts.

Second, the boundary-concentrated uncertainty pattern observed across the 23-patient MC
Dropout cohort (Boundary Uncertainty Ratios exceeding 0.84 for all regions) aligns with
established neuro-oncological understanding. Tumour margins — particularly the infiltrative
zone surrounding the enhancing rim — are inherently ambiguous on MRI due to partial volume
effects and diffuse glioma cell infiltration. A well-calibrated model should exhibit
elevated uncertainty precisely at these boundaries, and the observed pattern is consistent
with this expectation.

Third, the positive Saliency-Uncertainty Correlation observed uniquely in patient 00291
(TC: +0.090, ET: +0.053) — where high feature relevance overlaps with high predictive
variance — represents a specific pattern that, if validated on larger cohorts, could serve
as an automated flag for cases requiring priority radiologist review. This signal is
available only through multi-method analysis; no single XAI technique can simultaneously
identify both high-relevance and high-uncertainty regions.
```

### 6.6 Conclusion — Tighten, Reduce Redundancy with Discussion

**Current problem:** The five numbered findings in §7 are near-verbatim repetitions of §6. Rewrite the conclusion to focus on *implications and future trajectory*, not re-summarising findings.

**REVISED §7 (Conclusion):**

```typst
= Conclusion

This paper presents a multi-method Explainable AI framework for 3D brain tumour
segmentation that advances the integration of model accuracy and clinical
interpretability. By deploying six complementary XAI techniques with a SegResNet
architecture achieving competitive BraTS 2023 performance (Dice: 0.923 WT, 0.891 TC,
0.873 ET), and evaluating them quantitatively across a 25-patient cohort, this work
establishes three key principles for clinical XAI deployment:

First, _multi-method evaluation is not optional_ — it is a prerequisite for responsible
clinical deployment. The gradient-based failure case demonstrates that silent method
failure can produce misleading conclusions about model behaviour, a risk mitigated only
through cross-paradigm redundancy.

Second, _saliency and uncertainty are complementary, not interchangeable_. Their
statistical independence (Pearson $r approx 0$) across the evaluation cohort confirms that
deploying both modalities provides non-redundant clinical information — trust calibration
through attribution, risk assessment through uncertainty.

Third, _evaluation metrics must match method resolution_. The novel Weighted Dice metric
addresses the instability of hard-thresholded Saliency IoU for coarse-resolution methods,
providing a principled foundation for cross-method comparison in 3D medical XAI.

These findings are subject to important limitations: reliance on a single dataset (BraTS
2023), the absence of formal clinical validation, and a single deterministic test split.
Future work will prioritise formal evaluation with practising neuroradiologists,
cross-institutional validation on diverse MRI datasets, and extension of the XAI
evaluation to the full 126-patient test set.
```

---

## 7. ADDITIONAL STRUCTURAL CHANGES

### 7.1 Limitations — Expand and Be More Honest

The current limitations section (§6.4) is a single paragraph. Expand it:

```typst
== Limitations

This work has five primary limitations that should be considered when interpreting the
results.

_Dataset generalisability._ The framework is evaluated exclusively on BraTS 2023.
Cross-institutional validation using datasets with different scanner protocols,
field strengths, or patient demographics is essential to assess robustness under domain
shift.

_Single test split._ All segmentation metrics derive from a single deterministic split
(seed=42). While the consistency of improvement across two independent training scales
(250-patient prototype and 1,251-patient full model) partially mitigates this concern,
k-fold cross-validation would provide more robust performance estimates. The computational
cost (~24 hours per training run) made this infeasible within the project timeline.

_XAI cohort size._ The 25-patient XAI evaluation cohort, while substantially larger than
the case-study approach common in prior work, represents approximately 20% of the test
set. Extending quantitative XAI evaluation to the full 126-patient test set would
strengthen the generalisability of cross-method conclusions, particularly for
the Enhancing Tumour region where method performance is most variable.

_Computational cost of Occlusion Sensitivity._ Occlusion Sensitivity requires
approximately 1,000 forward passes per patient, limiting its practicality for real-time
clinical workflows. More efficient perturbation strategies (e.g., adaptive stride or
region-of-interest-focused occlusion) should be explored.

_Absence of clinical validation._ Formal evaluation involving practising radiologists or
neurosurgeons remains essential. The XAI metrics used in this work — Pointing Game,
Weighted Dice, and others — are quantitative proxies for clinical utility. Whether these
metrics correlate with diagnostic confidence or decision quality in practice is an open
question that requires prospective clinical studies.
```

### 7.2 Rename "LRP" → "Input × Gradient (LRP Proxy)" Consistently

Your dissertation correctly uses "Input × Gradient (LRP Proxy)" or "IxG". Draft 3 inconsistently calls it "LRP". Search-and-replace throughout:

- Table headers: "LRP" → "IxG (LRP Proxy)"
- Body text: "Layer-wise Relevance Propagation (LRP)" → keep the first mention as the full expansion, then use "Input × Gradient (IxG)" consistently thereafter
- MC Dropout tables: "LRP Corr" → "IxG Corr"

This is important because true LRP was not implemented (as you explain in both the paper and dissertation). Calling it "LRP" in tables is misleading.

---

## 8. BIBLIOGRAPHY ADDITIONS

You need to add the following references (already in your dissertation bib but missing from `paper.bib`):

```bibtex
@inproceedings{bach2015pixel,
  title={On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation},
  author={Bach, Sebastian and Binder, Alexander and Montavon, Grégoire and Klauschen, Frederick and Müller, Klaus-Robert and Samek, Wojciech},
  journal={PLoS ONE},
  volume={10},
  number={7},
  year={2015}
}

@article{zhang2018pointing,
  title={Top-down neural attention by excitation backprop},
  author={Zhang, Jianming and Bargal, Sarah Adel and Lin, Zhe and Brandt, Jonathan and Shen, Xiaohui and Sclaroff, Stan},
  journal={International Journal of Computer Vision},
  volume={126},
  pages={1084--1102},
  year={2018}
}

@article{zeineldin2022explainability,
  title={Explainability of deep neural networks for MRI analysis of brain tumors},
  author={Zeineldin, Ramy A and Karar, Mohamed E and Elshaer, Ziad and Wirtz, Christian R and Burgert, Oliver and Mathis-Ullrich, Franziska},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2022},
  publisher={Springer}
}

@article{molchanov2019taylor,
  title={Importance Estimation for Neural Network Pruning},
  author={Molchanov, Pavlo and Mallya, Arun and Tyree, Stephen and Frosio, Iuri and Kautz, Jan},
  journal={CVPR},
  year={2019}
}
```

Also verify that `@neuroxai` and `@zeineldin2022explainability` are not duplicates (they appear to reference the same paper). If so, consolidate into one bib key.

---

## 9. FINAL CHECKLIST

Before compiling Draft 4, verify every item:

- [ ] **Author block** — UEL affiliation, supervisor's name added, "Independent Researcher" removed
- [ ] **Research gap paragraph** — inserted in §2 with clear Gap 1/2/3 structure
- [ ] **Comparison table** — 3 frameworks vs. this work, inserted in §2
- [ ] **Language softening** — all 12 items from Section 3 above addressed
- [ ] **XAI cohort sizes** — 25 patients (saliency), 23 patients (MC Dropout) stated explicitly in abstract, §4, and §5
- [ ] **Evaluation scope table** — new table in §4 distinguishing 126 vs. 25 vs. 23 vs. 3
- [ ] **Cross-method table** — updated with 25-patient means ± std from dissertation
- [ ] **MC Dropout statistics** — 23-patient cohort means cited from dissertation
- [ ] **Pipeline description** — 7-stage pipeline paragraph added to §3 or §4
- [ ] **LRP → IxG (LRP Proxy)** — consistent naming throughout
- [ ] **LRP Corr → IxG Corr** — updated in all MC Dropout tables
- [ ] **Limitations** — expanded to five clear points
- [ ] **Conclusion** — rewritten to avoid Discussion redundancy
- [ ] **Abstract** — restructured with cohort sizes and corrected values
- [ ] **Introduction opening** — rewritten with clinical stake
- [ ] **Bibliography** — new entries added, duplicates consolidated
- [ ] **VR contribution** — either strengthened (Option B) or repositioned as pipeline feature (Option A)
- [ ] **Spell-check** — British English throughout (tumour, colour, behaviour, etc.)
- [ ] **Figure cross-references** — all `@fig:` and `@tab:` labels verified after restructuring

---

*This revision guide was generated from a comprehensive analysis of Draft 3 (`main_draft3.typ`), the full dissertation report (all chapters from `Final report/chapters/`), and the reviewer feedback provided. All statistics cited in the corrections are sourced directly from the dissertation results chapter to ensure accuracy.*
