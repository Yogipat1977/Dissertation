# Dissertation Editing Checklist
### "Illuminating the Black Box" — BSc Hons Data Science & AI
**Student:** Yogi Amitkumar Patel | **ID:** 2536809 | **University:** University of East London

---

> **How to use this file:** Work through each section in priority order. Tick each checkbox `[x]` when done. All changes are writing fixes only — no new experiments or code required.

---

## 🔴 PRIORITY 1 — Critical Content Changes

### Table 12 — Objective Satisfaction Matrix
- [ ] Change O6 status from *"not addressed / partially satisfied"* → **"Satisfied (Design Level)"**
- [ ] Add note to O6: *"Feasibility study conducted; implementation deferred due to OpenIGTLink resource constraints. Pipeline fully designed in Section 5.5.2."*
- [ ] Update O3 and O4 notes to explicitly reference clinical access constraints as a scoped-out project boundary

---

### Section 6.3 — Limitations
- [ ] Add **clinical validation** paragraph:
  > *"Formal clinical validation involving radiologists or neurosurgeons was beyond the ethical and logistical scope of a single-researcher BSc project, requiring IRB clearance and clinical access not available within the project timeline. This defines the boundary between a proof-of-concept framework and a clinically validated tool."*

- [ ] Add **single data split** paragraph:
  > *"All reported segmentation metrics are derived from a single stratified test split of 126 patients (seed=42). The computational cost of k-fold cross-validation across multiple full-scale 3D SegResNet training runs was prohibitive within project resource constraints. The 126-patient test set size and the consistency of improvement across two independent training scales partially mitigate this concern."*

---

### Abstract
- [ ] Change: *"demonstrates how immersive XAI can transform opaque neural networks into accountable decision-support systems"*
- [ ] To: *"proposes and technically demonstrates a framework through which immersive XAI **could** transform opaque neural networks into accountable decision-support systems"*

---

### Section 1.4 — Objectives
- [ ] Rewrite Objective 6 from: *"Leverage XAI insights for model pruning"*
- [ ] To: *"Conduct a feasibility analysis of XAI-guided architectural pruning, designing a conceptual pipeline and identifying the technical barriers to implementation"*

---

## 🔴 PRIORITY 2 — Section-Level Rewrites

### Section 5.5 — Objective 6
- [ ] Rename section title from: *"Objective 6 XAI-Guided Pruning Architectural Design"*
- [ ] To: *"Objective 6: XAI-Guided Architectural Analysis and Pruning Feasibility"*
- [ ] Add framing sentence at the **very start** of Section 5.5:
  > *"Given the computational and integration constraints encountered during implementation, Objective 6 was pursued as a rigorous design and feasibility contribution rather than a fully deployed system — a recognised and legitimate research output at proof-of-concept stage, consistent with the scope of a single-researcher BSc project."*

---

### Section 5.4 — VR Pipeline Evaluation
- [ ] Replace any language implying clinicians validated the system
- [ ] Change: *"allows clinicians to interactively navigate..."*
- [ ] To: *"is designed to allow clinicians to interactively navigate... This claim requires formal validation through a structured user study, proposed in Section 6.4."*
- [ ] Add heuristic defence sentence:
  > *"In the absence of clinical access, Nielsen's 10 Usability Heuristics were selected as an internationally validated, expert-driven evaluation framework widely used in HCI research for early-stage system evaluation, providing a structured and reproducible baseline for future comparative studies with domain experts."*

---

### Section 3.3.5 — Full-Scale Training Setup
- [ ] After reporting seed=42 split, add one-line qualifier:
  > *"These results are reported from a single deterministic test split; confidence intervals are not available without cross-validation, which was computationally infeasible at this scale."*

---

### Section 4.1 / 5.1 — Results & Evaluation
- [ ] Add progression consistency argument:
  > *"The consistency of metric improvement across the 250-patient prototype and 1,251-patient full model provides supplementary evidence that the reported performance is not an artefact of a single favourable test split, but reflects genuine model generalisation."*

---

## 🟡 PRIORITY 3 — Grammar & Sentence Fixes

| Location | Current Text | Fix |
|---|---|---|
| Section 2.1.2 | *"investigations findings"* | *"investigation's findings"* (add apostrophe) |
| Section 2.1.2 | *"...explains how this initiative..."* (second "explains") | Replace second "explains" with *"outlines"* |
| Section 2.2.1 | *"CNNs prefer an end-to-end approach"* | *"CNNs take an end-to-end approach"* |
| Section 2.2.1 | *"Magnetic Resonance Imaging MRI"* | *"Magnetic Resonance Imaging (MRI)"* — add parentheses |
| Section 2.3.1 | *"Just model prediction is insufficient"* | *"Model prediction alone is insufficient"* |
| Section 2.4 | *"This process requires time and is prone to spatial errors"* | *"This process is time-consuming and prone to spatial errors"* |
| Section 4.1 | *"The foundational technical phase of this study necessitated the development..."* | *"The first phase of this study involved developing..."* |
| Figure 18 Caption | *"warm tones envelops the tumor boundary"* | *"warm tones envelop the tumor boundary"* (subject-verb agreement) |

---

## 🟢 PRIORITY 4 — Whole-Document Consistency

### British English Spelling (Find & Replace)
Since you are submitting to a UK university (UEL), use British English throughout:

| Find | Replace |
|---|---|
| tumor | tumour |
| color | colour |
| utilizing | utilising |
| analyzing | analysing |
| visualize | visualise |
| characterize | characterise |
| recognize | recognise |
| optimize | optimise |
| behavior | behaviour |
| labeled | labelled |

> ⚠️ Run Find & Replace carefully — check each replacement to avoid changing proper nouns or dataset names (e.g., "BraTS" should remain unchanged).

---

### Abbreviation Formatting
- [ ] Ensure every abbreviation is introduced with parentheses on **first use in each chapter**
  - Example: *"Convolutional Neural Networks (CNNs)"* — not just in the Introduction
  - Check: CNN, MRI, XAI, VR, WT, TC, ET, GBP, LRP, MC Dropout

---

### PDF Layout Check
- [ ] Open the final submitted PDF and verify the running header (author name) does **not** bleed into the body text flow
- [ ] Verify all figures are captioned and referenced correctly in-text
- [ ] Verify Table 12 renders cleanly with updated O6 status

---

## ✅ Final Submission Checklist

- [ ] All `[x]` boxes above are ticked
- [ ] Abstract re-read for any remaining strong causal claims
- [ ] Conclusion re-read — ensure it does not claim clinical validation occurred
- [ ] Table 12 reviewed — all 7 objectives have a clear, honest status
- [ ] British English sweep completed
- [ ] PDF compiled and visually checked (headers, figures, tables)
- [ ] Word count within permitted range
- [ ] Submitted before deadline

---

*Checklist prepared following critical review of dissertation draft — May 2026*
