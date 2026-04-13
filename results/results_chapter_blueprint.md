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
> **UPCOMING PILLARS (Reserved Spaces)**
> The following sections are structurally defined but their content will be written in future stages of your project.

## 2. Pillar 2: Explainable AI (XAI) Interpretation
*(Placeholder - To be developed)*
* **Goal:** "Illuminating the Black Box." Explain *why* the model made the predictions established in Pillar 1.
* **Potential Inclusions:**
   - Visualizing 3D Grad-CAM (and other XAI) saliency maps.
   - Quantitative XAI metrics (`xai_gradcam_metrics.csv`, occlusion drop, etc.).
   - Comparison of which XAI method best aligns with pathological knowledge.

## 3. Pillar 3: Virtual Reality (VR) Immersive Visualisation
*(Placeholder - To be developed)*
* **Goal:** Demonstrating the end-user clinical application using SlicerVR/Unity3D.
* **Potential Inclusions:**
   - Showcasing the integration of MRI + Segmentation (Pillar 1) + XAI (Pillar 2) into the immersive 3D space.
   - Assessment of user experience, depth perception improvements, and potential utility for surgical planning.

---
## Chapter Summary
* Briefly tie all three pillars together. Emphasize that strong segmentation (Pillar 1) provides the foundation; interpretation (Pillar 2) provides the trust; and immersive visualization (Pillar 3) provides the clinical utility.
