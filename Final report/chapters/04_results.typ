= Results

This chapter presents a comprehensive appraisal of the 3D Convolutional Neural Network (SegResNet) and its efficacy. The results are organized into three sequential analyses: an evaluation of the model's predictive segmentation performance, the interpretability of its decisions via Explainable Artificial Intelligence (XAI) techniques, and the subsequent visualization of these complex metrics within an immersive Virtual Reality (VR) framework.

== Prediction and Segmentation Performance
The foundational technical phase of this study necessitated the development of a neural network architecture capable of accurately localizing brain tumor subregions, serving as a prerequisite for subsequent explainability analyses. Given the substantial histopathological heterogeneity and complex morphological variations characteristic of neoplastic lesions (actual tumor), establishing a robust segmentation backbone was imperative to ensure the validity of downstream interpretability methods.

=== Progression and Metric Analysis (Baseline vs. Final)
This section presents a comparative analysis of model iterations developed through the Agile-based methodology previously described. Beginning with a baseline prototype (Version 1) trained on minimal data and culminating in a fully-specified model trained on 1,251 patients from the BraTS cohort, the evaluation metrics reveal substantial performance divergences between these developmental phases.

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Figures/results_figures/bar_metrics_Baseline_version_01.png", width: 80%),
    image("../Figures/results_figures/bar_metrics_Final_version_10.png", width: 80%),
  ),
  caption: [Performance comparison between the Baseline Prototype (left) and the Final Model trained on 1,251 patients (right). Note the massive improvement in the ET metric.],
)

*Data Scaling Comparison*
To understand how dataset volume directly influences performance, we compared the Dice scores across three versions: the initial baseline, an intermediate training run of 250 patients, and the final run utilizing the complete 1,251-patient dataset.

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [], table.cell(colspan: 3, [*Dice Score*]), table.cell(colspan: 3, [*HD95 (mm)*]),
    table.hline(stroke: 0.5pt),
    [*Region*], [*Baseline*], [*250 Pts*], [*Final*], [*Baseline*], [*250 Pts*], [*Final*],
    table.hline(stroke: 0.5pt),

    [WT], [0.741], [0.908], [0.923], [48.63], [7.23], [5.60],
    [TC], [0.428], [0.837], [0.891], [57.54], [15.34], [4.54],
    [ET], [0.272], [0.742], [0.873], [75.91], [15.25], [3.66],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Test set Dice and HD95 progression across data scaling stages. Dice quantifies volumetric overlap (higher is better) while HD95 measures boundary error in millimetres (lower is better).],
)

Table \@tab:dice-hd95 presents segmentation metrics across training data scales (45 to 1,251 patients), utilizing Dice coefficients for volumetric overlap and 95th Percentile Hausdorff Distance (HD95) for boundary delineation. Whole Tumor segmentation exhibited asymptotic performance (Dice 0.908 to 0.923), suggesting efficient learning of macroscopic structures from limited data. In contrast, the Enhancing Tumor characterized by fragmented, heterogeneous morphology demonstrated substantial dependency on training volume, improving from 0.272 to 0.873. Corresponding boundary refinement was equally pronounced: ET HD95 decreased from 75.91 mm to 3.66 mm, progressing from clinically unusable deviations to sub-voxel precision requisite for safe neurosurgical planning.

*Computational vs. Performance Trade-off*

The escalation from 250 patients to 1,251 patients carried substantial hardware and temporal costs. Training iterations lasted exponentially longer on the GPU infrastructure. However, as the table indicates, the added computation was unequivocally justified. While general regions (WT) hit a point of diminishing returns in both Dice and HD95, the deeper feature extraction required to confidently segment the complex non-linear Enhancing Tumors and to tighten their boundary precision required the full breadth of the training corpus.

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Figures/results_figures/heatmap_sensitivity.png", width: 80%),
    image("../Figures/results_figures/heatmap_specificity.png", width: 80%),
  ),
  caption: [Sensitivity (left) and Specificity (right) heatmaps across model versions. Sensitivity measures the proportion of true tumor voxels correctly identified, while Specificity reflects the model's ability to correctly exclude healthy tissue.],
)


=== Qualitative Visual Evaluation (2D & 3D Comparisons)

Morphological fidelity is still necessary for therapeutic value, even though quantitative metrics demonstrate computational validity. Precise spatial localisation of complex tumour borders is necessary for accurate subregion delineation. In order to confirm spatial concordance and anatomical accuracy, SegResNet predictions are compared to expert-annotated ground truth using multi-planar (2D) and volumetric (3D) visual evaluations.

#figure(
  align(center, grid(
    columns: (auto, 18%, 18%, 18%),
    // 'auto' for labels, 18% to keep images close
    column-gutter: 4pt,
    row-gutter: 4pt,
    align: center + horizon,

    // Header Row: Patient IDs
    [], [*01661*], [*01663*], [*01666*],

    // Row 1: Vertical Header + Ground Truth (GT)
    rotate(-90deg, reflow: true, pad(x: 4pt)[*GT*]),
    image("../Figures/results_Slicer-img/01661/GT-01661.png", width: 100%),
    image("../Figures/results_Slicer-img/01663/GT-01663.png", width: 100%),
    image("../Figures/results_Slicer-img/01666/GT-01666.png", width: 100%),

    // Row 2: Vertical Header + Model Prediction (Pred)
    rotate(-90deg, reflow: true, pad(x: 4pt)[*Pred*]),
    image("../Figures/results_Slicer-img/01661/pred-01661.png", width: 100%),
    image("../Figures/results_Slicer-img/01663/pred-01663.png", width: 100%),
    image("../Figures/results_Slicer-img/01666/pred-01666.png", width: 100%),

    // Row 3: Vertical Header + 3D Volumetric Render
    rotate(-90deg, reflow: true, pad(x: 4pt)[*3D*]),
    image("../Figures/results_Slicer-img/01661/3D-img-2.png", width: 100%),
    image("../Figures/results_Slicer-img/01663/3D-img-3.png", width: 100%),
    image("../Figures/results_Slicer-img/01666/3D-img.png", width: 100%),
  )),
  caption: [Qualitative comparison of patients 01661, 01663, and 01666. Row 1: Ground Truth (GT). Row 2: Model Prediction (Pred). Row 3: 3D volumetric rendering of the predicted segmentation.],
)
*2D Alignment Analysis*
When examining the 2D planes, the model’s predicted boundary layers flawlessly align with the True Ground Truth masks. As established, the 3.66 mm precision in HD95 manifests visually here: the predicted layers perfectly hug the edges of the actual tumor on the base MRI scans rather than loosely bleeding into healthy tissue.

*3D Morphological Analysis*
While conventional slice-based analysis obscures the true volumetric extent of neoplastic growth, three-dimensional reconstruction (Panel D) reveals the complete spatial topography and structural depth of the lesion. SegResNet's native volumetric processing capability generates high-fidelity 3D masks that facilitate direct translation into immersive rendering environments. This dimensional accuracy establishes the essential foundation for the immersive visualization framework presented in Pillar 3.

== Explainable AI (XAI) Interpretation
// TODO: Reserve for Phase 2 XAI integration (Saliency maps, Grad-CAM metrics, etc)

== Virtual Reality (VR) Immersive Visualisation
// TODO: Reserve for Phase 3 VR evaluation and validation
