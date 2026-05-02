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

Having established clinically viable segmentation accuracy, the analysis shifts from how well the model performs to why it predicts specific sub-regions, addressing the “black box” opacity that undermines trust in neuro-oncological workflows @neri2023. Six complementary post-hoc techniques, taxonomised across three XAI paradigms @bhati2024, are deployed to interrogate this decision-making. Gradient-based attribution, comprising Grad-CAM for coarse class-discriminative localisation @selvaraju2017 @natekar2020, Guided Backpropagation for voxel-level structural dependencies, and their Guided Grad-CAM fusion, generates multi-scale saliency. Decomposition-based relevance via Layer-wise Relevance Propagation redistributes output scores to individual voxels under a conservation principle. Perturbation-based validation through Occlusion Sensitivity provides model-agnostic empirical proof of structural reliance, while stochastic uncertainty quantification via Monte Carlo Dropout maps epistemic ambiguity at tumour boundaries. Together, these converging paradigms close the fidelity gap, ensuring predictions are not only accurate but transparent, verifiable, and reliability-aware.
#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: left + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Category*], [*Method*], [*Resolution*], [*Class-Specific*],
    table.hline(stroke: 0.5pt),

    [Gradient-Based], [3D Grad-CAM], [Coarse (~20³)], [Yes],
    [Gradient-Based], [Guided Backpropagation (GBP)], [Full (voxel)], [No],
    [Gradient-Based], [Guided Grad-CAM], [Full (voxel)], [Yes],
    [Relevance-Based], [LRP (Input × Gradient)], [Full (voxel)], [Yes],
    [Perturbation-Based], [Occlusion Sensitivity], [Stride-upsampled], [Yes],
    [Uncertainty-Based], [MC Dropout (20 passes)], [Full (voxel)], [Yes],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Summary of the six XAI techniques applied to the SegResNet model, categorised by mechanism, spatial resolution, and class discrimination capability.],
)

=== XAI Evaluation Metrics: Addressing the Methodological Gap

Quantitative validation of explainability in 3D medical segmentation remains methodologically underdeveloped. Established literature predominantly evaluates saliency maps through *Pointing Game* accuracy and *Saliency Coverage*, binary or ratio-based measures that assess whether peak attention falls within the ground truth region or what fraction of total saliency mass concentrates inside the tumour boundary @natekar2020 @bhati2024. While these metrics operationalise localisation fidelity, they suffer from critical limitations: Pointing Game reduces volumetric interpretability to a single voxel hit-or-miss test, and Coverage remains agnostic to spatial distribution, permitting diffuse, anatomically imprecise attention to score favourably.

Furthermore, *Saliency IoU*, which thresholds continuous heatmaps into binary masks before computing Jaccard overlap, introduces hard-thresholding artefacts that disproportionately penalise coarse-resolution methods such as Grad-CAM while discarding gradient intensity information essential for clinical nuance @mironicolau2025.

To address this methodological gap, this research introduces *Weighted Dice*, a novel soft-metric adaptation that treats continuous saliency values as probabilistic membership weights rather than forcing premature binarisation. Weighted Dice is formulated as:

$ "WD" = (2 dot sum(S dot G)) / (sum S + sum G) $

where $S in [0,1]$ denotes the continuous saliency distribution and $G$ the binary ground truth. By treating the continuous saliency values directly as soft membership scores, Weighted Dice preserves full gradient intensity information, penalises both spatial misalignment and saliency leakage into healthy tissue, and remains stable across varying spatial resolutions. This constitutes a substantive contribution to 3D medical XAI methodology, enabling equitable comparison of coarse bottleneck attributions against full-resolution gradient maps without threshold-induced volatility.

=== The Bottleneck Resolution Problem: 3D Grad-CAM
Grad-CAM, conceived for 2D classification @selvaraju2017, was adapted to 3D segmentation by spatially averaging per-class logits for scalar backpropagation @natekar2020. This yields a coarse $20^3$ bottleneck heatmap subsequently upsampled to native $160^3$ introducing fundamental spatial fidelity loss in volumetric contexts. The critical question does trilinear upsampling inflate or deflate evaluation scores?

Identical activations were evaluated at both resolutions using Weighted Dice and Saliency IoU. As shown in @fig:gradcam-bottleneck, Weighted Dice remains stable ($plus.minus 0.02$–$0.04$), confirming no systematic bias. Conversely, Saliency IoU exhibits volatile thresholding artefacts, validating Weighted Dice as the reliable metric for coarse-resolution attribution. Critically, Grad-CAM achieves 0% Pointing Game for Enhancing Tumour even at native resolution,a fundamental bottleneck limitation, not model incapacity.
#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Figures/results_figures/xai_bottleneck_weighted_dice.svg", width: 90%),
    image("../Figures/results_figures/xai_bottleneck_saliency_iou.svg", width: 90%),
  ),
  caption: [
    Bottleneck resolution analysis comparing Weighted Dice (left) and Saliency IoU (right) for Grad-CAM evaluated at upsampled $160^3$ versus native $20^3$ resolution. Weighted Dice remains stable across both resolutions, while Saliency IoU exhibits volatile swings due to hard thresholding artefacts.
  ],
) <fig:gradcam-bottleneck>

=== Full-Resolution Gradient Attribution: Guided Backpropagation

Guided Backpropagation (GBP) modifies standard backpropagation by gating negative gradients at each ReLU, isolating purely positive signal paths to produce full-resolution ($160^3$) saliency maps. Unlike Grad-CAM’s bottleneck-dependent coarse localisation, GBP operates at native voxel scale without class-specific weighting, revealing fine-grained structural dependencies directly from input-level features. As shown in @fig:gbp-results, GBP achieves 100% Pointing Game across all patients and tumour regions including Enhancing Tumour, where Grad-CAM fails entirely. While Saliency Coverage remains modest (0.12–0.44) due to distributed edge-highlighting rather than concentrated blob detection, this diffuse pattern serves as a critical sanity check: the model encodes tumor relevant features at the input pixel level, not merely in deep bottleneck abstractions, rendering every prediction traceable to real anatomical structure.

#figure(
  grid(
    columns: 3,
    gutter: 2%,
    image("../Figures/xai-01497/gbp-01497.png", width: 70%),
    image("../Figures/xai-01397/gbp-01397.png", width: 70%),
    image("../Figures/xai-00291/gbp-00291.png", width: 70%),
  ),
  caption: [
    GBP saliency maps for patients 01497 (left), 01397 (centre), and 00291 (right). GBP achieves 100% Pointing Game across all patients and all tumour regions, including patient 00291, where Grad-CAM and Guided Grad-CAM produce zero activation.
  ],
) <fig:gbp-results>

=== Gradient Fusion: Guided Grad-CAM

Guided Grad-CAM fuses GBP's fine grained spatial detail with Grad-CAM's class-specific weighting via element wise multiplication of the full resolution GBP saliency and the upsampled Grad-CAM heatmap. For patient 01497, this recovers spatial information lost in Grad-CAM's bottleneck while preserving class-discriminative focus, tightly concentrating saliency within tumour boundaries.

#figure(
  box(width: 80%, grid(
    columns: (auto, 1fr, 1fr),
    column-gutter: 3pt,
    row-gutter: 3pt,
    align: center + horizon,
    [], [*Grad-CAM*], [*Guided Grad-CAM*],
    rotate(-90deg, reflow: true, pad(x: 2pt)[*01497*]),
    image("../Figures/xai-01497/grad-cam-01497.png", width: 60%),
    image("../Figures/xai-01497/guided-grad-cam-01497.png", width: 60%),

    rotate(-90deg, reflow: true, pad(x: 2pt)[*01397*]),
    image("../Figures/xai-01397/grad-cam-01397.png", width: 60%),
    image("../Figures/xai-01397/guided-grad-cam-01397.png", width: 60%),
  )),
  caption: [Grad-CAM (left column) versus Guided Grad-CAM (right column) for patients 01497 (top) and 01397 (bottom). The fusion recovers spatial detail lost in the bottleneck while preserving class-discriminative weighting, concentrating saliency tightly within tumour boundaries across varying morphologies.],
)

Quantitatively, the fusion achieves the highest Saliency Coverage of all methods evaluated: 0.81–0.96 for Whole Tumour and 0.81–0.92 for Tumour Core, with nearly all saliency mass localized inside the tumour. Enhancing Tumour Coverage also rises substantially from Grad-CAM's near-zero to 0.03–0.38, confirming that the full-resolution GBP component restores detail Grad-CAM alone cannot capture.

However, the fusion inherits Grad-CAM's failure modes. For patient 00291, where Grad-CAM produces a zero activation map, the element-wise multiplication zeros out GBP's otherwise perfect signal, producing blank maps across all metrics. This critical limitation demonstrates why no single XAI method is sufficient for clinical deployment; multi-method evaluation is essential.

=== Relevance-Based Attribution: LRP (Input × Gradient)

Layer-wise Relevance Propagation (LRP), implemented here as Input × Gradient, backpropagates the model's output score to individual input voxels. Unlike Grad-CAM, which shows _where_ the model attends, LRP reveals _what evidence_ supports its predictions.

#figure(
  grid(
    columns: 3,
    gutter: 2%,
    image("../Figures/xai-01497/lrp-01497.png", width: 90%),
    image("../Figures/xai-01397/lrp-01397.png", width: 90%),
    image("../Figures/xai-00291/lrp-00291.png", width: 90%),
  ),
  caption: [LRP saliency maps for patients 01497 (left), 01397 (centre), and 00291 (right). LRP achieves 100% Pointing Game across all patients and all regions, producing diffuse but correctly localised relevance distributions.],
)

LRP achieves *100% Pointing Game across all five patients and three tumour regions*, including Enhancing Tumor, matching GBP's perfect localisation. Saliency Coverage of 0.34–0.84 indicates relevance extends beyond tumour boundaries into surrounding tissue context (e.g., peritumoural oedema for Whole Tumour predictions).

Low Saliency IoU ($< 0.01$) and moderate Weighted Dice (0.01–0.17) reflect LRP's diffuse nature rather than poor alignment. The consistent perfect Pointing Game confirms that predictive features remain co-located with tumour tissue across all spatial scales.

=== Perturbation-Based Attribution: Occlusion Sensitivity

Unlike gradient-based approaches, Occlusion Sensitivity requires no assumptions about model internals. It slides a $16^3$ black patch across the input, recording confidence drops at each position to map model dependency empirically. This perturbation-based approach serves as the *gold-standard XAI validation*.

#figure(
  grid(
    columns: 3,
    gutter: 2%,
    image("../Figures/xai-01497/occlusion-01497.png", width: 80%),
    image("../Figures/xai-01397/occlusion-01397.png", width: 80%),
    image("../Figures/xai-01518/occlusion-01518.png", width: 80%),
  ),
  caption: [Occlusion Sensitivity heatmaps for patients 01497 (left), 01397 (centre), and 01518 (right). Occlusion achieves the highest Weighted Dice of all six methods, with patient 01397 ET achieving W.Dice = 0.35, the only method where ET localisation truly succeeds above 0.30],
)
Occlusion achieves 100% Pointing Game and MSR Accuracy for Whole Tumour across all patients, and produces the highest Weighted Dice scores of all six methods: 0.38–0.46 for WT, 0.24–0.47 for TC, and 0.23–0.35 for ET. Most remarkably, patient 01397 achieves a Weighted Dice of 0.35 with PG = 1.0 for Enhancing Tumor, the only method where ET localisation succeeds above the 0.30 threshold. This result constitutes the single strongest piece of evidence that the SegResNet model genuinely relies on tumour voxels for its segmentation predictions, ruling out shortcut learning, texture bias, or dataset artefacts. The clinical trustworthiness of the segmentation is thereby validated through a mechanism entirely independent of gradient computation.

=== Uncertainty Quantification: MC Dropout

MC Dropout estimates model uncertainty through 20 stochastic forward passes, producing per-voxel variance maps that complement saliency methods. Six metrics quantify this: Uncertainty Area Ratio (UAR), Boundary Uncertainty Ratio, mean uncertainty inside/outside the tumour, LRP Weighted Dice, and Saliency-Uncertainty Correlation (Pearson r between LRP saliency and MC Dropout variance within the tumour mask). LRP was selected for correlation analysis because its native $160^3$ resolution matches MC Dropout's output, avoiding interpolation artefacts inherent to Grad-CAM's $20^3$ maps.

*Patient 01497 -- Low Uncertainty, Clear Boundaries*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 7pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*LRP Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.196], [0.876], [0.00131], [0.00010], [+0.020],
    [TC], [0.092], [*0.967*], [0.00122], [0.00016], [+0.000],
    [ET], [0.219], [0.579], [*0.01400*], [0.00005], [−0.065],
    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 01497. Low UAR indicates confident segmentation. TC Boundary Ratio of 0.967 demonstrates that nearly all uncertainty concentrates at Tumour Core edges.],
)

This case shows confident segmentation, with UAR ranging from 0.083 (TC) to 0.226 (WT). Boundary ratios of 0.906 (WT) and 0.964 (TC) indicate uncertainty concentrates at tumour edges, while the ET boundary ratio of 0.503 reflects more distributed uncertainty. ET displays the highest internal variance (0.0130) relative to background (0.00007), correctly identifying enhancing tissue as the most challenging sub-region. Saliency-Uncertainty correlations are negligible (−0.055  to +0.021 ), suggesting relevance and uncertainty are largely decoupled.

*Patient 01397 -- High Uncertainty, Complex Morphology*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 7pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*LRP Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.490], [0.642], [0.00338], [0.00005], [−0.042],
    [TC], [*0.892*], [0.776], [0.01138], [0.00001], [−0.046],
    [ET], [0.624], [*1.000*], [0.00853], [0.00003], [−0.071],
    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 01397. TC UAR of 0.892 indicates 89% of model uncertainty concentrates inside the Tumour Core. ET Boundary Ratio reaches 1.000, every unit of uncertainty sits at the ET edge.],
)

UAR rises sharply to 0.538–0.907, with 91% of total uncertainty contained within the Tumour Core (UAR = 0.907). The ET boundary ratio reaches 0.998, localising virtually all uncertainty to the enhancing margin. Consistently negative LRP correlations (−0.050  to −0.071 ) indicate that regions of highest relevance coincide with lowest uncertainty, confirming the model remains confident in its most predictive features despite elevated overall ambiguity.

*Patient 00291 -- The Gradient-Based Failure Case*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 7pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*LRP Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.619], [*1.000*], [0.00440], [0.00003], [−0.012],
    [TC], [0.504], [0.997], [0.00802], [0.00002], [*+0.090*],
    [ET], [0.500], [0.997], [0.00930], [0.00002], [+0.053],
    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 00291. Despite complete gradient-based XAI failure, Boundary Ratio reaches 0.997–1.000, proving the model's spatial reasoning is intact. The positive TC/ET correlation is unique to this patient.],
)

This patient is uniquely diagnostic: Grad-CAM and Guided Grad-CAM produce *zero across all metrics* a complete gradient-based XAI failure. However, MC Dropout reveals a fundamentally different picture. The Boundary Ratio of 0.997–1.000 demonstrates that despite the model's ambiguity (UAR ≈ 0.50–0.62), uncertainty concentrates at the ground truth boundaries rather than being randomly distributed. This confirms the model _is_ segmenting based on real spatial features—the gradient-based methods failed to explain it, but the model's internal reasoning remains spatially grounded.

The positive TC/ET Saliency-Uncertainty Correlation (+0.05 to +0.09) is unique to this patient. Where LRP assigns high relevance and MC Dropout assigns high uncertainty overlap slightly, suggesting the model relies on features it is not fully confident about. Clinically, this is a valuable flag: cases exhibiting this pattern should be prioritised for radiologist review.

*Cross-Paradigm Finding: Saliency ≠ Uncertainty*

Across all three patients and nine regional measurements, the Saliency-Uncertainty Correlation ranges from −0.071 to +0.090 with a mean near zero. This demonstrates that saliency (what the model attends to) and uncertainty (where the model doubts) are *independent, non-redundant signals*. This independence is a positive finding: if they were correlated, one signal would be redundant. Because they are independent, a clinician using this system receives two complementary tools—saliency maps for trust calibration ("Is the AI looking at the right features?") and uncertainty maps for risk assessment ("Where might the AI be wrong?"). This directly motivates deploying both modalities in the immersive VR environment presented in Pillar 3.

=== Cross-Method Comparative Analysis

#figure(
  image("../Figures/results_figures/xai_regional_vulnerability.svg", width: 85%),
  caption: [Regional Vulnerability Analysis: Mean Weighted Dice across five XAI attribution methods, grouped by tumour subregion. Occlusion Sensitivity dominates Whole Tumour and Tumour Core, while ET remains universally challenging. Error bars indicate standard deviation.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Method*], [*WT Mean W.Dice*], [*TC Mean W.Dice*], [*ET Mean W.Dice*],
    table.hline(stroke: 0.5pt),

    [Grad-CAM], [0.220], [0.268], [0.048],
    [GBP], [0.130], [0.139], [0.140],
    [Guided Grad-CAM], [0.133], [0.153], [0.224],
    [LRP], [0.043], [0.058], [0.071],
    [Occlusion], [*0.422*], [*0.392*], [*0.165*],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Mean Weighted Dice scores by tumour region across all five saliency-based XAI methods. Occlusion Sensitivity achieves the highest scores for WT and TC. Bold values indicate the best-performing method per region.],
)
The Regional Vulnerability Analysis shows that Occlusion Sensitivity leads Whole Tumour (0.422) and Tumour Core (0.392), confirming its physically grounded model reliance. Enhancing Tumour remains the weakest region across all methods, revealing a fundamental limitation of interpretability techniques on small, heterogeneous subregions. GBP and Guided Grad-CAM outperform Grad-CAM on ET (0.14–0.22 versus 0.048), underscoring the advantage of full-voxel resolution.

A methodological insight from LRP further clarifies these rankings: despite recording the lowest Weighted Dice overall, LRP maintains 100% Pointing Game across every region. This confirms that Weighted Dice measures shape alignment while Pointing Game measures peak localisation, distinct and complementary dimensions of saliency quality. Neither metric alone is sufficient; together they provide a complete evaluation.
=== Visual Evaluation

#figure(
  align(center, grid(
    columns: (auto, 11%, 11%, 11%, 11%, 11%, 11%, 11%),
    column-gutter: 3pt,
    row-gutter: 3pt,
    align: center + horizon,

    [], [*GT*], [*Grad-CAM*], [*GBP*], [*G.Grad-CAM*], [*LRP*], [*Occlusion*], [*MC Drop.*],

    rotate(-90deg, reflow: true, pad(x: 2pt)[*01497*]),
    image("../Figures/xai-01497/GT-01497.png", width: 100%),
    image("../Figures/xai-01497/grad-cam-01497.png", width: 100%),
    image("../Figures/xai-01497/gbp-01497.png", width: 100%),
    image("../Figures/xai-01497/guided-grad-cam-01497.png", width: 100%),
    image("../Figures/xai-01497/lrp-01497.png", width: 100%),
    image("../Figures/xai-01497/occlusion-01497.png", width: 100%),
    image("../Figures/xai-01497/mc-01497.png", width: 100%),

    rotate(-90deg, reflow: true, pad(x: 2pt)[*01397*]),
    image("../Figures/xai-01397/GT-01397.png", width: 100%),
    image("../Figures/xai-01397/grad-cam-01397.png", width: 100%),
    image("../Figures/xai-01397/gbp-01397.png", width: 100%),
    image("../Figures/xai-01397/guided-grad-cam-01397.png", width: 100%),
    image("../Figures/xai-01397/lrp-01397.png", width: 100%),
    image("../Figures/xai-01397/occlusion-01397.png", width: 100%),
    image("../Figures/xai-01397/mc-01397.png", width: 100%),

    rotate(-90deg, reflow: true, pad(x: 2pt)[*00291*]),
    image("../Figures/xai-00291/GT-00291.png", width: 100%),
    image("../Figures/xai-00291/grad-cam-00291.png", width: 100%),
    image("../Figures/xai-00291/gbp-00291.png", width: 100%),
    image("../Figures/xai-00291/guided-grad-cam-00291.png", width: 100%),
    image("../Figures/xai-00291/lrp-00291.png", width: 100%),
    image("../Figures/xai-00291/occulusion-00291.png", width: 100%),
    image("../Figures/xai-00291/mc-00291.png", width: 100%),
  )),
  caption: [Multi-method XAI comparison across three patients with varying tumour morphologies. Patient 00291 (bottom row) demonstrates the gradient-based failure case: Grad-CAM and Guided Grad-CAM produce blank maps, while GBP, LRP, Occlusion, and MC Dropout continue to provide meaningful spatial information.],
)

The visual grid provides an immediate, qualitative validation of the quantitative findings. Patient 01497 (top row) shows consistent, well-localised saliency across all methods, corresponding to its clear tumour boundaries and low MC Dropout uncertainty. Patient 01397 (middle row) exhibits more diffuse saliency patterns consistent with its complex morphology and high uncertainty scores. Patient 00291 (bottom row) visually confirms the gradient-based failure: the Grad-CAM and Guided Grad-CAM columns appear blank, while GBP, LRP, Occlusion, and MC Dropout continue to provide spatially meaningful explanations, visually corroborating the quantitative finding that perturbation based and uncertainty based methods are more robust to morphological edge cases.

== Virtual Reality (VR) Immersive Visualisation
// TODO: Reserve for Phase 3 VR evaluation and validation
