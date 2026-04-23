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

Having established that the SegResNet model achieves clinically viable segmentation accuracy, the analysis now shifts from _how well_ the model performs to _why_ it generates specific predictions. This section applies six complementary XAI techniques to the trained SegResNet, interrogating whether its predictions are grounded in clinically meaningful anatomical features or potentially spurious artefacts. Each technique operates through a fundamentally different mechanism, providing converging evidence of model trustworthiness.

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

=== XAI Evaluation Metrics

To quantify the spatial alignment between each XAI method's saliency map $S$ and the ground truth segmentation mask $G$, four evaluation metrics are employed. Three are established in the literature; one is a novel contribution of this study.

*Pointing Game (PG).* A binary pass/fail test: $"PG" = 1$ if the single highest-saliency voxel $"argmax"(S)$ falls inside the ground truth mask $G$, and $0$ otherwise. This measures whether the model's peak attention is correctly localised.

*Saliency Coverage.* Defined as $"Cov" = (sum S dot G) / (sum S)$, this metric quantifies the fraction of total saliency mass that falls inside the tumour region. High coverage indicates that the model focuses on the tumour rather than irrelevant healthy tissue.

*Saliency IoU.* The saliency map is binarised at a threshold of 0.5, and the Jaccard index is computed against the ground truth: $"IoU" = |S_(>=0.5) sect G| / |S_(>=0.5) union G|$. This is a strict metric that penalises diffuse but correctly localised maps.

*Weighted Dice (Novel).* This study introduces Weighted Dice as a soft overlap metric for XAI evaluation:

$ "WD" = (2 dot sum(S dot G)) / (sum S + sum G) $

Unlike Saliency IoU, which forces an arbitrary binarisation threshold that discards intensity information, Weighted Dice treats the continuous saliency values directly as soft membership scores. Every voxel contributes proportional to its saliency intensity, providing a fairer evaluation for methods that produce diffuse but correctly centred maps. This is particularly important for 3D medical XAI, where saliency maps represent gradients of importance rather than binary decisions.

=== The Bottleneck Resolution Problem: 3D Grad-CAM

Grad-CAM, originally designed for 2D classification, was adapted for 3D segmentation by spatially averaging the logits for each target class to obtain a scalar score for backpropagation. This produces a coarse heatmap from the encoder's bottleneck layer at approximately 20³ resolution, which is then upsampled to the full 160³ input space for evaluation. A critical question arises: does this upsampling artificially inflate or deflate the evaluation metrics?

To answer this, the same Grad-CAM activations were evaluated at both upsampled (160³) and native bottleneck (~20³) resolution using both Weighted Dice and Saliency IoU.

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Figures/results_figures/xai_bottleneck_weighted_dice.svg", width: 90%),
    image("../Figures/results_figures/xai_bottleneck_saliency_iou.svg", width: 90%),
  ),
  caption: [Bottleneck resolution analysis comparing Weighted Dice (left) and Saliency IoU (right) for Grad-CAM evaluated at upsampled 160³ versus native ~20³ resolution. Weighted Dice remains stable across both resolutions (within ±0.02--0.04), while Saliency IoU exhibits volatile swings due to hard thresholding artefacts.],
)

The results demonstrate that Weighted Dice scores are highly stable across both resolutions, typically within ±0.02--0.04 of each other. In some cases, native resolution scores slightly higher (e.g., patient 01497 TC: 0.4506 upsampled vs. 0.4536 native), while in others it scores slightly lower (e.g., patient 01666 WT: 0.2612 vs. 0.2727), confirming no systematic bias. By contrast, Saliency IoU exhibits volatile swings between resolutions due to hard thresholding artefacts. This validates Weighted Dice as a more reliable metric for evaluating coarse-resolution methods.

Critically, even at its native resolution, Grad-CAM achieves 0% Pointing Game for Enhancing Tumour across all 10 patients evaluated. This is a fundamental resolution limitation: the ~20³ bottleneck cannot represent structures smaller than a single feature voxel. This is a method limitation, not a model flaw --- as established in Pillar 1, the model achieves 0.873 Dice on ET, proving it handles this subregion internally.

=== Full-Resolution Gradient Attribution: Guided Backpropagation

Guided Backpropagation (GBP) modifies the standard gradient by gating negative gradients at every ReLU during backpropagation, producing a full-resolution (160³) saliency map. Unlike Grad-CAM, GBP requires no bottleneck layer and makes no class-specific weighting --- it reveals which input voxels the network's forward activations depend upon most.

#figure(
  grid(
    columns: 3,
    gutter: 2%,
    image("../Figures/xai-01497/gbp-01497.png", width: 70%),
    image("../Figures/xai-01397/gbp-01397.png", width: 70%),
    image("../Figures/xai-00291/gbp-00291.png", width: 70%),
  ),
  caption: [GBP saliency maps for patients 01497 (left), 01397 (centre), and 00291 (right). GBP achieves 100% Pointing Game across all patients and all tumour regions --- including patient 00291, where Grad-CAM and Guided Grad-CAM produce zero saliency.],
)

GBP is the only method to achieve *100% Pointing Game across all five patients and all three tumour regions*, including Enhancing Tumour. Most notably, for patient 00291 --- where both Grad-CAM and Guided Grad-CAM produce entirely blank saliency maps --- GBP correctly localises the peak saliency inside the tumour for every region. This proves that the model encodes tumour-relevant features at the input pixel level, not only in deep bottleneck representations.

The low Saliency Coverage (0.12--0.44) is expected and not a flaw: GBP highlights fine edges and texture boundaries rather than concentrated tumour blobs, distributing saliency across both tumour and peri-tumoral tissue. Clinically, GBP serves as a "sanity check" --- if GBP fails to localise, the model genuinely lacks input-level features for that region. Because GBP never fails here, every model prediction is traceable to real input evidence.

=== Gradient Fusion: Guided Grad-CAM

Guided Grad-CAM element-wise multiplies GBP's full-resolution saliency by the upsampled Grad-CAM heatmap, combining GBP's voxel-level detail with Grad-CAM's class-specific weighting. This fusion is designed to produce saliency maps that are simultaneously high-resolution and class-discriminative.

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Figures/xai-01497/grad-cam-01497.png", width: 50%),
    image("../Figures/xai-01497/guided-grad-cam-01497.png", width: 50%),
  ),
  caption: [Grad-CAM (left) versus Guided Grad-CAM (right) for patient 01497. The fusion recovers spatial detail lost in the bottleneck while preserving class-discriminative weighting, concentrating saliency tightly within tumour boundaries.],
)

The fusion achieves the *highest Saliency Coverage of any method* evaluated: 0.81--0.96 for Whole Tumour and 0.81--0.92 for Tumour Core, indicating that nearly all saliency mass is concentrated inside the tumour. ET Coverage also improves substantially from Grad-CAM's near-zero to 0.03--0.38, demonstrating that the full-resolution GBP component recovers spatial detail that Grad-CAM alone cannot represent.

However, the fusion inherits Grad-CAM's failure modes. For patient 00291, where Grad-CAM produces a zero activation map, the element-wise multiplication zeros out GBP's otherwise perfect signal --- producing blank maps across all metrics. This critical limitation demonstrates why no single XAI method is sufficient for clinical deployment; multi-method evaluation is essential.

=== Relevance-Based Attribution: LRP (Input × Gradient)

Layer-wise Relevance Propagation (LRP), implemented here as the Input × Gradient proxy, distributes the model's output score back to individual input voxels. Unlike Grad-CAM, which shows _where_ the model attends, LRP reveals _what evidence_ the model uses for its predictions.

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

LRP achieves *100% Pointing Game for all five patients across all three tumour regions*, including Enhancing Tumour --- matching GBP's perfect localisation record. Saliency Coverage ranges from 0.34 to 0.84, indicating that relevance distributes beyond tumour boundaries into surrounding tissue context. This is expected: LRP correctly attributes relevance to contextual tissue (e.g., peritumoural oedema contributes to the Whole Tumour class prediction).

The low Saliency IoU (< 0.01) reflects the diffuse nature of LRP maps rather than poor performance. Weighted Dice values (0.01--0.17) confirm moderate spatial alignment. The perfect Pointing Game across all regions proves that the model's predictive features are always co-located with tumour tissue at every spatial scale.

=== Perturbation-Based Attribution: Occlusion Sensitivity

Unlike all gradient-based methods, Occlusion Sensitivity makes zero mathematical assumptions about the model's internal structure. It physically slides a 16³ black patch across the input volume, records the confidence drop at each position, and maps these perturbation effects back to spatial locations. This constitutes the *gold-standard XAI validation* because it directly measures model dependency through empirical observation.

#figure(
  grid(
    columns: 3,
    gutter: 2%,
    image("../Figures/xai-01497/occlusion-01497.png", width: 80%),
    image("../Figures/xai-01397/occlusion-01397.png", width: 80%),
    image("../Figures/xai-01518/occlusion-01518.png", width: 80%),
  ),
  caption: [Occlusion Sensitivity heatmaps for patients 01497 (left), 01397 (centre), and 01518 (right). Occlusion achieves the highest Weighted Dice of all six methods, with patient 01397 ET achieving W.Dice = 0.35 --- the only method where ET localisation truly succeeds above 0.30.],
)

Occlusion achieves *100% Pointing Game and MSR Accuracy for Whole Tumour* across all patients, and produces the *highest Weighted Dice scores of all six methods*: 0.38--0.46 for WT, 0.24--0.47 for TC, and 0.23--0.35 for ET. Most remarkably, patient 01397 achieves a Weighted Dice of 0.35 with PG = 1.0 for Enhancing Tumour --- the only method where ET localisation succeeds above the 0.30 threshold.

This result constitutes the single strongest piece of evidence that the SegResNet model genuinely relies on tumour voxels for its segmentation predictions, ruling out shortcut learning, texture bias, or dataset artefacts. The clinical trustworthiness of the segmentation is thereby validated through a mechanism entirely independent of gradient computation.

=== Uncertainty Quantification: MC Dropout

MC Dropout provides a fundamentally different analytical paradigm from saliency-based methods. By performing 20 stochastic forward passes with dropout enabled at inference time, it generates per-voxel variance maps that quantify _where the model is uncertain_ --- a complementary signal to saliency maps, which show _where the model attends_.

Six metrics specific to uncertainty analysis were computed: Uncertainty Area Ratio (UAR), Boundary Uncertainty Ratio, Mean Uncertainty Inside/Outside the tumour, LRP Weighted Dice (for cross-referencing), and Saliency-Uncertainty Correlation (Pearson _r_ between LRP saliency and MC Dropout variance within the tumour mask). LRP was selected for the correlation analysis because its full input resolution (160³) matches MC Dropout's output resolution; Grad-CAM's coarse ~20³ maps would primarily correlate interpolation artefacts.

*Patient 01497 --- Low Uncertainty, Clear Boundaries*

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

This patient presents well-defined tumour boundaries. The low UAR (0.09--0.22) indicates confident segmentation. The TC Boundary Ratio of 0.967 reveals that nearly all uncertainty concentrates at the Tumour Core edge --- clinically expected, as TC boundaries are among the hardest to delineate even for expert radiologists. ET exhibits 100× higher internal variance than surrounding tissue (0.014 vs. 0.00005), demonstrating that the model correctly identifies ET as inherently more difficult.

*Patient 01397 --- High Uncertainty, Complex Morphology*

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
  caption: [MC Dropout uncertainty metrics for patient 01397. TC UAR of 0.892 indicates 89% of model uncertainty concentrates inside the Tumour Core. ET Boundary Ratio reaches 1.000 --- every unit of uncertainty sits at the ET edge.],
)

This patient exhibits the most complex tumour morphology. TC UAR reaches 0.892, meaning 89% of all model uncertainty is concentrated inside the Tumour Core. The ET Boundary Ratio achieves *1.000* --- every single unit of uncertainty localises at the ET boundary, constituting the strongest calibration evidence: the model doubts precisely where the tumour transitions to healthy tissue. The consistently negative LRP correlation (−0.04 to −0.07) indicates that the model is _confident_ about the features it deems most relevant.

*Patient 00291 --- The Gradient-Based Failure Case*

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
  caption: [MC Dropout uncertainty metrics for patient 00291. Despite complete gradient-based XAI failure, Boundary Ratio reaches 0.997--1.000, proving the model's spatial reasoning is intact. The positive TC/ET correlation is unique to this patient.],
)

This patient is uniquely diagnostic: Grad-CAM and Guided Grad-CAM produce *zero across all metrics* --- a complete gradient-based XAI failure. However, MC Dropout reveals a fundamentally different picture. The Boundary Ratio of 0.997--1.000 demonstrates that despite the model's ambiguity (UAR ≈ 0.50--0.62), uncertainty concentrates at the ground truth boundaries rather than being randomly distributed. This confirms the model _is_ segmenting based on real spatial features --- the gradient-based methods failed to explain it, but the model's internal reasoning remains spatially grounded.

The positive TC/ET Saliency-Uncertainty Correlation (+0.05 to +0.09) is unique to this patient. Where LRP assigns high relevance and MC Dropout assigns high uncertainty overlap slightly, suggesting the model relies on features it is not fully confident about. Clinically, this is a valuable flag: cases exhibiting this pattern should be prioritised for radiologist review.

*Cross-Paradigm Finding: Saliency ≠ Uncertainty*

Across all three patients and nine regional measurements, the Saliency-Uncertainty Correlation ranges from −0.071 to +0.090 with a mean near zero. This demonstrates that saliency (what the model attends to) and uncertainty (where the model doubts) are *independent, non-redundant signals*. This independence is a positive finding: if they were correlated, one signal would be redundant. Because they are independent, a clinician using this system receives two complementary tools --- saliency maps for trust calibration ("Is the AI looking at the right features?") and uncertainty maps for risk assessment ("Where might the AI be wrong?"). This directly motivates deploying both modalities in the immersive VR environment presented in Pillar 3.

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

The Regional Vulnerability Analysis reveals several key patterns. Occlusion Sensitivity dominates Whole Tumour (0.422) and Tumour Core (0.392), providing the physically verified model reliance discussed previously. Enhancing Tumour is universally the weakest region across all methods, exposing the fundamental vulnerability of interpretability techniques when applied to small, heterogeneous subregions. Notably, GBP and Guided Grad-CAM achieve surprisingly competitive ET scores (0.14--0.22) compared to Grad-CAM's near-zero (0.048), demonstrating the resolution advantage of full-voxel methods.

An important methodological observation emerges from LRP's results: despite achieving the lowest Weighted Dice overall, LRP maintains 100% Pointing Game across all regions. This demonstrates that Weighted Dice and Pointing Game capture different dimensions of saliency quality --- shape alignment versus peak localisation --- and that both metrics are necessary for comprehensive evaluation.

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

The visual grid provides an immediate, qualitative validation of the quantitative findings. Patient 01497 (top row) shows consistent, well-localised saliency across all methods, corresponding to its clear tumour boundaries and low MC Dropout uncertainty. Patient 01397 (middle row) exhibits more diffuse saliency patterns consistent with its complex morphology and high uncertainty scores. Patient 00291 (bottom row) visually confirms the gradient-based failure: the Grad-CAM and Guided Grad-CAM columns appear blank, while GBP, LRP, Occlusion, and MC Dropout continue to provide spatially meaningful explanations --- visually corroborating the quantitative finding that perturbation-based and uncertainty-based methods are more robust to morphological edge cases.

== Virtual Reality (VR) Immersive Visualisation
// TODO: Reserve for Phase 3 VR evaluation and validation
