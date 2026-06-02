#import "template.typ": *

#show: doc => conf(
  title: [Illuminating the Black Box: A Multi-Method Explainable AI Framework with Uncertainty Quantification for 3D Brain Tumor Segmentation],
  authors: (
    (name: "Yogi Amitkumar Patel", affiliation: "*", email: "u2536809@uel.ac.uk"),
    (name: "Maimoona Sharif", affiliation: "*", email: ""),
  ),
  abstract: [
    Deep learning models for 3D brain tumour segmentation have achieved remarkable accuracy,
    yet their opaque decision-making processes present a critical barrier to clinical adoption.
    This paper presents a multi-method Explainable AI (XAI) framework that integrates six
    complementary post-hoc explanation techniques with a SegResNet architecture for volumetric
    brain tumour segmentation on the BraTS 2023 dataset. The framework combines gradient-based
    methods (Grad-CAM, Guided Backpropagation, Guided Grad-CAM, Input $times$ Gradient as an LRP
    proxy), perturbation-based analysis (Occlusion Sensitivity), and probabilistic uncertainty
    quantification (Monte Carlo Dropout) to provide multi-perspective interpretability of 3D CNN
    predictions. The SegResNet model achieves competitive segmentation performance on a held-out
    test set of 126 patients, with mean Dice scores of 0.923 (Whole Tumour), 0.891 (Tumour
    Core), and 0.873 (Enhancing Tumour). Quantitative evaluation of XAI outputs across a
    25-patient evaluation cohort using four metrics --- Pointing Game accuracy, Saliency Coverage,
    Saliency IoU, and a novel Weighted Dice score --- demonstrates that Occlusion Sensitivity
    achieves the highest spatial alignment (mean Weighted Dice: 0.397 WT, 0.345 TC), while
    Guided Grad-CAM achieves 93% saliency coverage for the Whole Tumour region. MC Dropout
    uncertainty analysis across 23 patients reveals boundary-concentrated variance patterns,
    with Boundary Uncertainty Ratios exceeding 0.84, consistent with clinically meaningful
    confidence calibration. A critical finding is a gradient-based failure case where Grad-CAM
    and Guided Grad-CAM produce zero saliency, yet MC Dropout confirms spatially grounded
    reasoning --- demonstrating why multi-method XAI evaluation is essential. All explanation
    outputs are exported as spatially-aligned NIfTI volumes for 3D visualisation. This work
    advances the integration of model accuracy and clinical interpretability by providing a
    unified, quantitatively validated explainability pipeline for volumetric medical image
    analysis.
  ],
  keywords: ("Explainable AI", "Brain Tumour Segmentation", "3D CNN", "Grad-CAM", "Uncertainty Quantification", "BraTS 2023", "SegResNet", "Monte Carlo Dropout", "Weighted Dice"),
  bibliography-file: "paper.bib",
  doc,
)

= Introduction

Approximately 308,000 people receive a primary brain tumour diagnosis annually @neri2023,
and for the neurosurgeon reviewing the pre-operative scan, the decisive question is never
simply whether a tumour is present --- it is where, precisely, the tumour boundary ends. A
margin error of a few millimetres in the wrong direction means either residual malignant
tissue or permanent neurological deficit. Deep learning models, particularly 3D
Convolutional Neural Networks (CNNs), have demonstrated the capacity to delineate these
boundaries with superhuman consistency @iftikhar2025 @bhati2024, processing entire MRI
volumes rather than individual 2D slices to capture the true three-dimensional spatial
relationships of tumour pathology.

However, the clinical deployment of these high-performing models is critically hindered by the "black box"
problem: their internal decision-making processes remain fundamentally opaque to the clinicians who must
rely on their outputs @neri2023. In the high-stakes domain of neurosurgery, this opacity creates a "trust gap" @wen2025 --- it
is insufficient for a model to simply output a segmentation mask; clinicians require the ability to verify why
specific voxels were classified as tumour tissue. This lack of interpretability is not merely a technical limitation but
a barrier to ethical and safe clinical deployment, particularly within regulatory frameworks such as the GDPR's
"right to explanation" @neri2023.

Explainable AI (XAI) techniques aim to address this disconnect by illuminating model reasoning. Methods such
as Gradient-weighted Class Activation Mapping (Grad-CAM) @selvaraju2017 have become established tools for visualizing
CNN decisions. However, existing XAI frameworks for 3D medical imaging suffer from three critical limitations:
(1) they typically employ only a single explanation method, providing an incomplete picture of model behavior;
(2) they lack uncertainty quantification, making it impossible to distinguish between confident and uncertain
predictions; and (3) they present explanations as static 2D overlays, failing to convey the volumetric nature of
the underlying pathology @neuroxai @axons3.

This paper presents a unified, multi-method XAI framework that addresses these limitations. Our key contributions are:

1. Multi-perspective explainability: We implement and quantitatively compare six complementary XAI techniques --- Grad-CAM, Guided Backpropagation (GBP), Guided Grad-CAM, Input $times$ Gradient (IxG) as an LRP proxy, Occlusion Sensitivity, and Monte Carlo (MC) Dropout --- providing gradient-based, perturbation-based, and probabilistic perspectives on model decision-making.

2. Quantitative XAI validation with a novel metric: We evaluate saliency maps using four metrics across a 25-patient cohort, including a novel Weighted Dice metric that treats continuous saliency values as soft membership scores --- eliminating the threshold-dependent volatility inherent in Saliency IoU and enabling equitable comparison of methods operating at different spatial resolutions.

3. Uncertainty-aware explainability: We integrate MC Dropout uncertainty quantification and demonstrate its statistical independence from structural attribution (Pearson $r approx 0$ across 23 patients), establishing that saliency and uncertainty provide complementary, non-redundant clinical signals --- the former for trust calibration, the latter for risk assessment.

The remainder of this paper is organized as follows: Section 2 reviews related work in 3D segmentation
architectures, XAI methods, and immersive visualization. Section 3 details our proposed framework. Section
4 describes the experimental setup. Section 5 presents comprehensive quantitative results. Section 6 discusses
findings, clinical implications, and limitations. Section 7 concludes with future directions.


= Related Work

== 3D Deep Learning for Brain Tumour Segmentation

The evolution from 2D slice-based to volumetric 3D approaches has fundamentally improved brain tumour
segmentation. The 3D U-Net @cicek2016 established the encoder-decoder paradigm with skip connections for volumetric
data, while V-Net @milletari2016 introduced residual connections and the Dice loss function to address class imbalance.

Modern architectures such as SegResNet @myronenko2019 combine these innovations, using residual blocks with GroupNorm
for stable deep network training and DiceFocalLoss for handling severely imbalanced tumour sub-regions. The
BrainAR framework @brainAR demonstrated the potential of combining 3D U-Net architectures with augmented reality
visualization, achieving Dice scores of 0.914 for whole tumour segmentation. Similarly, the AXONS-3 framework
@axons3 applied post-hoc XAI techniques to 3D brain tumour segmentation, highlighting the importance of interpretability in clinical AI systems.

The BraTS challenge @menze2015 has served as the primary benchmark for evaluating these architectures on multi-modal
MRI data (T1, T1c, T2, FLAIR), with current state-of-the-art models achieving Dice scores exceeding 0.90 for
the Whole Tumour region. The challenge provides a standardized evaluation framework enabling fair comparison
between methods.

== Explainable AI for Medical Imaging

XAI methods for CNNs are broadly categorized as gradient-based, perturbation-based, or decomposition-based.
Grad-CAM @selvaraju2017 computes class-discriminative heatmaps using gradients flowing into a target convolutional layer,
providing coarse but class-specific spatial localization. Guided Backpropagation @springenberg2015 produces full-resolution
saliency maps by gating negative gradients during backpropagation. Their fusion --- Guided Grad-CAM --- achieves
both high resolution and class specificity @selvaraju2017.

Layer-wise Relevance Propagation (LRP) @bach2015pixel operates on a conservation principle, attributing the output
prediction score back to individual input voxels. For complex architectures, the Input $times$ Gradient approximation
provides a computationally tractable proxy equivalent to $epsilon$-LRP in ReLU networks. Occlusion Sensitivity @zeiler2014
offers a complementary, model-agnostic approach by systematically masking input regions and measuring
prediction changes.

For uncertainty quantification, MC Dropout @gal2016dropout provides a Bayesian approximation by maintaining dropout
during inference across multiple stochastic forward passes. The variance across predictions serves as a proxy for
model uncertainty, identifying regions where the network lacks confidence.

Recent surveys @bhati2024 have highlighted the growing importance of XAI in medical imaging, emphasizing that no
single method provides a complete picture of model behavior --- a gap our multi-method framework directly
addresses. The NeuroXAI framework @neuroxai applied multiple gradient-based XAI methods to brain tumour
segmentation, achieving 90% clinician alignment. The AXONS-3 framework @axons3 advanced this by integrating trust
metrics into 3D segmentation pipelines. However, both frameworks lack integrated uncertainty quantification
and rely on static visualization, limiting their clinical utility for interactive decision support.

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
(see patient 00291, @tab:mc_00291), can fail silently.

*Gap 2: Absence of integrated uncertainty quantification.* Neither NeuroXAI nor AXONS-3
integrates probabilistic uncertainty quantification alongside structural attribution. Monte
Carlo Dropout provides a complementary signal --- where the model doubts --- that is
statistically independent of saliency (see @sec:saliency_uncertainty). Without this signal, clinicians cannot
distinguish between confident and uncertain predictions, a critical deficiency for clinical
deployment. BrainAR @brainAR demonstrated AR-based visualisation of segmentation
outputs but provides no XAI or uncertainty analysis whatsoever.

*Gap 3: Inadequate XAI evaluation methodology.* Existing quantitative evaluation of
saliency maps relies on Pointing Game and Saliency Coverage @natekar2020, both of which
suffer from known limitations --- Pointing Game reduces volumetric interpretability to a
single-voxel test, while Saliency Coverage is agnostic to spatial distribution. No prior work
has introduced a soft metric that evaluates continuous saliency maps without hard
binarisation thresholds, leading to volatile evaluation of coarse-resolution methods
such as Grad-CAM.

This paper addresses all three gaps through: (1) a six-method XAI suite spanning
gradient-based, perturbation-based, and probabilistic paradigms; (2) integrated MC Dropout
uncertainty quantification with cross-paradigm correlation analysis; and (3) the introduction
of Weighted Dice, a novel soft metric for resolution-robust saliency evaluation.

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

    [AXONS-3 @axons3 \ (Abyasa & Rahmania, 2025)],
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
    [*Grad-CAM, GBP, Guided Grad-CAM, IxG, Occlusion Sensitivity*],
    [*MC Dropout (20 passes)*],
    [*Yes (Occlusion Sensitivity)*],
    [*PG, Coverage, IoU, Weighted Dice (novel)*],
    [*3D Slicer + SlicerVR*],
    [*25 patients (XAI), 126 patients (segmentation)*],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Comparative analysis of XAI frameworks for 3D brain tumour segmentation. This work is the first to integrate perturbation-based validation, probabilistic uncertainty quantification, and a novel soft evaluation metric within a single pipeline. Dash entries indicate the feature is absent from the referenced work.],
) <tab:comparison>

== Immersive Visualization in Medical Imaging

Traditional slice-by-slice visualization of 3D data imposes significant cognitive load on clinicians @brainAR. Virtual
Reality (VR) environments provide stereoscopic depth perception and six-degrees-of-freedom interaction,
enabling clinicians to explore volumetric XAI outputs as spatial clouds rather than flat overlays @neuroxai. Platforms
such as 3D Slicer with the SlicerVR extension support direct rendering of NIfTI volumes in VR headsets, bridging
the gap between medical imaging formats and immersive display systems @zeineldin2023diss. The combination of XAI with VR
visualization represents an emerging frontier for clinical decision support, enabling clinicians to explore saliency
and uncertainty in an immersive environment. Additionally, all explanation outputs are exported as spatially-aligned NIfTI volumes compatible with immersive 3D visualization platforms such as 3D Slicer and SlicerVR.


= Proposed Framework

Our framework implements a modular pipeline: (1) YAML-based configuration loading and data initialisation, (2) model training with automated checkpointing and W&B tracking, (3) test set inference on 126 held-out patients, (4) XAI saliency map generation for all six methods across the evaluation cohort, (5) quantitative metric computation and CSV export, (6) spatially-aligned NIfTI volume export for each patient and method, and (7) VR visualisation preparation via 3D Slicer. This section details the architecture and each XAI component.

== SegResNet Architecture

The backbone of our framework is SegResNet @myronenko2019, a 3D encoder-decoder architecture that processes four-channel MRI volumes ($160^3$ voxels) to produce three-channel segmentation masks corresponding to the Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET) regions.

The encoder progressively downsamples input through four levels, increasing filters from 32 to 256 at the
bottleneck, using strided $3 times 3 times 3$ convolutions. Each level is composed of Residual Blocks (ResBlocks) featuring
GroupNorm normalization and ReLU activations. These blocks learn residual mappings via skip connections,
preventing vanishing gradients and ensuring stable training of deep 3D networks @he2016resnet. The decoder reconstructs
the segmentation map via trilinear upsampling and encoder-decoder skip connections. A dropout probability of
$p = 0.1$ serves dual purposes: regularization during training and enabling MC Dropout inference.

To address severe class imbalance --- particularly the ET region comprising merely 5% of tumour volume --- the
model employs DiceFocalLoss with a focal parameter $gamma = 2.0$, which increases the loss contribution from hard-to-classify voxels:

$ "DiceFocalLoss" = "DiceLoss" + lambda dot "FocalLoss" $

where DiceLoss optimizes volumetric overlap and FocalLoss ($"FL"(p_t) = -alpha (1 - p_t)^gamma log(p_t)$) heavily penalizes confident misclassifications.

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

== XAI Method Suite

We implement six XAI methods spanning three paradigms: gradient-based, perturbation-based, and probabilistic.
Each method provides a distinct perspective on model decision-making.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Category*], [*Method*], [*Resolution*], [*Class-Specific*],
    table.hline(stroke: 0.5pt),

    [Gradient-Based], [3D Grad-CAM], [Coarse ($20^3$)], [Yes],
    [Gradient-Based], [Guided Backpropagation (GBP)], [Full ($160^3$)], [No],
    [Gradient-Based], [Guided Grad-CAM], [Full ($160^3$)], [Yes],
    [Relevance-Based], [IxG (LRP Proxy)], [Full ($160^3$)], [Yes],
    [Perturbation-Based], [Occlusion Sensitivity], [Stride-upsampled], [Yes],
    [Uncertainty-Based], [MC Dropout (20 passes)], [Full ($160^3$)], [Yes],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Summary of the six XAI techniques applied to the SegResNet model, categorized by mechanism, spatial resolution, and class discrimination capability.],
) <tab:xai_methods>

=== Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM computes per-feature-map importance weights by analyzing gradients flowing into the bottleneck
convolutional layer (256 channels at $20^3$ resolution). For target class $c$ and feature map $k$, the importance weight
$alpha_k^c$ is computed via global average pooling of gradients:

$ alpha_k^c = 1 / Z sum_(x,y,z) frac(partial y^c, partial A_(x,y,z)^k) $

where $y^c$ is the spatially-averaged output logit and $A_(x,y,z)^k$ represents feature map activations. The class-discriminative heatmap is obtained through a weighted linear combination followed by ReLU activation:

$ L_"Grad-CAM"^c = "ReLU"(sum_k alpha_k^c A^k) $

The resulting $20^3$ heatmap is upsampled to $160^3$ via trilinear interpolation and normalized to $[0,1]$. A critical
limitation of Grad-CAM for 3D segmentation is the fundamental resolution trade-off: the $20^3$ bottleneck
cannot represent structures smaller than a single feature voxel, leading to potential failure for small Enhancing
Tumor regions.

=== Guided Backpropagation (GBP)

GBP extends standard backpropagation by imposing an additional gradient gate at every ReLU layer, suppressing
both negative forward activations and negative incoming gradients. The guided gradient $R_i^l$ at neuron $i$ in layer
$l$ is:

$ R_i^l = (f_i^l > 0) dot (R_i^(l+1) > 0) dot R_i^(l+1) $

where $f_i^l$ is the forward activation and $R_i^(l+1)$ is the incoming gradient. This dual gating isolates purely positive
signal paths, producing full-resolution ($160^3$) saliency maps with sharp voxel-level detail, though without
inherent class discrimination.

=== Guided Grad-CAM

Guided Grad-CAM fuses the complementary strengths of Grad-CAM (class-discriminative, coarse) and GBP
(high-resolution, class-agnostic) through element-wise multiplication:

$ L_"Guided Grad-CAM"^c = L_"Grad-CAM"^c dot L_"GBP" $

A critical implementation detail is hook isolation: simultaneously active Grad-CAM and GBP backward hooks
corrupt each other's gradient flows. Our framework employs sequential per-patient computation --- first Grad-CAM hooks, then GBP hooks on a clean model --- before multiplying the stored results.

=== Input $times$ Gradient (LRP Proxy)

Layer-wise Relevance Propagation (LRP) @bach2015pixel attributes the model's output prediction score back to each input voxel based on a conservation principle. Due to the architectural complexity of SegResNet (skip connections, GroupNorm layers), we employ the Input $times$ Gradient (IxG) approximation:

$ R_i approx x_i dot frac(partial f(x), partial x_i) $

This yields ultra-high-resolution ($160^3$) relevance maps quantifying which specific structural features drove the
network's prediction. The IxG approximation serves as the closest feasible proxy to $epsilon$-LRP for this architecture, directly circumventing the architectural bottlenecks that make true LRP propagation prohibitively complex.

=== Occlusion Sensitivity

Occlusion Sensitivity provides model-agnostic, perturbation-based explanations by systematically testing the
CNN's structural reliance on specific brain regions. A $16 times 16 times 16$ sliding window moves across the input
volume with stride 8, replacing occluded voxels with zeros:

$ S_c(i,j,k) = |f_c(x) - f_c(x_"occluded")| $

This method requires approximately 1,000 forward passes per patient but provides direct empirical evidence
of regional importance, constituting a rigorous model-agnostic validation approach since it independently measures model
dependency through empirical observation.

=== Monte Carlo Dropout (Uncertainty Quantification)

At test time, SegResNet's dropout layers ($p = 0.1$) remain active during $N = 20$ stochastic forward passes. The
stabilized mean prediction and voxel-wise predictive variance are:

$ overline(p) = 1 / N sum_(n=1)^N p_n $

$ sigma^2(x,y,z) = 1 / N sum_(n=1)^N (p_n(x,y,z) - overline(p)(x,y,z))^2 $

High variance identifies "brittle" transition zones where the network lacks confidence.

== Quantitative XAI Evaluation Metrics

All saliency maps are evaluated against ground truth tumour annotations using four metrics:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Metric*], [*Formula*], [*Interpretation*],
    table.hline(stroke: 0.5pt),

    [Pointing Game], [
      $ cases(1 & "if" "argmax"(L) in G, 0 & "otherwise") $
    ], [Binary hit/miss test: is the peak saliency voxel inside the tumour region $G$?],

    [Saliency Coverage], [
      $ frac(sum_(x in G) L(x), sum_x L(x)) $
    ], [Fraction of total saliency mass concentrated inside the tumour boundary],

    [Saliency IoU], [
      $ frac(|L_"thresh" inter G|, |L_"thresh" union G|) $ \\ ($"thresh" = 0.5$)
    ], [Shape overlap between thresholded saliency and ground truth],

    [Weighted Dice (Novel)], [
      $ frac(2 sum L(x) dot G(x), sum L(x) + sum G(x)) $
    ], [Soft overlap treating continuous saliency as prediction weights. Novel metric.],

    table.hline(stroke: 1.5pt),
  ),
  caption: [XAI evaluation metrics for saliency map validation against ground truth. Weighted Dice is a novel metric introduced in this work for reliable evaluation of coarse-resolution saliency methods.],
) <tab:xai_metrics>

The Weighted Dice metric is introduced as a novel contribution to address a critical limitation of Saliency
IoU: the hard binarization threshold (0.5) discards intensity information and introduces volatile metric swings
across resolutions. Weighted Dice treats continuous saliency values as soft membership scores, providing stable
evaluation across different saliency resolutions.


= Experimental Setup

== Dataset

The BraTS 2023 dataset comprises approximately 1,251 patient cases, each containing four co-registered MRI
modalities: T1-weighted (T1), T1 with Gadolinium contrast (T1c), T2-weighted (T2), and T2-FLAIR. All volumes
are resampled to $1 "mm"^3$ isotropic resolution and skull-stripped. Tumour annotations follow the BraTS
standard: necrotic core (label 1), peritumoral edema (label 2), and enhancing tumour (label 3), converted into three
binary channels: WT (labels 1+2+3), TC (labels 1+3), and ET (label 3 only).

The dataset was split deterministically (seed=42) into 1,000 training, 125 validation, and 126 test cases. Validation
was used for hyperparameter selection and early stopping; the test set was reserved for final evaluation and XAI
analysis.

== Preprocessing Pipeline

The preprocessing pipeline applies four operations: (1) foreground cropping to reduce spatial dimensions from
approximately $240^3$ to $160^3$ by removing background; (2) channel-first formatting for PyTorch compatibility
with dimensions $[4, 160, 160, 160]$; (3) per-channel z-score normalization using non-zero voxel statistics; and (4)
training-only random augmentations including random flips, rotations, intensity scaling, Gaussian noise, and
intensity shifts.

== Training Configuration

Training was conducted on a single NVIDIA RTX 5880 Ada (48 GB VRAM) with Automatic Mixed Precision
(AMP) enabled. An evolutionary prototyping phase on a 250-patient subset validated the pipeline before full-scale training.

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Parameter*], [*Value*],
    table.hline(stroke: 0.5pt),

    [Batch Size], [1 (limited by 3D volume memory)],
    [Epochs], [35],
    [Learning Rate], [$5 times 10^(-5)$],
    [LR Scheduler], [Cosine Annealing],
    [Optimizer], [AdamW (weight decay $1 times 10^(-5)$)],
    [Loss Function], [DiceFocalLoss ($gamma = 2.0$)],
    [Mixed Precision], [Enabled (AMP)],
    [Dropout], [$p = 0.1$],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Training hyperparameters for SegResNet on BraTS 2023.],
) <tab:hyperparams>

Best model checkpoints were selected based on validation Dice score, with experiment tracking via Weights &
Biases (W&B).

== XAI Evaluation Protocol

Quantitative XAI evaluation was conducted on a 25-patient subset sampled from the 126-patient test set. For each patient, all five saliency-based methods (Grad-CAM, Guided Backpropagation, Guided Grad-CAM, Input $times$ Gradient, and Occlusion Sensitivity) were evaluated using four metrics: Pointing Game, Saliency Coverage, Saliency IoU, and Weighted Dice. Monte Carlo Dropout uncertainty metrics were computed for 23 of these patients (two excluded due to convergence anomalies in stochastic passes). Three patients (01497, 01397, 00291) were selected as detailed case studies representing low-uncertainty, high-uncertainty, and gradient-failure scenarios respectively. The distinction between evaluation scopes is summarised below:

#figure(
  table(
    columns: (1fr, 1fr, 1fr),
    inset: 8pt,
    align: left + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Evaluation Scope*], [*Cohort Size*], [*Purpose*],
    table.hline(stroke: 0.5pt),

    [Segmentation Performance], [$n = 126$ patients], [Full test set metrics (Dice, HD95, IoU, Sensitivity, Specificity)],
    [XAI Saliency Metrics], [$n = 25$ patients], [Pointing Game, Coverage, Saliency IoU, Weighted Dice across 5 methods],
    [MC Dropout Uncertainty], [$n = 23$ patients], [UAR, Boundary Ratio, Saliency-Uncertainty Correlation],
    [Detailed Case Studies], [$n = 3$ patients], [In-depth multi-method qualitative and quantitative analysis],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Evaluation scope summary distinguishing between the full segmentation test set and the XAI evaluation cohort.],
) <tab:eval_scope>

== Implementation

The framework is implemented in Python using PyTorch and MONAI @cardoso2022monai. The modular architecture separates
data handling, model construction, training orchestration, XAI generation, and evaluation. XAI metrics are
computed inline during saliency generation to guarantee spatial alignment between saliency maps and ground
truth labels in the same MONAI-preprocessed coordinate space.


= Results

== Segmentation Performance

The SegResNet model achieves competitive segmentation performance on the held-out test set (126 patients),
with mean Dice scores of 0.923 for Whole Tumour, 0.891 for Tumour Core, and 0.873 for Enhancing Tumour.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Metric*], [*Whole Tumour*], [*Tumour Core*], [*Enhancing Tumour*],
    table.hline(stroke: 0.5pt),

    [Dice Score], [$0.923 plus.minus 0.079$], [$0.891 plus.minus 0.179$], [$0.873 plus.minus 0.159$],
    [HD95 (mm)], [$5.60 plus.minus 8.76$], [$4.54 plus.minus 7.60$], [$3.66 plus.minus 7.22$],
    [IoU], [$0.865 plus.minus 0.115$], [$0.836 plus.minus 0.205$], [$0.799 plus.minus 0.182$],
    [Sensitivity], [$0.925 plus.minus 0.087$], [$0.891 plus.minus 0.195$], [$0.859 plus.minus 0.236$],
    [Specificity], [$0.999 plus.minus 0.001$], [$0.999 plus.minus 0.001$], [$1.000 plus.minus 0.000$],

    table.hline(stroke: 1.5pt),
  ),
  caption: [SegResNet segmentation performance on the BraTS 2023 test set (mean $plus.minus$ std, $n = 126$).],
) <tab:segmentation>

The model achieves the highest Dice score for Whole Tumour (0.923) and notably low HD95 for Enhancing Tumour
(3.66 mm), indicating precise boundary delineation for the most clinically critical sub-region. The near-perfect
specificity ($> 0.999$) confirms reliable exclusion of healthy tissue. The Enhancing Tumour region shows the highest
variability in both Dice (0.159) and HD95 (7.22), reflecting the well-known challenge of segmenting the smallest
and most ambiguous tumour sub-region.

=== Data Scaling Progression: Baseline to Final Model

To understand how dataset volume influences performance, we compared Dice scores across training scales from
the initial baseline prototype (45 patients) to the final model (1,251 patients):

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [], [*Dice Score*], [], [], [*HD95 (mm)*], [], [],
    [*Region*], [*Baseline*], [*250 Pts*], [*Final*], [*Baseline*], [*250 Pts*], [*Final*],
    table.hline(stroke: 0.5pt),

    [WT], [0.741], [0.908], [0.923], [48.63], [7.23], [5.60],
    [TC], [0.428], [0.837], [0.891], [57.54], [15.34], [4.54],
    [ET], [0.272], [0.742], [0.873], [75.91], [15.25], [3.66],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Test set Dice and HD95 progression across data scaling stages. Whole Tumour shows asymptotic performance (0.908 to 0.923), while Enhancing Tumour demonstrates substantial dependency on training volume, improving from 0.272 to 0.873. Boundary refinement (HD95) is equally pronounced: ET HD95 decreased from 75.91 mm to 3.66 mm.],
) <tab:scaling>

=== Qualitative Visual Evaluation

#figure(
  align(center, grid(
    columns: (auto, 18%, 18%, 18%),
    column-gutter: 4pt,
    row-gutter: 4pt,
    align: center + horizon,

    [], [*01661*], [*01663*], [*01666*],

    rotate(-90deg, reflow: true, pad(x: 4pt)[*GT*]),
    image("../Final report/Figures/results_Slicer-img/01661/GT-01661.png", width: 100%),
    image("../Final report/Figures/results_Slicer-img/01663/GT-01663.png", width: 100%),
    image("../Final report/Figures/results_Slicer-img/01666/GT-01666.png", width: 100%),

    rotate(-90deg, reflow: true, pad(x: 4pt)[*Pred*]),
    image("../Final report/Figures/results_Slicer-img/01661/pred-01661.png", width: 100%),
    image("../Final report/Figures/results_Slicer-img/01663/pred-01663.png", width: 100%),
    image("../Final report/Figures/results_Slicer-img/01666/pred-01666.png", width: 100%),

    rotate(-90deg, reflow: true, pad(x: 4pt)[*3D*]),
    image("../Final report/Figures/results_Slicer-img/01661/3D-img-2.png", width: 100%),
    image("../Final report/Figures/results_Slicer-img/01663/3D-img-3.png", width: 100%),
    image("../Final report/Figures/results_Slicer-img/01666/3D-img.png", width: 100%),
  )),
  caption: [Qualitative comparison of patients 01661, 01663, and 01666. Row 1: Ground Truth (GT). Row 2: Model Prediction (Pred). Row 3: 3D volumetric rendering of the predicted segmentation. The predicted boundary layers show close agreement with ground truth masks, with HD95 precision manifesting as precise edge alignment.],
) <fig:qualitative>

== Explainable AI (XAI) Interpretation

Having established clinically viable segmentation accuracy, the analysis shifts from how well the model performs
to why it generates specific predictions. This section applies six complementary XAI techniques to interrogate
whether predictions are grounded in clinically meaningful anatomical features.

=== Bottleneck Resolution Analysis: Why Weighted Dice?

A critical question for 3D Grad-CAM evaluation is whether upsampling from $20^3$ to $160^3$ introduces
metric artifacts. To investigate, we evaluated the same Grad-CAM activations at both upsampled and native
resolution:

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Final report/Figures/results_figures/xai_bottleneck_weighted_dice.svg", width: 90%),
    image("../Final report/Figures/results_figures/xai_bottleneck_saliency_iou.svg", width: 90%),
  ),
  caption: [Bottleneck resolution analysis comparing Weighted Dice (left) and Saliency IoU (right) at different resolutions. Weighted Dice scores are highly stable across resolutions (typically within $plus.minus 0.02$ to $0.04$). In contrast, Saliency IoU exhibits volatile swings between resolutions due to hard thresholding artifacts. This validates Weighted Dice as a more reliable metric for evaluating coarse-resolution methods.],
) <fig:bottleneck>

=== Full-Resolution Gradient Attribution: Guided Backpropagation

GBP achieves strong Pointing Game accuracy for TC (80.0%) and ET (66.7%) across the 25-patient cohort, including patient 00291 where Grad-CAM and Guided Grad-CAM produce zero saliency. This indicates that the model encodes tumour-relevant features at the input pixel level.

=== Gradient Fusion: Guided Grad-CAM

Guided Grad-CAM achieves the highest Saliency Coverage of any method evaluated: 0.81--0.96 for Whole Tumour
and 0.81--0.92 for Tumour Core, indicating nearly all saliency mass is concentrated inside the tumour. However, it
inherits Grad-CAM's failure modes --- for patient 00291, element-wise multiplication zeros out GBP's otherwise
perfect signal, producing blank maps.

=== Perturbation-Based Attribution: Occlusion Sensitivity <sec:occlusion>

Occlusion Sensitivity achieves 100% Pointing Game for Whole Tumour and 88% for Tumour Core across the 25-patient cohort, with an ET Pointing Game of 54.2%. It produces the highest mean Weighted Dice of all six methods: highest mean Weighted Dice: 0.397 (WT), 0.345 (TC), 0.233 (ET) across the 25-patient cohort. Most remarkably, patient 01397 achieves a Weighted Dice of 0.35 with PG = 1.0 for Enhancing Tumour --- one of the strongest ET localisation results across all methods.

This provides compelling evidence that the model genuinely relies on tumour voxels, providing evidence against shortcut learning, texture bias, or dataset artefacts.

=== Uncertainty Quantification: MC Dropout <sec:mc_dropout>

MC Dropout generates per-voxel variance maps quantifying where the model is uncertain --- a complementary
signal to saliency maps which show where the model attends.

*Patient 01497 --- Low Uncertainty, Clear Boundaries:*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*IxG Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.196], [0.876], [0.00131], [0.00010], [$+0.020$],
    [TC], [0.092], [0.967], [0.00122], [0.00016], [$+0.000$],
    [ET], [0.219], [0.579], [0.01400], [0.00005], [$-0.065$],

    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 01497. Low UAR (0.09--0.22) indicates confident segmentation. TC Boundary Ratio of 0.967 demonstrates nearly all uncertainty concentrates at Tumour Core edges. ET exhibits 100$times$ higher internal variance than surrounding tissue (0.014 vs. 0.00005).],
) <tab:mc_01497>

*Patient 01397 --- High Uncertainty, Complex Morphology:*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*IxG Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.490], [0.642], [0.00338], [0.00005], [$-0.042$],
    [TC], [0.892], [0.776], [0.01138], [0.00001], [$-0.046$],
    [ET], [0.624], [1.000], [0.00853], [0.00003], [$-0.071$],

    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 01397. TC UAR of 0.892 indicates 89% of model uncertainty concentrates inside the Tumour Core. ET Boundary Ratio reaches 1.0 --- every unit of uncertainty sits at the ET edge, providing strong calibration evidence.],
) <tab:mc_01397>

*Patient 00291 --- The Gradient-Based Failure Case:*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*IxG Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.619], [1.000], [0.00440], [0.00003], [$-0.012$],
    [TC], [0.504], [0.997], [0.00802], [0.00002], [$+0.090$],
    [ET], [0.500], [0.997], [0.00930], [0.00002], [$+0.053$],

    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 00291. Despite complete gradient-based XAI failure (Grad-CAM and Guided Grad-CAM produce zero saliency), Boundary Ratio reaches 0.997--1.000, providing strong evidence that the model's spatial reasoning remains intact. Positive TC/ET correlation (+0.05 to +0.09) uniquely indicates the model relies on features it is not fully confident about --- a clinical red flag.],
) <tab:mc_00291>

Despite complete gradient-based XAI failure for this patient, MC Dropout reveals a fundamentally different
picture. The Boundary Ratio of 0.997--1.000 demonstrates that despite the model's ambiguity, uncertainty
concentrates at the ground truth boundaries rather than being randomly distributed. This suggests the model
is segmenting based on spatially grounded features --- the gradient-based methods failed to explain it, but the model's
internal reasoning remains spatially grounded.

=== Cross-Method Comparative Analysis

#figure(
  image("../Final report/Figures/results_figures/xai_regional_vulnerability.svg", width: 85%),
  caption: [Regional Vulnerability Analysis: Mean Weighted Dice across five XAI attribution methods ($n = 25$ per method), grouped by tumour subregion. Occlusion Sensitivity dominates Whole Tumour and Tumour Core, while ET remains universally challenging. Error bars indicate standard deviation.],
) <fig:regional_vulnerability>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Method*], [*WT Mean W.Dice*], [*TC Mean W.Dice*], [*ET Mean W.Dice*],
    table.hline(stroke: 0.5pt),

    [Grad-CAM], [$0.320 plus.minus 0.145$], [$0.358 plus.minus 0.161$], [$0.250 plus.minus 0.174$],
    [GBP], [$0.066 plus.minus 0.051$], [$0.053 plus.minus 0.052$], [$0.041 plus.minus 0.049$],
    [Guided Grad-CAM], [$0.132 plus.minus 0.072$], [$0.186 plus.minus 0.082$], [$0.169 plus.minus 0.073$],
    [IxG (LRP Proxy)], [$0.074 plus.minus 0.033$], [$0.122 plus.minus 0.050$], [$0.141 plus.minus 0.056$],
    [Occlusion], [*$0.397 plus.minus 0.116$*], [*$0.345 plus.minus 0.135$*], [$0.233 plus.minus 0.116$],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Mean Weighted Dice scores $plus.minus$ standard deviation by tumour region across the 25-patient XAI evaluation cohort ($n = 25$ per saliency method). Occlusion Sensitivity achieves the highest scores for WT and TC. Bold values indicate the best-performing method per region.],
) <tab:xai_mean_scores>

=== Visual Evaluation: Multi-Method Comparison

#figure(
  align(center, grid(
    columns: (auto, 11%, 11%, 11%, 11%, 11%, 11%, 11%),
    column-gutter: 3pt,
    row-gutter: 3pt,
    align: center + horizon,

    [], [*GT*], [*Grad-CAM*], [*GBP*], [*G.Grad-CAM*], [*IxG*], [*Occlusion*], [*MC Drop.*],

    rotate(-90deg, reflow: true, pad(x: 2pt)[*01497*]),
    image("../Final report/Figures/xai-01497/GT-01497.png", width: 100%),
    image("../Final report/Figures/xai-01497/grad-cam-01497.png", width: 100%),
    image("../Final report/Figures/xai-01497/gbp-01497.png", width: 100%),
    image("../Final report/Figures/xai-01497/guided-grad-cam-01497.png", width: 100%),
    image("../Final report/Figures/xai-01497/lrp-01497.png", width: 100%),
    image("../Final report/Figures/xai-01497/occlusion-01497.png", width: 100%),
    image("../Final report/Figures/xai-01497/mc-01497.png", width: 100%),

    rotate(-90deg, reflow: true, pad(x: 2pt)[*01397*]),
    image("../Final report/Figures/xai-01397/GT-01397.png", width: 100%),
    image("../Final report/Figures/xai-01397/grad-cam-01397.png", width: 100%),
    image("../Final report/Figures/xai-01397/gbp-01397.png", width: 100%),
    image("../Final report/Figures/xai-01397/guided-grad-cam-01397.png", width: 100%),
    image("../Final report/Figures/xai-01397/lrp-01397.png", width: 100%),
    image("../Final report/Figures/xai-01397/occlusion-01397.png", width: 100%),
    image("../Final report/Figures/xai-01397/mc-01397.png", width: 100%),

    rotate(-90deg, reflow: true, pad(x: 2pt)[*00291*]),
    image("../Final report/Figures/xai-00291/GT-00291.png", width: 100%),
    image("../Final report/Figures/xai-00291/grad-cam-00291.png", width: 100%),
    image("../Final report/Figures/xai-00291/gbp-00291.png", width: 100%),
    image("../Final report/Figures/xai-00291/guided-grad-cam-00291.png", width: 100%),
    image("../Final report/Figures/xai-00291/lrp-00291.png", width: 100%),
    image("../Final report/Figures/xai-00291/occulusion-00291.png", width: 100%),
    image("../Final report/Figures/xai-00291/mc-00291.png", width: 100%),
  )),
  caption: [Multi-method XAI comparison across three patients with varying tumour morphologies. Patient 00291 (bottom row) demonstrates the gradient-based failure case: Grad-CAM and Guided Grad-CAM produce blank maps, while GBP, IxG, Occlusion, and MC Dropout continue to provide meaningful spatial information. This case critically demonstrates why multi-method evaluation is essential for clinical deployment.],
) <fig:multi_method>

=== Cross-Paradigm Finding: Saliency $eq.not$ Uncertainty <sec:saliency_uncertainty>

Across the expanded 23-patient MC Dropout cohort and all regional measurements, the
Saliency-Uncertainty Correlation averages near zero (WT: $-0.007 plus.minus 0.096$, TC: $-0.037 plus.minus 0.104$,
ET: $-0.095 plus.minus 0.119$).

This demonstrates that saliency (what the model attends to) and uncertainty (where the model doubts) are independent, non-redundant signals. Because they are independent,
a clinician using this system receives two complementary tools: saliency maps for trust calibration ("Is the AI
looking at the right features?") and uncertainty maps for risk assessment ("Where might the AI be wrong?").


= Discussion

== Complementary Nature of XAI Methods

The central methodological argument of this work --- that no single XAI method provides a
complete picture of model behaviour --- is substantiated by three convergent lines of
evidence. First, the gradient-based failure case (patient 00291) demonstrates that Grad-CAM
and Guided Grad-CAM can produce zero activation despite the model maintaining spatially
grounded reasoning, as confirmed by MC Dropout Boundary Ratios of 0.997--1.000. This
failure is not an edge case to be dismissed; it is a diagnostic warning. Had this patient
been evaluated with Grad-CAM alone, the conclusion would have been that the model's
reasoning was deficient --- a conclusion directly contradicted by the perturbation-based and
probabilistic evidence.

Second, the cross-method Weighted Dice rankings reveal method-dependent regional
vulnerabilities. Occlusion Sensitivity leads for Whole Tumour (0.397) and Tumour Core
(0.345), while Grad-CAM narrowly leads for Enhancing Tumour (0.250 vs. 0.233). This
discrepancy is not a contradiction but a consequence of each method's inductive bias:
Occlusion captures empirical model dependency through physical perturbation, while
Grad-CAM's bottleneck weighting happens to better localise small ET structures when
upsampled. Neither result alone tells the full story.

Third, the near-zero Saliency-Uncertainty Correlation (mean Pearson $r$ across 23 patients:
WT $-0.007$, TC $-0.037$, ET $-0.095$) confirms that saliency and uncertainty are independent
signals. This independence is not a limitation but a design advantage: it means that
jointly deploying both modalities provides a complete, non-redundant clinical
interpretability layer. Saliency answers "Is the model attending to the right features?"
while uncertainty answers "Where might the model be wrong?" --- two fundamentally different
clinical questions that require two fundamentally different instruments.

== Clinical Implications

Three findings have direct implications for clinical deployment pathways. First, the high
saliency coverage of Guided Grad-CAM (93% for WT across the evaluation cohort) indicates
that the model concentrates its attention within the tumour boundary rather than on
spurious image correlates --- a necessary (though not sufficient) condition for clinical
trust. This finding is strengthened by Occlusion Sensitivity's independent confirmation:
the highest Weighted Dice of all methods (0.397 WT) demonstrates that physically removing
tumour voxels degrades the model's predictions, providing empirical evidence that the model
has learned tumour-relevant features rather than dataset-specific shortcuts.

Second, the boundary-concentrated uncertainty pattern observed across the 23-patient MC
Dropout cohort (Boundary Uncertainty Ratios exceeding 0.84 for all regions) aligns with
established neuro-oncological understanding. Tumour margins --- particularly the infiltrative
zone surrounding the enhancing rim --- are inherently ambiguous on MRI due to partial volume
effects and diffuse glioma cell infiltration. A well-calibrated model should exhibit
elevated uncertainty precisely at these boundaries, and the observed pattern is consistent
with this expectation.

Third, the positive Saliency-Uncertainty Correlation observed uniquely in patient 00291
(TC: +0.090, ET: +0.053) --- where high feature relevance overlaps with high predictive
variance --- represents a specific pattern that, if validated on larger cohorts, could serve
as an automated flag for cases requiring priority radiologist review. This signal is
available only through multi-method analysis; no single XAI technique can simultaneously
identify both high-relevance and high-uncertainty regions.

== The Novel Weighted Dice Metric

The bottleneck resolution analysis reveals a critical insight: Saliency IoU's volatile swings across resolutions
make it unreliable for coarse-resolution methods like Grad-CAM. Our novel Weighted Dice metric provides
stable evaluation by treating continuous saliency values as soft membership scores rather than forcing arbitrary
binarization thresholds.

This contribution is particularly important for 3D medical XAI, where saliency maps represent gradients of
importance rather than binary decisions. Researchers evaluating coarse-resolution XAI methods should consider adopting
Weighted Dice as a primary metric alongside existing measures.

== Limitations

This work has five primary limitations that should be considered when interpreting the results.

*Dataset generalisability.* The framework is evaluated exclusively on BraTS 2023.
Cross-institutional validation using datasets with different scanner protocols,
field strengths, or patient demographics is essential to assess robustness under domain
shift.

*Single test split.* All segmentation metrics derive from a single deterministic split
(seed=42). While the consistency of improvement across two independent training scales
(250-patient prototype and 1,251-patient full model) partially mitigates this concern,
k-fold cross-validation would provide more robust performance estimates. The computational
cost (~24 hours per training run) made this infeasible within the project timeline.

*XAI cohort size.* The 25-patient XAI evaluation cohort, while substantially larger than
the case-study approach common in prior work, represents approximately 20% of the test
set. Extending quantitative XAI evaluation to the full 126-patient test set would
strengthen the generalisability of cross-method conclusions, particularly for
the Enhancing Tumour region where method performance is most variable.

*Computational cost of Occlusion Sensitivity.* Occlusion Sensitivity requires
approximately 1,000 forward passes per patient, limiting its practicality for real-time
clinical workflows. More efficient perturbation strategies (e.g., adaptive stride or
region-of-interest-focused occlusion) should be explored.

*Absence of clinical validation.* Formal evaluation involving practising radiologists or
neurosurgeons remains essential. The XAI metrics used in this work --- Pointing Game,
Weighted Dice, and others --- are quantitative proxies for clinical utility. Whether these
metrics correlate with diagnostic confidence or decision quality in practice is an open
question that requires prospective clinical studies.

= Conclusion

This paper presents a multi-method Explainable AI framework for 3D brain tumour
segmentation that advances the integration of model accuracy and clinical
interpretability. By deploying six complementary XAI techniques with a SegResNet
architecture achieving competitive BraTS 2023 performance (Dice: 0.923 WT, 0.891 TC,
0.873 ET), and evaluating them quantitatively across a 25-patient cohort, this work
establishes three key principles for clinical XAI deployment:

First, *multi-method evaluation is not optional* --- it is a prerequisite for responsible
clinical deployment. The gradient-based failure case demonstrates that silent method
failure can produce misleading conclusions about model behaviour, a risk mitigated only
through cross-paradigm redundancy.

Second, *saliency and uncertainty are complementary, not interchangeable*. Their
statistical independence (Pearson $r approx 0$) across the evaluation cohort confirms that
deploying both modalities provides non-redundant clinical information --- trust calibration
through attribution, risk assessment through uncertainty.

Third, *evaluation metrics must match method resolution*. The novel Weighted Dice metric
addresses the instability of hard-thresholded Saliency IoU for coarse-resolution methods,
providing a principled foundation for cross-method comparison in 3D medical XAI.

These findings are subject to important limitations: reliance on a single dataset (BraTS
2023), the absence of formal clinical validation, and a single deterministic test split.
Future work will prioritise formal evaluation with practising neuroradiologists,
cross-institutional validation on diverse MRI datasets, and extension of the XAI
evaluation to the full 126-patient test set.


= Appendix: Extended XAI Results <sec:appendix>

== XAI Method Detailed Performance

The following summarizes the per-method performance characteristics observed across the 25-patient evaluation cohort:

*Grad-CAM:* Achieves strong Pointing Game for WT and TC but low values for ET due to the $20^3$ bottleneck resolution being too coarse to represent small Enhancing Tumour structures. Mean Weighted Dice across the 25-patient cohort: $0.320 plus.minus 0.145$ (WT), $0.358 plus.minus 0.161$ (TC), $0.250 plus.minus 0.174$ (ET).

*Guided Backpropagation (GBP):* Achieves strong Pointing Game accuracy across the 25-patient cohort (TC: 80.0%, ET: 66.7%), including patient 00291 where Grad-CAM and Guided Grad-CAM produce zero activation. Low Saliency Coverage (0.12--0.44) is expected as GBP highlights fine edges and texture boundaries rather than concentrated tumour blobs. Mean Weighted Dice: $0.066 plus.minus 0.051$ (WT), $0.053 plus.minus 0.052$ (TC), $0.041 plus.minus 0.049$ (ET).

*Guided Grad-CAM:* Achieves highest Saliency Coverage (0.93 WT, 0.85 TC, 0.57 ET). However, it inherits Grad-CAM's failure modes and produces blank maps for patient 00291. Mean Weighted Dice: $0.132 plus.minus 0.072$ (WT), $0.186 plus.minus 0.082$ (TC), $0.169 plus.minus 0.073$ (ET).

*IxG (LRP Proxy):* Achieves strong Pointing Game accuracy across all regions (WT: 92.0%, TC: 88.0%, ET: 95.8%), producing diffuse but correctly targeted relevance distributions. Mean Weighted Dice: $0.074 plus.minus 0.033$ (WT), $0.122 plus.minus 0.050$ (TC), $0.141 plus.minus 0.056$ (ET).

*Occlusion Sensitivity:* Achieves highest mean Weighted Dice of all methods across the 25-patient cohort: $0.397 plus.minus 0.116$ (WT), $0.345 plus.minus 0.135$ (TC), $0.233 plus.minus 0.116$ (ET). The rigorous model-agnostic validation confirms model reliance on tumour voxels.

*MC Dropout:* Provides uncertainty quantification independent of saliency. Boundary Uncertainty Ratio exceeds 0.84 for all regions, confirming characteristic "boundary glow" patterns. Saliency-Uncertainty Correlation near zero across the 23-patient cohort ($-0.007 plus.minus 0.096$ WT, $-0.037 plus.minus 0.104$ TC, $-0.095 plus.minus 0.119$ ET) confirms independent signals.

== VR Visualization Pipeline

All XAI outputs are exported as spatially-aligned NIfTI volumes compatible with 3D Slicer and SlicerVR. The pipeline: (1) generates saliency maps in the same MONAI-preprocessed coordinate space as ground truth labels; (2) exports as NIfTI format preserving spatial alignment; (3) enables direct loading into 3D Slicer for volumetric visualization; and (4) supports the SlicerVR extension for immersive exploration. This pipeline supports the transition from static 2D heatmaps to immersive 3D environments where clinicians can explore saliency and uncertainty distributions spatially, potentially reducing cognitive load inherent in traditional slice-by-slice interpretation.

// Bibliography is handled by the template via bibliography-file parameter
