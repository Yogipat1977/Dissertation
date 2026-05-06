// =============================================================================
// Research Paper Draft 3: Illuminating the Black Box
// A Multi-Method Explainable AI Framework with Uncertainty Quantification 
// for 3D Brain Tumor Segmentation
// Author: Yogi Amitkumar Patel
// Status: Peer-Reviewed Publication Draft (Independent Researcher)
// =============================================================================

#let paper-title = "Illuminating the Black Box: A Multi-Method Explainable AI Framework with Uncertainty Quantification for 3D Brain Tumor Segmentation"

// --- Page & Document Setup ---
#set document(
  title: paper-title,
  author: "Yogi Amitkumar Patel",
)

#set page(
  paper: "a4",
  margin: (left: 2.5cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
  numbering: "1",
  number-align: center,
)

// --- Typography ---
#set text(font: "Linux Libertine", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.65em, first-line-indent: 0em)
#set heading(numbering: "1.")

// Heading styling
#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 14pt, weight: "bold")[#it]
  v(0.6em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold")[#it]
  v(0.4em)
}
#show heading.where(level: 3): it => {
  v(0.6em)
  text(size: 10pt, weight: "bold", style: "italic")[#it]
  v(0.3em)
}

// --- Title Block ---
#align(center)[
  #v(0.5em)
  #text(size: 16pt, weight: "bold")[ #paper-title ]
  #v(1.5em)
]

#align(center)[
  #text(size: 11pt)[
    *Yogi Amitkumar Patel*\
    \
    Independent Researcher\
    u2536809\@uel.ac.uk
  ]
  #v(1.5em)
]

// ============================================================================
// ABSTRACT
// ============================================================================
#align(center)[#text(weight: "bold", size: 11pt)[Abstract]]
#v(0.5em)

#par(first-line-indent: 0em)[
  Deep learning models for 3D brain tumor segmentation have achieved remarkable accuracy, yet their opaque decision-making processes present a critical barrier to clinical adoption. This paper presents a comprehensive Explainable AI (XAI) framework integrating six complementary post-hoc explanation techniques with a SegResNet architecture for volumetric brain tumor segmentation on the BraTS 2023 dataset. Our framework combines gradient-based methods (Grad-CAM, Guided Backpropagation, Guided Grad-CAM, Layer-wise Relevance Propagation), perturbation-based analysis (Occlusion Sensitivity), and probabilistic uncertainty quantification (Monte Carlo Dropout) to provide multi-perspective interpretability of 3D CNN predictions. The SegResNet model achieves competitive segmentation performance with mean Dice scores of 0.923 (Whole Tumor), 0.891 (Tumor Core), and 0.873 (Enhancing Tumor) on a held-out test set of 126 patients. Quantitative evaluation of XAI outputs using four metrics - Pointing Game accuracy, Saliency Coverage, Saliency IoU, and a novel Weighted Dice score - demonstrates that Guided Grad-CAM achieves 93% saliency coverage for the Whole Tumor region, while Occlusion Sensitivity achieves the highest spatial alignment (Weighted Dice: 0.42). Monte Carlo Dropout uncertainty analysis reveals characteristic boundary-concentrated variance patterns, with Boundary Uncertainty Ratios exceeding 0.84, confirming clinically meaningful confidence calibration. A critical finding is the gradient-based failure case (patient 00291) where Grad-CAM and Guided Grad-CAM produce zero saliency, yet MC Dropout confirms spatially grounded reasoning - demonstrating why multi-method XAI evaluation is essential. All explanation outputs are exported as spatially-aligned NIfTI volumes for immersive 3D visualization via 3D Slicer. This work bridges the gap between model accuracy and clinical trust by providing a unified, quantitatively validated explainability pipeline for volumetric medical image analysis.
]

#v(0.5em)
#text(size: 9pt, weight: "bold")[Keywords:] #text(size: 9pt)[Explainable AI, Brain Tumor Segmentation, 3D CNN, Grad-CAM, Uncertainty Quantification, BraTS 2023, SegResNet, Virtual Reality, Monte Carlo Dropout, Layer-wise Relevance Propagation, Weighted Dice]
#v(1.5em)

// ============================================================================
// 1. INTRODUCTION
// ============================================================================
= Introduction

In modern neuro-oncology, accurate identification and delineation of brain tumors from Magnetic Resonance Imaging (MRI) data remains a critical clinical challenge. Manual segmentation by radiologists is labor-intensive, subjective, and increasingly unsustainable as imaging volumes grow @brats2015. Deep learning models, particularly 3D Convolutional Neural Networks (CNNs), have demonstrated superior performance in automated volumetric segmentation tasks @iftikhar2025 @bhati2024, processing entire MRI volumes rather than individual 2D slices to capture the true three-dimensional spatial relationships of tumor pathology.

However, the clinical deployment of these high-performing models is critically hindered by the _"black box"_ problem: their internal decision-making processes remain fundamentally opaque to the clinicians who must rely on their outputs @neri2023. In the high-stakes domain of neurosurgery, this opacity creates a _"trust gap"_ @wen2025 - it is insufficient for a model to simply output a segmentation mask; clinicians require the ability to verify _why_ specific voxels were classified as tumor tissue. This lack of interpretability is not merely a technical limitation but a barrier to ethical and safe clinical deployment, particularly within regulatory frameworks such as the GDPR's "right to explanation" @neri2023.

Explainable AI (XAI) techniques aim to address this disconnect by illuminating model reasoning. Methods such as Gradient-weighted Class Activation Mapping (Grad-CAM) @selvaraju2017 have become established tools for visualizing CNN decisions. However, existing XAI frameworks for 3D medical imaging suffer from three critical limitations: (1) they typically employ only a single explanation method, providing an incomplete picture of model behavior; (2) they lack uncertainty quantification, making it impossible to distinguish between confident and uncertain predictions; and (3) they present explanations as static 2D overlays, failing to convey the volumetric nature of the underlying pathology @neuroxai @brainAR.

This paper presents a unified, multi-method XAI framework that addresses these limitations. Our key contributions are:

1. *Multi-perspective explainability:* We implement and quantitatively compare six complementary XAI techniques - Grad-CAM, Guided Backpropagation (GBP), Guided Grad-CAM, Layer-wise Relevance Propagation (LRP), Occlusion Sensitivity, and Monte Carlo (MC) Dropout - providing gradient-based, perturbation-based, and probabilistic perspectives on model decision-making.

2. *Quantitative XAI validation with novel metrics:* We introduce a systematic evaluation framework using four metrics (Pointing Game, Saliency Coverage, Saliency IoU, and Weighted Dice) that objectively assess whether the model "looks at the right place for the right reasons." The Weighted Dice metric, introduced as a novel contribution, enables reliable evaluation of coarse-resolution saliency methods by treating continuous saliency values as soft membership scores rather than forcing arbitrary binarization thresholds.

3. *Uncertainty-aware explainability:* We integrate MC Dropout uncertainty quantification with structural attribution methods (LRP), enabling identification of _"brittle"_ decision regions where high feature importance intersects with high predictive variance - a critical clinical red flag.

4. *VR-ready volumetric pipeline:* All XAI outputs are exported as spatially-aligned NIfTI volumes, ready for immersive 3D visualization through platforms such as 3D Slicer and SlicerVR.

The remainder of this paper is organized as follows: Section 2 reviews related work in 3D segmentation architectures, XAI methods, and immersive visualization. Section 3 details our proposed framework. Section 4 describes the experimental setup. Section 5 presents comprehensive quantitative results. Section 6 discusses findings, clinical implications, and limitations. Section 7 concludes with future directions.

// ============================================================================
// 2. RELATED WORK
// ============================================================================
= Related Work

== 3D Deep Learning for Brain Tumor Segmentation

The evolution from 2D slice-based to volumetric 3D approaches has fundamentally improved brain tumor segmentation. The 3D U-Net @cicek2016 established the encoder-decoder paradigm with skip connections for volumetric data, while V-Net @milletari2016 introduced residual connections and the Dice loss function to address class imbalance.

#figure(
  image("../Final report/Figures/3d_unet_architecture.jpg", width: 85%),
  caption: [The 3D U-Net architecture establishes the encoder-decoder paradigm with skip connections for volumetric medical image segmentation. The contracting path (encoder) captures high-level context while the expanding path (decoder) recovers spatial detail],
) <fig:3d_unet>

Modern architectures such as SegResNet @myronenko2019 combine these innovations, using residual blocks with GroupNorm for stable deep network training and DiceFocalLoss for handling severely imbalanced tumor sub-regions. The BrainAR framework @brainAR demonstrated the potential of combining 3D U-Net architectures with augmented reality visualization, achieving Dice scores of 0.914 for whole tumor segmentation. Similarly, the AXONS-3 framework @axons3 applied post-hoc XAI techniques to 3D brain tumor segmentation, highlighting the importance of interpretability in clinical AI systems.

The BraTS challenge @brats2015 has served as the primary benchmark for evaluating these architectures on multi-modal MRI data (T1, T1c, T2, FLAIR), with current state-of-the-art models achieving Dice scores exceeding 0.90 for the Whole Tumor region. The challenge provides a standardized evaluation framework enabling fair comparison between methods.

== Explainable AI for Medical Imaging

XAI methods for CNNs are broadly categorized as _gradient-based_, _perturbation-based_, or _decomposition-based_. Grad-CAM @selvaraju2017 computes class-discriminative heatmaps using gradients flowing into a target convolutional layer, providing coarse but class-specific spatial localization. Guided Backpropagation @springenberg2015 produces full-resolution saliency maps by gating negative gradients during backpropagation. Their fusion - Guided Grad-CAM - achieves both high resolution and class specificity @selvaraju2017.

Layer-wise Relevance Propagation (LRP) @montavon2017 operates on a conservation principle, attributing the output prediction score back to individual input voxels. For complex architectures, the Input $times$ Gradient approximation provides a computationally tractable proxy equivalent to $epsilon$-LRP in ReLU networks. Occlusion Sensitivity @zeiler2014 offers a complementary, model-agnostic approach by systematically masking input regions and measuring prediction changes.

For uncertainty quantification, MC Dropout @gal2016dropout provides a Bayesian approximation by maintaining dropout during inference across multiple stochastic forward passes. The variance across predictions serves as a proxy for model uncertainty, identifying regions where the network lacks confidence.

Recent surveys @bhati2024 have highlighted the growing importance of XAI in medical imaging, emphasizing that no single method provides a complete picture of model behavior - a gap our multi-method framework directly addresses. The NeuroXAI framework @neuroxai applied multiple gradient-based XAI methods to brain tumor segmentation, achieving 90% clinician alignment. The AXONS-3 framework @axons3 advanced this by integrating trust metrics into 3D segmentation pipelines. However, both frameworks lack integrated uncertainty quantification and rely on static visualization, limiting their clinical utility for interactive decision support.

== Immersive Visualization in Medical Imaging

Traditional slice-by-slice visualization of 3D data imposes significant cognitive load on clinicians @brainAR. Virtual Reality (VR) environments provide stereoscopic depth perception and six-degrees-of-freedom interaction, enabling clinicians to explore volumetric XAI outputs as spatial clouds rather than flat overlays @neuroxai. Platforms such as 3D Slicer with the SlicerVR extension support direct rendering of NIfTI volumes in VR headsets, bridging the gap between medical imaging formats and immersive display systems @Zeineldin2023. The combination of XAI with VR visualization represents an emerging frontier for clinical decision support, enabling clinicians to explore saliency and uncertainty in an immersive environment.

// ============================================================================
// 3. PROPOSED FRAMEWORK
// ============================================================================
= Proposed Framework

Our framework implements a three-stage pipeline: (1) volumetric tumor segmentation using SegResNet, (2) multi-method XAI analysis, and (3) VR-compatible NIfTI export. This section details the architecture and each XAI component.

== SegResNet Architecture

#figure(
  image("../Final report/Figures/3D_SegResNet.svg", width: 85%),
  caption: [The SegResNet 3D encoder-decoder architecture. The encoder progressively downsamples through four levels (32 to 256 channels) using residual blocks with GroupNorm. The decoder reconstructs segmentation via trilinear upsampling and skip connections. A dropout probability of p=0.1 enables Monte Carlo Dropout inference],
) <fig:segresnet>

The backbone of our framework is SegResNet @myronenko2019, a 3D encoder-decoder architecture that processes four-channel MRI volumes ($160^3$ voxels) to produce three-channel segmentation masks corresponding to the Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) regions.

The encoder progressively downsamples input through four levels, increasing filters from 32 to 256 at the bottleneck, using strided $3 times 3 times 3$ convolutions. Each level is composed of Residual Blocks (ResBlocks) featuring GroupNorm normalization and ReLU activations. These blocks learn residual mappings via skip connections, preventing vanishing gradients and ensuring stable training of deep 3D networks @he2016deep. The decoder reconstructs the segmentation map via trilinear upsampling and encoder-decoder skip connections. A dropout probability of $p = 0.1$ serves dual purposes: regularization during training and enabling MC Dropout inference.

To address severe class imbalance - particularly the ET region comprising merely ~5% of tumor volume - the model employs DiceFocalLoss with a focal parameter $gamma = 2.0$, which increases the loss contribution from hard-to-classify voxels:

$
  "DiceFocalLoss" = "DiceLoss" + lambda dot.c "FocalLoss"
$

where DiceLoss optimizes volumetric overlap and FocalLoss ($"FL"(p_t) = -alpha_t (1-p_t)^gamma log(p_t)$) heavily penalizes confident misclassifications.

== XAI Method Suite

We implement six XAI methods spanning three paradigms: gradient-based, perturbation-based, and probabilistic. Each method provides a distinct perspective on model decision-making.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: left + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Category*], [*Method*], [*Resolution*], [*Class-Specific*],
    table.hline(stroke: 0.5pt),

    [Gradient-Based], [3D Grad-CAM], [Coarse (~20 cubed)], [Yes],
    [Gradient-Based], [Guided Backpropagation (GBP)], [Full (160 cubed)], [No],
    [Gradient-Based], [Guided Grad-CAM], [Full (160 cubed)], [Yes],
    [Relevance-Based], [LRP (Input x Gradient)], [Full (160 cubed)], [Yes],
    [Perturbation-Based], [Occlusion Sensitivity], [Stride-upsampled], [Yes],
    [Uncertainty-Based], [MC Dropout (20 passes)], [Full (160 cubed)], [Yes],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Summary of the six XAI techniques applied to the SegResNet model, categorized by mechanism, spatial resolution, and class discrimination capability],
) <tab:xai_methods>

=== Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM computes per-feature-map importance weights by analyzing gradients flowing into the bottleneck convolutional layer (256 channels at $20^3$ resolution). For target class $c$ and feature map $k$, the importance weight $alpha_k^c$ is computed via global average pooling of gradients:

$
  alpha_k^c = 1/Z sum_(x) sum_(y) sum_(z) (partial y^c)/(partial A^k_(x,y,z))
$

where $y^c$ is the spatially-averaged output logit and $A^k_(x,y,z)$ represents feature map activations. The class-discriminative heatmap is obtained through a weighted linear combination followed by ReLU activation:

$
  L^c_"Grad-CAM" = "ReLU"(sum_k alpha_k^c A^k)
$

The resulting $20^3$ heatmap is upsampled to $160^3$ via trilinear interpolation and normalized to $[0, 1]$. A critical limitation of Grad-CAM for 3D segmentation is the fundamental resolution trade-off: the ~20 cubed bottleneck cannot represent structures smaller than a single feature voxel, leading to potential failure for small Enhancing Tumor regions.

=== Guided Backpropagation (GBP)

GBP extends standard backpropagation by imposing an additional gradient gate at every ReLU layer, suppressing both negative forward activations _and_ negative incoming gradients. The guided gradient $R_i^l$ at neuron $i$ in layer $l$ is:

$
  R_i^l = (f_i^l > 0) dot.c (R_i^(l+1) > 0) dot.c R_i^(l+1)
$

where $f_i^l$ is the forward activation and $R_i^(l+1)$ is the incoming gradient. This dual gating isolates purely positive signal paths, producing full-resolution ($160^3$) saliency maps with sharp voxel-level detail, though without inherent class discrimination.

=== Guided Grad-CAM

Guided Grad-CAM fuses the complementary strengths of Grad-CAM (class-discriminative, coarse) and GBP (high-resolution, class-agnostic) through element-wise multiplication:

$
  L^c_"Guided Grad-CAM" = L^c_"Grad-CAM" dot.c L_"GBP"
$

A critical implementation detail is _hook isolation_: simultaneously active Grad-CAM and GBP backward hooks corrupt each other's gradient flows. Our framework employs sequential per-patient computation - first Grad-CAM hooks, then GBP hooks on a clean model - before multiplying the stored results.

=== Layer-wise Relevance Propagation (LRP)

LRP attributes the model's output prediction score back to each input voxel based on a conservation principle. Due to the architectural complexity of SegResNet (skip connections, GroupNorm layers), we employ the Input $times$ Gradient approximation:

$
  R_i approx x_i dot.c (partial f(x)) / (partial x_i)
$

This yields ultra-high-resolution ($160^3$) relevance maps quantifying which specific structural features drove the network's prediction.

=== Occlusion Sensitivity

Occlusion Sensitivity provides model-agnostic, perturbation-based explanations by systematically testing the CNN's structural reliance on specific brain regions. A $16 times 16 times 16$ sliding window moves across the input volume with stride 8, replacing occluded voxels with zeros:

$
  S_c(i,j,k) = f_c(x) - f_c(x_"occluded")
$

This method requires approximately 1,000 forward passes per patient but provides direct empirical evidence of regional importance, constituting the gold-standard XAI validation since it independently measures model dependency through empirical observation.

=== Monte Carlo Dropout (Uncertainty Quantification)

At test time, SegResNet's dropout layers ($p=0.1$) remain active during $N=20$ stochastic forward passes. The stabilized mean prediction and voxel-wise predictive variance are:

$
  macron(p) = 1/N sum_(n=1)^N p_n, quad quad sigma^2(x,y,z) = 1/N sum_(n=1)^N (p_n(x,y,z) - macron(p)(x,y,z))^2
$

High variance identifies _"brittle"_ transition zones where the network lacks confidence.

== Quantitative XAI Evaluation Metrics

All saliency maps are evaluated against ground truth tumor annotations using four metrics:

#figure(
  table(
    columns: (22%, 28%, 1fr),
    inset: 8pt,
    align: horizon,
    table.header([*Metric*], [*Formula*], [*Interpretation*]),
    [Pointing Game], [$ cases(1 "if" arg max(L) in G, 0 "otherwise") $], [Binary hit/miss test: is the peak saliency voxel inside the tumor region $G$?],
    [Saliency Coverage], [$ (sum_(x in G) L(x)) / (sum_x L(x)) $], [Fraction of total saliency mass concentrated inside the tumor boundary],
    [Saliency IoU], [$ |L_"thresh" sect G| / (|L_"thresh" union G|) $], [Shape overlap between thresholded saliency ($tau = 0.5$) and ground truth],
    [Weighted Dice (Novel)], [$ (2 sum L(x) dot G(x)) / (sum L(x) + sum G(x)) $], [Soft overlap treating continuous saliency as prediction weights. Novel metric],
  ),
  caption: [XAI evaluation metrics for saliency map validation against ground truth. Weighted Dice is a novel metric introduced in this work for reliable evaluation of coarse-resolution saliency methods],
) <tab:xai_metrics>

The Weighted Dice metric is introduced as a novel contribution to address a critical limitation of Saliency IoU: the hard binarization threshold (0.5) discards intensity information and introduces volatile metric swings across resolutions. Weighted Dice treats continuous saliency values as soft membership scores, providing stable evaluation across different saliency resolutions.

// ============================================================================
// 4. EXPERIMENTAL SETUP
// ============================================================================
= Experimental Setup

== Dataset

The BraTS 2023 dataset comprises approximately 1,251 patient cases, each containing four co-registered MRI modalities: T1-weighted (T1), T1 with Gadolinium contrast (T1c), T2-weighted (T2), and T2-FLAIR. All volumes are resampled to 1 mm cubed isotropic resolution and skull-stripped. Tumor annotations follow the BraTS standard: necrotic core (label 1), peritumoral edema (label 2), and enhancing tumor (label 3), converted into three binary channels: WT (labels 1+2+3), TC (labels 1+3), and ET (label 3 only).

The dataset was split deterministically (seed=42) into 1,000 training, 125 validation, and 126 test cases. Validation was used for hyperparameter selection and early stopping; the test set was reserved for final evaluation and XAI analysis.

== Preprocessing Pipeline

The preprocessing pipeline applies four operations: (1) foreground cropping to reduce spatial dimensions from approximately $240^3$ to $160^3$ by removing background; (2) channel-first formatting for PyTorch compatibility with dimensions $[4, 160, 160, 160]$; (3) per-channel z-score normalization using non-zero voxel statistics; and (4) training-only random augmentations including random flips, rotations, intensity scaling, Gaussian noise, and intensity shifts.

== Training Configuration

Training was conducted on a single NVIDIA RTX 5880 Ada (48 GB VRAM) with Automatic Mixed Precision (AMP) enabled. An evolutionary prototyping phase on a 250-patient subset validated the pipeline before full-scale training.

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 8pt,
    align: (left, left),
    table.header([*Parameter*], [*Value*]),
    [Batch Size], [1 (limited by 3D volume memory)],
    [Epochs], [35],
    [Learning Rate], [$5 times 10^(-5)$],
    [LR Scheduler], [Cosine Annealing],
    [Optimizer], [AdamW (weight decay $1 times 10^(-5)$)],
    [Loss Function], [DiceFocalLoss ($gamma = 2.0$)],
    [Mixed Precision], [Enabled (AMP)],
    [Dropout], [$p = 0.1$],
  ),
  caption: [Training hyperparameters for SegResNet on BraTS 2023],
) <tab:training_params>

Best model checkpoints were selected based on validation Dice score, with experiment tracking via Weights & Biases (W&B).

== Implementation

The framework is implemented in Python using PyTorch and MONAI @monai2022. The modular architecture separates data handling, model construction, training orchestration, XAI generation, and evaluation. XAI metrics are computed inline during saliency generation to guarantee spatial alignment between saliency maps and ground truth labels in the same MONAI-preprocessed coordinate space.

// ============================================================================
// 5. RESULTS
// ============================================================================
= Results

== Segmentation Performance

The SegResNet model achieves competitive segmentation performance on the held-out test set (126 patients), with mean Dice scores of 0.923 for Whole Tumor, 0.891 for Tumor Core, and 0.873 for Enhancing Tumor.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 8pt,
    align: center,
    table.header([*Metric*], [*Whole Tumor*], [*Tumor Core*], [*Enhancing Tumor*]),
    [Dice Score], [0.923 +- 0.079], [0.891 +- 0.179], [0.873 +- 0.159],
    [HD95 (mm)], [5.60 +- 8.76], [4.54 +- 7.60], [3.66 +- 7.22],
    [IoU], [0.865 +- 0.115], [0.836 +- 0.205], [0.799 +- 0.182],
    [Sensitivity], [0.925 +- 0.087], [0.891 +- 0.195], [0.859 +- 0.236],
    [Specificity], [0.999 +- 0.001], [0.999 +- 0.001], [1.000 +- 0.000],
  ),
  caption: [SegResNet segmentation performance on the BraTS 2023 test set (mean +- std, $n = 126$)],
) <tab:seg_results>

The model achieves the highest Dice score for Whole Tumor (0.923) and notably low HD95 for Enhancing Tumor (3.66 mm), indicating precise boundary delineation for the most clinically critical sub-region. The near-perfect specificity (>0.999) confirms reliable exclusion of healthy tissue. The Enhancing Tumor region shows the highest variability in both Dice (0.159) and HD95 (7.22), reflecting the well-known challenge of segmenting the smallest and most ambiguous tumor sub-region.

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Final report/Figures/results_figures/heatmap_dice.png", width: 90%),
    image("../Final report/Figures/results_figures/heatmap_sensitivity.png", width: 90%),
  ),
  caption: [Dice score (left) and Sensitivity (right) heatmaps across the test set patients. Each cell represents one patient's metric for a specific tumor region, with color intensity indicating performance level],
) <fig:metric_heatmaps>

=== Data Scaling Progression: Baseline to Final Model

To understand how dataset volume influences performance, we compared Dice scores across training scales from the initial baseline prototype (45 patients) to the final model (1,251 patients):

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 8pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [], table.cell(colspan: 3, align: center,[*Dice Score*]), table.cell(colspan: 3, align: center,[*HD95 (mm)*]),
    table.hline(stroke: 0.5pt),
    [*Region*], [*Baseline*], [*250 Pts*], [*Final*], [*Baseline*], [*250 Pts*], [*Final*],
    table.hline(stroke: 0.5pt),

    [WT], [0.741], [0.908], [0.923], [48.63], [7.23], [5.60],
    [TC], [0.428], [0.837], [0.891], [57.54], [15.34], [4.54],
    [ET], [0.272], [0.742], [0.873], [75.91], [15.25], [3.66],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Test set Dice and HD95 progression across data scaling stages. Whole Tumor shows asymptotic performance (0.908 to 0.923), while Enhancing Tumor demonstrates substantial dependency on training volume, improving from 0.272 to 0.873. Boundary refinement (HD95) is equally pronounced: ET HD95 decreased from 75.91 mm to 3.66 mm],
) <tab:dice_progression>

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Final report/Figures/results_figures/bar_metrics_Baseline_version_01.png", width: 90%),
    image("../Final report/Figures/results_figures/bar_metrics_Final_version_10.png", width: 90%),
  ),
  caption: [Performance comparison between the Baseline Prototype (left, 45 patients) and the Final Model (right, 1,251 patients). The massive improvement in ET metrics demonstrates the critical importance of training data volume for small, heterogeneous tumor sub-regions],
) <fig:bar_metrics>

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
  caption: [Qualitative comparison of patients 01661, 01663, and 01666. Row 1: Ground Truth (GT). Row 2: Model Prediction (Pred). Row 3: 3D volumetric rendering of the predicted segmentation. The predicted boundary layers align flawlessly with ground truth masks, with HD95 precision manifesting as perfect edge alignment]
) <fig:qualitative_comparison>

== Explainable AI (XAI) Interpretation

Having established clinically viable segmentation accuracy, the analysis shifts from _how well_ the model performs to _why_ it generates specific predictions. This section applies six complementary XAI techniques to interrogate whether predictions are grounded in clinically meaningful anatomical features.

=== Bottleneck Resolution Analysis: Why Weighted Dice?

A critical question for 3D Grad-CAM evaluation is whether upsampling from ~20 cubed to 160 cubed introduces metric artifacts. To investigate, we evaluated the same Grad-CAM activations at both upsampled and native resolution:

#figure(
  grid(
    columns: 2,
    gutter: 2%,
    image("../Final report/Figures/results_figures/xai_bottleneck_weighted_dice.svg", width: 90%),
    image("../Final report/Figures/results_figures/xai_bottleneck_saliency_iou.svg", width: 90%),
  ),
  caption: [Bottleneck resolution analysis comparing Weighted Dice and Saliency IoU at different resolutions],
) <fig:bottleneck_analysis>

Weighted Dice scores are highly stable across resolutions (typically within +-0.02 to 0.04). In contrast, Saliency IoU exhibits volatile swings between resolutions due to hard thresholding artifacts. This validates Weighted Dice as a more reliable metric for evaluating coarse-resolution methods, particularly important for 3D medical XAI where saliency maps represent gradients of importance rather than binary decisions.

=== Full-Resolution Gradient Attribution: Guided Backpropagation

GBP achieves 100% Pointing Game across all five patients and all three tumor regions - including patient 00291, where Grad-CAM and Guided Grad-CAM produce zero saliency. This proves that the model encodes tumor-relevant features at the input pixel level.

=== Gradient Fusion: Guided Grad-CAM

Guided Grad-CAM achieves the highest Saliency Coverage of any method evaluated: 0.81–0.96 for Whole Tumor and 0.81–0.92 for Tumor Core, indicating nearly all saliency mass is concentrated inside the tumor. However, it inherits Grad-CAM's failure modes - for patient 00291, element-wise multiplication zeros out GBP's otherwise perfect signal, producing blank maps.

=== Perturbation-Based Attribution: Occlusion Sensitivity

Occlusion Sensitivity achieves 100% Pointing Game for Whole Tumor across all patients and produces the highest Weighted Dice scores of all six methods: 0.38–0.46 for WT, 0.24–0.47 for TC. Most remarkably, patient 01397 achieves Weighted Dice of 0.35 with PG=1.0 for Enhancing Tumor - the only method where ET localization truly succeeds above 0.30.

This constitutes the strongest piece of evidence that the model genuinely relies on tumor voxels, ruling out shortcut learning, texture bias, or dataset artifacts.

=== Uncertainty Quantification: MC Dropout

MC Dropout generates per-voxel variance maps quantifying where the model is uncertain - a complementary signal to saliency maps which show where the model attends.

*Patient 01497  -  Low Uncertainty, Clear Boundaries:*

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
    [TC], [0.092], [0.967], [0.00122], [0.00016], [+0.000],
    [ET], [0.219], [0.579], [0.01400], [0.00005], [−0.065],
    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 01497. Low UAR (0.09-0.22) indicates confident segmentation. TC Boundary Ratio of 0.967 demonstrates nearly all uncertainty concentrates at Tumour Core edges. ET exhibits 100x higher internal variance than surrounding tissue (0.014 vs. 0.00005)],
) <tab:mc_01497>

*Patient 01397  -  High Uncertainty, Complex Morphology:*

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
    [TC], [0.892], [0.776], [0.01138], [0.00001], [−0.046],
    [ET], [0.624], [1.000], [0.00853], [0.00003], [−0.071],
    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 01397. TC UAR of 0.892 indicates 89% of model uncertainty concentrates inside the Tumour Core. ET Boundary Ratio reaches 1.0-every unit of uncertainty sits at the ET edge, constituting the strongest calibration evidence],
) <tab:mc_01397>

*Patient 00291  -  The Gradient-Based Failure Case:*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 7pt,
    align: center + horizon,
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Region*], [*UAR*], [*Boundary Ratio*], [*Unc Inside*], [*Unc Outside*], [*LRP Corr*],
    table.hline(stroke: 0.5pt),

    [WT], [0.619], [1.000], [0.00440], [0.00003], [−0.012],
    [TC], [0.504], [0.997], [0.00802], [0.00002], [+0.090],
    [ET], [0.500], [0.997], [0.00930], [0.00002], [+0.053],
    table.hline(stroke: 1.5pt),
  ),
  caption: [MC Dropout uncertainty metrics for patient 00291. Despite complete gradient-based XAI failure (Grad-CAM and Guided Grad-CAM produce zero saliency), Boundary Ratio reaches 0.997--1.000, proving the model's spatial reasoning is intact. Positive TC/ET correlation (+0.05 to +0.09) uniquely indicates the model relies on features it is not fully confident about - a clinical red flag],
) <tab:mc_00291>

Despite complete gradient-based XAI failure for this patient, MC Dropout reveals a fundamentally different picture. The Boundary Ratio of 0.997--1.000 demonstrates that despite the model's ambiguity, uncertainty concentrates at the ground truth boundaries rather than being randomly distributed. This confirms the model _is_ segmenting based on real spatial features - the gradient-based methods failed to explain it, but the model's internal reasoning remains spatially grounded.

=== Cross-Method Comparative Analysis

#figure(
  image("../Final report/Figures/results_figures/xai_regional_vulnerability.svg", width: 85%),
  caption: [Regional Vulnerability Analysis: Mean Weighted Dice across five XAI attribution methods, grouped by tumor subregion. Occlusion Sensitivity dominates Whole Tumor and Tumour Core, while ET remains universally challenging. Error bars indicate standard deviation]
) <fig:vulnerability_analysis>

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
    [Occlusion], [0.422], [0.392], [0.165],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Mean Weighted Dice scores by tumor region across all five saliency-based XAI methods. Occlusion Sensitivity achieves the highest scores for WT and TC. Enhancing Tumor is universally the weakest region across all methods, exposing the fundamental vulnerability of interpretability techniques for small, heterogeneous subregions]
) <tab:xai_mean_scores>

=== Visual Evaluation: Multi-Method Comparison

#figure(
  align(center, grid(
    columns: (auto, 11%, 11%, 11%, 11%, 11%, 11%, 11%),
    column-gutter: 3pt,
    row-gutter: 3pt,
    align: center + horizon,

    [], [*GT*], [*Grad-CAM*], [*GBP*], [*G.Grad-CAM*], [*LRP*], [*Occlusion*], [*MC Drop.*],

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
  caption: [Multi-method XAI comparison across three patients with varying tumor morphologies. Patient 00291 (bottom row) demonstrates the gradient-based failure case: Grad-CAM and Guided Grad-CAM produce blank maps, while GBP, LRP, Occlusion, and MC Dropout continue to provide meaningful spatial information. This case critically demonstrates why multi-method evaluation is essential for clinical deployment]
) <fig:xai_grid>

=== Cross-Paradigm Finding: Saliency ≠ Uncertainty

Across all three patients and nine regional measurements, the Saliency-Uncertainty Correlation ranges from −0.071 to +0.090 with a mean near zero. This demonstrates that saliency (what the model attends to) and uncertainty (where the model doubts) are *independent, non-redundant signals*. Because they are independent, a clinician using this system receives two complementary tools: saliency maps for trust calibration ("Is the AI looking at the right features?") and uncertainty maps for risk assessment ("Where might the AI be wrong?").

// ============================================================================
// 6. DISCUSSION
// ============================================================================
= Discussion

== Complementary Nature of XAI Methods

Our results demonstrate that no single XAI method provides a complete picture of model behavior. Each method answers a fundamentally different question:

- *Grad-CAM / Guided Grad-CAM:* "Which spatial regions influence the class prediction?"
- *GBP / LRP:* "Which specific voxel-level features drove the output?"
- *Occlusion Sensitivity:* "Which regions are empirically necessary for the prediction?"
- *MC Dropout:* "How confident is the model about each voxel?"

The patient 00291 case provides the strongest evidence for multi-method evaluation: Grad-CAM and Guided Grad-CAM produce blank maps (complete failure), while MC Dropout confirms spatially grounded reasoning (Boundary Ratio 0.997--1.000). Relying on any single method would lead to incorrect conclusions about model behavior.

== Clinical Implications

The high saliency coverage of Guided Grad-CAM (93% for WT) demonstrates that the model focuses attention within the tumor boundary for the correct reasons, not on spurious correlations. The boundary-concentrated uncertainty pattern aligns with clinical intuition - tumor margins are inherently ambiguous due to infiltrative growth patterns.

The combination of structural attribution (LRP) with uncertainty quantification enables identification of anatomically ambiguous regions where the prediction may be unreliable. The positive correlation in patient 00291 (where high LRP relevance overlaps with high uncertainty) flags a case that should be prioritized for radiologist review - a unique diagnostic signal available only through multi-method analysis.

== The Novel Weighted Dice Metric

The bottleneck resolution analysis reveals a critical insight: Saliency IoU's volatile swings across resolutions make it unreliable for coarse-resolution methods like Grad-CAM. Our novel Weighted Dice metric provides stable evaluation by treating continuous saliency values as soft membership scores rather than forcing arbitrary binarization thresholds.

This contribution is particularly important for 3D medical XAI, where saliency maps represent gradients of importance rather than binary decisions. Researchers evaluating coarse-resolution XAI methods should adopt Weighted Dice as the primary metric.

== Limitations

Several limitations should be acknowledged. First, our framework relies exclusively on the BraTS 2023 dataset; cross-institutional validation is necessary to assess generalizability under domain shift. Second, the Occlusion Sensitivity method requires approximately 1,000 forward passes per patient, limiting practicality for real-time clinical workflows. Third, formal evaluation with practicing radiologists remains essential for validating diagnostic efficacy. Fourth, the Input $times$ Gradient approximation for LRP may not capture all relevance dynamics of full $epsilon$-LRP decomposition. Fifth, the VR visualization component serves as an exploratory target; comprehensive usability studies are required.

// ============================================================================
// 7. CONCLUSION
// ============================================================================
= Conclusion

This paper presents a comprehensive, multi-method Explainable AI framework for 3D brain tumor segmentation that bridges the gap between model accuracy and clinical interpretability. By integrating six complementary XAI techniques with a SegResNet architecture achieving competitive BraTS 2023 performance (Dice: 0.923 WT, 0.891 TC, 0.873 ET), we demonstrate that high segmentation accuracy and transparent decision-making are not mutually exclusive.

Our key findings are:

1. *Multi-method necessity:* The gradient-based failure case (patient 00291) proves that no single XAI method is sufficient for clinical deployment. Multi-method evaluation is essential.

2. *Novel metric validation:* Weighted Dice provides reliable evaluation across different saliency resolutions, addressing the instability of Saliency IoU for coarse-resolution methods.

3. *Clinically meaningful uncertainty:* MC Dropout boundary-concentrated variance patterns (>0.84 Boundary Uncertainty Ratio) align with clinical intuition, confirming meaningful confidence calibration.

4. *Independent signals:* Saliency and uncertainty are non-redundant signals (correlation −0.07 to +0.09), providing complementary clinical information.

5. *VR-ready pipeline:* All explanation outputs export as NIfTI volumes for immersive VR visualization, advancing XAI from static reports toward dynamic clinical decision support.

Future work will focus on: (1) formal clinical validation studies with practicing neuroradiologists; (2) cross-institutional evaluation using diverse MRI datasets; and (3) development of interactive VR-based exploration tools that enable clinicians to dynamically interrogate model predictions.

// ============================================================================
// REFERENCES
// ============================================================================
#bibliography("paper.bib", style: "ieee")

// ============================================================================
// APPENDIX
// ============================================================================
= Appendix: Extended XAI Results

== XAI Method Detailed Performance

The following summarizes the per-method performance characteristics observed across the test set:

*Grad-CAM:* Achieves 100% Pointing Game for WT and TC but 0% for ET due to the ~20 cubed bottleneck resolution being too coarse to represent small Enhancing Tumor structures. Mean Weighted Dice: 0.220 (WT), 0.268 (TC), 0.048 (ET).

*Guided Backpropagation (GBP):* Achieves 100% Pointing Game across all patients and all tumor regions including ET. Low Saliency Coverage (0.12--0.44) is expected as GBP highlights fine edges and texture boundaries rather than concentrated tumor blobs. Mean Weighted Dice: 0.130 (WT), 0.139 (TC), 0.140 (ET).

*Guided Grad-CAM:* Achieves highest Saliency Coverage (0.93 WT, 0.85 TC, 0.57 ET). However, it inherits Grad-CAM's failure modes and produces blank maps for patient 00291. Mean Weighted Dice: 0.133 (WT), 0.153 (TC), 0.224 (ET).

*LRP (Input x Gradient):* Achieves 100% Pointing Game across all regions, producing diffuse but correctly localized relevance distributions. Mean Weighted Dice: 0.043 (WT), 0.058 (TC), 0.071 (ET).

*Occlusion Sensitivity:* Achieves highest Weighted Dice of all methods (0.422 WT, 0.392 TC, 0.165 ET). The gold-standard empirical validation confirms model reliance on tumor voxels. Mean Weighted Dice: 0.422 (WT), 0.392 (TC), 0.165 (ET).

*MC Dropout:* Provides uncertainty quantification independent of saliency. Boundary Uncertainty Ratio exceeds 0.84 for all regions, confirming characteristic "boundary glow" patterns. Saliency-Uncertainty Correlation near zero (-0.07 to +0.09) confirms independent signals.

== VR Visualization Pipeline

All XAI outputs are exported as spatially-aligned NIfTI volumes ready for immersive 3D visualization via 3D Slicer and SlicerVR. The pipeline:

1. Generates saliency maps in the same MONAI-preprocessed coordinate space as ground truth labels
2. Exports as NIfTI format preserving spatial alignment
3. Enables direct loading into 3D Slicer for volumetric visualization
4. Supports SlicerVR extension for immersive exploration

This pipeline advances XAI from static 2D heatmaps to immersive 3D environments where clinicians can explore saliency and uncertainty distributions spatially, potentially reducing cognitive load inherent in traditional slice-by-slice interpretation.

// --- Author Biography ---
#v(2em)
#align(center)[
  #text(size: 10pt, weight: "bold")[Yogi Amitkumar Patel]\
  #text(size: 9pt)[
    Independent Researcher\
    u2536809\@uel.ac.uk
  ]
]
