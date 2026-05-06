// =============================================================================
// Research Paper Draft 2: Illuminating the Black Box
// Author: Yogi Amitkumar Patel
// Template: IEEE-style conference/journal format
// =============================================================================

// --- Document Metadata ---
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
    School of Architecture, Computing and Engineering\
    University of East London\
    London, United Kingdom\
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
  Deep learning models for brain tumor segmentation have achieved remarkable accuracy, yet their opaque decision-making processes present a critical barrier to clinical adoption. This paper presents a comprehensive Explainable AI (XAI) framework that integrates six complementary post-hoc explanation techniques with a SegResNet architecture for volumetric brain tumor segmentation on the BraTS 2023 dataset. Our framework combines gradient-based methods (Grad-CAM, Guided Backpropagation, Guided Grad-CAM, and Layer-wise Relevance Propagation), perturbation-based analysis (Occlusion Sensitivity), and probabilistic uncertainty quantification (Monte Carlo Dropout) to provide multi-perspective interpretability of 3D CNN predictions. The SegResNet model achieves competitive segmentation performance with mean Dice scores of 0.923 (Whole Tumor), 0.891 (Tumor Core), and 0.873 (Enhancing Tumor) on the held-out test set comprising 126 patients. Quantitative evaluation of XAI outputs using four novel metrics - Pointing Game accuracy, Saliency Coverage, Saliency IoU, and Weighted Dice - demonstrates that Guided Grad-CAM achieves 93% saliency coverage for the Whole Tumor region, while Occlusion Sensitivity provides the highest spatial alignment (IoU: 0.19). Monte Carlo Dropout uncertainty analysis reveals characteristic boundary-concentrated variance patterns, with Boundary Uncertainty Ratios exceeding 0.84, confirming clinically meaningful confidence calibration. All explanation outputs are exported as spatially-aligned NIfTI volumes for immersive 3D visualization via 3D Slicer and SlicerVR. This work bridges the gap between model accuracy and clinical trust by providing a unified, quantitatively validated explainability pipeline for volumetric medical image analysis.
]

#v(0.5em)
#text(size: 9pt, weight: "bold")[Keywords:] #text(size: 9pt)[Explainable AI, Brain Tumor Segmentation, 3D CNN, Grad-CAM, Uncertainty Quantification, BraTS 2023, SegResNet, Virtual Reality, Monte Carlo Dropout, Layer-wise Relevance Propagation]
#v(1.5em)

// ============================================================================
// 1. INTRODUCTION
// ============================================================================
= Introduction

In modern neuro-oncology, accurate identification and delineation of brain tumors from Magnetic Resonance Imaging (MRI) data remains a critical clinical challenge. Manual segmentation by radiologists is labor-intensive, subjective, and increasingly unsustainable as imaging volumes grow @brats2015. Deep learning models, particularly 3D Convolutional Neural Networks (CNNs), have demonstrated superior performance in automated volumetric segmentation tasks @iftikhar2025 @bhati2024, processing entire MRI volumes rather than individual 2D slices to capture the true three-dimensional spatial relationships of tumor pathology. However, the clinical deployment of these high-performing models is critically hindered by the _"black box"_ problem: their internal decision-making processes remain fundamentally opaque to the clinicians who must rely on their outputs @neri2023. In the high-stakes domain of neurosurgery, this opacity creates a _"trust gap"_ @wen2025 - it is insufficient for a model to simply output a segmentation mask; clinicians require the ability to verify _why_ specific voxels were classified as tumor tissue. This lack of interpretability is not merely a technical limitation but a barrier to ethical and safe clinical deployment, particularly within regulatory frameworks such as the GDPR's "right to explanation" @neri2023.

Explainable AI (XAI) techniques aim to address this disconnect by illuminating model reasoning. Methods such as Gradient-weighted Class Activation Mapping (Grad-CAM) @selvaraju2017 have become established tools for visualizing CNN decisions. However, existing XAI frameworks for 3D medical imaging suffer from three critical limitations: (1) they typically employ only a single explanation method, providing an incomplete picture of model behavior; (2) they lack uncertainty quantification, making it impossible to distinguish between confident and uncertain predictions; and (3) they present explanations as static 2D overlays, failing to convey the volumetric nature of the underlying pathology @neuroxai @brainAR.

This paper presents a unified, multi-method XAI framework that addresses these limitations. Our key contributions are:

1. *Multi-perspective explainability:* We implement and quantitatively compare six complementary XAI techniques - Grad-CAM, Guided Backpropagation (GBP), Guided Grad-CAM, Layer-wise Relevance Propagation (LRP), Occlusion Sensitivity, and Monte Carlo (MC) Dropout - providing gradient-based, perturbation-based, and probabilistic perspectives on model decision-making.

2. *Quantitative XAI validation:* We introduce a systematic evaluation framework using four metrics (Pointing Game, Saliency Coverage, Saliency IoU, and Weighted Dice) that objectively assess whether the model "looks at the right place for the right reasons."

3. *Uncertainty-aware explainability:* We integrate MC Dropout uncertainty quantification with structural attribution methods (LRP), enabling identification of _"brittle"_ decision regions where high feature importance intersects with high predictive variance - a critical clinical red flag.

4. *VR-ready volumetric pipeline:* All XAI outputs are exported as spatially-aligned NIfTI volumes, ready for immersive 3D visualization through platforms such as 3D Slicer and SlicerVR.

The remainder of this paper is organized as follows: Section 2 reviews related work in 3D segmentation architectures, XAI methods, and immersive visualization. Section 3 details our proposed framework including the SegResNet architecture and all six XAI methods. Section 4 describes the experimental setup including dataset, preprocessing, and training configuration. Section 5 presents quantitative results for both segmentation performance and XAI evaluation. Section 6 discusses findings, clinical implications, and limitations. Section 7 concludes with future directions.

// ============================================================================
// 2. RELATED WORK
// ============================================================================
= Related Work

== 3D Deep Learning for Brain Tumor Segmentation

The evolution from 2D slice-based to volumetric 3D approaches has fundamentally improved brain tumor segmentation. The 3D U-Net @cicek2016 established the encoder-decoder paradigm with skip connections for volumetric data, while V-Net @milletari2016 introduced residual connections and the Dice loss function to address class imbalance. Modern architectures such as SegResNet @myronenko2019 combine these innovations, using residual blocks with GroupNorm for stable deep network training and DiceFocalLoss for handling severely imbalanced tumor sub-regions. The BraTS challenge @brats2015 has served as the primary benchmark for evaluating these architectures on multi-modal MRI data (T1, T1c, T2, FLAIR), with current state-of-the-art models achieving Dice scores exceeding 0.90 for the Whole Tumor region. The BrainAR framework @brainAR demonstrated the potential of combining 3D U-Net architectures with augmented reality visualization, achieving Dice scores of 0.914 for whole tumor segmentation. Similarly, the AXONS-3 framework @axons3 applied post-hoc XAI techniques to 3D brain tumor segmentation, highlighting the importance of interpretability in clinical AI systems.

== Explainable AI for Medical Imaging

XAI methods for CNNs are broadly categorized as _gradient-based_, _perturbation-based_, or _decomposition-based_. Grad-CAM @selvaraju2017 computes class-discriminative heatmaps using gradients flowing into a target convolutional layer, providing coarse but class-specific spatial localization. Guided Backpropagation @springenberg2015 produces full-resolution saliency maps by gating negative gradients during backpropagation. Their fusion - Guided Grad-CAM - achieves both high resolution and class specificity @selvaraju2017. Layer-wise Relevance Propagation (LRP) @montavon2017 operates on a conservation principle, attributing the output prediction score back to individual input voxels. For complex architectures, the Input $times$ Gradient approximation provides a computationally tractable proxy equivalent to $epsilon$-LRP in ReLU networks. Occlusion Sensitivity @zeiler2014 offers a complementary, model-agnostic approach by systematically masking input regions and measuring prediction changes.

For uncertainty quantification, MC Dropout @gal2016dropout provides a Bayesian approximation by maintaining dropout during inference across multiple stochastic forward passes. The variance across predictions serves as a proxy for model uncertainty, identifying regions where the network lacks confidence. Recent surveys @bhati2024 have highlighted the growing importance of XAI in medical imaging, emphasizing that no single method provides a complete picture of model behavior - a gap our multi-method framework directly addresses.

The NeuroXAI framework @neuroxai applied multiple gradient-based XAI methods to brain tumor segmentation, achieving 90% clinician alignment. The AXONS-3 framework @axons3 advanced this by integrating trust metrics into 3D segmentation pipelines. However, both frameworks lack integrated uncertainty quantification and rely on static visualization, limiting their clinical utility for interactive decision support. Similarly, BrainAR @brainAR demonstrated immersive tumor visualization but did not incorporate XAI techniques into the VR pipeline.

== Immersive Visualization in Medical Imaging

Traditional slice-by-slice visualization of 3D data imposes significant cognitive load on clinicians @brainAR. Virtual Reality (VR) environments provide stereoscopic depth perception and six-degrees-of-freedom interaction, enabling clinicians to explore volumetric XAI outputs as spatial clouds rather than flat overlays @neuroxai. Platforms such as 3D Slicer with the SlicerVR extension support direct rendering of NIfTI volumes in VR headsets, bridging the gap between medical imaging formats and immersive display systems @Zeineldin2023. The BrainAR framework demonstrated that AR-based tumor visualization can reduce diagnostic uncertainty and improve surgical planning accuracy, highlighting the clinical value of immersive visualization approaches. However, prior work has not combined XAI techniques with VR visualization for brain tumor segmentation, creating a gap between interpretability research and immersive clinical tools.

// ============================================================================
// 3. PROPOSED FRAMEWORK
// ============================================================================
= Proposed Framework

Our framework implements a three-stage pipeline: (1) volumetric tumor segmentation using SegResNet, (2) multi-method XAI analysis, and (3) VR-compatible NIfTI export. This section details the architecture and each XAI component.

== SegResNet Architecture

The backbone of our framework is SegResNet @myronenko2019, a 3D encoder-decoder architecture that processes four-channel MRI volumes ($160^3$ voxels) to produce three-channel segmentation masks corresponding to the Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) regions.

The encoder progressively downsamples input through four levels, increasing filters from 32 to 256 at the bottleneck, using strided $3 times 3 times 3$ convolutions. Each level is composed of Residual Blocks (ResBlocks) featuring GroupNorm normalization and ReLU activations. These blocks learn residual mappings via skip connections, preventing vanishing gradients and ensuring stable training of deep 3D networks @he2016deep. The decoder reconstructs the segmentation map via trilinear upsampling and encoder-decoder skip connections. A dropout probability of $p = 0.1$ serves dual purposes: regularization during training and enabling MC Dropout inference.

To address severe class imbalance - particularly the ET region comprising merely ~5% of tumor volume - the model employs DiceFocalLoss with a focal parameter $gamma = 2.0$, which increases the loss contribution from hard-to-classify voxels:

$
  "DiceFocalLoss" = "DiceLoss" + lambda dot.c "FocalLoss"
$

where DiceLoss optimizes volumetric overlap and FocalLoss ($"FL"(p_t) = -alpha_t (1-p_t)^gamma log(p_t)$) heavily penalizes confident misclassifications.

== XAI Method Suite

We implement six XAI methods spanning three paradigms: gradient-based, perturbation-based, and probabilistic. Each method provides a distinct perspective on model decision-making.

=== Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM computes per-feature-map importance weights by analyzing gradients flowing into the bottleneck convolutional layer (256 channels at $20^3$ resolution). For target class $c$ and feature map $k$, the importance weight $alpha_k^c$ is computed via global average pooling of gradients:

$
  alpha_k^c = 1/Z sum_(x) sum_(y) sum_(z) (partial y^c)/(partial A^k_(x,y,z))
$

where $y^c$ is the spatially-averaged output logit and $A^k_(x,y,z)$ represents feature map activations. The class-discriminative heatmap is obtained through a weighted linear combination followed by ReLU activation:

$
  L^c_"Grad-CAM" = "ReLU"(sum_k alpha_k^c A^k)
$

The resulting $20^3$ heatmap is upsampled to $160^3$ via trilinear interpolation and normalized to $[0, 1]$.

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

This effectively masks fine-grained GBP features by the class-discriminative Grad-CAM localization, yielding high-resolution attributions specific to each tumor sub-region.

A critical implementation detail is _hook isolation_: simultaneously active Grad-CAM and GBP backward hooks corrupt each other's gradient flows. Our framework employs sequential per-patient computation - first Grad-CAM hooks, then GBP hooks on a clean model - before multiplying the stored results.

=== Layer-wise Relevance Propagation (LRP)

LRP attributes the model's output prediction score back to each input voxel based on a conservation principle. Due to the architectural complexity of SegResNet (skip connections, GroupNorm layers), we employ the Input $times$ Gradient approximation, which is mathematically equivalent to $epsilon$-LRP in ReLU networks @montavon2017:

$
  R_i approx x_i dot.c (partial f(x)) / (partial x_i)
$

This yields ultra-high-resolution ($160^3$) relevance maps quantifying which specific structural features drove the network's prediction.

=== Occlusion Sensitivity

Occlusion Sensitivity provides model-agnostic, perturbation-based explanations by systematically testing the CNN's structural reliance on specific brain regions. A $16 times 16 times 16$ sliding window moves across the input volume with stride 8, replacing occluded voxels with zeros. At each position $(i,j,k)$, the sensitivity score measures the absolute prediction drop:

$
  S_c(i,j,k) = f_c(x) - f_c(x_"occluded")
$

The resulting $20^3$ sensitivity map is upsampled to $160^3$ via trilinear interpolation. This method requires approximately 1,000 forward passes per patient but provides direct empirical evidence of regional importance.

=== Monte Carlo Dropout (Uncertainty Quantification)

Unlike the preceding methods which address _"where is the model looking?"_, MC Dropout answers _"how confident is the model?"_ At test time, SegResNet's dropout layers ($p=0.1$) remain active during $N=20$ stochastic forward passes. The stabilized mean prediction and voxel-wise predictive variance are:

$
  macron(p) = 1/N sum_(n=1)^N p_n, quad quad sigma^2(x,y,z) = 1/N sum_(n=1)^N (p_n(x,y,z) - macron(p)(x,y,z))^2
$

High variance identifies _"brittle"_ transition zones where the network lacks confidence. Comparing LRP attributions against uncertainty maps enables a novel cross-method analysis: regions exhibiting both high relevance _and_ high uncertainty represent clinical red flags where the model heavily relies on features about which it is uncertain.

== Quantitative XAI Evaluation Metrics

All saliency maps are evaluated against ground truth tumor annotations using four metrics:

#figure(
  table(
    columns: (22%, 28%, 1fr),
    inset: 8pt,
    align: horizon,
    table.header([*Metric*], [*Formula*], [*Interpretation*]),
    [Pointing Game], [$ cases(1 "if" arg max(L) in G, 0 "otherwise") $], [Binary hit/miss test: is the peak saliency voxel inside the tumor region $G$?],
    [Saliency Coverage], [$ (sum_(x in G) L(x)) / (sum_x L(x)) $], [Fraction of total saliency mass concentrated inside the tumor boundary.],
    [Saliency IoU], [$ |L_"thresh" sect G| / (|L_"thresh" union G|) $], [Shape overlap between thresholded saliency ($tau = 0.5$) and ground truth.],
    [Weighted Dice], [$ (2 sum L(x) dot.c G(x)) / (sum L(x) + sum G(x)) $], [Soft overlap treating continuous saliency as prediction weights.],
  ),
  caption: [XAI evaluation metrics for saliency map validation against ground truth.],
) <xai_metrics>

Additionally, MC Dropout outputs are evaluated using the Uncertainty Area Ratio (UAR), Boundary Uncertainty Ratio, mean uncertainty inside/outside the tumor, and the Saliency-Uncertainty Correlation between LRP importance and variance maps.

// ============================================================================
// 4. EXPERIMENTAL SETUP
// ============================================================================
= Experimental Setup

== Dataset

The BraTS 2023 dataset comprises approximately 1,251 patient cases, each containing four co-registered MRI modalities: T1-weighted (T1), T1 with Gadolinium contrast (T1c), T2-weighted (T2), and T2-FLAIR. All volumes are resampled to 1 mm³ isotropic resolution and skull-stripped. Tumor annotations follow the BraTS standard: necrotic core (label 1), peritumoral edema (label 2), and enhancing tumor (label 3), converted into three binary channels: WT (labels 1+2+3), TC (labels 1+3), and ET (label 3 only).

The dataset was split deterministically (seed=42) into 1,000 training, 125 validation, and approximately 126 test cases. Validation was used for hyperparameter selection and early stopping; the test set was reserved for final evaluation and XAI analysis.

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
  caption: [Training hyperparameters for SegResNet on BraTS 2023.],
) <training_params>

Best model checkpoints were selected based on validation Dice score, with experiment tracking via Weights & Biases (W&B).

== Implementation

The framework is implemented in Python using PyTorch and MONAI @monai2022. The modular architecture separates data handling (`src/data`), model construction (`src/models`), training orchestration (`src/training`), XAI generation (`src/xai`), and evaluation (`src/evaluation`). XAI metrics are computed _inline_ during saliency generation to guarantee spatial alignment between saliency maps and ground truth labels in the same MONAI-preprocessed coordinate space.

// ============================================================================
// 5. RESULTS
// ============================================================================
= Results

== Segmentation Performance

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 8pt,
    align: center,
    table.header([*Metric*], [*Whole Tumor*], [*Tumor Core*], [*Enhancing Tumor*]),
    [Dice Score], [0.923 ± 0.079], [0.891 ± 0.179], [0.873 ± 0.159],
    [HD95 (mm)], [5.60 ± 8.76], [4.54 ± 7.60], [3.66 ± 7.22],
    [IoU], [0.865 ± 0.115], [0.836 ± 0.205], [0.799 ± 0.182],
    [Sensitivity], [0.925 ± 0.087], [0.891 ± 0.195], [0.859 ± 0.236],
    [Specificity], [0.999 ± 0.001], [0.999 ± 0.001], [1.000 ± 0.000],
  ),
  caption: [SegResNet segmentation performance on the BraTS 2023 test set (mean ± std, $n = 126$).],
) <seg_results>

The model achieves competitive performance across all three tumor sub-regions, with the highest Dice score for WT (0.923) and notably low HD95 for ET (3.66 mm), indicating precise boundary delineation for the most clinically critical sub-region. The near-perfect specificity (>0.999) confirms reliable exclusion of healthy tissue. The Enhancing Tumor region shows the highest variability in both Dice (0.159) and HD95 (7.22), reflecting the well-known challenge of segmenting the smallest and most ambiguous tumor sub-region.

== XAI Comparative Analysis

The XAI comparative analysis presents the performance of XAI methods evaluated on the test set.

#figure(
  table(
    columns: (25%, 15%, 18%, 14%, 18%),
    inset: 7pt,
    align: center,
    table.header([*Method*], [*PG (%)*], [*Coverage*], [*IoU*], [*W. Dice*]),
    table.hline(),
    [_Whole Tumor_], [], [], [], [],
    [GBP], [100], [0.223], [0.008], [0.130],
    [Guided Grad-CAM], [100], [*0.930*], [0.004], [0.133],
    [Occlusion], [100], [0.617], [*0.190*], [*0.422*],
    table.hline(),
    [_Tumor Core_], [], [], [], [],
    [GBP], [100], [0.282], [0.015], [0.139],
    [Guided Grad-CAM], [100], [*0.849*], [0.011], [0.153],
    [Occlusion], [80], [0.416], [*0.230*], [*0.392*],
    table.hline(),
    [_Enhancing Tumor_], [], [], [], [],
    [GBP], [100], [0.149], [0.069], [0.140],
    [Guided Grad-CAM], [100], [*0.570*], [0.059], [*0.224*],
    [Occlusion], [40], [0.193], [*0.115*], [0.205],
  ),
  caption: [Comparative XAI evaluation across methods and tumor sub-regions. PG = Pointing Game. Bold indicates best per-metric.],
) <xai_comparison>

Several key findings emerge from this analysis:

*Pointing Game accuracy:* Both GBP and Guided Grad-CAM achieve 100% Pointing Game accuracy across all regions, confirming that the model's peak attention consistently falls within the tumor boundary. Occlusion Sensitivity shows reduced accuracy for smaller sub-regions (40% for ET), attributable to the coarse $16^3$ sliding window resolution.

*Saliency Coverage:* Guided Grad-CAM achieves the highest coverage across all regions (0.93 for WT), indicating that 93% of the model's total attention mass is concentrated within the tumor boundary. This substantially outperforms standalone GBP (0.22 for WT), confirming the critical role of class-discriminative masking via the Grad-CAM component.

*Spatial Alignment (IoU and Weighted Dice):* Occlusion Sensitivity achieves the highest IoU scores (0.19 for WT), demonstrating superior spatial alignment between high-importance regions and ground truth despite its coarse native resolution. For Weighted Dice, which evaluates continuous saliency distributions, Occlusion again leads (0.42 for WT), while Guided Grad-CAM achieves the best ET Weighted Dice (0.22).

== Uncertainty Quantification

MC Dropout uncertainty statistics are presented in Table 6.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 7pt,
    align: center,
    table.header([*Region*], [*UAR*], [*Boundary Unc. Ratio*], [*Mean Unc. Inside*], [*Mean Unc. Outside*]),
    [Whole Tumor], [0.435], [0.839], [0.003], [0.00006],
    [Tumor Core], [0.496], [0.913], [0.007], [0.00006],
    [Enhancing Tumor], [0.447], [0.858], [0.011], [0.00003],
  ),
  caption: [MC Dropout uncertainty statistics ($N=20$ forward passes, $p=0.1$).],
) <mc_dropout>

The results reveal three critical patterns:

*Boundary-concentrated uncertainty:* The Boundary Uncertainty Ratio exceeds 0.84 for all regions, confirming that model uncertainty concentrates at tumor-healthy tissue boundaries - the characteristic _"boundary glow"_ pattern. This is clinically expected and desirable: the model is confident about the tumor interior and healthy tissue but appropriately uncertain at ambiguous transition zones.

*Asymmetric confidence:* Mean uncertainty inside the tumor (0.003–0.011) is 50–180 times higher than outside (0.00003–0.00006), indicating the model has substantially higher confidence when classifying healthy tissue compared to tumor tissue. The ET region exhibits the highest internal uncertainty (0.011), consistent with its smaller volume and more ambiguous boundaries.

*Cross-method analysis (LRP $times$ Uncertainty):* Low Saliency-Uncertainty Correlation values (-0.07 to +0.09) indicate that LRP importance and uncertainty are largely independent - regions the model relies upon are not preferentially uncertain. This is a positive finding, suggesting that the model's high-importance features are generally also its most stable predictions, rather than "brittle" decision points.

// ============================================================================
// 6. DISCUSSION
// ============================================================================
= Discussion

== Complementary Nature of XAI Methods

Our results demonstrate that no single XAI method provides a complete picture of model behavior. Grad-CAM excels at coarse class-discriminative localization, GBP provides fine-grained voxel-level detail, Guided Grad-CAM combines both strengths, LRP offers conservation-based relevance attribution, Occlusion Sensitivity provides direct empirical importance evidence, and MC Dropout quantifies prediction confidence. Each method answers a fundamentally different question about the model's decision-making process:

- *Grad-CAM / Guided Grad-CAM:* "Which spatial regions influence the class prediction?"
- *GBP / LRP:* "Which specific voxel-level features drove the output?"
- *Occlusion Sensitivity:* "Which regions are empirically necessary for the prediction?"
- *MC Dropout:* "How confident is the model about each voxel?"

This multi-perspective approach enables cross-validation of explanations. For instance, a region highlighted by Grad-CAM but showing high MC Dropout uncertainty would warrant clinical scrutiny, whereas a region consistently identified across gradient, perturbation, and uncertainty methods provides strong convergent evidence for the model's reasoning.

== Clinical Implications

The high saliency coverage of Guided Grad-CAM (93% for WT) provides clinically actionable information: the model demonstrably focuses its attention within the tumor boundary for the correct reasons, not on spurious correlations or imaging artifacts. This addresses the "right place for the right reasons" criterion essential for clinical trust @neri2023.

The boundary-concentrated uncertainty pattern from MC Dropout aligns with clinical intuition - tumor margins are inherently ambiguous due to infiltrative growth patterns, particularly in gliomas. The model's appropriate uncertainty at boundaries, combined with high confidence in tumor interiors and healthy tissue, suggests meaningful confidence calibration rather than overconfident predictions.

The combination of structural attribution (LRP) with uncertainty quantification enables identification of anatomically ambiguous regions where the prediction may be unreliable. This capability directly addresses the _"interactivity gap"_ identified in existing XAI frameworks @neuroxai, where static explanations cannot distinguish between confident and uncertain model behavior. Our VR-ready NIfTI export pipeline extends this by enabling immersive exploration of both saliency maps and uncertainty volumes in clinical VR environments.

== Limitations

Several limitations should be acknowledged. First, our framework relies exclusively on the BraTS 2023 dataset; cross-institutional validation is necessary to assess generalizability under domain shift. Second, the Occlusion Sensitivity method, while providing the highest spatial alignment metrics, requires approximately 1,000 forward passes per patient, limiting its practicality for real-time clinical workflows. Third, while our metrics quantitatively evaluate XAI outputs, they serve as academic proxies for clinical utility; formal evaluation with practicing radiologists remains essential for validating diagnostic efficacy. Fourth, the Input $times$ Gradient approximation for LRP, while computationally efficient, may not capture all relevance dynamics of the full $epsilon$-LRP decomposition in networks with complex skip connections. Fifth, the VR visualization component serves as an exploratory target; comprehensive usability studies with clinical end-users are required to establish its comparative advantage over traditional 2D viewing.

// ============================================================================
// 7. CONCLUSION
// ============================================================================
= Conclusion

This paper presents a comprehensive, multi-method Explainable AI framework for 3D brain tumor segmentation that bridges the gap between model accuracy and clinical interpretability. By integrating six complementary XAI techniques - spanning gradient-based, perturbation-based, and probabilistic approaches - with a SegResNet architecture achieving competitive BraTS 2023 performance (Dice: 0.923 WT, 0.891 TC, 0.873 ET), we demonstrate that high segmentation accuracy and transparent decision-making are not mutually exclusive objectives.

Our quantitative evaluation reveals that different XAI methods offer complementary strengths: Guided Grad-CAM achieves exceptional saliency coverage (93% for WT), while Occlusion Sensitivity provides superior spatial alignment (IoU: 0.19). MC Dropout uncertainty analysis confirms clinically meaningful confidence calibration, with boundary-concentrated variance patterns (Boundary Uncertainty Ratio > 0.84) and low saliency-uncertainty correlation indicating that the model's reasoning is both focused and stable.

The framework's export of all explanation outputs as spatially-aligned NIfTI volumes enables their visualization in immersive VR environments, offering a pathway to reduce the cognitive load inherent in interpreting complex 3D pathologies through traditional 2D interfaces.

Future work will focus on three directions: (1) formal clinical validation studies with practicing neuroradiologists to evaluate the diagnostic utility of multi-method XAI explanations; (2) cross-institutional evaluation using diverse MRI datasets to assess framework generalizability; and (3) development of interactive VR-based exploration tools that enable clinicians to dynamically interrogate model predictions, advancing XAI from static reports toward dynamic clinical decision support.

// ============================================================================
// REFERENCES
// ============================================================================
#bibliography("paper.bib", style: "ieee")

// ============================================================================
// APPENDIX (Optional)
// ============================================================================
= Appendix

== Supplementary Materials

Additional results, implementation details, and patient-level metrics are available in the supplementary materials accompanying this paper. The source code for the complete framework is publicly available at the project repository.

// --- Biographical Sketch ---
#v(2em)
#align(center)[
  #text(size: 10pt, weight: "bold")[Yogi Amitkumar Patel]\
  #text(size: 9pt)[
    School of Architecture, Computing and Engineering\
    University of East London\
    London, United Kingdom
  ]
]
