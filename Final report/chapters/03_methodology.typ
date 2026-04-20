= Methodology

== Introduction

This chapter delineates the methodological framework for developing an Explainable AI (XAI) system capable of both accurate brain tumor segmentation and transparent clinical decision-making. The central objective is to address the "black-box" limitation common in medical imaging AI by transforming a complex deep learning model into an interpretable system. This system aims to achieve high segmentation accuracy while providing clinicians with actionable insights into the underlying reasoning behind each prediction.

To ensure transparency, the framework integrates six complementary XAI methodologies, each validated against ground-truth annotations to confirm their clinical relevance. At the core of this system is SegResNet, a 3D residual encoder-decoder architecture designed to process full volumetric MRI data rather than individual 2D slices. This volumetric approach enables the model to effectively capture the spatial hierarchies across three critical tumor sub-regions: the Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

The research design follows a hybrid Agile/CRISP-DM methodology, strategically combining the systematic rigor required for medical data mining with the iterative flexibility necessary for deep learning experimentation. This dual approach ensures methodological robustness appropriate for safety-critical healthcare applications while accommodating the evolutionary nature of neural network development, which transitioned from initial rapid prototyping on a 250-patient subset to full-scale training on the complete BraTS corpus.

The remainder of this chapter is structured as follows: Section 3.2 outlines the overall research approach and rationale; Section 3.3 details the technical implementation, including data preprocessing, model architecture, and XAI methods; Section 3.4 describes the research design and evaluation strategy; Section 3.5 discusses technical challenges and methodological limitations; and Section 3.6 addresses ethical considerations and data governance.

== Approach

=== Overall Approach and Rationale

This study creates a three-stage pipeline using the BraTS 2023 dataset, which consists of about 1,250 multi-modal MRI patient cases: (1) training a SegResNet 3D CNN for volumetric tumour segmentation; (2) integrating multiple XAI techniques to illuminate model decision-making; and (3) exporting results into a VR-compatible format for immersive clinical evaluation. This volumetric methodology captures the true three-dimensional character of brain tumours, producing more anatomically accurate segmentations of the Whole Tumour, Tumour Core, and Enhancing Tumour areas, in contrast to 2D slice-by-slice methodologies that sacrifice inter-slice spatial context @milletari2016 @cicek2016.

SegResNet was selected as the primary architecture due to its residual connections, which provide robustness against vanishing gradients and enable effective training of deep networks @he2016deep. The architecture offers an optimal balance between performance and computational complexity, making it suitable for medical imaging applications where training data may be limited and model interpretability is essential.

The rationale for implementing multiple XAI methods stems from their complementary strengths. Gradient-based methods (Grad-CAM, Guided Backpropagation, LRP) provide different perspectives on feature importance, while perturbation-based approaches (Occlusion Sensitivity) offer model-agnostic explanations. Monte Carlo Dropout adds uncertainty quantification, enabling identification of regions where the model lacks confidence, a critical capability for clinical decision support.

=== Agile CRISP-DM Methodology and Its Use

Deep learning model training is very exploratory and unpredictable, making traditional waterfall development lifecycles inappropriate. A more flexible approach is required due to the intricacy of neural network designs, as well as the requirement for substantial hyperparameter adjustment and iterative improvement. The Cross-Industry Standard Process for Data Mining's (CRISP-DM) structured phases are combined with Agile development's rapid iteration and evolutionary prototyping concepts in this study's hybrid Agile/CRISP-DM approach.

#figure(
  image("../Figures/Agile_crisp_dm.svg", width: 75%),
  caption: [The Agile CRISP-DM methodology employed in this research, showing the iterative cycles between the six core phases.],
) <fig:agile-crisp-dm>

@fig:agile-crisp-dm illustrates the six interconnected phases of the methodology, with dashed arrows indicating the iterative nature of the development process. Each phase feeds into the next, but the Agile approach allows for rapid cycling back to earlier phases when issues are discovered or improvements are identified.

The framework, which was created using an iterative CRISP-DM process, was continuously validated against strict clinical criteria (Dice, HD95, IoU, sensitivity, and specificity) while containing Explainable AI components that were verified by saliency overlap and uncertainty quantification. Although the system is still an experimental research prototype awaiting clinical deployment, the pipeline produces standardised NIfTI outputs, containing segmentations, attention heatmaps, and variance arrays for direct 3D Slicer VR visualisation.

The Agile CRISP-DM approach ensured an organized, rapid-iteration development loop while maintaining the stringent robustness required for safety-critical medical machine learning components. Regular sprint reviews allowed for course correction, and the modular architecture enabled independent validation of each component.

== Implementation

=== Data Acquisition and Preprocessing

The BraTS 2023 dataset, provided through the MICCAI Brain Tumor Segmentation Challenge, serves as the foundation for this research. The dataset comprises approximately 1,251 patient cases, each containing four co-registered MRI modalities: T1-weighted (T1), T1-weighted with Gadolinium contrast enhancement (T1c), T2-weighted (T2), and T2 Fluid-Attenuated Inversion Recovery (FLAIR). All volumes have been resampled to an isotropic resolution of 1 mm³ and skull-stripped.

The tumor labels in BraTS follow a standard annotation scheme where each voxel is classified as background (0), necrotic tumor core (1), peritumoral edema (2), or enhancing tumor (3). For this research, these labels are converted into three binary channels representing clinically relevant regions: Whole Tumor (WT = necrosis + edema + enhancing), Tumor Core (TC = necrosis + enhancing), and Enhancing Tumor (ET = enhancing only). The preprocessing pipeline optimizes raw MRI volumes for efficient 3D model training through four core operations: foreground cropping reduces spatial dimensions from ~240³ to ~160³ voxels by removing extraneous background; channel-first formatting restructures data to [4, 160, 160, 160] for PyTorch compatibility; per-channel z-score normalization standardizes intensities across modalities using only brain tissue statistics; and random spatial augmentation (training only) applies geometric and intensity transforms  such as rendom flips, adding noice etc. to enhance model robustness.

=== SegResNet Model Architecture

#figure(
  image("../Figures/3D_SegResNet.svg", width: 85%),
  caption: [The SegResNet architecture, showing the encoder-decoder structure with residual connections and skip connections.],
) <fig:segresnet>

The implementation utilizes SegResNet, a 3D encoder-decoder architecture designed for volumetric segmentation that processes entire MRI volumes to capture global spatial dependencies. The encoder progressively downsamples input modalities through four levels, increasing filters from 32 to 256 channel bottleneck, using strided convolutions that reduce resolution from $160^3$ to $20^3$. Every layer is constructed from Residual Blocks (ResBlocks) featuring 3×3×3 kernels, GroupNorm, and ReLU activations. These blocks utilize skip connections to prevent vanishing gradients and ensure training stability by learning residual mappings rather than opaque transformations.

The bottleneck layer offers a compressed, globally-aware representation well-suited for Grad-CAM analysis. The decoder reverses the encoder via trilinear upsampling and skip connections, recovering fine-grained spatial detail for the final 160³ outputs. To mitigate severe class imbalance particularly the Enhancing Tumor (ET) region at merely ~5% volume, the model employs DiceFocalLoss (γ = 2.0) to emphasize misclassified voxels. A dropout rate of 0.1 serves dual purposes: regularization and enabling MC Dropout for uncertainty quantification in clinical review.

=== Evolutionary Prototyping Phase

Before committing to full-scale training, an evolutionary prototyping phase was conducted using a 250-patient subset randomly sampled from the BraTS training data. This phase served multiple purposes: validating the complex 3D data pipeline, rapidly testing candidate architectures, and verifying hardware memory limits.The initial prototype was implemented using an 2 NVIDIA GeForce RTX 2080 Ti (11 GB VRAM each), which allowed for training across 10 epochs in approximately 1 hours 2 min. The dataset was split into 200 training cases, 25 validation cases, and 25 test cases for this phase.

The prototype results demonstrated that SegResNet provided the optimal performance-to-complexity ratio among the tested architectures. The 250-patient subset achieved promising results on the prototype test set with Dice scores of 0.80 for the Whole Tumor (WT), 0.55 for the Tumor Core (TC), and 0.34 for the Enhancing Tumor (ET), confirming the viability of the approach for full-scale implementation. After validating this core functionality, the system was refactored into its final modular, configuration-driven PyTorch framework.

#figure(
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        #image("../Figures/Models_Monitor_graphs/250_train_loss.pdf", width: 100%)
        #align(center)[#text(size: 0.85em)[(a) Training Loss]]
      ],
      [
        #image("../Figures/Models_Monitor_graphs/250_batch_loss.pdf", width: 100%)
        #align(center)[#text(size: 0.85em)[(b) Batch Loss]]
      ],
    )
    #v(1em)
    #box(width: 60%)[
      #image("../Figures/Models_Monitor_graphs/250_val_WT.pdf", width: 100%)
      #align(center)[#text(size: 0.85em)[(c) Validation Whole Tumor (WT) Dice]]
    ]
  ],
  caption: [Training metrics for the 250-patient evolutionary prototype.],
) <fig:prototype_metrics>

=== Implementation Pipeline

The implementation follows a modular architecture designed for reproducibility and extensibility, organized into several key components. YAML-based configuration files manage all hyperparameters and paths, supporting accessible parameter sweeps across the experiments. The core data pipeline (`src/data`) manages dataset scanning, patient splitting, and MONAI transform pipelines, while the model factory (`src/models`) dynamically instantiates model architectures, loss functions, and optimizers. Operating in tandem, the training framework (`src/training`) orchestrates the complete training lifecycle, integrating automatic mixed precision, automated checkpointing, and tracking via Weights & Biases. Finally, the explainability aspect is controlled by the XAI suite (`src/xai`), which standardizes the generation of saliency maps and uncertainty metrics, supported directly by the evaluation tools (`src/evaluation`) responsible for computing both segmentation and novel XAI-specific metrics and exporting them to CSV configurations.

The end-to-end workflow proceeds through seven stages: (1) configuration loading and data initialization, (2) model training with validation, (3) test set inference, (4) XAI map generation for all methods, (5) metric computation and CSV export, (6) NIfTI volume export for 3D Slicer, and (7) VR visualization preparation.

#figure(
  image("../Figures/Implementation_pipeline.drawio.svg", width: 85%),
  caption: [The implementation pipeline, showing the modular architecture and end-to-end workflow.],
) <fig:implementation_pipeline>
=== Full-Scale Training Setup

The final training configuration was optimized for a single high-memory GPU (RTX 5880 Ada, 48 GB VRAM) with Automatic Mixed Precision (AMP) enabled to maximize training efficiency. Key hyperparameters include:

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 8pt,
    align: (left, left),
    table.header([*Parameter*], [*Value*]),
    [Batch Size], [1 (limited by 3D volume memory)],
    [Epochs], [35],
    [Learning Rate], [5×10⁻⁵],
    [LR Scheduler], [Cosine Annealing],
    [Optimizer], [AdamW (weight decay 1×10⁻⁵)],
    [Loss Function], [DiceFocalLoss],
    [Focal Gamma], [2.0],
    [Mixed Precision], [Enabled],
  ),
  caption: [SegResNet training hyperparameters for BraTS 2023 full dataset.],
) <tab:training-params>

#figure(
  [
    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        #image("../Figures/Models_Monitor_graphs/Final_train_loss.pdf", width: 100%)
        #align(center)[#text(size: 0.85em)[(a) Training Loss]]
      ],
      [
        #image("../Figures/Models_Monitor_graphs/Final_batch_loss.pdf", width: 100%)
        #align(center)[#text(size: 0.85em)[(b) Batch Loss]]
      ],
    )
    #v(1em)
    #box(width: 60%)[
      #image("../Figures/Models_Monitor_graphs/Final_val_WT.pdf", width: 100%)
      #align(center)[#text(size: 0.85em)[(c) Validation Whole Tumor (WT) Dice]]
    ]
  ],
  caption: [Training metrics for the full-scale SegResNet model (1,251 patients).],
) <fig:full_training_metrics>

The dataset was split into 1,000 training cases, 125 validation cases, and approximately 126 test cases. The validation set was used for hyperparameter tuning and early stopping, while the test set was reserved for final evaluation and XAI analysis. All splits were performed deterministically using a fixed random seed (42) to ensure reproducibility.

Training progress was monitored using Weights & Biases (W&B) for experiment tracking, with metrics logged at every iteration. Best model checkpoints were selected based on validation Dice score, and the final model was evaluated on the held-out test set to report unbiased performance estimates.

=== XAI Module Implementation

The XAI framework implements six complementary explanation techniques, each providing unique insights into model decision-making:

*Grad-CAM* (Gradient-weighted Class Activation Mapping) computes per-feature-map importance weights by analyzing gradients flowing into the final convolutional bottleneck layer. This provides class-discriminative localization indicating regions critical for predicting a specific tumor sub-region. For a target class $c$ and feature map $k$, the importance weight $alpha_k^c$ is calculated by global-average-pooling the gradients across the spatial dimensions:

$
  alpha_k^c = 1/Z sum_(x) sum_(y) sum_(z) (partial y^c)/(partial A^k_(x,y,z))
$

where $y^c$ is the spatially-averaged output logit, $A^k_(x,y,z)$ represents the feature map activation, and $Z$ is the total number of spatial elements. The gradient term $(partial y^c)/(partial A^k_(x,y,z))$ mathematically represents the sensitivity or "importance" of the target class score $y^c$ with respect to a specific pixel in the $k$-th feature map. By computing this partial derivative via backpropagation, the model quantifies exactly how much a change in that specific convolutional feature would objectively influence the final class probability. A linear combination of these feature maps is then computed, followed by a ReLU activation to preserve only features that positively support the target class:

$
  L^c_"Grad-CAM" = "ReLU"(sum_k alpha_k^c A^k)
$

The resulting heatmap is upsampled to the original input resolution, providing class-specific but coarse spatial localization.

*Guided Backpropagation* (GBP) adds an additional guidance signal to the standard backpropagation algorithm to highlight areas where significant features are present at a highly detailed, voxel-level scale. When passing gradients backward through a ReLU unit at layer $l$, GBP suppresses the flow of negative gradients by enforcing two conditions: the original forward activation must have been positive, and the incoming backward gradient must also be positive. The guided gradient $R_i^l$ propagated to neuron $i$ is formulated as:

$
  R_i^l = (f_i^l > 0) dot.c (R_i^(l+1) > 0) dot.c R_i^(l+1)
$

where $f_i^l$ is the forward activation and $R_i^(l+1)$ is the incoming gradient from the higher layer. Because both conditions act as binary indicator functions, purely positive signal paths are isolated. This generates a full-resolution (160³) saliency map with sharp edges and fine detail, although it is not inherently class-discriminative.

*Guided Grad-CAM* fuses the complementary strengths of the previous two methods to achieve visualizations that are both high-resolution and class-specific. It masks the high-frequency pixel details produced by GBP using the coarse, class-discriminative localization of Grad-CAM through element-wise multiplication:

$
  L^c_"Guided Grad-CAM" = L^c_"Grad-CAM" dot.c L_"Guided Backpropagation"
$

This process effectively masks out the fine-grained features that are irrelevant to the target class, resulting in high-resolution attributions that accurately highlight the structural features directly responsible for the model's prediction.

*Layer-wise Relevance Propagation* (LRP) operates on a deep Taylor decomposition "conservation principle," attributing a literal relevance score to each input voxel such that their sum equals the final model prediction score. Due to the architectural complexity of SegResNet, which encompasses numerous skip connections and specialized GroupNorm layers, developing manual backward hooks for every layer block is prohibitively complex. Therefore, the implementation utilizes the "Input × Gradient" approximation. For deep neural networks predominantly employing ReLU activations, this proxy is mathematically equivalent to the $epsilon$-LRP rule. The relevance $R_i$ assigned to the $i$-th voxel of the input $x_i$ concerning the spatial model prediction score $f(x)$ is formulated as:

$
  R_i approx x_i dot.c (partial f(x)) / (partial x_i)
$

This approach directly circumvents architectural bottlenecks, yielding an ultra-high resolution (160³) relevance map that quantifies which specific structural features explicitly drove the network's prediction.

*Occlusion Sensitivity* serves as a purely empirical, model-agnostic perturbation technique. It bypasses internal gradient calculations altogether, instead physically testing the 3D CNN's structural reliance by obscuring data. The implementation employs a 3D sliding window defined by a $16 times 16 times 16$ spatial occluding patch that moves across the target volume with a stride of 8 voxels. At each step, the occluded region within the input MRI modalities $x$ is replaced with a baseline value of zero, creating an occluded volume $x_"occluded"$. For a target class $c$, the spatial sensitivity score $S_c$ evaluated at the occluding window's center coordinate $(i, j, k)$ measures the absolute drop in the model's confidence $f_c$:

$
  S_c (i,j,k) = f_c (x) - f_c (x_"occluded")
$

Because the sliding window utilizes a stride of 8 across a 160³ input space, the resulting sensitivity map is initially evaluated at a dimension of 20³. It is subsequently reconstructed and upsampled back to 160³ via trilinear interpolation. This systematic validation step quantifies the clinical necessity of explicit brain regions on an undeniable, empirical level.

*Monte Carlo (MC) Dropout* quantifies confidence rather than providing structural attribution. Because traditional inference is deterministic, this implementation introduces test-time stochasticity by forcing SegResNet's built-in dropout layers (at a probability of $p=0.1$) to remain active during the evaluation phase. Generating $N=20$ stochastic forward passes for an identical volumetric input produces a probability distribution over the predicted segmentations. Given the predicted output probability $p_n$ from the $n$-th forward pass, the stabilized mean prediction $macron(p)$ and its corresponding voxel-wise predictive variance (uncertainty map) $sigma^2$ are evaluated as:

$
  macron(p) = 1/N sum_(n=1)^N p_n
$

$
  sigma^2(x, y, z) = 1/N sum_(n=1)^N (p_n(x,y,z) - macron(p)(x,y,z))^2
$

Where high variance occurs, it highlights "brittle" transition zones or anatomically ambiguous boundaries where the network remains unconfident. Comparing LRP attributions against these uncertainty maps isolates clinically critical structural dependencies that carry significant diagnostic risks.

All XAI outputs are saved as NIfTI volumes organized per patient and per method, ready for loading into 3D Slicer or VR visualization tools. The implementation ensures spatial alignment between saliency maps and ground truth labels by computing metrics inline during generation, using the same preprocessed coordinate space.

== Research Approach

=== Research Design

This study employs applied, quantitative research using secondary medical imaging data. The experimental design evaluates one primary architecture (SegResNet) combined with six XAI techniques, measuring both segmentation performance and explanation validity against ground truth annotations.

The experimental unit is a single BraTS patient case. Independent variables include the choice of XAI method and the tumor sub-region being evaluated (WT, TC, ET). Dependent variables encompass segmentation metrics (Dice, HD95, IoU, Sensitivity, Specificity), XAI metrics (Pointing Game, Coverage, IoU, Weighted Dice), and uncertainty statistics (UAR, boundary ratios).

=== Evaluation Strategy

Model performance and internal reliability are evaluated systematically. To ensure clinical interpretability, the quantitative assessment is divided into mathematical evaluations for structural segmentation accuracy and subsequent evaluations for XAI attribution and uncertainty. 

*1. Segmentation Performance Metrics*
The core segmentation performance of SegResNet is benchmarked against the clinically-annotated test set using the following spatial metrics:

#figure(
  table(
    columns: (22%, 26%, 1fr),
    inset: 5pt,
    align: horizon,
    table.header([*Metric Name*], [*Formula*], [*Description / Function*]),
    [*Dice Score (DSC)*], [$ (2|A inter B|) / (|A| + |B|) $], [Measures volumetric overlap between prediction ($A$) and ground truth ($B$) from 0 to 1. Primary BraTS benchmark.],
    [*HD 95th Percentile*], [$ P_(95) (d(a,B) union d(b,A)) $], [Measures 95th percentile surface boundary error in millimeters. Crucial for assessing margin precision.],
    [*Intersection over Union*], [$ |A inter B| / |A union B| $], [A stricter spatial overlap metric (Jaccard index) penalizing localized false positives more heavily than Dice.],
    [*Sensitivity (Recall)*], [$ "TP" / ("TP" + "FN") $], [The true positive rate. Measures the fraction of actual tumor voxels correctly identified to prevent missed diagnoses.],
    [*Specificity*], [$ "TN" / ("TN" + "FP") $], [The true negative rate. Measures reliability in excluding healthy tissue, directly preventing false alarms.]
  ),
  caption: [Summary of quantitative metrics used for evaluating 3D volumetric segmentation accuracy.],
) <tab:segmentation_metrics>

*2. Explainability and Uncertainty Metrics*
Beyond structural accuracy, the generated gradient-based visual attributions and the MC Dropout variance maps ($sigma^2$) are quantitatively assessed utilizing custom validation metrics:

#figure(
  table(
    columns: (22%, 26%, 1fr),
    inset: 9pt,
    align: horizon,
    table.header([*Metric Name*], [*Formula*], [*Description / Function*]),
    [*Pointing Game (PG)*], [$ cases(1 "if" arg max (L) in G, 0 "otherwise") $], [Binary test returning 1 if the highest activation peak in the saliency map $L$ falls inside the tumor region $G$.],
    [*Saliency Coverage*], [$ (sum_(x in G) L(x)) / (sum_x L(x)) $], [Calculates the percentage of total salient relevance mass concentrated inside the true tumor boundary.],
    [*Saliency IoU*], [$ |L_"thresh" inter G| / |L_"thresh" union G| $], [Evaluates shape overlap between a thresholded Boolean heatmap ($L_"thresh"$) and the ground truth mask ($G$).],
    [*Weighted Dice*], [$ (2 sum L(x) dot.c G(x)) / (sum L(x) + sum G(x)) $], [Treats continuous heatmaps as soft predictive weights, modeling organic relevance without binary thresholds.],
    [*Uncertainty Area Ratio*], [$ (sum_(x in G) sigma^2(x)) / (sum_x sigma^2(x)) $], [Computes the ratio of predictive uncertainty (variance $sigma^2$) trapped inside the pathogenic ROI versus total volume.],
    [*Saliency-Uncertainty*], [$ "Cov"(L, sigma^2) / (sigma_L sigma_(sigma^2)) $], [Pearson correlation isolating "brittle" behaviors where high relevance ($L$) intersects with high variance ($sigma^2$).]
  ),
  caption: [Summary of metrics utilized to evaluate XAI map alignment and model uncertainty.],
) <tab:xai_metrics>

=== Link to Research Questions

This methodological design directly addresses the research questions posed in Chapter 1:

- *Segmentation Accuracy* is measured by traditional metrics (Dice, HD95), answering whether the model achieves clinically acceptable performance.

- *Model Attention Alignment* is assessed by XAI metrics (Pointing Game, Coverage), answering whether the model "looks at the right place for the right reasons."

- *Reliability Characteristics* are quantified by uncertainty metrics, answering where the model is confident versus uncertain.

By combining these evaluation dimensions, this research can argue not only that the model is accurate, but also that its decision-making process is clinically interpretable and trustworthy.

== Challenges and Limitations

This research navigated several primary technical and methodological constraints:

*Computational and Architectural Constraints:* The memory-intensive nature of 3D CNNs mandated a limited batch size (1) and a $160^3$ volumetric ROI, affecting training efficiency (approx. 24 hours for 35 epochs). XAI generation proved exceptionally demanding; Occlusion Sensitivity required approximately 1,000 forward passes per patient. Architecturally, integrating gradient-based XAI methods required strict hook isolation. For instance, simultaneous Grad-CAM and Guided Backpropagation hooks inherently corrupt each other's gradients, demanding sequential, per-method computation workflows.

*Generalizability and Domain Shift:* Methodologically, relying exclusively on an isolated dataset (BraTS 2023) without external cross-institutional validation restricts the clinical generalization of the model. Its segmentation performance may degrade due to domain shift if exposed to MRI scans from alternate hospital scanners, protocols, or distinct neuropathologies.

*XAI Construct Validity:* Finally, while metrics like the Pointing Game and Weighted Dice effectively operationalize interpretability mathematically, they serve merely as academic proxies for clinical utility. Furthermore, the Virtual Reality (VR) application acts primarily as an exploratory visualization target; comprehensive studies involving practicing radiologists remain critical to validating the actual diagnostic efficacy of these immersive explanation maps.

== Ethical Considerations

=== Data Privacy and Governance

This research uses only de-identified BraTS data that has been stripped of all personally identifiable information. Data handling practices include local secure storage with restricted access, no attempt at re-identification, and adherence to dataset usage terms. The BraTS challenge operates under appropriate ethical approvals for secondary use of medical imaging data in research.

=== Clinical Safety and Responsible Use

It is essential to emphasize that this system is a research prototype and has not been approved for clinical decision-making. The intended use is as a decision-support and research tool to explore model behavior, not as an autonomous diagnostic system. Risks include over-reliance on AI outputs and potential misinterpretation of saliency maps or uncertainty maps by non-expert users.

Clear documentation accompanies all outputs explaining what XAI maps represent and what they do not. The system is designed to augment, not replace, clinical expertise.

=== Bias and Generalisation

The BraTS dataset, while large and diverse, may contain biases related to geography, scanner types, and pathology distribution. Limited external validation means performance may change on under-represented populations. Future work should include broader, more diverse datasets to ensure equitable performance across demographic groups.

=== Ethical Use of VR for Medical Imaging

Immersive visualization may amplify visual cues (e.g., bright saliency hotspots) and affect clinician perception. Safeguards for future deployment include clear documentation on XAI map interpretation, training materials for clinicians, and emphasis on combining AI insights with expert judgment rather than relying solely on automated outputs.

