= Methodology

== Introduction

This chapter presents the comprehensive methodology employed in developing an Explainable AI (XAI) framework for brain tumor segmentation using 3D Convolutional Neural Networks (CNNs). The primary objective of this research is to transform an opaque deep learning model into a transparent system that not only achieves high segmentation accuracy but also provides clinically meaningful insights into the decision-making process. This work addresses the critical challenge of removing the "black box" problem in medical imaging AI by implementing six complementary XAI techniques and evaluating them against ground truth annotations.

The research approach follows a hybrid Agile/CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which provides the structural rigor required for medical applications while allowing for iterative model improvements. The framework processes entire brain MRI volumes using SegResNet, a 3D residual encoder-decoder architecture, to produce volumetric tumor predictions across three clinically relevant regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).

This chapter is organized as follows: Section 2.2 describes the overall approach and development methodology; Section 2.3 details the implementation including data preprocessing, model architecture, and XAI methods; Section 2.4 presents the research approach and evaluation strategy; Section 2.5 discusses challenges and limitations; and Section 2.6 addresses ethical considerations.

== Approach

=== Overall Approach and Rationale

This research employs secondary data analysis using the publicly available BraTS 2023 dataset, which comprises approximately 1,250 patient cases with multi-modal MRI scans. The high-level strategy encompasses three interconnected components: (1) training a 3D CNN (SegResNet) to perform volumetric tumor segmentation, (2) integrating multiple XAI techniques to reduce the black-box nature of the model, and (3) exporting segmentation and XAI outputs into a VR-compatible pipeline for immersive clinical evaluation.

The decision to use 3D volumetric processing rather than 2D slice-by-slice analysis is grounded in clinical necessity. Brain tumors are inherently three-dimensional structures, and processing individual slices independently sacrifices critical spatial context that exists between adjacent slices @milletari2016 @cicek2016. Volumetric approaches preserve this contextual information, leading to more clinically realistic segmentations that better capture tumor boundaries and spatial extent.

SegResNet was selected as the primary architecture due to its residual connections, which provide robustness against vanishing gradients and enable effective training of deep networks @he2016deep. The architecture offers an optimal balance between performance and computational complexity, making it suitable for medical imaging applications where training data may be limited and model interpretability is essential.

The rationale for implementing multiple XAI methods stems from their complementary strengths. Gradient-based methods (Grad-CAM, Guided Backpropagation, LRP) provide different perspectives on feature importance, while perturbation-based approaches (Occlusion Sensitivity) offer model-agnostic explanations. Monte Carlo Dropout adds uncertainty quantification, enabling identification of regions where the model lacks confidence—a critical capability for clinical decision support.

=== Agile CRISP-DM Methodology and Its Use

Traditional waterfall development lifecycles are ill-suited for the unpredictable, highly experimental nature of deep learning model training. The complexity of neural network architectures, combined with the need for extensive hyperparameter tuning and iterative refinement, demands a more flexible approach. This research adopts a hybrid Agile/CRISP-DM methodology that combines the structured phases of the Cross-Industry Standard Process for Data Mining with the rapid iteration and evolutionary prototyping principles of Agile development.

#figure(
  image("../Figures/Agile_crisp_dm.svg", width: 75%),
  caption: [The Agile CRISP-DM methodology employed in this research, showing the iterative cycles between the six core phases.],
) <fig:agile-crisp-dm>

@fig:agile-crisp-dm illustrates the six interconnected phases of the methodology, with dashed arrows indicating the iterative nature of the development process. Each phase feeds into the next, but the Agile approach allows for rapid cycling back to earlier phases when issues are discovered or improvements are identified.

*Business and Clinical Understanding.* The clinical motivation for this research stems from two fundamental challenges in neuro-oncology: the black-box nature of 3D CNNs limits clinical trust in AI-assisted diagnoses, and traditional 2D slice-by-slice assessment introduces significant cognitive load for radiologists. The project scope was defined as developing a highly accurate 3D automated segmentation framework integrated with a comprehensive suite of Explainable AI methods, with outputs natively supporting immersive Virtual Reality tools for clinical review.

*Data Understanding and Preparation.* This phase involved extensive exploration of the BraTS 2023 dataset structure, including the four MRI modalities (T1, T1-contrast, T2, FLAIR) and their clinical significance for tumor characterization. A heavily engineered data pipeline was established to handle label conversion (separating WT, TC, and ET regions), foreground cropping to remove empty space around the brain, and intensity normalization to standardize input distributions.

*Modeling through Iterative Development.* An evolutionary prototyping Agile strategy was employed, beginning with a monolithic prototype trained on a small subset of 45 patients. This prototype served to validate the 3D data pipeline, test candidate model architectures (SwinUNETR, AttentionUnet, SegResNet), and verify hardware memory constraints before committing to full-scale training. After successful prototyping, the codebase was refactored into a scalable PyTorch/MONAI framework, with configuration-driven exploration leading to SegResNet selection for the primary training loop.

*Evaluation through Continuous Verification.* Model checkpoints were iteratively refined based on strict validation criteria encompassing Dice similarity, Hausdorff Distance 95th percentile (HD95), Intersection over Union (IoU), Sensitivity, and Specificity. XAI methods were integrated one sprint at a time, with targeted quality metrics (Saliency IoU, Uncertainty Area Ratio, Pointing Game) verifying feature relevance and uncertainty characteristics. Detailed metric definitions and quantitative results are presented in subsequent chapters.

*Deployment at Integration Level.* The final phase generates predictive outputs, heatmaps, and variance arrays as standardized NIfTI files, feeding directly into an immersive 3D Slicer/VR visualization pipeline for clinical review. It is important to note that this remains an experimental research prototype and has not been deployed for active clinical diagnostics.

The Agile CRISP-DM approach ensured an organized, rapid-iteration development loop while maintaining the stringent robustness required for safety-critical medical machine learning components. Regular sprint reviews allowed for course correction, and the modular architecture enabled independent validation of each component.

== Implementation

=== Data Acquisition and Preprocessing

The BraTS 2023 dataset, provided through the MICCAI Brain Tumor Segmentation Challenge, serves as the foundation for this research. The dataset comprises approximately 1,250 patient cases, each containing four co-registered MRI modalities: T1-weighted (T1), T1-weighted with Gadolinium contrast enhancement (T1c), T2-weighted (T2), and T2 Fluid-Attenuated Inversion Recovery (FLAIR). All volumes have been resampled to an isotropic resolution of 1 mm³ and skull-stripped.

The tumor labels in BraTS follow a standard annotation scheme where each voxel is classified as background (0), necrotic tumor core (1), peritumoral edema (2), or enhancing tumor (3). For this research, these labels are converted into three binary channels representing clinically relevant regions: Whole Tumor (WT = necrosis + edema + enhancing), Tumor Core (TC = necrosis + enhancing), and Enhancing Tumor (ET = enhancing only).

The preprocessing pipeline implements several critical transformations to prepare the data for model training:

- *Foreground Cropping:* Removes empty space surrounding the brain, reducing volume dimensions from approximately 240³ to around 160³ voxels per dimension, significantly reducing computational requirements.

- *Channel-First Formatting:* Reshapes volumes to [4, 160, 160, 160] to conform to PyTorch conventions.

- *Per-Channel Z-Score Normalization:* Computes mean and standard deviation only on non-zero voxels to preserve background structure, then normalizes each modality independently.

- *Random Spatial Augmentation:* During training only, applies random flips (across all three axes), 90-degree rotations, intensity scaling, intensity shifting, and Gaussian noise to improve model generalization.

=== SegResNet Model Architecture

#figure(
  image("../Figures/3D_SegResNet.svg", width: 85%),
  caption: [The SegResNet architecture, showing the encoder-decoder structure with residual connections and skip connections.],
) <fig:segresnet>

SegResNet is a 3D encoder-decoder architecture specifically designed for volumetric medical image segmentation. The model processes entire brain MRI volumes without sliding windows during training, enabling global context modeling that captures long-range spatial dependencies.

The encoder progressively downsamples the input volume across four levels using strided convolutions (stride = 2), extracting increasingly abstract features. The filter progression follows: 4 input channels (MRI modalities) → 32 filters (Level 1) → 64 filters (Level 2) → 128 filters (Level 3) → 256 filters (Level 4, bottleneck). Corresponding spatial resolutions are 160³ → 160³ → 80³ → 40³ → 20³.

At the bottleneck (Level 4), the 256-channel feature maps at 20³ resolution capture the entire brain tumor context in a highly compressed form. This layer serves as the target for Grad-CAM analysis, providing a coarse but globally-aware representation of tumor location.

The decoder mirrors the encoder but uses trilinear upsampling to restore spatial resolution. Skip connections directly concatenate features between matching encoder-decoder levels, preserving fine-grained spatial information and improving gradient flow during backpropagation. The final output produces 160³ volumes with 3 channels corresponding to WT, TC, and ET predictions.

Every layer in SegResNet is built from residual blocks (ResBlocks) that learn residual mappings rather than direct transformations. Each ResBlock consists of two 3D convolutions (3³ kernels) with GroupNorm and ReLU activations, plus a skip connection that adds the input to the output. This architecture provides several benefits: direct gradient paths that solve vanishing gradients, interpretable "correction" learning rather than opaque transformations, and training stability for very deep networks.

The model configuration uses 32 initial filters, a dropout probability of 0.1 (which serves dual purposes of regularization and enabling MC Dropout), and DiceFocalLoss as the training objective. The focal component (with γ = 2.0) is critical because Enhancing Tumor comprises only approximately 5% of voxels, making it easily overlooked by standard loss functions. Focal loss upweights misclassifications on ET, effectively balancing class imbalance.

=== Evolutionary Prototyping Phase

Before committing to full-scale training, an evolutionary prototyping phase was conducted using a 45-patient subset randomly sampled from the BraTS training data. This phase served multiple purposes: validating the complex 3D data pipeline, rapidly testing candidate architectures, and verifying hardware memory limits.

The initial prototype was implemented as a monolithic script that combined data loading, model definition, training, and evaluation in a single file. While this approach lacks modularity, it enabled rapid experimentation and quick iteration on architectural choices. After validating core functionality, the prototype was refactored into a modular, configuration-driven PyTorch framework with clear separation between data handling, model definitions, training logic, and evaluation.

Prototype results demonstrated that SegResNet provided the optimal performance-to-complexity ratio among the tested architectures. The 45-patient subset achieved Dice scores of 0.74 (WT), 0.43 (TC), and 0.27 (ET) on a 5-patient test set—promising results given the limited training data and confirming the viability of the approach for full-scale implementation.

=== Implementation Pipeline

*[Space reserved for detailed implementation pipeline description]*

The implementation follows a modular architecture designed for reproducibility and extensibility. The codebase is organized into several key components:

- *Configuration Management:* YAML-based configuration files control all hyperparameters, paths, and experiment settings, enabling reproducible experiments and easy parameter sweeps.

- *Data Pipeline:* The `src/data` module handles dataset scanning, patient splitting, and MONAI transform pipelines for both training and inference.

- *Model Factory:* The `src/models` module provides a unified interface for creating models, loss functions, optimizers, and learning rate schedulers from configuration.

- *Training Framework:* The `src/training` module implements the full training lifecycle with automatic mixed precision, best-model checkpointing, crash recovery, and Weights & Biases logging.

- *XAI Suite:* The `src/xai` module contains implementations for all six explanation techniques, with a consistent interface for generating saliency maps and uncertainty estimates.

- *Evaluation Tools:* The `src/evaluation` module computes both traditional segmentation metrics and novel XAI-specific metrics, exporting results to CSV files for analysis.

The end-to-end workflow proceeds through seven stages: (1) configuration loading and data initialization, (2) model training with validation, (3) test set inference, (4) XAI map generation for all methods, (5) metric computation and CSV export, (6) NIfTI volume export for 3D Slicer, and (7) VR visualization preparation.

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

The dataset was split into 1,000 training cases, 125 validation cases, and approximately 125 test cases. The validation set was used for hyperparameter tuning and early stopping, while the test set was reserved for final evaluation and XAI analysis. All splits were performed deterministically using a fixed random seed (42) to ensure reproducibility.

Training progress was monitored using Weights & Biases (W&B) for experiment tracking, with metrics logged at every iteration. Best model checkpoints were selected based on validation Dice score, and the final model was evaluated on the held-out test set to report unbiased performance estimates.

=== XAI Module Implementation

The XAI framework implements six complementary explanation techniques, each providing unique insights into model decision-making:

*Grad-CAM* (Gradient-weighted Class Activation Mapping) computes per-feature-map importance weights using gradients flowing into the bottleneck layer. For target class $c$ and feature map $k$, importance weights are calculated as:

$ alpha_k^c = (1/Z) sum_d sum_h sum_w (partial y^c)/(partial A^k_(d,h,w)) $

where $y^c$ is the spatially-averaged output logit and $Z$ is the number of spatial elements. The heatmap is computed as $L^c = "ReLU"(sum_k alpha_k^c A^k)$ and upsampled to input resolution. Grad-CAM provides class-discriminative but coarse localization (20³ → 160³).

*Guided Backpropagation* modifies the standard backpropagation algorithm by adding an additional gating mechanism at every ReLU layer during the backward pass. The modified gradient flow zeros out gradients where either the forward activation or the incoming gradient is negative, keeping only purely positive signal paths. This produces full-resolution (160³) saliency maps with sharp edges and fine detail, though it is not class-discriminative.

*Guided Grad-CAM* combines the strengths of both methods through element-wise multiplication: $"Guided_Grad-CAM" = "GBP" " " circle.stroked.tiny " " "upsample"("Grad-CAM")$. The Grad-CAM heatmap acts as a soft spatial mask, restricting the sharp GBP detail to class-relevant regions. This produces high-resolution, class-discriminative explanations that capture both spatial localization and fine-grained feature attribution.

*Layer-wise Relevance Propagation* (LRP) propagates the output prediction score backward through the network, distributing relevance based on activation magnitudes. This research uses the Input × Gradient proxy, which is mathematically equivalent to $epsilon$-LRP for ReLU networks: $R_i = x_i times (partial f(x))/(partial x_i)$. This provides ultra-high resolution relevance maps identifying which input voxels contributed most to the final prediction.

*Occlusion Sensitivity* tests model reliance on specific regions by systematically occluding (masking) spatial patches and recording the resulting drop in prediction confidence. A sliding 16³ window moves across the input with stride 8, and at each position, the masked region is set to baseline (0). The score drop $S_c(i,j,k) = f_c(x) - f_c(x_"occluded")$ indicates region importance. This model-agnostic approach requires approximately 1,000 forward passes per patient but provides robust, gradient-free explanations.

*Monte Carlo Dropout* quantifies model uncertainty by performing multiple stochastic forward passes with dropout layers active at inference time. For $N$ passes, the mean prediction $macron(p) = (1/N) sum_(n=1)^N p_n$ provides a stabilized output, while the variance $sigma^2(d,h,w) = (1/N) sum_(n=1)^N (p_n(d,h,w) - macron(p)(d,h,w))^2$ serves as an uncertainty estimate. High variance indicates regions where the model is inconsistent and therefore less trustworthy.

All XAI outputs are saved as NIfTI volumes organized per patient and per method, ready for loading into 3D Slicer or VR visualization tools. The implementation ensures spatial alignment between saliency maps and ground truth labels by computing metrics inline during generation, using the same preprocessed coordinate space.

== Research Approach

=== Research Design

This study employs applied, quantitative research using secondary medical imaging data. The experimental design evaluates one primary architecture (SegResNet) combined with six XAI techniques, measuring both segmentation performance and explanation validity against ground truth annotations.

The experimental unit is a single BraTS patient case. Independent variables include the choice of XAI method and the tumor sub-region being evaluated (WT, TC, ET). Dependent variables encompass segmentation metrics (Dice, HD95, IoU, Sensitivity, Specificity), XAI metrics (Pointing Game, Coverage, IoU, Weighted Dice), and uncertainty statistics (UAR, boundary ratios).

=== Experimental Setup

The computational environment consists of a high-performance workstation with an NVIDIA RTX 5880 Ada GPU (48 GB VRAM), 128 GB system RAM, and an AMD Ryzen Threadripper processor. The software stack includes Python 3.10, PyTorch 2.0+, MONAI 1.3+, and supporting libraries for medical image I/O and visualization.

Reproducibility measures include fixed random seeds where possible, comprehensive version control using Git, and saving configuration files alongside model checkpoints. All experiments are logged to Weights & Biases with full hyperparameter tracking and artifact storage.

=== Evaluation Strategy

Model performance is evaluated using five standard medical imaging metrics computed on the held-out test set:

- *Dice Score:* Measures volumetric overlap between prediction and ground truth, ranging from 0 (no overlap) to 1 (perfect overlap). This is the primary BraTS benchmark metric.

- *Hausdorff Distance 95th Percentile (HD95):* Measures the worst-case surface boundary error in millimeters, providing sensitivity to boundary accuracy that Dice may miss.

- *Intersection over Union (IoU):* A stricter overlap metric that penalizes false positives more heavily than Dice.

- *Sensitivity (Recall):* The true positive rate, measuring the proportion of actual tumor voxels correctly identified. High sensitivity is critical for clinical safety to avoid missing tumors.

- *Specificity:* The true negative rate, measuring the proportion of healthy tissue correctly excluded. High specificity prevents over-segmentation and false alarms.

XAI quality is evaluated using four metrics that compare saliency maps to ground truth:

- *Pointing Game:* Binary metric indicating whether the voxel with maximum saliency falls inside the tumor region.

- *Saliency Coverage:* The fraction of total saliency mass that falls inside the tumor, measuring attention concentration.

- *Saliency IoU:* Shape overlap between thresholded saliency and ground truth, assessing spatial alignment.

- *Weighted Dice:* A novel metric treating continuous saliency as soft prediction weights, capturing gradual alignment better than binary thresholding.

Uncertainty evaluation includes the Uncertainty Area Ratio (fraction of uncertainty inside tumor), Boundary Uncertainty Ratio (fraction at tumor edges), and correlation between LRP relevance and uncertainty to identify "brittle" decision regions.

=== Link to Research Questions

This methodological design directly addresses the research questions posed in Chapter 1:

- *Segmentation Accuracy* is measured by traditional metrics (Dice, HD95), answering whether the model achieves clinically acceptable performance.

- *Model Attention Alignment* is assessed by XAI metrics (Pointing Game, Coverage), answering whether the model "looks at the right place for the right reasons."

- *Reliability Characteristics* are quantified by uncertainty metrics, answering where the model is confident versus uncertain.

By combining these evaluation dimensions, this research can argue not only that the model is accurate, but also that its decision-making process is clinically interpretable and trustworthy.

== Challenges and Limitations

=== Technical Challenges

Several significant technical challenges were encountered during this research. Memory constraints of 3D CNNs necessitated careful ROI size selection (160³) and batch size limitation (1), impacting training efficiency. Long training times (approximately 24 hours for 35 epochs) and even longer XAI generation times (especially for Occlusion Sensitivity, requiring ~1,000 forward passes per patient) required workflow optimization and selective processing.

Integrating multiple XAI methods into a single coherent 3D framework presented architectural challenges, particularly regarding hook isolation for Guided Grad-CAM. When Grad-CAM and GBP hooks coexist on the same model, they corrupt each other's gradients, necessitating sequential computation per patient.

Workarounds included using Automatic Mixed Precision to reduce memory usage, limiting XAI runs to representative test subsets, and adjusting occlusion parameters (window size, stride) to balance resolution against computational cost.

=== Methodological Limitations

This research has several methodological limitations that should be considered when interpreting results. The use of a single benchmark dataset (BraTS 2023) without external clinical validation limits generalizability to other scanners, populations, or tumor types. Potential domain shift may affect performance if the model is applied to data from different institutions or acquisition protocols.

XAI methods themselves have known shortcomings: Grad-CAM produces blurry heatmaps due to upsampling, GBP is not class-discriminative, and Occlusion Sensitivity is computationally expensive. The XAI metrics serve as proxies for clinical interpretability but may not fully capture how radiologists actually perceive and use explanation maps.

The VR visualization component remains primarily a visualization target rather than an extensively evaluated clinical tool. User studies with practicing radiologists would be necessary to validate the clinical utility of the immersive explanation interface.

=== Threats to Validity

Internal validity may be threatened by preprocessing settings, train/test split randomness, and hyperparameter choices. External validity is limited by the single-dataset design and lack of cross-institutional validation. Construct validity concerns the extent to which metrics like Pointing Game and Weighted Dice truly measure "correct reasoning" by the model—these are operationalizations of interpretability that may not fully capture the clinical concept.

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

== Summary

This chapter has presented a comprehensive, modular methodology for generating and evaluating explainability in 3D brain tumor segmentation models. The key contributions include: (1) implementation of six XAI methods adapted for 3D volumetric data, (2) development of four quantitative XAI evaluation metrics including a novel Weighted Dice metric, (3) integration of MC Dropout for uncertainty quantification, (4) a complete evaluation pipeline bridging model performance with explanation validity, and (5) VR-ready outputs for immersive clinical evaluation.

The Agile CRISP-DM methodology provided an organized framework for iterative development while maintaining the rigor required for medical applications. The implementation is modular and reproducible, controlled primarily via YAML configuration files and scripted entry points. The following chapter presents the quantitative results of applying this framework to the BraTS 2023 test set, demonstrating which XAI methods most effectively explain SegResNet decisions and how well they align with ground truth tumors.


