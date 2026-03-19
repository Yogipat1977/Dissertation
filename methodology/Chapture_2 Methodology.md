
# Chapter 2 – Methodology

## 2.1 Introduction

- Briefly restate the project aim: 3D brain tumor segmentation using SegResNet, making the model’s decisions explainable (XAI), and preparing the outputs for VR-based visualisation instead of traditional 2D slice inspection.
- State the purpose of this chapter: to describe the methodological choices, including development approach (MeSDLC), implementation details, research design, challenges, and ethical aspects.
- outline the structure of the chapter by listing the main sections:
  - Section 2.2: Approach and MeSDLC methodology
  - Section 2.3: Implementation
  - Section 2.4: Research Approach
  - Section 2.5: Challenges and Limitations
  - Section 2.6: Ethical Considerations

---

## 2.2 Approach
### 2.2.1 Overall Approach and Rationale

- Describe the high-level strategy:
  - discribe about the data acquisition and usage type ( secondary data  analysis).
  - Use a 3D CNN (SegResNet) trained on BraTS 2023 to perform volumetric tumor segmentation (WT, TC, ET).
  - Integrate multiple XAI techniques (Grad-CAM, GBP, Guided Grad-CAM, LRP, Occlusion, MC Dropout) to reduce the black-box nature of the model.
  - Export segmentation and XAI outputs into a VR-compatible pipeline for immersive evaluation.
- Justify key design choices:
  - Why 3D instead of 2D slices (preserving spatial context, more clinically realistic).(cite: V-net fully convolutional, 3D U-Net, Demystifying Brain tumor segmentation)
  - Why SegResNet as the primary architecture (residual connections, robustness, performance vs complexity trade-off).( cite: V-Net, Deep Residual Learning for Image Recognition ( original resdual net paper by He et al. 2015 ))
  - Why multiple XAI methods instead of one (complementary strengths: gradient-based, perturbation-based, uncertainty-based).

### 2.2.2 Agile CRISP-DM Methodology and Its Use

- Introduce the Hybrid Agile/CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:
  - Explain why traditional, strict waterfall lifecycles like MeSDLC are ill-suited for the unpredictable, highly experimental nature of deep learning model training.
  - Describe how an Agile evolutionary prototyping approach—combined with the CRISP-DM data mining standard—provides the structural rigor required for medical applications while allowing for iterative model improvements.
- Map the project history to the CRISP-DM phases:

  - **Business & Clinical Understanding**
    - Clinical motivation: The "black-box" nature of 3D CNNs limits clinical trust, and traditional 2D slice-by-slice assessment introduces high cognitive load.
    - Project scope: Develop a highly accurate 3D automated segmentation framework integrated with a suite of Explainable AI (XAI) methods, outputting volumes natively supported by immersive Virtual Reality tools.

  - **Data Understanding and Preparation**
    - Exploring the BraTS 2023 dataset structure (T1, T1c, T2, FLAIR modalities).
    - Establishing a heavily engineered and cached data pipeline (`src/data`), handling label conversion (WT, TC, ET separation), foreground cropping, and normalization.

  - **Modeling (Iterative Development)**
    - Utilizing an "Evolutionary Prototyping" Agile strategy: The workflow started as a monolithic prototype trained on a small subset (45 patients), which was structurally refactored into a scalable PyTorch/MONAI framework (`src/models`).
    - After initial baselines, configuration-driven exploration led to the selection of SegResNet for the primary training loop over alternatives.

  - **Evaluation (Continuous Verification)**
    - Iterative refinement of model checkpoints based on strict validation criteria (Dice, HD95, IoU, Sensitivity, Specificity).
    - Integrating multiple XAI methods one sprint at a time (Grad-CAM, LRP, Occlusion, MC Dropout) and generating targeted quality metrics (Saliency IoU, UAR, Pointing Game) to verify feature relevance and uncertainty.
    - Mention that detailed metric definitions and results will be provided in later chapters.

  - **Deployment (Integration Level)**
    - Generating the final predictive outputs, heatmaps, and variance arrays as standardized NIfTI files.
    - Feeding these results directly into an immersive 3D Slicer / VR visualization pipeline for clinical review. 
    - Clarifying that this remains an experimental research prototype, not yet deployed for active clinical diagnostics.

- Conclude the section by emphasizing that the Agile CRISP-DM approach ensured an organized, rapid-iteration development loop while maintaining the stringent robustness required for safety-critical medical ML components.

---

## 2.3 Implementation

### 2.3.1 Data acquisition and Preprocessing

- Describe the BraTS 2023 dataset:
  - Mention the source of the dataset (e.g., MICCAI challenge).
  -Mention the type of data (e.g., multi-modal MRI, type of tumor).
  - Number of cases (approximate), four MRI modalities (T1, T1c, T2, FLAIR), and label classes.
- Explain label processing:
  - Conversion of single labelled volume into three channels: WT, TC, ET.
- Detail preprocessing steps:
  - Loading and channel arrangement.
  - Foreground cropping to remove empty space.
  - Intensity normalisation on non-zero voxels per modality.
  - Random cropping to a fixed ROI size (e.g., 160 × 160 × 160).
  - Data augmentation (random flips, rotations, intensity shifts/noise, padding) during training only, and the rationale for each.

### 2.3.2 SegResNet Model Architecture

- Provide an overview of SegResNet:
  - 3D encoder–decoder structure with residual blocks and skip connections.
  - Filter progression (e.g., 32 → 64 → 128 → 256) and corresponding changes in spatial resolution.
- Explain the residual block:
  - Structure (two 3D convolutions with normalisation and ReLU, plus skip connection).
  - Benefits for training stability and representational capacity.
- Summarise key hyperparameters:
  - Input channels (4 MRI modalities), output channels (3 tumor regions).
  - Dropout probability (e.g., 0.1) and its dual role (regularisation and enabling MC Dropout).
  - Loss function: DiceFocalLoss, parameter values (e.g., gamma), and why this loss is well suited to class imbalance and small ET regions.

### 2.3.3 Evolutionary Prototyping Phase

- Describe the initial 45-patient prototype loop (`Prototype_Data/`):
  - Purpose: To validate the complex 3D data pipeline, test candidate model architectures (SwinUNETR, AttentionUnet, SegResNet) rapidly on a small subset, and verify hardware memory limits before full-scale training.
  - Development approach: Creating an initial monolithic pipeline (`prototype.py`) which was subsequently refactored into a modular, configuration-driven PyTorch framework.
  - Outcomes: Proved that SegResNet provided the optimal performance-to-complexity ratio and validated the core segmentation pipeline.
-  Include the table of results from the prototype phase and the prototype pipeline pseudo code and images showing working pipeline.
### 2.3.4 Full-Scale Training Setup

- Describe the final training configuration (post-prototyping):
  - Hardware (e.g., single high-memory GPU).
  - Batch size, epochs, learning rate, weight decay.
  - Optimiser (AdamW) and learning rate scheduler (Cosine Annealing).
  - Use of automatic mixed precision if used.
- Explain the dataset splits:
  - Training, validation, and test splits; how they were created or given.
- Mention logging and checkpoints:
  - Include the different type of dice scores we have used (e.g., DiceFocalLoss, DiceLoss, CrossEntropyLoss).
  - How best-model checkpoints are selected (e.g., based on validation Dice).
  - Where metrics and configuration are stored (e.g., CSV files, experiment logs).

### 2.3.5 XAI Module Implementation

- For each XAI method, give a concise implementation overview:

  - **Grad-CAM**
    - Target convolutional layer (e.g., bottleneck).
    - Computation of feature-map importance via global average pooling of gradients.
    - ReLU and upsampling to input size.
    - Separate heatmaps for WT, TC, and ET.

  - **Guided Backpropagation**
    - Modified ReLU backward pass to block negative gradients.
    - Generation of full-resolution saliency maps highlighting fine details.

  - **Guided Grad-CAM**
    - Element-wise multiplication of GBP and upsampled Grad-CAM.
    - Benefit: class-specific, high-resolution saliency.

  - **LRP (Input × Gradient Proxy)**
    - Rationale for using gradient × input as a practical approximation of ε-LRP.
    - High-resolution relevance maps capturing which voxels contribute most to the final score.

  - **Occlusion Sensitivity**
    - Sliding 3D occluding window, baseline value, and step size.
    - Score drop calculation and conversion to a normalised relevance map.
    - Computational cost and design choices to manage runtime.

  - **MC Dropout (Uncertainty)**
    - Enabling dropout at inference and running multiple stochastic forward passes.
    - Computing mean prediction and voxel-level variance maps.
    - Typical pattern of high uncertainty at tumor boundaries.

- Mention export formats and directory structure:
  - All XAI outputs saved as NIfTI volumes organised per patient and per method, ready for loading into 3D Slicer / VR.

### 2.3.6 End-to-End Processing Pipeline

- Summarise the full workflow step-by-step:
  1. Load configuration and initialise data loaders.
  2. Train SegResNet on BraTS training set, validate, and save best model.
  3. Run inference on the test set to obtain segmentation predictions.
  4. Generate XAI maps for each test case using all selected methods.
  5. Compute segmentation metrics and XAI-specific metrics; save them as CSV files.
  6. Export MRI, ground truth, predictions, saliency maps, and uncertainty maps as NIfTI volumes for external tools (3D Slicer / VR).
- Emphasise that the implementation is modular and reproducible, controlled primarily via YAML configuration files and scripted entry points.

---

## 2.4 Research Approach

### 2.4.1 Research Design

- Position the study:
  - Applied, quantitative research using secondary medical imaging data.
  - Experimental evaluation of one primary architecture (SegResNet) plus several XAI techniques.
- Define your units and variables:
  - Experimental unit: single BraTS patient case.
  - Independent elements: choice of XAI method, tumor sub-region being evaluated (WT, TC, ET).
  - Dependent variables: segmentation metrics, XAI metrics, and uncertainty statistics.

### 2.4.2 Experimental Setup

- Specify the computational environment:
  - Hardware details (GPU model, VRAM, RAM, OS).
  - Software stack (Python version, PyTorch, MONAI, supporting libraries).
- Explain reproducibility measures:
  - Seed setting where possible.
  - Version control for code.
  - Saving configuration files and model checkpoints.

### 2.4.3 Evaluation Strategy

- Describe how you evaluate model performance:
  - Use of Dice, IoU, HD95, Sensitivity, Specificity on WT/TC/ET.
  - How metrics are computed per patient and then aggregated (mean, standard deviation, etc.).
- Describe how you evaluate XAI quality:
  - Metrics like Pointing Game, Saliency Coverage, Saliency IoU, Weighted Dice.
  - What each metric is intended to capture conceptually (location, coverage, shape overlap, soft overlap).
- Describe uncertainty evaluation:
  - Metrics such as Uncertainty Area Ratio, boundary-based ratios, and correlation between LRP relevance and uncertainty.
- Note that detailed numerical results, comparisons, and visual examples are presented in later chapters (Results/Discussion).

### 2.4.4 Link to Research Questions

- Map each component of the evaluation back to your main research questions:
  - Accuracy of segmentation → segmentation metrics.
  - Alignment of model attention with tumor regions → XAI metrics.
  - Reliability/confidence characteristics → uncertainty metrics.
- Explain that this methodological design allows you to argue not only that the model is accurate, but also whether it is “looking at the right place for the right reasons”.

---

## 2.5 Challenges and Limitations

### 2.5.1 Technical Challenges

- Describe major technical difficulties:
  - Memory constraints of 3D CNNs (need for specific ROI size and batch size).
  - Long training and inference times, especially for occlusion-based XAI.
  - Complexity of integrating multiple XAI methods into a single coherent 3D framework.
- Mention any workarounds:
  - Using mixed precision, limiting number of test patients for XAI runs, adjusting occlusion parameters.

### 2.5.2 Methodological Limitations

- Data-related limitations:
  - Use of a single benchmark dataset (BraTS 2023) with no external clinical dataset.
  - Potential domain shift if applied to other scanners/populations.
- XAI-related limitations:
  - Known shortcomings of specific methods (Grad-CAM blur, GBP not strictly class-specific, occlusion cost).
  - XAI metrics as proxies that may not fully capture clinical interpretability without expert feedback.
- VR-related limitations:
  - At this stage, VR is primarily a visualisation target rather than extensively evaluated with clinicians.

### 2.5.3 Threats to Validity

- Internal validity:
  - Possible confounders such as preprocessing settings or train/test split randomness.
- External validity:
  - Generalisability to other datasets, institutions, and anatomical regions.
- Construct validity:
  - The extent to which metrics like Pointing Game and Weighted Dice truly measure “correct reasoning” by the model.

---

## 2.6 Ethical Considerations

### 2.6.1 Data Privacy and Governance

- State that only de-identified BraTS data were used.
- Describe data handling practices:
  - Local secure storage, restricted access, no attempt at re-identification.
- Confirm adherence to dataset usage terms and institutional / course-level ethical requirements.

### 2.6.2 Clinical Safety and Responsible Use

- Emphasise that the system is a research prototype and not approved for clinical decision-making.
- Discuss risks:
  - Over-reliance on AI outputs.
  - Misinterpretation of saliency maps or uncertainty maps by non-expert users.
- Clarify intended use:
  - As a decision-support and research tool to explore model behaviour, not as an autonomous diagnostic system.

### 2.6.3 Bias and Generalisation

- Acknowledge possible biases:
  - Dataset composition (e.g., geography, scanner types, pathology distribution).
  - Limited external validation.
- Explain why these issues matter:
  - Performance may change on under-represented populations.
  - Importance of future work with broader, more diverse datasets.

### 2.6.4 Ethical Use of VR for Medical Imaging

- Reflect on the impact of immersive visualisation:
  - VR may amplify visual cues (e.g., bright saliency hotspots) and affect clinician perception.
- Suggest safeguards for future deployment:
  - Clear documentation on what XAI maps represent and what they do not.
  - Training materials and guidelines for clinicians.
  - Emphasis on combining AI insights with expert judgment.

---
```
