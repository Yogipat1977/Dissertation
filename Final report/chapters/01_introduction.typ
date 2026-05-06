= Introduction

Every year, approximately 308,000 people worldwide receive a primary brain tumour diagnosis @neri2023. For the neurosurgeon holding the pre-operative scan, the critical question is never simply *whether* a tumour is present it is *where, exactly, does it end?* A margin error of a few millimetres in the wrong direction means either residual malignant tissue or permanent neurological deficit. This dissertation begins at that boundary.

== Clinical Motivation

Magnetic Resonance Imaging (MRI) has become the gold standard for neuro-oncological assessment, offering the sub-millimetre volumetric resolution required to characterise the three clinically distinct tumour subregions defined by the Brain Tumour Segmentation (BraTS) challenge: the Whole Tumour (WT), encompassing all pathological tissue; the Tumour Core (TC), comprising the necrotic and actively growing regions; and the Enhancing Tumour (ET), the contrast-enhancing rim that directly governs radiotherapy dose planning @6975210. Manually delineating these boundaries from multi-modal MRI across T1, T1ce, T2, and FLAIR sequences is a labour-intensive, subjective, and anatomically demanding task. Inter-rater variability in manual segmentation has been documented at up to 28% for ET @neri2023, the subregion of highest clinical consequence. The case for automated, precise volumetric segmentation is unambiguous.

== The Black Box Paradox

Deep learning has answered the accuracy imperative. Modern 3D Convolutional Neural Networks (CNNs), trained on large-scale benchmarks such as BraTS, routinely achieve Dice scores above 0.90 for Whole Tumour performance that matches or exceeds expert radiologists on standardised test sets @bhati2024. Yet this technical success has precipitated a crisis of trust. As these models grow deeper and more accurate, their internal decision-making becomes progressively opaque: a clinician receives a segmentation mask with no insight into why specific voxels were classified as tumour tissue. In a medicolegal environment governed by the GDPR's "right to explanation" @neri2023, this opacity is not merely an academic inconvenience, it is a regulatory and ethical barrier to clinical deployment.

This is the *black box paradox* of medical AI: the more capable the model, the less accountable it becomes. A clinician cannot responsibly act on a prediction they cannot interrogate. Trust, in clinical practice, is not granted to accuracy figures; it is earned through transparency.

== The Research Response

Explainable AI (XAI) has emerged as the discipline that confronts this paradox directly. Post-hoc attribution techniques including gradient-based methods such as Grad-CAM and Guided Backpropagation, decomposition-based methods such as Layer-wise Relevance Propagation, perturbation-based methods such as Occlusion Sensitivity, and stochastic uncertainty methods such as Monte Carlo Dropout each offer a distinct window into the model's spatial reasoning @selvaraju2017 @natekar2020. However, a second barrier immediately presents itself: even when volumetric XAI saliency maps are successfully generated, they are routinely visualised as 2D slice overlays on flat-panel displays. This impoverished representation forces clinicians to mentally reconstruct a fundamentally three-dimensional attribution volume from sequential cross-sections reintroducing precisely the cognitive burden that automation was intended to eliminate @11083598.

The perceptual solution is immersive Virtual Reality (VR). By delivering stereoscopic depth cues and six degrees-of-freedom navigation, VR eliminates the dimensional compression of slice based workflows, allowing a clinician to inhabit the tumour space, rotate attribution volumes, and inspect the spatial relationship between model confidence and anatomical structure directly @zeineldin2022explainability. Yet no existing framework combines high fidelity 3D segmentation, a comprehensive multi paradigm XAI evaluation suite, and a reproducible open source VR visualisation pipeline into a single, end-to-end clinical proof-of-concept.

== Research Aim and Objectives

This dissertation develops and evaluates precisely such a framework. The central aim is:

#quote[To develop an Explainable AI framework that transforms an opaque 3D deep learning segmentation model into a transparent, trustworthy, and immersively interpretable decision-support system for brain tumour neuro-oncology.]

This aim is pursued through seven objectives: establishing voxel-wise XAI localisation mechanisms; analysing the accuracy-transparency trade-off; quantitatively and qualitatively evaluating 3D XAI visualisations in VR; conducting robustness testing on XAI-attributed regions; training SegResNet on BraTS 2023 and integrating six post-hoc XAI techniques; conducting a feasibility analysis of XAI-guided architectural pruning, designing a conceptual pipeline and identifying the technical barriers to implementation; and evaluating limitations with proposed future directions.

The black box is not an intrinsic property of deep learning. It is a design choice and this dissertation is the argument for choosing differently.
