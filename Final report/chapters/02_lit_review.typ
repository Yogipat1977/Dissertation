= Literature Review

== Introduction

=== Clinical Background and the "Black Box" Challenge

In modern oncology, properly identifying and managing brain tumours continues to be a major issue. Due to its simplicity and excellent volumetric resolution, Magnetic Resonance Imaging (MRI) has become the highest standard for analysis @6975210. However, there is a substantial barrier in the interpretation of this data. At the moment, the process of defining the boundaries of tumors is mostly done by hand. Subjectivity and labour intensity are introduced by this reliance on human interpretation, which may not be sustainable for scaling healthcare solutions @iftikhar2025. Automated technologies that can match or surpass human precision without the corresponding time restrictions are therefore desperately needed in medical applications.

The response to this clinical need has been the rapid adoption of Deep Learning (DL) technologies, particularly 3D Convolutional Neural Networks (CNNs) like the 3D U-Net. These models have demonstrated "superior performance" in handling volumetric data compared to traditional methods @iftikhar2025 @bhati2024. However, the integration of these advanced models has introduced a new, critical issue: the *"black box"* problem.

While these models are highly accurate, their internal decision-making processes are opaque @neri2023. *This creates a paradox in medical AI*: as models become more complex and accurate, they often become less interpretable to the clinicians using them. In a high-stakes environment like neurosurgery, a lack of transparency creates a "trust gap" @wen2025. It is not enough for a model to simply output a segmentation mask; clinicians require the ability to verify why specific voxels were classified as tumour tissue. Therefore, the lack of interpretability is no longer just a technical issue, but a barrier to ethical and safe clinical deployment @neri2023.

=== Research Aims and Scope of Literature Review

To address this disconnect between model accuracy and clinical trust, this literature review aims to critically evaluate the intersection of 3D deep learning, Explainable AI (XAI), and immersive visualisation. While these fields are often studied in isolation, this review argues that a unified framework is necessary to fully illuminate the "black box" of 3D CNNs.

The review is structured to build this argument logically:

*Section 2.2* establishes the technical capabilities of modern 3D CNN architectures for segmentation.

*Section 2.3* critically analyses current XAI methods (such as Grad-CAM and LIME), evaluating their limitations when applied to complex volumetric data.

*Section 2.4* explores the potential of immersive technologies (VR) to present these XAI outputs in a way that is intuitively understandable for clinicians.

*Section 2.5* synthesises these findings to identify the specific research gap this project will address: the lack of an integrated, 3D-visualised explainability pipeline for brain tumour segmentation.

*Section 2.6* explains the investigation's findings and explains how this initiative will fill the identified gap.

== 3D Deep Learning for Medical Brain Tumour Segmentation

=== Evaluation of 3D CNNs for Brain Tumour Segmentation

Over the past decade, the field of medical image analysis has experienced a significant paradigm shift, moving from *"reasoning-based"* systems that relied on manual feature engineering to *"learning-based"* deep learning (DL) approaches @neri2023. In the past, radiomics was used for the automated analysis of brain tumours. To train classifiers like Support Vector Machines (SVMs), domain specialists manually retrieved quantitative features including texture, intensity, and shape. However, this method was labour-intensive, prone to observer bias, and frequently failed to capture the intricate, non-linear spatial relationships present in volumetric data @bhati2024.

The introduction of Convolutional Neural Networks (CNNs) brought an end to this manual paradigm. CNNs prefer an end-to-end approach to extract features automatically by utilizing raw pixel data in contrast to classical machine learning. Initial implementation in neuro-oncology treated Magnetic Resonance Imaging (MRI) as a series of 2D slices. While this approach allowed taking pre-trained networks into consideration (such as VGG or ResNet), it has a significant drawback: it neglects the spatial context along the z-axis (depth).

In neuro-oncology, a tumor is an inherently volumetric entity; it defines the anatomical boundaries with continuation across the sagittal, coronal and axial planes. When a 3D volume is processed as a series of 2D slices it frequently results in *"discontinuous predictions"*, which produce uneven and anatomically impossible 3D models where a lesion is found in one slice but overlooked in the adjacent one @iftikhar2025. To overcome this problem, the field has moved towards 3D CNNs, which simultaneously conduct convolutions on the entire $(x, y, z)$ volume using volumetric kernels (such as $3 times 3 times 3$). This volumetric approach allows the model to learn complex spatial hierarchies and inter-slice connections, which is essential for precise segmentation of brain tumors using large-scale datasets like BraTS.

=== Evolution of Volumetric Segmentation Architectures

To handle the computational complexity of 3D data while conducting segmentation that is pixel-perfect, the research community has standardized the *"encoder-decoder"* architecture. Building upon the success of 2D U-Net, Çiçek et al. @cicek2016 introduced the 3D U-Net architecture, a fully convolutional network for dense volumetric image segmentation. A U-shaped symmetric model consists of two distinct pathways:

+ *The Contracting Path (Encoder)*: This pathway significantly considers feature extraction. It consists of max-pooling layers after recursive blocks of 3D convolutions. As data is processed through the encoder, the spatial resolution decreases while the number of features increases. This path allows the model to capture high-level conceptual data context ("where is the tumour present"), such as differentiating tumour tissues from healthy ones @cicek2016.

+ *The Expanding Path (Decoder)*: To regenerate the segmentation map, the spatial resolution that was lost during encoding needs to be restored. This path reconstructs the feature maps back to their initial input dimensions using up-convolution (transposed convolution) layers, also providing precise localisation ("where is the tumour").

#figure(
  image("../Figures/3d_unet_architecture.jpg", width: 90%),
  caption: [3D U-Net Architecture],
) <fig-3d-unet>

A critical innovation of the 3D U-Net introduced long-range skip connections. Deep neural networks often face difficulty acquiring fine-grained spatial information after applying multiple pooling operations. The 3D U-Net addresses this problem by concatenating high-resolution feature maps straight from the encoder to the appropriate decoder layers. According to Çiçek et al. @cicek2016, this mechanism allows the network to incorporate deep semantic features with shallow details, with precise boundary delineation even when trained with sparse annotations.

Despite its success, the standard U-Net struggles with small lesions. Medical images often contain irrelevant background information such as non-considerable tissue in the brain, while the standard U-Net treats all the pixels in a feature map equally. To overcome this issue, researchers have integrated "attention mechanisms" into the U-Net architecture. As explained by Natekar et al. @natekar2020, the Attention U-Net incorporates *"attention gates"* into the skip connections.

These gates utilise the coarser signal from the gating vector (the deep layer) to filter the activations from the input signal (the shallow layer) before they are merged. This enhances the activations in the Region of Interest (ROI) and suppresses the irrelevant background regions using mathematical equations. This "soft-attention" mechanism improves the model's sensitivity to small, irregular lesions such as glioma subregions without requiring additional guidance or complex cropping preprocessing @natekar2020.

Parallel to the U-Net, Milletari et al. @milletari2016 proposed the V-Net, an optimised model specifically for volumetric medical data. V-Net is completely distinct from other models, incorporating *residual connections* (short-skip connections) blocks. This model resembles the ResNet philosophy, adding the input of a block to its output, allowing gradients to flow more smoothly during backpropagation. Compared to convolutional U-Net, this reduces the vanishing gradient issue and makes it easier to train much deeper 3D networks @milletari2016.

Furthermore, the V-Net research has presented a solution to one of the most enduring challenges in medical segmentation: class imbalance. In a dataset, tumour occupies less than 1% of the total volumetric pixels while the rest of the area is unaffected and healthy background. Standard objective functions like cross-entropy fail in this scenario, resulting in a model achieving accuracy of 99% by predicting "background irrelevant data" for every voxel. To overcome this, Milletari et al. @milletari2016 introduced the *Dice coefficient*, which optimises the model by considering the gap between the ground truth and the expected segmentation:

$ D = frac(2 sum_(i)^(N) p_i g_i, sum_(i)^(N) p_i + sum_(i)^(N) g_i) $

Where the sums run through the $N$ voxels, of the predicted binary volume $p_i in P$ and the ground truth binary volume $g_i in G$. This function can be improved by rewriting it in a differentiable form. As a result, it can be optimised with gradient descent. It can be computed with respect to each predicted voxel, allowing the network to improve the overlap between the predictions and ground truth @milletari2016.

The gradient:

$
  frac(partial D, partial p_j) = 2 lr([frac(g_j, sum_i p_i^2 + sum_i g_i^2) - frac(2 p_j (sum_i p_i g_i), (sum_i p_i^2 + sum_i g_i^2)^2)])
$

== Explainable AI (XAI) for Explainability in Medicine

=== The Need for Explainability in Clinical Practice

The integration of Artificial Intelligence (AI) into medical practice, especially in high-stakes domains like neuro-oncology, has been made more difficult by the opacity of modern Deep Learning (DL) models. Convolutional Neural Networks have demonstrated superior performance in segmentation-like tasks, yet their complex, non-linear decision-making remains inaccessible. This "black box" nature has become a barrier to model adoption, often referred to as the *"trust gap"* @wen2025.

Just model prediction is insufficient in clinical practice; it must incorporate justification that follows medical knowledge. As highlighted by Neri et al. @neri2023, the ethical and legal framework governing medicine, such as the "right to explanation" under GDPR, demands that automated decisions be transparent and contestable. Clinicians must be able to identify the reason behind the model classification of a specific region as tumour-present or not to ensure patient safety @iftikhar2025.

Furthermore, XAI is not just a compliance tool but also a critical mechanism for model debugging and knowledge discovery. Researchers can detect "clever Hans" events, in which a model learns misleading connections (e.g., forecasting tumour based on a specific hospital watermark rather than anatomical pathology) @hou2024. When it comes to brain tumour segmentation, XAI refers to the validation of network focus on biologically relevant sub-regions such as the necrotic core or enhancing tumour @natekar2020.

=== Review of Voxel-Based XAI Techniques

The literature categorizes XAI approaches according to their scope and timing relative to model training. A primary distinction is made between ante-hoc (or "intrinsic") and post-hoc methods.

- *Ante-hoc:* Methods designed to be interpretable by their nature. For example, decision trees or linear regression models, where the relationships between them are explicit @agrawal2025. By integrating interpretable prototype blocks directly into the network architecture, recent advancements in "Self-Explainable AI (S-XAI)" plan to introduce transparency in deep learning @hou2024.

- *Post-hoc:* Methods designed for explainability after the model has been trained. They treat a model as a "black box" and try to improve and approximate its behaviour based on the given inputs and outputs. This method is preferred as standard for analysing medical imaging models like the 3D U-Net architecture to perform segmentation @bhati2024.

*Further XAI methods are divided by their scope:*

- *Global explanations:* Methods that explain the overall logic of the model across the entire dataset, such as feature importance in a model.

- *Local explanations:* Methods that explain the decision-making process of the model for specific inputs, such as the reason behind classifying a voxel region as tumour. Local explanation is important for clinical decision-making as they offer the patient-specific rationale for diagnosis and therapy planning @iftikhar2025.

For 3D medical imaging, the most common XAI techniques are those that can generate saliency maps, spatial representations of "feature importance". The three most prominent approaches are *Grad-CAM*, *LIME* and *SHAP*.

*Gradient-Weighted Class Activation Mapping (Grad-CAM)* has been recognised as the "golden standard" for visualising CNN decisions in medical imaging @bhati2024. Grad-CAM is model-agnostic and can be applied to any CNN model without retraining, compared to earlier methods that required architectural changes.

*Mechanism:* Grad-CAM utilises the gradient of the target concept (e.g., tumour) to produce a saliency map, which highlights the regions of the input image that contribute most to the prediction of the target class. It computes a weighted sum of the feature maps, where the weights represent the importance of each feature channel to the prediction. To eliminate the negative contributions, a Rectified Linear Unit (ReLU) is then applied, highlighting more specific regions that positively support the class of interest @selvaraju2017.

*Application in 3D:* Natekar et al. @natekar2020 successfully applied Grad-CAM to a 3D U-Net for brain tumour segmentation, as it was originally designed for 2D images. Their research showed that 3D Grad-CAM can localise tumours based on the aggregated gradients across volumetric feature maps. However, it has a critical limitation which is the resolution trade-off: because Grad-CAM uses feature maps from deep layers, the resulting heatmaps are often coarse and may bleed into surrounding tissue, reducing the voxel-level precision of the segmentation mask itself @natekar2020.

#pagebreak()

*Local Interpretable Model-Agnostic Explanations (LIME)* takes a different approach. LIME treats the model as a function only, without using internal gradients.

*Mechanism:* LIME modifies the model's predictions and observes the change by masking out random superpixels (segments). It then fits a simple, interpretable model to these disrupted samples to predict the model's behaviour locally around a specific image @ribeiro2016whyitrustyou.

*Evaluation in 3D:* While LIME is highly recommended for 2D image segmentation, its performance in 3D is computationally expensive. According to Agrawal et al. @agrawal2025, LIME requires thousands of forward passes to generate stable explanations. For data like 3D images which contain millions of voxels, generating valid superpixels and running sufficient distributions to achieve significant statistics is often too slow for real-time clinical workflows. Additionally, defining the "superpixels" in a 3D brain volume is quite difficult, frequently leading to unstable explanations that vary between runs.

*SHAP (SHapley Additive exPlanations)* provides a theoretically sound substitute based on cooperative game theory. Each feature (voxel) is given an "importance value" that represents its marginal contributions to the prediction, averaged over all possible combinations of features @ribeiro2016whyitrustyou.

*Strengths and Weaknesses:* SHAP provides the most mathematically consistent explanations, by conforming the summation of each feature's attributes to be equal to the model's output. However, it has extreme computational costs in high-dimensional spaces. "DeepSHAP," an approximation method, has been used to accelerate this process, but research suggests that SHAP values can mislead the explanation in deep networks, as they may violate axioms when features are highly correlated, which is often common in statistically coherent MRI data @mironicolau2025.

=== Comparative Synthesis and the Fidelity Gap

All three methods have their strengths and weaknesses, but Grad-CAM remains the best performing method for 3D medical image segmentation because of its computational efficiency (only one backward pass is needed) and its capacity to make use of the spatial information present in CNN feature maps @natekar2020.

However, a critical "fidelity gap" exists, as existing measures frequently show significant discrepancy between the predicted and ground truth segmentation @mironicolau2025. This discrepancy is exacerbated by the standard practice of analysing 3D attention maps as static 2D slices, a method that hides volumetric coherence and increases cognitive load for clinicians. Therefore, closing the "trust gap" calls for a paradigm shift towards immersive virtual reality, which provides the high-dimensional, intuitive visualisation needed to properly understand these deep volumetric insights, rather than simply algorithmic improvements.

== Immersive Visualisation in Medical Imaging

The interpretation of volumetric medical data has historically been analysed on 2D screens, which causes a lot of cognitive stress. According to Khedir et al. @11083598, traditional slice-by-slice navigation requires clinicians to mentally reconstruct volumetric architecture. This process requires time and is prone to spatial errors when assessing the depth of the tumours relative to prominent brain regions. Even though 3D Convolutional Neural Networks (CNNs) can process this volumetric data, the representation of their outputs remains largely stuck in 2D.

Emerging research suggests that immersive visualisation such as Virtual Reality (VR) and Augmented Reality (AR) can bridge this gap. VR provides *stereoscopic depth perception* and *6-Degrees-of-Freedom (6DoF)* interaction which goes well beyond standard monitors. This allows users to view the "saliency maps" not as flat overlays, which are generated by XAI, but as volumetric clouds suspended in 3D space. The NeuroXAI framework has shown that immersive environments significantly improve clinicians' ability to localise pathology and understand complex neural connectivity compared to standard desktop viewers @zeineldin2022explainability.

The application of VR extends well beyond passive viewing to active surgical planning. Defining tumour boundaries in neuro-oncology is critical. Recent advancements in Augmented Reality (AR) enable surgeons to simulate trajectories and visualise "risk maps" created by deep learning models before entering the operation theatre @Zeineldin2023 @11083598. For situations like complex glioma, where 2D scans may not represent the tumour with key blood arteries, this tool is a crucial resection corridor.

Additionally, the idea of interactive AI is essential to current medical systems. Static heatmaps (like standard Grad-CAM) offer *"take it or leave it"* explanations. VR environments, on the other hand, make *Human-in-the-Loop (HITL)* interaction easier. Immersive interfaces enable clinicians to interrogate the model, e.g., by digitally "erasing" a portion of the input volume to identify prediction changes @article. This interactivity transforms XAI from a static report into a dynamic conversation between the clinician and the AI, allowing the correction of model errors in real time @article.

To implement these immersive systems requires robust technological pipelines. Standard medical formats (DICOM/NIfTI) are not supported by native game engines like Unity. However, platforms like 3D Slicer have filled this bridge. The SlicerVR extension allows volume rendering of medical scans in VR headsets without any data conversion loss. Despite this, integrating XAI with VR still continues to be a major difficulty. Most pipelines concentrate on 2D slice-based visualisation, which fails to convey the volumetric feature importance and limits the clinical utility of XAI in more in-depth scenarios @zeineldin2022explainability. There is a need for distinct workflows that can consume a 3D CNN's attention weights and render them as interactive volumetric objects, as proposed in frameworks like DeepIGN @Zeineldin2023.

== Synthesis and Research Gap

=== The Disconnect Between Model Accuracy and Clinical Utility

After addressing and reviewing the three most significant chapters of this review, a distinct contrast has been established. Models like the 3D U-Net and its variants have achieved tremendous success in segmenting brain tumours @cicek2016 @milletari2016. On the other side, the opacity of the "black box" remains a critical barrier to clinical trust @neri2023. Although methods like Grad-CAM have successfully illuminated the internal logic of these networks @natekar2020, current implementations suffer from three major limitations that this project aims to address.

=== Identified Gaps in Current XAI Frameworks

Having global achievements in 3D XAI based on research such as NeuroXAI, there is still something which is called an *"interactivity gap"*; current systems use static visualisation rather than active user contribution @zeineldin2022explainability. This passivity leads to a serious methodological error that confuses feature relevance with model confidence. As mentioned in the DeepIGN study @Zeineldin2023, a network may exhibit high focal attention on a specific region while displaying high epistemic uncertainty. This could give a confident result but incorrect false positives. As a result, existing 3D XAI methods do not provide the required mechanisms for *interactive correction or uncertainty quantification*, leaving a functional gap where the user cannot distinguish between a confident prediction and a statistical guess, or change the model's focus @inproceedings.

Furthermore, the utility of these volumetric insights is severely constrained by the "cognitive friction" of traditional visualisation mediums. Khedir et al. @11083598 mentioned that analysing 3D volumetric pathology via slice-by-slice 2D navigation increases significant burden to mentally reconstruct spatial relationships from disconnected images. The mismatch between the 3D nature of the data and the 2D nature of the display limits the interpretability of the feature maps, as crucial depth signals are lost in translation. Therefore, a clear research gap exists for a unique framework that not only generates uncertainty-aware explanations but also represents them in an immersive virtual environment, to clarify whether or not engagement significantly lowers cognitive load compared to conventional medical workflows.

== Conclusion

This literature review establishes that while 3D deep learning architectures, particularly the 3D U-Net, have achieved superior success in volumetric tumour segmentation, their deployment is critically hindered by the "black box" paradox @neri2023. Clinical workflows still remain detached even if explainable methods like Grad-CAM provide mathematical solutions for transparency. The reliance on static 2D slice-by-slice visualisation fails to convey the complex spatial resolution of 3D pathologies and is hindered by a significant cognitive load that limits clinical trust @natekar2020.

Consequently, this research focusing on bridging the "trust gap" requires a paradigm shift from passive observation to an interactive environment. By developing a virtual reality environment that incorporates uncertainty quantification with real-time feedback, this project aims to "illuminate the black box" effectively. This innovative method will ensure that high-performance AI is not just accurate but also understandable and useful for medical practice.
