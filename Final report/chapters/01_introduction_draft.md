# Chapter 1: Introduction

## 1.1 Background and Motivation
Brain tumors represent a critical challenge in neuro-oncology, requiring precise diagnosis and immediate intervention. Among these, gliomas are the most prevalent and aggressive primary brain tumors in adults. Arising from glial cells, gliomas are characterized by rapid growth, diffuse infiltration into healthy tissue, and a highly heterogeneous nature. This heterogeneity means they vary drastically in shape, size, and texture across patients. Because of severe clinical implications, accurate identification of tumor boundaries is paramount for surgical planning. Consequently, the Brain Tumor Segmentation (BraTS) dataset, focusing primarily on gliomas, has emerged as the global benchmark for advancing computational diagnostic tools. Relying solely on manual assessment of these complex structures is increasingly unsustainable, highlighting an urgent need for automated, intelligent systems.

## 1.2 The Role of Medical Image Segmentation
In medical imaging, segmentation is the process of partitioning an image into clinically meaningful regions. For brain tumor diagnosis, this involves classifying every individual "voxel"—the 3D volumetric equivalent of a 2D pixel—within a Magnetic Resonance Imaging (MRI) scan as either healthy tissue or a pathological structure. Historically, segmentation has been performed manually by expert radiologists who trace boundaries slice by slice. While accurate, manual segmentation is labor-intensive, subject to operator variability, and time-consuming. Automated segmentation resolves these inefficiencies by using computational algorithms to delineate tumor margins rapidly and consistently. A successful approach must process the spatial complexities of MRI data, discerning subtle intensity variations across multiple modalities to accurately map the disease's extent.

## 1.3 Integrating 3D CNNs for Prediction and Classification
The advent of Convolutional Neural Networks (CNNs) has revolutionized automated medical image analysis. Early approaches relied on 2D CNNs, processing MRI scans slice by slice, which fundamentally ignores the rich anatomical context present in three-dimensional biological structures. To overcome this, 3D CNNs, such as the SegResNet architecture utilized in this research, are integrated to directly ingest full volumetric MRI data. 

By processing the entire volume simultaneously, a 3D CNN leverages continuous anatomical features across all three spatial axes. The network integrates these features to predict the presence of a tumor by identifying abnormal voxel clusters. Furthermore, it classifies these abnormalities into distinct sub-regions critical for clinical assessment: the necrotic Tumor Core (TC), the Enhancing Tumor (ET), and the broader surrounding edema, which together comprise the Whole Tumor (WT). This direct volumetric interpretation allows the model to capture the nuanced 3D morphology of gliomas that 2D cross-sections consistently miss.

## 1.4 The "Black-Box" Dilemma and Explainable AI (XAI)
Despite achieving unprecedented accuracy, modern 3D CNNs operate as complex "black boxes." They map high-dimensional input volumes to segmentation masks through millions of non-linear parameters, offering clinicians no insight into the reasoning behind their predictions. In high-stakes medical environments, accuracy alone is insufficient; clinicians must be able to trust and verify the AI's diagnostic rationale before making therapeutic decisions. 

To bridge this gap, this project introduces an Explainable AI (XAI) framework. By integrating voxel-based attribution methods, such as 3D Grad-CAM, the system quantifies and visualizes the specific regions of the MRI that most strongly influenced the model's predictions. Furthermore, this research innovates by translating these 3D XAI saliency maps into an immersive Virtual Reality (VR) environment. By overlaying the model's internal reasoning directly onto the 3D patient anatomy, the framework transforms abstract neural network activations into an intuitive, interactive tool designed to enhance clinical expertise.

## 1.5 Aim and Objectives
The central aim of this dissertation is to develop an Explainable AI framework that interprets 3D CNN models for brain tumor segmentation, visualizing these explanations in an immersive VR environment to support clinical decision-making. Specific objectives include:
1. Training a 3D CNN (SegResNet) on the BraTS 2023 dataset to classify glioma sub-regions.
2. Quantifying predictions using a 3D XAI framework adapted for volumetric data.
3. Developing a pipeline integrating 3D Slicer and VR to interactively visualize the MRI, segmentation masks, and XAI volumes.
4. Evaluating the trade-offs between model transparency and segmentation accuracy.

## 1.6 Dissertation Structure
Chapter 2 presents a literature review of deep learning and XAI methodologies. Chapter 3 details the methodological framework, including the SegResNet architecture and XAI integration. Chapter 4 documents the experimental results, while Chapter 5 critically evaluates the framework's clinical applicability. Finally, Chapter 6 concludes the research and suggests future work.
