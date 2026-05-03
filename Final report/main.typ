#import "template.typ": *

#show: project.with(
  title: "Final Year Project Report", // Replace with your title
  author: "Yogi Amitkumar Patel", //
  student_id: "2536809", //
  degree: "Data Science and Artificial Intelligence", //
  supervisor: "Maimoona Sarif", //
  date: datetime(year: 2026, month: 3, day: 16), //

  abstract: [
    Deep learning has achieved remarkable accuracy in medical image analysis, yet the "black-box" nature of 3D Convolutional Neural Networks (CNNs) remains a critical barrier to clinical adoption in neuro-oncology. This dissertation addresses this trust deficit by developing an end-to-end Explainable AI (XAI) framework for 3D brain tumor segmentation. A volumetric SegResNet architecture was trained on the BraTS 2023 dataset to accurately delineate glioma sub regions (Whole Tumor, Tumor Core, and Enhancing Tumor) directly from multi-modal MRI scans. To establish transparency, the framework integrates six voxel-level post-hoc XAI attribution methods including Grad-CAM, Guided Backpropagation, and Monte Carlo Dropout to spatially map the model's internal decision making process.

    Crucially, this research addresses the cognitive limitations of traditional 2D slice-based XAI displays by pioneering the visualization of these 3D saliency volumes within an immersive Virtual Reality (VR) environment. Leveraging 3D Slicer, the pipeline allows clinicians to interactively navigate stereoscopic representations of the patient's anatomy, the predicted segmentation masks, and the localized AI attribution maps. By systematically evaluating the trade-off between segmentation accuracy and model transparency, this project demonstrates how immersive XAI can transform opaque neural networks into accountable, transparent, and clinically viable decision-support systems for surgical planning.
  ],

  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Maimoona Sarif, for her invaluable guidance, expertise, and continuous support throughout the development of this dissertation. Her insightful feedback and encouragement were instrumental in shaping the direction and scope of this research.

    I also extend my thanks to the faculty at the University of East London for providing a rigorous academic environment and the foundational knowledge required to undertake this complex project in Data Science and Artificial Intelligence.

    Finally, I am deeply thankful to my family and friends for their unwavering patience, understanding, and motivation during the demanding months of this final year project. A special thank you goes to a close friend who stood by me and provided invaluable assistance in tackling technical difficulties whenever they arose.
  ],
)

// --- MAIN BODY ---

#include "chapters/01_introduction.typ"

#include "chapters/02_lit_review.typ"

#include "chapters/03_methodology.typ"

#include "chapters/04_results.typ"

#include "chapters/05_evaluation.typ"

#include "chapters/06_conclusion.typ"

// --- REFERENCES ---
#bibliography("works.bib", style: "harvard-cite-them-right")

// --- APPENDICES ---
#show: appendix
#include "chapters/appendix-a.typ"
