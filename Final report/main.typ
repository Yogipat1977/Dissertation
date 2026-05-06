#import "template.typ": *

#show: project.with(
  title: "Illuminating the Black Box: An Explainable AI Framework for Brain Tumour Segmentation Using Volumetric Data Interpretation in 3D CNNs", // Replace with your title
  author: "Yogi Amitkumar Patel", //
  student_id: "2536809", //
  degree: "Data Science and Artificial Intelligence", //
  supervisor: "Maimoona Sarif", //
  date: datetime(year: 2026, month: 3, day: 16), //

  abstract: [
    Despite achieving state-of-the-art performance in neuro-oncological imaging, deep learning models remain clinically underutilised due to their opacity - a critical barrier when automated segmentation informs high-stakes decisions such as brain tumour resection planning. This dissertation addresses the fundamental tension between predictive accuracy and clinical trustworthiness by developing an integrated framework for explainable and uncertainty-aware volumetric brain tumour segmentation. A 3D SegResNet architecture was trained on the BraTS 2023 dataset (n = 1,251 glioma patients) to segment three clinically salient sub-regions - Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET) - from multimodal MRI, achieving Dice scores of 0.923, 0.891, and 0.873 respectively, with an ET boundary precision of 3.66 mm HD95, surpassing BraTS 2023 challenge medians across all sub-regions.

    The principal contribution lies in a systematic, multi-modal explainability framework that renders the model's decision-making process auditable through six complementary post-hoc attribution methods: Grad-CAM, Guided Backpropagation, Guided Grad-CAM, Input × Gradient, Occlusion Sensitivity, and Monte Carlo Dropout. Saliency fidelity was quantitatively evaluated via Pointing Game accuracy, Saliency Coverage, and Saliency IoU, ensuring methodological rigour beyond qualitative impression. A key empirical finding, derived from analysis across 25 held-out patients, is that saliency and uncertainty constitute statistically independent signals (Pearson r ≈ 0), establishing their non-redundant complementarity for clinical decision support. To overcome the spatial information loss inherent in conventional 2D displays, 3D saliency and uncertainty volumes were streamed into an immersive Virtual Reality environment, rendered via GPU-accelerated ray-casting and delivered wirelessly to a Meta Quest 3 headset through 3D Slicer, enabling stereoscopic, six-degrees-of-freedom navigation within anatomical context.

    Current limitations include the absence of formal clinician user studies, reliance on a single data split without cross-validation, and partial implementation of XAI-guided model pruning. Future work will prioritise within-subjects radiologist evaluation measuring diagnostic accuracy and NASA-TLX cognitive load across flat-panel versus VR workflows, multi-institutional external validation, and completion of the feedback-driven pruning loop. Ultimately, this work demonstrates that the "black box" nature of deep learning is not an inherent epistemological limit but a tractable engineering problem; through rigorous volumetric attribution, principled uncertainty quantification, and immersive reproducible visualisation, we establish a pathway toward clinically trustworthy neuro-oncological AI.
  ],

  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Maimoona Sarif, for her invaluable guidance, expertise, and continuous support throughout the development of this dissertation. Her insightful feedback and encouragement were instrumental in shaping both the direction and scope of this research.

    I am deeply grateful to my close friend, Jayrup Nakawala, whose technical assistance was indispensable to this work. His support in configuring computational infrastructure and facilitating access to remote server resources for model training enabled the full-scale experiments that underpin the results presented herein.

    I also extend my heartfelt thanks to Maged Abdelmonem for generously providing access to a VR headset, which was essential to the immersive visualisation component of this framework and directly contributed to achieving the project's objectives.


  ],
)

// --- MAIN BODY --

#include "chapters/01_introduction.typ"

#include "chapters/02_lit_review.typ"

#include "chapters/03_methodology.typ"

#include "chapters/04_results.typ"

#include "chapters/05_evaluation.typ"

#include "chapters/06_conclusion.typ"

// --- REFERENCES ---
#set cite(style: "harvard-cite-them-right")

#show bibliography: it => {
  let b = counter("bib-internal")
  b.update(0)
  show block: it => {
    if type(it.body) == content {
      b.step()
      grid(
        columns: (2em, 1fr),
        gutter: 1em,
        align(right)[#context b.display().], it.body,
      )
    } else {
      it
    }
  }
  it
}

#bibliography("works.bib", style: "harvard-cite-them-right")

// --- APPENDICES ---
#show: appendix
#include "chapters/appendix-a.typ"
