= Conclusion

This dissertation set out to answer a single, clinically urgent question: can a 3D deep learning model for brain tumour segmentation be made simultaneously accurate, transparent, and immersively interpretable? The three-pillar framework, comprising SegResNet segmentation, multi-paradigm XAI, and VR visualisation, demonstrates that the answer is affirmative, within the scope of a research prototype.

== Research Aim Revisited

The primary aim to develop an Explainable AI framework that transforms opaque deep learning predictions into transparent, trustworthy clinical decision-support has been substantively achieved. SegResNet, trained on 1,251 BraTS 2023 patients, exceeds challenge-median performance across all three tumour subregions (WT: 0.923, TC: 0.891, ET: 0.873 Dice), with ET boundary precision of 3.66 mm HD95 sub-voxel accuracy sufficient for radiotherapy margin planning. The model is not only accurate; it is demonstrably grounded. Occlusion Sensitivity, the gold-standard model-agnostic validator, achieves the highest Weighted Dice of all six XAI methods (WT: 0.397, TC: 0.345), confirming that predictions depend on tumour voxels rather than spurious image correlates.

== Key Contributions

Three contributions extend beyond the immediate results. First, the *Weighted Dice* metric — an adaptation of the soft Dice formulation @milletari2016 for XAI saliency evaluation — fills a methodological gap in 3D medical XAI, enabling equitable cross-resolution comparison that hard-thresholded Saliency IoU cannot provide. Second, the *cross-paradigm independence finding* that saliency and MC Dropout uncertainty are statistically orthogonal signals (Pearson $r approx$ 0) establishes that jointly deploying attribution and uncertainty maps provides a complete, non-redundant clinical risk instrumentation layer. Third, the *reproducible VR pipeline*, encapsulating WiVRn wireless streaming, GPU Ray Cast rendering, and five auto-start functions into a single terminal command, constitutes a replicable open-source contribution to immersive medical AI infrastructure.

== Limitations

Three primary gaps bound the work's scope. The absence of a formal clinician user study means the VR pipeline's impact on diagnostic accuracy and cognitive load remains empirically unvalidated. Exclusive reliance on BraTS 2023 without cross-institutional validation limits generalisability to alternate scanner protocols and neuropathology distributions. Furthermore, the 25-patient evaluation cohort data derives from a single deterministic data split, necessitating future k-fold cross-validation to bound morphological variance. XAI-guided model pruning, identified as Objective 6, was not implemented; literature evidence suggests 40–60% parameter reduction is achievable with less than 1% Dice degradation, representing the most impactful unrealised technical objective.

== Future Directions

Four directions emerge directly from these gaps. A formal radiologist evaluation study measuring task completion time, diagnostic confidence, and error rate between flat-panel and VR conditions is the single highest-priority next step. External validation on UCSF-PDGM or MU-Glioma-Post datasets would establish cross-institutional robustness. XAI-guided pruning, using Occlusion Sensitivity attribution to identify low-relevance filters, would translate interpretability insights into model efficiency gains. Finally, extending the VR environment to multi-user shared tumour boards enabling collaborative neuro-oncological review would realise the full clinical deployment vision the framework currently points toward.

== Closing Statement

The black box is not a fixed property of deep learning; it is a design choice. This work demonstrates that with rigorous volumetric XAI, principled uncertainty quantification, and reproducible immersive visualisation, a 3D neuro-oncological AI system can be made accountable at the voxel level. The framework is not production-ready, but it is methodologically complete a foundation from which trustworthy clinical AI in neuro-oncology can be built.
