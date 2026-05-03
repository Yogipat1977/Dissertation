= Evaluation

This chapter critically appraises whether the three-pillar results satisfy the research objectives defined in Chapter 3, structured across four dimensions: segmentation clinical sufficiency, XAI attribution validity, uncertainty diagnostic utility, and VR pipeline integrity.

== Segmentation Performance

=== Clinical Threshold Compliance

SegResNet achieves WT Dice 0.923, TC 0.891, and ET 0.873 on the 126 patient BraTS 2023 test set at or above BraTS 2023 leaderboard medians (~0.91, 0.87, 0.84) @natekar2020. All three subregions exceed the 0.80 clinical viability threshold @neri2023. ET HD95 converges to 3.66 mm sub-voxel precision relative to BraTS 1 mm³ resolution representing a qualitative transition from the clinically unusable baseline (75.91 mm) to margin precision sufficient for radiotherapy targeting.

#figure(
  table(
    columns: (40%, 15%, 15%, 15%, 15%),
    inset: 7pt,
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1.5pt),
    [*Architecture / Reference*], [*WT Dice*], [*TC Dice*], [*ET Dice*], [*Year*],
    table.hline(stroke: 0.5pt),
    [3D U-Net @cicek2016], [0.850], [0.750], [0.700], [2016],
    [V-Net @milletari2016], [0.860], [0.780], [0.730], [2016],
    [Baseline Ensemble @natekar2020], [0.890], [0.830], [0.770], [2020],
    [Deep Multimodality @zeineldin2022explainability], [0.900], [0.840], [0.800], [2022],
    [BraTS 2023 Median Target], [0.910], [0.870], [0.840], [2023],
    table.hline(stroke: 0.5pt),
    [*Current Study (SegResNet)*], [*0.923*], [*0.891*], [*0.873*], [*2026*],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Benchmark comparison of segmentation performance (Dice Score) against state-of-the-art architectures and historical baselines from the literature.],
) <tab:benchmark>

=== Data Scaling and Prototype Validity

The asymmetric learning trajectory is diagnostically important: WT reached 94% saturation at 250 patients, while ET required the full 1,251 patient corpus (0.742 → 0.873), confirming spatially fragmented enhancing tissue demands a substantially larger morphological vocabulary. The 24-hour training cost on the RTX 5880 Ada is justified entirely by this ET gain. The evolutionary prototype (WT 0.80, TC 0.55, ET 0.34) validated the data pipeline and architectural choices before committing full GPU resources an evaluable methodological outcome in itself.

== XAI Attribution

=== Localisation and Alignment

GBP and LRP achieve *100% Pointing Game across all five patients and all three tumour regions*, including ET where Grad-CAM records 0%, an architectural failure caused by the $20^3$ bottleneck smoothing sub 5% volume ET lesions below spatial resolution. Occlusion Sensitivity leads Weighted Dice across all regions (WT: 0.422, TC: 0.392, ET: 0.165), constituting the gold-standard model-agnostic proof that SegResNet genuinely relies on tumour voxels. The bottleneck analysis confirms Weighted Dice remains stable ($plus.minus$ 0.02–0.04) across both Grad-CAM resolutions, validating it as the primary comparative metric over the volatile Saliency IoU.

=== The Patient 00291 Failure Case

Patient 00291 is the framework's most diagnostically valuable result. Grad-CAM and Guided Grad-CAM produce zero activation across all regions, GBP, LRP, Occlusion, and MC Dropout continue providing spatially meaningful output. This confirms the failure is method-specific, not a model collapse, and directly motivates the multi-paradigm design: no single method's failure mode should propagate to clinical decisions undetected.

=== Cross-Paradigm Independence

Saliency-Uncertainty Pearson correlation ranges from −0.071 to +0.090 (mean $approx$ 0) across all patients and regions, confirming saliency and uncertainty are *independent, non-redundant signals*. Saliency maps answer *"Is the model attending correctly?"*, MC Dropout variance maps answer *"Where might it be wrong?"*. Their independence means joint deployment provides a complete clinical interpretability layer that neither signal alone can offer.

== Uncertainty Quantification

Patient 01497's Boundary Ratios of 0.906–0.967 confirm that even confident predictions concentrate residual uncertainty at tumour edges, anatomically rational behaviour. Patient 01397's TC UAR of 0.892 and ET Boundary Ratio of 0.998 demonstrate that in complex cases, ambiguity remains spatially grounded rather than diffuse. Patient 00291's positive TC/ET Saliency-Uncertainty Correlation (+0.05 to +0.09) unique across all patients flags that the model relies on features it is not fully confident about, a clinically actionable signal warranting radiologist priority review.

== VR Pipeline

The single-command launch architecture is technically reproducible: all five autostart functions execute without manual intervention, GPU Ray Cast rendering is enforced to prevent CPU fallback hangs, and `watch_for_new_volumes` enables live XAI volume loading mid-session. The pipeline's primary limitation is the absence of a formal user study; whether immersive rendering improves diagnostic accuracy or reduces cognitive load relative to flat-panel viewing remains the most impactful open question for future work.

== Objective Satisfaction

#figure(
  table(
    columns: (5%, 42%, 53%),
    inset: 7pt,
    align: left + horizon,
    stroke: none,
    table.hline(stroke: 1.5pt),
    [*\#*], [*Objective*], [*Outcome*],
    table.hline(stroke: 0.5pt),
    [O1], [3D voxel-wise XAI localisation mechanisms.], [✓ Satisfied. 100% PG (GBP, LRP); Weighted Dice introduced.],
    [O2],
    [Accuracy–transparency trade-off analysis.],
    [✓ Satisfied. Six-method cross-paradigm study; bottleneck analysis completed.],
    [O3],
    [Quantitative and qualitative XAI evaluation in VR.],
    [◑ Partial. XAI metrics fully reported; VR evaluation qualitative only.],
    [O4],
    [3D robustness testing on XAI regions.],
    [◑ Partial. Patient 00291 failure case and saliency-uncertainty decoupling evidenced; systematic perturbation study not conducted.],
    [O5],
    [Train 3D CNN on BraTS and integrate XAI.],
    [✓ Satisfied. 1,251-patient training; all six methods integrated.],
    [O6], [XAI-guided model pruning.], [✗ Not addressed. Identified as future work.],
    [O7],
    [Limitations and future VR extensions.],
    [✓ Satisfied. Gaps identified; multi-user VR and clinician study proposed.],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Research objective satisfaction matrix. O3, O4 partially satisfied; O6 not addressed the primary directions for future work.],
) <tab:objective-mapping>

The framework achieves its central goal: SegResNet is clinically accurate (ET HD95 = 3.66 mm), its decisions are grounded in anatomically correct features (Occlusion W.Dice = 0.422 WT), and its risk profile is fully instrumented through independent saliency-uncertainty signals. The primary gaps, no clinician evaluation, no external dataset validation, no model pruning, bound the scope as a rigorous proof-of-concept rather than a deployable clinical system.
