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

It should be noted that all reported metrics derive from a single deterministic train/validation/test split (seed=42). The per-patient standard deviations (WT: 0.079, TC: 0.179, ET: 0.159) indicate substantial morphological variance. While k-fold cross-validation would provide more robust performance estimates, the computational cost of 3D CNN training (~24 hours per run) made multiple splits infeasible within the project timeline. Future work should employ at least 5-fold cross-validation to characterise split-dependent variance.

== XAI Attribution

=== Localisation and Alignment

Across the 25-patient evaluation cohort, GBP and IxG achieve strong Pointing Game accuracy (64–96%), significantly outperforming Grad-CAM on Enhancing Tumour (which scores only 33.3%). This confirms that the $20^3$ bottleneck smoothing causes severe spatial fidelity loss for sub-5% volume ET lesions. Occlusion Sensitivity leads Weighted Dice for WT (0.397) and TC (0.345), while Grad-CAM narrowly leads ET (0.250 vs Occlusion’s 0.233), constituting the gold-standard model-agnostic proof that SegResNet genuinely relies on tumour voxels. The bottleneck analysis confirms Weighted Dice remains stable ($plus.minus$ 0.02–0.04) across both Grad-CAM resolutions, validating it as the primary comparative metric over the volatile Saliency IoU.

=== The Patient 00291 Failure Case

Patient 00291 is the framework's most diagnostically valuable result. Grad-CAM and Guided Grad-CAM produce zero activation across all regions, GBP, IxG, Occlusion, and MC Dropout continue providing spatially meaningful output. This confirms the failure is method-specific, not a model collapse, and directly motivates the multi-paradigm design: no single method's failure mode should propagate to clinical decisions undetected.

=== Cross-Paradigm Independence

Saliency-Uncertainty Pearson correlation averages near zero across the expanded 23-patient cohort, confirming saliency and uncertainty are *independent, non-redundant signals*. Saliency maps answer *"Is the model attending correctly?"*, MC Dropout variance maps answer *"Where might it be wrong?"*. Their independence means joint deployment provides a complete clinical interpretability layer that neither signal alone can offer.

== Uncertainty Quantification

Patient 01497's Boundary Ratios of 0.906–0.967 confirm that even confident predictions concentrate residual uncertainty at tumour edges, anatomically rational behaviour. Patient 01397's TC UAR of 0.892 and ET Boundary Ratio of 0.998 demonstrate that in complex cases, ambiguity remains spatially grounded rather than diffuse. Patient 00291's positive TC/ET Saliency-Uncertainty Correlation (+0.05 to +0.09) unique across all patients flags that the model relies on features it is not fully confident about, a clinically actionable signal warranting radiologist priority review.

== VR Pipeline

The single-command launch architecture is technically reproducible: all five autostart functions execute without manual intervention, GPU Ray Cast rendering is enforced to prevent CPU fallback hangs, and `watch_for_new_volumes` enables live XAI volume loading mid-session.

=== Heuristic Usability Evaluation

To formally appraise the VR pipeline without a full user study, Nielsen's 10 Heuristics were applied.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 7pt,
    align: left + horizon,
    stroke: none,
    table.hline(stroke: 1.5pt),
    [*Heuristic*], [*Assessment*], [*Score (0-4)*],
    table.hline(stroke: 0.5pt),
    [Visibility of system status], [Volume rendering confirms loaded state; no progress indicator during NIfTI loading.], [2],
    [Match between system and real world], [Anatomically accurate NIfTI rendering; BraTS-standard colour coding.], [3],
    [User control and freedom], [6-DoF navigation; volume toggle; no undo functionality.], [2],
    [Consistency and standards], [Follows 3D Slicer @fedorov20123d conventions; standard OpenXR controller bindings.], [3],
    [Error prevention], [Auto-rendering via `watch_for_new_volumes`; no guard for corrupt NIfTI files.], [2],
    [Recognition over recall], [Volumes listed in Slicer @fedorov20123d panel; ray-march cursor provides spatial context.], [3],
    [Flexibility and efficiency], [Single-command launch; limited in-VR shortcuts.], [2],
    [Aesthetic and minimal design], [Clean volumetric rendering; no UI clutter in VR space.], [3],
    [Help users recover from errors], [No in-VR error messages; terminal-only diagnostics.], [1],
    [Help and documentation], [Launch script documented; no in-VR guidance.], [1],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Heuristic usability evaluation of the VR XAI pipeline. The pipeline achieves adequate scores for core visualisation functionality (Mean: 2.2/4.0) but identifies clear gaps in error handling and user guidance.],
)

=== Proposed Clinician Evaluation Protocol

The pipeline's primary limitation is the absence of a formal user study. Whether immersive rendering improves diagnostic accuracy or reduces cognitive load relative to flat-panel viewing remains the most impactful open question. Future work will deploy a within-subjects study recruiting 12-15 radiologists to evaluate XAI outputs via standard flat-panel versus immersive VR. Metrics will include task completion time for identifying false positives, boundary alignment Dice, and NASA-TLX @hart1988nasa scores for cognitive load assessment.

== Objective 6: XAI-Guided Pruning — Architectural Design

Objective 6 proposed using XAI attribution maps to guide model pruning, reducing SegResNet's parameter count while preserving segmentation accuracy. While full implementation was not completed within the project timeline, the architectural design was developed and is presented here for reproducibility.

=== Why True LRP Was Infeasible

Layer-wise Relevance Propagation (LRP) requires custom backward hooks at every layer to decompose relevance according to the $epsilon$-rule or $alpha beta$-rule. SegResNet's architecture presents three obstacles: (1) GroupNorm layers lack canonical LRP decomposition rules, unlike BatchNorm which admits closed-form propagation; (2) residual skip connections create forking relevance paths that require careful re-merging at each block; and (3) the encoder-decoder skip connections double the number of relevance streams. Implementing manual backward hooks for all 48+ layers was judged prohibitively complex for a single-developer project, motivating the Input × Gradient proxy adopted in this work.

=== Proposed VR-Interactive Pruning Loop

The proposed pipeline integrates the VR visualisation environment with XAI-guided filter ranking:

+ A clinician in VR identifies a false-positive region via the ray-march pointer in SlicerVR @fedorov20123d.
+ SlicerVR @fedorov20123d exports the spatial coordinates via OpenIGTLink to a Python backend running the SegResNet model.
+ Targeted backpropagation computes gradients _only_ for the clinician-defined error region, producing a spatially focused attribution map.
+ Taylor first-order pruning ranks convolutional filters by $|W times nabla L|$ computed on the error region, identifying filters that contribute most to the false positive.
+ The lowest-relevance filters are pruned, the model is fine-tuned for 2–3 epochs, and the updated predictions are re-exported to VR for validation.

This loop would enable clinician-in-the-loop model refinement, where domain expertise directly informs architectural decisions. Literature evidence suggests 40–60% parameter reduction is achievable with less than 1% Dice degradation in medical segmentation networks @myronenko2019, making this a high-impact direction for future work.

=== Why It Was Not Implemented

The critical dependency is a real-time SlicerVR @fedorov20123d ↔ Python bridge via OpenIGTLink. While OpenIGTLink is supported by 3D Slicer, configuring bidirectional coordinate streaming between VR controllers and a PyTorch inference backend required integration work beyond the project timeline. The architectural design above is provided as a concrete, implementable specification for future work.

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
    [O1], [3D voxel-wise XAI localisation mechanisms.], [✓ Satisfied. Strong PG (64–96% for GBP/IxG); Weighted Dice introduced.],
    [O2],
    [Accuracy–transparency trade-off analysis.],
    [✓ Satisfied. Six-method cross-paradigm study; bottleneck analysis completed.],
    [O3],
    [Quantitative and qualitative XAI evaluation in VR.],
    [◑ Partial. XAI metrics fully reported; VR evaluation limited to Nielsen's heuristics.],
    [O4],
    [3D robustness testing on XAI regions.],
    [◑ Partial. Patient 00291 failure case and saliency-uncertainty decoupling evidenced; systematic perturbation study not conducted.],
    [O5],
    [Train 3D CNN on BraTS and integrate XAI.],
    [✓ Satisfied. 1,251-patient training; all six methods integrated.],
    [O6], [XAI-guided model pruning.], [◑ Partial. Architectural design proposed; implementation deferred due to OpenIGTLink integration complexity.],
    [O7],
    [Limitations and future VR extensions.],
    [✓ Satisfied. Gaps identified; multi-user VR and clinician study proposed.],
    table.hline(stroke: 1.5pt),
  ),
  caption: [Research objective satisfaction matrix. O3, O4 partially satisfied; O6 not addressed the primary directions for future work.],
) <tab:objective-mapping>

The framework achieves its central goal: SegResNet is clinically accurate (ET HD95 = 3.66 mm), its decisions are grounded in anatomically correct features (Occlusion W.Dice = 0.397 WT), and its risk profile is fully instrumented through independent saliency-uncertainty signals. The primary gaps — no formal clinician user study, reliance on a single data split, and no XAI-guided model pruning — bound the scope as a rigorous proof-of-concept rather than a deployable clinical system.
