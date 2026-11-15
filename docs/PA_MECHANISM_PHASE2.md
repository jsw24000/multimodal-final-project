# PEARL attention mechanism, Phase 2

## Scope and controls

This continuation starts from the committed Phase 1 repository state `86edd05d70bf90b5363c8b246896d70028bddeb5`. New results use the same CLIP ViT-B/16 checkpoint (SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`), Reduced final visual block, text prototypes, validation pipelines, and PEARL postprocessing. VOC21 and VOC20 use their released protocols; ADE20K retains sliding-window inference. TLP is off unless explicitly identified. Phase 1 files and metrics remain unchanged.

The two aligners share `M=K_c^T Q_c`: released five-step Newton–Schulz (NS-5) and exact `U V^T` from SVD. The identity `softmax(Q(KR)^T/sqrt(d)) = softmax((QQ^T+QE^T)/sqrt(d))` holds for either R, because its omitted term is constant along each row. It does not require orthogonality. [The numerical test](../experiments/pa_mechanism/phase2/test_solvers.py) passed gamma endpoints, row-constant logits, value aggregation, SVD orthogonality, finite NS iterations, and row sums.

## Full-validation controlled performance

All numbers below are dataset mIoU (%) from complete validation splits. VOC21 and VOC20 have 1,449 images each; ADE20K has 2,000. The methods are swaps in the same Reduced CLIP implementation. The SCLIP-style rows do not claim to reproduce the separate official SCLIP codebase.

| Attention, TLP off | VOC21 | VOC20 | ADE20K |
|---|---:|---:|---:|
| Reduced QK | 35.28 | 78.98 | 11.60 |
| QQ | 57.82 | 84.64 | 17.32 |
| KK | 58.91 | 80.30 | 17.41 |
| QQ+KK sum | 58.91 | 83.06 | 17.47 |
| QQ+KK mean | 59.30 | 83.48 | 17.63 |
| Pure PA NS-5 | 59.00 | 84.72 | 17.75 |
| Pure PA SVD | 59.08 | 84.62 | 17.80 |
| PA+KK NS-5 | 59.19 | 84.95 | 17.82 |
| PA+KK SVD | 58.94 | 84.43 | 17.40 |

The [machine-readable combined table](../results/pa_mechanism_phase2_summary.csv) also records TLP-on rows, image counts, mean accuracy, and pixel accuracy. The [run manifest](../results/pa_mechanism_phase2_manifest.json) lists actual per-variant completion from raw records. The 17.82 ADE20K PA+KK NS-5 score matches the historical PA-only result to two decimals, and 19.36 with TLP matches historical Full PEARL.

Phase 1 full VOC21 results: QK 35.28, QQ 57.82, KK 58.91, SCLIP-style sum 58.91, SCLIP-style mean 59.30, pure PA NS-5 59.00, PA+KK NS-5 59.19 mIoU. Phase 1 full VOC20: QK 78.98, QQ 84.64, pure PA NS-5 84.72, PA+KK NS-5 84.95. The QQ+KK variants use the same Reduced CLIP encoder and are distinct from the official SCLIP repository protocol.

The historical **Vanilla CLIP** VOC21 result is 15.16 and differs from Reduced QK's 35.28 because the last residual/FFN architecture also changes. The Attention mechanism comparisons below all hold the Reduced architecture fixed; Vanilla-to-Reduced is reported only as an auxiliary architecture observation.

On the full 1,449-image VOC21 split, **pure PA SVD scores 59.08** and **PA+KK SVD scores 58.94**. The pure PA SVD gamma series at 0, .25, .5, .75, 1, 1.25 is **57.82, 58.24, 58.59, 58.86, 59.08, 59.24** mIoU. NS-5's corresponding values are 57.82, 58.22, 58.53, 58.78, 59.00, 59.14. The residual contributes under both solvers; NS-5 non-orthogonality is therefore not necessary for this VOC21 effect. Neither pure PA version surpasses the SCLIP-style mean's 59.30, while its output scale differs and requires controlled interpretation.

On the full 1,449-image VOC20 split, pure PA SVD scores **84.62**, PA+KK SVD **84.43**, versus QQ **84.64**, pure PA NS-5 **84.72**, and PA+KK NS-5 **84.95**. SVD gamma at 0/.25/.5/.75/1/1.25 scores **84.64/84.74/84.76/84.72/84.62/84.45**. Thus the residual at full SVD strength does not add a VOC20 gain, and the released NS-5 PA+KK has a larger advantage than its exact-SVD counterpart. This dataset dependence prevents a simple universal statement about the solver.

The remaining full VOC20 self-correlation controls are now available: KK **80.30**, QQ+KK sum **83.06**, QQ+KK mean **83.48** mIoU. Unlike VOC21, neither QQ+KK version reaches QQ's 84.64 or PA+KK NS-5's 84.95 on VOC20. This makes a blanket "QQ+KK replaces PA" conclusion incorrect even before adding TLP.

ADE20K's complete TLP-off SVD gamma series at 0/.25/.5/.75/1/1.25 is **17.32/17.49/17.62/17.72/17.80/17.86**; NS-5 gives **17.32/17.51/17.63/17.71/17.75/17.77**. VOC20's newly completed NS-5 series is **84.64/84.86/84.91/84.85/84.72/84.51**. The [three-dataset gamma curves](../results/figures/pa_mechanism_phase2/gamma_three_datasets_full.png) plot only measured full-validation values.

From 1,000 paired image bootstrap resamples of the Phase 1 confusion matrices, NS-5 pure PA minus QQ is +1.18 mIoU points (95% percentile interval +0.97 to +1.38) on VOC21, and +0.08 (−0.19 to +0.35) on VOC20. The resampling aggregates confusion matrices before computing dataset mIoU. These are within-split uncertainty intervals, not independent dataset replication.

Direct solver pairing gives pure-PA SVD minus NS-5 **+0.08 [−0.07,+0.22]** on VOC21 and **−0.10 [−0.29,+0.09]** on VOC20. For PA+KK, SVD minus NS-5 is **−0.25 [−0.40,−0.11]** on VOC21 and **−0.53 [−0.74,−0.33]** on VOC20. The NS-5 solver therefore matters more in combination with the KK term than for pure PA on these two datasets; this is an empirical interaction, not yet an identified spectral mechanism.

[The paired per-image NS-5/SVD scatter and difference distributions](../results/figures/pa_mechanism_phase2/solver_image_pairs.png) display all 1,449 images for each VOC protocol. Its image-level IoU is computed over classes present in that image's GT or prediction and is a diagnostic, not the official pooled dataset mIoU used in the tables and bootstrap.

## Original PEARL TLP on all full splits

The released TLP module was enabled without code changes. Every row evaluates the complete 1,449-image split using the same Reduced encoder and text protocol.

| Attention + TLP | VOC21 mIoU | VOC20 mIoU | ADE20K mIoU |
|---|---:|---:|---:|
| QQ | 61.70 | 86.40 | 18.45 |
| KK | 64.03 | 83.88 | 18.99 |
| QQ+KK sum | 63.67 | 85.60 | 18.81 |
| QQ+KK mean | 63.99 | 85.97 | 18.99 |
| Pure PA NS-5 | 63.21 | 86.44 | 18.69 |
| PA+KK NS-5 | 64.07 | 86.90 | 19.36 |
| PA+KK SVD | 64.16 | 86.73 | 19.20 |

The PA+KK NS-5 rows match the saved Full PEARL baselines to two decimal places. On VOC21, KK and QQ+KK mean nearly match PA+KK NS-5. On VOC20, PA+KK NS-5 leads QQ by about 0.50 points and QQ+KK mean by about 0.93 points. The differing rankings already argue against a universal statement that PA is either required or replaceable by one fixed self-correlation rule.

Paired image bootstrap sharpens that comparison. On VOC21 + TLP, PA+KK NS-5 minus KK is **+0.04 [−0.53,+0.64]** points, and minus QQ+KK mean **+0.08 [−0.24,+0.41]**. The interval spans gains and losses of practical size, so these data do not establish an advantage over those simpler alternatives. On VOC20 + TLP, PA+KK NS-5 minus QQ+KK mean is **+0.93 [+0.65,+1.24]**. VOC20 and VOC21 contain the same source images under different label protocols, so they are not independent replications.

The original TLP changes the mIoU relative to the same TLP-off attention on all three datasets. For QQ, the gains are **+3.88/+1.76/+1.14** on VOC21/VOC20/ADE20K; for QQ+KK mean **+4.69/+2.49/+1.36**; for PA+KK NS-5 **+4.87/+1.95/+1.54**. These are full-split paired configurations, not TLP architecture changes.

ADE20K uses released 224/112 sliding-window inference throughout. PA+KK NS-5 minus QQ+KK mean is **+0.37 [+0.23,+0.56]** under 1,000 paired image bootstrap resamples. On VOC21 that difference is only +0.08 [−0.24,+0.41]. TLP changes method rankings: pure PA NS-5 is weaker than QQ+KK mean on VOC21 and ADE20K, while the original PA+KK combination is stronger on VOC20 and ADE20K.

### Inference cost

CUDA-synchronized, warmed timings on the first VOC21 image (100 attention repetitions and 20 complete-image repetitions, no Phase-2 prefix cache) give the following measurements on the RTX PRO 6000 Blackwell. Attention timing includes aggregation with V; full-image timing includes preprocessing already prepared by the dataset, model prediction and postprocessing. The same image and process are used for each row, so these are local engineering measurements rather than a dataset-wide latency distribution.

| Variant | Attention + V, ms | Full image TLP off, ms | Full image TLP on, ms |
|---|---:|---:|---:|
| Reduced QK | 0.13 | 7.33 | 16.17 |
| QQ | 0.10 | 7.31 | 16.10 |
| KK | 0.10 | 7.41 | 16.25 |
| QQ+KK mean | 0.12 | 7.32 | 16.12 |
| Pure PA NS-5 | 0.77 | 8.09 | 16.90 |
| Pure PA SVD | 11.96 | 19.16 | 25.29 |
| PA+KK NS-5 | 0.78 | 7.87 | 13.19 |
| PA+KK SVD | 11.87 | 17.73 | 24.52 |

NS-5 performs a per-image/per-head matrix solve and costs roughly 0.7 ms above QQ for this attention operation; exact SVD costs roughly 12 ms. TLP raises the complete-image time by about 8.8 ms for QQ in this run. The peak allocated memory reported by PyTorch is about 848 MiB across variants and is dominated by the common model; these measurements do not resolve a method-specific incremental peak. Raw timing JSON is in `/data/Multimodal-final-project/outputs/pa_mechanism/phase2/metrics/benchmark_voc21_tlp_{off,on}.json`.

## GT-free entropy matching

As a diagnostic control, QQ logits are scaled by a per-image, per-head temperature chosen by bisection so their *mean row entropy* matches that of pure PA attention from the specified solver. The calculation uses Q/K only, never GT. It does compute PA to establish the entropy target, so it is a causal diagnostic of scale versus relation structure, **not** a cheaper deployable alternative. The numerical test checks that mean entropy differs by less than 0.01 nats on random inputs.

On VOC21, NS-5-matched QQ scores **57.67** versus NS-5 pure PA **59.00**; paired difference **+1.32 [1.10,1.54]** points. SVD-matched QQ scores **57.93** versus SVD pure PA **59.08**; difference **+1.15 [0.98,1.31]**. On VOC20, NS-5-matched QQ scores **84.75** versus NS-5 pure PA **84.72**, a difference of −0.03 [−0.29,+0.22]. ADE20K NS-5-matched QQ scores **17.23** against pure PA **17.75**, a paired difference of **+0.52 [+0.39,+0.63]**; SVD-matched QQ scores **17.33** against pure PA **17.80**, a difference of **+0.47 [+0.37,+0.56]**. Thus VOC21 and ADE20K PA gains are not explained merely by matching mean entropy, whereas VOC20 remains compatible with a scale-only explanation at this resolution. Matching only a mean entropy is not equivalent to matching the full row-wise entropy distribution.

## Fixed-case visual and feature diagnostics

[The fixed first image](../results/figures/pa_mechanism_phase2/2007_000033_segmentation.png), [a PA regression](../results/figures/pa_mechanism_phase2/2008_006143_segmentation.png), and [a PA improvement](../results/figures/pa_mechanism_phase2/2008_008252_segmentation.png) use true VOC21 RGB/GT and one checkpoint. [The bird query attention panel](../results/figures/pa_mechanism_phase2/2008_006143_foreground_attention.png) uses the same head, Query and log-scale color range across QK/QQ/KK/NS-5/SVD. The source labels and full-resolution PNG/SVG live under `/data/Multimodal-final-project/outputs/pa_mechanism/phase2/figures/`.

For these three images, [shared-basis PCA](../results/figures/pa_mechanism_phase2/2008_006143_feature_pca.png) and patch-text margins use the actual final projected patch tokens; they are diagnostics, not complete-dataset evidence. On the small bird case, foreground mean text margin is +0.0135 for QQ, −0.0151 for NS-5 pure PA and +0.0276 for SVD pure PA. The direct prediction panel likewise shows NS-5 nearly losing the bird. The class margin is computed from cosine similarities to the released text queries and does not itself reproduce all pixel postprocessing.

The same three-image token audit (`/data/Multimodal-final-project/outputs/pa_mechanism/phase2/features/voc21_fixed_cases_margin.json`) measures mean final projected token norms of **11.10 for QQ+KK sum** versus **9.39 for QQ+KK mean**, **9.25 for QQ**, **9.32 for pure PA NS-5** and **9.35 for pure PA SVD**. Sum and mean also differ in their per-image patch predictions, so the final visual LayerNorm and projection do not erase the scaling distinction completely. This is a fixed-case check, not a full-validation estimate or an attribution of the mIoU difference to norm alone.

GT-conditioned [corrected/damaged maps for the bird](../results/figures/pa_mechanism_phase2/2008_006143_error_changes.png) and [bus](../results/figures/pa_mechanism_phase2/2008_008252_error_changes.png) compare pure PA NS-5 to QQ pixel by pixel, excluding ignore pixels. On the bird image PA corrects 33 and damages 608 pixels; on the bus image it corrects 5,247 and damages 2,102. These selected examples show opposite outcomes and do not estimate the dataset-wide corrected/damaged balance.

Fixed-index real ADE20K panels, including [image 0001](../results/figures/pa_mechanism_phase2/ADE_val_00000001_ade_segmentation.png) and [image 1001](../results/figures/pa_mechanism_phase2/ADE_val_00001001_ade_segmentation.png), use the released 224/112 sliding inference and 150-class dataset palette. These are descriptive cases chosen before inspecting performance; the full-split aggregates above provide the quantitative evidence.

## Mechanism questions and evidence still required

**QK failure.** Raw foreground-to-background mass alone is not evidence of excess background attraction. The complete diagnostic divides it by available background key area, excludes CLS/ignore keys, separates diagonal from off-diagonal same-class mass, and records within/between-class Query row similarity.

The complete VOC21 attention diagnostic now covers **1,449 images × 12 heads = 17,388 image/head observations**. Ground truth is mapped to the patch grid; patch-class coverage is computed from resized pixel occupancy (mean high-purity patch fraction 0.932). CLS and ignore keys are excluded, and rows are renormalized over valid patch keys. The following are means of per-image/head scalar diagnostics, not pooled pixel metrics:

| Attention | Between-class Query row cosine | Same-class mass | Off-diagonal same-class mass | Foreground→background mass | Background area enrichment | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| QK | .732 | .681 | .678 | .625 | .856 | 6.128 |
| QQ | .296 | .815 | .780 | .485 | .657 | 5.653 |
| KK | .242 | .843 | .803 | .504 | .688 | 5.188 |
| Pure PA NS-5 | .385 | .816 | .808 | .516 | .698 | 5.691 |
| PA+KK NS-5 | .318 | .827 | .816 | .477 | .645 | 5.610 |
| Pure PA SVD | .311 | .825 | .808 | .484 | .656 | 5.605 |

Paired image bootstrap of per-image head means gives QQ−QK differences of **−.436 [−.438,−.433]** for between-class row cosine and **+.102 [.099,.105]** for off-diagonal same-class mass. The marked Query-row similarity under QK is direct evidence for query-indistinguishable attention. The off-diagonal result rules out a pure self-diagonal explanation for QQ/KK's semantic-affinity gain. However, QQ and KK also have lower entropy, so the geometry and temperature effects require separate interpretation.

QK foreground Queries send .625 mass to background Keys, but the available background area fraction averages .722 and the normalized enrichment averages **.856 < 1**. Thus raw background mass is high because background is abundant; there is no evidence here for extra QK background attraction under this metric. The top 1% received-Key share is .030 for QK and .069 for KK, so dominant-token concentration is not uniquely or most strongly a QK phenomenon. These findings reject a simple version of the background/dominant-token hypothesis while retaining the Query-collapse and poor-semantic-affinity hypotheses. [Query-row similarity](../results/figures/pa_mechanism_phase2/between_cos.png), [off-diagonal affinity](../results/figures/pa_mechanism_phase2/off_same.png), and [background enrichment](../results/figures/pa_mechanism_phase2/bg_enrichment.png) show the full-split statistics. Spatial-distance matching and boundary-stratified inference remain open.

An additional [full 1,449-image Key-receiver diagnostic](../results/pa_mechanism_phase2_key_receivers.json) measures the Pearson association between projected Key norm and total received Attention per image/head. The mean correlation is **−.576 for QK**, **+.558 for QQ**, **+.657 for KK**, and **+.698 for pure PA NS-5**. QK's top 1% receiving patches are 72.31% background against 72.22% background area, whereas QQ/KK/PA top receivers are 82.96%/90.63%/86.56% background. Thus QK's failure is not explained by attention to unusually high-norm Keys or excess background among its top receivers under these definitions. The norm correlation is descriptive, and the top-receiver class fraction is not itself an attention-mass enrichment estimate.

The curated [foreground](../results/figures/pa_mechanism_phase2/2008_006143_foreground_attention.png), [background](../results/figures/pa_mechanism_phase2/2008_006143_background_attention.png) and [boundary](../results/figures/pa_mechanism_phase2/2008_006143_boundary_attention.png) Query panels on a fixed VOC21 image share head 0 and one log color scale across methods within each panel. Additional fixed-first and PA-improvement cases, including their PNG/SVG attention sources, are in `results/figures/pa_mechanism_phase2/`. The Query location and case reason are printed in each panel title. Boundary Queries are defined by a neighboring patch with a different mapped GT label; this is a visualization choice, not a full boundary-stratified performance analysis.

ADE20K also has an [image-wide sampled-crop diagnostic](../results/pa_mechanism_phase2_ade_crop_attention_2000.json): for each of the 2,000 validation images, one crop is selected deterministically from the *released 224/112 sliding grid* by image index modulo that image's grid count. The crop coordinates and resized image shape are recorded per image in `/data/Multimodal-final-project/outputs/pa_mechanism/phase2/metrics/ade20k_crop_attention_2000.jsonl`; the original-resolution GT is mapped to that crop before patch occupancy is measured. This covers all images but only one of each image's overlapping crops. ADE's class 0 is a semantic category, so no VOC-style foreground/background metric is applied.

| ADE sampled-crop Attention | Between-class row cosine | Off-diagonal same-class mass | Entropy |
|---|---:|---:|---:|
| QK | .783 | .456 | 4.933 |
| QQ | .326 | .542 | 4.440 |
| KK | .316 | .555 | 4.058 |
| Pure PA NS-5 | .486 | .580 | 4.574 |
| Pure PA SVD | .393 | .575 | 4.483 |

The [ADE crop Attention chart](../results/figures/pa_mechanism_phase2/ade_crop_attention.png) includes 95% image-bootstrap intervals from 1,000 draws. Off-diagonal mass and entropy use 1,997 crops × 12 heads; three chosen crops had fewer than two valid patches. Between-class row cosine uses 1,943 crops × 12 heads because single-class crops lack a between-class pair. Mean high-purity patch fraction is .780; the raw summary also separates high-purity and mixed-patch same-class mass. QK's high between-class similarity and weak off-diagonal same-class affinity recur under this sampled ADE crop protocol. A full census of all overlapping ADE crops, including border-aware inference, remains future work.

**PA necessity.** The full VOC21 SCLIP-style mean slightly exceeds pure PA SVD and NS-5; bootstrap for mean versus NS-5 pure PA gives a small +0.30-point mean advantage with a percentile interval barely excluding zero. On VOC20, pure PA NS-5 adds only 0.08 points over QQ. ADE20K pure PA NS-5 adds +0.43 to QQ, while the best self-correlation mean reaches 17.63 against 17.75 for pure PA. Thus pure PA has a modest but dataset-dependent advantage. VOC21 + TLP puts KK and QQ+KK near PA+KK, whereas VOC20 and ADE20K + TLP favor PA+KK NS-5. These experiments support utility for PA in specific configurations, while its necessity as an independent module is not established across all protocols.

ADE20K paired bootstrap for TLP-off pure PA NS-5 minus QQ gives **+0.43 [+0.27,+0.54]**; PA+KK NS-5 minus QQ+KK mean gives **+0.18 [+0.08,+0.34]**. PA+KK SVD minus NS-5 is **−0.41 [−0.47,−0.33]**. Each interval uses 1,000 paired image resamples and recomputes pooled confusion mIoU. The corresponding per-image diagnostic shows 1,702 improved and 292 damaged images for pure PA NS-5 versus QQ, and 1,230 improved and 765 damaged images for PA+KK NS-5 versus QQ+KK mean; these counts use per-image present-class IoU and do not replace pooled mIoU.

For ADE pure PA, SVD minus NS-5 is only **+0.04 [−0.01,+0.15]** pooled mIoU, even though the per-image present-class IoU improves on 481 images and worsens on 1,512. This is a useful warning that image-level win counts and pooled macro mIoU answer different questions, especially with rare classes. The exact comparison outputs are in [paired-bootstrap JSON](../results/pa_mechanism_phase2_ade20k_paired_bootstrap.json).

[Per-class IoU, precision and recall](../results/pa_mechanism_phase2_ade20k_off.per_class.csv) are reported for every ADE20K variant, with a matching [TLP table](../results/pa_mechanism_phase2_ade20k_tlp.per_class.csv). Relative to QQ, pure PA NS-5 improves 91 and worsens 55 of the 146 ADE classes where both class IoUs are defined; relative to QQ+KK mean, PA+KK NS-5 improves 78 and worsens 68. This spread cautions against treating the small mean advantage as uniform semantic improvement.

The same TLP-off core-method per-class table is available for [VOC21](../results/pa_mechanism_phase2_voc21_off.per_class.csv) and [VOC20](../results/pa_mechanism_phase2_voc20_off.per_class.csv), assembled from their complete Phase 1 and Phase 2 raw confusions. TLP-on class tables are saved alongside the other `results/pa_mechanism_phase2_*per.class.csv` files. Per-class rows with zero GT and zero prediction union are undefined rather than assigned zero IoU.

To separate VOC background scoring from changes in the VOC20/VOC21 candidate label space, a [same-prediction VOC21 pixel audit](../results/pa_mechanism_phase2_voc21_fg_bg.json) reruns six methods on all 1,449 VOC21 images and partitions their valid GT pixels into foreground (65.34 million) and background (179.55 million). QK foreground/background pixel accuracy is **75.03%/62.06%**, QQ **80.20%/87.47%**, QQ+KK mean **79.12%/89.61%**, pure PA NS-5 **80.62%/88.32%**, and PA+KK NS-5 **81.66%/87.97%**. Relative to QQ, pure PA NS-5 corrects **4.27 million** pixels and damages **2.48 million**, including **1.42/1.15 million** corrected/damaged foreground pixels. Relative to QQ+KK mean, PA+KK NS-5 corrects **3.19 million** and damages **4.47 million** overall: **2.32/0.66 million** on foreground, but **0.87/3.81 million** on background. Thus PA+KK's stronger foreground recognition on VOC21 does not imply a better overall VOC21 mIoU; its background tradeoff is directly observed with an unchanged label set. The pixel audit is a descriptive partition and does not turn VOC20/VOC21 into independent image samples.

For the ADE20K bootstrap, a class enters a resample's mIoU when its pooled GT-or-prediction union is positive; absent classes are omitted in that resample, as in the evaluator. All 150 classes occur in the original split. A [1,000-draw GT support audit](../results/pa_mechanism_phase2_ade_bootstrap_coverage.json) found 159 resamples missing at least one GT class, with 148–150 GT classes present. Prediction union can keep a GT-absent class in the mIoU denominator, so this is an upper bound on union-absent frequency. The intervals are therefore sensitive to rare-class sampling and are presented as split-conditional uncertainty, not population confidence guarantees.

**Solver interpretation.** Across the complete 1,449-image VOC21 diagnostic, NS-5 mean `||R^T R-I||_F` is **7.302** versus SVD **8.1e-5**. NS-5 R singular values have means of **.00084 minimum, .216 across dimensions, 1.000 maximum**; SVD's are approximately 1 throughout. NS-5 alters the K Gram matrix by relative Frobenius distance **.168**, versus about **8.4e-6** for SVD. This is substantial direction-selective contraction, not an approximately orthogonal rotation. Mean normalized alignment errors are .694 for NS-5 and .724 for SVD, but their constraints differ; the lower NS-5 error is not evidence that it better solves the orthogonal Procrustes problem. [Orthogonality distribution](../results/figures/pa_mechanism_phase2/orthogonality.png) and [NS convergence](../results/figures/pa_mechanism_phase2/ns_convergence.png) show the corresponding measurements.

Mean NS orthogonality errors at 1/3/5/10/20 iterations are **7.82/7.64/7.30/5.35/1.90** across all image/head observations. Even 20 steps is not perfectly orthogonal by this measure, though its VOC21 segmentation score nearly matches exact SVD. The effect on the task can plateau before exact matrix convergence.

The fixed-seed [128-image singular-value spectrum](../results/figures/pa_mechanism_phase2/solver_spectrum.png) directly shows NS-5's strong contraction in most directions. Its source arrays include the singular spectrum of `M`, NS-5 `R` and SVD `R` for all 12 heads under `/data/Multimodal-final-project/outputs/pa_mechanism/phase2/metrics/voc21_solver_spectrum_128_24000.npz`. The [iteration/performance chart](../results/figures/pa_mechanism_phase2/ns_iterations.png) combines complete VOC21/VOC20/ADE20K scores with the 17,388 VOC21 image/head orthogonality records. VOC20's pure-PA NS 1/3/5/10/20 scores are **82.68/84.02/84.72/84.69/84.63**. ADE20K scores are **16.33/17.32/17.75/17.81/17.80**. NS-20 nearly matches exact SVD on all three datasets; NS-5 is modestly higher than SVD on VOC20.

On the 128 fixed VOC21 images × 12 heads, the [singular-spectrum rank summary](../results/pa_mechanism_phase2_solver_rank.json) gives mean entropy-effective rank **16.77 for M**, **32.50 for NS-5 R**, and **64.00 for SVD R** out of 64 dimensions. Mean stable ranks are **1.59/8.33/64.00** respectively. These measurements show the input cross-covariance spectrum is highly unequal and that five NS steps leave a strongly nonuniform R spectrum; the 128-image rank values are not presented as full-validation estimates.

The full VOC21 pure-PA iteration sweep gives mIoU **57.24/58.42/59.00/59.11/59.08** for NS 1/3/5/10/20 steps. At 20 steps it essentially matches exact SVD's 59.08. The 32-image pilot had the same broad trend but differed in intermediate magnitudes; only the full split is used for the conclusion. More iterations need not monotonically improve mIoU even when the matrix converges toward orthogonality.

## Synthesis

### Part I: Why does Reduced CLIP QK fail here?

On VOC21, QK's between-class Attention-row cosine (.732) is much larger than QQ's (.296), and its off-diagonal same-class mass (.678) is much lower than QQ's (.780) and KK's (.803). Its complete validation mIoU is 35.28, versus 57.82–59.30 for the self-correlation controls. These observations jointly support Query-indistinguishable Attention and weak semantic affinity as mechanisms consistent with poor dense segmentation. They do not by themselves establish that Attention geometry caused every final prediction gain. The area-normalized background and Key-receiver analyses do not support excess QK attraction to background or high-norm dominant Keys. The Vanilla CLIP figure is only an architecture reference; it is not an Attention-only comparison to Reduced QK.

### Part II: What does PA do?

The PA logit identity holds for either solver. Exact SVD is nearly orthogonal; released NS-5 is strongly direction selective, with mean R singular value .216 and K Gram distortion .168 on the full VOC21 diagnostic. SVD residual interpolation still raises VOC21 and ADE20K mIoU above QQ at gamma 1, so non-orthogonality is not required for every PA benefit. VOC20's SVD gamma curve peaks at gamma .5, and SVD PA+KK falls below NS-5 PA+KK on all three datasets. This supports a solver-dependent interaction with KK, but the present data do not isolate which contracted directions matter. Matching QQ's mean Attention entropy to PA leaves a clear PA gap on VOC21 and ADE20K, so a simple mean-temperature explanation is insufficient there. Fixed-case patch-text margins and shared-basis PCA show how attention choices can change final features, while remaining descriptive examples.

### Part III: Is PA necessary?

Pure PA is not uniformly best: QQ+KK mean leads it on VOC21 without TLP, and KK or QQ+KK mean closely match PA+KK NS-5 on VOC21 with TLP. On VOC20, QQ beats QQ+KK mean without TLP, while PA+KK NS-5 leads after TLP. On ADE20K, PA+KK NS-5 exceeds QQ+KK mean by .18 without TLP and .37 with TLP. Those gains and their paired intervals support PA's practical value in these particular settings. The extra per-image/head solve, variable solver ranking, and VOC21 counterexamples prevent a general claim that PA is a necessary module for open-vocabulary segmentation. The observed gain of the full PEARL configuration is a property of the PA+KK+TLP combination, not evidence that exact orthogonal Procrustes is uniquely responsible.

### Part IV: Limits and next research direction

The dataset-wide segmentation conclusions are based on complete released validation splits and fixed configurations. Fine-grained Attention diagnostics cover all VOC21 images and one released-grid crop per ADE20K image; they do not census every overlapping ADE crop. The full gamma sweeps did not entropy-match every gamma point; entropy controls target gamma 1 only. Spatial-distance-matched Query pairs, class-size effects, prediction-transfer matrices, and complete-dataset patch-text margins are still needed to identify where the residual helps and where it hurts. The original attention formulas and TLP implementation were left intact. There were no SVD convergence exceptions in completed evaluator runs; the raw runs did not separately count non-finite intermediate matrices, so absence of all NaN/Inf events cannot be asserted from those records alone.

## Provenance and unfinished work

New raw records are appended under `/data/Multimodal-final-project/outputs/pa_mechanism/phase2/`; configuration JSON identifies the dataset, variant list, seed, checkpoint, architecture and Git base commit. The evaluator skips already completed image/variant pairs on restart and the summarizer rejects incomplete splits. [The Phase 2 source](../experiments/pa_mechanism/phase2/) and [Phase 1 report](PA_MECHANISM_PHASE1.md) describe the reusable baseline.

All three core performance, solver, entropy-control, iteration and TLP matrices are complete at their stated full-validation sizes. VOC21 attention diagnostics cover all 1,449 images; ADE Attention diagnostics cover one official-grid crop from each of 2,000 images. Feature margins and visual panels are fixed-case diagnostics. Spatial-distance-matched Query comparisons, a census of all ADE slide crops, prediction-transfer matrices, size-stratified error analysis, and complete-dataset feature margins remain open. Thus the segmentation rankings have full-split support, while the fine-grained causal story of particular object sizes, boundaries, or all ADE crop positions is not yet established.
