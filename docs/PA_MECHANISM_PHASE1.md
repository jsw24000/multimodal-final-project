# PEARL PA mechanism analysis, Phase 1

## Research questions

**RQ1.** Do Q and K preserve useful spatial semantics, and does alignment improve cross-token affinity on held-out patches? Alignment error alone is insufficient: a Procrustes fit lowers it by construction. We therefore measure GT-conditioned attention, Q/K Gram similarity, and random and spatial held-out fits.

**RQ2.** Does PA help because it becomes QQ-like, or because its residual cross-space term is useful? With centered `K_c R = Q_c + E`, `softmax(Q(KR)^T/sqrt(d))` equals `softmax((QQ^T + QE^T)/sqrt(d))`, since the omitted term is constant along each attention row. The numerical test checks logits up to a row constant, attention, and value aggregation.

## Controlled setup and numerical checks

The starting point is commit `fd72506466f593533e5c10d98ae665a0373e11c6` and the already reproduced results in [REPRODUCTION_RESULTS.md](REPRODUCTION_RESULTS.md). All new inference uses the same CLIP ViT-B/16 checkpoint and PEARL VOC21 protocol, the Reduced final block, TLP off, and the same value and output projection. Strategies are QK, QQ, KK, SCLIP-style `softmax(QQ)+softmax(KK)`, its half-scaled version, pure PA, and PA+KK. The SCLIP-style sum has row sum 2; its average is a scale control and is not the official SCLIP computation. `gamma` uses `QQ + gamma*QE` with one alignment per forward pass. The official PA+KK coefficient is 0.08 on VOC21.

`experiments/pa_mechanism/test_numerics.py` passed on the local PyTorch environment. It checks gamma=0 against QQ, gamma=1 against pure PA, both PA variants against the original implementation, the row-constant logits identity, value aggregation, and the SCLIP-style row sum. Calculations are in float32. The unchanged default `pearl` code path remains available.

## Pilot results

The fixed-seed (`24000`) pilot contains 32 VOC21 validation images, sampled without replacement. The per-image confusion matrices are pooled before computing dataset-style mIoU. **These are pilot values, not full-validation scores.**

| Attention | Pilot mIoU (%) |
|---|---:|
| QK | 29.65 |
| QQ | 50.53 |
| KK | 48.95 |
| SCLIP-style sum | 49.18 |
| SCLIP-style mean | 49.80 |
| Pure PA | 51.29 |
| PA+KK | 52.19 |
| gamma=.25 | 51.72 |
| gamma=.5 | 51.73 |
| gamma=.75 | 51.60 |
| gamma=1.25 | 50.44 |

In this small sample, QQ captures much of the gain over QK. Positive residual strength gives a modest further gain, but the apparent peak around .25–.5 is exploratory and cannot select a parameter on the validation split. The official PA+KK gain is distinct from pure PA.

## Full VOC21 validation, 1,449 images

Every row below pooled the same 1,449 image-level confusion matrices. The run passed the expected-count guard. The checkpoint SHA256 is `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`. The QK and PA+KK mIoU values agree with the already saved Reduced CLIP and PA-only historical results to two decimals, an integration cross-check.

| Variant | mIoU | mAcc | pAcc |
|---|---:|---:|---:|
| Reduced QK | 35.28 | 76.82 | 65.52 |
| QQ | 57.82 | 82.47 | 85.53 |
| KK | 58.91 | 77.39 | 87.16 |
| SCLIP-style sum | 58.91 | 81.17 | 86.59 |
| SCLIP-style mean | 59.30 | 80.89 | 86.81 |
| Pure PA | 59.00 | 82.39 | 86.26 |
| PA+KK | 59.19 | 83.43 | 86.28 |

| gamma | 0 | .25 | .5 | .75 | 1 | 1.25 |
|---|---:|---:|---:|---:|---:|---:|
| mIoU | 57.82 | 58.22 | 58.53 | 58.78 | 59.00 | 59.14 |

The full split **reverses the pilot's apparent intermediate-gamma peak**. In the controlled VOC21 setup, QQ explains a large part of the improvement over QK, yet keeping the residual adds 1.18 mIoU points at gamma=1. An additional KK term adds 0.19 points. This is an empirical performance effect of the residual; it does not by itself establish that `QE^T` encodes a specific semantic relation. Gamma=1.25 was included as a prespecified diagnostic, not selected as a tuned model. The SCLIP-style average scoring 0.30 above pure PA also cautions against a unique PA-only explanation; its changed output scale is a confound for mechanistic attribution.

On the full VOC20 validation split (the same 1,449 images, with background ignored), QK/QQ/pure PA/PA+KK score **78.98/84.64/84.72/84.95 mIoU**. The residual's pure-PA increment above QQ is only 0.08 points here, versus 1.18 on VOC21. This is consistent with a stronger role for the explicit background class in VOC21, but that interpretation requires a controlled class-conditional analysis. ADE20K full controlled validation has not yet been run.

In a separate **32-image ADE20K pilot**, QK/QQ/pure PA/PA+KK score 7.19/9.91/11.10/10.80 pooled mIoU over classes represented in this small subset. These figures are not directly comparable to the 2,000-image ADE20K scores or usable for a cross-dataset performance claim. They establish only that the controlled implementation executes on the ADE sliding-window protocol; the apparent pure-PA advantage needs a full validation check.

## Per-head geometry and semantics, 16-image diagnostic sample

Across 16 fixed-seed images and all 12 heads (192 image/head observations), the mean normalized alignment error from the released five-step Newton–Schulz solver is 0.695; exact SVD yields 0.724. The mean Frobenius orthogonality error `||R^T R - I||_F` is 7.28 for the released solver and about 8.2e-5 for SVD. Thus the released transformation is markedly non-orthogonal at five iterations, and its lower reconstruction error cannot be interpreted as proof of a better *orthogonal* Procrustes solution. This numerical distinction needs further investigation.

SVD fits on half the patches yielded relative train/test errors of 0.695/0.805 for random splits and 0.665/1.005 for top-versus-bottom spatial splits. Held-out tokens did not contribute to either center or R. The larger spatial gap is evidence against assuming uniform alignment across the image; it does not alone identify the cause.

GT-conditioned patch attention excludes CLS and renormalizes each row over patch keys. Mean same-class mass is QK 0.675, QQ 0.800, KK 0.826, pure PA 0.801 and PA+KK 0.810. Mean foreground-to-background leakage is respectively 0.606, 0.454, 0.430, 0.471 and 0.437. These descriptive diagnostics differ in entropy (QK 6.17, QQ 5.70, KK 5.26, pure PA 5.75, PA+KK 5.67), so semantic mass differences cannot all be attributed to geometry alone. Background area and boundary conditioning remain to be analyzed.

## Figures and limitations

Full-resolution figures and source data reside under `/data/Multimodal-final-project/outputs/pa_mechanism/`. [Full gamma curve](../results/figures/pa_mechanism/gamma_full_voc21.png), [fixed-subset segmentation comparison](../results/figures/pa_mechanism/segmentation_2007_003611.png), [residual interpolation panel](../results/figures/pa_mechanism/residual_2007_003051.png), and [foreground attention](../results/figures/pa_mechanism/attention_2007_004121_foreground.png) are compressed report copies. Attention heatmaps use a common scale within the figure.

The full-split image-level diagnostic selected [a large PA improvement, `2008_008252`](../results/figures/pa_mechanism/segmentation_2008_008252.png), and [a PA regression, `2008_006143`](../results/figures/pa_mechanism/segmentation_2008_006143.png). Their image-level mean IoU differences (pure PA minus QQ) are +62.8 and -37.1 points respectively, calculated over classes present in that individual image's GT or prediction. These examples are deliberately extreme and cannot represent the dataset distribution. `2007_003051` is a near-unchanged fixed-subset example (75.5 QQ versus 75.4 pure PA image-level IoU). The actual pixel panels should be inspected for class-specific errors before assigning a mechanism to either extreme.

The improvement panel shows a bus at night: QQ includes several false colored patches inside and beside the bus, whereas pure PA largely preserves the bus and suppresses those errors. The regression panel shows a small bird against dark foliage: QQ retains a recognizable bird region while pure PA nearly loses it. These observations show residual benefit is conditional on the image and object; they do not establish that background suppression alone explains PA.

The 16-image attention diagnostics and 32-image qualitative subset are too small for general semantic-geometry claims, although VOC21 and the four VOC20 performance rows use complete validation sets. The current work has no boundary-stratified semantic analysis, no full-split statistical uncertainty, and no causal intervention isolating semantic geometry from entropy. ADE20K has only a 32-image pilot. The released five-step polar solver's large orthogonality error makes the idealized "rotation" interpretation particularly uncertain. A focused next experiment should compare pure PA using exact SVD versus released polar, then test whether the residual's benefit persists after entropy matching.
