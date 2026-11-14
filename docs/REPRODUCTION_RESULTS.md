# Full validation reproduction results

Date: 2026-11-14. All numbers below are local full-validation results from saved MMEngine logs and JSONs, not copied paper scores. The machine used one NVIDIA RTX PRO 6000 Blackwell GPU (driver 570.211.01), Python 3.10.20, PyTorch 2.8.0+cu128, MMCV 2.1.0, MMEngine 0.10.7 and MMSegmentation 1.2.2. `/data` is an ext4 mount on `/dev/nvme0n1` with 6.6T available at inspection. VOC21/VOC20 share the 1,449-image VOC2012 validation split; ADE20K uses 2,000 validation pairs. The dataset verifier and final MMEngine iteration were checked. Detailed per-class IoU, mAcc, pixel accuracy, logged GPU memory, runtime, source logs and commit IDs are in [the manifest](../results/experiment_manifest.json) and [the CSV](../results/performance_summary.csv). `elapsed_s` is computed from first/last MMEngine log timestamps and may exclude model setup; `peak_logged_memory_MiB` is the largest evaluator-reported allocation, not a whole-process device peak.

## Quantitative results

Each cell is local mIoU in percent. Parentheses give the PEARL paper Table 3 value for directly matched PEARL configurations. `N/A` means that a directly matched paper value has not been established. All rows marked Completed evaluated the complete split.

| Method | Status | VOC21 | VOC20 | ADE20K |
| --- | --- | ---: | ---: | ---: |
| Vanilla CLIP | Completed | 15.16 (N/A) | 38.01 (N/A) | 3.17 (N/A) |
| Reduced CLIP | Completed | 35.28 (N/A) | 78.98 (N/A) | 11.60 (N/A) |
| PEARL PA only | Completed | 59.19 (59.2) | 84.95 (85.0) | 17.82 (17.9) |
| PEARL TLP only | Completed | 35.37 (35.4) | 79.27 (79.3) | 11.71 (11.7) |
| Full PEARL | Completed | **64.07 (64.1)** | **86.90 (86.9)** | **19.36 (19.4)** |
| SCLIP official repo | Completed | 59.63 | 81.54 | 16.46 |
| NACLIP official repo | Completed | 64.06 | 83.03 | 19.07 |
| ClearCLIP | Not Run | — | — | — |

Full PEARL differs from the PEARL paper by -0.03, 0.00 and -0.04 points respectively. PA only differs by -0.01, -0.05 and -0.08; TLP only by -0.03, -0.03 and +0.01. These are rounded two-decimal comparisons, so the tiny differences can reflect reporting precision. This is strong agreement with the **released-code** protocol; it does not establish exact reproduction of every method-text detail in the paper.

SCLIP's own README reports 59.63/81.54/16.45; local results differ by 0.00/0.00/+0.01. NACLIP's own README reports 64.07/83.03/19.05; local results differ by -0.01/0.00/+0.02. The PEARL paper Table 1 reports different SCLIP and NACLIP figures (SCLIP 59.1/80.4/16.1, NACLIP 58.9/79.7/17.4). Their published settings or evaluation paths therefore cannot be treated as matched without further protocol reconciliation. The external rows above are official-repository reproductions on the shared local datasets and checkpoint, **not** strict controlled comparisons with PEARL.

| External baseline reference | VOC21 | VOC20 | ADE20K |
| --- | ---: | ---: | ---: |
| SCLIP repository README | 59.63 | 81.54 | 16.45 |
| SCLIP as reported in PEARL Table 1 | 59.1 | 80.4 | 16.1 |
| NACLIP repository README | 64.07 | 83.03 | 19.05 |
| NACLIP as reported in PEARL Table 1 | 58.9 | 79.7 | 17.4 |

The observed increase from Reduced CLIP to PA only is +23.91/+5.97/+6.22 points (VOC21/VOC20/ADE20K). Adding TLP to PA yields +4.88/+1.95/+1.54 points. Vanilla CLIP is much lower than Reduced CLIP in these dense prediction configs; collapsing those architectures into one baseline would obscure the source of improvement.

## Qualitative comparison

Ten fixed VOC21 validation image IDs, chosen by predetermined split indices rather than GT-based metric optimization, have full prediction panels under `/data/Multimodal-final-project/outputs/qualitative/voc21/panels/`. Each panel uses the same indexed VOC color mapping and shows `Input | Ground Truth | Vanilla CLIP | SCLIP | NACLIP | PEARL PA only | Full PEARL`. Raw prediction PNGs for each method and `selection.json` are beside the panels. Image-level IoU in `selection.json` is a diagnostic mean over classes present in that image's GT or prediction, not the dataset mIoU reported above.

Representative panels:

- `2007_000033.png`: aeroplane against a busy airport background. Full PEARL image-level IoU is 0.774 versus PA only 0.457, SCLIP 0.383 and NACLIP 0.371. The full method better retains the aircraft body while suppressing background clutter.
- `2008_002904.png`: Full PEARL reaches 0.905 versus NACLIP 0.612, SCLIP 0.513 and PA only 0.548; inspect this panel for a clear improvement example.
- `2008_006036.png`: Full PEARL reaches only 0.332 while SCLIP reaches 0.863. This is a useful failure case that prevents an overly uniform claim about PEARL.
- `2009_002549.png`: SCLIP reaches 0.933 while Full PEARL reaches 0.596. Inspect the class boundaries and background response before attributing the error to a particular mechanism.

The figures are illustrative, not a representative statistical sample. GT white regions are VOC ignore label 255. The visualized image IoU should not be compared numerically with validation mIoU.

## Protocol limits and next checks

PEARL VOC configs set `slide_crop=0`, despite the paper's method text describing overlapping windows. ADE20K keeps the released 224 crop / 112 stride and `alpha_kk=0.3`. The released TLP code also differs from a full text-relation-matrix reading of the paper equation. We made no silent corrections. Prompt lists, background handling and preprocessing in SCLIP/NACLIP remain method-specific, so the shared datasets and checkpoint alone do not make a controlled Attention ablation. ClearCLIP's first smoke import failed on missing `timm`; after an isolated `timm` install, a second smoke import failed on missing `fairscale`. Its older pinned PyTorch stack should be handled in an isolated compatible environment.

The most valuable next analysis steps are (1) compare foreground/background attention mass and high-norm K tokens on VOC21 success and failure panels, (2) quantify Q/K alignment error and PCA/SVD dominant directions before and after PA, and (3) separately inspect TLP's changes at class boundaries versus object interiors. Use the existing hooks identified in [ANALYSIS_INTERFACE.md](ANALYSIS_INTERFACE.md); keep these diagnostic experiments separate from the baseline scores.
