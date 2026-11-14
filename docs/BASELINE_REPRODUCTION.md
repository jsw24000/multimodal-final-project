# Performance baseline reproduction

## Scope and storage

This run evaluates the complete VOC2012 validation split (1,449 images) for VOC21 and VOC20, and the complete ADE20K validation split (2,000 images). Data, checkpoints, external source trees, logs and visualizations live under `${PEARL_DATA_ROOT}` (here `/data/Multimodal-final-project`). Only small result summaries belong in Git. The CLIP checkpoint is `checkpoints/clip/ViT-B-16.pt`, SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.

The project Git commit recorded in `results/experiment_manifest.json` is the **base commit**, since the new scripts and documentation were uncommitted when the runs were made. External Git commit IDs are recorded there as well. The repository URLs are [PEARL](https://github.com/PGSmall/PEARL), [SCLIP](https://github.com/wangf3014/SCLIP), [NACLIP](https://github.com/sinahmr/NACLIP), and [ClearCLIP](https://github.com/mc-lan/ClearCLIP). SCLIP and NACLIP have MIT license files in their source trees. ClearCLIP's README refers to the NTU S-Lab License 1.0; its source was not copied into this project.

## PEARL official-code protocol

The public PEARL code and dataset configs were retained. We added only an optional `--arch` CLI override in `eval.py`; its empty default preserves the model's original `reduced` architecture. All five PEARL variants use the same official configs, validation data, checkpoint and MMSeg `IoUMetric`:

| Variant | Vision architecture | Last attention | TLP |
| --- | --- | --- | --- |
| Vanilla CLIP | vanilla | vanilla | off |
| Reduced CLIP | reduced | vanilla | off |
| PA only | reduced | pearl | off |
| TLP only | reduced | vanilla | on |
| Full PEARL | reduced | pearl | on |

`Vanilla CLIP` and `Reduced CLIP` are distinct; the latter removes the final residual/FFN path in the published visual encoder. The paper's Table 3 row with both PA and TLP off has not been assigned to either row above because its architecture and evaluation path have not been independently matched.

VOC21 includes background class 0; VOC20 uses the same JPEGs and indexed PNG masks but `reduce_zero_label=True`, treating raw background as ignore and mapping raw 1–20 to 0–19. ADE20K uses raw 0 as ignore and raw 1–150 as classes via `LoadAnnotations(reduce_zero_label=True)`. Masks use 255 for ignore. All configs resize with `(2048, 336)` and `keep_ratio=True`; VOC20/21 set `slide_crop=0`. ADE20K retains the model default 224 crop and 112 stride. PEARL uses the official CLIP ViT-B/16 weight, class/synonym lists in `configs/cls_*.txt`, and `openai_imagenet_template` in `prompts/imagenet_template.py`. Per-template text features are normalized, averaged and normalized again; predictions use logit scale 40 and the official per-dataset probability threshold. PA uses norm-weighted Q/K centering, five Newton–Schulz polar iterations, and K–K coefficient 0.08 except ADE20K's configured 0.3. TLP retains the published implementation with grid 80; no paper-equation replacement was applied.

The [PEARL paper](https://arxiv.org/html/2603.21528) describes overlapping sliding windows in its method text, while the released VOC configs disable cropping. These results follow the released code. The released TLP uses a diagonal `einsum('cc,...')` text-relation operation; any full-matrix paper-equation variant would be a separate experiment, not part of these baselines.

## External official-code protocol

SCLIP and NACLIP were cloned into `cache/external_repos/` and imported unchanged by `experiments/performance/run_external.py`. The wrapper changes only the checkpoint path, validation data root, class-file path and work directory. Each method retains its own official model code, class names, prompts, preprocessing and evaluation config. These are **official-code reproductions**, not a strict controlled Attention comparison. They share the physical VOC/ADE images, ground truth, CLIP ViT-B/16 checkpoint and MMSeg `IoUMetric`, but method-specific prompt and preprocessing choices remain. Their own README reference values differ from values reported in PEARL Table 1; compare local runs with their own repository values first.

ClearCLIP was cloned separately but not run: its smoke import first required `timm==0.4.12`. We installed that package into the isolated data-disk path `cache/external_deps/clearclip` with `--no-deps`, then its smoke import stopped on missing `fairscale`. The logs are `outputs/performance/clearclip_smoke*.log`. Its pinned PyTorch 1.13.1 / torchvision 0.14.1 stack is incompatible with this Blackwell GPU's tested PyTorch 2.8.0+cu128 environment. No incompatible dependency was installed into the PEARL environment. A future ClearCLIP run needs a separate compatible environment and a tested adaptation; its metrics remain `Not Run`.

## Reproduce

```bash
cd /home/3dsm/Desktop/JSWprobe/Multimodal-final-project
export PEARL_DATA_ROOT=/data/Multimodal-final-project
export HF_HOME=$PEARL_DATA_ROOT/cache/huggingface
export TORCH_HOME=$PEARL_DATA_ROOT/cache/torch
export XDG_CACHE_HOME=$PEARL_DATA_ROOT/cache/xdg
export TMPDIR=$PEARL_DATA_ROOT/cache/tmp
conda activate "$PEARL_DATA_ROOT/cache/conda_envs/pearl"
python scripts/verify_datasets.py --dataset voc21
python scripts/verify_datasets.py --dataset ade20k
python eval.py --config configs/cfg_voc21.py --attn pearl --prop on
bash experiments/performance/run_pearl_matrix.sh
bash experiments/performance/run_external_matrix.sh
python scripts/collect_performance_results.py
```

The matrix scripts skip runs with existing result JSONs and refuse to overwrite an existing unfinished log. Inspect a partial log and choose an unused work directory before any retry. The collector checks for the final validation iteration before accepting a result. For single-run commands, see the `commands` and `runs` fields in `results/experiment_manifest.json`.
