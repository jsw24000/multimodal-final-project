# Phase 3: residual cross-projection

Run from the project root with `PEARL_DATA_ROOT=/data/Multimodal-final-project`, `PYTHONPATH=$PWD`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=1`, and `TMPDIR=/data/Multimodal-final-project/cache/tmp`. Use `/data/Multimodal-final-project/cache/conda_envs/pearl/bin/python` as `python` in the commands below.

All Phase 3 alignment variants use the unchanged CLIP ViT-B/16 checkpoint, the Reduced final block and exact SVD. The evaluator writes one confusion record per image and variant to the data disk. Repeating an identical command resumes missing records; configuration sidecars are checked before inference. It uses the opt-in Phase 2 prefix cache and an opt-in Phase 3 SVD alignment cache; `test_cache.py` checks cached and uncached ADE sliding predictions pixelwise.

Variant names:

- `g150`: Pure PA QQ-residual strength `gamma=1.50`.
- `kb050`: KK-to-PA *logit* interpolation `beta=.50`.
- `cb050`: CSA-mean-to-PA *post-softmax* interpolation `beta=.50`.
- `cw075`: weighted CSA with QQ weight `lambda=.75`.
- `rk_g150_a008`: optional residual plus centered KK, `gamma=1.50`, `alpha=.08`; this is not Pure PA.
- `g100`: exact Pure PA SVD.

The off-matrix candidate values were fixed before full validation: `g150,g200,g250,g300,kb025,kb050,kb075,cb025,cb050,cb075,cw025,cw075`. Their endpoints and the earlier `g025,g050,g075,g100,g125` come from complete Phase 1/2 runs. Gamma is not an alignment quality parameter once it differs from 1. The `kb` path is a logit interpolation; the `cb` and `cw` paths combine normalized Attention matrices. These are different mechanisms.

```bash
python experiments/pa_mechanism/phase3/test_interpolation.py
python experiments/pa_mechanism/phase3/test_cache.py
python experiments/pa_mechanism/phase3/test_statistics.py
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc21 --tlp off --variants g150,g200,g250,g300,kb025,kb050,kb075,cb025,cb050,cb075,cw025,cw075
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc20 --tlp off --variants g150,g200,g250,g300,kb025,kb050,kb075,cb025,cb050,cb075,cw025,cw075
python experiments/pa_mechanism/phase3/run_phase3.py --dataset ade20k --tlp off --variants g150,g200,g250,g300,kb025,kb050,kb075,cb025,cb050,cb075,cw025,cw075
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc21 --tlp on --variants g100
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc20 --tlp on --variants g100
python experiments/pa_mechanism/phase3/run_phase3.py --dataset ade20k --tlp on --variants g100
python experiments/pa_mechanism/phase3/aggregate_results.py
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc21 --tlp on --variants g125,g150,cw075
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc20 --tlp on --variants g125,g150,cw075
python experiments/pa_mechanism/phase3/run_phase3.py --dataset ade20k --tlp on --variants g125,g150,cw075
```

The frozen `results/pa_mechanism_phase3_selection_protocol.json` uses aligned VOC21/VOC20 image indices and a separate ADE partition. It selects one shared gamma and one shared CSA weight on labeled development subsets, then reports non-overlapping holdout performance. This is weight-free, **labeled hyperparameter selection**. The selected values were `g125` and `cw075`. The additional `g150` TLP run is exploratory. Complete validation curves are exploratory, not independent validation after selection.

The conditional centered-KK extension was recorded in `results/pa_mechanism_phase3_kk_extension_protocol.json` after the core results, so it is explicitly exploratory. Full TLP-off commands are:

```bash
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc21 --tlp off --variants rk_g125_a003,rk_g125_a008,rk_g125_a015
python experiments/pa_mechanism/phase3/run_phase3.py --dataset voc20 --tlp off --variants rk_g125_a003,rk_g125_a008,rk_g125_a015
python experiments/pa_mechanism/phase3/run_phase3.py --dataset ade20k --tlp off --variants rk_g125_a003,rk_g125_a008,rk_g125_a015,rk_g125_a030
python experiments/pa_mechanism/phase3/run_phase3.py --dataset ade20k --tlp on --variants rk_g125_a008
```

After all full runs, use:

```bash
python experiments/pa_mechanism/phase3/aggregate_results.py
python experiments/pa_mechanism/phase3/run_bootstrap_matrix.py
python experiments/pa_mechanism/phase3/run_kk_extension_bootstrap.py
python experiments/pa_mechanism/phase3/class_deltas.py
python experiments/pa_mechanism/phase3/plot_results.py
python experiments/pa_mechanism/phase3/build_manifest.py
```

`aggregate_results.py` merges complete historical endpoints and Phase 3 records; `build_manifest.py` checks actual image counts. Full raw JSONL remains under `/data/Multimodal-final-project/outputs/pa_mechanism/phase3/`; compact tables, the report and selected figures enter Git.
