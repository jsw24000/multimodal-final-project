# Phase 2 reproducible runs

Run from the project root using `/data/Multimodal-final-project/cache/conda_envs/pearl/bin/python`. Set `PEARL_DATA_ROOT=/data/Multimodal-final-project`, `PYTHONPATH=$PWD`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=1` and `TMPDIR=$PEARL_DATA_ROOT/cache/tmp`.

The evaluator writes one JSONL row per completed image and variant under `/data/Multimodal-final-project/outputs/pa_mechanism/phase2/runs/`. Repeating an identical command resumes missing pairs. Its sidecar configuration must match exactly; changed settings require a distinct variant list, seed or run name. `clip/model.py` enables a prefix cache only when `run_matrix.py` explicitly sets `phase2_cache_enabled`; the ADE sliding-window equivalence test compares cached and uncached predictions pixel by pixel.

```bash
python experiments/pa_mechanism/phase2/test_solvers.py
python experiments/pa_mechanism/phase2/test_cache.py
python experiments/pa_mechanism/phase2/test_statistics.py
python experiments/pa_mechanism/phase2/test_ade_crop.py
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc21 --variants svd_pa,svd_pakk,svd_g025,svd_g050,svd_g075,svd_g125
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc20 --variants svd_pa,svd_pakk,svd_g025,svd_g050,svd_g075,svd_g125
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc20 --variants ns5_g025,ns5_g050,ns5_g075,ns5_g125
python experiments/pa_mechanism/phase2/run_matrix.py --dataset ade20k --variants qk,qq,kk,sclip,sclip_mean,ns5_pa,ns5_pakk,ns5_g025,ns5_g050,ns5_g075,ns5_g125,svd_pa,svd_pakk,svd_g025,svd_g050,svd_g075,svd_g125
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc21 --tlp on --variants qq,kk,sclip,sclip_mean,ns5_pa,ns5_pakk,svd_pakk
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc20 --tlp on --variants qq,kk,sclip,sclip_mean,ns5_pa,ns5_pakk,svd_pakk
python experiments/pa_mechanism/phase2/run_matrix.py --dataset ade20k --tlp on --variants qq,kk,sclip,sclip_mean,ns5_pa,ns5_pakk,svd_pakk
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc21 --variants ns5_qqmatch,svd_qqmatch
python experiments/pa_mechanism/phase2/run_matrix.py --dataset voc20 --variants ns5_qqmatch,svd_qqmatch
python experiments/pa_mechanism/phase2/run_matrix.py --dataset ade20k --variants ns5_qqmatch,svd_qqmatch
python experiments/pa_mechanism/phase2/run_matrix.py --dataset ade20k --variants ns1_pa,ns3_pa,ns10_pa,ns20_pa
python experiments/pa_mechanism/phase2/diagnose_attention.py --count 0
python experiments/pa_mechanism/phase2/diagnose_key_receivers.py
python experiments/pa_mechanism/phase2/diagnose_ade_crop.py --count 0
python experiments/pa_mechanism/phase2/plot_ade_crop.py
python experiments/pa_mechanism/phase2/foreground_background.py
python experiments/pa_mechanism/phase2/make_full_per_class.py
python experiments/pa_mechanism/phase2/plot_gamma.py
python experiments/pa_mechanism/phase2/plot_iterations.py
python experiments/pa_mechanism/phase2/bootstrap_coverage.py
python experiments/pa_mechanism/phase2/solver_rank.py
python experiments/pa_mechanism/phase2/benchmark.py --tlp off
python experiments/pa_mechanism/phase2/benchmark.py --tlp on
python experiments/pa_mechanism/phase2/build_manifest.py
```

Use `summarize.py RUN.jsonl --out RESULT.csv` only after all expected rows are present; it raises on an incomplete split. `per_class.py` gives class IoU/precision/recall. `paired_compare.py` runs paired image bootstrap on any two complete Phase 1/2 runs. Full-split runs use the original dataset configs, including ADE20K 224/112 sliding inference. Phase 1 results are read-only comparison inputs.
