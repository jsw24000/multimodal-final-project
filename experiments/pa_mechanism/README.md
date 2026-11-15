# Controlled PA mechanism experiments

Run from repository root in the existing data-disk environment:

```bash
export PEARL_DATA_ROOT=/data/Multimodal-final-project
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
python experiments/pa_mechanism/test_numerics.py
python experiments/pa_mechanism/run_pilot.py --count 32 --seed 24000
python experiments/pa_mechanism/analyze_geometry.py --count 16 --seed 24000
python experiments/pa_mechanism/visualize.py
python experiments/pa_mechanism/run_pilot.py --count 1449 --seed 24000
python experiments/pa_mechanism/summarize.py /data/Multimodal-final-project/outputs/pa_mechanism/runs/eval_1449_24000_11.json --expected 1449 --output results/pa_mechanism_summary.csv
```

Large per-image confusion matrices, predictions, and full-resolution figures stay on `/data`. The `--expected` guard prevents a partial run from being mislabeled as full validation. `run_pilot.py` uses a single model instance and switches only final-block attention. Results are saved after every 25 images and on completion. For a new run, choose an unused seed or distinct output path; the script currently writes its named run file.

`analyze_geometry.py` produces per-image, per-head measurements; GT is used only for diagnostics after inference. Its patch GT is nearest-neighbor resized to the encoder grid. This is a pragmatic diagnostic mapping, not a boundary-purity filter.
