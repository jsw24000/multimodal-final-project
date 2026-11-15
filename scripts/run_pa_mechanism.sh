#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
export PEARL_DATA_ROOT="$root" PYTHONPATH="$project" OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
export TMPDIR="$root/cache/tmp" XDG_CACHE_HOME="$root/cache/xdg"
cd "$project"
python="$root/cache/conda_envs/pearl/bin/python"
"$python" experiments/pa_mechanism/test_numerics.py
"$python" experiments/pa_mechanism/run_pilot.py --count 32 --seed 24000
"$python" experiments/pa_mechanism/analyze_geometry.py --count 16 --seed 24000
"$python" experiments/pa_mechanism/visualize.py
