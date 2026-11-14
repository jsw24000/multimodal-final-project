#!/usr/bin/env bash
set -uo pipefail
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
python="$root/cache/conda_envs/pearl/bin/python"
out="$root/outputs/qualitative/voc21"
mkdir -p "$out"
export PEARL_DATA_ROOT="$root" OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
export HF_HOME="$root/cache/huggingface" TORCH_HOME="$root/cache/torch"
export XDG_CACHE_HOME="$root/cache/xdg" TMPDIR="$root/cache/tmp"
cd "$project" || exit 1
while systemctl --user is-active --quiet pearl-external-matrix.service; do
  sleep 20
done
for method in vanilla sclip naclip pa_only full_pearl; do
  echo "RUNNING: $method $(date -Is)"
  "$python" experiments/visualization/export_voc21_predictions.py "$method" \
    > "$out/${method}_export.log" 2>&1
  code=$?
  echo "FINISHED: $method exit=$code $(date -Is)"
done
"$python" experiments/visualization/make_voc21_panels.py > "$out/panels.log" 2>&1
echo "PANELS: exit=$? $(date -Is)"
