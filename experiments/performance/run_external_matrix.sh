#!/usr/bin/env bash
set -uo pipefail
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
python="$root/cache/conda_envs/pearl/bin/python"
out="$root/outputs/performance"
[[ -x "$python" && -d "$out" ]] || exit 1
export PEARL_DATA_ROOT="$root" OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
export HF_HOME="$root/cache/huggingface" TORCH_HOME="$root/cache/torch"
export XDG_CACHE_HOME="$root/cache/xdg" TMPDIR="$root/cache/tmp"
cd "$project" || exit 1
for method in sclip naclip; do
  for dataset in voc21 voc20 ade20k; do
    case "$dataset" in voc21|voc20) expected=1449 ;; ade20k) expected=2000 ;; esac
    name="${method}_${dataset}"
    if find "$out/$name" -name '*.json' -type f -print -quit 2>/dev/null | grep -q . \
       && [[ -f "$out/$name.log" ]] && grep -Eq "Iter\(test\) \[ *$expected/$expected\]" "$out/$name.log"; then
      echo "COMPLETED (reuse): $name"
      continue
    fi
    if [[ -e "$out/$name.log" ]]; then
      echo "BLOCKED (existing log): $name"
      continue
    fi
    echo "RUNNING: $name $(date -Is)"
    "$python" experiments/performance/run_external.py "$method" "$dataset" > "$out/$name.log" 2>&1
    code=$?
    if [[ $code -eq 0 ]] && find "$out/$name" -name '*.json' -type f -print -quit | grep -q . \
       && grep -Eq "Iter\(test\) \[ *$expected/$expected\]" "$out/$name.log"; then
      echo "COMPLETED: $name $(date -Is)"
    else
      echo "FAILED: $name exit=$code log=$out/$name.log $(date -Is)"
    fi
  done
done
