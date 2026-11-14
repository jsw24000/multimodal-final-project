#!/usr/bin/env bash
set -uo pipefail

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
python="$root/cache/conda_envs/pearl/bin/python"
out="$root/outputs/performance"
[[ -x "$python" && -d "$out" ]] || { echo 'PEARL environment/output missing' >&2; exit 1; }
[[ "$(findmnt -n -o SOURCE -T "$root")" != "$(findmnt -n -o SOURCE -T /)" ]] || exit 1
export PEARL_DATA_ROOT="$root" OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
export HF_HOME="$root/cache/huggingface" TORCH_HOME="$root/cache/torch"
export XDG_CACHE_HOME="$root/cache/xdg" TMPDIR="$root/cache/tmp"
cd "$project" || exit 1

for dataset in voc21 voc20 ade20k; do
  case "$dataset" in voc21|voc20) expected=1449 ;; ade20k) expected=2000 ;; esac
  for variant in vanilla reduced pa_only tlp_only; do
    case "$variant" in
      vanilla) arch=vanilla; attn=vanilla; prop=off ;;
      reduced) arch=reduced; attn=vanilla; prop=off ;;
      pa_only) arch=reduced; attn=pearl; prop=off ;;
      tlp_only) arch=reduced; attn=vanilla; prop=on ;;
    esac
    name="${dataset}_${variant}"
    work="$out/$name"
    log="$out/$name.log"
    if find "$work" -name '*.json' -type f -print -quit 2>/dev/null | grep -q . \
       && [[ -f "$log" ]] && grep -Eq "Iter\(test\) \[ *$expected/$expected\]" "$log"; then
      echo "COMPLETED (reuse): $name"
      continue
    fi
    if [[ -e "$log" ]]; then
      echo "BLOCKED (existing log needs inspection): $name"
      continue
    fi
    echo "RUNNING: $name $(date -Is)"
    "$python" eval.py --config "configs/cfg_${dataset}.py" --arch "$arch" \
      --attn "$attn" --prop "$prop" --work-dir "$work" > "$log" 2>&1
    code=$?
    if [[ $code -eq 0 ]] && find "$work" -name '*.json' -type f -print -quit | grep -q . \
       && grep -Eq "Iter\(test\) \[ *$expected/$expected\]" "$log"; then
      echo "COMPLETED: $name $(date -Is)"
    else
      echo "FAILED: $name exit=$code log=$log $(date -Is)"
    fi
  done
done
