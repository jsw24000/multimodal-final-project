#!/usr/bin/env bash
set -euo pipefail

root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
mount=${PEARL_DATA_MOUNT:-/data}
printf 'user=%s\nproject=%s\ndata_root=%s\n' "$(whoami)" "$(pwd)" "$root"
printf '\nMount:\n'
findmnt -T "$mount" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
df -hT "$mount" /
if [[ ! -d "$mount" ]] || [[ ! -w "$mount" ]]; then
  printf 'BLOCKED: %s is absent or not writable by %s\n' "$mount" "$(whoami)" >&2
fi
if [[ -e "$root" ]]; then
  findmnt -T "$root" -o TARGET,SOURCE,FSTYPE,AVAIL
fi
printf '\nGPU and tools:\n'
for command_name in nvidia-smi nvcc python conda git gh wget; do
  if command -v "$command_name" >/dev/null 2>&1; then
    case "$command_name" in
      nvidia-smi) nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader ;;
      nvcc) nvcc --version | tail -1 ;;
      python) python --version ;;
      conda) conda --version ;;
      git) git --version ;;
      gh) gh --version | head -1 ;;
      wget) wget --version | head -1 ;;
    esac
  else
    printf '%s: MISSING\n' "$command_name"
  fi
done
