#!/usr/bin/env bash
set -euo pipefail

root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
mount=${PEARL_DATA_MOUNT:-/data}
printf 'user=%s\nproject=%s\ndata_root=%s\n' "$(whoami)" "$(pwd)" "$root"
printf '\nMount:\n'
findmnt -T "$mount" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
df -hT "$mount" /
if [[ -e "$root" ]]; then
  findmnt -T "$root" -o TARGET,SOURCE,FSTYPE,AVAIL
  if [[ ! -d "$root" || ! -w "$root" ||
        "$(findmnt -n -o SOURCE -T "$root")" != "$(findmnt -n -o SOURCE -T "$mount")" ||
        "$(findmnt -n -o SOURCE -T "$root")" == "$(findmnt -n -o SOURCE -T /)" ]]; then
    printf 'BLOCKED: data root %s is not a writable directory on the large mount\n' "$root" >&2
  else
    printf 'data root writable: yes\n'
  fi
else
  printf 'BLOCKED: data root %s does not exist\n' "$root" >&2
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
