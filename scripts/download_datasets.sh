#!/usr/bin/env bash
set -euo pipefail

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
mount=${PEARL_DATA_MOUNT:-/data}
dataset=${1:-}
case "$dataset" in voc2012|ade20k|context|coco_val) ;; *)
  echo 'Usage: scripts/download_datasets.sh {voc2012|ade20k|context|coco_val}' >&2; exit 2 ;;
esac

# Verify the mount before *any* directory creation or download.
test -d "$mount" && mountpoint -q "$mount" || { echo "BLOCKED: $mount is not a mount" >&2; exit 3; }
test -w "$mount" || { echo "BLOCKED: $mount is not writable" >&2; exit 3; }
mount_source=$(findmnt -n -o SOURCE -T "$mount")
root_source=$(findmnt -n -o SOURCE -T "$(dirname "$(realpath -m "$root")")")
system_source=$(findmnt -n -o SOURCE -T /)
[[ "$(realpath -m "$root")" == "$(realpath "$mount")"/* && "$mount_source" != "$system_source" && "$root_source" == "$mount_source" ]] || {
  echo "BLOCKED: $root is not on the separate mounted disk" >&2; exit 3;
}
mkdir -p "$root"/{datasets,checkpoints,cache,outputs,downloads}
[[ "$(findmnt -n -o SOURCE -T "$root")" == "$mount_source" ]] || exit 3
available=$(df -B1 --output=avail "$root" | tail -1)

fetch() {
  local url=$1 file=$2 expected=$3
  if [[ -s "$file" ]]; then
    echo "Reuse archive: $file"
  else
    (( available > expected * 2 )) || { echo "BLOCKED: insufficient free space for $file" >&2; exit 4; }
    echo "Download $url -> $file.part"
    wget --no-hsts --https-only --continue --tries=3 --timeout=45 --waitretry=5 -O "$file.part" "$url"
    [[ $(stat -c %s "$file.part") -eq $expected ]] || { echo "FAILED: archive size mismatch; preserving .part" >&2; exit 5; }
    mv -n "$file.part" "$file"
  fi
  [[ $(stat -c %s "$file") -eq $expected ]] || { echo "FAILED: existing archive size mismatch; preserving it" >&2; exit 5; }
}

case "$dataset" in
  voc2012)
    if python "$project/scripts/verify_datasets.py" --dataset voc21 --data-root "$root" --quiet; then
      echo 'COMPLETED: VOC2012 already verified'; exit 0
    fi
    target="$root/datasets/VOCdevkit/VOC2012"
    [[ ! -e "$target" ]] || { echo "BLOCKED: existing unverified $target; inspect without overwriting" >&2; exit 6; }
    archive="$root/downloads/VOCtrainval_11-May-2012.tar"
    fetch 'https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' "$archive" 1999639040
    tar -tf "$archive" >/dev/null
    stage="$root/downloads/staging/voc2012"
    mkdir -p "$stage"
    tar -xf "$archive" -C "$stage" --skip-old-files
    PEARL_DATA_ROOT="$stage" python "$project/scripts/verify_datasets.py" --dataset voc21 --dataset-dir "$stage/VOCdevkit/VOC2012" --quiet
    mkdir -p "$root/datasets/VOCdevkit"
    mv -n "$stage/VOCdevkit/VOC2012" "$target"
    python "$project/scripts/verify_datasets.py" --dataset voc21 --data-root "$root"
    ;;
  ade20k)
    if python "$project/scripts/verify_datasets.py" --dataset ade20k --data-root "$root" --quiet; then
      echo 'COMPLETED: ADE20K already verified'; exit 0
    fi
    target="$root/datasets/ADEChallengeData2016"
    [[ ! -e "$target" ]] || { echo "BLOCKED: existing unverified $target; inspect without overwriting" >&2; exit 6; }
    archive="$root/downloads/ADEChallengeData2016.zip"
    fetch 'https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip' "$archive" 967382037
    unzip -tq "$archive" >/dev/null
    stage="$root/downloads/staging/ade20k"
    mkdir -p "$stage"
    unzip -nq "$archive" -d "$stage"
    python "$project/scripts/verify_datasets.py" --dataset ade20k --dataset-dir "$stage/ADEChallengeData2016" --quiet
    mv -n "$stage/ADEChallengeData2016" "$target"
    python "$project/scripts/verify_datasets.py" --dataset ade20k --data-root "$root"
    ;;
  context)
    echo 'PENDING: VOC2010 images, trainval_merged.json and MMSegmentation Detail converter required.'
    echo 'See docs/DATASETS.md. No unverified GT is generated.'
    ;;
  coco_val)
    echo 'PENDING: COCO val2017 plus COCO-Stuff PNG maps; validation-only conversion protocol needs confirmation.'
    echo 'See docs/DATASETS.md. No unverified GT is generated.'
    ;;
esac
