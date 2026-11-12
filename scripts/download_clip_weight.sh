#!/usr/bin/env bash
set -euo pipefail

root=${PEARL_DATA_ROOT:-/data/Multimodal-final-project}
if [[ ! -d "$root" ]]; then
  echo "BLOCKED: PEARL_DATA_ROOT does not exist: $root" >&2
  exit 1
fi
source_disk=$(findmnt -n -o SOURCE -T "$root")
system_disk=$(findmnt -n -o SOURCE -T /)
disk_size=$(df -B1 --output=size "$root" | tail -n 1)
if [[ ! -w "$root" || "$source_disk" == "$system_disk" || "$disk_size" -lt 6000000000000 ]]; then
  echo "BLOCKED: PEARL_DATA_ROOT is not writable on the external disk: $root" >&2
  exit 1
fi

target_dir="$root/checkpoints/clip"
target="$target_dir/ViT-B-16.pt"
partial="$target.part"
log="$target_dir/download.log"
url='https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt'
sha256='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
expected_size=350837078
mkdir -p "$target_dir"

if [[ -f "$target" ]]; then
  printf '%s  %s\n' "$sha256" "$target" | sha256sum -c -
  exit 0
fi

stalled=0
for ((attempt=1; attempt<=30; attempt++)); do
  before=$(stat -c%s "$partial" 2>/dev/null || echo 0)
  if ((before >= expected_size)); then
    break
  fi
  echo "CLIP download attempt $attempt: $before/$expected_size bytes"
  # This host can reach the official CLIP endpoint directly via IPv6; its
  # desktop proxy has caused repeated read timeouts for this endpoint.
  wget -6 --no-proxy --no-hsts --https-only --continue --tries=1 --timeout=20 \
    --progress=dot:giga -O "$partial" "$url" >> "$log" 2>&1 || true
  after=$(stat -c%s "$partial" 2>/dev/null || echo 0)
  if ((after == before)); then
    ((stalled += 1))
  else
    stalled=0
  fi
  if ((stalled >= 3)); then
    echo "BLOCKED: no download progress in three attempts; partial file retained: $partial" >&2
    exit 1
  fi
done

actual_size=$(stat -c%s "$partial" 2>/dev/null || echo 0)
if ((actual_size != expected_size)); then
  echo "BLOCKED: CLIP checkpoint incomplete ($actual_size/$expected_size bytes): $partial" >&2
  exit 1
fi
printf '%s  %s\n' "$sha256" "$partial" | sha256sum -c -
mv -n "$partial" "$target"
echo "COMPLETED: $target"
