# Setup

This repository retains the upstream PEARL code and Git history. `origin` is intended for `jsw24000/multimodal-final-project`; `upstream` is `PGSmall/PEARL`. The upstream repository does not contain an explicit `LICENSE` file at the inspected revision, so no new license is asserted here.

## Storage and cache

The host's large ext4 disk is mounted at `/data`, but user `3dsm` currently cannot write to it. See [ENVIRONMENT_REPORT.md](ENVIRONMENT_REPORT.md) for the limited manual permission step. Once writable, the project data root contains `datasets/`, `checkpoints/`, `cache/`, `outputs/`, and `downloads/`. All large artifacts and package caches belong there. Set variables from `.env.example` in the shell after verifying the mount. `.env` is ignored by Git. The default CLIP ViT-B/16 weight download goes to `checkpoints/clip`, and default evaluator work directories go to `outputs/work_dirs`.

```bash
export PEARL_DATA_ROOT=/data/Multimodal-final-project
export HF_HOME=$PEARL_DATA_ROOT/cache/huggingface
export TORCH_HOME=$PEARL_DATA_ROOT/cache/torch
export XDG_CACHE_HOME=$PEARL_DATA_ROOT/cache/xdg
export PIP_CACHE_DIR=$PEARL_DATA_ROOT/cache/pip
export CONDA_PKGS_DIRS=$PEARL_DATA_ROOT/cache/conda_pkgs
bash scripts/check_environment.sh
```

## Dependencies

Upstream documents Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, CUDA 11.8 wheels, MMEngine 0.10.7, MMCV 2.2.0, and MMSegmentation 1.2.2. Its displayed `pip install` URLs contain Markdown link markup and need plain URLs when used. The host has a Blackwell RTX PRO 6000 GPU, so the upstream CUDA 11.8 wheel must be checked for compute capability before installing or attempting inference. No PEARL environment was installed in this phase because its packages and caches should be placed on the large disk. Do not claim protocol equivalence if a newer PyTorch/MMCV combination is required for this GPU; record the actual versions and run an inference smoke test first.

After storage permission, create the environment under the large disk and install only after selecting a Blackwell-compatible PyTorch/MMCV combination:

```bash
conda create --prefix "$PEARL_DATA_ROOT/cache/conda_envs/pearl" python=3.10 -y
conda activate "$PEARL_DATA_ROOT/cache/conda_envs/pearl"
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability(0))'
```

The final installation commands are pending a compatible wheel/extension check. The official dependency versions and links remain in the upstream README below the local project section.
