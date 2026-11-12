# Setup

This repository retains the upstream PEARL code and Git history. `origin` is intended for `jsw24000/multimodal-final-project`; `upstream` is `PGSmall/PEARL`. The upstream repository does not contain an explicit `LICENSE` file at the inspected revision, so no new license is asserted here.

## Storage and cache

The host's large ext4 disk is mounted at `/data`, and user `3dsm` can write to the project subdirectory `/data/Multimodal-final-project`. See [ENVIRONMENT_REPORT.md](ENVIRONMENT_REPORT.md) for the mount check. The project data root contains `datasets/`, `checkpoints/`, `cache/`, `outputs/`, and `downloads/`. All large artifacts and package caches belong there. Set variables from `.env.example` in the shell after verifying the mount. `.env` is ignored by Git. The default CLIP ViT-B/16 weight download goes to `checkpoints/clip`, and default evaluator work directories go to `outputs/work_dirs`.

```bash
export PEARL_DATA_ROOT=/data/Multimodal-final-project
export HF_HOME=$PEARL_DATA_ROOT/cache/huggingface
export TORCH_HOME=$PEARL_DATA_ROOT/cache/torch
export XDG_CACHE_HOME=$PEARL_DATA_ROOT/cache/xdg
export PIP_CACHE_DIR=$PEARL_DATA_ROOT/cache/pip
export CONDA_PKGS_DIRS=$PEARL_DATA_ROOT/cache/conda_pkgs
bash scripts/check_environment.sh
```

## PEARL runtime on this host

The isolated environment is `/data/Multimodal-final-project/cache/conda_envs/pearl`. It was cloned from the existing `wan22_videogpa` environment to reuse its **tested** PyTorch 2.8.0+cu128 build; no packages were installed into the source environment. The clone contains additional packages from that source environment. Its broken cloned `pip` was replaced inside the PEARL environment, and the unrelated `decord` package was removed because `pip check` reported that it was unsupported on this platform. `pip check` now reports no broken requirements.

| Component | Installed version | Note |
| --- | --- | --- |
| Python | 3.10.20 | From the source environment |
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 | CUDA tensor operation and `sm_120` support checked |
| MMEngine / MMSegmentation | 0.10.7 / 1.2.2 | MMSeg module registration checked |
| MMCV | 2.1.0 | Full package with CPU C++ ops, built from source against this PyTorch; CUDA ops were not built |
| setuptools | 80.9.0 | Supplies `pkg_resources`, imported by the bundled CLIP code |

The upstream README requests PyTorch 2.4.1+cu118 and MMCV 2.2.0. PyTorch 2.4.1+cu118 predates Blackwell support, while MMSegmentation 1.2.2 requires `mmcv<2.2.0` in its own runtime check. These hardware and dependency changes are recorded explicitly; do not describe this environment as the exact upstream software stack. The displayed upstream pip URLs also contain Markdown syntax and must be converted to plain URLs before shell use.

The full MMCV package is necessary: `mmcv-lite` 2.1.0 cannot load `mmcv._ext` during MMSegmentation registration. The PEARL model and VOC21 data pipeline do not call MMCV CUDA ops, so a CPU C++ ops build was used. A later experiment using MMCV CUDA ops will require a separate CUDA-enabled build and validation. The locally built wheel is under `cache/wheels/` on the data disk; SHA256: `f356d0508ba9b5b29158134c232c78a1a7222043f5b83eb156f18005b5514af6`.

Activate and check the installed environment from the project root:

```bash
export PEARL_DATA_ROOT=/data/Multimodal-final-project
export HF_HOME=$PEARL_DATA_ROOT/cache/huggingface
export TORCH_HOME=$PEARL_DATA_ROOT/cache/torch
export XDG_CACHE_HOME=$PEARL_DATA_ROOT/cache/xdg
export PIP_CACHE_DIR=$PEARL_DATA_ROOT/cache/pip
export CONDA_PKGS_DIRS=$PEARL_DATA_ROOT/cache/conda_pkgs
conda activate "$PEARL_DATA_ROOT/cache/conda_envs/pearl"
python scripts/check_pearl_environment.py --check-voc21
python -m pip check
```

The first PEARL inference also needs the official CLIP ViT-B/16 checkpoint. Download it to the data disk with the resumable script below. The script verifies its 350,837,078-byte size and the SHA256 encoded in the official URL before renaming the `.part` file. The checkpoint remains outside Git.

```bash
bash scripts/download_clip_weight.sh
```

After the checkpoint checksum and VOC21 data check pass, run the single-image inference smoke test:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python scripts/smoke_test_voc21.py
```

This loads one validation image and its mask, runs full PEARL (`--attn pearl --prop on`), and checks the output shape. It does not compute validation mIoU or write predictions. The first full benchmark is `python eval.py --config configs/cfg_voc21.py --attn pearl --prop on`; its work directory is routed to the data disk. Run it only after confirming the environment and protocol checks above.

For a clean rebuild on the same GPU, first verify the `/data` mount, set the cache variables above, and create a new prefix on that disk. The following are the relevant package pins; the installed environment above was created by cloning `wan22_videogpa`, then installing the missing OpenMMLab packages.

```bash
conda create --prefix "$PEARL_DATA_ROOT/cache/conda_envs/pearl-new" python=3.10 pip -y
conda activate "$PEARL_DATA_ROOT/cache/conda_envs/pearl-new"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'setuptools==80.9.0' wheel ninja psutil ftfy regex
python -m pip install 'mmengine==0.10.7' 'mmsegmentation==1.2.2'
CUDA_VISIBLE_DEVICES='' MMCV_WITH_OPS=1 MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-binary mmcv 'mmcv==2.1.0'
```

## Terminal proxy on this host

The desktop system proxy does not automatically reach every shell. On this host, Clash Verge currently listens at `127.0.0.1:7897`. When `wget` prints `Connecting to 127.0.0.1:7897`, it is using the proxy. Export the variables in the same terminal as a download command if direct connections time out:

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=$http_proxy
```

This is a host-specific convenience, not a project dependency. Download `.part` files remain on the large disk and can be resumed by rerunning the script.
