# Environment report

Inspection date: 2026-11-12 (Asia/Shanghai). Commands: `pwd`, `whoami`, `lsblk -f`, `df -hT`, `findmnt`, `stat`, `getfacl`, `nvidia-smi`, `nvcc --version`, `python --version`, `conda --version`, `git --version`, and connectivity HEAD requests.

| Item | Observed result | Status |
| --- | --- | --- |
| User | `3dsm` | COMPLETED |
| Project directory | `/home/3dsm/Desktop/JSWprobe/Multimodal-final-project` was empty before clone; its parent is a separate Git repository with unrelated changes | COMPLETED |
| Large disk | `/dev/nvme0n1` mounted read/write at `/data`, ext4, 7.0 TB total, 6.6 TB available | COMPLETED |
| Write access | `/data` is owned by `root:root`, mode `755`; the project subdirectory `/data/Multimodal-final-project` is now owned by `3dsm` and writable | COMPLETED |
| System disk | `/dev/nvme1n1p2` mounted at `/`, ext4, 938 GB total, 146 GB available | COMPLETED |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB | COMPLETED |
| Driver / CUDA toolkit | NVIDIA 570.211.01, reported CUDA 12.8; nvcc 12.8.93 | COMPLETED |
| Python / Conda | Python 3.13.11 in base; Conda 26.1.0 | COMPLETED |
| Git / GitHub CLI | Git 2.34.1; `gh` not installed | BLOCKED for host CLI authentication |
| Git author identity | No global identity; repository-local `jsw24000` and GitHub noreply address tied to verified account ID 192295254 configured | COMPLETED |
| Python packages in base | `torch`, `mmengine`, `mmseg` absent; Pillow and NumPy present | BLOCKED for model test |
| Network | GitHub and PyTorch wheel index returned HTTP 200; direct official VOC2012/VOC2010 archive URLs and ADE archive returned HTTP 200; COCO HTTPS had a certificate mismatch; SSH to GitHub ports 22 and 443 timed out | PARTIAL |
| GitHub repository | `jsw24000/multimodal-final-project` now exists, Public, empty, `fork: false`; connected account is `jsw24000` | PENDING push; GitHub fork relation absent |

The target data root now exists on `/dev/nvme0n1` and is writable by `3dsm`. VOC2010 raw data and verified VOC2012 data and archives are stored there. ADE20K is downloading to a resumable `.part` file; its log is at `/data/Multimodal-final-project/downloads/ade20k-download.log`. No data, model, package cache, or large experimental output was written to the system disk. The official PEARL README pins PyTorch 2.4.1 with CUDA 11.8. This Blackwell GPU needs an explicit wheel compatibility check; do not assume the official pin can execute CUDA kernels on it. The local Git commit preserves upstream PEARL history, but the newly created GitHub repository is an independent repository, not a native Fork. The host cannot push over HTTPS without local GitHub authentication.

Storage access was granted to the project subdirectory without changing mount configuration or existing data. The previously proposed manual command was:

```bash
sudo install -d -o 3dsm -g 3dsm -m 750 /data/Multimodal-final-project
```

The assistant did **not** execute this sudo command. `bash scripts/check_environment.sh` and `findmnt -T /data/Multimodal-final-project` now confirm the data root is on `/dev/nvme0n1` and writable. The environment and download scripts were corrected to require write access to the project directory rather than the mount root.

## PEARL runtime update: 2026-11-13

An isolated Conda environment now exists at `/data/Multimodal-final-project/cache/conda_envs/pearl`. It was cloned from `wan22_videogpa` without modifying that source. The PEARL environment has Python 3.10.20, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, MMEngine 0.10.7, full MMCV 2.1.0 with CPU C++ ops, MMSegmentation 1.2.2, and setuptools 80.9.0. The bundled CLIP implementation requires `pkg_resources`, which setuptools 83 in the source environment no longer supplies. The cloned `pip` was repaired only in the new environment. `pip check` passes after removing an unrelated, unsupported `decord` package from the clone.

The check script `python scripts/check_pearl_environment.py --check-voc21` passed: CUDA tensor computation worked on the RTX PRO 6000 (`sm_120`), `mmcv._ext` loaded, MMSegmentation and PEARL registered, and one sample loaded from the 1,449-entry VOC21 validation dataset. The environment uses a documented Blackwell-compatible software variant because the upstream PyTorch 2.4.1+cu118 pin predates Blackwell support and the upstream MMCV 2.2.0 pin conflicts with the MMSegmentation 1.2.2 runtime bound `mmcv<2.2.0`. See [SETUP.md](SETUP.md) for exact versions and rebuilding steps.

The official CLIP ViT-B/16 checkpoint was downloaded to `/data/Multimodal-final-project/checkpoints/clip/ViT-B-16.pt` with resumable direct access and verified against SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` (350,837,078 bytes). `python scripts/smoke_test_voc21.py` passed with full PEARL on one validation image: prediction and mask shape `(1, 366, 500)`, predicted label IDs `[0, 1]`, and peak allocated GPU memory 840.2 MiB. This is an integration test, not a full validation mIoU result.

GitHub CLI 2.103.3 is now available at `/data/Multimodal-final-project/cache/tools/gh_2.103.3_linux_amd64/bin/gh`; `gh api user` returned `jsw24000`. The 2026-11-12 GitHub CLI/auth status in the inspection table above is historical.
