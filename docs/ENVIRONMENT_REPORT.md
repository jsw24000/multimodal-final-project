# Environment report

Inspection date: 2026-11-12 (Asia/Shanghai). Commands: `pwd`, `whoami`, `lsblk -f`, `df -hT`, `findmnt`, `stat`, `getfacl`, `nvidia-smi`, `nvcc --version`, `python --version`, `conda --version`, `git --version`, and connectivity HEAD requests.

| Item | Observed result | Status |
| --- | --- | --- |
| User | `3dsm` | COMPLETED |
| Project directory | `/home/3dsm/Desktop/JSWprobe/Multimodal-final-project` was empty before clone; its parent is a separate Git repository with unrelated changes | COMPLETED |
| Large disk | `/dev/nvme0n1` mounted read/write at `/data`, ext4, 7.0 TB total, 6.6 TB available | COMPLETED |
| Write access | `/data` is owned by `root:root`, mode `755`; `3dsm` has no write access or ACL grant | BLOCKED |
| System disk | `/dev/nvme1n1p2` mounted at `/`, ext4, 938 GB total, 146 GB available | COMPLETED |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB | COMPLETED |
| Driver / CUDA toolkit | NVIDIA 570.211.01, reported CUDA 12.8; nvcc 12.8.93 | COMPLETED |
| Python / Conda | Python 3.13.11 in base; Conda 26.1.0 | COMPLETED |
| Git / GitHub CLI | Git 2.34.1; `gh` not installed | BLOCKED for CLI fork |
| Git author identity | No global identity; repository-local `jsw24000` and GitHub noreply address tied to verified account ID 192295254 configured | COMPLETED |
| Python packages in base | `torch`, `mmengine`, `mmseg` absent; Pillow and NumPy present | BLOCKED for model test |
| Network | GitHub and PyTorch wheel index returned HTTP 200; target GitHub repo returned 404; direct official VOC2012/VOC2010 archive URLs and ADE archive returned HTTP 200; COCO HTTPS had a certificate mismatch | PARTIAL |
| Connected GitHub account | `jsw24000`, verified by GitHub connector; connector does not expose repository fork/create | BLOCKED for fork |

The target data root has not been created. No data, model, package cache, or large experimental output was written to the system disk. The official PEARL README pins PyTorch 2.4.1 with CUDA 11.8. This Blackwell GPU needs an explicit wheel compatibility check; do not assume the official pin can execute CUDA kernels on it.

The administrator or disk owner should grant access only to a new project directory on `/data`, without changing mount configuration or existing data:

```bash
sudo install -d -o 3dsm -g 3dsm -m 750 /data/Multimodal-final-project
```

This command is a proposed manual action and was **not** executed. Afterwards run `bash scripts/check_environment.sh` and verify `findmnt -T /data/Multimodal-final-project` reports `/dev/nvme0n1` before downloading.
