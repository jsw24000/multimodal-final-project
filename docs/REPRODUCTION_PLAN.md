# Reproduction plan

## Phase 1: establish a faithful baseline

1. Record upstream commit, dataset source/archive hashes, conversion tool revision, config file, Python/PyTorch/MMCV/MMEngine/MMSeg versions, GPU capability, and command in a small metadata file under `results/`.
2. Verify VOC2012 validation image/mask pairs and `val.txt`, then check the loaded MMSegmentation dataset length and one batch. The first benchmark is VOC21, with its background class retained.
3. Run a one-image PEARL smoke test with fixed config and CLIP ViT-B/16 checkpoint. Keep predictions and logs under `${PEARL_DATA_ROOT}/outputs`; store only light metrics and provenance in `results/`.
4. Run the full validation evaluation only after the environment and dataset checks pass. Compare to the paper with exact protocol and separate any compatibility changes required by this Blackwell GPU.

## Later experiments

- Performance: Vanilla CLIP, PEARL PA only, PEARL TLP only, full PEARL, SCLIP, NACLIP.
- Attention and features: final attention, Q/K distribution and alignment error, foreground/background and same-class attention mass, key norms and high-norm background tokens, PCA/SVD directions.
- Mechanisms: Procrustes/random orthogonal rotation, SVD/Newton–Schulz, QK/QQ/KK, TLP off, and official TLP versus paper-equation TLP.

New work belongs in `experiments/`. Changes to the official evaluation protocol should be explicit named variants rather than silent edits to `configs/` or `pearl/`. Shared datasets and checkpoint paths come from `project_paths.py`.
