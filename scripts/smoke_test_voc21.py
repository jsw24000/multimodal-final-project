"""Run PEARL on one VOC21 validation image without evaluating the full split."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="validation image index")
    parser.add_argument("--attn", choices=("pearl", "vanilla"), default="pearl")
    parser.add_argument("--prop", choices=("on", "off"), default="on")
    args = parser.parse_args()

    from project_paths import checkpoint_path, require_large_disk

    require_large_disk()
    checkpoint = Path(checkpoint_path("clip", "ViT-B-16.pt"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CLIP checkpoint missing: {checkpoint}; run scripts/download_clip_weight.sh")

    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules
    import custom_datasets  # noqa: F401
    import pearl_ovss  # noqa: F401

    register_all_modules()
    cfg = Config.fromfile(str(PROJECT_ROOT / "configs/cfg_voc21.py"))
    cfg.model.clip_path = str(checkpoint)
    cfg.model.name_path = str(PROJECT_ROOT / "configs/cls_voc21.txt")
    cfg.model.attn_strategy = args.attn
    cfg.model.use_prop = args.prop
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index {args.index} outside VOC21 validation split of {len(dataset)}")

    model = MODELS.build(cfg.model).cuda().eval()
    batch = pseudo_collate([dataset[args.index]])
    with torch.inference_mode():
        result = model.test_step(batch)
    prediction = result[0].pred_sem_seg.data
    ground_truth = result[0].gt_sem_seg.data
    assert prediction.shape == ground_truth.shape, "Prediction and mask sizes differ"
    print(f"dataset_len={len(dataset)} index={args.index} attn={args.attn} prop={args.prop}")
    print(f"prediction_shape={tuple(prediction.shape)} labels={torch.unique(prediction).cpu().tolist()}")
    print(f"gpu_peak_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print("VOC21 single-image smoke test: OK")


if __name__ == "__main__":
    main()
