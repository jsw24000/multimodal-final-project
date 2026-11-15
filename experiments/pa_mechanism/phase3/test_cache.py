"""Exact prediction equivalence with the opt-in SVD alignment cache."""
import os
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

import custom_datasets  # noqa: F401
import pearl_ovss  # noqa: F401
from experiments.pa_mechanism.phase3.run_phase3 import configure


@torch.inference_mode()
def test():
    root = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
    register_all_modules()
    cfg = Config.fromfile('configs/cfg_ade20k.py')
    cfg.model.clip_path = str(root/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.arch = 'reduced'
    cfg.model.use_prop = 'off'
    cfg.model.attn_strategy = 'qq'
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    model = MODELS.build(cfg.model).cuda().eval()
    item = dataset[0]
    visual = model.net.visual
    original_alpha = float(cfg.model.alpha_kk)
    uncached = {}
    for name in ('g150', 'kb050', 'cb050'):
        configure(visual, name, original_alpha)
        visual.phase2_cache_enabled = False
        visual.phase3_svd_cache_enabled = False
        uncached[name] = model.test_step(pseudo_collate([item]))[0].pred_sem_seg.data.clone()
    visual.phase2_cache_enabled = True
    visual.phase3_svd_cache_enabled = True
    visual._phase2_prefix_cache = []
    visual._phase3_svd_cache = []
    for name in ('g150', 'kb050', 'cb050'):
        configure(visual, name, original_alpha)
        cached = model.test_step(pseudo_collate([item]))[0].pred_sem_seg.data
        assert torch.equal(uncached[name], cached), name


if __name__ == '__main__':
    test()
    print('Phase 3 ADE cache test: PASS')
