"""Export ten fixed VOC21 predictions with a method's original implementation."""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
PROJECT = Path(__file__).resolve().parents[2]
INDICES = (0, 56, 111, 222, 333, 444, 555, 777, 1000, 1448)
EXTERNAL = {'sclip': ('SCLIP', 'clip_segmentor'), 'naclip': ('NACLIP', 'naclip')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=('vanilla', 'pa_only', 'full_pearl', *EXTERNAL))
    a = p.parse_args()
    assert ROOT.is_dir() and ROOT.stat().st_dev != Path('/').stat().st_dev
    if a.method in EXTERNAL:
        name, module = EXTERNAL[a.method]
        repo = ROOT/'cache/external_repos'/name
    else:
        repo, module = PROJECT, 'pearl_ovss'
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    from PIL import Image
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules
    register_all_modules()
    importlib.import_module('custom_datasets')
    importlib.import_module(module)
    cfg = Config.fromfile(str(repo/'configs/cfg_voc21.py'))
    cfg.model.clip_path = str(ROOT/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.name_path = str(repo/'configs/cls_voc21.txt')
    cfg.test_dataloader.dataset.data_root = str(ROOT/'datasets/VOCdevkit/VOC2012')
    if a.method == 'vanilla':
        cfg.model.arch, cfg.model.attn_strategy, cfg.model.use_prop = 'vanilla', 'vanilla', 'off'
    elif a.method == 'pa_only':
        cfg.model.arch, cfg.model.attn_strategy, cfg.model.use_prop = 'reduced', 'pearl', 'off'
    elif a.method == 'full_pearl':
        cfg.model.arch, cfg.model.attn_strategy, cfg.model.use_prop = 'reduced', 'pearl', 'on'
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    out = ROOT/'outputs/qualitative/voc21'/a.method
    out.mkdir(parents=True, exist_ok=True)
    if all((out/f'{Path(dataset.get_data_info(i)["img_path"]).stem}.png').is_file() for i in INDICES):
        print(f'REUSE: {a.method} all ten predictions', flush=True)
        return
    model = MODELS.build(cfg.model).cuda().eval()
    for i in INDICES:
        item = dataset[i]
        image_id = Path(dataset.get_data_info(i)['img_path']).stem
        target = out/f'{image_id}.png'
        if target.exists(): continue
        with torch.inference_mode():
            result = model.test_step(pseudo_collate([item]))[0]
        prediction = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        assert prediction.shape == tuple(result.gt_sem_seg.data.shape[-2:])
        Image.fromarray(prediction).save(target)
        print(f'COMPLETED: {a.method} {i} {image_id}', flush=True)


if __name__ == '__main__': main()
