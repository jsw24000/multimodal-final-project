"""Run an unmodified external OVSS baseline against shared local validation data."""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
REPOS = {'sclip': ('SCLIP', 'clip_segmentor'), 'naclip': ('NACLIP', 'naclip'),
         'clearclip': ('ClearCLIP', 'clearclip_segmentor')}
DATA = {'voc21': 'VOCdevkit/VOC2012', 'voc20': 'VOCdevkit/VOC2012',
        'ade20k': 'ADEChallengeData2016'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=REPOS)
    p.add_argument('dataset', choices=DATA)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    assert ROOT.is_dir() and ROOT.stat().st_dev != Path('/').stat().st_dev
    repo_name, module = REPOS[a.method]
    repo = ROOT / 'cache/external_repos' / repo_name
    checkpoint = ROOT / 'checkpoints/clip/ViT-B-16.pt'
    assert (repo / '.git').is_dir() and checkpoint.is_file()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import Runner
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules
    register_all_modules()
    importlib.import_module('custom_datasets')
    importlib.import_module(module)
    cfg = Config.fromfile(str(repo / 'configs' / f'cfg_{a.dataset}.py'))
    cfg.model.clip_path = str(checkpoint)
    cfg.model.name_path = str(repo / 'configs' / f'cls_{a.dataset}.txt')
    cfg.test_dataloader.dataset.data_root = str(ROOT / 'datasets' / DATA[a.dataset])
    cfg.work_dir = str(ROOT / 'outputs/performance' / f'{a.method}_{a.dataset}')
    if a.smoke:
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        model = MODELS.build(cfg.model).cuda().eval()
        with torch.inference_mode():
            pred = model.test_step(pseudo_collate([dataset[0]]))
        assert pred[0].pred_sem_seg.data.shape == pred[0].gt_sem_seg.data.shape
        print(f'SMOKE OK: {a.method} {a.dataset} {len(dataset)} samples', flush=True)
    else:
        Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
