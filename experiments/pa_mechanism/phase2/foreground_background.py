"""Same-VOC21-prediction foreground/background and corrected-pixel audit."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

import custom_datasets  # noqa: F401
import pearl_ovss  # noqa: F401
from experiments.pa_mechanism.phase2.run_matrix import configure


ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
OUT = ROOT/'outputs/pa_mechanism/phase2/metrics/voc21_fg_bg_1449.jsonl'
SUMMARY = Path('results/pa_mechanism_phase2_voc21_fg_bg.json')
VARIANTS = ('qk', 'qq', 'kk', 'sclip_mean', 'ns5_pa', 'ns5_pakk')
PAIRS = (('qq', 'ns5_pa'), ('sclip_mean', 'ns5_pakk'))


@torch.inference_mode()
def main():
    register_all_modules()
    cfg = Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path = str(ROOT/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.arch = 'reduced'
    cfg.model.use_prop = 'off'
    cfg.model.attn_strategy = 'qk'
    ds = DATASETS.build(cfg.test_dataloader.dataset)
    model = MODELS.build(cfg.model).cuda().eval()
    model.net.visual.phase2_cache_enabled = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.open():
            done.add(json.loads(line)['index'])
    with OUT.open('a', buffering=1) as stream:
        for index in range(len(ds)):
            if index in done:
                continue
            model.net.visual._phase2_prefix_cache = []
            item = ds[index]
            preds = {}
            gt = None
            for variant in VARIANTS:
                configure(model.net.visual, variant)
                sample = model.test_step(pseudo_collate([item]))[0]
                if gt is None:
                    gt = sample.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.int16)
                preds[variant] = sample.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int16)
            valid = gt != 255
            fg = valid & (gt != 0)
            bg = valid & (gt == 0)
            metrics = {}
            for variant, pred in preds.items():
                correct = pred == gt
                metrics[variant] = dict(correct=int((valid & correct).sum()),
                                        foreground_correct=int((fg & correct).sum()),
                                        background_correct=int((bg & correct).sum()),
                                        foreground_predicted_background=int((fg & (pred == 0)).sum()),
                                        background_predicted_foreground=int((bg & (pred != 0)).sum()))
            changes = {}
            for first, second in PAIRS:
                a = preds[first] == gt
                b = preds[second] == gt
                changes[f'{first}_to_{second}'] = dict(
                    corrected=int((valid & ~a & b).sum()), damaged=int((valid & a & ~b).sum()),
                    foreground_corrected=int((fg & ~a & b).sum()), foreground_damaged=int((fg & a & ~b).sum()),
                    background_corrected=int((bg & ~a & b).sum()), background_damaged=int((bg & a & ~b).sum()))
            stream.write(json.dumps(dict(index=index, image_id=Path(ds.get_data_info(index)['img_path']).stem,
                                         valid_pixels=int(valid.sum()), foreground_pixels=int(fg.sum()),
                                         background_pixels=int(bg.sum()), variants=metrics, pairs=changes),
                                    separators=(',', ':'))+'\n')
            if (index+1) % 50 == 0:
                print(index+1, '/', len(ds), flush=True)
    totals = dict(images=0, valid_pixels=0, foreground_pixels=0, background_pixels=0,
                  variants={variant: {} for variant in VARIANTS}, pairs={f'{a}_to_{b}': {} for a, b in PAIRS})
    for line in OUT.open():
        row = json.loads(line)
        totals['images'] += 1
        for key in ('valid_pixels', 'foreground_pixels', 'background_pixels'):
            totals[key] += row[key]
        for group in ('variants', 'pairs'):
            for variant, metrics in row[group].items():
                target = totals[group][variant]
                for key, value in metrics.items():
                    target[key] = target.get(key, 0)+value
    assert totals['images'] == len(ds)
    SUMMARY.write_text(json.dumps(totals, indent=2))
    print(SUMMARY, totals['images'], flush=True)


if __name__ == '__main__':
    main()
