"""Resumable full-split Phase 3 evaluation with one confusion row per image/variant."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

import custom_datasets  # noqa: F401
import pearl_ovss  # noqa: F401


ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
RUNS = ROOT/'outputs/pa_mechanism/phase3/runs'
NCLASSES = {'voc21': 21, 'voc20': 20, 'ade20k': 150}
BASE_COMMIT = '62a0cb25d4aab7572f178374596333d1836c2c2b'


def configure(visual, name, original_alpha):
    """All Phase 3 alignment variants use exact SVD; no variant updates model weights."""
    visual.align_solver = 'svd'
    visual.polar_iters = 5
    visual.gamma = 1.
    visual.beta = .5
    visual.lambda_csa = .5
    visual.alpha_kk = original_alpha
    if name == 'qq':
        visual.attn_strategy = 'qq'
    elif name == 'kk':
        visual.attn_strategy = 'kk'
    elif name == 'csa':
        visual.attn_strategy = 'sclip_mean'
    elif name == 'pa':
        visual.attn_strategy = 'pure_pa'
    elif name.startswith('g') and name[1:].isdigit():
        visual.attn_strategy = 'gamma'
        visual.gamma = int(name[1:])/100
    elif name.startswith('kb') and name[2:].isdigit():
        visual.attn_strategy = 'kk_pa_beta'
        visual.beta = int(name[2:])/100
    elif name.startswith('cb') and name[2:].isdigit():
        visual.attn_strategy = 'csa_pa_beta'
        visual.beta = int(name[2:])/100
    elif name.startswith('cw') and name[2:].isdigit():
        visual.attn_strategy = 'csa_weighted'
        visual.lambda_csa = int(name[2:])/100
    elif name.startswith('rk_g'):
        pieces = name.split('_')
        if len(pieces) != 3 or not pieces[1][1:].isdigit() or not pieces[2].startswith('a') or not pieces[2][1:].isdigit():
            raise ValueError(name)
        visual.attn_strategy = 'residual_kk'
        visual.gamma = int(pieces[1][1:])/100
        visual.alpha_kk = int(pieces[2][1:])/100
    else:
        raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=NCLASSES, required=True)
    parser.add_argument('--tlp', choices=('on', 'off'), default='off')
    parser.add_argument('--variants', required=True, help='Comma-separated g150,kb025,cb050,cw075,pa,...')
    parser.add_argument('--count', type=int, default=0, help='0 means the complete validation split')
    parser.add_argument('--seed', type=int, default=24003)
    args = parser.parse_args()
    variants = args.variants.split(',')
    if len(set(variants)) != len(variants):
        raise ValueError('duplicate variants')
    register_all_modules()
    cfg = Config.fromfile(f'configs/cfg_{args.dataset}.py')
    cfg.model.clip_path = str(ROOT/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.arch = 'reduced'
    cfg.model.use_prop = args.tlp
    cfg.model.attn_strategy = 'qk'
    original_alpha = float(cfg.model.get('alpha_kk', .08))
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if args.count == 0:
        indices = list(range(len(dataset)))
    else:
        indices = sorted(map(int, np.random.default_rng(args.seed).choice(
            len(dataset), min(args.count, len(dataset)), replace=False)))
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f'{args.dataset}_{args.tlp}_{len(indices)}_{args.seed}_{"-".join(variants)}'
    path = RUNS/f'{name}.jsonl'
    config_path = RUNS/f'{name}.config.json'
    config = dict(phase=3, dataset=args.dataset, tlp=args.tlp, variants=variants,
                  indices=indices, seed=args.seed, nclasses=NCLASSES[args.dataset],
                  checkpoint=cfg.model.clip_path, arch='reduced', solver='svd',
                  original_alpha_kk=original_alpha,
                  slide_crop=int(cfg.model.get('slide_crop', 224)),
                  slide_stride=int(cfg.model.get('slide_stride', 112)),
                  name_path=cfg.model.name_path,
                  git_base_commit=BASE_COMMIT)
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('run configuration mismatch')
    else:
        config_path.write_text(json.dumps(config, indent=2))
    done = set()
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            pair = (row['index'], row['variant'])
            if pair in done:
                raise RuntimeError(f'duplicate record {pair}')
            done.add(pair)
    model = MODELS.build(cfg.model).cuda().eval()
    model.net.visual.phase2_cache_enabled = True
    model.net.visual.phase3_svd_cache_enabled = True
    n = NCLASSES[args.dataset]
    with path.open('a', buffering=1) as stream:
        for position, index in enumerate(indices):
            model.net.visual._phase2_prefix_cache = []
            model.net.visual._phase3_svd_cache = []
            item = None
            for variant in variants:
                if (index, variant) in done:
                    continue
                if item is None:
                    item = dataset[index]
                configure(model.net.visual, variant, original_alpha)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    result = model.test_step(pseudo_collate([item]))[0]
                torch.cuda.synchronize()
                pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
                gt = result.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
                if pred.shape != gt.shape:
                    raise RuntimeError((pred.shape, gt.shape))
                good = gt != 255
                if good.any() and (gt[good].max() >= n or pred[good].max() >= n or pred[good].min() < 0):
                    raise RuntimeError('invalid class id')
                counts = np.bincount(gt[good]*n+pred[good], minlength=n*n)
                nonzero = np.flatnonzero(counts)
                row = dict(index=index, image_id=Path(dataset.get_data_info(index)['img_path']).stem,
                           variant=variant, confusion_indices=nonzero.tolist(),
                           confusion_counts=counts[nonzero].tolist(), seconds=time.perf_counter()-start,
                           gt_pixels=int(good.sum()))
                stream.write(json.dumps(row, separators=(',', ':'))+'\n')
            if (position+1) % 25 == 0 or position+1 == len(indices):
                print(args.dataset, args.tlp, position+1, '/', len(indices), flush=True)


if __name__ == '__main__':
    main()
