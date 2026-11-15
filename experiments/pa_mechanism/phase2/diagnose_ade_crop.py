"""One deterministic released-grid ADE crop per validation image, with exact GT crop."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

import custom_datasets  # noqa: F401
import pearl_ovss  # noqa: F401
from experiments.pa_mechanism.attention_variants import attention


ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
METRICS = ROOT/'outputs/pa_mechanism/phase2/metrics'
VARIANTS = (('qk', 'qk', 'polar'), ('qq', 'qq', 'polar'),
            ('kk', 'kk', 'polar'), ('pure_pa_ns5', 'pure_pa', 'polar'),
            ('pure_pa_svd', 'pure_pa', 'svd'))


def crop_coordinates(h, w, crop, stride, index):
    hn = max(h-crop+stride-1, 0)//stride+1
    wn = max(w-crop+stride-1, 0)//stride+1
    chosen = index % (hn*wn)
    hi, wi = divmod(chosen, wn)
    y2, x2 = min(hi*stride+crop, h), min(wi*stride+crop, w)
    y1, x1 = max(y2-crop, 0), max(x2-crop, 0)
    return y1, y2, x1, x2, chosen, hn*wn


def semantic_metrics(attn, labels, purity):
    valid = labels != 255
    a = attn[1:, 1:].float()[valid][:, valid]
    a = a/a.sum(-1, keepdim=True).clamp_min(1e-9)
    y = labels[valid]
    p = purity[valid]
    n = len(y)
    if n < 2:
        return None
    same = y[:, None] == y[None, :]
    diagonal = torch.eye(n, dtype=torch.bool, device=y.device)
    same_rows = (a*same).sum(-1)
    off_rows = (a*(same & ~diagonal)).sum(-1)
    key_area = same.float().mean(-1).clamp_min(1e-9)
    high, mixed = p >= .9, p < .9
    unit = F.normalize(a, dim=-1)
    row_cos = unit@unit.T
    entropy = -(a*a.clamp_min(1e-9).log()).sum(-1)
    return dict(valid_patches=n, high_purity_fraction=float(high.float().mean()),
                same=float(same_rows.mean()), off_same=float(off_rows.mean()),
                same_enrichment=float((same_rows/key_area).mean()),
                high_same=float(same_rows[high].mean()) if high.any() else None,
                mixed_same=float(same_rows[mixed].mean()) if mixed.any() else None,
                within_cos=float(row_cos[same & ~diagonal].mean()) if (same & ~diagonal).any() else None,
                between_cos=float(row_cos[~same].mean()) if (~same).any() else None,
                entropy=float(entropy.mean()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=0, help='0 means full 2,000 images')
    args = p.parse_args()
    register_all_modules()
    cfg = Config.fromfile('configs/cfg_ade20k.py')
    cfg.model.clip_path = str(ROOT/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.arch = 'reduced'
    cfg.model.use_prop = 'off'
    cfg.model.attn_strategy = 'qq'
    ds = DATASETS.build(cfg.test_dataloader.dataset)
    total = len(ds) if args.count == 0 else min(args.count, len(ds))
    model = MODELS.build(cfg.model).cuda().eval()
    captures = []
    hook = model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(
        lambda _m, _i, out: captures.__setitem__(slice(None), [out.detach()]))
    METRICS.mkdir(parents=True, exist_ok=True)
    path = METRICS/f'ade20k_crop_attention_{total}.jsonl'
    done = set()
    if path.exists():
        for line in path.open():
            done.add(json.loads(line)['index'])
    with path.open('a', buffering=1) as stream:
        for index in range(total):
            if index in done:
                continue
            item = ds[index]
            processed = model.data_preprocessor(pseudo_collate([item]), training=False)
            img = processed['inputs']
            gt = processed['data_samples'][0].gt_sem_seg.data.squeeze().cuda()
            h_img, w_img = img.shape[-2:]
            # ADE annotations remain at original resolution in the packed sample.
            # Map them to the resized model input before selecting official slide coordinates.
            if gt.shape != (h_img, w_img):
                gt = F.interpolate(gt[None, None].float(), size=(h_img, w_img), mode='nearest').long().squeeze()
            y1, y2, x1, x2, chosen, n_crops = crop_coordinates(h_img, w_img, model.slide_crop, model.slide_stride, index)
            crop = img[:, :, y1:y2, x1:x2]
            gt_crop = gt[y1:y2, x1:x2]
            captures.clear()
            model.forward_feature(crop)
            z = captures[-1]
            layer = model.net.visual.transformer.resblocks[-1].attn
            q, k, _ = F.linear(z, layer.in_proj_weight, layer.in_proj_bias).chunk(3, dim=-1)
            t, b, d = q.shape
            heads = layer.num_heads
            q = q.view(t, b*heads, d//heads).transpose(0, 1).float()
            k = k.view(t, b*heads, d//heads).transpose(0, 1).float()
            hp, wp = crop.shape[-2]//16, crop.shape[-1]//16
            assert t == hp*wp+1
            label_pixels = F.interpolate(gt_crop[None, None].float(), size=(hp*16, wp*16), mode='nearest').long().squeeze()
            valid_pixels = (label_pixels != 255).float()[None, None]
            one_hot = F.one_hot(label_pixels.clamp(0, 149), 150).permute(2, 0, 1).float()[None]
            area = F.avg_pool2d(one_hot*valid_pixels, 16, 16).flatten(2)[0]
            denom = F.avg_pool2d(valid_pixels, 16, 16).flatten().clamp_min(1e-9)
            purity, labels = area.max(0)
            purity = purity/denom
            labels[denom < .5] = 255
            head_rows = []
            for name, strategy, solver in VARIANTS:
                matrix = attention(q, k, strategy, solver=solver)
                for head in range(heads):
                    head_rows.append(dict(variant=name, head=head,
                                          metrics=semantic_metrics(matrix[head], labels, purity)))
            stream.write(json.dumps(dict(index=index, image_id=Path(ds.get_data_info(index)['img_path']).stem,
                                         crop=dict(y1=y1, y2=y2, x1=x1, x2=x2, chosen=chosen,
                                                   released_grid_crops=n_crops, resized_image_shape=[h_img, w_img]),
                                         heads=head_rows), separators=(',', ':'))+'\n')
            if (index+1) % 25 == 0 or index+1 == total:
                print(index+1, '/', total, flush=True)
    hook.remove()
    sums = {name: {} for name, _, _ in VARIANTS}
    counts = {name: {} for name, _, _ in VARIANTS}
    images = 0
    for line in path.open():
        row = json.loads(line)
        images += 1
        for entry in row['heads']:
            metrics = entry['metrics']
            if metrics is None:
                continue
            name = entry['variant']
            for key, value in metrics.items():
                if value is None:
                    continue
                sums[name][key] = sums[name].get(key, 0.)+value
                counts[name][key] = counts[name].get(key, 0)+1
    assert images == total
    aggregate = {name: {key: sums[name][key]/counts[name][key] for key in sums[name]}
                 for name in sums}
    out = Path(f'results/pa_mechanism_phase2_ade_crop_attention_{total}.json')
    out.write_text(json.dumps(dict(dataset='ade20k', images=images, crops_per_image=1,
                                   crop_selection='index modulo official slide-grid count',
                                   heads_per_crop=heads, means=aggregate,
                                   valid_head_counts=counts), indent=2))
    print(out, images, flush=True)


if __name__ == '__main__':
    main()
