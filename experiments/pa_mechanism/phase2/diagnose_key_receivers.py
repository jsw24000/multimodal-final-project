"""Full VOC21 check of dominant receiving Keys and their projected K norms."""
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
OUT = ROOT / 'outputs/pa_mechanism/phase2/metrics/voc21_key_receivers_1449.jsonl'


@torch.inference_mode()
def main():
    register_all_modules()
    cfg = Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path = str(ROOT / 'checkpoints/clip/ViT-B-16.pt')
    cfg.model.arch = 'reduced'
    cfg.model.use_prop = 'off'
    cfg.model.attn_strategy = 'qq'
    ds = DATASETS.build(cfg.test_dataloader.dataset)
    model = MODELS.build(cfg.model).cuda().eval()
    capture = []
    hook = model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(
        lambda _m, _i, o: capture.__setitem__(slice(None), [o.detach()]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.open():
            done.add(json.loads(line)['index'])
    with OUT.open('a', buffering=1) as stream:
        for index in range(len(ds)):
            if index in done:
                continue
            item = ds[index]
            capture.clear()
            sample = model.test_step(pseudo_collate([item]))[0]
            z = capture[-1]
            layer = model.net.visual.transformer.resblocks[-1].attn
            q, k, _ = F.linear(z, layer.in_proj_weight, layer.in_proj_bias).chunk(3, dim=-1)
            t, b, d = q.shape
            h = layer.num_heads
            q = q.view(t, b*h, d//h).transpose(0, 1).float()
            k = k.view(t, b*h, d//h).transpose(0, 1).float()
            hp, wp = item['inputs'].shape[-2]//16, item['inputs'].shape[-1]//16
            gt = sample.gt_sem_seg.data[None].float().cuda()
            gt = F.interpolate(gt, size=(hp*16, wp*16), mode='nearest').long().squeeze()
            valid = (gt != 255).float()[None, None]
            classes = F.one_hot(gt.clamp(0, 20), 21).permute(2, 0, 1).float()[None]
            occupancy = F.avg_pool2d(classes*valid, 16, 16).flatten(2)[0]
            denominator = F.avg_pool2d(valid, 16, 16).flatten().clamp_min(1e-9)
            purity, labels = occupancy.max(0)
            purity = purity/denominator
            labels[denominator < .5] = 255
            keep = labels != 255
            bg = labels[keep] == 0
            rows = []
            for strategy in ('qk', 'qq', 'kk', 'pure_pa'):
                a = attention(q, k, strategy)[:, 1:, 1:]
                a = a[:, keep][:, :, keep]
                a = a/a.sum(-1, keepdim=True).clamp_min(1e-9)
                received = a.sum(1)
                knorm = k[:, 1:][:, keep].norm(dim=-1)
                top_count = max(1, received.shape[-1]//100)
                top_indices = received.topk(top_count, dim=-1).indices
                rec_center = received-received.mean(-1, keepdim=True)
                norm_center = knorm-knorm.mean(-1, keepdim=True)
                corr = (rec_center*norm_center).sum(-1)/(rec_center.norm(dim=-1)*norm_center.norm(dim=-1)).clamp_min(1e-9)
                for head in range(h):
                    top = top_indices[head]
                    rows.append(dict(variant=strategy, head=head,
                                     key_norm_received_pearson=float(corr[head]),
                                     top1pct_receiver_mass=float(received[head, top].sum()/received[head].sum()),
                                     top1pct_bg_fraction=float(bg[top].float().mean()),
                                     bg_area_fraction=float(bg.float().mean()),
                                     high_purity_fraction=float((purity[keep] >= .9).float().mean())))
            stream.write(json.dumps(dict(index=index, rows=rows), separators=(',', ':'))+'\n')
            if (index+1) % 50 == 0:
                print(index+1, '/', len(ds), flush=True)
    hook.remove()
    aggregates = {}
    for line in OUT.open():
        for row in json.loads(line)['rows']:
            aggregates.setdefault(row['variant'], []).append(row)
    summary = {variant: {key: float(np.mean([r[key] for r in records]))
                         for key in ('key_norm_received_pearson', 'top1pct_receiver_mass',
                                     'top1pct_bg_fraction', 'bg_area_fraction', 'high_purity_fraction')}
               for variant, records in aggregates.items()}
    Path('results/pa_mechanism_phase2_key_receivers.json').write_text(
        json.dumps(dict(dataset='voc21', images=len(ds), heads=h, variants=summary), indent=2))
    print('complete', len(done), len(ds), flush=True)


if __name__ == '__main__':
    main()
