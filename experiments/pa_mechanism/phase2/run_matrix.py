"""Resumable paired evaluation. One JSONL line per completed image/variant."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules
import custom_datasets, pearl_ovss

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
OUT=ROOT/'outputs/pa_mechanism/phase2/runs'
N={'voc21':21,'voc20':20,'ade20k':150}
DEFAULT=('qk','qq','kk','sclip','sclip_mean','ns5_pa','ns5_pakk','svd_pa','svd_pakk',
         'svd_g025','svd_g050','svd_g075','svd_g125')

def configure(visual, name):
    visual.gamma=1.;visual.polar_iters=5;visual.align_solver='polar'
    if name in ('qk','qq','kk','sclip','sclip_mean'):
        visual.attn_strategy=name
    elif name.startswith(('ns','svd')):
        solver,kind=name.split('_',1)
        visual.align_solver='svd' if solver=='svd' else 'polar'
        if solver.startswith('ns'):visual.polar_iters=int(solver[2:])
        if kind=='pa':visual.attn_strategy='pure_pa'
        elif kind=='pakk':visual.attn_strategy='pa_kk'
        elif kind=='qqmatch':visual.attn_strategy='qq_entropy_match'
        elif kind.startswith('g'):
            visual.attn_strategy='gamma';visual.gamma=int(kind[1:])/100
        else:raise ValueError(name)
    else:raise ValueError(name)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=N,required=True)
    p.add_argument('--tlp',choices=['on','off'],default='off')
    p.add_argument('--variants',default=','.join(DEFAULT))
    p.add_argument('--count',type=int,default=0,help='0 means full split')
    p.add_argument('--seed',type=int,default=24000)
    args=p.parse_args()
    variants=args.variants.split(',');register_all_modules()
    cfg=Config.fromfile(f'configs/cfg_{args.dataset}.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.arch='reduced';cfg.model.use_prop=args.tlp;cfg.model.attn_strategy='qk'
    ds=DATASETS.build(cfg.test_dataloader.dataset)
    ids=list(range(len(ds))) if args.count==0 else sorted(map(int,np.random.default_rng(args.seed).choice(len(ds),min(args.count,len(ds)),replace=False)))
    OUT.mkdir(parents=True,exist_ok=True)
    run_id=f'{args.dataset}_{args.tlp}_{len(ids)}_{args.seed}_{"-".join(variants)}'
    path=OUT/f'{run_id}.jsonl'
    config=OUT/f'{run_id}.config.json'
    config_data=dict(dataset=args.dataset,tlp=args.tlp,variants=variants,indices=ids,seed=args.seed,nclasses=N[args.dataset],
                     checkpoint=str(ROOT/'checkpoints/clip/ViT-B-16.pt'),arch='reduced',git_commit=os.popen('git rev-parse HEAD').read().strip())
    if config.exists() and json.loads(config.read_text())!=config_data:raise RuntimeError('run configuration mismatch')
    config.write_text(json.dumps(config_data,indent=2))
    done=set()
    if path.exists():
        with path.open() as f:
            for line in f:
                r=json.loads(line);done.add((r['index'],r['variant']))
    model=MODELS.build(cfg.model).cuda().eval();n=N[args.dataset]
    model.net.visual.phase2_cache_enabled=True
    with path.open('a',buffering=1) as f:
        for j,index in enumerate(ids):
            model.net.visual._phase2_prefix_cache=[]
            item=None
            for variant in variants:
                if (index,variant) in done:continue
                if item is None:item=ds[index]
                configure(model.net.visual,variant)
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():result=model.test_step(pseudo_collate([item]))[0]
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                pred=result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
                gt=result.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
                if pred.shape!=gt.shape:raise RuntimeError((pred.shape,gt.shape))
                good=gt!=255
                if good.any() and (gt[good].max()>=n or pred[good].max()>=n):raise RuntimeError('invalid class id')
                bins=np.bincount(gt[good]*n+pred[good],minlength=n*n)
                nz=np.flatnonzero(bins)
                r=dict(index=index,image_id=Path(ds.get_data_info(index)['img_path']).stem,variant=variant,
                       confusion_indices=nz.tolist(),confusion_counts=bins[nz].tolist(),seconds=elapsed,
                       gt_pixels=int(good.sum()))
                f.write(json.dumps(r,separators=(',',':'))+'\n')
            if (j+1)%25==0 or j+1==len(ids):print(args.dataset,args.tlp,j+1,'/',len(ids),flush=True)

if __name__=='__main__':main()
