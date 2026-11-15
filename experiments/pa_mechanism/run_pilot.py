"""Paired VOC21 pilot; saves per-image confusion and selected raw predictions."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules
import custom_datasets, pearl_ovss

ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
OUT = ROOT/'outputs/pa_mechanism'
VARIANTS = ('qk','qq','kk','sclip','sclip_mean','pure_pa','pa_kk','gamma_0.25','gamma_0.5','gamma_0.75','gamma_1.25')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--seed',type=int,default=24000)
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--dataset',choices=['voc21','voc20','ade20k'],default='voc21')
    a=p.parse_args()
    variants=tuple(a.variants.split(','))
    register_all_modules()
    cfg=Config.fromfile(f'configs/cfg_{a.dataset}.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt')
    cfg.model.use_prop='off'; cfg.model.attn_strategy='qk'; cfg.model.arch='reduced'
    dataset=DATASETS.build(cfg.test_dataloader.dataset)
    ids=np.random.default_rng(a.seed).choice(len(dataset),size=min(a.count,len(dataset)),replace=False)
    ids=sorted(map(int,ids))
    model=MODELS.build(cfg.model).cuda().eval()
    nclasses={'voc21':21,'voc20':20,'ade20k':150}[a.dataset]
    filename=f'eval_{a.count}_{a.seed}_{len(variants)}.json' if a.dataset=='voc21' else f'eval_{a.dataset}_{a.count}_{a.seed}_{len(variants)}.json'
    (OUT/'runs').mkdir(parents=True,exist_ok=True)
    (OUT/'predictions').mkdir(parents=True,exist_ok=True)
    records=[]
    for index in ids:
        item=dataset[index]
        image_id=Path(dataset.get_data_info(index)['img_path']).stem
        for variant in variants:
            if variant.startswith('gamma_'):
                model.net.visual.attn_strategy='gamma'
                model.net.visual.gamma=float(variant.split('_')[1])
            else: model.net.visual.attn_strategy=variant
            start=time.perf_counter()
            with torch.inference_mode(): result=model.test_step(pseudo_collate([item]))[0]
            torch.cuda.synchronize()
            pred=result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
            gt=result.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
            valid=gt!=255
            confusion=np.bincount((gt[valid].astype(int)*nclasses+pred[valid].astype(int)),minlength=nclasses*nclasses).reshape(nclasses,nclasses)
            records.append(dict(index=index,image_id=image_id,variant=variant,confusion=confusion.tolist(),seconds=time.perf_counter()-start))
            if len(records)%11==0: print(index,image_id,'complete',flush=True)
            if index in ids[:5]:
                folder=OUT/'predictions'/a.dataset/variant; folder.mkdir(parents=True,exist_ok=True)
                Image.fromarray(pred).save(folder/f'{image_id}.png')
        if len(records) % (len(variants)*25) == 0 or index == ids[-1]:
            (OUT/'runs'/filename).write_text(json.dumps(dict(dataset=a.dataset,seed=a.seed,indices=ids,variants=variants,records=records)))
    for variant in variants:
        cm=sum((np.asarray(r['confusion']) for r in records if r['variant']==variant))
        union=cm.sum(0)+cm.sum(1)-np.diag(cm)
        iou=np.divide(np.diag(cm),union,out=np.full(nclasses,np.nan),where=union!=0)
        print(variant,round(float(np.nanmean(iou)*100),2),flush=True)

if __name__=='__main__': main()
