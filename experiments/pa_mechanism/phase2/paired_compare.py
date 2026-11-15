"""Pair methods across separate Phase 1/2 run files and bootstrap dataset mIoU delta."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.pa_mechanism.phase2.summarize import load,miou

def array(path,variant):
    if path.suffix=='.jsonl':
        cfg,m=load(path);return cfg['indices'],np.stack([m[variant][i] for i in cfg['indices']])
    d=json.loads(path.read_text());r=[x for x in d['records'] if x['variant']==variant]
    return [x['index'] for x in r],np.stack([np.asarray(x['confusion'],dtype=np.int64) for x in r])

def main():
    p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('first_variant')
    p.add_argument('second',type=Path);p.add_argument('second_variant');p.add_argument('--reps',type=int,default=1000)
    a=p.parse_args();i,x=array(a.first,a.first_variant);j,y=array(a.second,a.second_variant)
    assert i==j and x.shape==y.shape,(len(i),len(j),x.shape,y.shape)
    rng=np.random.default_rng(24000);delta=miou(y.sum(0))-miou(x.sum(0));boot=[]
    for _ in range(a.reps):
        ix=rng.integers(len(i),size=len(i));boot.append(miou(y[ix].sum(0))-miou(x[ix].sum(0)))
    # Per-image IoU is explicitly a diagnostic over classes present in each image.
    image_delta=np.asarray([miou(y[t])-miou(x[t]) for t in range(len(i))])
    print(json.dumps(dict(first=a.first_variant,second=a.second_variant,n=len(i),delta_mIoU=delta,
        ci95=np.percentile(boot,[2.5,97.5]).tolist(),image_delta_median=float(np.nanmedian(image_delta)),
        image_improved=int((image_delta>0).sum()),image_damaged=int((image_delta<0).sum()))))

if __name__=='__main__':main()
