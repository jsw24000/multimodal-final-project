"""Aggregate sparse confusion records and paired image bootstrap."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def load(path):
    cfg=json.loads(path.with_suffix('.config.json').read_text())
    n=cfg['nclasses'];expected=set(cfg['indices']); variants=cfg['variants']
    matrices={v:{} for v in variants}
    for line in path.read_text().splitlines():
        r=json.loads(line);v=r['variant'];i=r['index']
        if i in matrices[v]:raise RuntimeError(f'duplicate {v} {i}')
        m=np.zeros(n*n,dtype=np.int64);m[r['confusion_indices']]=r['confusion_counts']
        matrices[v][i]=m.reshape(n,n)
    for v in variants:
        if set(matrices[v])!=expected:raise RuntimeError(f'incomplete {v}: {len(matrices[v])}/{len(expected)}')
    return cfg,matrices

def miou(m,include=None):
    if include is not None:m=m[include][:,include]
    union=m.sum(0)+m.sum(1)-np.diag(m)
    return float(np.nanmean(np.divide(np.diag(m),union,out=np.full(len(union),np.nan),where=union>0))*100)

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=24000);a=p.parse_args()
    cfg,mat=load(a.run);ids=cfg['indices'];variants=cfg['variants']
    arrays={v:np.stack([mat[v][i] for i in ids]) for v in variants}
    base=variants[0];rng=np.random.default_rng(a.seed)
    # Stream bootstrap samples to bound memory for ADE's 150-class matrices.
    delta={v:[] for v in variants if v!=base}
    for _ in range(a.bootstrap):
        ix=rng.integers(0,len(ids),len(ids))
        base_m=arrays[base][ix].sum(0);base_score=miou(base_m)
        for v in delta:delta[v].append(miou(arrays[v][ix].sum(0))-base_score)
    rows=[]
    for v in variants:
        cm=arrays[v].sum(0);u=cm.sum(0)+cm.sum(1)-np.diag(cm)
        acc=np.divide(np.diag(cm),cm.sum(1),out=np.full(len(cm),np.nan),where=cm.sum(1)>0)
        ci=np.percentile(delta[v],[2.5,97.5]).tolist() if v!=base and a.bootstrap else ([0,0] if v==base else [float('nan'),float('nan')])
        rows.append(dict(dataset=cfg['dataset'],tlp=cfg['tlp'],variant=v,n=len(ids),mIoU=miou(cm),mAcc=float(np.nanmean(acc)*100),pAcc=float(np.trace(cm)/cm.sum()*100),
                         delta_vs_first=miou(cm)-miou(arrays[base].sum(0)),ci_low=ci[0],ci_high=ci[1]))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for r in rows:print(r,flush=True)

if __name__=='__main__':main()
