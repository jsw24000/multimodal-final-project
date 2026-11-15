"""Pair Phase 1 NS-5 and Phase 2 SVD image confusion matrices."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from experiments.pa_mechanism.phase2.summarize import load,miou

ROOT=Path('/data/Multimodal-final-project/outputs/pa_mechanism')

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['voc21','voc20']);p.add_argument('svd_run',type=Path)
    p.add_argument('--out',type=Path,default=Path('results/pa_mechanism_phase2_solver_comparison.csv'));a=p.parse_args()
    old=ROOT/'runs'/('eval_1449_24000_11.json' if a.dataset=='voc21' else 'eval_voc20_1449_24000_4.json')
    data=json.loads(old.read_text());cfg,svd=load(a.svd_run)
    ids=cfg['indices'];n=cfg['nclasses'];assert len(ids)==1449
    ns={v:{} for v in ('pure_pa','pa_kk')}
    for r in data['records']:
        if r['variant'] in ns:ns[r['variant']][r['index']]=np.asarray(r['confusion'],dtype=np.int64)
    rng=np.random.default_rng(24000);rows=[]
    for name,svd_name in [('pure_pa','svd_pa'),('pa_kk','svd_pakk')]:
        x=np.stack([ns[name][i] for i in ids]);y=np.stack([svd[svd_name][i] for i in ids])
        dx=miou(y.sum(0))-miou(x.sum(0));samples=[]
        for _ in range(1000):
            ix=rng.integers(len(ids),size=len(ids));samples.append(miou(y[ix].sum(0))-miou(x[ix].sum(0)))
        ci=np.percentile(samples,[2.5,97.5]);perclass=[]
        for m in (x.sum(0),y.sum(0)):
            u=m.sum(0)+m.sum(1)-np.diag(m);perclass.append(np.divide(np.diag(m),u,out=np.full(n,np.nan),where=u>0)*100)
        rows.append(dict(dataset=a.dataset,variant=name,n=len(ids),ns_mIoU=miou(x.sum(0)),svd_mIoU=miou(y.sum(0)),svd_minus_ns=dx,
                         ci_low=ci[0],ci_high=ci[1],per_class_delta=json.dumps((perclass[1]-perclass[0]).tolist())))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    mode='a' if a.out.exists() else 'w'
    with a.out.open(mode,newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        if mode=='w':w.writeheader()
        w.writerows(rows)
    for row in rows:print({k:v for k,v in row.items() if k!='per_class_delta'})

if __name__=='__main__':main()
