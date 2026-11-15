"""Paired bootstrap for Phase 1 full VOC records, without modifying them."""
import argparse,csv,json
from pathlib import Path
import numpy as np

ROOT=Path('/data/Multimodal-final-project/outputs/pa_mechanism/runs')
FILES={'voc21':ROOT/'eval_1449_24000_11.json','voc20':ROOT/'eval_voc20_1449_24000_4.json'}

def score(cm):
    u=cm.sum(0)+cm.sum(1)-np.diag(cm)
    return np.nanmean(np.divide(np.diag(cm),u,out=np.full(len(u),np.nan),where=u>0))*100

def main():
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=1000);p.add_argument('--out',type=Path,default=Path('results/pa_mechanism_phase2_phase1_bootstrap.csv'));a=p.parse_args()
    rng=np.random.default_rng(24000);rows=[]
    for dataset,path in FILES.items():
        data=json.loads(path.read_text());n=len(data['indices']);variants=data['variants']
        arrays={v:np.stack([np.asarray(r['confusion'],dtype=np.int64) for r in data['records'] if r['variant']==v]) for v in variants}
        assert all(x.shape[0]==n for x in arrays.values())
        for comp in [('qq','pure_pa'),('qq','pa_kk')]+([('sclip_mean','pure_pa'),('sclip_mean','pa_kk')] if dataset=='voc21' else []):
            base,candidate=comp;delta=score(arrays[candidate].sum(0))-score(arrays[base].sum(0));samples=[]
            for _ in range(a.reps):
                ix=rng.integers(n,size=n)
                samples.append(score(arrays[candidate][ix].sum(0))-score(arrays[base][ix].sum(0)))
            low,high=np.percentile(samples,[2.5,97.5])
            rows.append(dict(dataset=dataset,base=base,candidate=candidate,n=n,delta_mIoU=delta,ci_low=low,ci_high=high,reps=a.reps))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for row in rows:print(row)

if __name__=='__main__':main()
