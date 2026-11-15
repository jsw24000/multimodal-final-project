"""Image-level paired bootstrap of attention scalar diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,default=Path('results/pa_mechanism_phase2_diagnostic_bootstrap.csv'))
    a=p.parse_args();data=[json.loads(x) for x in a.run.read_text().splitlines()]
    rng=np.random.default_rng(24000);rows=[]
    for candidate in ('qq','kk','pure_pa','pa_kk','svd_pa'):
        for key in ('same','off_same','bg_enrichment','between_cos','entropy','top1pct_received'):
            x=np.asarray([[h['variants']['qk'][key] for h in r['heads']] for r in data],dtype=float)
            y=np.asarray([[h['variants'][candidate][key] for h in r['heads']] for r in data],dtype=float)
            diff=np.nanmean(y-x,axis=1)
            samples=[]
            for _ in range(1000):samples.append(np.nanmean(diff[rng.integers(len(diff),size=len(diff))]))
            ci=np.percentile(samples,[2.5,97.5])
            rows.append(dict(dataset='voc21',base='qk',candidate=candidate,metric=key,images=len(data),delta=float(np.nanmean(diff)),ci_low=ci[0],ci_high=ci[1]))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for r in rows:print(r)

if __name__=='__main__':main()
