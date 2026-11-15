"""Summarize paired confusion matrices; only accept a complete split when requested."""
import argparse, csv, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--expected',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.run.read_text()); variants=data['variants']; records=data['records']
    counts={v:sum(r['variant']==v for r in records) for v in variants}
    if a.expected and any(n!=a.expected for n in counts.values()): raise ValueError(f'incomplete: {counts}')
    rows=[]
    for variant in variants:
        rr=[r for r in records if r['variant']==variant]
        cm=sum((np.asarray(r['confusion']) for r in rr))
        union=cm.sum(0)+cm.sum(1)-np.diag(cm)
        iou=np.divide(np.diag(cm),union,out=np.full(cm.shape[0],np.nan),where=union>0)
        accuracy=np.divide(np.diag(cm),cm.sum(1),out=np.full(cm.shape[0],np.nan),where=cm.sum(1)>0)
        rows.append(dict(variant=variant,n=len(rr),mIoU=float(np.nanmean(iou)*100),mAcc=float(np.nanmean(accuracy)*100),pAcc=float(np.trace(cm)/cm.sum()*100),mean_seconds=float(np.mean([r['seconds'] for r in rr]))))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for row in rows:print(row)

if __name__=='__main__':main()
