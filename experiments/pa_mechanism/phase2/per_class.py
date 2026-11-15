"""Per-class IoU, precision, recall from a verified complete run."""
import argparse,csv
from pathlib import Path
import numpy as np
from experiments.pa_mechanism.phase2.summarize import load

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg,mat=load(a.run);rows=[]
    for v in cfg['variants']:
        cm=sum(mat[v].values());tp=np.diag(cm);u=cm.sum(0)+cm.sum(1)-tp
        for c in range(cfg['nclasses']):
            rows.append(dict(dataset=cfg['dataset'],tlp=cfg['tlp'],variant=v,class_id=c,gt_pixels=int(cm[c].sum()),pred_pixels=int(cm[:,c].sum()),
                iou=float(tp[c]/u[c]*100) if u[c] else float('nan'),
                precision=float(tp[c]/cm[:,c].sum()*100) if cm[:,c].sum() else float('nan'),
                recall=float(tp[c]/cm[c].sum()*100) if cm[c].sum() else float('nan')))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(len(rows),'class rows')

if __name__=='__main__':main()
