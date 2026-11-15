"""Build one long-form controlled performance table from verified complete CSVs."""
import csv
from pathlib import Path

OUT=Path('results/pa_mechanism_phase2_summary.csv')

def read(path):
    with open(path) as f:return list(csv.DictReader(f))

def main():
    rows=[]
    for dataset in ('voc21','voc20','ade20k'):
        if dataset=='voc21':
            old=read('results/pa_mechanism_summary.csv');extra=[]
        elif dataset=='voc20':
            old=read('results/pa_mechanism_voc20_summary.csv');extra=read('results/pa_mechanism_phase2_voc20_selfcorr.csv')
        else:
            old=read('results/pa_mechanism_phase2_ade20k_off.csv');extra=[]
        for source,tlp in ((old,'off'),(extra,'off'),(read(f'results/pa_mechanism_phase2_{dataset}_svd.csv') if dataset!='ade20k' else [],'off'),
                           (read(f'results/pa_mechanism_phase2_{dataset}_tlp.csv'),'on')):
            for r in source:
                variant=r['variant']
                if variant.startswith(('gamma_','svd_g','ns5_g')):continue
                rows.append(dict(dataset=dataset,tlp=tlp,variant=variant,n=int(r['n']),mIoU=float(r['mIoU']),mAcc=float(r['mAcc']),pAcc=float(r['pAcc'])))
    with OUT.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(OUT,len(rows),'rows')

if __name__=='__main__':main()
