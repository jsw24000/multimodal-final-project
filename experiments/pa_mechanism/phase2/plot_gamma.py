"""Plot full-validation gamma curves from verified CSV files."""
import csv
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/figures/pa_mechanism_phase2');OUT.mkdir(parents=True,exist_ok=True)

def rows(path):
    with open(path) as f:return {r['variant']:float(r['mIoU']) for r in csv.DictReader(f)}

def main():
    gammas=[0,.25,.5,.75,1,1.25]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,dataset in zip(axes,['voc21','voc20','ade20k']):
        if dataset=='ade20k':
            full=rows('results/pa_mechanism_phase2_ade20k_off.csv')
            ns=[full['qq'],full['ns5_g025'],full['ns5_g050'],full['ns5_g075'],full['ns5_pa'],full['ns5_g125']]
            sy=[full['qq'],full['svd_g025'],full['svd_g050'],full['svd_g075'],full['svd_pa'],full['svd_g125']]
            count='2,000 images'
        else:
            phase1=rows('results/pa_mechanism_summary.csv' if dataset=='voc21' else 'results/pa_mechanism_voc20_summary.csv')
            svd=rows(f'results/pa_mechanism_phase2_{dataset}_svd.csv')
            if dataset=='voc21':
                ns=[phase1['qq']]+[phase1[f'gamma_{g}'] for g in ('0.25','0.5','0.75')]+[phase1['pure_pa'],phase1['gamma_1.25']]
            else:
                ns_extra=rows('results/pa_mechanism_phase2_voc20_ns_gamma.csv')
                ns=[phase1['qq'],ns_extra['ns5_g025'],ns_extra['ns5_g050'],ns_extra['ns5_g075'],phase1['pure_pa'],ns_extra['ns5_g125']]
            sy=[phase1['qq'],svd['svd_g025'],svd['svd_g050'],svd['svd_g075'],svd['svd_pa'],svd['svd_g125']]
            count='1,449 images'
        ax.plot(gammas,ns,'o-',color='#F4A7AE',label='NS-5')
        ax.plot(gammas,sy,'s-',color='#5878A6',label='SVD')
        ax.set(xlabel='Residual strength γ',ylabel='mIoU (%)',title=f'{dataset.upper()}, {count}')
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(OUT/'gamma_three_datasets_full.png',dpi=190)
    fig.savefig(OUT/'gamma_three_datasets_full.svg')
    plt.close(fig)

if __name__=='__main__':main()
