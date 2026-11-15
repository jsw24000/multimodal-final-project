"""Per-image NS-5 versus SVD diagnostic scatter and difference histogram."""
import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.pa_mechanism.phase2.summarize import load,miou

ROOT=Path('/data/Multimodal-final-project/outputs/pa_mechanism')
OUT=Path('results/figures/pa_mechanism_phase2');OUT.mkdir(parents=True,exist_ok=True)

def main():
    fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for row,dataset in enumerate(('voc21','voc20')):
        old=json.loads((ROOT/'runs'/('eval_1449_24000_11.json' if dataset=='voc21' else 'eval_voc20_1449_24000_4.json')).read_text())
        svd_path=ROOT/'phase2/runs'/f'{dataset}_off_1449_24000_svd_pa-svd_pakk-svd_g025-svd_g050-svd_g075-svd_g125.jsonl'
        cfg,svd=load(svd_path)
        ns={r['index']:np.asarray(r['confusion']) for r in old['records'] if r['variant']=='pure_pa'}
        ids=cfg['indices'];x=np.asarray([miou(ns[i]) for i in ids]);y=np.asarray([miou(svd['svd_pa'][i]) for i in ids]);delta=y-x
        axes[row,0].scatter(x,y,s=9,alpha=.35,color='#5878A6');axes[row,0].plot([0,100],[0,100],ls='--',color='#F4A7AE')
        axes[row,0].set(xlim=(0,100),ylim=(0,100),xlabel='NS-5 image diagnostic IoU (%)',ylabel='SVD image diagnostic IoU (%)',title=f'{dataset.upper()}, 1,449 images')
        axes[row,1].hist(delta,bins=40,color='#A7C7E7',edgecolor='#5878A6');axes[row,1].axvline(0,color='#F4A7AE',ls='--')
        axes[row,1].set(xlabel='SVD − NS-5 image diagnostic IoU (points)',ylabel='Images',title=f'median {np.median(delta):+.2f}, improved {(delta>0).sum()}, damaged {(delta<0).sum()}')
    fig.savefig(OUT/'solver_image_pairs.png',dpi=180);fig.savefig(OUT/'solver_image_pairs.svg');plt.close(fig)

if __name__=='__main__':main()
