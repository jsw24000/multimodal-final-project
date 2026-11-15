import csv,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def csvmap(path):
    with open(path) as f:return {r['variant']:float(r['mIoU']) for r in csv.DictReader(f)}

def main():
    h=[v for x in map(json.loads,open('/data/Multimodal-final-project/outputs/pa_mechanism/phase2/metrics/voc21_attention_1449_24000.jsonl')) for v in x['heads']]
    it=(1,3,5,10,20)
    orth=[np.mean([v['ns_iterations'][str(i)]['orth_error'] for v in h]) for i in it]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot(it,orth,'o-',color='#5878A6');axes[0].set(xlabel='NS iterations',ylabel='Mean ||RᵀR−I||F',title='VOC21, 1,449 images × 12 heads');axes[0].grid(alpha=.25)
    for dataset,color in [('voc21','#F4A7AE'),('voc20','#A6D0C6'),('ade20k','#5878A6')]:
        full=csvmap(f'results/pa_mechanism_phase2_ns_iters_{dataset}.csv')
        old=csvmap('results/pa_mechanism_summary.csv' if dataset=='voc21' else 'results/pa_mechanism_voc20_summary.csv' if dataset=='voc20' else 'results/pa_mechanism_phase2_ade20k_off.csv')
        vals=[full['ns1_pa'],full['ns3_pa'],old['pure_pa' if dataset!='ade20k' else 'ns5_pa'],full['ns10_pa'],full['ns20_pa']]
        axes[1].plot(it,vals,'o-',label=dataset.upper(),color=color)
    axes[1].set(xlabel='NS iterations',ylabel='Full validation mIoU (%)',title='Pure PA, TLP off');axes[1].grid(alpha=.25);axes[1].legend()
    out=Path('results/figures/pa_mechanism_phase2');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'ns_iterations.png',dpi=190);fig.savefig(out/'ns_iterations.svg');plt.close(fig)

if __name__=='__main__':main()
