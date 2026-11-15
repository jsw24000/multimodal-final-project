"""Aggregate complete per-head records and render mechanism plots."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,default=Path('results/pa_mechanism_phase2_diagnostics.csv'))
    p.add_argument('--figdir',type=Path,default=Path('results/figures/pa_mechanism_phase2'));a=p.parse_args()
    data=[json.loads(x) for x in a.run.read_text().splitlines()];heads=[h for x in data for h in x['heads']]
    variants=list(heads[0]['variants']);keys=list(heads[0]['variants'][variants[0]])
    rows=[]
    for v in variants:
        row=dict(variant=v,images=len(data),image_head_observations=len(heads))
        for k in keys:
            values=np.asarray([h['variants'][v][k] for h in heads],dtype=float)
            row[k]=float(np.nanmean(values))
        rows.append(row)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    a.figdir.mkdir(parents=True,exist_ok=True)
    palette=['#A7C7E7','#F4A7AE','#A6D0C6','#5878A6','#FBD9C9','#DADDE3']
    for key,title in [('same','Same-class attention mass'),('off_same','Off-diagonal same-class mass'),('bg_enrichment','Foreground to background, area normalized'),('between_cos','Between-class query-row cosine')]:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        ax.bar(range(len(rows)),[r[key] for r in rows],color=palette[:len(rows)])
        ax.set_xticks(range(len(rows)),variants,rotation=25,ha='right');ax.set_ylabel(key);ax.set_title(f'VOC21, {len(data)} images, 12 heads | {title}')
        fig.savefig(a.figdir/f'{key}.png',dpi=180);fig.savefig(a.figdir/f'{key}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for solver,color in [('polar','#F4A7AE'),('svd','#5878A6')]:
        vals=[h['solver'][solver]['orth_error'] for h in heads]
        ax.hist(vals,bins=30,alpha=.65,label=solver,color=color)
    ax.set(xlabel='Frobenius orthogonality error',ylabel='Image/head count',title=f'VOC21, {len(data)} images');ax.legend()
    fig.savefig(a.figdir/'orthogonality.png',dpi=180);fig.savefig(a.figdir/'orthogonality.svg');plt.close(fig)
    iters=(1,3,5,10,20)
    ns_heads=[h for h in heads if 'ns_iterations' in h]
    if ns_heads:
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        ax.plot(iters,[np.mean([h['ns_iterations'][str(i)]['orth_error'] for h in ns_heads]) for i in iters],'o-',color='#5878A6')
        ax.set(xlabel='Newton-Schulz iterations',ylabel='Mean orthogonality error',title=f'VOC21, {len(ns_heads)//12} images, 12 heads');ax.set_xscale('log')
        fig.savefig(a.figdir/'ns_convergence.png',dpi=180);fig.savefig(a.figdir/'ns_convergence.svg');plt.close(fig)
    for row in rows:print(row,flush=True)
    print('solver means')
    for solver in ('polar','svd'):
        print(solver,{k:float(np.mean([h['solver'][solver][k] for h in heads])) for k in heads[0]['solver'][solver]})

if __name__=='__main__':main()
