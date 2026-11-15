"""Fixed-seed VOC21 Q/K cross-covariance and R singular-value spectra."""
import json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS,MODELS
from mmseg.utils import register_all_modules
import custom_datasets,pearl_ovss
from experiments.pa_mechanism.attention_variants import alignment

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
OUT=ROOT/'outputs/pa_mechanism/phase2/metrics';OUT.mkdir(parents=True,exist_ok=True)
FIG=Path('results/figures/pa_mechanism_phase2');FIG.mkdir(parents=True,exist_ok=True)

@torch.inference_mode()
def main():
    register_all_modules();cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop='off';cfg.model.attn_strategy='qq'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval()
    indices=sorted(map(int,np.random.default_rng(24000).choice(len(ds),size=128,replace=False)))
    capture=[];hook=model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(lambda m,i,o:capture.append(o.detach()))
    spectra={'M':[],'NS5':[],'SVD':[]}
    for index in indices:
        capture.clear();model.test_step(pseudo_collate([ds[index]]))
        z=capture[-1];layer=model.net.visual.transformer.resblocks[-1].attn
        q,k,_=F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,dim=-1)
        t,b,d=q.shape;h=layer.num_heads;q=q.view(t,b*h,d//h).transpose(0,1).float();k=k.view(t,b*h,d//h).transpose(0,1).float()
        qc,kc,r_ns,_=alignment(q,k,'polar');_,_,r_svd,_=alignment(q,k,'svd')
        m=kc.transpose(1,2)@qc
        sv_m=torch.linalg.svdvals(m);sv_ns=torch.linalg.svdvals(r_ns);sv_svd=torch.linalg.svdvals(r_svd)
        for head in range(h):
            spectra['M'].append(sv_m[head].cpu().numpy());spectra['NS5'].append(sv_ns[head].cpu().numpy());spectra['SVD'].append(sv_svd[head].cpu().numpy())
        if (indices.index(index)+1)%25==0:print(indices.index(index)+1,'/128',flush=True)
    hook.remove();arrays={k:np.stack(v) for k,v in spectra.items()}
    np.savez_compressed(OUT/'voc21_solver_spectrum_128_24000.npz',indices=np.asarray(indices),**arrays)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key in zip(axes,['M','NS5']):
        vals=arrays[key]
        if key=='M':vals=vals/np.maximum(vals[:,:1],1e-9)
        mean=vals.mean(0);lo,hi=np.percentile(vals,[10,90],axis=0)
        ax.plot(range(1,len(mean)+1),mean,color='#5878A6',label=key)
        ax.fill_between(range(1,len(mean)+1),lo,hi,color='#A7C7E7',alpha=.55,label='10–90 percentile')
        if key=='NS5':ax.axhline(1,color='#F4A7AE',ls='--',label='orthogonal SVD')
        ax.set(xlabel='Singular value index',ylabel='Singular value'+(' / top singular value' if key=='M' else ''),title=f'{key} | 128 images × 12 heads');ax.legend()
    fig.savefig(FIG/'solver_spectrum.png',dpi=190);fig.savefig(FIG/'solver_spectrum.svg');plt.close(fig)
    print('spectrum complete')

if __name__=='__main__':main()
