"""Streaming per-image/head VOC21 attention and solver diagnostics."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS,MODELS
from mmseg.utils import register_all_modules
import custom_datasets,pearl_ovss
from experiments.pa_mechanism.attention_variants import attention,alignment

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))

def metrics(a,labels,purity):
    a=a[1:,1:].float()
    valid=labels!=255
    a=a[:,valid];a=a/a.sum(-1,keepdim=True).clamp_min(1e-9)
    rows=a[valid]
    y=labels[valid];p=purity[valid]
    if not len(y):return {}
    n=len(y);same=y[:,None]==y[None,:];diag=torch.eye(n,device=y.device,dtype=torch.bool)
    fg=y!=0;bg=y==0
    same_mass=(rows*same).sum(-1).mean();diag_mass=rows[diag].mean()
    off_same=(rows*(same&~diag)).sum(-1).mean()
    area=same.float().mean(-1).clamp_min(1e-9)
    enrichment=((rows*same).sum(-1)/area).mean()
    leak=rows[fg][:,bg].sum(-1).mean() if fg.any() and bg.any() else torch.tensor(float('nan'),device=y.device)
    bg_frac=bg.float().mean();rho=leak/bg_frac if bg_frac>0 else torch.tensor(float('nan'),device=y.device)
    recv=rows.sum(0);k=max(1,n//100);top=recv.topk(k).values.sum()/recv.sum()
    entropy=-(rows*rows.clamp_min(1e-9).log()).sum(-1).mean()
    # Deterministic, distance-controlled pair sample: same displacement of 3 patches where available.
    norm=F.normalize(rows,dim=-1);cos=norm@norm.T
    eye=~diag
    within=cos[same&eye].mean() if (same&eye).any() else torch.tensor(float('nan'),device=y.device)
    between=cos[~same].mean() if (~same).any() else torch.tensor(float('nan'),device=y.device)
    high=p>=.9; mixed=p<.9
    return dict(same=float(same_mass),off_same=float(off_same),diagonal=float(diag_mass),enrichment=float(enrichment),
                bg_leak=float(leak),bg_enrichment=float(rho),bg_area=float(bg_frac),top1pct_received=float(top),
                entropy=float(entropy),within_cos=float(within),between_cos=float(between),
                high_purity_frac=float(high.float().mean()),mixed_frac=float(mixed.float().mean()))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=0);p.add_argument('--seed',type=int,default=24000);a=p.parse_args()
    register_all_modules();cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.use_prop='off';cfg.model.attn_strategy='qq'
    ds=DATASETS.build(cfg.test_dataloader.dataset);ids=list(range(len(ds))) if a.count==0 else sorted(map(int,np.random.default_rng(a.seed).choice(len(ds),a.count,replace=False)))
    model=MODELS.build(cfg.model).cuda().eval();capture=[]
    hook=model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(lambda m,i,o:capture.append(o.detach()))
    out=ROOT/'outputs/pa_mechanism/phase2/metrics';out.mkdir(parents=True,exist_ok=True)
    path=out/f'voc21_attention_{len(ids)}_{a.seed}.jsonl'
    done=set()
    if path.exists():
        for line in path.read_text().splitlines():done.add(json.loads(line)['index'])
    with path.open('a',buffering=1) as f:
        for index in ids:
            if index in done:continue
            item=ds[index];capture.clear();result=model.test_step(pseudo_collate([item]))[0]
            z=capture[-1];layer=model.net.visual.transformer.resblocks[-1].attn
            q,k,_=F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,dim=-1)
            t,b,d=q.shape;h=layer.num_heads;q=q.view(t,b*h,d//h).transpose(0,1).float();k=k.view(t,b*h,d//h).transpose(0,1).float()
            hp,wp=item['inputs'].shape[-2]//16,item['inputs'].shape[-1]//16
            gt=result.gt_sem_seg.data[None].float().cuda()
            gt=F.interpolate(gt,size=(hp*16,wp*16),mode='nearest').long().squeeze()
            one=F.one_hot(gt.clamp(0,20),21).permute(2,0,1).float()[None]
            valid=(gt!=255).float()[None,None]
            area=F.avg_pool2d(one*valid,16,16).flatten(2)[0]
            denom=F.avg_pool2d(valid,16,16).flatten().clamp_min(1e-9)
            pure,label=area.max(0);pure=pure/denom;label[denom<.5]=255
            maps={v:attention(q,k,v,solver=s) for v,s in [('qk','polar'),('qq','polar'),('kk','polar'),('pure_pa','polar'),('pa_kk','polar')]}
            maps['svd_pa']=attention(q,k,'pure_pa',solver='svd')
            rows=[]
            for head in range(h):
                row=dict(head=head,variants={v:metrics(att[head],label,pure) for v,att in maps.items()})
                row['solver']={}
                row['ns_iterations']={}
                for steps in (1,3,5,10,20):
                    _,_,rr,_=alignment(q[head:head+1],k[head:head+1],'polar',steps)
                    sv_it=torch.linalg.svdvals(rr[0])
                    row['ns_iterations'][str(steps)]=dict(orth_error=float((rr[0].T@rr[0]-torch.eye(rr.shape[-1],device=rr.device)).norm()),
                        singular_min=float(sv_it.min()),singular_mean=float(sv_it.mean()),singular_max=float(sv_it.max()))
                for solver in ('polar','svd'):
                    qc,kc,r,e=alignment(q[head:head+1],k[head:head+1],solver)
                    sv=torch.linalg.svdvals(r[0]);gram0=k[head]@k[head].T;gram1=(k[head]@r[0])@(k[head]@r[0]).T
                    row['solver'][solver]=dict(orth_error=float((r[0].T@r[0]-torch.eye(r.shape[-1],device=r.device)).norm()),
                        align_error=float(e[:,1:].norm()/qc[:,1:].norm()),singular_min=float(sv.min()),singular_max=float(sv.max()),
                        singular_mean=float(sv.mean()),gram_distortion=float((gram1-gram0).norm()/gram0.norm()))
                rows.append(row)
            f.write(json.dumps(dict(index=index,image_id=Path(ds.get_data_info(index)['img_path']).stem,heads=rows),separators=(',',':'))+'\n')
            if (index+1)%25==0:print(index+1,'/',len(ids),flush=True)
    hook.remove()

if __name__=='__main__':main()
