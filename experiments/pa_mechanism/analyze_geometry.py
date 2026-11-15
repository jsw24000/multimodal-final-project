"""VOC21 per-head geometry and GT-conditioned attention diagnostics."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules
import custom_datasets, pearl_ovss
from experiments.pa_mechanism.attention_variants import attention, alignment

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))

def metric(a, labels):
    # CLS excluded; patch rows renormalized across patch keys.
    a=a[:,1:,1:].float()
    a=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
    good=labels!=255
    same=(labels[:,None]==labels[None,:]) & good[:,None] & good[None,:]
    fg=(labels!=0)&good
    bg=(labels==0)&good
    result=[]
    for head in a:
        same_mass=head[same].sum()/good.sum().clamp_min(1)
        leak=head[fg][:,bg].sum()/fg.sum().clamp_min(1)
        entropy=-(head*head.clamp_min(1e-9).log()).sum(-1).mean()
        result.append(dict(same_mass=float(same_mass),fg_bg_leak=float(leak),entropy=float(entropy)))
    return result

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--count',type=int,default=16); p.add_argument('--seed',type=int,default=24000); a=p.parse_args()
    register_all_modules(); cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt'); cfg.model.use_prop='off'; cfg.model.arch='reduced'; cfg.model.attn_strategy='qq'
    ds=DATASETS.build(cfg.test_dataloader.dataset); model=MODELS.build(cfg.model).cuda().eval()
    indices=sorted(map(int,np.random.default_rng(a.seed).choice(len(ds),size=a.count,replace=False)))
    captured=[]
    hook=model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(lambda mod,inp,out:captured.append(out.detach()))
    rows=[]
    for index in indices:
        item=ds[index]; captured.clear()
        with torch.inference_mode(): result=model.test_step(pseudo_collate([item]))[0]
        z=captured[-1]; layer=model.net.visual.transformer.resblocks[-1].attn
        q,k,v=F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,dim=-1)
        t,b,d=q.shape; h=layer.num_heads
        q=q.view(t,b*h,d//h).transpose(0,1).float(); k=k.view(t,b*h,d//h).transpose(0,1).float()
        hp,wp=item['inputs'].shape[-2]//16,item['inputs'].shape[-1]//16
        gt=result.gt_sem_seg.data[None].float().cuda()
        labels=F.interpolate(gt,size=(hp,wp),mode='nearest').long().flatten()
        qc,kc,r,e=alignment(q,k)
        qc_s,kc_s,r_s,e_s=alignment(q,k,solver='svd')
        patches=t-1
        rng=np.random.default_rng(a.seed+index)
        random_a=torch.as_tensor(rng.permutation(patches)[:patches//2]+1,device=q.device)
        yy=torch.arange(patches,device=q.device)//wp
        spatial_a=torch.where(yy < hp//2)[0]+1
        heldout={}
        for split,fit_ids in [('random',random_a),('spatial',spatial_a)]:
            mask=torch.ones(patches+1,dtype=torch.bool,device=q.device);mask[0]=False;mask[fit_ids]=False
            test_ids=torch.where(mask)[0]
            qfit,kfit=q[:,fit_ids],k[:,fit_ids]
            # Fit centers and R only on A; no B statistics enter this calculation.
            weight=qfit.norm(dim=-1,keepdim=True);weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-8)
            qm=(weight*qfit).sum(1,keepdim=True);km=(weight*kfit).sum(1,keepdim=True)
            m=(kfit-km).transpose(1,2)@(qfit-qm)
            u,_,vh=torch.linalg.svd(m,full_matrices=False);rr=u@vh
            train=((kfit-km)@rr-(qfit-qm)).norm(dim=(1,2))/(qfit-qm).norm(dim=(1,2))
            test=((k[:,test_ids]-km)@rr-(q[:,test_ids]-qm)).norm(dim=(1,2))/(q[:,test_ids]-qm).norm(dim=(1,2))
            heldout[split]=(train,test)
        gram_q=qc[:,1:]@qc[:,1:].transpose(1,2); gram_k=kc[:,1:]@kc[:,1:].transpose(1,2)
        variants={name:metric(attention(q,k,name),labels) for name in ('qk','qq','kk','pure_pa','pa_kk')}
        if index == indices[0]:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from PIL import Image
            figure_dir=ROOT/'outputs/pa_mechanism/figures'; figure_dir.mkdir(parents=True,exist_ok=True)
            image_id=Path(ds.get_data_info(index)['img_path']).stem
            rgb=Image.open(ROOT/'datasets/VOCdevkit/VOC2012/JPEGImages'/f'{image_id}.jpg').convert('RGB')
            gt_np=np.asarray(Image.open(ROOT/'datasets/VOCdevkit/VOC2012/SegmentationClass'/f'{image_id}.png'))
            for query_type,query_pool in [('foreground',torch.where((labels>0)&(labels<255))[0]),('background',torch.where(labels==0)[0])]:
                if not len(query_pool): continue
                query=int(query_pool[len(query_pool)//2]); names=('qk','qq','kk','pure_pa','pa_kk')
                maps=[attention(q,k,name)[0,query+1,1:].reshape(hp,wp).cpu().numpy() for name in names]
                vmax=max(x.max() for x in maps)
                fig,axes=plt.subplots(1,7,figsize=(19,3.2),layout='constrained')
                axes[0].imshow(rgb); axes[1].imshow(gt_np,cmap='tab20',vmin=0,vmax=21)
                for ax,name,map_ in zip(axes[2:],names,maps):
                    ax.imshow(map_,vmin=0,vmax=vmax,cmap='magma',interpolation='nearest');ax.set_title(name)
                for ax in axes: ax.axis('off')
                axes[0].set_title('Input');axes[1].set_title('GT')
                fig.suptitle(f'VOC21 {image_id} | head 0 | {query_type} query patch ({query//wp},{query%wp}) | common scale')
                fig.savefig(figure_dir/f'attention_{image_id}_{query_type}.png',dpi=170)
                fig.savefig(figure_dir/f'attention_{image_id}_{query_type}.svg');plt.close(fig)
        for gamma in (0,.25,.5,.75,1.,1.25): variants[f'gamma_{gamma}']=metric(attention(q,k,'gamma',gamma=gamma),labels)
        for head in range(h):
            rows.append(dict(index=index,image_id=Path(ds.get_data_info(index)['img_path']).stem,head=head,
                alignment_error=float((e[head,1:].norm()/qc[head,1:].norm()).item()),
                residual_ratio=float((e[head,1:].norm()/kc[head,1:].norm()).item()),
                orthogonality_error=float((r[head].T@r[head]-torch.eye(r.shape[-1],device=r.device)).norm().item()),
                svd_orthogonality_error=float((r_s[head].T@r_s[head]-torch.eye(r.shape[-1],device=r.device)).norm().item()),
                svd_alignment_error=float((e_s[head,1:].norm()/qc_s[head,1:].norm()).item()),
                heldout={split:dict(train=float(vals[0][head]),test=float(vals[1][head])) for split,vals in heldout.items()},
                gram_cosine=float(F.cosine_similarity(gram_q[head].flatten(),gram_k[head].flatten(),dim=0).item()),
                metrics={name:vals[head] for name,vals in variants.items()}))
        print(index,'geometry complete',flush=True)
    hook.remove()
    out=ROOT/'outputs/pa_mechanism/metrics'; out.mkdir(parents=True,exist_ok=True)
    (out/f'geometry_{a.count}_{a.seed}.json').write_text(json.dumps(dict(indices=indices,rows=rows)))

if __name__=='__main__': main()
