"""Real VOC21 segmentation and head-0 attention panels for fixed and extreme cases."""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS,MODELS
from mmseg.utils import register_all_modules
import custom_datasets,pearl_ovss
from experiments.pa_mechanism.attention_variants import attention
from experiments.pa_mechanism.phase2.run_matrix import configure

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
FIG=ROOT/'outputs/pa_mechanism/phase2/figures';FIG.mkdir(parents=True,exist_ok=True)
PALETTE=np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128],[255,255,255]],dtype=np.uint8)

@torch.inference_mode()
def main():
    register_all_modules();cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop='off';cfg.model.attn_strategy='qk'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval()
    captures=[];hook=model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(lambda m,i,o:captures.append(o.detach()))
    choices=[(0,'fixed-first'),(560,'PA-regression-vs-QQ'),(619,'PA-improvement-vs-QQ')]
    variants=['qk','qq','kk','sclip_mean','ns5_pa','svd_pa','ns5_pakk','svd_pakk']
    names=['QK','QQ','KK','QQ+KK mean','PA NS-5','PA SVD','PA+KK NS-5','PA+KK SVD']
    for index,reason in choices:
        item=ds[index];image_id=Path(ds.get_data_info(index)['img_path']).stem
        rgb=np.asarray(Image.open(ROOT/'datasets/VOCdevkit/VOC2012/JPEGImages'/f'{image_id}.jpg').convert('RGB'))
        gt=np.asarray(Image.open(ROOT/'datasets/VOCdevkit/VOC2012/SegmentationClass'/f'{image_id}.png'))
        preds=[];captures.clear()
        for variant in variants:
            configure(model.net.visual,variant)
            result=model.test_step(pseudo_collate([item]))[0]
            preds.append(result.pred_sem_seg.data.squeeze().cpu().numpy())
        np.savez_compressed(FIG/f'{image_id}_labels.npz',rgb=rgb,gt=gt,variants=np.asarray(variants),predictions=np.stack(preds))
        fig,axes=plt.subplots(2,5,figsize=(16,7),layout='constrained')
        for ax in axes.ravel():ax.axis('off')
        entries=[('RGB',rgb),('GT',PALETTE[np.minimum(gt,21)])]+[(name,PALETTE[pred]) for name,pred in zip(names,preds)]
        for ax,(title,img) in zip(axes.ravel(),entries):ax.imshow(img);ax.set_title(title)
        fig.suptitle(f'VOC21 {image_id} | {reason} | Reduced, TLP off, NS-5/SVD | KK=.08',fontsize=14)
        fig.savefig(FIG/f'{image_id}_segmentation.png',dpi=180);fig.savefig(FIG/f'{image_id}_segmentation.svg');plt.close(fig)
        # Last hook belongs to final variant; q/k do not depend on the attention choice.
        z=captures[-1];layer=model.net.visual.transformer.resblocks[-1].attn
        q,k,_=F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,dim=-1)
        t,b,d=q.shape;h=layer.num_heads;q=q.view(t,b*h,d//h).transpose(0,1).float();k=k.view(t,b*h,d//h).transpose(0,1).float()
        hp,wp=item['inputs'].shape[-2]//16,item['inputs'].shape[-1]//16
        lab=F.interpolate(torch.as_tensor(gt,device=q.device)[None,None].float(),size=(hp,wp),mode='nearest').long().flatten()
        patch_labels=lab.reshape(hp,wp)
        boundary=torch.zeros_like(patch_labels,dtype=torch.bool)
        boundary[1:] |= patch_labels[1:] != patch_labels[:-1]
        boundary[:-1] |= patch_labels[:-1] != patch_labels[1:]
        boundary[:,1:] |= patch_labels[:,1:] != patch_labels[:,:-1]
        boundary[:,:-1] |= patch_labels[:,:-1] != patch_labels[:,1:]
        pools=[('foreground',torch.where((lab>0)&(lab<255))[0]),
               ('background',torch.where(lab==0)[0]),
               ('boundary',torch.where(boundary.flatten()&(lab>0)&(lab<255))[0])]
        for query_type,pool in pools:
            if not len(pool):continue
            if query_type=='foreground':
                coords=np.argwhere((gt>0)&(gt<255))
                cy,cx=np.median(coords,axis=0)
                query=min(hp-1,int(cy*hp/gt.shape[0]))*wp+min(wp-1,int(cx*wp/gt.shape[1]))
            else:
                query=int(pool[len(pool)//2])
            show=['qk','qq','kk','ns5_pa','svd_pa']
            maps=[]
            for v in show:
                solver='svd' if v=='svd_pa' else 'polar';strategy='pure_pa' if v.endswith('_pa') else v
                maps.append(attention(q,k,strategy,solver=solver)[0,query+1,1:].reshape(hp,wp).cpu().numpy())
            np.savez_compressed(FIG/f'{image_id}_{query_type}_attention.npz',maps=np.stack(maps),variants=np.asarray(show),
                                query_row=query//wp,query_col=query%wp,head=0,image_id=image_id)
            vmax=max(m.max() for m in maps)
            fig,axes=plt.subplots(1,7,figsize=(19,3.5),layout='constrained')
            axes[0].imshow(rgb);axes[1].imshow(PALETTE[np.minimum(gt,21)])
            px=(query%wp+.5)*rgb.shape[1]/wp;py=(query//wp+.5)*rgb.shape[0]/hp
            axes[0].plot(px,py,'o',ms=10,mec='white',mfc='#F4A7AE',mew=2)
            for ax,title,map_ in zip(axes[2:],['QK','QQ','KK','PA NS-5','PA SVD'],maps):
                ax.imshow(map_,norm=LogNorm(vmin=1e-5,vmax=vmax),cmap='magma',interpolation='nearest');ax.set_title(title)
            for ax in axes:ax.axis('off')
            axes[0].set_title('RGB + query');axes[1].set_title('GT')
            fig.suptitle(f'VOC21 {image_id} | {reason} | {query_type} query ({query//wp},{query%wp}) | head 0 | common log scale')
            fig.savefig(FIG/f'{image_id}_{query_type}_attention.png',dpi=180);fig.savefig(FIG/f'{image_id}_{query_type}_attention.svg');plt.close(fig)
        print(image_id,'figures complete',flush=True)
    hook.remove()

if __name__=='__main__':main()
