"""Patch feature/text margin and shared-basis PCA on fixed VOC21 images."""
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
from experiments.pa_mechanism.phase2.run_matrix import configure

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
OUT=ROOT/'outputs/pa_mechanism/phase2/features';OUT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'outputs/pa_mechanism/phase2/figures';FIG.mkdir(parents=True,exist_ok=True)

@torch.inference_mode()
def main():
    register_all_modules();cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop='off';cfg.model.attn_strategy='qk'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval()
    variants=['qk','qq','sclip','sclip_mean','ns5_pa','svd_pa'];results=[]
    for index in (0,560,619):
        item=ds[index];image_id=Path(ds.get_data_info(index)['img_path']).stem
        batch=model.data_preprocessor(pseudo_collate([item]),training=False);img=batch['inputs']
        hp,wp=img.shape[-2]//16,img.shape[-1]//16
        gt=batch['data_samples'][0].gt_sem_seg.data[None].float().cuda()
        label=F.interpolate(gt,size=(hp,wp),mode='nearest').long().flatten()
        text=F.normalize(model.query_features.float(),dim=-1)
        all_features=[];rows=[]
        for variant in variants:
            configure(model.net.visual,variant)
            tokens=model.net.encode_image(img,return_all=True)[:,1:].float().squeeze(0)
            features=F.normalize(tokens,dim=-1)
            query_scores=features@text.T
            cls_scores=torch.stack([query_scores[:,model.query_idx==c].max(-1).values for c in range(21)],dim=-1)
            valid=label!=255
            truth=cls_scores[valid].gather(1,label[valid,None]).squeeze(1)
            other=cls_scores[valid].clone();other.scatter_(1,label[valid,None],-1e9)
            margin=truth-other.max(-1).values
            rows.append(dict(image_id=image_id,index=index,variant=variant,valid_patches=int(valid.sum()),
                patch_accuracy=float((cls_scores[valid].argmax(-1)==label[valid]).float().mean()),mean_margin=float(margin.mean()),
                projected_token_norm=float(tokens.norm(dim=-1).mean()),
                mean_foreground_margin=float(margin[label[valid]!=0].mean()) if (label[valid]!=0).any() else float('nan')))
            all_features.append(features.cpu().numpy())
        x=np.concatenate(all_features,axis=0);center=x.mean(0,keepdims=True);_,_,vh=np.linalg.svd(x-center,full_matrices=False)
        xy=[(features-center)@vh[:2].T for features in all_features]
        fig,axes=plt.subplots(1,len(variants),figsize=(17,3.5),layout='constrained')
        labels=label.cpu().numpy();good=labels!=255
        all_xy=np.concatenate(xy);xlim=(all_xy[:,0].min(),all_xy[:,0].max());ylim=(all_xy[:,1].min(),all_xy[:,1].max())
        for ax,v,points in zip(axes,variants,xy):
            ax.scatter(points[good,0],points[good,1],c=labels[good],s=7,cmap='tab20',vmin=0,vmax=20,alpha=.7)
            ax.set_title(v);ax.set_xlabel('Shared PC1');ax.set_ylabel('Shared PC2');ax.set_xlim(xlim);ax.set_ylim(ylim)
        fig.suptitle(f'VOC21 {image_id} | shared PCA basis | GT patch labels')
        fig.savefig(FIG/f'{image_id}_feature_pca.png',dpi=180);fig.savefig(FIG/f'{image_id}_feature_pca.svg');plt.close(fig)
        results.extend(rows);print(image_id,'features complete',flush=True)
    (OUT/'voc21_fixed_cases_margin.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
