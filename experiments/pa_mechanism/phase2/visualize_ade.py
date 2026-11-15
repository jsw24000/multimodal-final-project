"""Fixed ADE20K real-image segmentation panels under released sliding-window inference."""
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS,MODELS
from mmseg.utils import register_all_modules
import custom_datasets,pearl_ovss
from experiments.pa_mechanism.phase2.run_matrix import configure

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
OUT=ROOT/'outputs/pa_mechanism/phase2/figures';OUT.mkdir(parents=True,exist_ok=True)

@torch.inference_mode()
def main():
    register_all_modules();cfg=Config.fromfile('configs/cfg_ade20k.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop='off';cfg.model.attn_strategy='qk'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval()
    palette=np.asarray(ds.metainfo['palette'],dtype=np.uint8);palette=np.vstack([palette,np.array([[255,255,255]],dtype=np.uint8)])
    variants=['qk','qq','kk','sclip_mean','ns5_pa','svd_pa','ns5_pakk','svd_pakk']
    names=['QK','QQ','KK','QQ+KK mean','PA NS-5','PA SVD','PA+KK NS-5','PA+KK SVD']
    for index in (0,1000,1999):
        item=ds[index];info=ds.get_data_info(index);image_id=Path(info['img_path']).stem
        rgb=np.asarray(Image.open(info['img_path']).convert('RGB'))
        preds=[];gt=None
        for v in variants:
            configure(model.net.visual,v)
            result=model.test_step(pseudo_collate([item]))[0]
            preds.append(result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8))
            gt=result.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        np.savez_compressed(OUT/f'{image_id}_ade_labels.npz',rgb=rgb,gt=gt,variants=np.asarray(variants),predictions=np.stack(preds))
        fig,axes=plt.subplots(2,5,figsize=(16,7),layout='constrained')
        for ax in axes.ravel():ax.axis('off')
        panels=[('RGB',rgb),('GT',palette[np.minimum(gt,150)])]+[(name,palette[pred]) for name,pred in zip(names,preds)]
        for ax,(title,arr) in zip(axes.ravel(),panels):ax.imshow(arr);ax.set_title(title)
        fig.suptitle(f'ADE20K {image_id} | fixed index {index} | released 224/112 sliding, TLP off, KK=.3')
        fig.savefig(OUT/f'{image_id}_ade_segmentation.png',dpi=180);fig.savefig(OUT/f'{image_id}_ade_segmentation.svg');plt.close(fig)
        print(image_id,'ADE figure complete',flush=True)

if __name__=='__main__':main()
