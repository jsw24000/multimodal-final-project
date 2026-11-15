"""Re-run selected full-validation IDs and save prediction panels."""
import argparse, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules
import custom_datasets, pearl_ovss

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))

def main():
    p=argparse.ArgumentParser();p.add_argument('indices',nargs='+',type=int);a=p.parse_args()
    register_all_modules();cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop='off';cfg.model.attn_strategy='qk'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval()
    out=ROOT/'outputs/pa_mechanism/predictions/voc21'
    for i in a.indices:
        item=ds[i];image_id=Path(ds.get_data_info(i)['img_path']).stem
        for variant in ('qk','qq','kk','pure_pa','pa_kk'):
            model.net.visual.attn_strategy=variant
            with torch.inference_mode():result=model.test_step(pseudo_collate([item]))[0]
            folder=out/variant;folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)).save(folder/f'{image_id}.png')
        print(i,image_id,flush=True)

if __name__=='__main__':main()
