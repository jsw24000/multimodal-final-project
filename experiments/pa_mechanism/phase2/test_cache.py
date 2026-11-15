"""Ensure cached prefix gives identical ADE sliding-window predictions."""
import os
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS,MODELS
from mmseg.utils import register_all_modules
import custom_datasets,pearl_ovss
from experiments.pa_mechanism.phase2.run_matrix import configure

def main():
    root=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
    register_all_modules();cfg=Config.fromfile('configs/cfg_ade20k.py')
    cfg.model.clip_path=str(root/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop='off';cfg.model.attn_strategy='qq'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval();item=ds[0]
    originals={}
    for v in ('qk','qq','ns5_pakk','svd_pa'):
        configure(model.net.visual,v)
        with torch.inference_mode():originals[v]=model.test_step(pseudo_collate([item]))[0].pred_sem_seg.data.clone()
    model.net.visual.phase2_cache_enabled=True;model.net.visual._phase2_prefix_cache=[]
    for v in originals:
        configure(model.net.visual,v)
        with torch.inference_mode():cached=model.test_step(pseudo_collate([item]))[0].pred_sem_seg.data
        assert torch.equal(originals[v],cached),v
    assert len(model.net.visual._phase2_prefix_cache)>0
    print('ADE sliding cached predictions: exact match')

if __name__=='__main__':main()
