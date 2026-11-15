"""Warm, synchronized VOC21 attention and full-image timings without prefix cache."""
import argparse,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.registry import DATASETS,MODELS
from mmseg.utils import register_all_modules
import custom_datasets,pearl_ovss
from experiments.pa_mechanism.attention_variants import attention
from experiments.pa_mechanism.phase2.run_matrix import configure

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tlp',choices=['on','off'],default='off');args=p.parse_args()
    register_all_modules();cfg=Config.fromfile('configs/cfg_voc21.py')
    cfg.model.clip_path=str(ROOT/'checkpoints/clip/ViT-B-16.pt');cfg.model.arch='reduced';cfg.model.use_prop=args.tlp;cfg.model.attn_strategy='qk'
    ds=DATASETS.build(cfg.test_dataloader.dataset);model=MODELS.build(cfg.model).cuda().eval();item=ds[0]
    capture=[];hook=model.net.visual.transformer.resblocks[-1].ln_1.register_forward_hook(lambda m,i,o:capture.__setitem__(slice(None),[o.detach()]))
    model.test_step(pseudo_collate([item]));z=capture[-1]
    layer=model.net.visual.transformer.resblocks[-1].attn;q,k,v=F.linear(z,layer.in_proj_weight,layer.in_proj_bias).chunk(3,dim=-1)
    t,b,d=q.shape;h=layer.num_heads
    q=q.view(t,b*h,d//h).transpose(0,1).float();k=k.view(t,b*h,d//h).transpose(0,1).float();v=v.view(t,b*h,d//h).transpose(0,1).float()
    specs=[('qk','qk','polar'),('qq','qq','polar'),('kk','kk','polar'),('sclip_mean','sclip_mean','polar'),
           ('ns5_pa','pure_pa','polar'),('svd_pa','pure_pa','svd'),('ns5_pakk','pa_kk','polar'),('svd_pakk','pa_kk','svd')]
    rows=[]
    for name,strategy,solver in specs:
        for _ in range(10):_ = attention(q,k,strategy,solver=solver)@v
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):_ = attention(q,k,strategy,solver=solver)@v
        torch.cuda.synchronize();kernel_ms=(time.perf_counter()-start)*10
        configure(model.net.visual,name);model.net.visual.phase2_cache_enabled=False
        for _ in range(5):model.test_step(pseudo_collate([item]))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(20):model.test_step(pseudo_collate([item]))
        torch.cuda.synchronize();image_ms=(time.perf_counter()-start)*50
        rows.append(dict(variant=name,attention_aggregation_ms=kernel_ms,full_image_ms=image_ms,
                         peak_process_memory_MiB=torch.cuda.max_memory_allocated()/1048576))
        print(rows[-1],flush=True)
    hook.remove()
    out=ROOT/f'outputs/pa_mechanism/phase2/metrics/benchmark_voc21_tlp_{args.tlp}.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(image_index=0,attention_repetitions=100,full_repetitions=20,warmup_attention=10,warmup_full=5,
                                   gpu=torch.cuda.get_device_name(),rows=rows),indent=2))

if __name__=='__main__':main()
