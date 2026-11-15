"""True-pixel corrected/damaged maps for fixed VOC21 examples."""
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.pa_mechanism.phase2.visualize_cases import PALETTE

ROOT=Path('/data/Multimodal-final-project/outputs/pa_mechanism/phase2/figures')
OUT=Path('results/figures/pa_mechanism_phase2');OUT.mkdir(parents=True,exist_ok=True)

def main():
    for image_id in ('2007_000033','2008_006143','2008_008252'):
        data=np.load(ROOT/f'{image_id}_labels.npz');rgb=data['rgb'];gt=data['gt'];variants=list(data['variants']);preds=data['predictions']
        qq=preds[variants.index('qq')];pa=preds[variants.index('ns5_pa')]
        valid=gt!=255;better=(qq!=gt)&(pa==gt)&valid;worse=(qq==gt)&(pa!=gt)&valid
        change=np.ones((*gt.shape,3),dtype=np.uint8)*245;change[better]=[80,160,125];change[worse]=[230,100,115];change[~valid]=[180,180,180]
        fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
        for ax,title,img in zip(axes,['RGB','GT labels','PA NS-5 vs QQ: green corrected, coral damaged'],[rgb,PALETTE[np.minimum(gt,21)],change]):
            ax.imshow(img)
            ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle(f'VOC21 {image_id} | corrected {better.sum():,} pixels, damaged {worse.sum():,} pixels | valid GT only')
        fig.savefig(OUT/f'{image_id}_error_changes.png',dpi=180);fig.savefig(OUT/f'{image_id}_error_changes.svg');plt.close(fig)
        print(image_id,'corrected',better.sum(),'damaged',worse.sum())

if __name__=='__main__':main()
