"""Render pilot figures from immutable prediction and metrics artifacts."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(os.environ.get('PEARL_DATA_ROOT','/data/Multimodal-final-project'))
OUT=ROOT/'outputs/pa_mechanism'
FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
PRED=OUT/'predictions'
REPO=Path(__file__).resolve().parents[2]

def pred_path(variant,image_id):
    newer=PRED/'voc21'/variant/f'{image_id}.png'
    return newer if newer.exists() else PRED/variant/f'{image_id}.png'

def scores(records,variants):
    values=[]
    for variant in variants:
        cm=sum((np.asarray(r['confusion']) for r in records if r['variant']==variant))
        union=cm.sum(0)+cm.sum(1)-np.diag(cm)
        iou=np.divide(np.diag(cm),union,out=np.full(21,np.nan),where=union>0)
        values.append(float(np.nanmean(iou)*100))
    return values

def main():
    run=json.loads((OUT/'runs/pilot_32_24000.json').read_text())
    ids=[run['records'][j*11]['image_id'] for j in range(5)]
    ids += [x for x in ('2008_006143','2008_008252','2007_000033') if pred_path('pa_kk',x).exists()]
    palette=np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128],[255,255,255]],dtype=np.uint8)
    variants=['qk','qq','kk','pure_pa','pa_kk']
    for image_id in ids:
        base=ROOT/'datasets/VOCdevkit/VOC2012'
        rgb=np.asarray(Image.open(base/'JPEGImages'/f'{image_id}.jpg').convert('RGB'))
        gt=np.asarray(Image.open(base/'SegmentationClass'/f'{image_id}.png'))
        fig,axes=plt.subplots(1,7,figsize=(20,3.5),layout='constrained')
        arrays=[rgb,palette[np.minimum(gt,21)]]+[palette[np.asarray(Image.open(pred_path(v,image_id)))] for v in variants]
        for ax,title,arr in zip(axes,['Input','GT','Reduced QK','QQ','KK','Pure PA','PA + KK'],arrays):
            ax.imshow(arr);ax.set_title(title);ax.axis('off')
        provenance='full-validation extreme' if image_id in ('2008_006143','2008_008252') else 'fixed subset'
        fig.suptitle(f'VOC21 {image_id} | {provenance} | TLP off',fontsize=13)
        fig.savefig(FIG/f'segmentation_{image_id}.png',dpi=170);fig.savefig(FIG/f'segmentation_{image_id}.svg');plt.close(fig)
    image_id=ids[0]
    base=ROOT/'datasets/VOCdevkit/VOC2012'
    rgb=np.asarray(Image.open(base/'JPEGImages'/f'{image_id}.jpg').convert('RGB'))
    gt=np.asarray(Image.open(base/'SegmentationClass'/f'{image_id}.png'))
    gamma_variants=['qq','gamma_0.25','gamma_0.5','gamma_0.75','pure_pa','gamma_1.25']
    fig,axes=plt.subplots(1,8,figsize=(21,3.5),layout='constrained')
    arrays=[rgb,palette[np.minimum(gt,21)]]+[palette[np.asarray(Image.open(pred_path(v,image_id)))] for v in gamma_variants]
    for ax,title,arr in zip(axes,['Input','GT','γ=0','γ=.25','γ=.5','γ=.75','γ=1','γ=1.25'],arrays):
        ax.imshow(arr);ax.set_title(title);ax.axis('off')
    fig.suptitle(f'VOC21 {image_id} | same checkpoint and image | TLP off')
    fig.savefig(FIG/f'residual_{image_id}.png',dpi=170);fig.savefig(FIG/f'residual_{image_id}.svg');plt.close(fig)
    gammas=[0,.25,.5,.75,1,1.25]
    vals=scores(run['records'],['qq','gamma_0.25','gamma_0.5','gamma_0.75','pure_pa','gamma_1.25'])
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained');ax.plot(gammas,vals,'o-',color='#5878A6');ax.set(xlabel='Residual strength γ',ylabel='VOC21 pilot mIoU (%)',title='32 fixed images; descriptive only');ax.grid(alpha=.25)
    fig.savefig(FIG/'gamma_pilot.png',dpi=180);fig.savefig(FIG/'gamma_pilot.svg');plt.close(fig)
    full=OUT/'runs/eval_1449_24000_11.json'
    if full.exists():
        data=json.loads(full.read_text())
        if len(data['records'])==1449*11:
            vals=scores(data['records'],['qq','gamma_0.25','gamma_0.5','gamma_0.75','pure_pa','gamma_1.25'])
            fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
            ax.plot(gammas,vals,'o-',color='#5878A6');ax.set(xlabel='Residual strength γ',ylabel='VOC21 mIoU (%)',title='Full validation, 1,449 images');ax.grid(alpha=.25)
            fig.savefig(FIG/'gamma_full_voc21.png',dpi=180);fig.savefig(FIG/'gamma_full_voc21.svg');plt.close(fig)
    geo=OUT/'metrics/geometry_16_24000.json'
    if geo.exists():
        rows=json.loads(geo.read_text())['rows']
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for ax,key,title in zip(axes,['alignment_error','orthogonality_error'],['Relative alignment error','Polar orthogonality error']):
            ax.hist([r[key] for r in rows],bins=18,color='#A7C7E7',edgecolor='#5878A6');ax.set_title(title);ax.set_xlabel('Per image/head');ax.set_ylabel('Count')
        fig.savefig(FIG/'geometry_pilot.png',dpi=180);fig.savefig(FIG/'geometry_pilot.svg');plt.close(fig)
    print(FIG)

if __name__=='__main__':main()
