"""Compose fixed-ID VOC21 comparison panels from saved predictions."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
PROJECT = Path(__file__).resolve().parents[2]
INDICES = (0, 56, 111, 222, 333, 444, 555, 777, 1000, 1448)
METHODS = ('vanilla', 'sclip', 'naclip', 'pa_only', 'full_pearl')
TITLES = {'vanilla':'Vanilla CLIP', 'sclip':'SCLIP', 'naclip':'NACLIP',
          'pa_only':'PEARL PA only', 'full_pearl':'Full PEARL'}
LABELS = ('background','aeroplane','bicycle','bird','boat','bottle','bus','car',
          'cat','chair','cow','diningtable','dog','horse','motorbike','person',
          'pottedplant','sheep','sofa','train','tvmonitor')


def voc_palette():
    colors = []
    for label in range(21):
        r = g = b = 0
        c = label
        for bit in range(8):
            r |= ((c >> 0) & 1) << (7-bit)
            g |= ((c >> 1) & 1) << (7-bit)
            b |= ((c >> 2) & 1) << (7-bit)
            c >>= 3
        colors.append((r, g, b))
    return np.array(colors + [(255,255,255)] * 235, dtype=np.uint8)


def main():
    assert ROOT.is_dir() and ROOT.stat().st_dev != Path('/').stat().st_dev
    val = (ROOT/'datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt').read_text().splitlines()
    base = ROOT/'datasets/VOCdevkit/VOC2012'
    out = ROOT/'outputs/qualitative/voc21/panels'
    out.mkdir(parents=True, exist_ok=True)
    pal = voc_palette()
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font = ImageFont.truetype(font_path, 22)
    legend_font = ImageFont.truetype(font_path, 13)
    summary = []
    for index in INDICES:
        image_id = val[index]
        image = Image.open(base/'JPEGImages'/f'{image_id}.jpg').convert('RGB')
        gt = np.array(Image.open(base/'SegmentationClass'/f'{image_id}.png'))
        masks = {'Ground Truth': gt}
        for method in METHODS:
            path = ROOT/'outputs/qualitative/voc21'/method/f'{image_id}.png'
            if path.is_file(): masks[method] = np.array(Image.open(path))
        cols = [('Input', image), ('Ground Truth', Image.fromarray(pal[gt]).convert('RGB'))]
        for method in METHODS:
            if method in masks:
                cols.append((TITLES[method], Image.fromarray(pal[masks[method]]).convert('RGB')))
        width, height = 480, 360
        legend_rows = (len(LABELS)+1+len(cols)-1)//len(cols)
        canvas = Image.new('RGB', (width*len(cols), height+8+legend_rows*42), '#f7f7f7')
        draw = ImageDraw.Draw(canvas)
        for col, (title, picture) in enumerate(cols):
            panel = ImageOps.contain(picture, (width-16,height-16))
            x = col*width + (width-panel.width)//2
            y = 38+(height-40-panel.height)//2
            canvas.paste(panel,(x,y))
            draw.text((col*width+12,8),title,fill='black',font=title_font)
        for label, name in enumerate(LABELS):
            x = 14+(label%len(cols))*width
            y = height+8+(label//len(cols))*42
            color = tuple(map(int,pal[label]))
            draw.rectangle((x,y,x+18,y+18),fill=color,outline='black')
            draw.text((x+24,y),f'{label}: {name}',fill='black',font=legend_font)
        ignore_index = len(LABELS)
        x = 14+(ignore_index%len(cols))*width
        y = height+8+(ignore_index//len(cols))*42
        draw.rectangle((x,y,x+18,y+18),fill='white',outline='black')
        draw.text((x+24,y),'255: ignore',fill='black',font=legend_font)
        target=out/f'{image_id}.png'
        canvas.save(target, optimize=True)
        def iou(pred):
            valid = gt != 255
            classes = np.union1d(gt[valid],pred[valid])
            values = []
            for c in classes:
                inter = np.logical_and(pred[valid]==c,gt[valid]==c).sum()
                union = np.logical_or(pred[valid]==c,gt[valid]==c).sum()
                if union: values.append(inter/union)
            return round(float(np.mean(values)),3) if values else None
        summary.append(dict(index=index,image_id=image_id,panels=[c[0] for c in cols],
                            per_image_iou={m:iou(masks[m]) for m in METHODS if m in masks}))
        print(target,flush=True)
    (out/'selection.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__': main()
