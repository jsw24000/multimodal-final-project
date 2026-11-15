"""Combine Phase 1 and Phase 2 raw confusions into controlled per-class tables."""
import csv
import json
from pathlib import Path

import numpy as np

from experiments.pa_mechanism.phase2.summarize import load


DATA = Path('/data/Multimodal-final-project/outputs/pa_mechanism')
HERE = Path('results')
PHASE1 = {'voc21': DATA/'runs/eval_1449_24000_11.json',
          'voc20': DATA/'runs/eval_voc20_1449_24000_4.json'}


def phase1(dataset):
    source = json.loads(PHASE1[dataset].read_text())
    nclasses = 21 if dataset == 'voc21' else 20
    totals = {}
    counts = {}
    for row in source['records']:
        if row['variant'].startswith('gamma_'):
            continue
        totals.setdefault(row['variant'], np.zeros((nclasses, nclasses), dtype=np.int64))
        totals[row['variant']] += np.asarray(row['confusion'], dtype=np.int64)
        counts[row['variant']] = counts.get(row['variant'], 0)+1
    assert all(count == 1449 for count in counts.values()), counts
    return totals


def phase2(dataset):
    run_dir = DATA/'phase2/runs'
    prefixes = [f'{dataset}_off_1449_24000_svd_pa-svd_pakk-svd_g025-svd_g050-svd_g075-svd_g125.jsonl']
    if dataset == 'voc20':
        prefixes.append('voc20_off_1449_24000_kk-sclip-sclip_mean.jsonl')
    totals = {}
    for name in prefixes:
        _, matrices = load(run_dir/name)
        for variant, by_image in matrices.items():
            if variant.startswith('svd_g'):
                continue
            totals[variant] = sum(by_image.values())
    return totals


def main():
    for dataset in ('voc21', 'voc20'):
        matrices = {**phase1(dataset), **phase2(dataset)}
        rows = []
        for variant, cm in matrices.items():
            tp = np.diag(cm)
            union = cm.sum(0)+cm.sum(1)-tp
            for c in range(len(cm)):
                pred, gt = cm[:, c].sum(), cm[c].sum()
                rows.append(dict(dataset=dataset, tlp='off', variant=variant, class_id=c,
                                 gt_pixels=int(gt), pred_pixels=int(pred),
                                 iou=float(tp[c]/union[c]*100) if union[c] else float('nan'),
                                 precision=float(tp[c]/pred*100) if pred else float('nan'),
                                 recall=float(tp[c]/gt*100) if gt else float('nan')))
        out = HERE/f'pa_mechanism_phase2_{dataset}_off.per_class.csv'
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(dataset, len(matrices), len(rows), out)


if __name__ == '__main__':
    main()
