"""Merge complete Phase 1/2 endpoints with Phase 3 runs and frozen dev/holdout splits."""
import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get('PEARL_DATA_ROOT', '/data/Multimodal-final-project'))
OLD = ROOT/'outputs/pa_mechanism'
NEW = OLD/'phase3/runs'
RESULTS = Path('results')
PROTOCOL = json.loads((RESULTS/'pa_mechanism_phase3_selection_protocol.json').read_text())
SIZES = {'voc21': (1449, 21), 'voc20': (1449, 20), 'ade20k': (2000, 150)}


def split_indices(dataset, n):
    rule = PROTOCOL['ade20k'] if dataset == 'ade20k' else PROTOCOL['voc21_voc20']
    permutation = np.random.default_rng(rule['seed']).permutation(n)
    dev = set(map(int, permutation[:rule['development_images']]))
    return dev


def metrics(cm):
    tp = np.diag(cm)
    union = cm.sum(0)+cm.sum(1)-tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union>0)
    acc = np.divide(tp, cm.sum(1), out=np.full(len(tp), np.nan), where=cm.sum(1)>0)
    return dict(mIoU=float(np.nanmean(iou)*100), mAcc=float(np.nanmean(acc)*100),
                pAcc=float(tp.sum()/cm.sum()*100))


def consume(path, dataset, tlp, mapping, source_phase, dev):
    n, classes = SIZES[dataset]
    targets = {alias for aliases in mapping.values() for alias in aliases}
    source_for_alias = {alias: source for source, aliases in mapping.items() for alias in aliases}
    totals = {alias: [np.zeros((classes, classes), dtype=np.int64) for _ in range(3)] for alias in targets}
    seen = {alias: set() for alias in targets}
    if path.suffix == '.jsonl':
        iterator = (json.loads(line) for line in path.open())
    else:
        iterator = iter(json.loads(path.read_text())['records'])
    for record in iterator:
        source = record['variant']
        if source not in mapping:
            continue
        index = record['index']
        if 'confusion' in record:
            cm = np.asarray(record['confusion'], dtype=np.int64)
        else:
            cm = np.zeros(classes*classes, dtype=np.int64)
            cm[record['confusion_indices']] = record['confusion_counts']
            cm = cm.reshape(classes, classes)
        assert cm.shape == (classes, classes)
        for alias in mapping[source]:
            if index in seen[alias]:
                raise RuntimeError((path, alias, index))
            seen[alias].add(index)
            totals[alias][0] += cm
            totals[alias][1 if index in dev else 2] += cm
    rows = []
    for alias in sorted(targets):
        if seen[alias] != set(range(n)):
            raise RuntimeError(f'incomplete {path} {alias}: {len(seen[alias])}/{n}')
        full, development, holdout = totals[alias]
        row = dict(dataset=dataset, tlp=tlp, variant=alias, n=n,
                   dev_n=len(dev), holdout_n=n-len(dev), source_phase=source_phase,
                   source_file=str(path), source_variant=source_for_alias[alias], **metrics(full))
        row['dev_mIoU'] = metrics(development)['mIoU']
        row['holdout_mIoU'] = metrics(holdout)['mIoU']
        rows.append(row)
    return rows


def one(pattern):
    matches = list(OLD.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError((pattern, matches))
    return matches[0]


def history(dataset, dev):
    rows = []
    if dataset == 'voc21':
        rows += consume(OLD/'runs/eval_1449_24000_11.json', dataset, 'off',
                        {'qq': ['qq', 'g000', 'cw100'], 'kk': ['kk', 'kb000', 'cw000'],
                         'sclip_mean': ['csa', 'cb000', 'cw050'],
                         'pa_kk': ['original_pearl_ns5']}, 1, dev)
    elif dataset == 'voc20':
        rows += consume(OLD/'runs/eval_voc20_1449_24000_4.json', dataset, 'off',
                        {'qq': ['qq', 'g000', 'cw100'],
                         'pa_kk': ['original_pearl_ns5']}, 1, dev)
        rows += consume(one('phase2/runs/voc20_off_1449_24000_kk-sclip-sclip_mean.jsonl'), dataset, 'off',
                        {'kk': ['kk', 'kb000', 'cw000'], 'sclip_mean': ['csa', 'cb000', 'cw050']}, 2, dev)
    else:
        rows += consume(one('phase2/runs/ade20k_off_2000_24000_qk-qq-kk-sclip-sclip_mean-*.jsonl'), dataset, 'off',
                        {'qq': ['qq', 'g000', 'cw100'], 'kk': ['kk', 'kb000', 'cw000'],
                         'sclip_mean': ['csa', 'cb000', 'cw050'],
                         'svd_pa': ['pa', 'g100', 'kb100', 'cb100'],
                         'ns5_pakk': ['original_pearl_ns5'],
                         'svd_pakk': ['svd_pa_kk'],
                         'svd_g025': ['g025'], 'svd_g050': ['g050'],
                         'svd_g075': ['g075'], 'svd_g125': ['g125']}, 2, dev)
    if dataset != 'ade20k':
        rows += consume(one(f'phase2/runs/{dataset}_off_1449_24000_svd_pa-svd_pakk-*.jsonl'), dataset, 'off',
                        {'svd_pa': ['pa', 'g100', 'kb100', 'cb100'],
                         'svd_pakk': ['svd_pa_kk'],
                         'svd_g025': ['g025'], 'svd_g050': ['g050'],
                         'svd_g075': ['g075'], 'svd_g125': ['g125']}, 2, dev)
    rows += consume(one(f'phase2/runs/{dataset}_on_{SIZES[dataset][0]}_24000_qq-kk-sclip-sclip_mean-ns5_pa-ns5_pakk-svd_pakk.jsonl'),
                    dataset, 'on',
                    {'qq': ['qq', 'cw100'], 'kk': ['kk', 'cw000'],
                     'sclip_mean': ['csa', 'cw050', 'cb000'],
                     'ns5_pakk': ['original_pearl_ns5'],
                     'svd_pakk': ['svd_pa_kk']}, 2, dev)
    return rows


def phase3_rows(dataset, dev):
    n = SIZES[dataset][0]
    rows = []
    for config_path in sorted(NEW.glob(f'{dataset}_*_{n}_24003_*.config.json')):
        config = json.loads(config_path.read_text())
        path = config_path.with_name(config_path.name.replace('.config.json', '.jsonl'))
        mapping = {variant: [variant] for variant in config['variants']}
        rows += consume(path, dataset, config['tlp'], mapping, 3, dev)
    return rows


def choose(rows):
    index = {(r['dataset'], r['tlp'], r['variant']): r for r in rows}
    gamma_candidates = [f'g{round(v*100):03d}' for v in PROTOCOL['gamma_candidates']]
    lambda_candidates = [f'cw{round(v*100):03d}' for v in PROTOCOL['weighted_csa_lambda_candidates']]
    def mean_dev_delta(candidate):
        return np.mean([index[(ds, 'off', candidate)]['dev_mIoU']-index[(ds, 'off', 'csa')]['dev_mIoU']
                        for ds in SIZES])
    best_gamma = max(gamma_candidates, key=lambda v: (mean_dev_delta(v), -int(v[1:])))
    best_lambda = max(lambda_candidates, key=lambda v: (mean_dev_delta(v), -int(v[2:])))
    report = dict(protocol=PROTOCOL, shared_gamma=best_gamma, shared_lambda=best_lambda,
                  candidates={v: float(mean_dev_delta(v)) for v in gamma_candidates+lambda_candidates},
                  dataset_checks={ds: {variant: dict(development_mIoU=index[(ds, 'off', variant)]['dev_mIoU'],
                                                    holdout_mIoU=index[(ds, 'off', variant)]['holdout_mIoU'],
                                                    full_mIoU=index[(ds, 'off', variant)]['mIoU'])
                                       for variant in (best_gamma, best_lambda, 'csa', 'g100')}
                                  for ds in SIZES})
    (RESULTS/'pa_mechanism_phase3_selection.json').write_text(json.dumps(report, indent=2))
    return report


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    for dataset, (n, _) in SIZES.items():
        dev = split_indices(dataset, n)
        rows += history(dataset, dev)
        rows += phase3_rows(dataset, dev)
    keys = [(r['dataset'], r['tlp'], r['variant']) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('duplicate dataset/tlp/variant')
    rows.sort(key=lambda r: (r['dataset'], r['tlp'], r['variant']))
    write_csv(RESULTS/'pa_mechanism_phase3_summary.csv', rows)
    selection = choose(rows)
    index = {(r['dataset'], r['tlp'], r['variant']): r for r in rows}
    comparisons = []
    for row in rows:
        ds, tlp, variant = row['dataset'], row['tlp'], row['variant']
        if row['source_phase'] != 3:
            continue
        refs = {name: index.get((ds, tlp, name)) for name in ('qq', 'kk', 'csa', 'pa' if tlp == 'off' else 'g100')}
        base = 'qq' if variant.startswith('g') or variant.startswith('rk_') else 'kk' if variant.startswith(('kb', 'cw')) else 'csa'
        for reference, baseline in refs.items():
            if baseline is None:
                continue
            comparisons.append(dict(dataset=ds, tlp=tlp, variant=variant,
                                    reference=reference, full_delta_mIoU=row['mIoU']-baseline['mIoU'],
                                    development_delta_mIoU=row['dev_mIoU']-baseline['dev_mIoU'],
                                    holdout_delta_mIoU=row['holdout_mIoU']-baseline['holdout_mIoU'],
                                    n=row['n'], holdout_n=row['holdout_n']))
    if comparisons:
        write_csv(RESULTS/'pa_mechanism_phase3_comparisons.csv', comparisons)
    print('rows', len(rows), 'shared gamma', selection['shared_gamma'],
          'shared lambda', selection['shared_lambda'])


if __name__ == '__main__':
    main()
