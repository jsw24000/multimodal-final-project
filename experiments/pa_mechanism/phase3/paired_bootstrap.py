"""Exact pooled-mIoU image bootstrap using TP/GT/pred marginals instead of dense confusions."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.pa_mechanism.phase3.aggregate_results import SIZES, split_indices


@lru_cache(maxsize=64)
def load_marginals(path, variant, nclasses):
    path = Path(path)
    if path.suffix == '.jsonl':
        records = (json.loads(line) for line in path.open())
    else:
        records = iter(json.loads(path.read_text())['records'])
    found = {}
    for row in records:
        if row['variant'] != variant:
            continue
        if 'confusion' in row:
            matrix = np.asarray(row['confusion'], dtype=np.int64)
            tp = np.diag(matrix)
            gt = matrix.sum(1)
            pred = matrix.sum(0)
        else:
            flat = np.asarray(row['confusion_indices'], dtype=np.int64)
            counts = np.asarray(row['confusion_counts'], dtype=np.int64)
            gt = np.bincount(flat//nclasses, weights=counts, minlength=nclasses).astype(np.int64)
            pred = np.bincount(flat%nclasses, weights=counts, minlength=nclasses).astype(np.int64)
            tp = np.bincount(flat[flat//nclasses == flat%nclasses]//nclasses,
                             weights=counts[flat//nclasses == flat%nclasses],
                             minlength=nclasses).astype(np.int64)
        index = row['index']
        if index in found:
            raise RuntimeError(f'duplicate {variant} index {index}: {path}')
        found[index] = np.stack((tp, gt, pred))
    return found


def miou(marginals):
    tp, gt, pred = marginals
    union = gt+pred-tp
    return float(np.nanmean(np.divide(tp, union, out=np.full(len(union), np.nan), where=union>0))*100)


def compare(dataset, first_path, first_variant, second_path, second_variant,
            scope='full', reps=1000, seed=24003):
    n, classes = SIZES[dataset]
    first = load_marginals(first_path, first_variant, classes)
    second = load_marginals(second_path, second_variant, classes)
    if set(first) != set(range(n)) or set(second) != set(range(n)):
        raise RuntimeError('incomplete paired split')
    dev = split_indices(dataset, n)
    indices = [i for i in range(n) if scope == 'full' or (i in dev if scope == 'development' else i not in dev)]
    if scope not in ('full', 'development', 'holdout'):
        raise ValueError(scope)
    a = np.stack([first[i] for i in indices])
    b = np.stack([second[i] for i in indices])
    delta = miou(b.sum(0))-miou(a.sum(0))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for rep in range(reps):
        sample = rng.integers(len(indices), size=len(indices))
        draws[rep] = miou(b[sample].sum(0))-miou(a[sample].sum(0))
    low, high = np.percentile(draws, [2.5, 97.5])
    return dict(dataset=dataset, scope=scope, n=len(indices), first=first_variant,
                second=second_variant, first_file=str(first_path), second_file=str(second_path),
                delta_mIoU=delta, ci95_low=float(low), ci95_high=float(high),
                repetitions=reps, seed=seed,
                class_inclusion='classes with positive pooled GT-or-pred union in each resample')
