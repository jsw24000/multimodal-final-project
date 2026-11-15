"""Per-class pooled IoU changes for selected residual controls, with GT support."""
import csv
import json
from pathlib import Path

import numpy as np

from experiments.pa_mechanism.phase3.aggregate_results import SIZES
from experiments.pa_mechanism.phase3.paired_bootstrap import load_marginals


RESULTS = Path('results')


def aggregate(row, nclasses):
    found = load_marginals(row['source_file'], row['source_variant'], nclasses)
    return np.stack(list(found.values())).sum(0)


def class_iou(stats):
    tp, gt, pred = stats
    union = gt + pred - tp
    return np.divide(tp, union, out=np.full(len(union), np.nan), where=union > 0) * 100


def main():
    with (RESULTS/'pa_mechanism_phase3_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selection = json.loads((RESULTS/'pa_mechanism_phase3_selection.json').read_text())
    index = {(r['dataset'], r['tlp'], r['variant']): r for r in rows}
    output = []
    for dataset, (_, nclasses) in SIZES.items():
        for tlp in ('off', 'on'):
            for method in (selection['shared_gamma'], 'g150'):
                references = ('csa', 'original_pearl_ns5') if tlp == 'on' else ('csa', 'g100')
                method_stats = aggregate(index[(dataset, tlp, method)], nclasses)
                method_iou = class_iou(method_stats)
                for reference in references:
                    ref_stats = aggregate(index[(dataset, tlp, reference)], nclasses)
                    ref_iou = class_iou(ref_stats)
                    for class_id in range(nclasses):
                        before = ref_iou[class_id]
                        after = method_iou[class_id]
                        output.append(dict(dataset=dataset, tlp=tlp, method=method,
                                           reference=reference, class_id=class_id,
                                           gt_pixels=int(ref_stats[1, class_id]),
                                           reference_iou=before, method_iou=after,
                                           delta_iou=after-before if np.isfinite(before) and np.isfinite(after) else float('nan')))
    path = RESULTS/'pa_mechanism_phase3_class_deltas.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(path, len(output), 'class comparisons')


if __name__ == '__main__':
    main()
