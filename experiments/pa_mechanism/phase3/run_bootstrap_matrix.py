"""Paired image bootstrap for selected Phase 3 comparisons on full and holdout splits."""
import csv
import json
from pathlib import Path

from experiments.pa_mechanism.phase3.paired_bootstrap import compare


RESULTS = Path('results')


def main():
    with (RESULTS/'pa_mechanism_phase3_summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    selection = json.loads((RESULTS/'pa_mechanism_phase3_selection.json').read_text())
    index = {(row['dataset'], row['tlp'], row['variant']): row for row in rows}
    pairs = []
    for dataset in ('voc21', 'voc20', 'ade20k'):
        for tlp in ('off', 'on'):
            for method in (selection['shared_gamma'], 'g150'):
                references = ['csa', selection['shared_lambda'], 'g100']
                if tlp == 'on':
                    references.append('original_pearl_ns5')
                for reference in references:
                    if reference == method:
                        continue
                    if (dataset, tlp, method) not in index or (dataset, tlp, reference) not in index:
                        raise RuntimeError((dataset, tlp, method, reference))
                    pairs.append((dataset, tlp, reference, method))
    results = []
    for dataset, tlp, first, second in pairs:
        a, b = index[(dataset, tlp, first)], index[(dataset, tlp, second)]
        for scope in ('full', 'holdout'):
            result = compare(dataset, a['source_file'], a['source_variant'],
                             b['source_file'], b['source_variant'], scope=scope,
                             reps=1000, seed=24003)
            result['tlp'] = tlp
            result['first_alias'] = first
            result['second_alias'] = second
            result['method_status'] = ('development_selected' if second == selection['shared_gamma']
                                       else 'additional_exploratory_gamma')
            results.append(result)
            print(dataset, tlp, scope, first, second,
                  round(result['delta_mIoU'], 4),
                  [round(result['ci95_low'], 4), round(result['ci95_high'], 4)], flush=True)
    out = RESULTS/'pa_mechanism_phase3_bootstrap.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(out, len(results), 'paired comparisons')


if __name__ == '__main__':
    main()
