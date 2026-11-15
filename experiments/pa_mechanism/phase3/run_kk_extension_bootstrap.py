"""Focused paired intervals for the conditional centered-KK extension."""
import csv
from pathlib import Path

from experiments.pa_mechanism.phase3.paired_bootstrap import compare


RESULTS = Path('results')


def main():
    with (RESULTS/'pa_mechanism_phase3_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    index = {(r['dataset'], r['tlp'], r['variant']): r for r in rows}
    pairs = [(ds, 'off', 'g125', 'rk_g125_a003') for ds in ('voc21', 'voc20', 'ade20k')]
    pairs += [('ade20k', 'off', 'g125', 'rk_g125_a008'),
              ('ade20k', 'on', 'g125', 'rk_g125_a008'),
              ('ade20k', 'on', 'original_pearl_ns5', 'rk_g125_a008')]
    output = []
    for dataset, tlp, first, second in pairs:
        a = index[(dataset, tlp, first)]
        b = index[(dataset, tlp, second)]
        for scope in ('full', 'holdout'):
            result = compare(dataset, a['source_file'], a['source_variant'],
                             b['source_file'], b['source_variant'],
                             scope=scope, reps=1000, seed=24003)
            result.update(tlp=tlp, first_alias=first, second_alias=second)
            output.append(result)
            print(dataset, tlp, scope, first, second,
                  round(result['delta_mIoU'], 4),
                  [round(result['ci95_low'], 4), round(result['ci95_high'], 4)], flush=True)
    path = RESULTS/'pa_mechanism_phase3_kk_extension_bootstrap.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(path)


if __name__ == '__main__':
    main()
