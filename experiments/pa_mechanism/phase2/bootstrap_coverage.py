"""Check ADE20K class coverage under the paired-image bootstrap rule."""
import json
from pathlib import Path

import numpy as np

RUN = Path('/data/Multimodal-final-project/outputs/pa_mechanism/phase2/runs/ade20k_off_2000_24000_qk-qq-kk-sclip-sclip_mean-ns5_pa-ns5_pakk-ns5_g025-ns5_g050-ns5_g075-ns5_g125-svd_pa-svd_pakk-svd_g025-svd_g050-svd_g075-svd_g125.jsonl')
OUT = Path('results/pa_mechanism_phase2_ade_bootstrap_coverage.json')


def main():
    by_image = {}
    for line in RUN.open():
        row = json.loads(line)
        if row['variant'] != 'qq':
            continue
        ids = np.asarray(row['confusion_indices'], dtype=np.int64)
        counts = np.asarray(row['confusion_counts'], dtype=np.int64)
        by_image[row['index']] = np.bincount(ids//150, weights=counts, minlength=150)
    assert len(by_image) == 2000
    support = np.stack([by_image[i] for i in range(2000)])
    rng = np.random.default_rng(24000)
    class_count = []
    for _ in range(1000):
        draw = rng.integers(2000, size=2000)
        class_count.append(int((support[draw].sum(0) > 0).sum()))
    payload = dict(images=2000, classes=150, resamples=1000, seed=24000,
                   full_split_present_classes=int((support.sum(0)>0).sum()),
                   resample_present_class_min=min(class_count),
                   resample_present_class_max=max(class_count),
                   resamples_missing_any_class=sum(x<150 for x in class_count))
    OUT.write_text(json.dumps(payload, indent=2))
    print(payload)


if __name__ == '__main__':
    main()
