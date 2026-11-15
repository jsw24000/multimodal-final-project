"""ADE sampled-crop Attention means with image-bootstrap uncertainty."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path('/data/Multimodal-final-project/outputs/pa_mechanism/phase2/metrics/ade20k_crop_attention_2000.jsonl')
OUT = Path('results/figures/pa_mechanism_phase2')
STATS = Path('results/pa_mechanism_phase2_ade_crop_bootstrap.json')
VARIANTS = ('qk', 'qq', 'kk', 'pure_pa_ns5', 'pure_pa_svd')
NAMES = ('QK', 'QQ', 'KK', 'PA NS-5', 'PA SVD')
KEYS = ('between_cos', 'off_same', 'entropy')
TITLES = ('Between-class row cosine', 'Off-diagonal same-class mass', 'Attention entropy (nats)')
COLORS = ('#DADDE3', '#A7C7E7', '#5878A6', '#F4A7AE', '#A6D0C6')


def main():
    records = [json.loads(line) for line in SOURCE.open()]
    assert len(records) == 2000
    array = np.full((len(records), len(VARIANTS), len(KEYS)), np.nan)
    for i, record in enumerate(records):
        assert record['index'] == i
        for j, variant in enumerate(VARIANTS):
            group = [entry['metrics'] for entry in record['heads']
                     if entry['variant'] == variant and entry['metrics'] is not None]
            for k, key in enumerate(KEYS):
                values = [row[key] for row in group if row[key] is not None]
                if values:
                    array[i, j, k] = np.mean(values)
    rng = np.random.default_rng(24000)
    samples = np.empty((1000, len(VARIANTS), len(KEYS)))
    for rep in range(1000):
        draw = rng.integers(len(records), size=len(records))
        samples[rep] = np.nanmean(array[draw], axis=0)
    means = np.nanmean(array, axis=0)
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    summary = {variant: {key: dict(mean=float(means[j, k]), ci95=[float(low[j, k]), float(high[j, k])],
                                   contributing_images=int(np.isfinite(array[:, j, k]).sum()))
                         for k, key in enumerate(KEYS)}
               for j, variant in enumerate(VARIANTS)}
    STATS.write_text(json.dumps(dict(dataset='ade20k', images=2000, crops_per_image=1,
                                     bootstrap_images=1000, seed=24000, variants=summary), indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    x = np.arange(len(VARIANTS))
    for k, ax in enumerate(axes):
        ax.bar(x, means[:, k], color=COLORS, edgecolor='#5878A6', linewidth=.5)
        ax.errorbar(x, means[:, k], yerr=[means[:, k]-low[:, k], high[:, k]-means[:, k]],
                    fmt='none', ecolor='#27364B', capsize=3, linewidth=1)
        ax.set_xticks(x, NAMES, rotation=25, ha='right')
        ax.set_title(TITLES[k])
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('ADE20K: 2,000 images, one official-grid crop per image, 12 heads; 95% image bootstrap')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'ade_crop_attention.png', dpi=190)
    fig.savefig(OUT/'ade_crop_attention.svg')
    plt.close(fig)
    print(STATS)


if __name__ == '__main__':
    main()
