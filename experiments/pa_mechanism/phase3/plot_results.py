"""Report-ready full-validation interpolation and final-pipeline figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROWS = Path('results/pa_mechanism_phase3_summary.csv')
SELECTION = Path('results/pa_mechanism_phase3_selection.json')
OUT = Path('results/figures/pa_mechanism_phase3')
COLORS = dict(gamma='#F4A7AE', kk='#5878A6', csa_pa='#A6D0C6',
              csa='#A7C7E7', weighted='#CFEAF6', original='#DADDE3')
DATASETS = ('voc21', 'voc20', 'ade20k')
COUNTS = {'voc21': '1,449', 'voc20': '1,449', 'ade20k': '2,000'}


def main():
    with ROWS.open() as f:
        rows = list(csv.DictReader(f))
    score = {(r['dataset'], r['tlp'], r['variant']): float(r['mIoU']) for r in rows}
    selected = json.loads(SELECTION.read_text())
    gamma = selected['shared_gamma']
    weight = selected['shared_lambda']
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout='constrained')
    paths = (([0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3],
              ['g000', 'g025', 'g050', 'g075', 'g100', 'g125', 'g150', 'g200', 'g250', 'g300'],
              'QQ → Pure PA SVD', 'Residual strength γ', COLORS['gamma']),
             ([0, .25, .5, .75, 1],
              ['kb000', 'kb025', 'kb050', 'kb075', 'kb100'],
              'KK → Pure PA SVD', 'Logit interpolation β', COLORS['kk']),
             ([0, .25, .5, .75, 1],
              ['cb000', 'cb025', 'cb050', 'cb075', 'cb100'],
              'CSA mean → Pure PA SVD', 'Attention mixture β', COLORS['csa_pa']))
    for i, dataset in enumerate(DATASETS):
        for j, (x, variants, title, xlabel, color) in enumerate(paths):
            ax = axes[i, j]
            values = [score[(dataset, 'off', v)] for v in variants]
            ax.plot(x, values, 'o-', color=color, linewidth=2)
            if j == 0:
                ax.axhline(score[(dataset, 'off', 'csa')], color=COLORS['kk'],
                           linestyle='--', linewidth=1, alpha=.8, label='CSA mean')
                ax.legend(fontsize=8)
            ax.set(title=f'{dataset.upper()} · {title}', xlabel=xlabel, ylabel='mIoU (%)')
            ax.title.set_fontsize(10)
            ax.grid(alpha=.2)
    fig.suptitle('Three distinct paths toward Pure PA; complete validation, TLP off', fontsize=16)
    fig.savefig(OUT/'three_paths.png', dpi=190)
    fig.savefig(OUT/'three_paths.svg')
    plt.close(fig)

    labels = ['CSA 1:1', f'CSA λ={int(weight[2:])/100:g}', 'Pure PA γ=1',
              f'Pure PA γ={int(gamma[1:])/100:g}', 'Pure PA γ=1.5*', 'Original PEARL']
    variants = ['csa', weight, 'g100', gamma, 'g150', 'original_pearl_ns5']
    colors = [COLORS['csa'], COLORS['weighted'], COLORS['gamma'], '#E27D87',
              '#B95F6A', COLORS['original']]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for col, dataset in enumerate(DATASETS):
        for row, tlp in enumerate(('off', 'on')):
            ax = axes[row, col]
            values = [score[(dataset, tlp, v)] for v in variants]
            baseline = score[(dataset, tlp, 'csa')]
            deltas = [value-baseline for value in values]
            bars = ax.bar(range(len(values)), deltas, color=colors, edgecolor='#5878A6', linewidth=.5)
            ax.axhline(0, color='#5878A6', linewidth=.8)
            ax.set_xticks(range(len(values)), labels, rotation=30, ha='right', fontsize=8)
            ax.set_title(f'{dataset.upper()} ({COUNTS[dataset]} images), TLP {tlp}')
            ax.set_ylabel('ΔmIoU vs CSA mean (pp)')
            ax.grid(axis='y', alpha=.2)
            margin = max(.2, (max(deltas)-min(deltas))*.12)
            ax.set_ylim(min(0, min(deltas))-margin, max(0, max(deltas))+margin)
            for bar, value, delta in zip(bars, values, deltas):
                ax.text(bar.get_x()+bar.get_width()/2, delta+(.04 if delta >= 0 else -.04),
                        f'{value:.2f}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=8)
    fig.suptitle('Development-selected γ and CSA weight; *additional exploratory gamma; full validation', fontsize=15)
    fig.savefig(OUT/'residual_tlp.png', dpi=190)
    fig.savefig(OUT/'residual_tlp.svg')
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
