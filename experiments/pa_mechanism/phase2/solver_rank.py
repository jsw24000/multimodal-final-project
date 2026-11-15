"""Effective and stable ranks from the fixed 128-image solver spectrum sample."""
import json
from pathlib import Path

import numpy as np

SOURCE = Path('/data/Multimodal-final-project/outputs/pa_mechanism/phase2/metrics/voc21_solver_spectrum_128_24000.npz')
OUT = Path('results/pa_mechanism_phase2_solver_rank.json')


def main():
    source = np.load(SOURCE)
    summaries = {}
    for key in ('M', 'NS5', 'SVD'):
        s = source[key].astype(np.float64)
        p = s/s.sum(-1, keepdims=True)
        entropy = -(p*np.log(p.clip(1e-300))).sum(-1)
        effective = np.exp(entropy)
        stable = (s*s).sum(-1)/(s.max(-1)**2)
        summaries[key] = dict(effective_rank_mean=float(effective.mean()),
                              effective_rank_median=float(np.median(effective)),
                              stable_rank_mean=float(stable.mean()),
                              singular_min_mean=float(s.min(-1).mean()),
                              singular_max_mean=float(s.max(-1).mean()))
    result = dict(dataset='voc21', images=len(source['indices']), heads=12,
                  matrices=len(source['M']), dimensions=64, seed=24000,
                  definition='effective rank = exp(entropy of normalized singular values); stable rank = sum(s^2)/max(s)^2',
                  summaries=summaries)
    OUT.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
