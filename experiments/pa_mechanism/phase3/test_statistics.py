"""Check sparse marginals reproduce confusion-matrix mIoU exactly."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.pa_mechanism.phase2.summarize import miou as dense_miou
from experiments.pa_mechanism.phase3.paired_bootstrap import load_marginals, miou


def test():
    matrices = [np.array([[8, 2, 1], [1, 9, 0], [0, 2, 3]], dtype=np.int64),
                np.array([[4, 1, 0], [3, 2, 1], [1, 0, 6]], dtype=np.int64)]
    with TemporaryDirectory() as directory:
        path = Path(directory)/'test.jsonl'
        with path.open('w') as f:
            for index, matrix in enumerate(matrices):
                flat = matrix.flatten()
                nz = np.flatnonzero(flat)
                f.write(json.dumps(dict(index=index, variant='a',
                                        confusion_indices=nz.tolist(),
                                        confusion_counts=flat[nz].tolist()))+'\n')
        marginals = load_marginals(path, 'a', 3)
        for i, matrix in enumerate(matrices):
            assert miou(marginals[i]) == dense_miou(matrix)
        assert abs(miou(sum(marginals.values()))-dense_miou(sum(matrices))) < 1e-12


if __name__ == '__main__':
    test()
    print('Phase 3 statistics tests: PASS')
