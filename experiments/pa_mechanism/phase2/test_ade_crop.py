"""Check released-grid coordinates and ADE class-0/ignore handling."""
import torch

from experiments.pa_mechanism.phase2.diagnose_ade_crop import crop_coordinates, semantic_metrics


def test():
    assert crop_coordinates(336, 448, 224, 112, 0) == (0, 224, 0, 224, 0, 6)
    assert crop_coordinates(336, 448, 224, 112, 5) == (112, 336, 224, 448, 5, 6)
    a = torch.tensor([[1., 0., 0., 0.],
                      [0., .2, .4, .4],
                      [0., .1, .6, .3],
                      [0., .2, .3, .5]])
    labels = torch.tensor([0, 1, 255])
    purity = torch.ones(3)
    result = semantic_metrics(a, labels, purity)
    assert result['valid_patches'] == 2
    assert result['between_cos'] is not None
    assert result['high_purity_fraction'] == 1.


if __name__ == '__main__':
    test()
    print('ADE crop tests: PASS')
