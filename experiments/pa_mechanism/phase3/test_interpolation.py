"""Endpoint, orthogonality, probability and value-aggregation checks."""
import torch

from experiments.pa_mechanism.attention_variants import alignment, attention


def test():
    torch.manual_seed(24003)
    q = torch.randn(12, 71, 64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qq = attention(q, k, 'qq')
    kk = attention(q, k, 'kk')
    csa = attention(q, k, 'sclip_mean')
    pa = attention(q, k, 'pure_pa', solver='svd')
    _, kc, r, e = alignment(q, k, 'svd')
    eye = torch.eye(64)
    assert (r.transpose(-2, -1) @ r-eye).abs().max() < 2e-5
    assert torch.allclose((k@r)@(k@r).transpose(-2, -1), k@k.transpose(-2, -1), atol=2e-4)
    assert torch.allclose(attention(q, k, 'gamma', gamma=0., solver='svd'), qq, atol=2e-6)
    assert torch.allclose(attention(q, k, 'gamma', gamma=1., solver='svd'), pa, atol=2e-6)
    assert torch.allclose(attention(q, k, 'kk_pa_beta', beta=0., solver='svd'), kk, atol=2e-6)
    assert torch.allclose(attention(q, k, 'kk_pa_beta', beta=1., solver='svd'), pa, atol=2e-5)
    assert torch.allclose(attention(q, k, 'csa_pa_beta', beta=0., solver='svd'), csa, atol=2e-6)
    assert torch.allclose(attention(q, k, 'csa_pa_beta', beta=1., solver='svd'), pa, atol=2e-6)
    assert torch.allclose(attention(q, k, 'csa_weighted', lambda_csa=.5), csa, atol=2e-6)
    assert torch.allclose(attention(q, k, 'csa_weighted', lambda_csa=0.), kk, atol=2e-6)
    assert torch.allclose(attention(q, k, 'csa_weighted', lambda_csa=1.), qq, atol=2e-6)
    for beta in (0., .25, .5, .75, 1.):
        for strategy in ('csa_pa_beta', 'csa_weighted'):
            a = attention(q, k, strategy, beta=beta, lambda_csa=beta, solver='svd')
            assert (a >= 0).all()
            assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=2e-6)
            assert torch.isfinite(a@v).all()
    for gamma in (0., 1., 1.5, 3.):
        a = attention(q, k, 'gamma', gamma=gamma, solver='svd')
        cached = attention(q, k, 'gamma', gamma=gamma, solver='svd', alignment_result=alignment(q, k, 'svd'))
        assert torch.allclose(a, cached, atol=2e-6)
        ext = attention(q, k, 'residual_kk', gamma=gamma, alpha_kk=0., solver='svd')
        assert torch.allclose(a, ext, atol=2e-6)
        assert torch.allclose(a@v, ext@v, atol=2e-6)
    for alpha in (.08, .3):
        ext = attention(q, k, 'residual_kk', gamma=1., alpha_kk=alpha, solver='svd')
        old = attention(q, k, 'pa_kk', alpha_kk=alpha, solver='svd')
        assert torch.allclose(ext, old, atol=2e-6)
        assert torch.allclose(ext@v, old@v, atol=2e-6)
    # The omitted centering term is row constant for every Query.
    scale = q.shape[-1]**-.5
    direct = (q@(k@r).transpose(-2, -1))*scale
    decomposed = (q@q.transpose(-2, -1)+q@e.transpose(-2, -1))*scale
    shift = direct-decomposed
    assert (shift-shift[..., :1]).abs().max() < 1e-4
    assert torch.allclose(torch.softmax(direct, -1), torch.softmax(decomposed, -1), atol=2e-6)


if __name__ == '__main__':
    test()
    print('Phase 3 interpolation tests: PASS')
