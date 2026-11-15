import torch
from pearl.attn import pa_attn
from experiments.pa_mechanism.attention_variants import attention, alignment


def test_decomposition():
    torch.manual_seed(17)
    q, k, v = [torch.randn(3, 31, 64) for _ in range(3)]
    a0 = attention(q, k, 'gamma', gamma=0)
    a1 = attention(q, k, 'gamma', gamma=1)
    qq = attention(q, k, 'qq')
    pa = attention(q, k, 'pure_pa')
    official = pa_attn(q, k, v, use_kk=False, return_attn=True)
    official_kk = pa_attn(q, k, v, use_kk=True, alpha_kk=.08, return_attn=True)
    assert torch.allclose(a0, qq, atol=2e-6)
    assert torch.allclose(a1, pa, atol=2e-6)
    assert torch.allclose(pa, official, atol=2e-6)
    assert torch.allclose(attention(q,k,'pa_kk'), official_kk, atol=2e-6)
    assert torch.allclose(a1 @ v, pa @ v, atol=2e-6)
    qc, kc, r, e = alignment(q, k)
    scale = 64**-.5
    log_direct = (q @ (k @ r).transpose(1, 2))*scale
    log_decomp = (q @ q.transpose(1, 2) + q @ e.transpose(1, 2))*scale
    shifted = log_direct - log_decomp
    assert (shifted - shifted[..., :1]).abs().max() < 2e-5
    assert torch.allclose(attention(q,k,'sclip').sum(-1), torch.full((3,31),2.), atol=2e-6)


if __name__ == '__main__':
    test_decomposition()
    print('numerical equivalence: PASS')
