import torch
import torch.nn.functional as F
from experiments.pa_mechanism.attention_variants import alignment, attention
from pearl.attn import pa_attn

def test():
    torch.manual_seed(24000)
    q=torch.randn(12,71,64);k=torch.randn_like(q);v=torch.randn_like(q)
    eye=torch.eye(64)
    scale=q.shape[-1]**-.5
    qq=F.softmax((q@q.transpose(1,2))*scale,dim=-1)
    kk=F.softmax((k@k.transpose(1,2))*scale,dim=-1)
    assert torch.allclose(attention(q,k,'qq'),qq,atol=1e-6)
    assert torch.allclose(attention(q,k,'kk'),kk,atol=1e-6)
    assert torch.allclose(attention(q,k,'sclip'),qq+kk,atol=1e-6)
    assert torch.allclose(attention(q,k,'sclip_mean'),(qq+kk)*.5,atol=1e-6)
    assert torch.allclose(attention(q,k,'pa_kk'),pa_attn(q,k,v,return_attn=True),atol=2e-6)
    for solver in ('polar','svd'):
        qc,kc,r,e=alignment(q,k,solver)
        assert torch.isfinite(r).all()
        a0=attention(q,k,'gamma',gamma=0,solver=solver)
        a1=attention(q,k,'gamma',gamma=1,solver=solver)
        pa=attention(q,k,'pure_pa',solver=solver)
        assert torch.allclose(a0,attention(q,k,'qq'),atol=2e-6)
        assert torch.allclose(a1,pa,atol=2e-6)
        assert torch.allclose(a1@v,pa@v,atol=2e-6)
        assert torch.allclose(a1.sum(-1),torch.ones_like(a1.sum(-1)),atol=2e-6)
        direct=q@(k@r).transpose(1,2)
        decomposed=q@q.transpose(1,2)+q@e.transpose(1,2)
        delta=direct-decomposed
        assert (delta-delta[...,:1]).abs().max()<1e-4
        if solver=='svd':assert (r.transpose(1,2)@r-eye).abs().max()<1e-4
    for n in (1,3,5,10,20):
        _,_,r,_=alignment(q,k,'polar',n)
        assert torch.isfinite(r).all(),n
    for solver in ('polar','svd'):
        target=attention(q,k,'pure_pa',solver=solver)
        matched=attention(q,k,'qq_entropy_match',solver=solver)
        entropy=lambda x:-(x*x.clamp_min(1e-12).log()).sum(-1).mean(-1)
        assert (entropy(target)-entropy(matched)).abs().max()<.01

if __name__=='__main__':
    test();print('solver tests: PASS')
