import torch
import torch.nn.functional as F
from pearl.attn import _polar_unitary_newton_schulz


def alignment(q, k, solver='polar'):
    q, k = q.float(), k.float()
    w = q.norm(dim=-1, keepdim=True)
    w[:, :1] = 0
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    qc, kc = q - (w*q).sum(1, keepdim=True), k - (w*k).sum(1, keepdim=True)
    m = kc.transpose(1, 2) @ qc
    if solver == 'polar':
        r = _polar_unitary_newton_schulz(m)
    elif solver == 'svd':
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        r = u @ vh
    else:
        raise ValueError(solver)
    return qc, kc, r, kc @ r - qc


def attention(q, k, strategy, *, gamma=1., alpha_kk=.08, solver='polar'):
    q, k = q.float(), k.float()
    scale = q.shape[-1] ** -.5
    qq = (q @ q.transpose(1, 2)) * scale
    kk = (k @ k.transpose(1, 2)) * scale
    if strategy == 'qk': logits = (q @ k.transpose(1, 2)) * scale
    elif strategy == 'qq': logits = qq
    elif strategy == 'kk': logits = kk
    elif strategy in ('sclip', 'sclip_mean'):
        a = F.softmax(qq, -1) + F.softmax(kk, -1)
        return a if strategy == 'sclip' else a * .5
    elif strategy in ('pure_pa', 'pa_kk', 'gamma'):
        qc, kc, r, e = alignment(q, k, solver)
        if strategy == 'gamma': logits = (qq + gamma * (q @ e.transpose(1, 2)) * scale)
        else:
            logits = (q @ (k @ r).transpose(1, 2)) * scale
            if strategy == 'pa_kk': logits = logits + alpha_kk * (kc @ kc.transpose(1, 2)) * scale
    else: raise ValueError(strategy)
    return F.softmax(logits, -1)
