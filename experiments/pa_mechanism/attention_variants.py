import torch
import torch.nn.functional as F
from pearl.attn import _polar_unitary_newton_schulz


def alignment(q, k, solver='polar', polar_iters=5):
    q, k = q.float(), k.float()
    w = q.norm(dim=-1, keepdim=True)
    w[:, :1] = 0
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    qc, kc = q - (w*q).sum(1, keepdim=True), k - (w*k).sum(1, keepdim=True)
    m = kc.transpose(1, 2) @ qc
    if solver == 'polar':
        r = _polar_unitary_newton_schulz(m, iters=polar_iters)
    elif solver == 'svd':
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        r = u @ vh
    else:
        raise ValueError(solver)
    return qc, kc, r, kc @ r - qc


def attention(q, k, strategy, *, gamma=1., beta=.5, lambda_csa=.5,
              alpha_kk=.08, solver='polar', polar_iters=5, alignment_result=None):
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
    elif strategy == 'csa_weighted':
        return lambda_csa * F.softmax(qq, -1) + (1. - lambda_csa) * F.softmax(kk, -1)
    elif strategy == 'csa_pa_beta':
        _, _, r, _ = alignment_result if alignment_result is not None else alignment(q, k, solver, polar_iters)
        pa = F.softmax((q @ (k @ r).transpose(1, 2)) * scale, -1)
        csa = .5 * (F.softmax(qq, -1) + F.softmax(kk, -1))
        return (1. - beta) * csa + beta * pa
    elif strategy == 'kk_pa_beta':
        _, _, r, _ = alignment_result if alignment_result is not None else alignment(q, k, solver, polar_iters)
        kr = k @ r
        logits = kk + beta * ((q - kr) @ kr.transpose(1, 2)) * scale
    elif strategy == 'qq_entropy_match':
        qc, kc, r, e = alignment(q, k, solver, polar_iters)
        target = F.softmax((q @ (k @ r).transpose(1, 2)) * scale, -1)
        target_entropy = -(target * target.clamp_min(1e-12).log()).sum(-1).mean(-1)
        lo = torch.full_like(target_entropy, .01)
        hi = torch.full_like(target_entropy, 100.)
        for _ in range(16):
            mid = (lo + hi) * .5
            trial = F.softmax(qq * mid[:, None, None], -1)
            ent = -(trial * trial.clamp_min(1e-12).log()).sum(-1).mean(-1)
            lo = torch.where(ent > target_entropy, mid, lo)
            hi = torch.where(ent > target_entropy, hi, mid)
        return F.softmax(qq * ((lo+hi)*.5)[:, None, None], -1)
    elif strategy in ('pure_pa', 'pa_kk', 'gamma', 'residual_kk'):
        qc, kc, r, e = alignment_result if alignment_result is not None else alignment(q, k, solver, polar_iters)
        if strategy in ('gamma', 'residual_kk'):
            logits = qq + gamma * (q @ e.transpose(1, 2)) * scale
            if strategy == 'residual_kk':
                logits = logits + alpha_kk * (kc @ kc.transpose(1, 2)) * scale
        else:
            logits = (q @ (k @ r).transpose(1, 2)) * scale
            if strategy == 'pa_kk': logits = logits + alpha_kk * (kc @ kc.transpose(1, 2)) * scale
    else: raise ValueError(strategy)
    return F.softmax(logits, -1)
