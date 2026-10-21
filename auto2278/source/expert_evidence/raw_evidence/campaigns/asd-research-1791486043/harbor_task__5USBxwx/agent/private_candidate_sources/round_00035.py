"""Optimized multihead retrieval kernel: Gram norms, G-first ordering, in-place ops."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    B, L, C = x.shape
    H, D = wq.shape[0], wq.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(D)
    HD = H * D
    BH = B * H
    HL = H * L
    xr = x.reshape(B * L, C)

    wqr = wq.reshape(HD, C)
    wkr = wk.reshape(HD, C)
    wvr = wv.reshape(HD, C)

    # Compute Gram matrix first (small footprint, stays in cache for later use)
    G = torch.bmm(x, x.transpose(1, 2))

    # Q, K projections in 2D then reshape to (BH, L, D)
    q = (xr @ wqr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
    k = (xr @ wkr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
    q.mul_(inv_sqrt_d)

    # logits = q @ k^T
    logits = torch.bmm(q, k.transpose(1, 2))
    del q, k

    # p0 = softmax(logits)
    p0 = logits.softmax(dim=-1)

    if alpha == 1.0:
        # Fast path: p1 == p0, ratio == 1, output = p0 @ v
        del G, logits
        v = (xr @ wvr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
        y = torch.bmm(p0, v)
        del p0, v
        return y.view(B, H, L, D).transpose(1, 2).flatten(2).contiguous()

    # p1 = softmax(a_p * logits + a_m * logits^T)
    a_p = 0.5 + 0.5 * alpha
    a_m = 0.5 - 0.5 * alpha
    p1_logits = logits.mul(a_p).add_(logits.transpose(1, 2), alpha=a_m)
    del logits
    p1 = p1_logits.softmax(dim=-1)
    del p1_logits

    # p_mix = (1 - beta) * p0 + beta * p1
    p_mix = torch.lerp(p0, p1, beta)
    del p0

    # Norms via Gram: ||p @ x||^2 = (p * (p @ G)).sum(-1)
    # G was computed early; reuse it here
    p1G = torch.bmm(p1.view(B, HL, L), G).view(BH, L, L)
    pmG = torch.bmm(p_mix.view(B, HL, L), G).view(BH, L, L)
    del G

    norm1 = p1G.mul_(p1).sum(dim=-1, keepdim=True).sqrt_().clamp_min_(1e-6)
    del p1, p1G
    normm = pmG.mul_(p_mix).sum(dim=-1, keepdim=True).sqrt_().clamp_min_(1e-6)
    del pmG

    ratio = norm1.div_(normm).clamp_(0.25, 4.0)
    del norm1, normm

    # V projection + output
    v = (xr @ wvr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
    y = torch.bmm(p_mix, v).mul_(ratio)
    del p_mix, v, ratio

    return y.view(B, H, L, D).transpose(1, 2).flatten(2).contiguous()
