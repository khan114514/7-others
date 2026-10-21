"""Optimized multihead retrieval kernel: 3D bmm, q-scaling, tiled Gram, in-place ops."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    B, L, C = x.shape
    H, D = wq.shape[0], wq.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(D)
    HD = H * D
    BH = B * H
    xr = x.reshape(B * L, C)

    q = (xr @ wq.reshape(HD, C).T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
    k = (xr @ wk.reshape(HD, C).T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)

    q.mul_(inv_sqrt_d)

    logits = torch.bmm(q, k.transpose(1, 2))

    p0 = logits.softmax(dim=-1)

    if alpha == 1.0:
        v = (xr @ wv.reshape(HD, C).T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
        y = torch.bmm(p0, v)
    else:
        a_p = 0.5 * (1.0 + alpha)
        a_m = 0.5 * (1.0 - alpha)
        p1_logits = logits.mul(a_p)
        p1_logits.add_(logits.transpose(1, 2), alpha=a_m)
        p1 = p1_logits.softmax(dim=-1)

        p_mix = torch.lerp(p0, p1, beta)

        G = x @ x.transpose(-1, -2)
        G_tiled = G.expand(BH, L, L).contiguous()

        p1G = torch.bmm(p1, G_tiled)
        norm1_sq = p1G.mul_(p1).sum(dim=-1, keepdim=True)
        pmG = torch.bmm(p_mix, G_tiled)
        normm_sq = pmG.mul_(p_mix).sum(dim=-1, keepdim=True)
        norm1 = norm1_sq.sqrt_().clamp_min_(1e-6)
        normm = normm_sq.sqrt_().clamp_min_(1e-6)
        ratio = norm1.div_(normm).clamp_(0.25, 4.0)

        v = (xr @ wv.reshape(HD, C).T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
        y = torch.bmm(p_mix, v).mul_(ratio)

    return y.view(B, H, L, D).transpose(1, 2).flatten(2).contiguous()
