"""Optimized multihead retrieval kernel: 3D bmm, q-scaling, Gram norms, in-place ops."""
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

    wqr = wq.reshape(HD, C)
    wkr = wk.reshape(HD, C)
    wvr = wv.reshape(HD, C)

    q = (xr @ wqr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
    k = (xr @ wkr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)

    q.mul_(inv_sqrt_d)

    logits = torch.bmm(q, k.transpose(1, 2))

    p0 = logits.softmax(dim=-1)

    if alpha == 1.0:
        v = (xr @ wvr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
        y = torch.bmm(p0, v)
    else:
        a_p = 0.5 + 0.5 * alpha
        a_m = 0.5 - 0.5 * alpha
        p1_logits = logits.mul(a_p).add_(logits.transpose(1, 2), alpha=a_m)
        p1 = p1_logits.softmax(dim=-1)

        p_mix = torch.lerp(p0, p1, beta)

        G = torch.bmm(x, x.transpose(1, 2))
        G_tiled = G.unsqueeze(1).expand(B, H, L, L).reshape(BH, L, L)

        p1G = torch.bmm(p1, G_tiled)
        norm1 = p1G.mul_(p1).sum(dim=-1, keepdim=True).sqrt_().clamp_min_(1e-6)
        pmG = torch.bmm(p_mix, G_tiled)
        normm = pmG.mul_(p_mix).sum(dim=-1, keepdim=True).sqrt_().clamp_min_(1e-6)
        ratio = norm1.div_(normm).clamp_(0.25, 4.0)

        v = (xr @ wvr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
        y = torch.bmm(p_mix, v).mul_(ratio)

    return y.view(B, H, L, D).transpose(1, 2).flatten(2).contiguous()
