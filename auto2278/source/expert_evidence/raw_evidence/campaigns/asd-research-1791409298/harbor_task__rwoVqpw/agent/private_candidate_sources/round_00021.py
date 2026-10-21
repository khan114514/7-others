"""Optimized multihead retrieval kernel: v-space projections, 2D softmax, Gram norms, 4D p@G."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    B, L, C = x.shape
    H, D = wq.shape[0], wq.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(D)
    HD = H * D
    BHL = B * H * L
    xr = x.reshape(B * L, C)

    q = (xr @ wq.reshape(HD, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()
    k = (xr @ wk.reshape(HD, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()
    v = (xr @ wv.reshape(HD, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()

    logits = q @ k.transpose(-1, -2)
    logits.mul_(inv_sqrt_d)

    p0 = logits.view(BHL, L).softmax(dim=-1).view(B, H, L, L)

    if alpha == 1.0:
        y = p0 @ v
    else:
        a_p = 0.5 * (1.0 + alpha)
        a_m = 0.5 * (1.0 - alpha)
        p1_logits = logits.mul(a_p)
        p1_logits.add_(logits.transpose(-1, -2), alpha=a_m)
        p1 = p1_logits.view(BHL, L).softmax(dim=-1).view(B, H, L, L)

        p_mix = torch.lerp(p0, p1, beta)

        G = x @ x.transpose(-1, -2)
        G_exp = G.unsqueeze(1)

        p1G = p1 @ G_exp
        norm1_sq = p1G.mul_(p1).sum(dim=-1, keepdim=True)
        pmG = p_mix @ G_exp
        normm_sq = pmG.mul_(p_mix).sum(dim=-1, keepdim=True)
        norm1 = norm1_sq.sqrt_().clamp_min_(1e-6)
        normm = normm_sq.sqrt_().clamp_min_(1e-6)
        ratio = norm1.div_(normm).clamp_(0.25, 4.0)

        y = (p_mix @ v).mul_(ratio)

    return y.transpose(1, 2).flatten(2).contiguous()

