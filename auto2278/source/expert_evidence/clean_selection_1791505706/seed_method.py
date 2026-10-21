"""Optimized multihead retrieval kernel: 3D bmm, q-scaling, Gram norms via 2D matmul, in-place ops."""
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

        # Gram matrix for norm computation: G = x @ x^T, shape (B, L, L)
        G = torch.bmm(x, x.transpose(1, 2))

        # Compute p@G via 3D bmm with reshaped p: (B, H*L, L) @ (B, L, L) -> (B, H*L, L)
        # This avoids tiling G across heads and uses a more efficient BLAS call shape.
        p1G = torch.bmm(p1.view(B, HL, L), G).view(BH, L, L)
        pmG = torch.bmm(p_mix.view(B, HL, L), G).view(BH, L, L)

        # ||p@x||^2 = sum_j (p_ij * (p@G)_ij) along last axis
        norm1 = p1G.mul_(p1).sum(dim=-1, keepdim=True).sqrt_().clamp_min_(1e-6)
        normm = pmG.mul_(p_mix).sum(dim=-1, keepdim=True).sqrt_().clamp_min_(1e-6)
        ratio = norm1.div_(normm).clamp_(0.25, 4.0)

        v = (xr @ wvr.T).view(B, L, H, D).permute(0, 2, 1, 3).contiguous().view(BH, L, D)
        y = torch.bmm(p_mix, v).mul_(ratio)

    return y.view(B, H, L, D).transpose(1, 2).flatten(2).contiguous()
