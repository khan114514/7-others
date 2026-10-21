"""Optimized multihead retrieval kernel: Gram norms, v-space output, cache-aware ordering."""
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

    # Gram matrix x @ x^T for norm computation (small, stays in cache)
    G = x @ x.transpose(-1, -2).unsqueeze(1)

    # Projections for Q and K (used for attention logits)
    wqr = wq.reshape(HD, C)
    wkr = wk.reshape(HD, C)

    q = (xr @ wqr.T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()
    k = (xr @ wkr.T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()

    # Scaled attention logits
    logits = q @ k.transpose(-1, -2)
    logits.mul_(inv_sqrt_d)

    # p0 = softmax(A)
    p0 = logits.view(BHL, L).softmax(dim=-1).view(B, H, L, L)

    if alpha == 1.0:
        # Compute v projection late for cache locality
        v = (xr @ wv.reshape(HD, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()
        y = p0 @ v
    else:
        a_p = 0.5 * (1.0 + alpha)
        a_m = 0.5 * (1.0 - alpha)

        # p1 = softmax(a_p * A + a_m * A^T)
        p1_logits = logits.mul(a_p)
        p1_logits.add_(logits.transpose(-1, -2), alpha=a_m)
        del logits
        p1 = p1_logits.view(BHL, L).softmax(dim=-1).view(B, H, L, L)
        del p1_logits

        # p_mix = (1 - beta) * p0 + beta * p1
        p_mix = torch.lerp(p0, p1, beta)
        del p0

        # Norms via Gram matrix: ||p @ x||^2 = (p @ G * p).sum(-1)
        p1G = p1 @ G
        norm1_sq = p1G.mul_(p1).sum(dim=-1, keepdim=True)
        del p1

        pmG = p_mix @ G
        normm_sq = pmG.mul_(p_mix).sum(dim=-1, keepdim=True)

        norm1 = norm1_sq.sqrt_().clamp_min_(1e-6)
        normm = normm_sq.sqrt_().clamp_min_(1e-6)
        ratio = norm1.div_(normm).clamp_(0.25, 4.0)

        # v projection computed late for cache locality with p_mix @ v
        v = (xr @ wv.reshape(HD, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3).contiguous()
        y = (p_mix @ v).mul_(ratio)

    return y.transpose(1, 2).flatten(2).contiguous()
