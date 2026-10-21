"""Optimized multihead retrieval kernel using v-first ordering and Gram-matrix norms."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    B, L, C = x.shape
    H, D = wq.shape[0], wq.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(D)

    xr = x.reshape(B * L, C)
    q = (xr @ wq.reshape(H * D, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3)
    k = (xr @ wk.reshape(H * D, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3)
    v = (xr @ wv.reshape(H * D, C).T).reshape(B, L, H, D).permute(0, 2, 1, 3)

    logits = q @ k.transpose(-1, -2) * inv_sqrt_d
    p0 = logits.softmax(dim=-1)

    if alpha == 1.0:
        y = p0 @ v
    else:
        a_p = 0.5 * (1.0 + alpha)
        a_m = 0.5 * (1.0 - alpha)
        p1 = (a_p * logits + a_m * logits.transpose(-1, -2)).softmax(dim=-1)
        p_mix = torch.lerp(p0, p1, beta)

        # Gram matrix for norm computation (avoids full p@x einsum in C-dim)
        G = x @ x.transpose(-1, -2)  # (B, L, L)
        G_exp = G.unsqueeze(1)       # (B, 1, L, L) for head broadcasting

        # ||u1||^2 per head/position: diag(p1 @ G @ p1^T)
        p1G = p1 @ G_exp
        norm1_sq = (p1G * p1).sum(dim=-1, keepdim=True)

        # ||m||^2 per head/position: diag(p_mix @ G @ p_mix^T)
        pmG = p_mix @ G_exp
        normm_sq = (pmG * p_mix).sum(dim=-1, keepdim=True)

        norm1 = norm1_sq.sqrt().clamp_min(1e-6)
        normm = normm_sq.sqrt().clamp_min(1e-6)
        ratio = (norm1 / normm).clamp(0.25, 4.0)

        y = (p_mix @ v) * ratio

    return y.transpose(1, 2).flatten(2).contiguous()
