"""Optimized multihead retrieval kernel."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    D = wq.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(D)

    q = torch.einsum("blc,hdc->bhld", x, wq)
    k = torch.einsum("blc,hdc->bhld", x, wk)
    logits = (q @ k.transpose(-1, -2)) * inv_sqrt_d
    p0 = logits.softmax(dim=-1)
    base = torch.einsum("bhjk,bkc->bhjc", p0, x)

    if alpha == 1.0:
        modified = base
    else:
        a_p = 0.5 * (1 + alpha)
        a_m = 0.5 * (1 - alpha)
        perturbed = a_p * logits + a_m * logits.transpose(-1, -2)
        p1 = perturbed.softmax(dim=-1)
        modified = torch.einsum("bhjk,bkc->bhjc", p1, x)

    blended = base.lerp(modified, beta)

    ratio = (modified.norm(dim=-1, keepdim=True).clamp_min(1e-6)
             / blended.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(.25, 4.)

    y = torch.einsum("bhlc,hdc->bhld", blended * ratio, wv)
    return y.transpose(1, 2).flatten(2).contiguous()
