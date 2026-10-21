"""Direct, exact implementation of the published operator contract."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    q = torch.einsum("blc,hdc->bhld", x, wq)
    k = torch.einsum("blc,hdc->bhld", x, wk)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(wq.shape[1])
    p0 = logits.softmax(dim=-1)
    base = p0 @ x.unsqueeze(1)
    if alpha == 1.0:
        modified = base
    else:
        perturbed = .5 * (1 + alpha) * logits + .5 * (1 - alpha) * logits.transpose(-1, -2)
        modified = perturbed.softmax(dim=-1) @ x.unsqueeze(1)
    blended = base + beta * (modified - base)
    ratio = (modified.norm(dim=-1, keepdim=True).clamp_min(1e-6)
             / blended.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(.25, 4.)
    y = torch.einsum("bhlc,hdc->bhld", blended * ratio, wv)
    return y.transpose(1, 2).flatten(2).contiguous()
