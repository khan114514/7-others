"""Exact norm recovery in token space, with a shape-dependent direct path."""
import math
import torch


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    b, length, channels = x.shape
    heads, dim, _ = wq.shape
    q = torch.nn.functional.linear(x, wq.flatten(0, 1)).view(b, length, heads, dim).transpose(1, 2)
    k = torch.nn.functional.linear(x, wk.flatten(0, 1)).view(b, length, heads, dim).transpose(1, 2)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
    p0 = logits.softmax(-1)
    if alpha == 1.0:
        v = torch.nn.functional.linear(x, wv.flatten(0, 1)).view(b, length, heads, dim).transpose(1, 2)
        return (p0 @ v).transpose(1, 2).reshape(b, length, heads * dim).contiguous()
    p1 = (.5 * (1 + alpha) * logits + .5 * (1 - alpha) * logits.transpose(-1, -2)).softmax(-1)
    if length * 4 < channels:
        mix = (1. - beta) * p0 + beta * p1
        gram = (x @ x.transpose(-1, -2)).unsqueeze(1)
        n1 = ((p1 @ gram) * p1).sum(-1, keepdim=True).clamp_min(0.).sqrt().clamp_min(1e-6)
        nm = ((mix @ gram) * mix).sum(-1, keepdim=True).clamp_min(0.).sqrt().clamp_min(1e-6)
        ratio = (n1 / nm).clamp(.25, 4.)
        v = torch.nn.functional.linear(x, wv.flatten(0, 1)).view(b, length, heads, dim).transpose(1, 2)
        y = (mix @ v) * ratio
    else:
        base = p0 @ x.unsqueeze(1)
        modified = p1 @ x.unsqueeze(1)
        mix = base + beta * (modified - base)
        ratio = (modified.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                 / mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(.25, 4.)
        y = (mix * ratio) @ wv.transpose(-1, -2).unsqueeze(0)
    return y.transpose(1, 2).reshape(b, length, heads * dim).contiguous()
