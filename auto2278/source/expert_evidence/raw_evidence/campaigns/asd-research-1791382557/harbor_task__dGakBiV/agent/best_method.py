"""Exact retrieval with dense projections and symmetric Gram quadratics."""
import math
import torch


def project(x, weight):
    b, length, channels = x.shape
    heads, dim, _ = weight.shape
    if b * length <= 128:
        return torch.mm(weight.reshape(heads * dim, channels),
                        x.reshape(b * length, channels).t()).reshape(
                            heads, dim, b, length).permute(2, 0, 3, 1)
    return torch.mm(x.reshape(b * length, channels),
                    weight.reshape(heads * dim, channels).t()).reshape(
                        b, length, heads, dim).transpose(1, 2)


def gram_norm_squared(probabilities, gram):
    b, heads, length, _ = probabilities.shape
    coefficients = probabilities.reshape(b, heads * length, length)
    if length <= 64:
        return torch.bmm(coefficients, gram).mul_(coefficients).sum(
            dim=-1, keepdim=True).reshape(b, heads, length, 1).clamp_min_(1e-12)

    # The two off-diagonal Gram blocks are transposes, so their
    # contributions to each scalar quadratic form are identical.
    squared = coefficients.new_zeros((b, heads * length, 1))
    for start in range(0, length, 64):
        end = min(start + 64, length)
        part = coefficients[:, :, start:end]
        diagonal = torch.bmm(part, gram[:, start:end, start:end])
        squared.add_(diagonal.mul_(part).sum(dim=-1, keepdim=True))
        if end < length:
            cross = torch.bmm(part, gram[:, start:end, end:])
            squared.add_(cross.mul_(coefficients[:, :, end:]).sum(
                dim=-1, keepdim=True), alpha=2.0)
    return squared.reshape(b, heads, length, 1).clamp_min_(1e-12)


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    b, length, channels = x.shape
    heads, dim, _ = wq.shape
    q = project(x, wq)
    k = project(x, wk)
    logits = torch.matmul(q, k.transpose(-1, -2)).div_(math.sqrt(dim))
    p0 = logits.softmax(dim=-1)

    # Equal probability matrices make both the blend and norm ratio trivial.
    if alpha == 1.0:
        if dim < channels:
            y = torch.matmul(p0, project(x, wv))
        else:
            base = torch.matmul(p0, x.unsqueeze(1))
            y = torch.matmul(base, wv.transpose(-1, -2))
        return y.transpose(1, 2).flatten(2).contiguous()

    perturbed = logits.mul(.5 * (1 + alpha))
    perturbed.add_(logits.transpose(-1, -2), alpha=.5 * (1 - alpha))
    p1 = perturbed.softmax(dim=-1)

    # ||p x||^2 = p (x x^T) p^T.  Sharing x x^T avoids
    # constructing two feature-space retrievals for every head.
    if 2 * length + dim + channels / heads < 2 * channels:
        gram = torch.bmm(x, x.transpose(-1, -2))
        norm1 = gram_norm_squared(p1, gram)
        p1.sub_(p0)
        p0.add_(p1, alpha=beta)
        norm_mix = gram_norm_squared(p0, gram)
        ratio = norm1.div_(norm_mix).clamp_(.0625, 16.).sqrt_()
        y = torch.matmul(p0, project(x, wv))
        y.mul_(ratio)
    else:
        base = torch.matmul(p0, x.unsqueeze(1))
        modified = torch.matmul(p1, x.unsqueeze(1))
        ratio = modified.norm(dim=-1, keepdim=True).clamp_min_(1e-6)
        modified.sub_(base)
        base.add_(modified, alpha=beta)
        ratio.div_(base.norm(dim=-1, keepdim=True).clamp_min_(1e-6))
        ratio.clamp_(.25, 4.)
        y = torch.matmul(base.mul_(ratio), wv.transpose(-1, -2))
    return y.transpose(1, 2).flatten(2).contiguous()
