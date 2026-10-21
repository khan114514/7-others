"""Exact retrieval with a joint factor-panel and value contraction."""
import math
import torch


def project(x, weight):
    b, length, channels = x.shape
    heads, dim, _ = weight.shape
    if b > 1:
        # Keep batch and head adjacent so the attention products can flatten
        # those dimensions without copying each projected tensor.
        projected = torch.bmm(
            weight.reshape(1, heads * dim, channels).expand(b, -1, -1),
            x.transpose(1, 2))
        return projected.reshape(b, heads, dim, length).transpose(2, 3)
    if b * length <= 128:
        return torch.mm(weight.reshape(heads * dim, channels),
                        x.reshape(b * length, channels).t()).reshape(
                            heads, dim, b, length).permute(2, 0, 3, 1)
    return torch.mm(x.reshape(b * length, channels),
                    weight.reshape(heads * dim, channels).t()).reshape(
                        b, length, heads, dim).transpose(1, 2)


def project_qkv(x, wq, wk, wv, need_gram=False):
    b, length, channels = x.shape
    heads, dim, _ = wq.shape
    rows = heads * dim
    maps = (wq.reshape(rows, channels), wk.reshape(rows, channels),
            wv.reshape(rows, channels))
    # The Gram matrix uses the same right operand as all three maps.
    # Including its rows in the product also keeps tokens contiguous.
    if b == 1 and need_gram:
        weights = torch.cat((*maps, x.reshape(length, channels)), dim=0)
    else:
        weights = torch.cat(maps, dim=0)
    if b > 1:
        projected = torch.bmm(weights.unsqueeze(0).expand(b, -1, -1),
                              x.transpose(1, 2))
        q, k, values = projected.reshape(b, 3, heads, dim, length).permute(
            1, 0, 2, 4, 3).unbind(0)
        return q, k, values, None
    if length >= 256 and heads >= 20:
        # A wide output product is faster for the large projection.  The
        # appended feature rows still produce the shared token Gram matrix.
        projected = torch.mm(x.reshape(length, channels), weights.t())
        q, k, values = projected[:, :3 * rows].reshape(
            b, length, 3, heads, dim).permute(2, 0, 3, 1, 4).unbind(0)
        gram = projected[:, 3 * rows:].unsqueeze(0) if need_gram else None
        return q, k, values, gram
    projected = torch.mm(weights, x.reshape(length, channels).t())
    q, k, values = projected[:3 * rows].reshape(
        3, b, heads, dim, length).transpose(3, 4).unbind(0)
    gram = projected[3 * rows:].unsqueeze(0) if need_gram else None
    return q, k, values, gram


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


def factored_norms_and_values(coefficients, factor, values):
    b, heads, _, length, _ = coefficients.shape
    rows = 2 * heads * length
    width = min(64, length)

    # Each head shares the norm factor, but has its own projected values.
    # Retrieve both with one product for the first triangular panel.
    joint = torch.cat((factor[:, :, :width].unsqueeze(1).expand(
        b, heads, length, width), values), dim=-1)
    projected = torch.matmul(
        coefficients.reshape(b, heads, 2 * length, length), joint)
    squared = projected[:, :, :, :width].norm(
        dim=-1, keepdim=True).square_().reshape(b, rows, 1)
    y = projected[:, :, length:, width:]

    # Later panels contain only norm coordinates.  Their zero upper rows
    # can be omitted from both coefficient and factor operands.
    flat = coefficients.reshape(b, rows, length)
    for start in range(width, length, 64):
        end = min(start + 64, length)
        if b == 1 and heads >= 20:
            panel = torch.mm(flat[0, :, start:],
                             factor[0, start:, start:end]).unsqueeze(0)
        else:
            panel = torch.bmm(flat[:, :, start:],
                              factor[:, start:, start:end])
        squared.add_(panel.norm(dim=-1, keepdim=True).square_())
    squared.clamp_min_(1e-12)
    norms = squared.reshape(b, heads, 2, length, 1)
    return norms[:, :, 0], norms[:, :, 1], y


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    b, length, channels = x.shape
    heads, dim, _ = wq.shape

    # When the two distributions coincide, the retrieval norm ratio is one.
    # Pack tokens across batches in one projection and keep head dimensions
    # contiguous for the native CPU attention kernel.
    if alpha == 1.0 and dim < channels:
        maps = torch.cat((wq.reshape(heads * dim, channels),
                          wk.reshape(heads * dim, channels),
                          wv.reshape(heads * dim, channels)), dim=0)
        projected = torch.mm(x.reshape(b * length, channels), maps.t())
        q, k, values = projected.reshape(
            b, length, 3, heads, dim).permute(2, 0, 3, 1, 4).unbind(0)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, values)
        return y.transpose(1, 2).flatten(2).contiguous()

    use_gram = 2 * length + dim + channels / heads < 2 * channels
    values = None
    gram = None
    if use_gram or (alpha == 1.0 and dim < channels):
        q, k, values, gram = project_qkv(
            x, wq, wk, wv, use_gram and alpha != 1.0)
    else:
        q = project(x, wq)
        k = project(x, wk)
    logits = torch.matmul(q, k.transpose(-1, -2)).div_(math.sqrt(dim))
    p0 = logits.softmax(dim=-1)

    # Equal probability matrices make both the blend and norm ratio trivial.
    if alpha == 1.0:
        if dim < channels:
            y = torch.matmul(p0, values)
        else:
            base = torch.matmul(p0, x.unsqueeze(1))
            y = torch.matmul(base, wv.transpose(-1, -2))
        return y.transpose(1, 2).flatten(2).contiguous()

    # A + t (A^T - A) equals the prescribed symmetric logit blend.
    # lerp evaluates it in one tensor pass without a scaled-logit buffer.
    perturbed = torch.lerp(logits, logits.transpose(-1, -2),
                           .5 * (1 - alpha))
    p1 = perturbed.softmax(dim=-1)

    # ||p x||^2 = p (x x^T) p^T.  Sharing x x^T avoids
    # constructing two feature-space retrievals for every head.
    factor = None
    if use_gram:
        if gram is None:
            gram = torch.bmm(x, x.transpose(-1, -2))
        if heads >= 4 and length >= 128:
            factor, info = torch.linalg.cholesky_ex(gram, check_errors=False)
            # Rank-deficient features still use the direct operator below.
            use_gram = not bool(info.any())

    if use_gram:
        if factor is None:
            norm1 = gram_norm_squared(p1, gram)
            torch.lerp(p0, p1, beta, out=p0)
            norm_mix = gram_norm_squared(p0, gram)
        else:
            # Keep the two coefficient sets adjacent within each head so a
            # joint factor/value contraction can also produce the output.
            torch.lerp(p0, p1, beta, out=p0)
            coefficients = torch.stack((p1, p0), dim=2)
            norm1, norm_mix, y = factored_norms_and_values(
                coefficients, factor, values)
        ratio = norm1.div_(norm_mix).clamp_(.0625, 16.).sqrt_()
        if factor is None:
            y = torch.matmul(p0, values)
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
