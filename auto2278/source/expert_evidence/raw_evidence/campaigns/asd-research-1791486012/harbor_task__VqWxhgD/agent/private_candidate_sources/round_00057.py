"""Exact retrieval with shared projection and softmax workspaces."""
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
    if b == 1 and need_gram and (length <= 128 or channels >= 1024):
        # Write each independent product into one workspace.  The maps
        # remain in their original storage, so wide projections avoid
        # copying all three weight matrices into a temporary packed map.
        projected = x.new_empty((3 * rows + length, length))
        tokens = x.reshape(length, channels)
        token_columns = tokens.t()
        torch.mm(maps[0], token_columns, out=projected[:rows])
        torch.mm(maps[1], token_columns, out=projected[rows:2 * rows])
        torch.mm(maps[2], token_columns, out=projected[2 * rows:3 * rows])
        torch.mm(tokens, token_columns, out=projected[3 * rows:])
        q, k, values = projected[:3 * rows].reshape(
            3, b, heads, dim, length).transpose(3, 4).unbind(0)
        return q, k, values, projected[3 * rows:].unsqueeze(0)
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


def factored_norms(coefficients, factor):
    b, rows, length = coefficients.shape
    heads = rows // (2 * length)
    if b == 1 and length >= 128 and heads >= 10:
        # Keep the triangular reduction, but use 2-D matmuls to avoid the
        # singleton batch overhead on the long-sequence paths.
        squared = coefficients.new_zeros((rows, 1))
        panel = 64 if heads >= 20 else 48
        for start in range(0, length, panel):
            end = min(start + panel, length)
            projected = torch.mm(coefficients[0, :, start:], factor[0, start:, start:end])
            # The norm kernel fuses the elementwise square and reduction,
            # so only the short vector of panel norms needs to be squared.
            squared.add_(projected.norm(dim=-1, keepdim=True).square_())
        return squared.unsqueeze(0).clamp_min_(1e-12)
    squared = coefficients.new_zeros((b, rows, 1))
    # G = F F^T, hence ||p x||^2 = ||p F||^2.  Only the
    # rows on or below each diagonal block of F contribute.
    panel = 48 if b == 1 and length >= 128 else 64
    for start in range(0, length, panel):
        end = min(start + panel, length)
        projected = torch.bmm(coefficients[:, :, start:],
                              factor[:, start:, start:end])
        squared.add_(projected.norm(dim=-1, keepdim=True).square_())
    return squared.clamp_min_(1e-12)


@torch.no_grad()
def attention(x, wq, wk, wv, alpha, beta):
    b, length, channels = x.shape
    heads, dim, _ = wq.shape

    # When the two distributions coincide, the retrieval norm ratio is one.
    # Pack tokens across batches and write each independent projection into
    # one workspace, avoiding a concatenated copy of the three weight maps.
    # The resulting views keep head dimensions contiguous for native attention.
    if alpha == 1.0 and dim < channels:
        tokens = x.reshape(b * length, channels)
        projected = x.new_empty((3, b * length, heads * dim))
        torch.mm(tokens, wq.reshape(heads * dim, channels).t(),
                 out=projected[0])
        torch.mm(tokens, wk.reshape(heads * dim, channels).t(),
                 out=projected[1])
        torch.mm(tokens, wv.reshape(heads * dim, channels).t(),
                 out=projected[2])
        q, k, values = projected.reshape(
            3, b, length, heads, dim).permute(0, 1, 3, 2, 4).unbind(0)
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
    coefficients = None
    if use_gram and alpha != 1.0 and heads >= 4 and length >= 128:
        # Put both distributions directly in the contraction workspace.
        # The first half holds P1, and the second half becomes the blend.
        coefficients = logits.new_empty((b, 2 * heads, length, length))
        p1 = coefficients[:, :heads]
        p0 = coefficients[:, heads:]
        torch.softmax(logits, dim=-1, out=p0)
    else:
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
    if coefficients is None:
        p1 = perturbed.softmax(dim=-1)
    else:
        torch.softmax(perturbed, dim=-1, out=p1)

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
            # Blend in a single pass, then contract both sets of coefficients
            # with each triangular factor panel in one batched product.
            torch.lerp(p0, p1, beta, out=p0)
            mixed = coefficients[:, heads:]
            squared = factored_norms(
                coefficients.reshape(b, 2 * heads * length, length), factor)
            norm1 = squared[:, :heads * length].reshape(b, heads, length, 1)
            norm_mix = squared[:, heads * length:].reshape(b, heads, length, 1)
            p0 = mixed
        ratio = norm1.div_(norm_mix).clamp_(.0625, 16.).sqrt_()
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
