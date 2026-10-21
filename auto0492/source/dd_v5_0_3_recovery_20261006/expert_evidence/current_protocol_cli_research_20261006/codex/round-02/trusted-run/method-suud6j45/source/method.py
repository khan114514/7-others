"""Median-agreement teacher targets with complementary uniform time pairs.

All method tensors and random draws stay on CPU. The frozen trainer supplies
the official DD EMA initialization, loss, optimizer and evaluation protocol.
"""
from dataclasses import dataclass, field


@dataclass
class TeacherConfig:
    num_samples: int = 128
    teacher_steps: int = 10
    guidance: float = 2.0
    filter_rule: str = 'student_agreement_per_label'
    weighting: str = 'uniform'
    reuse: int = 1
    top_k: int = 900
    top_p: float = .95
    ode_nfe: int = 32
    ode_grid: str = 'logSNR'
    proposal_multiplier: int = 3


@dataclass
class MethodSpec:
    name: str = 'dd_median3_paired_uniform128'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = (
        'Each adjacent pair has complementary times low=0.5u and high=1-low '
        'for u~U[0,1), with random assignment to its samples. The mixed '
        'marginal time distribution is uniform on [0,1] and every pair spans '
        'the two half-intervals.'
    )
    train_notes: str = (
        'Official DD EMA initialization under dd-selfproto-v1.3. '
        '128 targets, three proposals per original label, stable lower-median '
        'LPIPS-distance selection, one pass, uniform weights. Time pairs use '
        'complementary half-interval antithetic sampling. Frozen training and '
        'evaluation settings are supplied exclusively by the trainer.'
    )


def fm_time_sampler(batch_shape, device, generator=None):
    import torch
    if torch.device(device).type != 'cpu':
        raise ValueError('Method time sampling requires a CPU device')
    if generator is None or str(generator.device) != 'cpu':
        raise ValueError('Method time sampling requires a CPU generator')
    values = torch.empty(batch_shape, dtype=torch.float32, device='cpu')
    pairs = (values.numel() + 1) // 2
    u = torch.rand(pairs, device='cpu', generator=generator)
    low = .5 * u
    high = 1 - low
    swap = torch.randint(0, 2, (pairs,), device='cpu', generator=generator).bool()
    # Random orientation prevents the first shuffled sample being privileged.
    first = torch.where(swap, high, low)
    second = torch.where(swap, low, high)
    return torch.stack((first, second), dim=1).flatten()[:values.numel()].reshape(values.shape)


def select_teacher_candidates(agreement, multiplier):
    import torch
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError('Proposal multiplier must be a positive integer')
    if agreement.device.type != 'cpu' or agreement.ndim != 1:
        raise ValueError('Agreement must be a one-dimensional CPU tensor')
    if agreement.numel() % multiplier or not torch.isfinite(agreement).all():
        raise ValueError('Agreement must contain complete finite proposal groups')
    groups = agreement.reshape(-1, multiplier)
    rank = (multiplier - 1) // 2
    local = torch.argsort(groups, dim=1, stable=True)[:, rank]
    return torch.arange(len(groups), dtype=torch.long, device='cpu') * multiplier + local


def pair_weights(class_labels, target_ids):
    """Keep relative loss weights uniform; no token or class preference."""
    import torch
    if class_labels.device.type != 'cpu' or target_ids.device.type != 'cpu':
        raise ValueError('Pair weighting requires CPU tensors')
    if class_labels.ndim != 1 or target_ids.ndim != 2 or len(target_ids) != len(class_labels):
        raise ValueError('Pair labels and token rows must have matching batch dimensions')
    if len(class_labels) == 0:
        raise ValueError('A weighted batch must have positive mass')
    return torch.ones(len(class_labels), dtype=torch.float32, device='cpu')


def get_method():
    return MethodSpec()
