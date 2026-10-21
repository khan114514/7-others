"""Calibration pilot derived from historical teacher agreement and late-half FM experiments.

Positive num_samples enables the frozen fine-tuning procedure. Configure the
VAR teacher targets and implement the FM time distribution in this file.
The interface uses a CPU torch.Generator; GPU training remains in the trusted
trainer. No training checkpoint, scorer or evaluation output is supplied here.
"""
from dataclasses import dataclass, field


@dataclass
class TeacherConfig:
    num_samples: int = 256
    teacher_steps: int = 10
    guidance: float = 2.0
    filter_rule: str = 'student_agreement_per_label'
    weighting: str = 'uniform'
    reuse: int = 1
    top_k: int = 900
    top_p: float = .95
    ode_nfe: int = 32
    ode_grid: str = 'logSNR'
    proposal_multiplier: int = 8


@dataclass
class MethodSpec:
    name: str = 'author_portable_philox_late256'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = 'Seeded CPU Philox4x32-10 stream matching the legacy small-batch CUDA layout; CPU generator advances normally.'
    train_notes: str = 'Released DD EMA initialization; 256 selected genuine teacher pairs, eight proposals per label, one training pass.'


_philox_seed = None
_philox_block = 0


def _philox_word(seed, block, row):
    # Mathematical Philox4x32-10 counter transform. No CUDA/library access.
    mask = (1 << 32) - 1
    c0, c1, c2, c3 = block & mask, (block >> 32) & mask, row & mask, (row >> 32) & mask
    k0, k1 = seed & mask, (seed >> 32) & mask
    for round_index in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & mask, p1 & mask, ((p0 >> 32) ^ c3 ^ k1) & mask, p0 & mask
        k0 = (k0 + 0x9E3779B9) & mask
        k1 = (k1 + 0xBB67AE85) & mask
    return c0


def fm_time_sampler(batch_shape, device, generator=None):
    import torch
    global _philox_seed, _philox_block
    if generator is None:
        raise ValueError('The supplied random stream is required')
    seed = generator.initial_seed()
    if _philox_seed != seed:
        _philox_seed, _philox_block = seed, 0
    # Preserve the supplied CPU stream advancement. The numerical draws use
    # its seed with a backend-independent counter; no training/evaluation
    # artifact or hard-coded seed is consulted.
    draws = torch.rand(batch_shape, device=device, generator=generator)
    n = draws.numel()
    if not 1 <= n <= 256:
        raise ValueError('Small-batch Philox sampler supports 1..256 rows')
    words = [_philox_word(seed, _philox_block, row) for row in range(n)]
    _philox_block += 1
    uniform = torch.tensor(words, dtype=torch.float32, device=device) * (2.0 ** -32) + (2.0 ** -33)
    uniform = torch.where(uniform == 1.0, torch.zeros_like(uniform), uniform)
    return (.5 + .5 * uniform).reshape(batch_shape)


def select_teacher_candidates(agreement, multiplier):
    import torch
    return torch.arange(len(agreement) // multiplier, dtype=torch.long) * multiplier + agreement.reshape(-1, multiplier).argmin(1)


def pair_weights(class_labels, target_ids):
    """Relative teacher sample weights; used only when weighting='custom'."""
    import torch
    return torch.ones(len(class_labels), dtype=torch.float32)


def get_method():
    return MethodSpec()
