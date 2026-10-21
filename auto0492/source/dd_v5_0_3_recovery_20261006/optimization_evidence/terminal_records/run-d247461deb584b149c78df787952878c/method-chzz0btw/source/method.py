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
    name: str = 'calibration_inference_atom_mixture256'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = 'Mixture: probability 0.5 exact t=0, otherwise Uniform U(0.5,1).'
    train_notes: str = 'Released DD EMA initialization; 256 selected genuine teacher pairs, eight proposals per label, one training pass. Allocate half the FM training mass to the exact inference t=0 stage; retain late-prefix supervision for the other half.'


def fm_time_sampler(batch_shape, device, generator=None):
    import torch
    u = torch.rand(batch_shape, device=device, generator=generator)
    return torch.where(u < .5, torch.zeros_like(u), u)


def select_teacher_candidates(agreement, multiplier):
    import torch
    return torch.arange(len(agreement) // multiplier, dtype=torch.long) * multiplier + agreement.reshape(-1, multiplier).argmin(1)


def pair_weights(class_labels, target_ids):
    """Relative teacher sample weights; used only when weighting='custom'."""
    import torch
    return torch.ones(len(class_labels), dtype=torch.float32)


def get_method():
    return MethodSpec()
