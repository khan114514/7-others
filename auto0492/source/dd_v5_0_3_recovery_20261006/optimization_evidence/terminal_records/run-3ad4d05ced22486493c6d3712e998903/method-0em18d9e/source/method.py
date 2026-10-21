"""Start from the published one-step DD model with no further adaptation.

Positive num_samples enables the frozen fine-tuning procedure. Configure the
VAR teacher targets and implement the FM time distribution in this file.
The interface uses a CPU torch.Generator; GPU training remains in the trusted
trainer. No training checkpoint, scorer or evaluation output is supplied here.
"""
from dataclasses import dataclass, field


@dataclass
class TeacherConfig:
    num_samples: int = 16
    teacher_steps: int = 10
    guidance: float = 2.0
    filter_rule: str = 'student_agreement_per_label'
    weighting: str = 'custom'
    reuse: int = 1
    top_k: int = 900
    top_p: float = .95
    ode_nfe: int = 32
    ode_grid: str = 'logSNR'
    proposal_multiplier: int = 2


@dataclass
class MethodSpec:
    name: str = 'weighted_filtered_interface_smoke'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = 'Uniform U(0,1) if adaptation is enabled.'
    train_notes: str = 'Released DD EMA initialization; no adaptation by default.'


def fm_time_sampler(batch_shape, device, generator=None):
    import torch
    return .5 + .5 * torch.rand(batch_shape, device=device, generator=generator)


def select_teacher_candidates(agreement, multiplier):
    import torch
    return torch.arange(len(agreement) // multiplier, dtype=torch.long) * multiplier


def pair_weights(class_labels, target_ids):
    """Relative teacher sample weights; used only when weighting='custom'."""
    import torch
    return 1. + (class_labels % 2).float()


def get_method():
    return MethodSpec()
