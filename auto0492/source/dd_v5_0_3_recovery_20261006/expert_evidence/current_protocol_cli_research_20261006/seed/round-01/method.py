"""Max-disagreement proposal selection with early-scale time bias.

Hypothesis:
  1. Among multiple teacher proposals per class, samples where the initialized
     student disagrees most (highest LPIPS) carry the strongest gradient signal
     because they expose the student's largest gaps relative to the teacher.
     This is hard-example mining within the candidate set.
  2. Biasing FM time samples toward early scales (small t) improves one-step
     generation quality because the inference task — predicting from noise —
     most resembles training at low t, where more tokens are unknown.
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
    proposal_multiplier: int = 4


@dataclass
class MethodSpec:
    name: str = 'max_disagree_early_scale'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = 'Power-law bias toward early scales: t ~ U(0,1)^2.5, concentrating ~50% of mass on scales 0-2 (t<0.3). More relevant to one-step generation which starts from pure noise.'
    train_notes: str = 'Hard-example proposal selection: pick the teacher proposal with maximum LPIPS distance from the initialized student in each group. 256 samples x 4 proposals = 1024 generated, 256 selected. Uniform weighting.'


def fm_time_sampler(batch_shape, device, generator=None):
    """Early-scale-biased time distribution via power transform.

    t = u^(power) with power > 1 skews density toward t=0 (early scales).
    power=2.5 puts ~50% of mass on scales 0-2, while still covering all 10 scales.
    """
    import torch
    u = torch.rand(batch_shape, device=device, generator=generator)
    return torch.pow(u, 2.5)


def select_teacher_candidates(agreement, multiplier):
    """Select the teacher proposal with MAXIMUM LPIPS distance per label group.

    Implements hard-example mining: the proposal where the initialized student
    disagrees most with the teacher provides the strongest learning signal.
    Each group of size `multiplier` corresponds to one original class label;
    we must return exactly one index within each group's range.
    """
    import torch
    groups = len(agreement) // multiplier
    distances = agreement.view(groups, multiplier)
    _, within_group_idx = distances.max(dim=1)
    base = torch.arange(groups, dtype=torch.long) * multiplier
    return base + within_group_idx


def pair_weights(class_labels, target_ids):
    """Relative teacher sample weights; used only when weighting='custom'."""
    import torch
    return torch.ones(len(class_labels), dtype=torch.float32)


def get_method():
    return MethodSpec()
