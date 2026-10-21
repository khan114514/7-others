"""Easy-example proposal selection with uniform time sampling.

Hypothesis (joint, round-02 diagnostic vs round-01):
  1. Among multiple teacher proposals per class, samples where the initialized
     student agrees most (lowest LPIPS) provide the cleanest, most stable
     gradient signal.  Hard-example selection (round-01) picks outlier/noisy
     proposals that cause destructive overfitting at small sample budgets.
  2. Uniform time sampling across all 10 FM scales gives balanced gradient
     coverage.  Early-scale bias (round-01) starved mid-to-late refinement
     scales, which are equally important for one-step generation quality.
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
    name: str = 'min_disagree_uniform_time'
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    time_sampler_note: str = 'Uniform time sampling: t ~ U(0,1). Balanced gradient coverage across all 10 FM scales. No early-scale bias.'
    train_notes: str = 'Easy-example proposal selection: pick the teacher proposal with minimum LPIPS distance from the initialized student in each group. 256 samples x 4 proposals = 1024 generated, 256 selected. Uniform weighting.'


def fm_time_sampler(batch_shape, device, generator=None):
    """Uniform time distribution across [0,1].

    Provides balanced gradient coverage across all 10 denoising scales.
    Each scale receives approximately 10% of training mass.
    """
    import torch
    return torch.rand(batch_shape, device=device, generator=generator)


def select_teacher_candidates(agreement, multiplier):
    """Select the teacher proposal with MINIMUM LPIPS distance per label group.

    Implements easy-example / clean-gradient selection: the proposal where the
    initialized student already agrees most with the teacher provides the
    cleanest, most representative learning signal.
    Each group of size `multiplier` corresponds to one original class label;
    we must return exactly one index within each group's range.
    """
    import torch
    groups = len(agreement) // multiplier
    distances = agreement.view(groups, multiplier)
    _, within_group_idx = distances.min(dim=1)
    base = torch.arange(groups, dtype=torch.long) * multiplier
    return base + within_group_idx


def pair_weights(class_labels, target_ids):
    """Relative teacher sample weights; used only when weighting='custom'."""
    import torch
    return torch.ones(len(class_labels), dtype=torch.float32)


def get_method():
    return MethodSpec()
