"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 8）

Round 8: warmstart_official_dd + t0_only + very_low_lr (1e-6) + low_ce (0.01) + 1000 steps

Diagnostic round: test whether gentle fine-tuning can preserve baseline quality.
Round 6 finding: full FT (lr=5e-5, 37k steps) causes catastrophic forgetting (9.66 → 166.83).
Hypothesis: very low LR + few steps + t0_only + minimal CE weight = gentle adaptation
that preserves official DD quality while potentially improving slightly.

方法差异（仅优化面内）：
- 教师样本: 50k, cfg=3.0, 复用 round 6 (72min 计入预算)
- 时间采样: t0_only (100% t=0)
- 暖启动: 官方 VAR-DD-d16 权重
- 超参: lr=1e-6, 1000 steps, ce_weight=0.01, bs=8
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TeacherSampleConfig:
    num_samples: int = 50_000
    guidance: float = 3.0
    top_k: int = 900
    top_p: float = 0.95
    reuse: int = 1
    reuse_source: str = "round06_warmstart_t0_cfg3"


def fm_time_sampler(batch_shape, device, generator=None,
                    sampler_type="t0_only", num_levels=10,
                    step=0, total_steps=1):
    b = batch_shape if isinstance(batch_shape, int) else batch_shape[0]
    if sampler_type == "t0_only":
        return torch.zeros(b, dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")


@dataclass
class MethodSpec:
    name: str = "warmstart_t0only_1e6_lr_lowce_diag"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_only"
    train_steps: int = 1000
    train_bs: int = 8
    train_lr: float = 1e-6
    loss_type: str = "mse_ce"
    ce_weight: float = 0.01
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "Diagnostic round: very gentle fine-tuning to test baseline preservation. lr=1e-6 (50x lower than round6), 1000 steps, t0_only, ce_weight=0.01."


def get_method() -> MethodSpec:
    return MethodSpec()
