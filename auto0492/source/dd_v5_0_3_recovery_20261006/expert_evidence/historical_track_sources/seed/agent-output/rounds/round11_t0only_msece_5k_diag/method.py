"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 11）

Round 11: warmstart_official_dd + t0_only + MSE+CE (ce_weight=0.3) + 5000 steps + lr=2e-6

Diagnostic round: test MSE+CE vs MSE-only hypothesis.
Round 10 finding: MSE-only + 5k steps + lr=2e-6 causes severe degradation (5k-FID=62.6).
Hypothesis: MSE-only training drifts shared transformer backbone, degrading untrained
logits head used for coarse levels (split_t=5 in get_final_sequence).
Prediction: MSE+CE (ce_weight=0.3) will preserve quality much better than MSE-only.

方法差异（仅优化面内）：
- 教师样本: 32k, cfg=3.0, 复用 round8 (generation time 72min 计入预算)
- 时间采样: t0_only (100% t=0)
- 暖启动: 官方 VAR-DD-d16 权重
- 超参: lr=2e-6, 5000 steps, ce_weight=0.3, bs=8
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TeacherSampleConfig:
    num_samples: int = 32_000
    guidance: float = 3.0
    top_k: int = 900
    top_p: float = 0.95
    reuse: int = 1
    reuse_source: str = "round08_gentle_ft_cfg3_32k"


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
    name: str = "warmstart_t0only_msece_5k_diag"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_only"
    train_steps: int = 5000
    train_bs: int = 8
    train_lr: float = 2e-6
    loss_type: str = "mse_ce"
    ce_weight: float = 0.3
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "Diagnostic: test MSE+CE (ce=0.3) vs MSE-only to verify logits head drift hypothesis."


def get_method() -> MethodSpec:
    return MethodSpec()
