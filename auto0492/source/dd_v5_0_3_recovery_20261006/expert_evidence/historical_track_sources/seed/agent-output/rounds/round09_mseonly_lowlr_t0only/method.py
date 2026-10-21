"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 9）

Round 9: warmstart + t0_only + lr=1e-6 + MSE-only (no CE)

Diagnostic: isolate whether CE loss is causing catastrophic degradation.
Round 8 finding: even gentle FT (1000 steps, lr=1e-6, ce_w=0.01) degrades FID from 9.66 to 28.29 (2k sample diag).
Hypothesis: CE loss (initially 20, near-random plateau at 8.2) produces toxic gradients
that disrupt internal representations, even at low ce_weight.
Test: remove CE entirely, train with MSE only.

方法差异（仅优化面内）：
- 教师样本: 10k, cfg=3.0 (from round 8 teacher)
- 时间采样: t0_only (100% t=0)
- 暖启动: 官方 VAR-DD-d16 权重
- 超参: lr=1e-6, 1000 steps, ce_weight=0.0 (MSE only), bs=8
"""

from dataclasses import dataclass, field
import torch

@dataclass
class TeacherSampleConfig:
    num_samples: int = 10_000
    guidance: float = 3.0
    top_k: int = 900
    top_p: float = 0.95
    reuse: int = 1
    reuse_source: str = "round08_gentle_ft"

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
    name: str = "warmstart_t0only_1e6_mseonly_diag"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_only"
    train_steps: int = 1000
    train_bs: int = 8
    train_lr: float = 1e-6
    loss_type: str = "mse"
    ce_weight: float = 0.0
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "Diagnostic: MSE-only training to test if CE loss causes degradation."

def get_method() -> MethodSpec:
    return MethodSpec()
