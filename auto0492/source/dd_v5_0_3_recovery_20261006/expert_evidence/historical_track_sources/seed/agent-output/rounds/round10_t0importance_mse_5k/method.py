"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 10）

Round 10: warmstart + t0_importance (70% t=0) + MSE-only + 5000 steps + lr=2e-6

Diagnostic: test if mixed t time sampling improves quality vs t0_only.
Hypothesis: training at multiple t levels gives better representations,
improving t=0 generation quality.

方法差异（仅优化面内）：
- 教师样本: 50k, cfg=3.0
- 时间采样: t0_importance (70% t=0, 30% uniform)
- 暖启动: 官方 VAR-DD-d16 权重
- 超参: lr=2e-6, 5000 steps, ce_weight=0.0 (MSE only), bs=8
"""

from dataclasses import dataclass, field
import torch

@dataclass
class TeacherSampleConfig:
    num_samples: int = 50_000
    guidance: float = 3.0
    top_k: int = 900
    top_p: float = 0.95
    reuse: int = 0
    reuse_source: str = ""

def fm_time_sampler(batch_shape, device, generator=None,
                    sampler_type="t0_importance", num_levels=10,
                    step=0, total_steps=1):
    b = batch_shape if isinstance(batch_shape, int) else batch_shape[0]
    if sampler_type == "t0_importance":
        weights = torch.ones(num_levels, device=device) * 0.3 / num_levels
        weights[0] += 0.7
        return torch.multinomial(weights, b, replacement=True, generator=generator)
    elif sampler_type == "uniform":
        return torch.randint(0, num_levels, (b,), device=device, generator=generator)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

@dataclass
class MethodSpec:
    name: str = "warmstart_t0importance_2e6_mse_5k"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_importance"
    train_steps: int = 5000
    train_bs: int = 8
    train_lr: float = 2e-6
    loss_type: str = "mse"
    ce_weight: float = 0.0
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "Diagnostic: t0_importance (70% t=0) + MSE-only + 5k steps @ lr=2e-6."

def get_method() -> MethodSpec:
    return MethodSpec()
