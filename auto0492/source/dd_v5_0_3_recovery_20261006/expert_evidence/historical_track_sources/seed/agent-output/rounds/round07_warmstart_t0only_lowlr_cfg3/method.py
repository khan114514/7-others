"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 7）

Round 7: warmstart_official_dd + t0_only + very_low_lr (1e-6) + cfg=3.0 teacher

关键发现（Round 6）：全量微调（lr=5e-5, 37k steps）导致灾难性遗忘，FID 从 9.66 退化到 166.83。
Round 7 假设：极轻量微调（1e-6, 5k steps, t0_only）可在保留官方 DD 大部分能力的前提下，
小幅适应 cfg=3.0 教师分布，可能获得 FID 改进。

方法差异：
- 教师样本: 50k, cfg=3.0, 复用 round 6
- 时间采样: t0_only (100% t=0)
- 暖启动: 官方 DD 权重
- 超参: lr=1e-6, 5k steps, ce_weight=0.3
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
    elif sampler_type == "uniform":
        return torch.randint(0, num_levels, (b,), device=device, generator=generator)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

@dataclass
class MethodSpec:
    name: str = "warmstart_t0only_1e6_lr_cfg3"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_only"
    train_steps: int = 5000
    train_bs: int = 8
    train_lr: float = 1e-6
    loss_type: str = "mse_ce"
    ce_weight: float = 0.3
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"

def get_method() -> MethodSpec:
    return MethodSpec()
