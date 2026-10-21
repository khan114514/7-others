"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 13）

Round 13: warmstart_official_dd + t0_only + MSE+CE (ce=0.3) + 600 steps + lr=5e-7 + cfg=7.0 teacher

Full candidate: higher-CFG teacher (cfg=7.0) + ultra-gentle fine-tuning.
Hypothesis: cfg=7.0 teacher produces higher-quality samples than cfg=5.0.
Ultra-gentle training (fewer steps, lower LR) minimizes drift from baseline
while potentially absorbing small quality gains from the better teacher.

方法差异（仅优化面内）：
- 教师样本: 50k, cfg=7.0, top_k=900, top_p=0.95, 新生成
- 时间采样: t0_only (100% t=0)
- 暖启动: 官方 VAR-DD-d16 权重
- 超参: lr=5e-7, 600 steps, ce_weight=0.3, bs=8, ema_decay=0.9999
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TeacherSampleConfig:
    num_samples: int = 50_000
    guidance: float = 7.0
    top_k: int = 900
    top_p: float = 0.95
    reuse: int = 0
    reuse_source: str = ""


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
    name: str = "warmstart_cfg7_t0only_ultragentle_msece"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_only"
    train_steps: int = 600
    train_bs: int = 8
    train_lr: float = 5e-7
    loss_type: str = "mse_ce"
    ce_weight: float = 0.3
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "Full candidate: cfg=7.0 teacher + ultra-gentle FT (600 steps, lr=5e-7) to test if higher CFG teacher improves FID over baseline."


def get_method() -> MethodSpec:
    return MethodSpec()
