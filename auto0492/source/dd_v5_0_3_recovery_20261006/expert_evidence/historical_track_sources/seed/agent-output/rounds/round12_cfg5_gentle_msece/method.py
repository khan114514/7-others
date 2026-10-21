"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 12）

Round 12: warmstart_official_dd + t0_only + MSE+CE (ce=0.3) + 1000 steps + lr=1e-6 + cfg=5.0 teacher

Full candidate: higher quality teacher (cfg=5.0) + very gentle fine-tuning.
Hypothesis: teacher samples with higher guidance scale produce higher-quality outputs.
Gentle fine-tuning on these higher-quality samples nudges the DD model toward
better sample quality, potentially improving FID over baseline.

方法差异（仅优化面内）：
- 教师样本: 50k, cfg=5.0, top_k=900, top_p=0.95, 新生成
- 时间采样: t0_only (100% t=0)
- 暖启动: 官方 VAR-DD-d16 权重
- 超参: lr=1e-6, 1000 steps, ce_weight=0.3, bs=8
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TeacherSampleConfig:
    num_samples: int = 50_000
    guidance: float = 5.0
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
    name: str = "warmstart_cfg5_t0only_gentle_msece"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_only"
    train_steps: int = 1000
    train_bs: int = 8
    train_lr: float = 1e-6
    loss_type: str = "mse_ce"
    ce_weight: float = 0.3
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "Full candidate: cfg=5.0 teacher + very gentle FT (1k steps, lr=1e-6) to test quality improvement hypothesis."


def get_method() -> MethodSpec:
    return MethodSpec()
