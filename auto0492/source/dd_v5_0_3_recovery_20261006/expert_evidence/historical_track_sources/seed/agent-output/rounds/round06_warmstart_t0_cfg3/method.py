"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，Round 6）

Round 6: warmstart_official_dd + t0_dominant time sampling + cfg=3.0 teacher

方法假设：
1. 暖启动：从官方 DD 权重初始化（v1.1 协议允许），大幅降低训练难度
2. 教师样本：cfg=3.0 更高引导强度，提升教师样本质量（样本 FID 更低）
3. 时间采样：t0_dominant（90% t=0），聚焦单步推理实际任务
4. 更高 CE 权重（0.3）：低分辨率层用离散预测，影响整体结构

方法差异（仅优化面内）：
- 教师样本策略: 50k, cfg=3.0, top_k=900, top_p=0.95
- FM 时间采样: t0_dominant (90% t=0, 10% uniform)
- 暖启动: 官方 VAR-DD-d16 权重（v1.1 协议允许，须披露）
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TeacherSampleConfig:
    num_samples: int = 50_000
    teacher_steps: int = 10
    guidance: float = 3.0
    top_k: int = 900
    top_p: float = 0.95
    filter_rule: str = "none"
    weighting: str = "uniform"
    reuse: int = 0
    reuse_source: str = ""


def fm_time_sampler(batch_shape, device, generator=None,
                    sampler_type="t0_dominant", num_levels=10,
                    step=0, total_steps=1):
    b = batch_shape if isinstance(batch_shape, int) else batch_shape[0]
    
    if sampler_type == "uniform":
        return torch.randint(0, num_levels, (b,), device=device, generator=generator)
    elif sampler_type == "level_weighted":
        token_counts = torch.tensor(
            [680, 679, 675, 666, 650, 625, 589, 525, 425, 256],
            dtype=torch.float32, device=device
        )
        weights = token_counts / token_counts.sum()
        return torch.multinomial(weights, b, replacement=True, generator=generator)
    elif sampler_type == "t0_importance":
        weights = torch.ones(num_levels, device=device) * 0.3 / num_levels
        weights[0] += 0.7
        return torch.multinomial(weights, b, replacement=True, generator=generator)
    elif sampler_type == "t0_dominant":
        weights = torch.ones(num_levels, device=device) * 0.1 / num_levels
        weights[0] += 0.9
        return torch.multinomial(weights, b, replacement=True, generator=generator)
    elif sampler_type == "curriculum_hl":
        progress = step / max(1, total_steps)
        if progress < 0.2:
            lo, hi = 6, 9
            w = torch.zeros(num_levels, device=device)
            w[lo:hi+1] = 1.0 / (hi - lo + 1)
        elif progress < 0.4:
            lo, hi = 3, 7
            w = torch.zeros(num_levels, device=device)
            w[lo:hi+1] = 1.0 / (hi - lo + 1)
        elif progress < 0.6:
            lo, hi = 0, 5
            w = torch.zeros(num_levels, device=device)
            w[lo:hi+1] = 1.0 / (hi - lo + 1)
        elif progress < 0.8:
            w = torch.zeros(num_levels, device=device)
            w[0:3] = 1.0 / 3.0
            w[0] = w[0] + 0.5
            w = w / w.sum()
        else:
            w = torch.zeros(num_levels, device=device)
            w[0] = 1.0
        return torch.multinomial(w, b, replacement=True, generator=generator)
    elif sampler_type == "logit_normal":
        t_cont = torch.sigmoid(torch.randn(b, device=device, generator=generator))
        return (t_cont * num_levels).long().clamp(0, num_levels - 1)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")


@dataclass
class MethodSpec:
    name: str = "warmstart_t0_dominant_cfg3_teacher"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "t0_dominant"
    loss_weighting: str = "uniform"
    train_steps: int = 30000
    train_bs: int = 8
    train_lr: float = 5e-5
    loss_type: str = "mse_ce"
    ce_weight: float = 0.3
    ema_decay: float = 0.9999
    warmstart: bool = True
    warmstart_source: str = "official_VAR-DD-d16"
    train_notes: str = "FM 训练器为自行重实现；v1.1 暖启动自官方 DD 权重；t0_dominant 90% t=0 聚焦单步推理；教师样本 cfg=3.0 更高质量。"


def get_method() -> MethodSpec:
    return MethodSpec()
