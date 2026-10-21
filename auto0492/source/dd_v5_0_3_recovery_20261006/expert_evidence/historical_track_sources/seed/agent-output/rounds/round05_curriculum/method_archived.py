"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.0，Round 5）

Round 5: curriculum_hl 时间采样课程（高 t → 低 t 渐进式）

关键洞察：
- 单步推理 (t=0) 需要模型从全噪声预测所有 680 个 token，极难直接学习
- 教师模型是自回归的，在高 t 情形（只预测最高分辨率 token）上表现很好
- 从高 t 开始训练、逐步降低 t，让模型渐进式学习去噪任务
  （课程式学习，明确属于时间采样优化面）

方法差异：
1. 时间采样: curriculum_hl — 五段式课程（高t→中高t→中低t→低t+t0侧重→t0_only）
2. 教师样本: 复用 round3 的 50k (cfg=1.5, top_k=900, top_p=0.95)
3. Loss: mse+ce (ce_weight=0.1, 保持与之前一致)
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TeacherSampleConfig:
    num_samples: int = 50_000
    teacher_steps: int = 10
    guidance: float = 1.5
    top_k: int = 900
    top_p: float = 0.95
    filter_rule: str = "none"
    weighting: str = "uniform"
    reuse: int = 1
    reuse_source: str = "round03_levelweighted"


def fm_time_sampler(batch_shape, device, generator=None,
                    sampler_type="curriculum_hl", num_levels=10,
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
    name: str = "curriculum_hl_timesampling"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_type: str = "curriculum_hl"
    loss_weighting: str = "uniform"
    train_steps: int = 30000
    train_bs: int = 8
    train_lr: float = 1e-4
    loss_type: str = "mse_ce"
    ce_weight: float = 0.1
    ema_decay: float = 0.9999
    train_notes: str = "FM 训练器为自行重实现；curriculum_hl 课程式时间采样：从高t(类AR教师任务)渐进到低t(单步目标)。"


def get_method() -> MethodSpec:
    return MethodSpec()
