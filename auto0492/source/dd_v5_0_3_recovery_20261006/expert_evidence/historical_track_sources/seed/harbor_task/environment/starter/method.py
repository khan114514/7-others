"""
Distilled Decoding 1 — 优化面入口（Starter 骨架，dd-selfproto-v1.0）

本文件是候选之间【唯一允许有方法差异】的文件。两个可研究变量：

1. 教师样本策略（TeacherSampleConfig）：
   教师（原始 VAR-d16, historical-resources/models/dd1/VAR-teacher/var_d16.pth）
   生成训练用样本的数量、教师侧生成配置、筛选/加权/复用方式。

2. 流匹配训练时间采样（fm_time_sampler）：
   FM 训练时时间步 t ∈ [0,1] 的采样分布。默认 uniform（论文式朴素起点）。

固定面（见 instruction.md §3）：VAE、depth=16、256px、单步推理协议、FID-50k 评测协议。
预算：教师样本生成 + 训练总 GPU 墙钟 ≤ 4.0h（A100-40GB）。

注意：官方未发布 DD 训练代码，训练器为自行重实现，须在文档中如实标注。
"""

from dataclasses import dataclass, field


@dataclass
class TeacherSampleConfig:
    """教师样本策略。候选可修改数量/教师侧配置/筛选加权逻辑。"""
    num_samples: int = 50_000          # 教师样本总数（计入预算）
    teacher_steps: int = 10            # 教师 AR 采样步数（VAR 原生为 10 尺度）
    guidance: float = 1.0              # 教师侧指导强度（若教师实现支持）
    filter_rule: str = "none"          # 样本筛选规则标识，如 "none" / "topk:<k>"
    weighting: str = "uniform"         # 样本加权方案
    reuse: int = 1                     # 每个教师样本在训练中的复用次数


def fm_time_sampler(batch_shape, device, generator=None):
    """
    流匹配训练时间步采样，返回 t ∈ [0,1] 的张量，形状 batch_shape。
    默认：均匀分布 U(0,1)（朴素起点）。
    候选可改为 logit-normal、分段、课程式等任意不依赖评测信息的分布。
    """
    import torch
    return torch.rand(batch_shape, device=device, generator=generator)


@dataclass
class MethodSpec:
    """一个候选方法的完整声明。优化迭代 = 修改本 spec 并重新闭环。"""
    name: str = "starter_uniform"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_note: str = "uniform U(0,1)"
    train_notes: str = "FM 训练器为自行重实现；超参（lr/steps/bs）为配套工程参数，须在证据中记录实际值。"


def get_method() -> MethodSpec:
    return MethodSpec()
