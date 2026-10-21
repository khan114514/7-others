"""
Distilled Decoding 1 — 优化面入口（dd-selfproto-v1.1，显式官方DD暖启动）

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
    num_samples: int = 64              # 暖启动短程适配，限制教师数量和一遍更新次数
    teacher_steps: int = 10            # 教师 AR 采样步数（VAR 原生为 10 尺度）
    guidance: float = 2.0              # 教师 CFG，沿 VAR 原生尺度线性递增
    filter_rule: str = "none"          # 样本筛选规则标识，如 "none" / "topk:<k>"
    weighting: str = "uniform"         # 样本加权方案
    reuse: int = 1                     # 一次完整打乱遍历，保持固定训练超参
    top_k: int = 900                   # 教师合法的概率截断，所有学生保持一致
    top_p: float = 0.95
    ode_nfe: int = 32                  # Gaussian到教师token的确定耦合数值精度
    ode_grid: str = "logSNR"           # 解析诊断中比均匀时间网格更稳定
    source_manifest: str = ""
    source_budget_record: str = ""


def fm_time_sampler(batch_shape, device, generator=None):
    """
    流匹配训练时间步采样，返回 t ∈ [0,1] 的张量，形状 batch_shape。
    均匀覆盖十个前缀尺度，与round05保持一致。
    """
    import torch
    return torch.rand(batch_shape, device=device, generator=generator)


@dataclass
class MethodSpec:
    """一个候选方法的完整声明。优化迭代 = 修改本 spec 并重新闭环。"""
    name: str = "warmstart_official_dd_teacher64_onepass_uniform"
    protocol_version: str = "dd-selfproto-v1.1"
    initialization: str = "warmstart_official_dd_ema"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_note: str = "uniform U(0,1); floor(10*t) selects native prefix scale 0..9"
    train_notes: str = "自行重实现训练器；从官方DD的EMA参数暖启动，与Baseline权重逐参数相等。独立生成64个教师耦合样本，实际生成时间收费；一遍数据4次更新。固定bs2/accum8/lr1e-4/EMA.99/损失。"
    attribution_note: str = "与round05相比，只改教师样本数量及生成/复用方式：8192复用样本改为64新生成样本。一遍复用上限、uniform时间分布、初始化和训练超参不变，计费上限同为1小时。后续8192冷启动对照独立量化初始化贡献。"


def get_method() -> MethodSpec:
    return MethodSpec()
