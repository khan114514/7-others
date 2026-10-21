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
    num_samples: int = 256             # 暖启动短程适配，限制教师数量和一遍更新次数
    teacher_steps: int = 10            # 教师 AR 采样步数（VAR 原生为 10 尺度）
    guidance: float = 2.0              # 教师 CFG，沿 VAR 原生尺度线性递增
    filter_rule: str = "student_agreement_per_label"
    weighting: str = "uniform"         # 样本加权方案
    reuse: int = 1                     # 一次完整打乱遍历，保持固定训练超参
    top_k: int = 900                   # 教师合法的概率截断，所有学生保持一致
    top_p: float = 0.95
    ode_nfe: int = 32                  # Gaussian到教师token的确定耦合数值精度
    ode_grid: str = "logSNR"           # 解析诊断中比均匀时间网格更稳定
    proposal_multiplier: int = 8
    source_manifest: str = ""
    source_budget_record: str = ""


def select_teacher_candidates(agreement, multiplier):
    """Keep one VAR teacher proposal per original label group."""
    import torch
    distances = agreement.reshape(-1, multiplier)
    chosen = distances.argmin(dim=1)
    return torch.arange(len(distances), dtype=torch.long) * multiplier + chosen


def fm_time_sampler(batch_shape, device, generator=None):
    """
    流匹配训练时间步采样，返回 t ∈ [0,1] 的张量，形状 batch_shape。
    均匀覆盖前缀尺度5..9，与round12保持一致。
    """
    import torch
    return 0.5 + 0.5 * torch.rand(batch_shape, device=device, generator=generator)


@dataclass
class MethodSpec:
    """一个候选方法的完整声明。优化迭代 = 修改本 spec 并重新闭环。"""
    name: str = "warmstart_official_dd_teacher256_agreement8x_latehalf"
    protocol_version: str = "dd-selfproto-v1.1"
    initialization: str = "warmstart_official_dd_ema"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_note: str = "uniform U(0.5,1); P(prefix5..9)=0.2 each; frozen floor(10*t) prefix mapping"
    train_notes: str = "自行重实现训练器；官方DD EMA暖启动。为256个固定标签各生成八个VAR教师噪声/目标配对，以初始化学生单步输出到教师目标的LPIPS距离选出一个。训练目标全部来自VAR教师，生成2048候选及筛选计算全额计费；一遍256数据16次更新。固定bs2/accum8/lr1e-4/EMA.99/损失。"
    attribution_note: str = "与round12相比，仅将每标签提案数从2增加到8，保持一致性筛选准则。256标签序列、数量、后半区间时间分布、暖启动、16次更新、训练超参与1小时预算上限保持一致。筛选会偏置噪声及目标联合分布，正式推理仍使用标准Gaussian，不预设泛化改善；暖启动的独立贡献仍未量化。"


def get_method() -> MethodSpec:
    return MethodSpec()
