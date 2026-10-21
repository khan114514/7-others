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
    source_manifest: str = "historical-codex-models/round15/pairs/manifest.json"
    source_budget_record: str = "historical-codex-workspace/agent-output/rounds/round15/teacher_generation_execution.json"


def select_teacher_candidates(agreement, multiplier):
    """Keep one VAR teacher proposal per original label group."""
    import torch
    distances = agreement.reshape(-1, multiplier)
    chosen = distances.argmin(dim=1)
    return torch.arange(len(distances), dtype=torch.long) * multiplier + chosen


def fm_time_sampler(batch_shape, device, generator=None):
    """
    流匹配训练时间步采样，返回 t ∈ [0,1] 的张量，形状 batch_shape。
    只改变FM时间分布，均匀覆盖原生前缀尺度7..9；教师配对与round15完全一致。
    """
    import torch
    return 0.7 + 0.3 * torch.rand(batch_shape, device=device, generator=generator)


@dataclass
class MethodSpec:
    """一个候选方法的完整声明。优化迭代 = 修改本 spec 并重新闭环。"""
    name: str = "warmstart_official_dd_reuse256_agreement8x_tail3"
    protocol_version: str = "dd-selfproto-v1.1"
    initialization: str = "warmstart_official_dd_ema"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_note: str = "uniform U(0.7,1); P(prefix7..9)=1/3 each; frozen floor(10*t) prefix mapping"
    train_notes: str = "自行重实现训练器；官方DD EMA暖启动。原样复用round15从2048提案筛选的256个VAR教师噪声/目标配对；全部生成与筛选的实测时间全额计入本候选预算。一遍256数据16次更新。固定bs2/accum8/lr1e-4/EMA.99/损失。"
    attribution_note: str = "与round15相比，仅将FM时间分布从U(0.5,1)改成U(0.7,1)；相同256教师配对、8倍筛选策略、官方DD EMA初始化、16次更新、训练超参与1小时执行上限。保持相同执行环境与无损optimizer存储。检验教师后期前缀是否进一步缓解耦合误差；单次训练不保证逐位复现，不预设统计显著，暖启动独立贡献未量化。"


def get_method() -> MethodSpec:
    return MethodSpec()
