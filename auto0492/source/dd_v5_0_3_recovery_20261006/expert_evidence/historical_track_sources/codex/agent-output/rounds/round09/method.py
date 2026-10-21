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
    filter_rule: str = "none"          # 样本筛选规则标识，如 "none" / "topk:<k>"
    weighting: str = "uniform"         # 样本加权方案
    reuse: int = 1                     # 一次完整打乱遍历，保持固定训练超参
    top_k: int = 900                   # 教师合法的概率截断，所有学生保持一致
    top_p: float = 0.95
    ode_nfe: int = 32                  # Gaussian到教师token的确定耦合数值精度
    ode_grid: str = "logSNR"           # 解析诊断中比均匀时间网格更稳定
    source_manifest: str = "historical-codex-models/round08/pairs/manifest.json"
    source_budget_record: str = "historical-codex-workspace/agent-output/rounds/round08/teacher_generation_execution.json"


def fm_time_sampler(batch_shape, device, generator=None):
    """
    流匹配训练时间步采样，返回 t ∈ [0,1] 的张量，形状 batch_shape。
    尺度0概率50%，其余九个前缀尺度平分剩余概率。
    """
    import torch
    u = torch.rand(batch_shape, device=device, generator=generator)
    return torch.where(u < .5, u * .2, .1 + (u - .5) * 1.8)


@dataclass
class MethodSpec:
    """一个候选方法的完整声明。优化迭代 = 修改本 spec 并重新闭环。"""
    name: str = "warmstart_official_dd_reuse256_onepass_prefix0half"
    protocol_version: str = "dd-selfproto-v1.1"
    initialization: str = "warmstart_official_dd_ema"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_note: str = "P(prefix0)=0.5; scales1..9 equally share0.5; frozen floor(10*t) mapping"
    train_notes: str = "自行重实现训练器；从官方DD的EMA参数暖启动，与Baseline权重逐参数相等。复用round08的256个教师耦合样本，全额计入原生成时间；一遍数据16次更新。固定bs2/accum8/lr1e-4/EMA.99/损失。"
    attribution_note: str = "与round08相比，仅改变FM时间采样。教师数据逐字节一致，数据顺序、初始化、16次更新、训练超参与1小时预算上限保持一致。教师复用全额收费，实际墙钟可能不同；暖启动的独立贡献仍未量化。"


def get_method() -> MethodSpec:
    return MethodSpec()
