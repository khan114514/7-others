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
    num_samples: int = 8_192           # 限定生成费用，为完整学生训练保留预算
    teacher_steps: int = 10            # 教师 AR 采样步数（VAR 原生为 10 尺度）
    guidance: float = 2.0              # 教师 CFG，沿 VAR 原生尺度线性递增
    filter_rule: str = "none"          # 样本筛选规则标识，如 "none" / "topk:<k>"
    weighting: str = "uniform"         # 样本加权方案
    reuse: int = 256                   # 最多256次完整打乱遍历，预算到期提前停止
    top_k: int = 900                   # 教师合法的概率截断，所有学生保持一致
    top_p: float = 0.95
    ode_nfe: int = 32                  # Gaussian到教师token的确定耦合数值精度
    ode_grid: str = "logSNR"           # 解析诊断中比均匀时间网格更稳定


def fm_time_sampler(batch_shape, device, generator=None):
    """
    流匹配训练时间步采样，返回 t ∈ [0,1] 的张量，形状 batch_shape。
    一半概率均匀覆盖纯噪声尺度区间[0,.1)，另一半覆盖[.1,1)。
    对照上一轮均匀方案，单步所用尺度0的概率由10%提高到50%。
    """
    import torch
    u = torch.rand(batch_shape, device=device, generator=generator)
    return torch.where(u < .5, u * .2, .1 + (u - .5) * 1.8)


@dataclass
class MethodSpec:
    """一个候选方法的完整声明。优化迭代 = 修改本 spec 并重新闭环。"""
    name: str = "coupled8192_cfg2_nfe32_prefix0half"
    teacher: TeacherSampleConfig = field(default_factory=TeacherSampleConfig)
    time_sampler_note: str = "50%概率均匀覆盖尺度0，剩余50%均匀覆盖尺度1..9；映射floor(10*t)，端点1截到尺度9。教师策略与训练工程保持第二轮不变，以隔离时间分布。"
    train_notes: str = "FM 训练器为自行重实现；超参（lr/steps/bs）为配套工程参数，须在证据中记录实际值。"


def get_method() -> MethodSpec:
    return MethodSpec()
