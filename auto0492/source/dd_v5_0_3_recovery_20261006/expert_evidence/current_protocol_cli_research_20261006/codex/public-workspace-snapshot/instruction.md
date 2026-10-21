# 单步图像生成：教师样本策略与时间采样

## Goal

在固定的 ImageNet-256 单步图像学生上，设计教师样本策略、训练时间采样、提案筛选或相对样本加权，改善生成质量。主指标是 FID-50k，越低越好。

## Task Setting

初始 `/workspace/solution/method.py` 使用发布模型的 EMA 权重，不做额外适配，是可运行的 Baseline。`num_samples=0` 表示保留该起点；正整数启用同一初始化上的训练。每次评测从固定初始化重新开始。本题使用公开采样代码与预训练权重，训练器为独立重实现。

CPU Agent 与独立 CPU Verifier 通过各自的 Unix socket 请求受控外部 GPU 后端；容器内不提供 GPU。每个研究轨道拥有独立作业范围，只能查阅自己的回执。最终 Verifier 使用独立的评测通道。

## Objective and Metrics

评测重新运行冻结采样器，生成 50000 张 256×256 图，batch 8，单次学生预测、固定混合输出头，使用固定预训练 Inception 与 ImageNet 统计独立计算 FID。公开评测固定 evaluation seed=0；教师生成 seed=1001，训练 seed=2001。正式私有评测使用事先封存的非公开评测 seed，保持教师与训练 seed 不变；其 Baseline 和校准模型均须在同一私有随机流完成实际完整评测。私有通道只在真实校准冻结后启用，公开 seed 的结果不能称为 Hidden 成绩。不能提交权重、预计算图像或自报成绩。评测保留完整 PNG 编码字节的哈希清单和前 64 张预览；完整编码字节在评测时保存在主机内存中。

正式奖励使用不裁剪的 `(B - FID)/(B - U)`。B 来自同协议、同评测随机流的发布模型；U 来自事先选定模型的实际完整复验。作者已允许适度放宽验收：完整成对评测显示实际改善，并对保存模型完成重新加载与完整复验；三倍标准差、独立重训练的逐字节一致性不作为通过门槛。未完成真实校准或未冻结锚点时，正式通道拒绝启动。公开评测回执返回实际原始 FID，不代表最终验收。

## Allowed Scope

只修改 `/workspace/solution/method.py`。允许配置教师样本数量、教师指导强度、概率截断、ODE 求解方式、提案数量、筛选方式、样本复用与加权；允许实现下面三个函数，编写可组合的采样、选择或加权算法。

- `get_method()` 返回带 `name`、`teacher` 的对象；`time_sampler_note` 与 `train_notes` 为可选说明。
- `fm_time_sampler(batch_shape, device, generator)` 返回指定形状的有限张量，值在 `[0,1]`。方法使用 CPU 张量与 CPU `torch.Generator`；训练器检查后将值送到 GPU。时间流与教师、数据打乱、评测随机流分开。
- `filter_rule` 为 `none` 或 `student_agreement_per_label`。后者提供每个原标签的一组教师提案，以及初始化学生与对应教师目标之间的 LPIPS 距离。`select_teacher_candidates(agreement, multiplier)` 返回 CPU LongTensor，每组必须且只能选一个组内索引。
- `weighting` 为 `uniform` 或 `custom`。后者调用 `pair_weights(class_labels, target_ids)`，输入均为 CPU LongTensor，输出每样本一个 `[0,16]` 有限权重。每批权重须有正质量，训练器归一到均值 1 后，加权同一批的 CE 与 LPIPS 损失。

## Hard Boundaries

学生架构、VAE、初始化、优化器、学习率、损失定义、推理头、推理步数与评测器固定。不能修改冻结实现或资产，读取其他作业、Reference、专家证据或私有评测材料。后端方法进程只能读取提交源文件、标准运行库和函数输入，不能访问 GPU 设备或联网。Agent 可使用其公开网络查阅资料；网络权限不扩展后端方法进程的权限。

配置边界如下：

- `num_samples` 为 0 或不超过 100000 的 16 的整数倍；`reuse` 为 1 到 100000 的整数；`teacher_steps` 固定为 10。
- `proposal_multiplier` 为 1 到 16，未筛选时必须为 1。提案总数不超过 131072，这是固定主机内存的容量约束。
- `guidance` 在 `[0,20]`；`top_k` 为 `[0,4096]` 的整数；`top_p` 在 `[0,1]`；有效 `ode_nfe` 为 3 到 128 的整数；`ode_grid` 为 `logSNR` 或 `time_uniform`。

每次候选的教师生成、筛选、方法调用、训练、序列化和训练状态保存合计不超过四个 GPU 墙钟小时。排队与之后的冻结评测不计入训练预算。设备为单卡 A100-40GB 或 H100，主机内存至少 64GB。

训练固定 micro-batch 2、累积 8、AdamW、lr=1e-4、betas=(0.9,0.95)、weight decay=0.05、梯度裁剪 1、EMA=0.99、bf16 前向与 fp32 权重。损失为 LPIPS-alex-v0.1 加 0.1 倍生成 token CE；时间映射为 `floor(10*t)` 截断到 0…9。

## Submission Instructions

最终提交仅为 `/workspace/solution/method.py` 的源码。不能提交检查点、辅助权重、评分文件或预计算输出。保留符合接口的完整实现；最终独立 Verifier 会重新执行教师生成、训练与评测。

## Workflow & Iteration

通过 `bash /workspace/environment/public_eval/solve.sh` 提交当前方法，等待自己作业的可信回执。依据公开原始 FID、成功或失败状态迭代；没有成功出分的作业不能当作有效结果。无效值、缺失输出、预算超时或冻结资产变化会使作业失败。

非作弊的接口或约束违规为无效提交，正式无效分记为 `-1`，不参加有效方案排序。格式、预算、资源、数值与基础设施错误分别保留失败类别，不作为正常质量成绩。单次配置读取上限 60 秒；单次采样、筛选或加权调用上限 15 秒，超时会终止方法进程。

## Completion Criteria

方法须满足全部执行边界，从固定初始化完成重新运行，并由独立评测器产出有限、有效的真实 FID。改善主张需要预先声明协议下的完整成对重复证据；成功烟测、历史报告或尚未冻结的奖励不能替代正式验收。
