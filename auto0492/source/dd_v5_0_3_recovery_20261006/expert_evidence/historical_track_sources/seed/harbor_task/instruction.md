# Distilled Decoding 1 复现与优化任务书（自建协议 v1.1，2026-10-03 12:30 修订）

> 本题无官方 harbor 任务包。本题面由运行方依据论文与优化面描述自建，当前协议版本 `dd-selfproto-v1.1`。
> 任何协议变更必须重新版本化并建立可比起点，新旧分数不得混算。

## 0. 协议变更 v1.0 → v1.1（2026-10-03 12:30 CST，运行方发布，两条轨道同步生效）

1. **暖启动放行**：候选方法**允许**以官方 VAR-DD-d16 蒸馏权重作为训练初始化（v1.0 中该权重仅作 Baseline）。
   - 必须在 policy_name 或 method_summary 中显式标注暖启动（如 `warmstart_official_dd`），不得隐瞒初始化来源。
   - 必须做归因记录：候选相对 Baseline 的 FID 差值、以及在相同预算下暖启动与修改优化面变量各自的贡献说明。
   - 优化面不变：方法差异仍只允许落在「教师样本策略 + FM 训练时间采样」。暖启动只是初始化选择，不是新优化面。
   - Baseline 不变：官方权重实测 FID-50k（本轨道 round 1）仍是唯一 B 锚点；暖启动候选与 Baseline 仍按同协议成对比较。
2. **教师样本跨候选复用**：允许复用先前候选生成的教师样本，但必须披露复用关系，且**被复用样本的生成时间按原计费计入复用候选的预算**（不得零成本继承）。预算上限仍为 4.0 GPU·h/候选。
3. **磁盘保洁条款**：单轮 FID 评分完成后，允许删除该轮 50k 采样原始图片以保磁盘，前提是已保存：FID 原始输出、样本清单 SHA-256、64 图网格证据。评分结果原文件与哈希不得删除。
4. v1.0 下的全部历史轮次保持有效记录；v1.1 起的新候选在证据中注明协议版本。

## 1. 题目

论文：*Distilled Decoding 1: One-step Sampling of Image Auto-regressive Models with Flow Matching*（ICLR 2025，arXiv:2412.17153）。
官方仓库（只读参考，commit 3b95d33）：`historical-resources/autoresearch_tasks/repos/distilled_decoding`。

**官方仅发布了推理代码与蒸馏后权重；训练代码（教师样本生成 + 流匹配训练）未发布（仓库 TODO 明示）。**
本题的所有训练实现均为基于论文方法描述的自行复现，必须在文档与交付中如实标注为「自行重实现（reimplementation）」，
不得宣称复现了官方训练流程。

**优化面（题面给定，不得扩大）**：优化 Distilled Decoding 的**教师样本（teacher samples）**与
**流匹配训练时间采样（FM training time-step sampling）**，在固定训练预算下**降低单步图像 FID**。

## 2. 两阶段结构

### 阶段一：文献复现（必须先完成）

用官方发布的 VAR-DD-d16 蒸馏权重，按官方命令完成 **ImageNet-256 单步采样 FID-50k** 评测：

```
python sample.py --ckpt_path <VAR-DD-d16> --vae_ckpt <VAE> \
  --sample_dir <本轨道工作区>/outputs/repro_d16_step1 --sample_num 50000 \
  --t_list 0 --loss_split_t 5 --depth 16 --bs 64 --fid_statistic <fid_stats 目录>
```

- 论文参考值：VAR-DD-d16 单步 FID = **9.94**（50k，ImageNet-256）。
- 复现目的是验证评测链路可信并冻结评分锚点，不是追求逐位一致；实测值与 9.94 的偏差必须记录并讨论。
- 阶段一完成 = 拿到一次完整 50k FID 结果并保存全部证据。该结果即本轨道 **Baseline（B 锚点）**。
- 阶段一工时计入有效优化时间（属于有证据的方法诊断与真实评分）。

### 阶段二：预算内优化（复现完成后开始）

在优化面内提出并验证改进，使**自行蒸馏训练**得到的模型单步 FID-50k **低于阶段一 Baseline**。

允许的研究变量（且仅允许这些）：
1. **教师样本策略**：教师样本数量、生成配置（如教师采样步数/指导强度等教师侧合法参数）、样本筛选或加权、样本复用方式。
2. **流匹配训练时间采样**：FM 训练时时间步 t 的采样分布/加权方案（如 uniform、logit-normal、分段、课程式等）。

**固定训练预算（冻结）**：每个候选方法从教师样本生成到训练结束的**总 GPU 墙钟时间 ≤ 4.0 GPU·h**
（本实例单卡 A100-40GB 实测墙钟；含教师样本生成时间；不含后续 50k 评测时间）。
预算必须计时留证（开始/结束时间戳、nvidia-smi 记录）。超预算的候选不得作为正式候选评分。

## 3. 固定面（不得修改）

- VAE 权重 `vae_ch160v4096z32.pth`、官方蒸馏权重 `VAR-DD/var-d16.pth`（Baseline 锚点；v1.1 起亦允许作候选训练初始化，见第 0 节）、
  教师权重 `VAR-teacher/var_d16.pth`（仅作教师，不得作为候选模型本身提交）。
- 模型规模 depth=16、分辨率 256、单步推理协议（`t_list 0`、`loss_split_t 5`）。
- 评测协议：50,000 样本、官方 `statistic.pth`、官方 `sample.py` 调用链与本题面指定的 FID 计算方式。
- 随机性控制：正式评分的采样 seed 集必须冻结并在证据中记录；同一协议下重跑应使用相同 seed 集。
- 主指标：**FID-50k（越低越好）**，原始值记入轨迹 score 字段（不做归一化、不换算）。

## 4. 质量门

- 候选必须是「完整训练闭环 + 完整 50k 评分」；诊断性小规模 FID（<50k）可自由用于迭代，
  但只能作为诊断证据，不得写入轨迹 score 字段（该轮 score 记 null 并在 method_summary 说明为诊断轮）。
- 候选模型输出必须有限、可解码、无 NaN；样本网格目检无全黑/全同崩溃。
- 不得利用评测器漏洞（如直接向 statistic 分布注入样本、复用 Baseline 样本充当候选样本）。

## 5. 资源与路径（本机实际映射，优先级最高）

- 只读资产（禁止写入/移动）：
  - 仓库：`historical-resources/autoresearch_tasks/repos/distilled_decoding`
  - 权重与统计：`historical-resources/models/dd1/{VAR-DD/var-d16.pth, VAR/vae_ch160v4096z32.pth, VAR-teacher/var_d16.pth, fid_stats/statistic.pth}`
- Python 环境：`historical-resources/venvs/dd1/bin/python`（torch 2.5.1+cu121、pytorch-fid；缺包可装到本轨道自己的 venv，不得修改共享 venv）。
- 本轨道可写区：`/workspace`（= harbor_task）、`/agent-output`、以及 `historical-resources/work_dd_<轨道名>/`（大文件如教师样本、50k 采样、训练 ckpt 一律放这里， overlay 根盘只有 24G）。
- GPU 与另一轨道共享单卡 A100-40GB：每个进程必须用 `PYTORCH_CUDA_ALLOC_CONF` 或等价手段把显存控制在 20GB 以内；
  训练与 50k 评测前检查 `nvidia-smi` 余量，OOM 记为真实失败并调整 batch 重试。
- 方法代码（solution、训练、采样）禁止使用网络；Agent 进程本身的模型 API 流量不受此限。

## 6. solution 接口

- `solution/method.py`：优化面入口。声明教师样本策略与 FM 时间采样方案的配置与实现，是候选之间唯一允许有方法差异的文件。
- `solution/solve.sh`：一条命令跑完「教师样本生成 → 预算内训练 → 50k 单步采样 → FID 评分」，输出 FID 到 stdout 与结果 json。
- 训练框架代码（FM 训练器、模型定义）可放在 solution 外的本轨道工作区并在 method.py 中引用，但方法差异必须收敛到 method.py 可读的配置/函数上，便于审阅。

## 7. 交付

- 轨迹 `trajectory.jsonl`（八字段）→ 收尾转 `trajectory_<轨道>.json`。
- `best_method/`：最佳候选的 method.py + 训练/采样代码 + 权重路径 + 复验证据。
- 运行摘要：复现结果 vs 9.94、各候选 FID 与预算使用、最佳方法、协议偏差说明。
