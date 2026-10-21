# 实际执行代码与原生结果索引

本页索引完整候选中已经提供的可信运行源码和真实历史记录，不增加科学结果、Job/Trial、模型调用或研究时长。`execution-contract-index.json` 给出源码指纹、行号、原生类型校验和原记录与公开路径版本的独立指纹。

## 评分

`numerical_sources/gpu_fixed_evaluator.py` 是实际数值父进程的评分实现。五 seed 指标来自 `gpu_jail_supervisor.py` 在候选退出后的独立指标计算；父进程使用五 seed 平均 MSE，以 `(B-MSE)/(B-U)` 计算分数。有效结果的公式不使用 clip/min/max。Dev anchor 与独立私有校准 anchor 分开；数值索引给出实际 B/U 值和读取路径。硬门槛失败的 -1 是失败哨兵，不是有效 MSE 的归一化分数。私有校准不是官方平台 Hidden 验收。

## 实际隔离与资源边界

本次混合 profile 的真实 GPU 路径为 `gpu_jail_launcher.py` → `gpu_jail_entry.py`，由 `gpu_jail_supervisor.py` 管理固定输入、输出描述符、CPU/RAM/时间/checkpoint 与独立 reload。entry 关闭外部目录和 socket FD，chroot、清空 groups、降为独占 UID/GID，启用 no-new-privileges 与全线程 seccomp。代码拒绝网络、进程创建、权限/挂载改变和文件创建；CUDA/BLAS 所需线程保留。执行前实际断言探针、执行后父进程读取 target 并校验 shape/finite/nonnegative。具体实现和行号在 JSON 中，不能把教学 bubblewrap 示例当成本次实际数值 provider。完整冻结和既有运行回执仍在包内。

## Harbor 0.23 原生记录

`native_schema/TrialResult.schema.json` 来自本次实际安装的 Harbor 0.23.0。该原生类型没有顶层 `status` 字段，使用 .get 读取它得到 null 不能推断 Trial 未完成。两次真实 NOP 的原生 config/result 已分别完成目标版本类型校验；实际 `finished_at` 存在、`exception_info` 为 null、Verifier 为 separate，实际五 seed fresh/reload 与 reward 附件保留。原记录没有被补写 COMPLETE。原生 Job 示例只做过类型校验，没有执行；不能代替已有真实 Trial，也不要求另造 job.yaml。

## 双构建上下文与资产注入

源码采用 teaching-task-root-v1；`platform_adapter/export_native.py` 显式转换为 Harbor 的 environment/ Agent context 和 tests/ Verifier context。私有资产在导出后由 `inject_private_assets.py` 注入 tests/hidden_assets，并验证 Agent context 完全未变。`native_schema/existing-private-injection-receipt.json` 是已完成 native_task_v7 注入的原回执；索引重新核对包内供应文件、实际原生 Verifier 文件和回执的两个资产指纹，Agent context 不含这些文件。现有 Trial 的 provider 与配置分别参照 existing_config_validation.json。目标平台的官方 task_id、部署和 Hidden 验收仍未取得，不能用专家本地成功冒充。

## GPU active-entry 的准备顺序与现有冻结绑定

新环境由可信管理员在自己的数据盘完成以下步骤，且必须在 numerical-runtime-binding.json 首次创建之前：先运行 build_gpu_jail_runtime.py --root TASK_ROOT --version v2，完成依赖 runtime；确认审定的 preparation/gpu_jail_entry.py 已逐字节安装为 private/gpu-jail-runtime-v2/runner.py；再运行 prepare_gpu_active_entry.py --root TASK_ROOT；执行实际隔离/CUDA 探针通过后，运行 freeze_gpu_binding.py --root TASK_ROOT 一次。probe_gpu_jail.py --root TASK_ROOT 必须经 gpu_lease.py 包裹，指定现场核验授权的 GPU UUID 与全部既有共享锁（逐一 --lock），锁忙就排队；不能直接绕过锁调用 CUDA 探针。这里 TASK_ROOT 是自己的任务根目录；所有源文件均来自 numerical_sources。不得向已冻结环境重新执行准备或冻结。

active-entry 的严格字段只有 entry_sha256，取值为实际 runner.py 字节的 SHA256，schema 与真实非秘密回执见 native_schema/ActiveEntry.schema.json 和 existing-active-entry.json。新准备入口拒绝已有绑定、已有回执、符号链接及 runner/source 不一致。gpu_fixed_evaluator.py 与 gpu_jail_supervisor.py 每次核对 runner 指纹；freeze_gpu_binding.py 把 active-entry 文件本身的指纹写入完整原数值绑定。

active-entry-materialization.json 是本次只读核验：现有 runner、包内 gpu_jail_entry.py、实际 active-entry.entry_sha256 三者相同，active-entry 文件指纹与原 be21... 冻结绑定相同。本次没有写远端 runner、回执或绑定。初始 build 记录的旧 runner 指纹与后续审定 active-entry 不同，不能把初始 builder 当成后续 activation 的记录。新增准备入口只补齐可复现步骤，没有冒称历史上由该新增脚本执行，也不新增 GPU/Trial 结果。
