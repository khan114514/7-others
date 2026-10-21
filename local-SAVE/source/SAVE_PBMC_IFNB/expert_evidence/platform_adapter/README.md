# 实际 Harbor 0.23 执行路径

任务源使用 teaching-task-root-v1。Agent 构建上下文是任务根；仅复制 environment/public_eval、Starter、公开资产和初始 solution。独立 Verifier 使用 tests/ 上下文，不与 Agent 共用私有评测目录。

原生 Harbor 0.23 的构建上下文分别为 environment/ 和 tests/。先执行 `python export_native.py TASK_ROOT NEW_NATIVE_TASK`，再通过 `inject_private_assets.py` 向新导出的 tests/ 注入可信私有资产及其冻结 anchor。注入器核对两份 HDF5 指纹并确认 Agent 构建上下文完全未变。源码 tests/hidden_assets 是空目录，必须在独立 Verifier 构建前完成此注入；没有注入的导出不构成可评分任务。

本次实际执行采用 `local-docker-cli-remote-cuda-chroot-v1`：原生 Harbor 管理本地 Docker Actor、提交物移交、独立 Docker Verifier 和 reward 收集；本地可信主控通过各角色独占的无网络队列代理模型请求及数值工作。模型凭据只在可信主控内存中读取，不进入 Actor、数值工作进程、容器环境变量或本交付包。

GPU 数值执行位于独立 A100 工作进程。执行器先运行固定 CUDA 探针，再关闭外部目录/套接字描述符、进入只读 chroot、降为独占无权限 UID，并向全部线程应用 seccomp。候选只接收训练资产与当前 context 的控制细胞。刺激目标只在候选退出后由可信评分器读取。固定输出描述符、checkpoint 容量、CPU 亲和性、RAM、每 seed 时限和独立 checkpoint reload 由评分器执行。

这是实际使用的混合执行 profile。当前 A100 宿主没有 Docker socket、SYS_ADMIN 或可用的用户 namespace；本地 Docker 为 CPU 容器，未声明其分配了 NVIDIA GPU。平台若使用其他 provider，须验证等价隔离和执行条件。

同版本真实 NOP 已完成：run_id `222d2e75b7184ce9880b224325f53163`，Harbor Trial `dbf1dd7f-b1bb-47a7-bd2b-b53fcd736199`；独立 Verifier 完成全部五 seed fresh CUDA 执行和五次新进程 reload，得到有限 reward 0，正常结束。该 0 分对应未修改 Starter 在私有校准 context 的完整评测。校准 context 属专家本地验证，不是平台官方 Hidden 验收。

两条精确模型的真实 CLI 工具自检均已通过。正式迭代、最终方法复验及容器稳定性另按真实记录核验；NOP 与连通性测试不计入有效研究时长，也不能替代最终双 AI 审查。
