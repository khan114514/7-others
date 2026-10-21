# Heima / Efficient Reasoning with Hidden Thinking：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `ce7d24591a81f080954e9910b1e02d77e093260ba205fa3a524fbfb693dcf017`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA15, QA17, H02, H03, H05。待核项：无。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

本次只读复核结论：NOT READY。公开Dev正式B/R五组证据和提升通过；Baseline在已交范围内合理，双轨保守有效时长各≥10h。请处理以下核心问题：

C01：正式运行依赖 Compose、两个额外 broker 容器和 SSH 外部 GPU 后端；task.toml 却声明 gpus=0，不符合每 sandbox 单容器并直接使用自身 GPU 的当前合同。
证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:217；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:39；workspace/host_adapter/host_paths.py#gpu_command；workspace/harbor_task/task.toml:24
修改：删除 environment/ 和 tests/ 直属 Compose 文件及外部 GPU/额外 broker 容器依赖；将可信推理、私有评分及候选隔离部署进各自单容器，用本 sandbox 分配的 GPU。
验收：重交完整三目录包；给出单容器 Dockerfile、直接 GPU 配置、公开 Dev/私有 Verifier 实际调用、候选移交和有限 reward 记录。

C02：任务声明并强制 no-network，配置 storage_mb=512；这些是当前内场明确不声明的约束，且适配器仅接受禁网策略。
证据：workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:23；workspace/harbor_task/task.toml:27；workspace/host_adapter/heima_harbor_environment.py:74
修改：移除 task.toml 的 storage_mb 与 no-network 声明，并按内场实际联网/资源能力同步题面、镜像和约束实现。
验收：更新后的 TOML、题面和约束执行点；真实格式/GPU 脚本检查结果，以及目标平台接受的资源配置。

C03：完整私有协议现有 NOP 的 GPU 活动为64740.009秒，独立 Verifier为66585.789秒；支持入口明确取消外层超时。该实际运行设计尚未满足完整单轮≤7200秒。
证据：expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch,completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:17；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/task.toml:17
修改：将完整评分设计纳入单轮≤7200秒预算并保留可中断的总超时；不要用 NoWallPrivateTrial 取消时限作为正常入口。若无法满足，向平台提交实测与明确例外申请。
验收：同一最终入口的完整训练/推理、重载和评分耗时范围与资源说明；可中断时限配置及完整有限 reward。仅补换卡计时不要求当前自动重跑。

非阻断待补：官方四项指纹、A100向H20/L20迁移影响及目标耗时、NOP文档/默认入口版本关联可后补；不单凭这些要求全量重跑。实际额外容器/远程GPU和取消时限的问题不属于放宽范围。
本机无真实官方0.3.10工具；内容复核不冒称官方新版自检或平台通过。静态材料一致不等于独立复现；本次未运行提交代码、模型、Docker或训练。


## 独立审查 B 的具体建议

Heima核心结论：FAIL / NOT READY。仅静态审查完整ZIP，未运行提交代码。
ZIP SHA256: ce7d24591a81f080954e9910b1e02d77e093260ba205fa3a524fbfb693dcf017
正式G01/G02/G03通过；需处理以下核心问题：

C01：正式执行采用Compose、额外broker容器及包外GPU评分，未满足每sandbox单容器直接使用自身GPU的V4合同。
事实位置：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:53；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:51；workspace/host_adapter/host_paths.py:73；workspace/harbor_task/task.toml:24
最小修改：将公开Dev与可信评分依赖部署到对应sandbox的单一容器，声明真实GPU资源并直接使用本sandbox GPU；移除两份直属Compose和必需的broker/外部计算服务。保留候选受限子进程及独立Verifier边界。
复验材料：完整新ZIP；容器构建/资源配置与调用链显示每sandbox仅一个工作容器；受限候选及Dev/Hidden评分使用本sandbox分配GPU，候选移交与reward链可复核。

C02：task.toml声明storage_mb及三处no-network；provider只接受禁网并以该能力作为必要运行条件，违反当前内场配置口径。
事实位置：workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:23；workspace/harbor_task/task.toml:27；workspace/host_adapter/heima_harbor_environment.py:74
最小修改：删除不受当前内场支持的storage_mb和禁网声明；按实际联网环境实现候选权限与私有数据保护，同步题面、配置及provider。
复验材料：更新的task.toml、题面和单容器provider；明确核心保护由受限候选/可信评分实现，不依赖平台禁网或磁盘限额。

C03：完整私有评分已有实测GPU区间64740.009402秒，Verifier共66585.788957秒；支持入口把总Verifier超时设为None，超过单轮7200秒且失去总时间上限。
事实位置：expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch；expert_evidence/native_harbor/current-nop/scorer-result.json#completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:11；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/instruction.md:42
最小修改：在合规资源下优化完整评分或提交有依据的协议/预算调整供平台确认，并提供可中断的有限Agent、Verifier、单次候选与完整单轮预算；不可用无限总超时掩盖已发生的超限。
复验材料：完整计时起止、GPU实际设备/数量及CPU资源、队列扣除、每阶段成本与限时中断实现；证明完整必要评分满足7200秒，或提供平台对该具体超限的明确例外确认。

非阻断待补（不单独要求返修或自动补跑）：

run_summary缺task_binding.hash_profile/task_content_sha256；未用自制算法伪造官方四项指纹。
位置：expert_evidence/run_summary.json
后补建议：后续取得真实task_fingerprint.py，保留原值并补四项指纹；输出放四项之外。
材料：真实工具身份/哈希、输入范围、当前ZIP对应指纹结果。

已有NOP是特定自定义无总超时调用；文档修改与记录源码版本差异已显式披露，当前普通配置NOP未获同版本验证。
位置：expert_evidence/supported_invocation/README.md:22；expert_evidence/native_harbor/current-nop/result.json；expert_evidence/native_harbor/current-nop/status.json
后补建议：保留既有Trial原件及真实边界；合规部署修复后可补同配置/同版本NOP，不因本待补项自动开跑。
材料：同一次Trial config/result/reward/log和版本关联；无需为仅文档指纹差异重跑。

现有补充Dev验证明确记录A100，未见H20/L20迁移后的实测；历史分数不改标目标卡。
位置：expert_evidence/current-runtime-anchor-source-binding.json#actual_gpu
后补建议：保留真实历史卡型与正式B/R；H20/L20换卡成绩、耗时和锚点影响可后补。
材料：已有设备证明与目标卡估算/后续结果分开；不作为当前核心通过前提。

官方0.3.10及独立脚本未提供，本报告为local-v4-content-review；未做动态验证，不代表平台验收。

