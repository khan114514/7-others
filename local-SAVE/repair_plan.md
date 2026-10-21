# SAVE / PBMC IFN-beta：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `487dfcc7e7dfebdd261ee7f59eb2098f0a89308557c07a9af6ad65d6a0b6aad4`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA06, QA15, QA17, QA20, H02, H03, H04。待核项：无。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

专家返修说明：本轮当前V4静态内容复核不通过。Baseline与Reference五seed科学对照及两条≥10h轨迹材料在已交范围内可接受，请保留原始A100记录、全部seed/失败/重载和轨迹，不为迁卡或哈希差异自动全量重跑。

需修复的核心问题：
1. 将 Public/Dev 数值执行放入 Agent 自身分配的 GPU 容器；将私有评分及受限候选执行放入独立 Verifier 自身 GPU 容器。移除额外 broker 容器、任务专用 Compose 依赖和 SSH/远端 GPU 队列；同步构建、入口、资源与 artifacts，保留历史真实 A100 记录。
事实：expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60-62/147-153 与 model_relay.py:59-72。该实现将本地GPU改0、启动额外broker并将工作送远端A100。验收：重交单容器 Agent/Verifier 的 Dockerfile/可信执行调用链及 H20/L20 资源配置，给出公开Dev和最终受限评分入口；已有记录需清楚区分历史混合拓扑与新配置。新卡耗时/重算及新NOP依当前放宽口径可后补。

2. 删除 task.toml 中两处 storage_mb 与 no-network/network_mode 限制，更新题面和 provider，使任务在当前内场实际联网条件下仍保持私有数据不可见；不能仅删文字而保留强制禁网代码。
事实：workspace/harbor_task/task.toml:15-18/31-34，provider environment.py:39-46/123-124。当前内场不接收这两类环境条件。验收：新的 task.toml、instruction、provider 与目标平台支持字段一致；提供保留 Hidden/冻结面隔离的源码路径。

3. 可信数值父进程对各seed MSE、聚合raw_metric和归一化score逐级 math.isfinite；写reward前再次校验有限标量，并用 allow_nan=False 序列化。溢出归为明确无效/数值失败，score=null，清理旧reward，不标COMPLETE。
事实：numerical_sources/independent_metrics.py:14-22 仅查输入有限，有限大预测的MSE平方可溢出；gpu_fixed_evaluator.py:86-94 与 tests/grader.py:26-29 不拒绝非有限score即写reward。本次静态定位，未运行攻击/边界候选。验收：提交有限大预测导致平方溢出的边界用例结果、正常候选结果、无reward/失败分类记录；证明 NaN/Inf 不会进入 reward.txt/reward.json。本次仅静态定位，未运行该用例。

非阻断后补：官方四项指纹/task_binding；迁卡成绩/锚点影响；迁卡后耗时；当前精确版本NOP关联；换卡后重算。保留真实历史设备和分数，不能改标新卡实测。真实官方0.3.10格式/GPU/指纹工具验收尚待取得，报告不冒充官方PASS或平台通过。

可见隐私：非官方模型网关端点类型位于 expert_evidence/runtime_integration/portable_runtime_v4/model_transport.py:7，建议外置为可信部署配置并确认外发授权，不回显值。完整附件/二进制扫描由主控另作。

复验应重新提交同一完整三目录ZIP，保存其SHA；本报告的模型/版本与动态边界在report.json中如实记录。


## 独立审查 B 的具体建议

本轮结论：NOT READY。优化面、健康Baseline与全五seed Reference提升通过现有材料静态复核；当前需修正的是环境执行方式与配置。

1. provider环境源码60–62行改gpus=0并要求外置A100，120行加入extra_docker_compose，147–150行创建broker；grader22–23行经队列外发评分。
影响：违反当前V4每sandbox单容器、直接自身GPU且正式执行不依赖额外工作环境要求；不是仅换卡耗时补证。
证据：expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:120；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:147；workspace/harbor_task/tests/grader.py:22
最小修改：把可信数值评分和受限候选进程迁到每个Solver/Verifier sandbox自身的单容器及分配GPU；去掉外部A100分发、额外broker容器和自定义Compose依赖。保留separate、method.py移交、最小输入和可信父侧评分；交付可按environment/、tests/直接构建的最终上下文或平台支持的显式准备链。
验收材料：最终双context Dockerfile/依赖/入口、目标provider配置、容器和GPU归属证据、公开Dev/独立Verifier完整评分及移交/reward记录；历史科学数据保留原设备，不因仅换卡强制全部B/R重跑。

2. task.toml第15/31行storage_mb，第18/34行no-network；instruction第19/24行依赖禁网。
影响：当前内场不声明磁盘限额及禁网/网络白名单，不能以旧版本schema曾接受取代当前契约。
证据：workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:31；workspace/harbor_task/task.toml:34；workspace/harbor_task/instruction.md:24
最小修改：删除task.toml两处storage_mb和no-network声明，按当前内场真实网络/工具策略同步题面及环境；用最小输入、文件权限和受限候选执行实现安全边界，不能依赖平台禁网。
验收材料：原生配置和题面一致的完整包、可用真实格式/GPU前检的原始输出、资源/网络/超时执行说明。

非阻断后补：四项指纹/历史版本关联、换卡成绩影响、目标卡耗时、历史NOP当前版本绑定、换卡后分数/锚点重算。保留真实旧卡证据，不为这些缺口单独自动补跑。
工具缺口：真实官方0.3.10和独立格式/GPU/指纹脚本未取得；以后补真实版本/SHA与原始输出，禁止改旧工具版本冒充。
科学复算：B均值0.0036568199948745547，R均值0.0026325329327709819，σ_B=0.00025142367973954213，改善4.073948258σ，归一化0.280103222。无需主动削弱Baseline。
两轨有效时间36508.691003、36639.615035秒，八字段及40/69成功公开评分核对一致；建议分别解释Seed摘要159轮、154正计时native turn、376事件/69成功评分计数，并显式写时区。此计数建议不单独否定10h时长。
本报告为local-v4-content-review，未独立动态复跑或官方Hidden验收。

