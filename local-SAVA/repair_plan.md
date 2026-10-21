# SAVA 数据估值 / hierarchical transport valuation：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `8e950e70c9434f14da4ae64890a90074860a1598d1c4f6f543a9cf69d8165979`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA17, QA18, H02, H03。待核项：无。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

SAVA 当前V4静态复核结论：NOT READY。研究优化面、Baseline合理性、Reference正式提升通过；请只修复下列已定位部署/约束问题，保留真实历史证据与有效数值，不为放宽项重跑。

1. environment/ 直属存在 docker-compose.yaml，内容对 main service 设置 capability/seccomp；当前任务文件绑定亦包含该文件。
位置：workspace/harbor_task/environment/docker-compose.yaml:1-8；expert_evidence/runtime_repairs/current_binding.json#task_files_sha256。
影响：当前V4明确禁止 environment/ 和 tests/ 直属四种 Compose 文件，正式部署应每sandbox单容器且不依赖Compose。既有本地NOP成功不能豁免当前部署规则。
最小修改：删除任务context内的Compose文件；用平台认可的单容器部署参数落实服务所需UID、能力和超时清理。不要直接去掉CAP_KILL而使既有killpg退化。
验收材料：完整重打包；无四种禁用文件的成员清单；平台可识别的单容器部署配置、Actor降权/冻结面与超时清理记录。

2. 根task.toml及environment/task.toml包含两处storage_mb和network_mode="no-network"；题面第33行承诺禁网和仅模型传输白名单。
位置：workspace/harbor_task/task.toml:15-19；workspace/harbor_task/task.toml:33-37；workspace/harbor_task/environment/task.toml:15-19；workspace/harbor_task/instruction.md:33。
影响：当前内场不声明storage_mb、禁网或网络白名单。将本地Docker网络隔离作为平台核心前提不符合本批部署合同；不是换卡待补。
最小修改：两份配置同步删除不受支持的磁盘/网络限制字段；题面按平台实际联网与默认工具策略改写，保留候选接口内的文件/网络API禁令和可信评分边界，不以基础设施禁网作为正确性前提。
验收材料：同步后的题面和配置；目标平台可识别资源配置及真实网络条件说明；允许联网环境中的公开Dev与独立Verifier隔离验收材料。

3. 顶层artifacts除/workspace/solution外还包含/opt/public_eval/results；已有NOP artifacts manifest对该目录记录status=ok。该目录由公开评分服务保存报告和历史method快照。
位置：workspace/harbor_task/task.toml:2；workspace/harbor_task/environment/public_eval/dev_service.py:21-25；expert_evidence/harbor_nop/current/artifacts/manifest.json。
影响：当前移交合同仅传最终候选代码及必要模型/配置。公开历史结果目录不是评分候选必需产物，扩大移交面且把历史报告/方法快照带入独立Verifier。未发现它已篡改本次reward的证据。
最小修改：artifacts仅保留/workspace/solution；公开历史报告留在专家/Agent日志归档，不作为Verifier候选移交。同步两份TOML和任何调用覆盖。
验收材料：新完整包配置与移交manifest，证明仅最终method.py、冻结solve.sh进入Verifier所需候选路径；Verifier重新计算并原子写reward。

4. invoke在select就绪后调用阻塞readline；候选仅输出无换行片段时可停在readline而不再检查60s deadline。公开dev_client有7200s等待上限，但dev_service/evaluator没有可信整轮截止与清理；客户端退出不终止后台评测。
位置：workspace/harbor_task/tests/evaluator.py:47-49；workspace/harbor_task/tests/evaluator.py:56-68；workspace/harbor_task/environment/public_eval/evaluator.py:47-68；workspace/harbor_task/environment/public_eval/dev_client.py:12-15；workspace/harbor_task/environment/public_eval/dev_service.py:23。
影响：题面承诺单调用/初始化60s及完整评测7200s。现成正常sleep超时探针只覆盖带完整行的分支，不能补足阻塞管道和整轮超时。此为静态可定位的实现缺口，不声称本次实际触发挂死。
最小修改：使用带剩余deadline的非阻塞/分块管道读取，处理无换行、输出上限、EOF与清理；在可信父侧为整个Dev评价设置独立截止时间，超时杀掉进程组并等待回收，服务返回TIMEOUT且可接受下一请求。
验收材料：受隔离环境中的无换行import输出、无换行call输出和多个单调用均未超时但整轮超时三类负测；有限时返回、无残留worker及下一次Dev成功记录；同步公开/私有evaluator。

非阻断待补：run_summary.task_binding缺官方四项指纹；取得真实发布脚本后补版本/SHA及结果。CPU任务无换卡成绩、耗时、重算要求；已有NOP无需为凑材料重跑。这些不单独阻断。
工具缺口：真实官方0.3.10及独立格式/GPU/指纹工具未获得/未运行；后续保存真实工具输出，不将本报告称为官方通过。
交付：重交包含三职责目录的完整ZIP，保留既有540次原始材料及历史版本；按实际影响做目标环境验收，不修改历史分数或补造记录。
边界：本报告是独立本地V4静态内容复核，不是本次Docker/Verifier/训练/模型实跑，也不代表平台或发布隐私扫描通过。


## 独立审查 B 的具体建议

SAVA 专家返修说明
结论：当前完整ZIP为NOT READY；方法、Baseline、正式成对改善及非训练七小时例外在已提供证据内成立。无需仅因五类放宽事项重跑全部B/R或双模型轨迹。
ZIP SHA256：8e950e70c9434f14da4ae64890a90074860a1598d1c4f6f543a9cf69d8165979

1. 移除当前V4禁用的Compose、磁盘与禁网配置
事实与位置：environment/直属存在docker-compose.yaml；task.toml的两个environment段声明storage_mb及network_mode=no-network；instruction.md第33行依赖禁网/模型白名单。 workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:19；workspace/harbor_task/task.toml:33；workspace/harbor_task/task.toml:37；workspace/harbor_task/instruction.md:33
影响：V4当前内场明确不接受直属Compose以及storage_mb、禁网/网络白名单声明；历史Docker实验不能证明这一部署合同符合当前平台。
最小修改：删除直属Compose文件；移除任务及镜像内副本的storage_mb/network_mode声明，更新题面网络/工具策略为平台实际可用能力。将必要的可信服务启动、UID降权与超时清理放入单sandbox单容器且平台支持的实现，不靠额外容器。
重交验收材料：重交完整ZIP，静态配置与题面一致；目标平台确认服务启动、Actor UID与候选UID、CAP_KILL等必要权限及真实资源落实。保留历史记录而非改标为新平台实测。

2. 将artifacts缩至候选所需资产
事实与位置：task.toml第2行除候选目录还收集公开评分结果目录；dev_service.py会在该目录保存评分JSON和方法历史快照。 workspace/harbor_task/task.toml:2；workspace/harbor_task/environment/public_eval/dev_service.py:21；workspace/harbor_task/environment/public_eval/dev_service.py:24；workspace/harbor_task/tests/verifier.py:15
影响：V4只移交候选代码及必要配置/模型，公开评分结果和历史源码并非Verifier输入。即使现有Verifier不信任这些结果，扩大artifacts仍违反交付边界。
最小修改：从artifacts删除公开结果目录，仅保留题面规定的两文件候选目录；专家过程日志留在expert_evidence，不作为Agent候选移交。
重交验收材料：更新配置与候选移交清单；新版本Verifier只接收必要候选并产生自身评分。既有NOP可保留，版本关联后补。

3. 修复无换行stdout导致的超时失效
事实与位置：两份evaluator.py在select确认可读后调用阻塞readline（第49、60行）；之后还有无deadline的stdout.read（第68行）。候选输出少量不带换行字节并继续计算时，readline可阻塞，后续60秒剩余量检查无法执行。 workspace/harbor_task/tests/evaluator.py:47；workspace/harbor_task/tests/evaluator.py:49；workspace/harbor_task/tests/evaluator.py:58；workspace/harbor_task/tests/evaluator.py:60；workspace/harbor_task/tests/evaluator.py:68；workspace/harbor_task/environment/public_eval/evaluator.py:49
影响：单次初始化/方法60秒约束是题面核心规则；select只保证有字节可读，不保证完整行到达。7200秒外层Verifier时限不等于单次60秒，正常TIMEOUT探针也不能覆盖该分支。
最小修改：使用非阻塞、有上限的字节读取与累计缓冲解析行，所有read/wait均服从同一monotonic deadline；在finally中终止并回收进程组。公开服务端也应执行完整评测deadline，而非只由客户端放弃等待。
重交验收材料：授权隔离环境中的负向记录：导入/调用阶段无换行输出后持续运行、超长单行、DONE后不退出均在指定deadline内终止；无遗留worker，分类保留TIMEOUT/RESOURCE。同步两份评分器。

4. 八字段score恢复V4原始指标口径
事实与位置：两轨每轮均有八字段且全部79份完成报告可匹配，但score保存归一化奖励，例如Codex第1轮-0.374633743283，而对应原始耗时为0.716864103102秒。Codex score_semantics明确写normalized；Seed也使用同口径。 expert_evidence/trajectory_codex.json#score_semantics；expert_evidence/trajectory_codex.json#rounds[0].score；expert_evidence/trajectory_seed.json#rounds[0].score；expert_evidence/native_records/codex/protected_reports/2caed5f0a5cb48978475ac0f0c33f046.json#raw_metric
影响：当前V4要求每轮score为实际Public/Dev原始指标或声明代理指标；当前使用主指标的归一化reward，未声明独立代理指标。可追溯但字段语义不符合当前格式。
最小修改：按现成protected_reports将有完整测量轮的score写为raw_metric；未出分保持null，失败原因和质量失败状态保留。归一化reward与质量门用于选优，可在外部映射/说明保存，八字段不扩列。
重交验收材料：两轨逐轮映射到原报告的raw_metric、时间、状态、方法标识；保留原轨迹原样为历史材料，新轨迹不改实验事实/顺序/有效时间。无需重跑模型或正式B/R。

非阻断待补：真实四项指纹及新版本NOP关联可后补；本CPU题无换卡三类事项。缺真实官方0.3.10/独立格式GPU/指纹工具仅如实保留工具验收缺口，不能伪造官方PASS。
按上述修改后重交workspace、expert_evidence、optimization_evidence完整ZIP，保留旧证据原值与修订说明，不只交补丁。此说明仅本地保存，未发送给专家。
边界：本轮仅静态读取ZIP文本/元数据及复算日志；没有执行提交代码、Docker、训练或模型调用，也没有完成正式平台Hidden验收。

