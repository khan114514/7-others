# Heima 独立静态质检

核心结论：**FAIL / NOT READY**。内容复核完成。G01–G03通过；当前V4部署/资源/耗时合同存在三项核心失败。

输入：heima-submission.zip；SHA256 `ce7d24591a81f080954e9910b1e02d77e093260ba205fa3a524fbfb693dcf017`。
审查：local-v4-content-review/1；会话heima_b；可信派发配置gpt-6-astra/high，运行自报GPT-6，精确后端版本未暴露。仅ZIP输入；未引用包内旧QA或其他审查结论。

## 优化面、Baseline与Reference

冻结视觉推理模型的可执行投机草稿策略；降低同批普通AR校准后的平均问题时延比。
固定目标普通自回归；draft返回空列表，经同一策略IPC及目标采样逐token解码。Starter即正式Baseline，B为其全五seed聚合。
冻结分词器得到结构前缀IDS；历史完全匹配时提议最多16个后续token，不匹配则普通AR；目标验证每token并在低裕度时回放AR。

固定模型推理，无训练/微调。正式seed为17/29/43/71/101；两侧共享数据、随机性、质量门和解码预算。

| seed | Baseline | Reference | 改善 | 质量门 |
|---|---:|---:|---:|---|
| 17 | 1.00010338017657 | 0.373456636115598 | 0.626646744060975 | 通过 |
| 29 | 0.999924867484366 | 0.407006021007487 | 0.592918846476879 | 通过 |
| 43 | 1.00090799096054 | 0.412585396817574 | 0.588322594142969 | 通过 |
| 71 | 0.999648023906308 | 0.389729277890587 | 0.609918746015721 | 通过 |
| 101 | 1.00010121076607 | 0.398433284779739 | 0.601667925986333 | 通过 |

均值B=1.00013709465877，R=0.396242123322197；n−1样本标准差σ_B=0.000469290955224339，σ_R=0.0154058499709953。Δ=0.603894971336576，3σ_B=0.00140787286567302，改善=1286.824228σ_B。U=0，归一化R=0.603812191910163，剩余原始空间R−U=0.396242123322197。

来源：optimization_evidence/{baseline_runs,reference_runs}/seed_{17,29,43,71,101}/{result.json,per-sample.jsonl,run.log}。逐一读取全部10份result及对应960行角色记录，重算每seed均值；每seed96唯一ID，tokens/答案一致率1、准确率差0，最低解析率95/96。方法/原始日志/逐样本哈希一致。共享run.log明确是同一个五seed三角色进程，不把共享文件误认复制独立训练。

额外当前部署Dev验证也逐行复算：B=0.9988075356161995，R=0.39498650780291444，σ_B=0.006585376983396364，σ_R=0.013960103155429945，归一化0.6045419225244097、改善91.69118629589349σ_B。来源expert_evidence/runtime_dev_validation/；只作补充，未覆盖正式冻结B。其source_results仍指旧正式目录，建议纠正引用。

## Baseline专项

总评：在已提供材料范围内合理。Starter=正式普通AR；B=全部正式Baseline聚合；额外sanity=同批独立AR校准。未发现直接弱化红旗。

| 项目 | 状态 | 证据与理由 |
|---|---|---|
| B01 | 通过 | 普通自回归是冻结目标上自然的无草稿对照；Starter、正式Baseline和主评分B映射一致；不以年代或低分判弱。 证据：workspace/harbor_task/environment/starter.py；optimization_evidence/baseline_runs/method.py；expert_evidence/专家作业说明文档.md:3 |
| B02 | 通过 | 推理型无训练，empty draft沿实际decoder执行完整普通目标步；全部480条Baseline输出及实际前向记录存在，无未训练模型冒充训练证据。 证据：workspace/backend/policy_decoder_v1.py:97；optimization_evidence/baseline_runs/seed_17/result.json#metrics；optimization_evidence/训练证据说明.md:3 |
| B03 | 通过 | 已有科学对照共享冻结权重、96行、五seed、BF16及输出/前向预算，三角色交错计时。部署资源不合规独立列C01-C03，不反推B/R之间不公平。 证据：workspace/backend/run_policy_score_v2.py:101；optimization_evidence/baseline_runs/seed_17/result.json#protocol；optimization_evidence/reference_runs/seed_17/result.json#protocol |
| B04 | 通过 | Baseline为无草稿算法而非待调差参数；Reference固定结构前缀。双方温度、top-k、随机数、质量门相同，未见仅Baseline失常配置。 证据：workspace/backend/policy_decoder_v1.py:17；workspace/backend/policy_decoder_v1.py:53；optimization_evidence/reference_runs/method.py:5 |
| B05 | 通过 | 17/29/43/71/101完整成对；seed与样本ID生成RNG并实际进入torch采样，同方法非seed代码不变。属于固定模型重复随机评测。 证据：workspace/backend/run_policy_score_v2.py:108；workspace/backend/policy_decoder_v1.py:17；optimization_evidence/comparison_summary.json#seeds |
| B06 | 通过 | 适用sanity即同批独立普通AR calibration，Baseline/Calibration比值接近1；未见低于已交基本锚点的功能异常。 证据：optimization_evidence/baseline_runs/seed_17/per-sample.jsonl；optimization_evidence/comparison_summary.json#paired_results |
| B07 | 通过 | 改动是draft候选结构，token验证及低裕度重放同规则；逐条tokens一致、准确率无损，拒绝/IPC/启动计入耗时。 证据：workspace/backend/run_policy_score_v2.py:116；workspace/backend/policy_decoder_v1.py:149；optimization_evidence/reference_runs/seed_17/per-sample.jsonl |
| B08 | 通过 | 专家说明披露普通AR对照、冻结模型、v1漂移改为同批校准、数学U、训练不适用及原始证据位置。 证据：optimization_evidence/训练证据说明.md；expert_evidence/专家作业说明文档.md |

五维：
- training_sufficiency: not_applicable；冻结推理，不训练或微调；不要求空checkpoint/reload。 证据：optimization_evidence/训练证据说明.md:3
- method_representativeness: pass；普通AR是相同目标的合理自然对照。 证据：optimization_evidence/baseline_runs/method.py；workspace/harbor_task/environment/starter.py
- budget_comparability: pass；B/R同固定模型、数据、seed、解码预算；整体部署超限另列。 证据：optimization_evidence/baseline_runs/seed_17/result.json#protocol；optimization_evidence/reference_runs/seed_17/result.json#protocol
- implementation_correctness: pass；真实AR循环及候选验证路径，480行输出/前向记录与质量门一致。 证据：workspace/backend/policy_decoder_v1.py；workspace/backend/run_policy_score_v2.py
- selection_integrity: pass；五组全集及非seed配置一致，未抽取最差B或最优单seed。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_17/result.json#execution

## 三项内容门

| 项目 | 状态 | 结论 |
|---|---|---|
| G01 | pass | 开放可执行draft(state)支持结构构造、历史复用、接受率适应；Reference真实执行前缀匹配及多token草稿，非纯参数搜索。 |
| G02 | pass | 在已提供材料范围内合理。固定模型随机推理，普通AR作为正式B；成对协议公平，未发现直接弱化或实现故障。 |
| G03 | pass | 正式五组均有效；改善0.603894971336576=1286.824228σ_B，归一化0.603812191910163在[0.15,0.8]，剩余空间充足。 |

## 双轨及格式

| 模型 | 包内轮数 | 完整有效评分 | 保守有效秒 | 小时 | 最佳Dev原始指标 |
|---|---:|---:|---:|---:|---:|
| gpt-6.1-sol | 19 | 5 | 36749.336764097 | 10.208149101 | 0.423135843763386 |
| doubao-seed-2-1-turbo-260628 | 8 | 5 | 41497.018062592 | 11.526949462 | 0.32156455143956 |

19+8轮均恰好八字段，成功分数与10份实际反馈匹配；失败保留null与原因，retained_best按越低越好核对。GPT从后缀复制到结构补全；Seed从模板到动态答案与结构组合，最终唯一best_method哈希与Seed胜出方法相同。保守时段来自clock_proof/公开投影及实际执行引用，逐段核对扣除并求全轨并集；缺日志的GPT3段、Seed2段全部计零，重复重叠不重复计。未执行包内核算脚本，未把原始记录上界或容器在线时长当有效时间。

格式：三大目录、正式成对记录及唯一best_method齐全。额外历史证据/适配代码可保留；直属Compose为核心违规。补充验证source_results归位、陈旧“仍在评分”说明更新为建议。

## 21项检查

| ID | 状态 | 原因和证据 |
|---|---|---|
| QA01 | pass | 八部分齐全，题面未给论文标题/ID/解法来源或Reference实现。 证据：workspace/harbor_task/instruction.md |
| QA02 | pass | Dev固定B与Hidden同批独立普通控制B均按(B-M)/(B-U)连续未裁剪；Hidden的B计算独立于候选值；质量门失败-1。 证据：workspace/harbor_task/instruction.md:13；workspace/backend/run_policy_score_v2.py:136；workspace/backend/run_private_hidden_score_v2.py:186 |
| QA03 | pass | Reference源码、全五seed非空原始日志和逐样本指标可追溯，正式归一化0.603812191910163。推理无新训练checkpoint。 证据：workspace/reference/method.py；optimization_evidence/comparison_summary.json；optimization_evidence/reference_runs/seed_17/run.log |
| QA04 | pass | 随机评价按全部正式值复算n−1样本标准差；改善1286.824228σ_B，超过3σ。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_17/result.json#metrics.primary.value |
| QA05 | fail | 候选大小、权限、质量门和0.5秒IPC有实现；但内场资源配置、额外工作容器/外部GPU和无总时限不满足V4。 证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:53；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:51；workspace/host_adapter/host_paths.py:73；workspace/harbor_task/task.toml:24；workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:23；workspace/harbor_task/task.toml:27；workspace/host_adapter/heima_harbor_environment.py:74；expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch；expert_evidence/native_harbor/current-nop/scorer-result.json#completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:11；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/instruction.md:42 |
| QA06 | pass | test.sh单命令调用grade.py，可信私有结果经有限数值核验写reward；异常保留错误并非伪造0分。 证据：workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/grade.py:111；workspace/harbor_task/tests/grade.py:122 |
| QA07 | pass | 只检查instruction：未暴露Hidden样本、标签或可恢复答案。 证据：workspace/harbor_task/instruction.md |
| QA08 | pass | 只检查instruction：未暴露Reference源码、具体IDS或参考得分；公开sigma阈值不是私有答案。 证据：workspace/harbor_task/instruction.md |
| QA09 | pass | 评分前清理reward两种格式及pending；最终pending.replace原子写入。 证据：workspace/harbor_task/tests/grade.py:57；workspace/harbor_task/tests/grade.py:131 |
| QA10 | pass | 可信源与Hidden指纹校验、Root私有目录、受限chroot降权后导入候选；候选不在GPU评分器进程导入。物理隔离本次未实跑。 证据：workspace/backend/trusted_private_backend_support.py:59；workspace/backend/run_private_hidden_score_v2.py:81；workspace/backend/policy_jail_launcher.py:17；workspace/backend/policy_jail_runner.py:34 |
| QA11 | pass | 目标、场景、指标、修改面、硬边界、提交、迭代与完成标准均有实质内容。 证据：workspace/harbor_task/instruction.md |
| QA12 | pass | ZIP完整成员清点无.git目录或Git对象历史。 证据：@inventory |
| QA13 | pass | 全部五seed双方数据和协议一致；独立复算每seed 96行×两角色指标及质量门，Reference逐seed均改善。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_17/per-sample.jsonl；optimization_evidence/reference_runs/seed_17/per-sample.jsonl |
| QA14 | pass | 可组合历史检索、结构语法和接受历史控制；Reference只覆盖固定前缀，分数未饱和；可探索上下文索引与草稿决策组合。 证据：workspace/harbor_task/instruction.md:36；workspace/reference/method.py:5；expert_evidence/trajectory_seed.json |
| QA15 | fail | 已有完整私有GPU执行64740.009402秒，超过7200秒；这不是仅缺换卡补证。A100→H20/L20补证独立列非阻断。 证据：expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch；expert_evidence/native_harbor/current-nop/scorer-result.json#completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:11；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/instruction.md:42 |
| QA16 | pass | 保守时段逐段扣除/并集复算：GPT36749.336764秒、Seed41497.018063秒，各≥10h；未用7h例外、未将两轨相加。 证据：expert_evidence/clock_proof/clock-source-projections.json；expert_evidence/clock_proof/conservative-clock-result.json；expert_evidence/run_summary.json#independent_clock_verification |
| QA17 | fail | 有separate及真实既有reward链，但Compose/额外broker/外部GPU、内场配置与无限总时限使当前接入不通过；NOP版本待补单列。 证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:53；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:51；workspace/host_adapter/host_paths.py:73；workspace/harbor_task/task.toml:24；workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:23；workspace/harbor_task/task.toml:27；workspace/host_adapter/heima_harbor_environment.py:74；expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch；expert_evidence/native_harbor/current-nop/scorer-result.json#completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:11；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/instruction.md:42 |
| QA18 | pass | GPT19轮/Seed8轮逐轮恰八字段，分别5轮成功；10个成功分数与回执逐项匹配、失败null、最佳标记符合minimize。两条历史来源及方法演进可追溯。 证据：expert_evidence/trajectory_codex.json；expert_evidence/trajectory_seed.json；expert_evidence/trajectory_records/primary-receipt-index.json#round_bindings；expert_evidence/best_method/method.py |
| QA19 | pass | 题面明确随机评价、五seed、Baseline n−1 sigma与3σ有效提升阈值，区分快速/完整评分。 证据：workspace/harbor_task/instruction.md:15；workspace/harbor_task/instruction.md:26 |
| QA20 | pass | 题面有可改范围、实际公开工具命令及禁止其他模型/私有访问、候选网络与写入边界；配置兼容缺陷按QA05另列。 证据：workspace/harbor_task/instruction.md:19；workspace/harbor_task/instruction.md:22；workspace/harbor_task/instruction.md:30；workspace/harbor_task/instruction.md:32 |
| QA21 | manual（非阻断） | 缺run_summary.task_binding两项官方指纹字段，按2026-10-21口径非阻断待补；未臆造官方算法。 证据：expert_evidence/run_summary.json |

## Harbor六项

| ID | 状态 | 原因和证据 |
|---|---|---|
| H01 | pass | 任务根/双Dockerfile/test.sh齐；公开96条数据入Agent，1380条Hidden采用root私有注入且有哈希/调用与既有评分材料，非空占位声明。 证据：workspace/harbor_task/task.toml；workspace/harbor_task/environment/Dockerfile；workspace/harbor_task/tests/Dockerfile；workspace/backend/trusted_private_backend_support.py:91；expert_evidence/current_private_validation/primary_nop/paired-summary.json |
| H02 | fail | 目标Harbor0.23.0及自定义provider有包内schema记录，显式separate；但当前V4不接受storage/禁网声明。 证据：workspace/harbor_task/task.toml；expert_evidence/harbor-0.23.0-native-schema.json；workspace/host_adapter/heima_harbor_environment.py:74 |
| H03 | fail | COPY上下文内来源存在、候选artifacts路径一致；实际运行却必须Compose、额外broker与外部GPU评分，违反单容器本地GPU合同。 证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:53；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:51；workspace/host_adapter/host_paths.py:73；workspace/harbor_task/task.toml:24 |
| H04 | pass | test.sh→grade.py→可信私有结果；绑定候选源码/notes，检查完整1380×5，清理预写reward并原子写有限主分数。 证据：workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/grade.py:111；workspace/harbor_task/tests/grade.py:120；workspace/harbor_task/tests/grade.py:131 |
| H05 | fail | 支持operator显式选择NoWallPrivateTrial，覆盖总Verifier超时为None；已有结果无法证明普通默认时限链路。 证据：workspace/operator/run_recover_private_native.py:173；expert_evidence/supported_invocation/README.md:11；expert_evidence/native_harbor/current-nop/status.json#actual_verifier_timeout_sec |
| H06 | manual（非阻断） | 已有同一NOP Trial的配置、结果、reward与日志可对应，reward=0.00015455815448575576；是自定义无总超时调用且文档后改，非当前普通配置验收。 证据：expert_evidence/native_harbor/current-nop/config.json；expert_evidence/native_harbor/current-nop/result.json；expert_evidence/native_harbor/current-nop/reward.txt；expert_evidence/native_harbor/current-nop/trial.log；expert_evidence/supported_invocation/README.md:22 |

## 可直接给专家的退回说明

当前完整ZIP需返修三项核心问题；研究方法、正式Baseline及Dev改善证据无需因本结论被否定。
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


## 非阻断待补

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


## 工具与运行边界

报告真实schema是local-v4-content-review/1，不是官方0.3.10，也不是旧0.3.2工具release。官方发布包及独立格式/GPU/指纹脚本待补；没有把旧PASS改标。内容复核已经完成。

已有NOP奖励0.00015455815448575576、separate及无异常结果可关联；它是NoWallPrivateTrial，实际GPU区间64740.009402275秒，不能冒充普通10800秒或V4 7200秒验收。当前文档版本关联及普通配置NOP属非阻断待核，已记录超限仍为核心失败。

没有运行/导入提交代码、训练、Docker、Verifier、安装器、pickle或包内核算器；只用可信标准库读取ZIP文本/元数据和自行复算数字。大模型通过外部清单列出，未读取权重。已有稳定性材料只保持其历史范围，未作独立12小时压力测试。

所有证据路径按包根相对记录；未外发、未改包、未读取凭据、未调用模型。未执行整个包的严格发布隐私扫描，本报告不声明主包可公开发布。
