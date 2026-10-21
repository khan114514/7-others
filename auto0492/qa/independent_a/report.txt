# DD 当前 V4 独立静态复核（A）

结论：**FAIL / NOT READY**。复核完成度：已完成安全静态内容审查；目标运行未执行。ZIP SHA-256：`4f3ed4b5164bfd052d9e699da6b47726450fe1cd0dac3af65896732295ebb096`。

configured_model=gpt-6.1-sol；runtime_reported_model=not_exposed；runtime_version=not_exposed；session_id=dd_a。本报告使用local-v4-content-review/schema4，不是官方0.3.10输出。

## 优化面及实际边界

固定发布 DD 单步 ImageNet-256 学生、初始化、损失、优化器与评测；方法可编程改变 FM 时间分布、同标签教师提案选择及样本相对权重。实际 guard 传递张量和配置，而非只允许常量。

FID-50000 minimize；有限数值、完整50000样本、合法候选及冻结面；V4 Public/Dev Reference归一化[.15,.8]。当前题面另写四小时外部A100/H100及撤销3σ，与本次V4冲突。

实际链：Agent改method.py → Public client/role socket → 外部可信supervisor → CPU guard/RPC → 教师生成与固定学生适配 → 冻结FID → client原子reward。公开和最终私有通道分离，候选函数不是直接导入高权限grader。开放接口的研究价值成立；运行部署模式不符合当前内场。

## Baseline 来源、训练与公平预算

正式B=未改Starter的发布预训练DD EMA，num_samples=0，不增加适配；其已有训练权重是合法固定起点。公开和私有评测复用同一完成训练的发布模型，不声称多次训练。

256个教师目标，每标签8个提案，以初始化学生到教师目标的LPIPS最小值选择；从同一发布EMA进行一次256样本适配（16次AdamW更新），便携Philox4x32-10时间采样限制在后半区间[.5,1]；均匀样本权重。

Baseline总评：在已提供材料范围内合理。不能把零新增适配误判成未训练模型或无依据削弱。既有对照共用发布EMA、评价数据/流程与A100上限，适配量是允许变量；没有发现B被异常截短或只用更差seed的直接证据。

| 五维事实 | 核对结果 |
|---|---|
| training_sufficiency | B是完成训练的发布EMA而非随机初始化；新增updates=0为合法零适配，不能因不新增训练判不足。R的16updates有逐更新曲线和正常停止原因。 |
| implementation_correctness | 更新循环包含loss.backward/optimizer.step/EMA，固定推理加载保存checkpoint；原始50k日志可对应result。只读不认证权重数值或独立重现。 |
| method_representativeness | 有可追溯官方发布模型作为朴素固定起点，未见任务内更强常用锚点足以证明此对照缺乏代表性。 |
| budget_comparability | 既有B/R同卡、同评价流、同资源上限；适配量可优化。迁移H20/L20影响延期，外部计算模式本身仍需修复。 |
| selection_integrity | 公开正式一对和私有正式一对完整披露，额外seed1/2与失败保留；未见挑最差B或复制checkpoint冒充训练。统计充分性未成立。 |

| 项目 | 状态 | 事实与路径 |
|---|---|---|
| B01 | pass | 发布DD EMA为任务的可执行预训练起点及正式B；有固定版本资产URL/大小/SHA，不因年代或无新增适配判弱基线。 workspace/harbor_task/environment/assets.json；workspace/harbor_task/environment/starter/method.py:11-31；optimization_evidence/训练证据说明.md:5-11 |
| B02 | pass | Baseline不新增训练是明确零适配方案，已有50k原始日志和模型；Reference实际16updates/256seen_pairs，正常teacher_reuse_limit结束，梯度/EMA更新路径正常。 optimization_evidence/baseline_runs/seed_0/result.json#training；optimization_evidence/terminal_records/run-390fddf193f5436c91f4c7a0a1bed55a/training/train_result.json；workspace/harbor_task/environment/engine/framework/train_student.py:178-283 |
| B03 | pass | 在既有比较范围内同一发布初始化、A100、数据/评价、四小时上限；适配量属于允许变量，0与16updates之差不构成无依据截短B。当前内场资源架构问题单独判定。 optimization_evidence/baseline_runs/seed_0/result.json#protocol；optimization_evidence/reference_runs/seed_0/result.json#protocol；workspace/harbor_task/instruction.md:9,21,38-40 |
| B04 | pass | B为未改Starter，R改变的是允许的教师筛选与时间分布；没有可定位的只给B异常学习率/错误损失证据，不要求替作者重做最优调参。 workspace/harbor_task/environment/starter/method.py；workspace/reference/method.py；workspace/harbor_task/environment/engine/method_contract.py:33-66 |
| B05 | pass | 已声明的公开seed0及封存私有流各成对，训练seed2001/教师1001实际进入setup和Generator；没有把保存模型重载冒充独立训练。n=1不支持统计稳定性，另列提升证据缺口。 optimization_evidence/comparison_summary.json:6-24；optimization_evidence/current_formal_comparison.json:9-26；workspace/harbor_task/environment/engine/framework/common.py:48-65；workspace/harbor_task/environment/engine/framework/train_student.py:182-183 |
| B06 | pass | 未改Starter本身是适用的公开基本锚点，实测FID9.6632620588；没有包内同协议简单sanity明显胜过B却被忽视的直接证据。 optimization_evidence/baseline_runs/seed_0/result.json#metrics；workspace/harbor_task/environment/starter/method.py |
| B07 | pass | R以同一固定初始化/损失/推理链适配；教师生成数据是声明研究变量，无可见私有标签进入候选。只支持整个方法的一次改善，未证明各因素独立贡献。 workspace/harbor_task/environment/engine/framework/generate_pairs.py:109-190；workspace/harbor_task/environment/engine/framework/train_student.py:189-245；workspace/reference/method.py:11-82 |
| B08 | pass | 说明披露发布EMA来源、零适配角色、训练器独立重实现、16更新、失败及n=1限制，B与评分锚点映射一致。 optimization_evidence/训练证据说明.md:5-31；expert_evidence/专家作业说明文档.md:3-9 |

## Reference 全部正式成对数值与复算

| 协议/评测流 | B | R | U | 改善B−R | 归一化 | 原始result来源 |
|---|---:|---:|---:|---:|---:|---|
| Public/Dev seed0 | 9.663262058820692 | 9.511905516911611 | 9.497105760787463 | 0.151356541909081 | 0.9109287080939392 | optimization_evidence/baseline_runs/seed_0/result.json；optimization_evidence/reference_runs/seed_0/result.json |
| 封存私有单流 | 9.800031891769606 | 9.74957468191576 | 9.646181753456233 | 0.050457209853846 | 0.3279633701145659 | expert_evidence/current_native_nop_20261005_mac/verifier/result.json；expert_evidence/current_formal_reference_20261005/result.json |

两组分别n=1，均值即表中单值，双方n−1样本标准差=null，3σ门槛不可计算。公开相对改善1.566308985390%；私有相对改善0.514867812790%。不把两种随机流混在一起估计方差，也不把重载同模型当独立重训。公开R完整流水3142.311075秒，教师训练478.143044秒、16updates、256样本；正常停止teacher_reuse_limit。逐更新梯度有限，CE约4.37→4.14、LPIPS约.181→.173，未以曲线未收敛单独判退。

公开原始终态位于 optimization_evidence/terminal_records/run-6103fc4495c64392aeb05229602fe9a7/result.json 和 run-390fddf193f5436c91f4c7a0a1bed55a/result.json。两份run.log分别513506/522920字节，含warning和50k/FID进度；两个sample_manifest各50000行，SHA与索引一致。私有B/U校准回执在expert_evidence/current_formal_anchor_calibration/，在正式R之前完成；保存正式EMA完整重载result在expert_evidence/current_saved_formal_ema_reload_20261005/，FID差0，清单SHA一致。这些是包内记录的交叉核验，不是本次动态重现。

额外Baseline evaluation seed1=9.799929104416378、seed2=9.759412334388799（guard04_baseline_seed1_summary.json/seed2_summary.json）未被隐藏。与seed0的补充均值9.740867832541957、样本σ=.07019540048695508、3σ=.21058620146086524；只有B而无完整R配对，故不改正式B锚点，不以此宣布正式统计通过或失败。

| 模型逻辑路径 | 字节 | ZIP内流式SHA-256 |
|---|---:|---|
| optimization_evidence/baseline_runs/seed_0/model/ema_inference_checkpoint.pth | 1312736238 | 387bd60acaa30b114ebe39188e5d174175a04ab39393d947ef2af24c43a2259d |
| optimization_evidence/reference_runs/seed_0/model/ema_inference_checkpoint.pth | 1312736238 | bf07515aad89a3ca0832f16eab9c1be9669810c5ee4d1a15d06473696c7fdb79 |
| optimization_evidence/current_formal_reference/model/ema_inference_checkpoint.pth | 1312736238 | d449833fbe7249f6e212f84794be2bf46583b0719b80f9c9960573448b649a86 |

三份EMA实测SHA与索引一致；未反序列化。公开EMA记录只是CPU导出重载，私有正式EMA另有完整FID重载记录。原始全训练checkpoint/optimizer归档的包外持久位置未独立验收，不伪称已确认包外恢复。

## 三项内容门

| 门 | 状态 | 结论/证据 |
|---|---|---|
| G01 | pass | 三个可执行张量算法接口确实经RPC进入训练/选择/加权链，方法空间超出固定数字搜索。Reference包含选择算法和后半时间采样。 workspace/harbor_task/instruction.md:19-40；workspace/harbor_task/environment/engine/method_contract.py:33-94；workspace/harbor_task/environment/engine/method_proxy.py:106-136；workspace/harbor_task/environment/engine/method_worker.py:30-63 |
| G02 | pass | 在已提供静态材料范围内，发布预训练DD EMA作为未改Starter和正式B合理，未发现直接弱化或预算不公平反证。统计充分性与当前内场架构另行审查。 workspace/harbor_task/environment/starter/method.py；optimization_evidence/baseline_runs/seed_0/result.json；optimization_evidence/训练证据说明.md |
| G03 | fail | 公开Reference归一化0.9109287080939392超出V4[.15,.8]；公开/私有各仅n=1，σ_B与3σ无法估计。作者包内撤销统计门槛及放宽至1.0的口径不覆盖本轮当前V4。 optimization_evidence/comparison_summary.json:6-32；optimization_evidence/reference_runs/seed_0/result.json:58-62；optimization_evidence/训练证据说明.md:29,49,62；workspace/harbor_task/instruction.md:17 |

## 双轨、八字段与有效时长

GPT轨19轮（旧17+当前2），当前中位数教师选择/时间配对FID9.730977102619079→9.627520442260902；Seed轨15轮（旧13+当前2），max-LPIPS+early到min-LPIPS+uniform的联合变化FID10.126636628087738→9.784449511236346，仍保留公开Baseline为该轨最佳。Seed首轮为同checkpoint评测恢复，未当新训练；原失败保留。两个当前下一方向均有分析记录。

两轨主JSON每轮恰好八字段、轮号递增、时间可解析，但当前成功status=SUCCESS应规范为ok。累计小时10.301257/11.721739来自历史36431.318982/41400秒加当前653.207145/798.261664秒。旧Codex从头训练R2/R3、旧Seed改变MSE/CE/LR/EMA的R3/R5/R6不符合当前冻结优化面，不能仅凭概念可迁移计入同任务合规研究。当前计入秒数是下界，额外CLI1047.178/1595.988秒是观察值；本次不把下界不足直接当实际时长上界，但当前各≥10h仍无合格证明，记待核。

最终全局选定源码仍是preselected_reference SHA69db146f135ccc6ff9a6dc57f2aa99af51602993cddf2a75951851e64005798f，旧完整公开/私有复验保留；不是两条当前新候选更优的证明。best_method归位/轨迹快照应统一索引。当前CPU镜像稳定性记录actual_stability_complete=false，不代表12h。

## 格式与Harbor

三大目录齐全，原始日志/模型与多种历史附件均保留。未见expert_annotation.json和顶层expert_evidence/best_method/；后者可由已明确绑定的workspace/reference源码等价归位，不仅因目录名判退。正式私有对照在current_formal_comparison与expert_evidence下，需提供清晰映射。

- expert_evidence/expert_annotation.json：misplaced；缺少轻量注释文件；从现有真实说明整理类型/方向/优化面、同split B/R/U与隔离声明，不塞QA PASS。
- workspace/reference/：merge_candidate；与全局preselected_reference绑定说明合并，唯一最终方法明确归位best_method或记录等价路径；保留当前源码SHA，不造不同方法。
- optimization_evidence/historical_trajectories/：misplaced；旧轨迹可移expert_evidence历史附件；不要与当前冻结轨迹的有效时长混计。
- optimization_evidence/initial_smoke/：extra_allowed；烟测有价值可保留，不能替代正式50k对照。
- expert_evidence/current_protocol_cli_research_20261006/：extra_allowed；源码/原始模型输出/回执有审计价值，可保留索引，规范主轨迹单独生成。
- optimization_evidence/current_formal_reference/：merge_candidate；私有正式对照与标准baseline_runs/reference_runs索引显式映射；保留私有seed承诺，不公开数字seed。

上述额外/归位建议不单独判退；明确缺口已映射到相应检查。通用Harbor文档允许更多配置，不代表本批内场允许外部工作环境。参考：[Configuration](https://docs.harborframework.com/tasks/configuration)、[Separate verifier](https://docs.harborframework.com/tasks/separate-verifier)，2026-10-22静态读取；包内运行目标0.23.0，未做原生schema加载。

COPY/context来源一致，实际提交路径为/workspace/solution/method.py，经collect复制到/logs/artifacts/solution/method.py，再由Verifier恢复相同源码；没有误把Oracle当运行时提交。专用tests镜像不含torch、可信engine/资产，只通过formal.sock调用外部主控，因此整包不是可直接使用自身GPU的当前内场实现。

| Harbor项 | 状态 | 证据/结论 |
|---|---|---|
| H01 | pass | 任务根、双Dockerfile、test.sh和实际封存私有流逻辑/校准回执存在；Hidden是外部受控策略注入，非空目录占位。其目标内场迁移不合规归路径/环境项。 workspace/harbor_task/task.toml；workspace/harbor_task/tests/test.sh；workspace/harbor_task/environment/engine/evaluation_policy.py:11-46；expert_evidence/current_formal_anchor_calibration/baseline-actual-50000-receipt.json |
| H02 | fail | TOML语法可解析、schema1.3和separate明确；但storage_mb、verifier no-network及0GPU+external-controlled-A100与当前内场合同冲突。包内目标为Harbor0.23，未做原生schema加载。 workspace/harbor_task/task.toml:15-47；expert_evidence/current_native_nop_20261006_recovery/preparation.json#runtime |
| H03 | fail | 双context的COPY来源均存在，源码-only经/logs/artifacts移交路径一致；但tests镜像只装Python和client，实际训练/私有评分由外部socket服务执行，且两侧直属Compose必需。 workspace/harbor_task/environment/Dockerfile:9-16；workspace/harbor_task/tests/Dockerfile:3-7；workspace/harbor_task/tests/test.sh:15-18；workspace/harbor_task/environment/docker-compose.yaml:15-19；workspace/harbor_task/tests/docker-compose.yaml:14-18 |
| H04 | pass | 入口清理旧结果，client校验新job/源码/完整样本/有限metric和reward，stage原子替换到/logs/verifier/reward.txt；成功Baseline0是合法值。 workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/runtime_client.py:66-89,108-124,139-168 |
| H05 | pass | 已有NOP配置选择同题、agent=nop、environment=docker、verifier.disable=false；config/result同Trial，没有以普通Reference复测冒充该NOP。 expert_evidence/current_native_nop_20261006_recovery/config.json；expert_evidence/current_native_nop_20261006_recovery/native-trial/result.json |
| H06 | pass_with_deferred | 旧Harbor0.23同次NOP config/result/log/reward完整：separate、nop、无exception、FID9.8000318918/reward0；103项任务文件列表逐SHA匹配。该证据支持外部后端旧链路，迁移后NOP可后补。 expert_evidence/current_native_nop_20261006_recovery/native-trial/result.json；expert_evidence/current_native_nop_20261006_recovery/native-trial/verifier/reward.txt；expert_evidence/current_native_nop_20261006_recovery/native-trial/verifier/test-stdout.txt；expert_evidence/current_native_nop_20261006_recovery/native-trial/trial.log；expert_evidence/current_native_nop_20261006_recovery/verified-current-nop.json |

## 21项实现检查

| 项目 | 状态 | 原因及证据 |
|---|---|---|
| QA01 | pass | 八部分有实质内容；题面没有论文标题、arXiv、参考实现代码或可恢复Hidden答案。 workspace/harbor_task/instruction.md:1-54 |
| QA02 | pass | normalize=(B−metric)/(B−U)，有限数值、U<B、无裁剪；静态代入B→0、U→1，按FID改善方向连续单调。 workspace/harbor_task/environment/engine/scoring.py:5-15；workspace/harbor_task/environment/engine/supervisor.py:162-166 |
| QA03 | fail | Reference源码、正式公开seed0结果和非空原始日志/模型均在；实际公开归一化0.9109287081>.8且default_public_reference_interval_pass=false，属已证实门槛失败。 optimization_evidence/reference_runs/seed_0/result.json:49-62；workspace/reference/reproduce.py；optimization_evidence/reference_runs/seed_0/run.log |
| QA04 | manual | 固定seed不能证明完整训练链确定；公开/私有各n=1，样本标准差null，无法复算有效3σ。额外B seed1/2没有同协议R配对，不能拼成完整统计。 optimization_evidence/comparison_summary.json:22-32；optimization_evidence/current_formal_comparison.json:24-26；expert_evidence/guard04_baseline_seed1_summary.json；expert_evidence/guard04_baseline_seed2_summary.json |
| QA05 | fail | 方法数值guard、冻结校验和超时存在，但核心运行依赖外部GPU；题面四小时且supervisor教师训练14400秒+评测21600秒，未落实完整单轮7200秒。TOML还声明storage_mb/no-network，违背当前内场合同。 workspace/harbor_task/instruction.md:11,38；workspace/harbor_task/task.toml:15-47；workspace/harbor_task/environment/engine/supervisor.py:21-22,122-149 |
| QA06 | pass | test.sh单命令→可信client校验新job/source/result→有限reward.txt；基础设施错误不伪造0，已分类Hard Gate输出−1。 workspace/harbor_task/tests/test.sh:1-18；workspace/harbor_task/tests/runtime_client.py:66-118,127-168 |
| QA07 | pass | 题面仅说明封存非公开评测seed及访问禁令，未披露私有具体seed、标签或样本答案。 workspace/harbor_task/instruction.md:15,30 |
| QA08 | pass | 题面未给出Reference的256/8x/Philox late-half组合及私有参考代码；接口说明是可编程任务合同。 workspace/harbor_task/instruction.md:19-26 |
| QA09 | pass | 入口清理已有reward/result/stage，client写stage后replace；失败trap清理部分成功输出。 workspace/harbor_task/tests/test.sh:3-14；workspace/harbor_task/tests/runtime_client.py:121-124,139-164 |
| QA10 | pass | 可信supervisor运行前后校验protected/assets/checkpoint；候选仅进CPU RPC子进程并先施加Landlock/seccomp，不能直接导入高权限grader。静态不认证完整物理隔离。 workspace/harbor_task/environment/engine/supervisor.py:54-60,82-87,156-159；workspace/harbor_task/environment/engine/method_worker.py:13-24；workspace/harbor_task/environment/engine/worker_guard.py:11-121 |
| QA11 | pass | 目标输入输出、固定环境、指标公式、可改函数、硬边界、源码提交、公开迭代与完成条件均有对应内容。 workspace/harbor_task/instruction.md:3-54 |
| QA12 | pass | ZIP成员清点无.git目录/对象，不存在可读remotes/reflog的提交历史。 @inventory |
| QA13 | pass | 公开声明seed0及独立封存私有流均有实际B/R原始结果、同流正向改善和有限完整FID；不是以归一化B=0当平凡方法。统计充分性和公开区间失败已独立列出。 optimization_evidence/baseline_runs/seed_0/result.json；optimization_evidence/reference_runs/seed_0/result.json；optimization_evidence/current_formal_comparison.json；expert_evidence/current_formal_reference_20261005/result.json |
| QA14 | pass | 可实现题面未给出的中位数提案选择、分层/反相关时间配对及custom权重；现有公开FID距U仍有.0147997561，未达到数学饱和。公开归一化过高仍影响提升门。 workspace/harbor_task/environment/engine/method_contract.py:69-94；expert_evidence/current_protocol_cli_research_20261006/codex/round-02/analysis.json；workspace/reference/method.py |
| QA15 | pass_with_deferred | 既有A100完整公开B2630.654秒/R3142.311秒，R教师训练478.143秒；已交当前候选未证明实际超2h。H20/L20完整耗时未知，可按换卡放宽后补；4h预算和外部架构仍属核心配置冲突。 optimization_evidence/baseline_runs/seed_0/result.json#execution；optimization_evidence/reference_runs/seed_0/result.json#execution；workspace/harbor_task/instruction.md:38 |
| QA16 | manual | 累计10.3013h/11.7217h包含旧协议不可移植训练；Seed的11.5h全部来自旧R3/R5/R6损失/LR等实验。当前各2轮计入653.207/798.262秒是下界，不能证明当前冻结任务各≥10h，也不能反推绝对实际时长。 expert_evidence/run_summary.json:6-49；expert_evidence/duration_audit/codex.json；expert_evidence/duration_audit/seed.json；expert_evidence/protocol-evolution-map-20261006.json；expert_evidence/轨迹协议适用性说明.md |
| QA17 | fail | 显式separate与移交/reward链存在，既有NOP有正证据；但正式工作依赖外部A100、角色socket和直属Compose，镜像自身不含可直接执行的GPU评分依赖，违反当前内场单容器/自身GPU合同。 workspace/harbor_task/task.toml；workspace/harbor_task/environment/docker-compose.yaml；workspace/harbor_task/tests/docker-compose.yaml；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/tests/test.sh:18；author_tools/RUNTIME.md:5-17 |
| QA18 | fail | 两轨八字段及真实当前2轮分数可追溯，但当前成功行status写SUCCESS而非规定ok；历史不合法协议与当前行混为主轨迹，累计时长与当前任务资格未闭环。 expert_evidence/trajectory_codex.json#rounds[17:19]；expert_evidence/trajectory_seed.json#rounds[13:15]；expert_evidence/protocol-evolution-map-20261006.json |
| QA19 | fail | 题面说明evaluation/teacher/training seed，但第17行明确撤销3σ，以一次改善替代；在尚未证明完整评估确定时，与当前有效提升阈值冲突。 workspace/harbor_task/instruction.md:15-17,54 |
| QA20 | pass | 可改method.py、工具调用/公开网络和后端进程权限边界明确；公开单命令入口存在。具体环境禁网配置的当前不兼容已另列。 workspace/harbor_task/instruction.md:21-30,44-50 |
| QA21 | manual_nonblocking | run_summary缺task_binding.hash_profile/task_content_sha256；已有NOP独立文件列表103项与ZIP当前对应文件SHA全匹配，但不冒充官方四项指纹算法。 expert_evidence/run_summary.json；expert_evidence/current_native_nop_20261006_recovery/verified-current-nop.json#current_task_files |

## 失败/待核项的最小修改与验收材料

- G03（fail）：修改：保留全部真实分数及既有B/U，不裁剪、不倒调U、不削弱B。若保持当前题设计，提交满足V4公开区间且有实际同协议成对重复的Reference；随机过程至少3组并按n−1算σ_B。若认为可按确定性评估审查，提供覆盖完整重训/评测链的确定性依据并按适用正式协议提交；仅保存模型重载一致不足以证明。 验收：最终冻结协议、全部配对原始result/run.log/训练模型索引与独立重载复评分；Public/Dev归一化[.15,.8]及随机模式Δ≥3σ_B的可复算表，或可核验的确定性协议材料。
- QA03（fail）：修改：保留全部真实分数及既有B/U，不裁剪、不倒调U、不削弱B。若保持当前题设计，提交满足V4公开区间且有实际同协议成对重复的Reference；随机过程至少3组并按n−1算σ_B。若认为可按确定性评估审查，提供覆盖完整重训/评测链的确定性依据并按适用正式协议提交；仅保存模型重载一致不足以证明。 验收：最终冻结协议、全部配对原始result/run.log/训练模型索引与独立重载复评分；Public/Dev归一化[.15,.8]及随机模式Δ≥3σ_B的可复算表，或可核验的确定性协议材料。
- QA04（manual）：修改：明确确定性边界；若含随机训练/评估，补全同协议成对重复并按Baseline n−1标准差计算，保留失败与选择规则。 验收：完整正式seed/replicate表、B/R原始result/log、训练seed与evaluation replicate区分、σ_B及Δ≥3σ计算。
- QA05（fail）：修改：将教师/训练/评测迁入各自sandbox单容器直接使用自身H20/L20，移除外部GPU请求与直属Compose；删除storage_mb和禁网声明，按实际联网条件实现边界；对完整单轮设置可中断7200秒预算。 验收：更新题面/TOML/Docker与完整调用链；目标资源配置、全部必要阶段计时范围及超时终止证据。换卡后的耗时实测仍可非阻断后补。
- QA15（pass_with_deferred）：修改：保留旧卡真实记录，后续补目标资源估算依据或完整单轮目标卡计时；不要把旧卡标成新卡。 验收：H20/L20端到端阶段计时或明确估算依据、CPU/GPU数量和计时范围；该迁移补证非当前独立阻断项。
- QA16（manual）：修改：将旧协议历史附件与当前合规轨迹分开，按现有起止和扣除区间重新核算当前同任务有效时长；不能直接把旧禁止控制实验计入。缺口由后续获授权研究或已有合法区间材料补齐，本次不启动实验。 验收：每轨当前冻结任务≥36000秒的可核算方法/训练/分析区间、故障/等待扣除和最佳源码复验；训练题不得用7h例外。
- QA17（fail）：修改：保留源码-only移交、候选guard、公开/私有边界和可信reward；将所需训练/评测依赖与资产准备放入支持目标GPU的本sandbox环境，取消外部工作服务/Compose/socket前置。 验收：更新环境与Verifier路径/依赖/Hidden准备/候选移交证据；平台运行状态单列，NOP可非阻断后补。
- QA18（fail）：修改：保留原始模型输出，另生成当前冻结任务的八字段规范轨迹：成功status=ok、失败按实际分类；历史轨迹单列。明确retained_best只在对应当前可见候选集合内比较，并绑定最终候选快照。 验收：两条规范轨迹、逐轮当前源码/完整Public结果/决策、与run_summary一致的时间和最终最佳源码绑定；不得造新轮或改原分数。
- QA19（fail）：修改：同步题面的统计口径与实际随机性：随机模式写明Baseline样本标准差和有效改善≥3σ_B；确定性模式提供可核验依据并保留预声明阈值。不要求为本项额外堆重复次数文字。 验收：修订instruction的seed/有效提升阈值说明及对应冻结协议；统计证据另由成对原始记录验收。
- QA21（manual_nonblocking）：修改：取得真实发布task_fingerprint.py后按environment/、tests/、task.toml、instruction.md全递归计算并补task_binding；保留旧值与范围差异。 验收：真实脚本版本/SHA、四项完整范围输出及run_summary关联；该缺口非阻断，不要求为此全量重跑。
- H02（fail）：修改：将教师/训练/评测迁入各自sandbox单容器直接使用自身H20/L20，移除外部GPU请求与直属Compose；删除storage_mb和禁网声明，按实际联网条件实现边界；对完整单轮设置可中断7200秒预算。 验收：更新题面/TOML/Docker与完整调用链；目标资源配置、全部必要阶段计时范围及超时终止证据。换卡后的耗时实测仍可非阻断后补。
- H03（fail）：修改：保留源码-only移交、候选guard、公开/私有边界和可信reward；将所需训练/评测依赖与资产准备放入支持目标GPU的本sandbox环境，取消外部工作服务/Compose/socket前置。 验收：更新环境与Verifier路径/依赖/Hidden准备/候选移交证据；平台运行状态单列，NOP可非阻断后补。
- H06（pass_with_deferred）：修改：原样保留旧NOP与原registry失败材料；迁移当前合规环境后如需平台补验，提供同一Trial链。 验收：目标版本同次config/result/reward/log与任务关联；此NOP补证按当前授权可非阻断后补。

## 五类非阻断后补

- fingerprint：task_binding缺失，已有旧NOP列表103项字节SHA匹配不等于官方四项指纹。 已有依据：expert_evidence/current_native_nop_20261006_recovery/verified-current-nop.json。后补：真实task_fingerprint.py版本/SHA及全递归四项输出。
- card_effect：B/R实测为A100，H20/L20迁移对分数与锚点影响尚未核验。 已有依据：optimization_evidence/baseline_runs/seed_0/result.json#protocol.resources。后补：目标卡影响说明或已有校准；不强制仅为换卡复跑。
- card_runtime：目标卡完整耗时没有实测，旧卡当前R约52.37分钟。 已有依据：optimization_evidence/reference_runs/seed_0/result.json#execution.elapsed_seconds。后补：资源估算依据或H20/L20端到端计时；非阻断。
- nop：旧外部后端同版本NOP完整，迁移后当前内场NOP未运行。 已有依据：expert_evidence/current_native_nop_20261006_recovery/native-trial/result.json。后补：平台目标同Trial链；可非阻断后补。
- card_rescore：当前保留真实旧卡公开/私有分数，不把它们改标H20/L20。 已有依据：optimization_evidence/current_formal_comparison.json。后补：仅换卡引出的重算/标定可延期；不能豁免已有公开区间失败或原始统计缺口。

## 当前规则冲突与工具缺口

- workspace/harbor_task/instruction.md:17：包内作者旧授权撤销3σ；当前合同：本轮V4仅五类放宽，不豁免统计充分性与Public区间。
- optimization_evidence/训练证据说明.md:29：作者将公开Reference区间改[.15,1.0]；当前合同：本轮Public/Dev仍为[.15,.8]。
- workspace/harbor_task/instruction.md:11,38：外部A100/H100、训练单独4h且不计评价；当前合同：各sandbox单容器自身H20/L20；完整单轮≤2h。
- workspace/harbor_task/task.toml:17,28,43：禁网/storage_mb；当前合同：本批内场不声明禁网/白名单/storage_mb。

未获真实官方0.3.10发布包、独立格式/GPU/指纹脚本；未执行旧v0.3.2收集器，没有旧工具原始结论可转标。该报告为本地V4内容审查，不是官方工具PASS。

## 本次运行边界与风险

- 本次只流式hash ZIP/三份EMA，读取文本/元数据，解析TOML与Python AST；不执行或导入提交代码。
- 未运行Docker、Verifier、训练、模型调用或反序列化；未独立认证权重数值、checkpoint重载、网络物理隔离、12h稳定或目标平台验收。
- 未读包外历史QA/作者修复说明/其他审查者报告；包内历史QA排除。源码与小文本可见隐私抽查无直接真实home/邮箱/私网IP命中；不是完整二进制隐私门禁，逐件严格扫描由主控执行。
- 没有访问当前权威题库，不作期次/题号对外填报或已提交声明。

风险限三项：
- 公开Reference超上限且n=1无3σ证据，私有一次得分合规不能替代公开门槛。
- 外部GPU与Compose链路在旧NOP可运行，但当前内场不接收该工作环境架构。
- 跨协议时长与SUCCESS状态尚未按当前轨迹合同闭环，不能将累计历史小时数当当前任务完成。

## 可复制给专家的返修说明

当前完整包需要返修；本轮为静态内容复核，不是独立复跑或官方0.3.10验收。

1. 保留真实B/R/U和全部失败。公开seed0 B=9.663262058820692、R=9.511905516911611、U=9.497105760787463，复算归一化0.9109287080939392，超过当前V4的0.8（comparison_summary.json及reference_runs/seed_0/result.json）。私有一次0.3279633701145659不能替代Public/Dev门槛。包内作者将区间放宽至1.0、撤销3σ的旧口径不覆盖本轮合同。不要裁剪分数、倒调U或削弱B；交付满足公开区间的实际方法及全部正式成对记录，并明确随机性。随机模式至少3组、按n−1计算Baseline σ及Δ≥3σ；完整保存模型重载不是独立训练重复。若主张确定性，补覆盖完整重训/评测链的依据。

2. workspace/harbor_task/instruction.md:11,38、task.toml:15-47、environment/docker-compose.yaml、tests/docker-compose.yaml、tests/test.sh:18显示CPU sandbox依赖外部A100/H100及socket/SSH工作后端，且训练4h与后评测分开计。请迁移为每sandbox单容器直接用自身H20/L20，取消直属Compose与外部工作服务；删除storage_mb/禁网声明；对完整必要教师/训练/重载/评分轮设置≤7200秒的可中断预算。保留guard、冻结校验、源码-only移交、Hidden与可信reward。验收需更新Docker/context/依赖与资产准备链、TOML和真实调用路径。旧卡已交数值保真；换卡导致的成绩/耗时补证可以延期。

3. expert_evidence/run_summary.json的10.3013h/11.7217h混入旧协议实验；duration_audit/seed.json的11.5h来自旧R3/R5/R6，protocol-evolution-map-20261006.json已说明旧损失、LR和从头训练不属于当前合法轮。不要删历史或把它们改名当前轮。按当前冻结任务逐轨重算起止、排队/安装/故障/阻塞扣除，提供各≥10h的现有合法证据，或在另获执行授权后补足研究。本次只确认当前各2个真实方法闭环，653.207/798.262秒是保守下界而不是总时长上界，当前有效总时长仍未核验。训练题不适用7h例外。

4. 两份主轨迹当前成功行status=SUCCESS（Codex R18/R19、Seed R14/R15），请保留原始输出并生成当前八字段规范副本，成功status=ok；当前与历史分开，逐轮源码、真实Public结果、决策与run_summary时间一致。最终唯一方法源码及必要入口/配置明确归位或映射至workspace/reference；补轻量expert_annotation及私有正式对照索引，避免按目录名造无用材料。

非阻断后补：官方四项指纹/task_binding、换卡影响、目标卡完整耗时、迁移后NOP、仅换卡引出的重算/标定。已有旧Harbor0.23 NOP同Trial无异常、separate、完整50k、reward0，103项文件SHA与现包匹配；应原样保留，不把它改成目标平台通过。真实官方0.3.10及独立格式/GPU/指纹工具仍待取得，不得改旧工具版本或伪造PASS。当前CPU稳定性记录未完成，不代表12h压力通过。

