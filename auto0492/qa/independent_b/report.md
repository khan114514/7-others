# DD 当前V4独立只读复核 B

核心结论：**FAIL / NOT READY**。NOT READY：公开Reference归一化超界，运行架构/资源配置不符当前单sandbox契约，当前轨迹与随机证据仍有核心缺口。Baseline本身在已交材料范围内合理。

ZIP SHA256：`4f3ed4b5164bfd052d9e699da6b47726450fe1cd0dac3af65896732295ebb096`。配置模型 gpt-6-astra；runtime型号/版本 not_exposed；独立会话 dd_b。本报告为 local-v4-content-review，非官方0.3.10、自验工具或平台通过。

## 优化面与实际边界

固定DD单步ImageNet256学生，仅提交method.py；RPC检查后让采样/选样/加权参与真实教师与训练流水线。

可实现时间分布、每标签提案选择及相对权重算法；RPC只允许经验证的标量与CPU张量跨边界，可信训练器不直接import候选。冻结学生、VAE、初始化、AdamW与损失。Landlock/seccomp、响应时限、前后冻结哈希有实现。但任务依赖外部GPU及Compose，与当前环境要求冲突。

## Baseline与Reference

发布完成训练的DD EMA单步解码器，num_samples=0不额外适配；同一模型是Starter和B锚点。来源manifest指向提交记录的commit 3b95d33f381d23f66c42d4d230233c32c27eb3c3，未外部重新验证。

从同一EMA开始，256教师目标、每标签8提案、最小LPIPS一致性筛选；CPU Philox4x32-10采样U(.5,1)，uniform weights，固定训练器16updates；优化样本与采样算法而非只改数值。

Baseline专项：**在已提供材料范围内合理**。无适配是合法预训练对照角色，不能称为未训练或故意少训练。

- training_sufficiency：B 为完成训练的发布 EMA，额外训练0是定义；R 实际16updates/256对，停于teacher_reuse_limit，不能据此断言少训练。

- method_representativeness：发布同架构单步模型作为固定初始化与对照合理；本审查不作包外 SOTA 判断。

- budget_comparability：历史同A100/同评价与14400秒适配上限；适配开关及样本策略可变。当前内场资源配置不合规另列，不等于 B 被削弱。

- implementation_correctness：源码有固定初始化、梯度回传、optimizer.step、EMA存储与独立FID；原始日志支持成功。模型仅查索引和元数据，本次未反序列化。

- selection_integrity：B固定发布模型；R选取与失败运行披露，公开/私有分别索引，未将复评包装为多次训练。完整外部搜索历史无法从ZIP证明。

| 检查项 | 状态 | 事实与来源 | 修改与验收 |
|---|---|---|---|
| B01 | pass | 正式 B 与 Starter 均是已发布 DD EMA 的零额外适配；来源有 commit 与资产清单，可作为适配研究的合理对照。<br>`expert_evidence/official_source_manifest.json`；`workspace/harbor_task/environment/starter/method.py`；`workspace/harbor_task/environment/assets.json` | 无需修改。  |
| B02 | pass | B 不重新训练是合法模型角色，非未训练模型；R 原始 updates 为16、seen_pairs=256、teacher_reuse_limit 正常停止，优化器更新与 EMA 保存存在。<br>`optimization_evidence/baseline_runs/seed_0/result.json#training`；`optimization_evidence/terminal_records/run-390fddf193f5436c91f4c7a0a1bed55a/training/train_result.json`；`workspace/harbor_task/environment/engine/framework/train_student.py:248` | 无需修改。  |
| B03 | pass | 双方同发布初始化、公开 seed0、50k、batch8、FID，R 478.143秒适配且共用14400秒历史预算；额外适配是允许变量，不机械要求B也训练。当前平台预算另审。<br>`optimization_evidence/baseline_runs/seed_0/result.json#protocol`；`optimization_evidence/reference_runs/seed_0/result.json#training` | 无需修改。  |
| B04 | pass | B 是未修改发布模型，无异常学习率/故障参数；R 搜索与失败保留。未独立证明所有包外搜索完整，但未发现只弱化 B 的直接反证。<br>`workspace/harbor_task/environment/starter/method.py`；`optimization_evidence/训练证据说明.md`；`expert_evidence/trajectory_codex.json` | 无需修改。  |
| B05 | pass | 公开及私有各一组成对流，训练 seed1001/2001 与评估 seed 分开；没有把 EMA 重载称为独立重训。只有一对不足显著性，但不证明 B 不公平。<br>`optimization_evidence/comparison_summary.json`；`optimization_evidence/current_formal_comparison.json`；`expert_evidence/current_saved_formal_ema_reload_20261005/comparison.json` | 无需修改。  |
| B06 | pass | 发布模型本身作为 Starter 锚点有真实非平凡 FID；未提供更简单随机锚点不单独判错。<br>`optimization_evidence/baseline_runs/seed_0/run.log:6528` | 无需修改。  |
| B07 | pass | 教师提案筛选、时间采样和可选加权是开放变量；B/R 固定模型、损失、评价，未见隐藏数据用于训练的代码路径；只能归因整体适配方案。<br>`workspace/harbor_task/environment/engine/framework/train_student.py`；`workspace/harbor_task/instruction.md:30`；`workspace/reference/method.py` | 无需修改。  |
| B08 | pass | 说明公开了发布 EMA 来源、无适配角色、实际16次更新、预算、重载边界与 n=1，不隐瞒统计不足。<br>`expert_evidence/专家作业说明文档.md`；`optimization_evidence/训练证据说明.md` | 无需修改。  |

全部声明正式成对原值（两种split分开，不能合并求σ）：

| 协议/评估seed | Baseline | Reference | U | 改善 | 归一化 |
|---|---:|---:|---:|---:|---:|
| Public/Dev seed0 | 9.663262058820692 | 9.511905516911611 | 9.497105760787463 | 0.151356541909081 | 0.9109287080939392 |
| 单个封存私有流（commitment见JSON） | 9.800031891769606 | 9.74957468191576 | 9.646181753456233 | 0.05045720985384605 | 0.3279633701145659 |

公开原始来源：`optimization_evidence/baseline_runs/seed_0/result.json` 与 `optimization_evidence/reference_runs/seed_0/result.json`；原始run.log分别第6528/6591行记录FID。底层真实result为 `optimization_evidence/terminal_records/run-6103fc4495c64392aeb05229602fe9a7/result.json`、`optimization_evidence/terminal_records/run-390fddf193f5436c91f4c7a0a1bed55a/result.json`。

私有原始来源：`expert_evidence/current_native_nop_20261005_mac/verifier/result.json` 与 `expert_evidence/current_formal_reference_20261005/result.json`。5.0.3恢复NOP另有同Baseline FID，不当作新增独立seed。

各组n=1，所以均值就是表中原值；B和R的n−1样本标准差均为null，3σ不可计算。训练seed1001/2001与evaluation seed0不是重复训练编号。固定模型/固定随机流的重载一致不证明训练流水线确定。辅助B seed1=9.799929104416378、seed2=9.759412334388799；其与seed0合算σ=0.07019540048695508，但没有配对R，不加入正式判定。

B/R公开全流程2630.654/3142.311秒；R教师训练478.143秒，16updates、256对，原始训练日志正常止于teacher_reuse_limit；私有R教师训练484.119秒。当前历史预算14400秒，部署仍A100。公开R模型只有CPU重载索引且明确未完整重评分；私有R交付EMA另有50k重载FID9.74957468191576及50000PNG哈希一致的既有记录。本次未加载模型。

## 三项内容门

| 检查项 | 状态 | 事实与来源 | 修改与验收 |
|---|---|---|---|
| G01 | pass | 方法边界开放：自定义时间分布、每标签提案选择、相对权重经CPU RPC返回训练器；guard检查形状/范围，不限制为常量搜索。<br>`workspace/harbor_task/instruction.md:21`；`workspace/harbor_task/environment/engine/method_proxy.py`；`workspace/harbor_task/environment/engine/method_contract.py`；`workspace/reference/method.py` | 无需修改。  |
| G02 | pass | 在已提供材料范围内 Baseline 合理；发布 EMA 是健康、可运行且共享初始化的对照，未见不公平削弱的直接反证。<br>`optimization_evidence/baseline_runs/seed_0/result.json`；`optimization_evidence/baseline_runs/seed_0/run.log:6528`；`expert_evidence/official_source_manifest.json` | 无需修改。  |
| G03 | fail | 公开 B=9.663262058820692、R=9.511905516911611、U=9.497105760787463，归一化0.9109287080939392>0.8；私有0.3279633701145659不能替代公开门槛。各n=1亦不能估计随机显著性。<br>`optimization_evidence/comparison_summary.json`；`optimization_evidence/reference_runs/seed_0/result.json#metrics`；`optimization_evidence/current_formal_comparison.json` | 保留现有结果与固定 B/U；删除包内自行扩大的 [0.15,1.0] 验收及三倍标准差豁免。选择满足当前 Public/Dev 归一化区间的真实 Reference，或由当前人类另行明确改变验收合同；不要移动 U 或削弱 B 造达标。 明确评估随机性与训练随机性的区别，补齐同方法同非 seed 配置的成对评估重复；如主张完整流水线确定性，给出充分的确定性依据，不能以固定 seed 或保存模型重载相同替代。 所选方法在同一预先冻结协议的原始成对结果、日志、模型索引与该模型干净重载评分；当前要求仍为 [0.15,0.8]。 随机情形至少各 3 次同 seed 完整原始评估（高波动至少 5），复算 n−1 的 σ_B、改善与 3σ 判定；不要求把固定模型的合法评估重复伪装为独立训练。 |

## 两轨时长与格式

| 轨道 | 打包轮数/当前轮数 | 作者混合时长 | 当前教师训练下界 | 当前有效时长结论 |
|---|---:|---:|---:|---|
| codex | 19/2 | 10.301257258 h | 653.207145 s | 未核清，不能认定≥10h |
| seed | 15/2 | 11.721739351 h | 798.261664 s | 未核清，不能认定≥10h |

当前Codex两轮公开FID为9.730977102619079、9.627520442260902；Seed为10.126636628087738、9.784449511236346。方法确有中位提案/互补时间及最大-最小不一致筛选变化。原始失败与恢复分开保留；恢复同一checkpoint只算一次方法轮。

历史Codex时长下界36431.318982秒，Seed41400秒。旧MSE、从头训练、lr/EMA变化不是当前固定协议合法实验；作者自己已披露。当前材料未给足可计入当前协议的兼容区间，不能以两种不同协议的总数自动达标。本题训练型，不使用7h例外；也不恢复旧12h/15轮门槛。

两轨每轮八字段数量正确；新增成功条目status=SUCCESS需要规范为ok。原始模型输出不修改，规范化副本保留来源。格式缺轻量expert_annotation和统一唯一best_method；可复用现有选定源码补齐，不需要重造历史证据。额外terminal_records/公开implementation允许保留，历史轨迹宜归专家附件。

## Harbor与运行边界

已有恢复NOP记录为nop、separate、无最终异常、reward0，与50000样本回执一致。日志含Compose复制失败后tar回退；不将已恢复的前置错误写为最终失败。此记录使用缓存镜像适配及外部GPU，不能覆盖当前单sandbox部署要求。准确Harbor版本未从已读依据锁定；本次未加载原生schema或动态运行。

| 检查项 | 状态 | 事实与来源 | 修改与验收 |
|---|---|---|---|
| H01 | manual | 任务根/两Dockerfile/入口齐全，私有seed策略由包外formal_evaluation_policy供给，已有私有真实回执；当前交付仅提供外部部署示例，尚无目标单sandbox内Hidden安全准备及依赖入口。<br>`workspace/harbor_task/task.toml`；`workspace/harbor_task/tests/Dockerfile`；`workspace/harbor_task/environment/engine/evaluation_policy.py:10`；`author_tools/deployment.example.json` | 把真实私有策略准备与可信评分器部署到独立Verifier；保留安全注入所需平台操作/不可变入口和版本依据。 目标独立Verifier可获得Hidden与评分依赖的具体准备链；不向Agent提供私有值。 |
| H02 | fail | TOML显式separate，包内NOP支持历史docker配置；当前V4禁用的storage_mb/no-network仍存在，准确Harbor软件版本未锁定可见。<br>`workspace/harbor_task/task.toml:17`；`workspace/harbor_task/task.toml:28`；`workspace/harbor_task/task.toml:43`；`expert_evidence/current_native_nop_20261006_recovery/native-trial/config.json` | 移除当前不支持的storage_mb/禁网声明，按真实目标Harbor版本整理资源字段，记录版本依据。 完整TOML及实际目标配置核验记录；不把TOML语法可读称原生schema已加载。 |
| H03 | fail | 两个Dockerfile的COPY均可在各自context解析，但实际训练/评分需要Compose挂载public/formal socket再转外部GPU；tests镜像没有本地训练评分依赖，无法按当前自身GPU契约交付。<br>`workspace/harbor_task/environment/Dockerfile`；`workspace/harbor_task/tests/Dockerfile`；`workspace/harbor_task/environment/docker-compose.yaml:1`；`workspace/harbor_task/tests/docker-compose.yaml:1`；`workspace/harbor_task/tests/test.sh:20`；`author_tools/ssh_gateway.py:1` | 将教师生成、训练和评分放入各自 sandbox 的单容器并使用本 sandbox 的 H20/L20；移除两处直属 Compose、外部 GPU socket/SSH 依赖和 storage_mb/no-network 声明；同步公开 Dev、独立 Verifier、资产准备与候选移交。 重交完整任务源码、镜像依赖/资产准备和平台资源配置；静态核对本容器完整调用链，目标运行材料按授权补验。历史 A100 成绩原样保留，换卡重算/计时不作为当前单独阻断项。 |
| H04 | pass | 可信客户端检查完整50k、来源hash、协议与有限metric/reward，原子写/logs/verifier/reward.txt；不同失败类别有明确处理。<br>`workspace/harbor_task/tests/runtime_client.py:66`；`workspace/harbor_task/tests/runtime_client.py:108`；`workspace/harbor_task/tests/runtime_client.py:161` | 无需修改。  |
| H05 | pass | 已交NOP调用选择本任务、docker、nop，verifier.disable=false；结果展示separate及有效reward。仅静态核对已有配置，未加载原生schema。<br>`expert_evidence/current_native_nop_20261006_recovery/native-trial/config.json`；`expert_evidence/current_native_nop_20261006_recovery/native-trial/result.json` | 无需修改。  |
| H06 | manual（非阻断） | 已有同次NOP config/result/reward/stdout，separate、无异常、50k、reward0吻合；使用缓存镜像适配及旧外部GPU结构，目标V4迁移后的关联待补。原Compose cp失败后已有tar回退成功，不误判为最终失败。<br>`expert_evidence/current_native_nop_20261006_recovery/native-trial/result.json`；`expert_evidence/current_native_nop_20261006_recovery/native-trial/verifier/reward.txt`；`expert_evidence/current_native_nop_20261006_recovery/native-trial/trial.log` | 保留该历史成功记录；在已获授权目标环境可复用或后补迁移后的同版NOP，不为本静态审查自动启动。 同次Trial配置、结束结果、reward、日志及任务版本对应；此材料缺口非阻断。 |

## 21项检查

| 检查项 | 状态 | 事实与来源 | 修改与验收 |
|---|---|---|---|
| QA01 | pass | 题面八部分齐备，未直接提供参考方法代码、论文题名、arXiv或解法仓库；允许的接口与公开权重背景属任务必要信息。<br>`workspace/harbor_task/instruction.md` | 无需修改。  |
| QA02 | pass | normalize=(B−FID)/(B−U)，检查有限数及0≤U<B；未裁剪，B映射0、U映射1。<br>`workspace/harbor_task/environment/engine/scoring.py:20`；`workspace/harbor_task/environment/anchors.json` | 无需修改。  |
| QA03 | fail | 参考源码/入口、真实 seed0 结果、非空原始日志与EMA存在；但公开归一化0.910929超上限，公开EMA索引明确未完成该模型完整重评分，不能用另一私有新训模型的重载替代。<br>`workspace/reference/reproduce.py`；`optimization_evidence/reference_runs/seed_0/result.json`；`optimization_evidence/reference_runs/seed_0/model/artifact.json#full_saved_model_rescore_of_this_current_reference_complete` | 保留现有结果与固定 B/U；删除包内自行扩大的 [0.15,1.0] 验收及三倍标准差豁免。选择满足当前 Public/Dev 归一化区间的真实 Reference，或由当前人类另行明确改变验收合同；不要移动 U 或削弱 B 造达标。 补该公开交付EMA独立重载评分记录。 所选方法在同一预先冻结协议的原始成对结果、日志、模型索引与该模型干净重载评分；当前要求仍为 [0.15,0.8]。 |
| QA04 | manual | 正式公开与私有各n=1，样本标准差应为null；固定seed及同一保存模型复评相同不证明整个训练评估确定，无法给出有效3σ结论。<br>`optimization_evidence/comparison_summary.json#baseline_sample_standard_deviation`；`optimization_evidence/current_formal_comparison.json#declared_run_count`；`expert_evidence/trajectory_codex.json#rounds[11]` | 明确评估随机性与训练随机性的区别，补齐同方法同非 seed 配置的成对评估重复；如主张完整流水线确定性，给出充分的确定性依据，不能以固定 seed 或保存模型重载相同替代。 随机情形至少各 3 次同 seed 完整原始评估（高波动至少 5），复算 n−1 的 σ_B、改善与 3σ 判定；不要求把固定模型的合法评估重复伪装为独立训练。 |
| QA05 | fail | 存在RPC约束/超时/冻结哈希实现，但当前资源合同仍是外部GPU、4h训练另加6h评测，TOML含storage_mb及no-network，未落实当前内场完整单轮和部署边界。<br>`workspace/harbor_task/instruction.md:11`；`workspace/harbor_task/instruction.md:38`；`workspace/harbor_task/task.toml:17`；`workspace/harbor_task/environment/engine/supervisor.py:19` | 将教师生成、训练和评分放入各自 sandbox 的单容器并使用本 sandbox 的 H20/L20；移除两处直属 Compose、外部 GPU socket/SSH 依赖和 storage_mb/no-network 声明；同步公开 Dev、独立 Verifier、资产准备与候选移交。 将完整单轮预算约束为≤7200秒，区分阶段超时与总时限。 重交完整任务源码、镜像依赖/资产准备和平台资源配置；静态核对本容器完整调用链，目标运行材料按授权补验。历史 A100 成绩原样保留，换卡重算/计时不作为当前单独阻断项。 提供可信进程终止及完整单轮预算的实现。 |
| QA06 | pass | test.sh调用可信客户端；校验成功回执和有限分后写reward.txt，Hard Gate -1，基础设施错误保留失败且不伪装0分。<br>`workspace/harbor_task/tests/test.sh`；`workspace/harbor_task/tests/runtime_client.py:108`；`workspace/harbor_task/tests/runtime_client.py:161` | 无需修改。  |
| QA07 | pass | 只按题面核验：未暴露私有seed值、样本、标签或可恢复答案；只声明封存私有流。<br>`workspace/harbor_task/instruction.md:15` | 无需修改。  |
| QA08 | pass | 只按题面核验：未给Reference具体代码、数值配置或结果；接口说明并非直接答案。<br>`workspace/harbor_task/instruction.md` | 无需修改。  |
| QA09 | pass | 入口清理预写reward/result及stage，可信客户端用stage.replace原子发布，失败清理无效结果。<br>`workspace/harbor_task/tests/test.sh:4`；`workspace/harbor_task/tests/runtime_client.py:139`；`workspace/harbor_task/tests/runtime_client.py:164` | 无需修改。  |
| QA10 | pass | 后端verify_code/verify_assets前后校验冻结面；候选只经独立worker+Landlock/seccomp运行，输入为源码与授权张量。静态认可实现，不认证目标环境隔离实测。<br>`workspace/harbor_task/environment/engine/supervisor.py:52`；`workspace/harbor_task/environment/engine/method_worker.py`；`workspace/harbor_task/environment/engine/worker_guard.py` | 无需修改。  |
| QA11 | pass | 八部分有目标输入输出、方法接口、指标、边界、提交、迭代和完成条件的实质内容；其中过时约束由其他检查明确判退。<br>`workspace/harbor_task/instruction.md` | 无需修改。  |
| QA12 | pass | ZIP成员清点未见.git目录；此结论限已提交包。<br>`@inventory` | 无需修改。  |
| QA13 | pass | 声明的公开 seed0 及单私有流都有成对有效50k原值且R改善，未把保存模型重载算独立训练。统计重复不足和公开区间失败单独处理；换卡无需先重算。<br>`optimization_evidence/baseline_runs/seed_0/result.json`；`optimization_evidence/reference_runs/seed_0/result.json`；`optimization_evidence/current_formal_comparison.json` | 无需修改。  |
| QA14 | pass | CPU函数允许实现非题面直接给出的分层/互补采样及数据相关相对加权；Reference虽超过当前验收上限仍未到U，存在可探索空间，不要求本次发现新解。<br>`workspace/harbor_task/environment/engine/method_proxy.py`；`workspace/harbor_task/environment/engine/method_contract.py`；`optimization_evidence/comparison_summary.json` | 无需修改。  |
| QA15 | fail | 现有A100公开B/R全流程2630.654/3142.311秒均<2h，但当前配置仍明确外部A100/H100、4h训练及6h评测，未迁移H20/L20单sandbox执行；这不是仅缺换卡计时。<br>`optimization_evidence/baseline_runs/seed_0/result.json#execution`；`optimization_evidence/reference_runs/seed_0/result.json#execution`；`workspace/harbor_task/instruction.md:38`；`workspace/harbor_task/task.toml:47` | 将教师生成、训练和评分放入各自 sandbox 的单容器并使用本 sandbox 的 H20/L20；移除两处直属 Compose、外部 GPU socket/SSH 依赖和 storage_mb/no-network 声明；同步公开 Dev、独立 Verifier、资产准备与候选移交。 修正允许预算覆盖训练、重载与评分的完整单轮。 重交完整任务源码、镜像依赖/资产准备和平台资源配置；静态核对本容器完整调用链，目标运行材料按授权补验。历史 A100 成绩原样保留，换卡重算/计时不作为当前单独阻断项。 已有A100耗时原样保留；H20/L20耗时补证可非阻断后补。 |
| QA16 | manual | 宣称10.3013/11.7217h由旧v1.0/v1.1时长加当前653.207/798.262秒；旧轨含从头训练、MSE、lr/EMA变动，未给出可计入当前固定协议的逐区间等价证明。不能认定两轨已各≥10h。<br>`expert_evidence/run_summary.json#tracks`；`expert_evidence/duration_audit/codex.json`；`expert_evidence/duration_audit/seed.json`；`expert_evidence/轨迹协议适用性说明.md` | 将旧协议与当前固定训练器的有效区间分开，逐区间证明协议兼容后才计入；继续保留原始历史证据。只对可承认时长的实际缺口补当前双轨迭代，不把故障、排队或复验重复计入。 两条独立轨迹各自可追溯的 ≥36000 秒合规有效时长、起止/扣除区间和方法—公开评分—决策记录；本题有训练，7h 例外不适用。 |
| QA17 | fail | 独立separate和artifacts/reward链存在，但两个直属Compose及socket转交外部GPU违反当前单容器、自身GPU要求；历史NOP成功不豁免部署结构。<br>`workspace/harbor_task/task.toml`；`workspace/harbor_task/environment/docker-compose.yaml`；`workspace/harbor_task/tests/docker-compose.yaml`；`workspace/harbor_task/tests/test.sh` | 将教师生成、训练和评分放入各自 sandbox 的单容器并使用本 sandbox 的 H20/L20；移除两处直属 Compose、外部 GPU socket/SSH 依赖和 storage_mb/no-network 声明；同步公开 Dev、独立 Verifier、资产准备与候选移交。 重交完整任务源码、镜像依赖/资产准备和平台资源配置；静态核对本容器完整调用链，目标运行材料按授权补验。历史 A100 成绩原样保留，换卡重算/计时不作为当前单独阻断项。 |
| QA18 | fail | 两轨19/15轮均具八字段，末两轮有真实公开50k回执；但当前4条成功状态为SUCCESS而非ok，主轨混入已声明不属当前协议的旧轮。当前长时依据和唯一best_method交付也需清楚映射。<br>`expert_evidence/trajectory_codex.json#rounds[17:19]`；`expert_evidence/trajectory_seed.json#rounds[13:15]`；`expert_evidence/轨迹协议适用性说明.md`；`expert_evidence/current_public_best_retention_20261006/verified-retained-source.json` | 把两份规范化轨迹的成功状态改为 ok，保留原始 SUCCESS 回执；把旧协议历史轨迹归档，与当前轮次及选优范围分开；提交唯一最终 best_method 的源码/入口映射和同公开口径选择证据。 两轨每轮恰好八字段、连续轮号、实际公开分数、失败原因和布尔最佳标记；历史与当前方法边界、来源记录及最终提交源码可逐轮追溯。 |
| QA19 | fail | 题面列出教师/训练/公开评估seed，却明示三倍标准差不作为门槛，没有当前随机评估有效提升阈值；该作者旧豁免不在当前五类放宽内。<br>`workspace/harbor_task/instruction.md:15`；`workspace/harbor_task/instruction.md:17` | 同步当前随机性与有效改善阈值，明确保持固定的训练和评价随机流；不要把旧作者豁免写为当前有效规则。 修订题面及配套冻结协议；随机性证据与题面阈值一致。 |
| QA20 | manual | 题面明确仅改method.py和Agent公开网络/候选网络限制，但未明确可用工具策略或平台默认工具策略的适用对象。<br>`workspace/harbor_task/instruction.md:21`；`workspace/harbor_task/instruction.md:30` | 在题面加入明确的平台默认工具策略引用及工具使用边界，并与当前允许联网的内场配置一致。 修订instruction与实际环境配置对齐；不需要固定webSearch拼写。 |
| QA21 | manual（非阻断） | run_summary缺task_binding.hash_profile/task_content_sha256；旧B/R frozen_manifest摘要不同。保留原值，按当前口径非阻断待补，不据此推定模型损坏或要求全量重跑。<br>`expert_evidence/run_summary.json`；`optimization_evidence/baseline_runs/seed_0/result.json#protocol.actual_frozen_manifest_sha256`；`optimization_evidence/reference_runs/seed_0/result.json#protocol.actual_frozen_manifest_sha256` | 获得真实发布task_fingerprint.py后补四项全递归指纹与各轨版本映射；输出置四项之外。 真实工具版本/SHA及指纹结果，保留历史摘要，不改写旧运行身份。 |

## 当前规则冲突与五类非阻断待补

- `workspace/harbor_task/instruction.md:17`：包内自述3σ豁免与公开Reference上界[0.15,1.0]来自旧作者规则，不是当前用户五类放宽，不能沿用。

- `workspace/harbor_task/instruction.md:11`：CPU sandbox外接A100/H100 GPU后端，与当前单容器本sandbox GPU要求不一致。

- `expert_evidence/轨迹协议适用性说明.md`：旧协议研究时长不能无等价证明合并为当前固定协议10h；不恢复旧12h+15轮门槛。

- `review-policy`：旧参考文档强制双Dockerfile/缺NOP即失败的内容由V4覆盖；本报告不因这些旧门槛判退。

- 指纹/哈希：task_binding缺失，历史frozen_manifest值不同；不推断伪造。 后补：真实官方指纹工具和版本映射。

- 换卡对Baseline/提升的影响：已有真实A100配对可继续使用；迁移H20/L20后的影响未核。 后补：后续核对卡型变化与锚点影响。

- 换卡耗时：已交A100全流程耗时清楚；目标H20/L20计时未提供。 后补：按授权后补目标卡估算/实测。

- NOP：旧external-GPU设计的同次NOP成功；目标V4改造后链路尚未验收。 后补：目标同版Trial配置/result/reward/log关联。

- 换卡重算：不因缺新卡B/R重算阻断；公开0.910929超界是已有原值，不属于换卡补证豁免。 后补：保留旧成绩并按需后补。

## 工具缺口与本次边界

- 真实官方0.3.10、独立格式/GPU检查及task_fingerprint.py未提供；本报告为本地内容审查，非官方工具输出。
- 只对同一完整ZIP流式SHA256与选定文本/JSON/日志/源码和成员元数据读取。
- 未执行/导入提交代码、Docker、安装器或Verifier，未训练/模型调用/权重反序列化。
- 未读取包外历史QA或同伴报告；包内历史QA未用作结论。
- 未完成全二进制隐私发布扫描；由主控逐附件独立执行，不能据本报告外发。

风险仅限：

- 公开单对提升真实但统计稳定性尚未成立，保存EMA复评不能支持重训稳定性。
- 历史长轨主要包含当前冻结面不允许的变化，混合总时长可能误示当前协议达标。
- 既有NOP只支持外部GPU/Compose部署，不能推为当前内场结构通过。

## 可复制专家退回说明

DD 当前完整包需返修，当前口径 NOT READY。以下是具体动作，不要求为五类放宽事项立即补跑。

1. 公开Reference真实归一化0.9109287080939392超过当前0.8；私有0.3279633701145659不能代替公开门槛。见optimization_evidence/comparison_summary.json与reference_runs/seed_0/result.json。保留现有结果与固定 B/U；删除包内自行扩大的 [0.15,1.0] 验收及三倍标准差豁免。选择满足当前 Public/Dev 归一化区间的真实 Reference，或由当前人类另行明确改变验收合同；不要移动 U 或削弱 B 造达标。 所选方法在同一预先冻结协议的原始成对结果、日志、模型索引与该模型干净重载评分；当前要求仍为 [0.15,0.8]。

2. 成对公开/私有均只有一个流。σ_B不可估计；同一保存模型重载不是独立随机重复。见comparison_summary.json及current_formal_comparison.json。明确评估随机性与训练随机性的区别，补齐同方法同非 seed 配置的成对评估重复；如主张完整流水线确定性，给出充分的确定性依据，不能以固定 seed 或保存模型重载相同替代。 随机情形至少各 3 次同 seed 完整原始评估（高波动至少 5），复算 n−1 的 σ_B、改善与 3σ 判定；不要求把固定模型的合法评估重复伪装为独立训练。

3. 题面第11/38行、task.toml及environment/tests直属docker-compose.yaml仍依赖外部A100/H100、4h训练另计评分、storage_mb与禁网。将教师生成、训练和评分放入各自 sandbox 的单容器并使用本 sandbox 的 H20/L20；移除两处直属 Compose、外部 GPU socket/SSH 依赖和 storage_mb/no-network 声明；同步公开 Dev、独立 Verifier、资产准备与候选移交。 完整单轮限制应覆盖训练/重载/评分且≤7200秒。重交完整任务源码、镜像依赖/资产准备和平台资源配置；静态核对本容器完整调用链，目标运行材料按授权补验。历史 A100 成绩原样保留，换卡重算/计时不作为当前单独阻断项。

4. run_summary和两份duration_audit把旧协议时长加当前短轮。将旧协议与当前固定训练器的有效区间分开，逐区间证明协议兼容后才计入；继续保留原始历史证据。只对可承认时长的实际缺口补当前双轨迭代，不把故障、排队或复验重复计入。 两条独立轨迹各自可追溯的 ≥36000 秒合规有效时长、起止/扣除区间和方法—公开评分—决策记录；本题有训练，7h 例外不适用。

5. trajectory_codex末2轮与trajectory_seed末2轮成功状态为SUCCESS；规范化轨迹改为ok并保留原始回执。把两份规范化轨迹的成功状态改为 ok，保留原始 SUCCESS 回执；把旧协议历史轨迹归档，与当前轮次及选优范围分开；提交唯一最终 best_method 的源码/入口映射和同公开口径选择证据。 两轨每轮恰好八字段、连续轮号、实际公开分数、失败原因和布尔最佳标记；历史与当前方法边界、来源记录及最终提交源码可逐轮追溯。 补轻量expert_annotation与明确工具策略引用。

6. 公开Reference模型artifact索引明确full_saved_model_rescore=false；私有新训模型已有完整重载是另一产物。为实际公开评分模型补独立干净进程重载评分或明确可接受的完整等效证据。

非阻断后补：四项官方指纹及历史hash差异、换卡成绩影响、换卡耗时、迁移后NOP版本关联、换卡重算。已有A100原值和旧NOP照实保留；不改标卡型、不补造hash、不重复计算恢复/重载为新研究。

工具验收：真实官方0.3.10及格式/GPU/指纹脚本未提供，后补真实版本验收；本次仅静态读取，无训练、Docker、模型调用或反序列化。

