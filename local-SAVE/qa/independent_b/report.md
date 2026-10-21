# SAVE PBMC IFNB 独立V4只读质检（reviewer B）

输入：SAVE_PBMC_IFNB_TASK.zip；逻辑包根：SAVE_PBMC_IFNB。
SHA256：487dfcc7e7dfebdd261ee7f59eb2098f0a89308557c07a9af6ad65d6a0b6aad4
配置模型：gpt-6-astra（派发元数据）；runtime model/version：not_exposed；session_id：save_b。
报告类型：local-v4-content-review。本次独立输入仅同一完整ZIP和通用审查规则；未读取历史QA/同伴报告。

## 优化面与实际调用边界
三 callable 可实现模型、条件表示、训练目标/更新及推理算法；实际执行器仅约束源码文件/安全边界，非数字参数解析器。
可改：method.py 中 build_model/train_model/predict 与内部实现。
固定：数据切分和MaxAbsScaler预处理；全五seed、原尺度MSE和质量门；固定入口及提交文件限制；公共/私有数据边界与共同资源预算。
minimize mean-expression MSE；非负/有限/shape、checkpoint<=256MiB、fresh/reload；有效分数连续不裁剪。
源码提交guard只检查普通文件、尺寸和三callable；受限执行器加载任意合法方法模块。Candidate仅取得train/control，父进程在其结束后读target并评分。
证据：workspace/harbor_task/instruction.md:16；workspace/harbor_task/tests/grader.py:11；expert_evidence/runtime_integration/numerical_sources/gpu_jail_entry.py:136。

## Baseline来源、训练与公平预算
包内声明来源为SAVE指定commit的条件VAE，约38.02M参数；随机初始化、200 epoch AdamW训练，直接解码stimulated条件并逆缩放。Starter=正式Baseline=归一化B。
只确认包内来源披露及源码/日志一致；外部官方默认来源真实性未联网复核，静态一致不等于独立复现。
Reference：模型与独立同seed训练完全相同；推理改为观测控制表达+stimulated decode-control decode，并保证非负，两个decode重置相同inference seed。
训练计划与实际区分：method传iter=3200，但后端对应epoch调整被注释；不能称实际3200步。十份原始训练日志均200/200；最终训练state_dict评分，未以私有标签选择checkpoint。

- **training_sufficiency**：十份训练log均200/200，recon_loss从约587–633降至51.37–51.53，正常退出；没有逐update计数，iter=3200只是未生效参数。 证据：optimization_evidence/baseline_runs/seed_17/run.log；optimization_evidence/reference_runs/seed_97/run.log；workspace/harbor_task/environment/starter/conditional_backend/model.py:326。
- **method_representativeness**：来源为指定commit的条件扰动VAE，任务限定在冻结数据协议；未发现包内常见更强可比锚点要求。 证据：expert_evidence/source_provenance.json。
- **budget_comparability**：同数据/训练配置/时间上限和设备类型；Reference额外控制解码仍在预算内。 证据：workspace/reference/method.py:44；optimization_evidence/训练证据说明.md:13。
- **implementation_correctness**：损失、参数更新、随机初始化、原尺度评分和weights_only strict重载都有实现；十个模型流式哈希吻合，未加载。 证据：workspace/harbor_task/environment/starter/conditional_backend/VAE_train.py:214；workspace/harbor_task/tests/train_eval.py:62；workspace/harbor_task/tests/metrics.py:16。
- **selection_integrity**：追加71/97后保留初始三seed，最终五组全交、非seed配置固定；未独立认证事先声明的历史时间真实性。 证据：optimization_evidence/训练证据说明.md:9；optimization_evidence/comparison_summary.json#paired_results。

### Baseline八项专项

| 项目 | 状态 | 事实、依据与必要动作 |
|---|---|---|
| B01 | pass | Starter/正式Baseline/B锚点一致；包内记录来源commit和任务限定，没有必须换SOTA的任务内反证。 证据：expert_evidence/source_provenance.json#commit；workspace/harbor_task/environment/starter/method.py。 |
| B02 | pass | AdamW、反传和optimizer.step健全；十次日志均到200/200并正常退出、独立重载。iter=3200未用于后端训练循环，不能视作实际步数。 证据：workspace/harbor_task/environment/starter/conditional_backend/VAE_train.py:214；workspace/harbor_task/environment/starter/conditional_backend/model.py:326；optimization_evidence/baseline_runs/seed_17/run.log；optimization_evidence/reference_runs/seed_97/run.log。 |
| B03 | pass | 同训练数据/模型/200epoch/batch1024、每seed900秒和4线程；实际B约359.22–361.26秒，R约362.06–365.99秒，无单侧削减。 证据：workspace/harbor_task/environment/starter/method.py；optimization_evidence/训练证据说明.md；optimization_evidence/comparison_summary.json。 |
| B04 | pass | 双方训练配置同一，Reference只改推理；没有Baseline异常参数或仅保留最差运行的直接证据。 证据：workspace/harbor_task/environment/starter/method.py:7；workspace/reference/method.py:7；optimization_evidence/训练证据说明.md:9。 |
| B05 | pass | 五seed实际进入初始化和训练；跨seed权重不同，同seed B/R相同符合完全同训练；原始log/PID/时长支持分别执行，不据相同哈希判复制。 证据：workspace/harbor_task/environment/starter/conditional_backend/model.py:179；workspace/harbor_task/environment/starter/conditional_backend/VAE_train.py:164；optimization_evidence/baseline_runs/seed_17/execution/execution_receipt.json；optimization_evidence/reference_runs/seed_17/execution/execution_receipt.json。 |
| B06 | pass | 正式可执行Starter即基本锚点，B取全五seed均值；未另设更弱B。未另交随机/常数sanity不自动判退。 证据：workspace/harbor_task/tests/anchors.json#dev.baseline；optimization_evidence/comparison_summary.json#statistics.baseline_mean。 |
| B07 | pass | Reference为raw_control+stimulated_decode-control_decode后非负投影；固定训练与评分协议一致，支持该整体推理变化收益。 证据：workspace/reference/method.py:43；workspace/harbor_task/tests/metrics.py:7。 |
| B08 | pass | 来源、配方、公平训练与非整篇论文复现范围已披露；外部官方默认真实性不在ZIP-only核验范围内。 证据：expert_evidence/source_provenance.json；expert_evidence/专家作业说明文档.md:7；optimization_evidence/训练证据说明.md。 |

## Reference全部正式成对数值
每条路径均为包内真实result.json；同目录run.log和model/artifact.json、model/reload.log与其对应。
| seed | Baseline MSE | Reference MSE | B−R | 原始来源 |
|---|---:|---:|---:|---|
| 17 | 0.0035213459653151711 | 0.0025195494768602486 | 0.0010017964884549225 | optimization_evidence/baseline_runs/seed_17/result.json；optimization_evidence/reference_runs/seed_17/result.json |
| 29 | 0.0033021915259081455 | 0.0025885801697712648 | 0.00071361135613688068 | optimization_evidence/baseline_runs/seed_29/result.json；optimization_evidence/reference_runs/seed_29/result.json |
| 43 | 0.0037422385312627805 | 0.0026323001823446378 | 0.0011099383489181426 | optimization_evidence/baseline_runs/seed_43/result.json；optimization_evidence/reference_runs/seed_43/result.json |
| 71 | 0.0039580559147975088 | 0.0026512386258838146 | 0.0013068172889136942 | optimization_evidence/baseline_runs/seed_71/result.json；optimization_evidence/reference_runs/seed_71/result.json |
| 97 | 0.0037602680370891666 | 0.0027709962089949437 | 0.00098927182809422287 | optimization_evidence/baseline_runs/seed_97/result.json；optimization_evidence/reference_runs/seed_97/result.json |

B均值=0.0036568199948745547，R均值=0.0026325329327709819。n−1样本标准差σ_B=0.00025142367973954213，σ_R=9.2492084357402582e-05。
正向改善Δ=0.0010242870621035727，3σ_B=0.00075427103921862633，Δ/σ_B=4.073948258035。U=0是数学MSE下界；归一化=(B−R)/B=0.280103221799，相对改善28.01032218%。
剩余headroom R−U=0.0026325329327709819，约10.470505σ_B。
十组COMPLETE、退出码0、独立reload差值0；十个152124106-byte模型仅流式哈希且全部匹配，未反序列化。
五组都同向改善。B/R同seed权重相同符合完全相同训练，原始日志/PID/耗时分别存在；跨seed权重不同。这是五次训练随机性的成对证据，不把固定seed或最终最佳统计方法的确定性当作整个评估无波动。

| 角色 | seed | 日志epoch | train秒 | 完整运行秒 | train/reload PID |
|---|---:|---:|---:|---:|---|
| baseline | 17 | 200 | 360.633858 | 376.296416 | 286016 / 292997 |
| reference | 17 | 200 | 364.140445 | 380.269231 | 272654 / 279657 |
| baseline | 29 | 200 | 359.219803 | 374.224877 | 293254 / 300210 |
| reference | 29 | 200 | 362.062423 | 378.304558 | 279903 / 287272 |
| baseline | 43 | 200 | 359.220861 | 374.256949 | 300453 / 307494 |
| reference | 43 | 200 | 365.994668 | 382.199520 | 287518 / 294683 |
| baseline | 71 | 200 | 361.256119 | 377.410397 | 308404 / 316308 |
| reference | 71 | 200 | 364.401687 | 381.322829 | 295448 / 303686 |
| baseline | 97 | 200 | 360.860195 | 376.322278 | 316613 / 324430 |
| reference | 97 | 200 | 363.524746 | 379.382326 | 304333 / 312726 |

## 三项内容门结论

| 项目 | 状态 | 事实、依据与必要动作 |
|---|---|---|
| G01 | pass | 可执行三callable开放模型/目标/更新/推理，guard不只接受字面常量；Reference有实际条件残差推理变化。 证据：workspace/harbor_task/instruction.md:16；workspace/harbor_task/tests/grader.py:11；expert_evidence/runtime_integration/numerical_sources/gpu_jail_entry.py:136；workspace/reference/method.py:43。 |
| G02 | pass | 现有材料范围内合理：健康来源VAE、200epoch真实完成、共同预算、全五seed与独立重载，无单侧弱化直接反证。 证据：workspace/harbor_task/environment/starter/method.py；optimization_evidence/训练证据说明.md；optimization_evidence/comparison_summary.json。 |
| G03 | pass | 全五组同向且有效；增益4.073948σ_B，归一化0.280103，满足[0.15,0.8]与3σ，属3–5σ可接受证据。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_17/result.json；optimization_evidence/baseline_runs/seed_29/result.json；optimization_evidence/baseline_runs/seed_43/result.json；optimization_evidence/baseline_runs/seed_71/result.json；optimization_evidence/baseline_runs/seed_97/result.json；optimization_evidence/reference_runs/seed_17/result.json；optimization_evidence/reference_runs/seed_29/result.json；optimization_evidence/reference_runs/seed_43/result.json；optimization_evidence/reference_runs/seed_71/result.json；optimization_evidence/reference_runs/seed_97/result.json。 |

## 双轨时长、八字段与最终方法
| 轨迹 | 打包事件 / 成功评分 | 正计时native turn | 有效秒 / 小时 | 最终Public MSE / 归一化 |
|---|---|---|---|---|
| gpt-6.1-sol | 52 / 40 | 37 | 36508.691003 / 10.141303056 | 0.001351913106690589 / 0.630303622113 |
| doubao-seed-2-1-turbo-260628 | 376 / 69 | 154 | 36639.615035 / 10.177670843 | 0.0015912749345400839 / 0.564847343656 |

gpt-6.1-sol：从 nearest context/cell-type response transfer 演进到 donor/context 与 detection/intensity、比例调制；失败和未改善轮保留。
计入区间总长38021.260218秒 − 扣除并集1512.569216秒 = 36508.691003秒。
逐段 native end-start 减已并集处理的下载、队列、失败 provider、阻塞等待；正计时区间无重叠，最终复验 research_credit=0。本次静态重算不认证原宿主时间真实性。
证据：expert_evidence/actual_runtime_evidence/gpt-clock.json；expert_evidence/actual_runtime_evidence/clock_policy_source.py.txt:169；expert_evidence/trajectory_codex.json。

doubao-seed-2-1-turbo-260628：从 nearest context/cell-type response transfer 演进到 donor/context 与 detection/intensity、比例调制；失败和未改善轮保留。
计入区间总长59012.813187秒 − 扣除并集22373.198152秒 = 36639.615035秒。
逐段 native end-start 减已并集处理的下载、队列、失败 provider、阻塞等待；正计时区间无重叠，最终复验 research_credit=0。本次静态重算不认证原宿主时间真实性。
证据：expert_evidence/actual_runtime_evidence/seed-clock.json；expert_evidence/actual_runtime_evidence/clock_policy_source.py.txt:169；expert_evidence/trajectory_seed.json。

全部记录逐条核对八字段、递增round、时间格式、有限成功score/null失败分数及布尔retained_best。40/69成功分数与full_public_result原值多重集合完全一致。GPT最佳为第49轮，Seed最佳为第367轮；后续未改善/失败保留。最终best_method SHA与GPT公开复验一致；canonical选优按Public Dev，已有私有复验不是官方Hidden验收。
训练题不适用7h例外。已有容器liveness不能替代研究时长，本次没有12h压力测试。

## 格式层
三主目录和必需科学材料齐全，八字段格式通过；源任务仍是teaching profile，经adapter/injection才生成可构建上下文。不能将历史混合provider当当前V4部署。
以下仅为整理建议，不单独扣错：
- expert_evidence/actual_runtime_evidence/（extra_allowed）：保留有效计时和原始公开评分证据；精简恢复副本时保持来源可追溯。
- expert_evidence/作者交接记录.md（merge_candidate）：与专家说明内容重复，可合并并保持引用。
- expert_evidence/runtime_integration/（extra_allowed）：作为历史实现证据保留；当前部署入口切至V4兼容运行方式。
- expert_evidence/run_summary.json#agent_execution[1].effective_rounds（merge_candidate）：摘要159 effective rounds、clock154正计时native turns、轨迹376事件/69成功评分，应分别命名并解释单位；不是10h不足证据。
- expert_evidence/expert_annotation.json（merge_candidate）：建议补明确task_type/research_direction；审核状态归run_summary，annotation保持轻量。
- expert_evidence/trajectory_codex.json#rounds.time（extra_allowed）：显式标明轮次时间时区，避免与run_metadata UTC混淆。

## Harbor静态与运行边界
目标Harbor 0.23.0，依据仅包内schema/原始配置与源码；未导入原生schema。Source Dockerfile经export_native转换标准context，Hidden注入已有实际资产与实现；不是“完全缺环境”或“没有Hidden”。只移交method.py，由固定runner重新训练，符合题目提交合同。历史NOP和最终复验仅支持旧混合profile。

### Harbor六项

| 项目 | 状态 | 事实、依据与必要动作 |
|---|---|---|
| H01 | pass | 任务根/入口/双Dockerfile存在；Hidden有真实controls/targets、私有契约和注入器，不把源码目录未预置资产误判为无Hidden。 证据：workspace/harbor_task/task.toml；workspace/harbor_task/tests/test.sh；expert_evidence/private_verifier_provisioning/private_contract.json；expert_evidence/platform_adapter/inject_private_assets.py:23；expert_evidence/runtime_integration/numerical_sources/gpu_jail_supervisor.py:122。 |
| H02 | fail | 显式separate且包内有Harbor0.23类型校验，但当前task.toml仍在两环境声明storage_mb/no-network，违反当前内场要求。 证据：workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:31；workspace/harbor_task/task.toml:34；expert_evidence/runtime_integration/native_schema/existing_config_validation.json。 最小修改/补证：删除task.toml两处storage_mb和no-network声明，按当前内场真实网络/工具策略同步题面及环境；用最小输入、文件权限和受限候选执行实现安全边界，不能依赖平台禁网。 验收：原生配置和题面一致的完整包、可用真实格式/GPU前检的原始输出、资源/网络/超时执行说明。 |
| H03 | fail | 实际provider将容器GPU设为0、要求外置A100，创建broker并依赖Compose；评分靠远程队列，违反每sandbox单容器/自身GPU。源码Dockerfile需已供adapter转换context。 证据：expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:120；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:147；workspace/harbor_task/tests/grader.py:22；expert_evidence/platform_adapter/export_native.py:24。 最小修改/补证：把可信数值评分和受限候选进程迁到每个Solver/Verifier sandbox自身的单容器及分配GPU；去掉外部A100分发、额外broker容器和自定义Compose依赖。保留separate、method.py移交、最小输入和可信父侧评分；交付可按environment/、tests/直接构建的最终上下文或平台支持的显式准备链。 验收：最终双context Dockerfile/依赖/入口、目标provider配置、容器和GPU归属证据、公开Dev/独立Verifier完整评分及移交/reward记录；历史科学数据保留原设备，不因仅换卡强制全部B/R重跑。 |
| H04 | pass | test.sh到grader的reward链路明确，清理旧分数并原子替换；历史NOP五seed与reward=0一致。接口通过不代表拓扑符合V4。 证据：workspace/harbor_task/tests/test.sh:3；workspace/harbor_task/tests/grader.py:17；workspace/harbor_task/tests/grader.py:26；expert_evidence/nop_trial/verifier/result.json；expert_evidence/nop_trial/verifier/reward.json。 |
| H05 | pass | 真实NOP和Job示例任务/provider/nop身份可追溯，Verifier未禁用；示例明确需物化路径且未执行。provider不合规在环境项处理。 证据：expert_evidence/nop_trial/config.json；expert_evidence/portable_runtime_nop/config.json；expert_evidence/runtime_integration/job.example.json；expert_evidence/runtime_integration/launch_nop.py。 |
| H06 | manual（非阻断） | 两次历史NOP均nop/separate、无异常、有finished_at、五seed完整且reward=0；当前ZIP缺V4四项绑定，版本关联非阻断待核，不冒称当前平台通过。 证据：expert_evidence/nop_trial/result.json；expert_evidence/portable_runtime_nop/result.json；expert_evidence/runtime_integration/native_schema/existing_config_validation.json；expert_evidence/run_summary.json。 最小修改/补证：保留原NOP数据；后续用真实四项指纹建立同版本映射，可复用平台既有记录，不为凑NOP自动运行。 验收：当前任务版本映射及同Trial配置/result/reward/日志；迁移拓扑后的对应验收记录。 |

## 核心结论与实现检查
**FAIL / NOT READY；静态内容复核已完成。** 科学优化面、健康Baseline和Reference提升通过现有证据静态复核；实际运行依赖远程GPU、额外broker容器/Compose，且配置声明storage_mb和禁网，违反当前V4核心环境要求。
确定性核心环境失败与五类放宽待补、工具缺口及本次未实跑边界分别列出；科学三门通过不能抵消交付环境不合规。

### 21项实现检查

| 项目 | 状态 | 事实、依据与必要动作 |
|---|---|---|
| QA01 | pass | 题面八部分齐全，无论文题名、arXiv、来源仓库或直接参考答案。 证据：workspace/harbor_task/instruction.md:1。 |
| QA02 | pass | 实际评分为(B-meanMSE)/(B-U)且有效分数不裁剪；Dev B取正式五seed均值，Hidden B独立校准，U=0。 证据：expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:86；workspace/harbor_task/tests/anchors.json；expert_evidence/private_verifier_provisioning/private_contract.json。 |
| QA03 | pass | Reference源码/入口、五seed result/log/model/artifact/reload齐全，归一化0.280103。换卡补证另列。 证据：workspace/reference/method.py；optimization_evidence/reference_runs/seed_17/result.json；optimization_evidence/reference_runs/seed_97/model/reload.log；optimization_evidence/comparison_summary.json。 |
| QA04 | pass | 随机训练与seeded latent推理按五组成对复算：n−1 σ_B=0.000251423680，增益0.001024287062，即4.073948σ。 证据：optimization_evidence/comparison_summary.json；workspace/harbor_task/environment/starter/method.py:15。 |
| QA05 | fail | chroot/seccomp、时间/内存/重载限制有实现；但仍声明禁网/storage_mb并依赖外置GPU和额外broker，违反当前平台核心约束。 证据：workspace/harbor_task/instruction.md:24；workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:18；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:147。 最小修改/补证：把可信数值评分和受限候选进程迁到每个Solver/Verifier sandbox自身的单容器及分配GPU；去掉外部A100分发、额外broker容器和自定义Compose依赖。保留separate、method.py移交、最小输入和可信父侧评分；交付可按environment/、tests/直接构建的最终上下文或平台支持的显式准备链。 删除task.toml两处storage_mb和no-network声明，按当前内场真实网络/工具策略同步题面及环境；用最小输入、文件权限和受限候选执行实现安全边界，不能依赖平台禁网。 验收：最终双context Dockerfile/依赖/入口、目标provider配置、容器和GPU归属证据、公开Dev/独立Verifier完整评分及移交/reward记录；历史科学数据保留原设备，不因仅换卡强制全部B/R重跑。 原生配置和题面一致的完整包、可用真实格式/GPU前检的原始输出、资源/网络/超时执行说明。 |
| QA06 | pass | solve.sh公开入口和test.sh最终入口明确；MSE聚合为标量、失败分类，完整结果写Harness reward。 证据：workspace/harbor_task/environment/public_eval/solve.sh；workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/grader.py:26；expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:83。 |
| QA07 | pass | 仅查题面：描述withheld context但无样本、刺激标签或可恢复Hidden答案。 证据：workspace/harbor_task/instruction.md:7；workspace/harbor_task/instruction.md:19。 |
| QA08 | pass | 仅查题面：未泄露Reference条件差分公式/源码/结果；接口和初始模型背景属于必要说明。 证据：workspace/harbor_task/instruction.md:16；workspace/harbor_task/instruction.md:26。 |
| QA09 | pass | grader删除reward.txt/reward.json旧值，pending文件replace原子写最终结果与reward。 证据：workspace/harbor_task/tests/grader.py:17；workspace/harbor_task/tests/grader.py:25；workspace/harbor_task/tests/grader.py:29。 |
| QA10 | pass | 构建后公开资产/评分器只读，可信evaluator逐次校验冻结与私有资产；候选在UID/chroot内，target仅由可信父侧读。 证据：workspace/harbor_task/environment/Dockerfile:11；expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:12；expert_evidence/runtime_integration/numerical_sources/gpu_jail_entry.py:55；expert_evidence/runtime_integration/numerical_sources/gpu_jail_supervisor.py:122。 |
| QA11 | pass | 目标输入输出、环境、公式、可改面、硬边界、三callable提交、公开迭代和完成标准均有实质内容。 证据：workspace/harbor_task/instruction.md:3；workspace/harbor_task/instruction.md:9；workspace/harbor_task/instruction.md:16；workspace/harbor_task/instruction.md:21；workspace/harbor_task/instruction.md:26；workspace/harbor_task/instruction.md:35；workspace/harbor_task/instruction.md:38。 |
| QA12 | pass | ZIP清点无.git目录/成员，无Git历史泄露面。 证据：@inventory。 |
| QA13 | pass | 全五seed B/R真实fresh日志、退出0、独立reload和质量门均完整且同向改善；十个模型流式SHA256与索引吻合。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_17/result.json；optimization_evidence/baseline_runs/seed_29/result.json；optimization_evidence/baseline_runs/seed_43/result.json；optimization_evidence/baseline_runs/seed_71/result.json；optimization_evidence/baseline_runs/seed_97/result.json；optimization_evidence/reference_runs/seed_17/result.json；optimization_evidence/reference_runs/seed_29/result.json；optimization_evidence/reference_runs/seed_43/result.json；optimization_evidence/reference_runs/seed_71/result.json；optimization_evidence/reference_runs/seed_97/result.json。 |
| QA14 | pass | 可探索条件表示、控制适配和推理；Reference0.280103未饱和，实际最佳统计训练方法说明非参数方法方向可执行。 证据：workspace/harbor_task/instruction.md:17；workspace/reference/method.py:43；expert_evidence/best_method/method.py:129；expert_evidence/best_method/method.py:287。 |
| QA15 | pass | 历史单A100/4CPU；训练359–366秒/seed，完整五seed B/R约1878.51/1901.48秒，轨迹最长公开评分2220.68秒，均低于2h。目标卡耗时可后补。 证据：optimization_evidence/训练证据说明.md:13；optimization_evidence/baseline_runs/seed_17/result.json#execution；expert_evidence/actual_runtime_evidence/gpt-public-experiments.json。 |
| QA16 | pass | 逐段起止减并集扣除复算：GPT36508.691秒、Seed36639.615秒，各>10h；正计时区间无重叠，未把uptime/最终复验计入。 证据：expert_evidence/actual_runtime_evidence/gpt-clock.json#rows；expert_evidence/actual_runtime_evidence/seed-clock.json#rows；expert_evidence/actual_runtime_evidence/clock_policy_source.py.txt:169。 |
| QA17 | fail | separate、Hidden注入和reward存在；但禁网/磁盘声明及broker/Compose/远程GPU正式provider违反V4环境核心要求。历史NOP绑定缺口本身非阻断。 证据：workspace/harbor_task/task.toml；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:120；expert_evidence/runtime_integration/portable_runtime_v4/environment.py:147；expert_evidence/runtime_integration/README.md:5。 最小修改/补证：把可信数值评分和受限候选进程迁到每个Solver/Verifier sandbox自身的单容器及分配GPU；去掉外部A100分发、额外broker容器和自定义Compose依赖。保留separate、method.py移交、最小输入和可信父侧评分；交付可按environment/、tests/直接构建的最终上下文或平台支持的显式准备链。 删除task.toml两处storage_mb和no-network声明，按当前内场真实网络/工具策略同步题面及环境；用最小输入、文件权限和受限候选执行实现安全边界，不能依赖平台禁网。 验收：最终双context Dockerfile/依赖/入口、目标provider配置、容器和GPU归属证据、公开Dev/独立Verifier完整评分及移交/reward记录；历史科学数据保留原设备，不因仅换卡强制全部B/R重跑。 原生配置和题面一致的完整包、可用真实格式/GPU前检的原始输出、资源/网络/超时执行说明。 |
| QA18 | pass | GPT52/Seed376记录逐条恰好八字段，40/69成功score与公开评分原值多重集合完全一致；失败null，最终SHA/Dev分数与复验一致。 证据：expert_evidence/trajectory_codex.json；expert_evidence/trajectory_seed.json；expert_evidence/actual_runtime_evidence/gpt-public-experiments.json；expert_evidence/actual_runtime_evidence/seed-public-experiments.json；expert_evidence/final_revalidation/gpt/agent/dev-result.original.json；expert_evidence/final_revalidation/seed/agent/dev-result.original.json。 |
| QA19 | pass | 题面明确五training seeds、inference seed+10000、全seed聚合和三倍Baseline样本标准差阈值。 证据：workspace/harbor_task/instruction.md:14。 |
| QA20 | pass | 可改文件、平台coding/terminal工具及network/web search规则写明；当前平台规则内容冲突已由核心环境项处理。 证据：workspace/harbor_task/instruction.md:17；workspace/harbor_task/instruction.md:19；workspace/harbor_task/instruction.md:24。 |
| QA21 | manual（非阻断） | run_summary缺task_binding.hash_profile/task_content_sha256；旧数值绑定不能替V4四项指纹，仅非阻断待补。 证据：expert_evidence/run_summary.json。 最小修改/补证：取得真实task_fingerprint.py后全量计算environment/、task.toml、tests/、instruction.md；输出放四项外，映射两轨历史版本，不造算法或历史哈希。 验收：真实工具版本/脚本SHA、原输出、task_binding和历史版本对应材料。 |

## 五类非阻断待补
- **四项指纹/哈希**：run_summary无task_binding；本次模型哈希实际全部匹配。 补证：获得真实发布脚本后补四项指纹和历史版本映射。 验收材料：真实工具版本/SHA、原始输出及task_binding。 来源：expert_evidence/run_summary.json。
- **Baseline/Reference换卡影响**：正式证据为真实A100成对结果，无H20/L20迁移成绩。 补证：保留旧卡事实，后补平台认可迁移可比性说明，不强制完整重跑。 验收材料：准确新旧卡、协议和影响判断或后续实测材料。 来源：optimization_evidence/训练证据说明.md:13。
- **换卡后运行时间**：历史整轮满足2h，目标卡未实测。 补证：后续补目标卡耗时及资源估算依据，区分实测与估算。 验收材料：目标卡配置、计时范围与日志或认可估算。 来源：expert_evidence/actual_runtime_evidence/gpt-public-experiments.json。
- **NOP**：有两次混合profile成功NOP；当前四项版本关联待核。 补证：建立同版本映射或复用平台记录；不为此自动启动NOP。 验收材料：任务版本映射与同Trial config/result/reward/log。 来源：expert_evidence/nop_trial/result.json；expert_evidence/portable_runtime_nop/result.json。
- **换卡后重算分数**：已有旧卡全五组分数和统计达标，未改标新卡。 补证：换卡导致的锚点/分数复核可延期，保留真实历史值。 验收材料：后续迁移评分/锚点影响依据；当前不要求全部seed重算。 来源：optimization_evidence/comparison_summary.json。

## 工具缺口与本次未实跑边界
未取得真实官方0.3.10发布包及独立格式/GPU/指纹脚本；本报告真实类型为local-v4-content-review，不是官方工具PASS。
获得真实脚本后记录版本/SHA并按实际CLI运行；禁止改旧工具版本号或拼造官方输出。
需补验收材料：真实工具/版本/SHA、命令与原始输出。
没有执行/导入提交代码、Docker、安装脚本、模型反序列化、训练或模型API调用。
只读核对已有成对训练、NOP和最终复验；未认证记录宿主时间真实性。
未核验：当前V4目标环境动态验收；H20/L20计时和换卡成绩；12h独立压力测试；官方Hidden验收。
源码/题面可见隐私类型检查未发现作者home、邮箱、私网IPv4；完整逐件文本/二进制扫描由主控执行，本报告不代表外发隐私门禁通过。

## 具体风险（最多三条）
- 远程A100、额外broker和Compose是源码证实的当前环境不合规，不属于仅换卡耗时待补。
- 科学增益处于3–5σ区间，按当前规则接受；不扩展为本次独立复现或全部上下文普遍提升。
- 两轨超10h余量约8.5/10.7分钟；本次已按所交起止扣除复算，新证据需保真更新，不能用容器uptime填补。

## 可直接复制给专家的退回说明
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

