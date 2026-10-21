# Bifrost 独立只读内容复核（reviewer B）

核心结论：**NOT READY / FAIL**。安全静态复核完成；平台运行拓扑、失败分数分类及轨迹格式有明确失败，公开 Dev Reference 证据待核。
输入ZIP SHA256：`c97e90c1e23768b61b992fa0b03b0894a4a7d425b33bafe077160ff911aa2e3c`。session_id=bifrost_b；configured_model=gpt-6-astra；runtime_reported_model/runtime_version=not_exposed。

## 优化面与实际执行边界
固定公开预训练模型的等价推理图算法优化，不是数值超参搜索。
题面允许1–256节点data-only DAG，执行器实际支持算子组合、attention、切分重组、prefix缓存和固定Inductor编译区；字段/权重白名单、meta shape、7GiB cache等guard实装。Verifier只读program.json，未执行提交的Python builder。数值执行实际走独立远程A100 worker；当前每角色还启动broker容器和Compose override，这是当前V4核心架构冲突。

## Baseline 来源、训练与公平性
Starter=published eager dense DAG；正式Baseline=相同语义；主评分B=同一次正式配对Baseline聚合；NOP为额外同语义sanity。
published eager dense 双分支decoder；两个模态分支按mask混合，在全部行上计算，每步重算，无训练。
训练不适用。来源版本由benchmark.json、public-checkpoint-assets.json和源码绑定；实际使用固定28层BF16、三采样seed、每seed两例×三重复×256阶段，双方同一CUDA child交替计时并统一质量门。Reference实际变化为七个text节点prefix复用。Baseline正常功能及公平性可由源码与已有原始记录支持，当前无法证明独立运行真实性。

五维事实：
- training_sufficiency：固定预训练模型，无训练/微调；训练步数、学习曲线和重训 checkpoint 不适用；每 seed 的推理记录完整。
- method_representativeness：published eager dense 双分支 decoder 是图执行优化的直接可解释对照；并非仅以年代或分数低判弱。
- budget_comparability：同一工作、A100、同权重与输入，18 次成对重复、28层、256阶段，统一34GiB质量/显存门。
- implementation_correctness：代码 strict=True 加载；warmup 比较官方形状 eager 源；已有正式六例 source_alignment=0，exit=0。
- selection_integrity：全部三个正式 seed 六组 result 均回溯同一次成对工作；日志明确是同一次 CPU 复算 stdout 的重复，不声称逐seed GPU stdout。

| 项目 | 状态 | 事实/证据 |
|---|---|---|
| B01 | pass | 正式 Baseline 为可运行 Starter 的 published eager dense dual-branch decoder；固定 BF16 28 层模型，适合算子/推理图优化对照，未发现必须替换为更强锚点的直接证据。 证据：Task/environment/starter/method.py:9；Task/environment/public_eval/protocol/benchmark.json:52；optimization_evidence/README.md |
| B02 | pass | 不训练；evaluator 严格加载权重并在每例 warmup 对齐 published decoder。正式记录 source_alignment 六例为零误差，正常退出。不能把不训练判为训练不足。 证据：Task/trusted/full_decoder_evaluator.py:65；Task/trusted/full_decoder_evaluator.py:135；expert_evidence/release-full-validation/oracle/result.json#source_alignment |
| B03 | pass | B/R 在同一 A100 CUDA child、同输入/固定权重、同 sampler/种子/重复/质量门；交替顺序，首次缓存构建计入计时。当前部署不合 V4 单列，不构成已证明 B/R 不公平。 证据：Task/trusted/full_decoder_evaluator.py:147；optimization_evidence/baseline_runs/seed_101/result.json#protocol |
| B04 | pass | Baseline 使用冻结 eager 图，无异常学习率/早停等训练参数；Reference 研究变量为中间结果生存期，配置变化符合开放图接口。没有强迫双方搜索次数相同。 证据：Task/environment/starter/method.py；Task/solution/reference_method.py:34 |
| B05 | pass | 101/202/303 实际进入 torch.manual_seed/randperm；每 seed 两例×三次重复，完整配对。它们是固定预训练模型的采样种子与延迟重复，不是独立训练。 证据：Task/trusted/full_decoder_evaluator.py:126；Task/trusted/full_decoder_evaluator.py:149；optimization_evidence/训练证据说明.md |
| B06 | pass | 另有 NOP/Starter 相同语义的实测锚点：B=58274.88579305013ms、T=58294.56111907959ms、reward=-0.000337629593978607；保留计时噪声且质量通过，不硬置零。 证据：expert_evidence/release-full-validation/nop/result.json |
| B07 | pass | Reference 仅将七个 text 节点改为 prefix 生命周期；权重/数据/质量门不变，改善可归于复用。来源为固定模型推理而非训练或 FLUX 图像质量提升。 证据：Task/solution/reference_method.py:34；expert_evidence/release-full-validation/oracle/raw-stage-times.json；expert_evidence/release-full-validation/oracle/result.json |
| B08 | pass | README 与训练说明披露来源、Baseline/Starter/锚点映射及完整冻结协议、日志复用范围；未发现挑最差 B 或把同 checkpoint 冒充独立训练。 证据：optimization_evidence/README.md；optimization_evidence/训练证据说明.md；optimization_evidence/comparison_summary.json |

## Reference 方法和全部正式原始数值
维持dense形状与同权重，将text input norm、Q/K/V、attention out、post norm、MLP七个节点设prefix，跨阶段复用，每prompt/rep重置，首次构建计时。
正式数据为formal/Hidden，而非公开Dev。`optimization_evidence/训练证据说明.md`明确六个run.log是同一原CPU复算stdout的重复；原GPU stdout未附，但全部阶段/重复、原生Trial stdout、CPU原stdout与六份latent已提供。不要求对非训练任务造model目录。

| seed | B(ms) | R(ms) | 改善(ms) | 原始result路径 |
|---|---:|---:|---:|---|
| 101 | 58263.6082611084 | 33606.80603027344 | 24656.80223083496 | optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_101/result.json |
| 202 | 58411.00238800049 | 33728.2001953125 | 24682.80219268799 | optimization_evidence/baseline_runs/seed_202/result.json；optimization_evidence/reference_runs/seed_202/result.json |
| 303 | 58622.83100128174 | 33852.496841430664 | 24770.334159851074 | optimization_evidence/baseline_runs/seed_303/result.json；optimization_evidence/reference_runs/seed_303/result.json |

均值 B=58432.48055013021 ms、R=33729.16768900553 ms；n−1样本SD B=180.57194646667463 ms、R=122.8482629295528 ms。改善=24703.31286112468 ms；3σ=541.7158394000239 ms；改善=136.80592885276224σ；U=0，归一化=0.422766800733905。剩余到U空间=33729.16768900553 ms。
逐case先取三重复中位数，再取每seed两个case平均，再跨seed平均。4608条阶段数值求和与下表18条原始重复完全一致，误差最大0。每个seed的两份角色result均与该共享原始记录相符。

以下全部重复来源：`expert_evidence/release-full-validation/oracle/result.json#records`；阶段原始来源：`expert_evidence/release-full-validation/oracle/raw-stage-times.json`。

| seed | case | repetition | B total(ms) | R total(ms) |
|---|---:|---:|---:|---:|
| 101 | 0 | 0 | 57705.82113647461 | 33308.66484069824 |
| 101 | 0 | 1 | 57946.475524902344 | 33426.11553955078 |
| 101 | 0 | 2 | 58105.807861328125 | 33530.78276062012 |
| 101 | 1 | 0 | 58428.64637756348 | 33679.11723327637 |
| 101 | 1 | 1 | 58580.74099731445 | 33787.496520996094 |
| 101 | 1 | 2 | 58696.246337890625 | 33925.676025390625 |
| 202 | 2 | 0 | 58668.26139831543 | 33851.179931640625 |
| 202 | 2 | 1 | 58763.357192993164 | 33945.01222229004 |
| 202 | 2 | 2 | 58811.39796447754 | 34008.9384765625 |
| 202 | 3 | 0 | 58277.298080444336 | 33616.94012451172 |
| 202 | 3 | 1 | 57982.105560302734 | 33472.25196838379 |
| 202 | 3 | 2 | 58058.64758300781 | 33511.38816833496 |
| 303 | 4 | 0 | 58285.94079589844 | 33607.761962890625 |
| 303 | 4 | 1 | 58554.48667907715 | 33812.86805725098 |
| 303 | 4 | 2 | 58707.5491027832 | 33936.20474243164 |
| 303 | 5 | 0 | 58549.045333862305 | 33782.75941467285 |
| 303 | 5 | 1 | 58691.17532348633 | 33892.12562561035 |
| 303 | 5 | 2 | 58713.41368103027 | 33944.01988220215 |

质量：全部阶段及六例最终误差记录均0，峰值27697030656bytes，child_exit_code=0。仅静态核对，没有加载latent或权重。
额外最终复验：GPT自身配对B=58570.95049285889、T=14484.337151845297、reward=0.7527044203660095；Seed自身配对B=58277.173444112144、T=15967.34054629008、reward=0.7260103810353333。两条不是相互配对显著性实验。

## 三项内容门
| 项目 | 状态 | 结论及证据 | 最小修改与验收 |
|---|---|---|---|
| G01 | pass | 开放 1–256 节点数据 DAG、算子组合、张量分区、attention、prefix 生命周期与受限编译区；真实 guard 与执行链允许方法更新，Reference 七个节点复用是算法性变化。 证据：Task/instruction.md#Allowed Scope；Task/trusted/tensor_dag.py:26；Task/trusted/tensor_dag.py:160；Task/solution/reference_method.py:34 | 无需核心修复。  |
| G02 | pass | 在已提供材料内 Baseline 合理；固定预训练 eager 源与 Starter 对齐，同数据和同工作配对，未发现无依据少训练、故障、额外数据或挑 seed。 证据：optimization_evidence/README.md；Task/trusted/full_decoder_evaluator.py:135；optimization_evidence/comparison_summary.json | 无需核心修复。  |
| G03 | manual | 正式 Hidden 的全部成对数值满足正向改善、归一化区间和 3σ；但交付正式 Reference 证据明确为 formal/Hidden，未定位同一 Reference 的公开 Dev 成对自验，不能把 formal 结果改标 Dev。 证据：expert_evidence/release-full-validation/oracle/result.json#profile；Task/trusted/full_decoder_evaluator.py:21；optimization_evidence/reference_runs/seed_101/result.json#protocol | 补作者 Reference 同一源码/graph 在公开 Dev 协议下的正式成对结果；明确公开与 Hidden 分别使用的案例、重复数、聚合和 B/R/U，现有 Hidden 全量结果继续保留。 公开 Dev 的原始阶段/重复记录、全部声明 seed、结果与日志，绑定当前 Reference；复算归一化 [0.15,0.8] 和 ≥3σ。若已有结果可直接补入，无需为本静态审查自动启动运行。 |

## 双轨时长与方法演进
采用非训练且短迭代的7h例外，不认定10h已核。GPT保守下界25828.664409399033秒=7.174629h；Seed27860.4020447731秒=7.739001h。160/363个区间求和与摘要一致、内部无重叠，与全部已知timed exclusion及bounded infrastructure区间交叠均0。原始native/provider时钟投影与失败扣除材料保留；不计未知未闭合、长期阻塞、准备、最终复验及容器uptime。
GPT有48个源码变更→Dev反馈窗口，中位153.186秒；Seed20个，中位168.496秒，均超过3个方法闭环；合法失败也保留。两个最终候选分别有独立native正式复验，文件SHA与各自selected源码一致。后续方向由原始trajectory及源方法分析支持：GPT可继续改变attention存储/编译边界，Seed保留rotary/RMS/softmax草稿且未编造新分数。
GPT102轮、Seed59轮，不等同102/59次独立有效方法实验。GPT从dense text缓存、分区/融合投影发展到post-softmax编译；Seed从BF16切片误差排查发展到融合QKV及flattened image MLP，随后编译view/SiLU收益接近计时噪声。Seed保留较简单exp91而非拿微小噪声宣称更强，选择理由有记录。

## 格式、Harbor及21项
Task/与workspace/README映射接受等价布局，不强制改名。optimization_evidence正式结果与raw来源清楚；轨迹字段不合V4，expert_annotation含过多运行史且未简明给同split的B/R/U，唯一最终best_method未明确。

格式建议（等价布局或额外交付本身不扣错）：
- Task/ [extra_allowed]：可保留当前真实task根，提交清单明确映射；不因非workspace/harbor_task名直接扣错。
- expert_evidence/trajectory_codex.json [misplaced]：每轮来源/哈希/时钟扩展移入旁表，主轨迹保留恰八字段。
- expert_evidence/expert_annotation.json [merge_candidate]：将运行史迁入run_summary，annotation仅题型/研究方向/优化面/指标/同split B/R/U及基本隔离。
- Trajectories/GPT/selected-method [misplaced]：两轨方法可作证据保留，但明确唯一最终交付及Public/Dev选择依据，并设best_method或明确等价目录。
- expert_evidence/historical-* [merge_candidate]：归档过期副本并给单一当前索引；旧QA未作为本次证据。
- optimization_evidence/*_runs/seed_*/run.log [extra_allowed]：保留明确说明：六个run.log是同一原CPU审计stdout的显式重复，真实GPU stdout未附；不要改名声称逐seed GPU日志。

| 项目 | 状态 | 结论及证据 | 最小修改与验收 |
|---|---|---|---|
| H01 | pass | Task/为等价任务根；双Dockerfile、test.sh齐全。Hidden三个chunk共40912872字节，由tests镜像prepare_hidden校验合并，grade核同一SHA；不按固定目录名扣错。 证据：Task/task.toml；Task/tests/Dockerfile；Task/tests/hidden-chunks/manifest.json；Task/tests/prepare_hidden.py；Task/tests/grade.py:24 | 无需核心修复。  |
| H02 | fail | Harbor0.23.0及custom provider明确，TOML已有separate；但当前配置仍声明storage_mb和no-network，并以gpus=0托管远程GPU，未按V4目标内场配置。 证据：Runtime/requirements.txt；Task/task.toml | 将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。 同步删除当前内场不支持的storage_mb/禁网声明。 提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。 |
| H03 | fail | 两个build context的COPY源和最终路径有效、artifacts覆盖JSON/方法。失败点是数值依赖不在sandbox镜像内，adapter另启动broker并依赖外部controller/SSH worker。 证据：Task/environment/Dockerfile；Task/tests/Dockerfile；Runtime/hybrid_harbor_environment.py:137；Runtime/hybrid_relay.py:102；Runtime/numerical_controller.py:28 | 将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。 提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。 |
| H04 | fail | 有效候选确写有限reward且原子替换；但异常分支写-1/0问题使基础设施失败可生成正常分数哨兵，不符合当前错误分类要求。 证据：Task/tests/test.sh；Task/tests/grade.py:53；Task/trusted/numerical_worker.py:50 | 区分质量/内存 Hard Gate 与格式、超时、基础设施异常；只有已完成质量评估的无效候选写 -1。未出分异常记录 score=null 并按 Harness 异常通道失败，不产生正常 reward 哨兵。 给出正常有限分数、质量失败、格式错误、队列失联/超时四分支的状态/reward 对照及更新源码；异常不得被收集成普通 0 或 -1 排名。 |
| H05 | pass | 已有native NOP/Oracle/selected Trial使用Task、指定custom provider，verifier.disable=false、separate，结果没有exception。配置确实支撑该旧拓扑运行，不等同目标内场接受。 证据：expert_evidence/current-package-nop/trial/config.json；expert_evidence/current-package-nop/trial/result.json；expert_evidence/release-full-validation/oracle/trial/result.json | 无需核心修复。  |
| H06 | pass | 同一NOP Trial config/result/reward/stdout/log齐全，nop+separate、正常结束、reward=-0.000337629593978607相符，Task48逐文件SHA均匹配当前ZIP；已有证据通过，非本次复跑。 证据：expert_evidence/current-package-nop/trial/config.json；expert_evidence/current-package-nop/trial/result.json；expert_evidence/current-package-nop/trial/verifier/reward.txt；expert_evidence/current-package-nop/trial/verifier/test-stdout.txt；expert_evidence/current-package-nop/trial/trial.log；expert_evidence/current-package-nop/trial_task_binding.json | 无需核心修复。  |

| 项目 | 状态 | 结论及证据 | 最小修改与验收 |
|---|---|---|---|
| QA01 | pass | 题面八部分齐全，明确固定预训练模型和可读 published 源；未给出作者 Reference 的七个缓存节点或解法配置。 证据：Task/instruction.md | 无需核心修复。  |
| QA02 | pass | 评分 (B-T)/B，U=0ms 是预先声明非负耗时理想下限；有效结果连续、单调、无裁剪，同语义 NOP 的实测噪声保留。 证据：Task/trusted/full_decoder_evaluator.py:193；Task/trusted/full_decoder_evaluator.py:212；Task/environment/public_eval/protocol/benchmark.json:95 | 无需核心修复。  |
| QA03 | manual | Reference 源码、入口、全部正式 seed/result 与非空日志齐全；formal 分数0.4227668有效。尚未核验公开 Dev 下同一 Reference 自验，不要求无训练任务伪造 model/。 证据：Task/solution/reference_method.py；Task/solution/solve.sh；optimization_evidence/reference_runs/seed_101/result.json；expert_evidence/release-full-validation/oracle/result.json | 补作者 Reference 同一源码/graph 在公开 Dev 协议下的正式成对结果；明确公开与 Hidden 分别使用的案例、重复数、聚合和 B/R/U，现有 Hidden 全量结果继续保留。 公开 Dev 的原始阶段/重复记录、全部声明 seed、结果与日志，绑定当前 Reference；复算归一化 [0.15,0.8] 和 ≥3σ。若已有结果可直接补入，无需为本静态审查自动启动运行。 |
| QA04 | pass | GPU延迟有随机波动；三 seed B 样本标准差180.5719465ms，改善24703.3128611ms，≥541.7158394ms；不是用标准误或配对差方差。 证据：expert_evidence/release-full-validation/oracle/raw-stage-times.json；optimization_evidence/comparison_summary.json | 无需核心修复。  |
| QA05 | fail | DAG、shape、cache和质量门有真实 guard；但 task.toml 声明 storage_mb、no-network、gpus=0，实际依赖远程GPU和额外broker，违反当前内场资源/单容器边界。 证据：Task/task.toml:16；Task/task.toml:24；Task/task.toml:38；Runtime/hybrid_harbor_environment.py:95；Runtime/hybrid_harbor_environment.py:137 | 将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。 删除 storage_mb 与禁网声明，按现有平台联网能力配置。 提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。 |
| QA06 | fail | test.sh→grade.py 有单命令和标量出口；但 grade.py 全部异常都写 -1，worker 异常统一 reward=0，把无评估分数的基础设施/格式失败与正常评分哨兵混用。 证据：Task/tests/grade.py:53；Task/trusted/numerical_worker.py:50；Task/tests/test.sh | 区分质量/内存 Hard Gate 与格式、超时、基础设施异常；只有已完成质量评估的无效候选写 -1。未出分异常记录 score=null 并按 Harness 异常通道失败，不产生正常 reward 哨兵。 给出正常有限分数、质量失败、格式错误、队列失联/超时四分支的状态/reward 对照及更新源码；异常不得被收集成普通 0 或 -1 排名。 |
| QA07 | pass | 仅审题面：描述六个 withheld prompts 和质量门，没有 Hidden embedding、标签或可恢复答案。 证据：Task/instruction.md | 无需核心修复。  |
| QA08 | pass | 仅审题面：未披露作者 Reference 代码、七个 prefix 节点清单或正式分数。通用允许复用不等于提供具体答案。 证据：Task/instruction.md | 无需核心修复。  |
| QA09 | pass | test.sh及grade.py清理既有reward；write_reward写pending再replace，候选仅JSON，不运行其Python。 证据：Task/tests/test.sh:3；Task/tests/grade.py:12；Task/tests/grade.py:21 | 无需核心修复。  |
| QA10 | pass | 只接收受限JSON图；固定权重名字、算子/字段白名单及只读Agent镜像边界；Hidden重新校验SHA，候选不导入高权限grader。静态通过不声称完整物理认证。 证据：Task/trusted/tensor_dag.py:26；Task/tests/grade.py:24；Runtime/hybrid_harbor_environment.py:95 | 无需核心修复。  |
| QA11 | pass | 八部分均含实质目标、输入、指标、可改范围、禁止边界、提交JSON、Dev迭代和最终恢复要求。 证据：Task/instruction.md | 无需核心修复。  |
| QA12 | pass | ZIP成员清点未见 .git 目录，不需要推断源仓库历史已清理。 证据：@inventory | 无需核心修复。  |
| QA13 | pass | 三个正式seed六组result全部存在且角色/种子相符；4608阶段→18重复→六case中位数→三seed平均全部精确对上，质量门通过。 证据：expert_evidence/release-full-validation/oracle/raw-stage-times.json；optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_303/result.json；optimization_evidence/comparison_summary.json | 无需核心修复。  |
| QA14 | pass | 公开图接口允许组合新方法；例如融合QKV后重排attention contraction/编译边界是题面未给定具体解法。Reference仅0.4228，留有空间；两条候选0.7527/0.7260属额外结果。 证据：Task/trusted/tensor_dag.py；Trajectories/GPT/selected-method/method.py；Trajectories/Seed/selected-method/method.py；expert_evidence/current-release-comparison.json | 无需核心修复。  |
| QA15 | fail | 已有四个A100完整native Trial约24–41分钟，未证实超2h；但当前仍是A100远程拓扑，TOML无H20/L20资源配置，不能把尚未迁移的实现当成仅缺换卡耗时。 证据：Task/task.toml；Task/environment/public_eval/protocol/benchmark.json:104；expert_evidence/current-package-nop/trial/result.json；expert_evidence/release-full-validation/oracle/trial/result.json | 将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。 提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。 |
| QA16 | pass | 适用非训练短迭代7h例外：保守不重叠区间复算GPT25828.6644秒、Seed27860.4020秒；已知排除区间交叠0。多轮方法闭环、下一方向及各自独立最终复验齐全；不声称10h已证实。 证据：expert_evidence/original-source-method-loops/conservative-clock-source-accounting.json；expert_evidence/original-source-method-loops/gpt-actual-method-feedback-windows.json；expert_evidence/original-source-method-loops/seed-actual-method-feedback-windows.json；expert_evidence/release-full-validation/gpt-selected/trial/result.json；expert_evidence/release-full-validation/seed-selected/trial/result.json | 无需核心修复。  |
| QA17 | fail | 显式separate、两context COPY/入口和artifacts一致；已有同包NOP成功。但custom adapter强依赖额外broker容器、Compose override及远程A100，当前V4单容器自身GPU部署未满足。 证据：Task/task.toml；Runtime/hybrid_harbor_environment.py:107；Runtime/hybrid_harbor_environment.py:137；Runtime/hybrid_relay.py:102 | 将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。 提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。 |
| QA18 | fail | GPT102轮/Seed59轮齐全，但GPT48–102轮有额外字段，45轮成功status=COMPLETED而非ok；两轨161条时间均不合指定格式，长摘录替代1–3句摘要，GPT后段摘要仅实验标签。 证据：expert_evidence/trajectory_codex.json#rounds[47:102]；expert_evidence/trajectory_codex.json#rounds[101]；expert_evidence/trajectory_seed.json#rounds[0:59] | 保留原始轨迹另存；生成恰八字段规范版，成功状态统一ok，时间统一为YYYY-MM-DD HH:mm:ss并在摘要声明时区；每轮写1–3句实际方法与结果，来源/hash/命令放旁表。明确唯一最终候选及公开Dev选择依据，提供等价best_method交付目录。 提交规范化映射与两条原值保真轨迹；逐轮核对状态、失败/null、原始Dev分数、retained_best及最终移交候选一致。 |
| QA19 | pass | 题面明确采样101/202/303及B-T≥3σ_B，样本标准差n-1；不把固定采样seed等同确定性延迟。 证据：Task/instruction.md#Task Setting；Task/instruction.md#Objective and Metrics | 无需核心修复。  |
| QA20 | pass | 可改范围、shell/Python工具和网络策略明确；本项检查是否说明，禁网与远程资源策略的不合当前内场已在核心配置项处理。 证据：Task/instruction.md#Allowed Scope；Task/instruction.md#Hard Boundaries | 无需核心修复。  |
| QA21 | manual | run_summary缺标准task_binding.hash_profile/task_content_sha256。自定义Task48 manifest及NOP逐文件哈希确有来源且48项匹配，不能冒充官方四项指纹。 证据：expert_evidence/run_summary.json；expert_evidence/current-package-nop/trial_task_binding.json | 保留当前原值，取得真实发布的task_fingerprint.py后按四项范围计算并补task_binding；勿改历史哈希或自造兼容算法。 官方脚本版本/SHA及四项指纹输出置于计算范围之外；历史证据关联说明。 |

## 五类非阻断待补
- QA21：non_blocking_pending。标准四项指纹字段/官方算法验收未齐；现有Task48与NOP逐件匹配。 后补动作：按真实发布脚本补标准四项字段，保留旧值。 验收：脚本版本/SHA、四项指纹输出和版本对应。
- G02/G03 换卡影响：non_blocking_pending。已有成对数值为真实A100；换卡影响未核，不推翻已有Baseline公平性与数值改善。 后补动作：后续补H20/L20影响说明或已有测量。 验收：设备保真、同协议补充结果；不强制当前全量重跑。
- QA15 换卡耗时：non_blocking_pending。H20/L20耗时未实测，A100 native单轮均小于2h。配置/远程架构问题另外是核心失败。 后补动作：目标资源部署修复后可后补耗时。 验收：旧卡实测与目标估算/后续实测分列。
- QA17/H06 NOP：already_supported。已有同包NOP记录完整；本次未实跑。 后补动作：迁移后如已有平台记录可补入。 验收：同一Trial配置、result/reward/log与版本绑定；缺口可后补。
- QA03/QA13 换卡重算：non_blocking_pending。不要求仅因换卡重算已有全部seed或归一化；公开Dev Reference缺证是不同的核心范围问题。 后补动作：保留A100正式源值，后续补迁移对照。 验收：保真设备及对应协议/版本。

## 工具缺口与运行边界
本次是local-v4-content-review，不是官方0.3.10发布工具输出。真实官方格式/GPU/四项指纹脚本未取得，官方工具验收待补；未运行旧脚本或改版本字符串。未执行/导入提交代码、Docker/Verifier/安装/训练/外部模型，未反序列化模型。本次未做AIDP/Harbor或12h动态验收；已有同版本NOP仅称材料支持其历史custom拓扑成功。
已读题面/源码未发现需回显的实际秘密；完整逐件二进制隐私扫描由主控执行，本报告不背书整包隐私。没有读取包外历史QA/作者修复说明或同伴结论，包内旧QA未用作依据。

## 具体风险（最多三条）
- 三seed样本标准差同时包含不同prompt/采样case差异与计时波动，按声明协议达标不等于候选间显著性；GPT/Seed不同成对工作不能直接宣称二者显著优劣。
- eager/Reference的收益绑定decoder CUDA-event范围，未覆盖FLUX渲染、首次编译或端到端输送；现有题面有披露，推广需另证。
- 保守活动下界支持7h例外，完整有效时长仍未知；不能把多日wall span或practical_estimate改写成已证实10h。

## 可复制给专家的退回说明
Bifrost 完整题包退回说明（独立静态复核）
结论：NOT READY。研究优化面成立，Baseline在材料范围内合理，正式Hidden全量B/R改善强；以下核心交付问题仍须处理。

1. 执行架构与当前平台契约不符。Task/task.toml的gpus=0/storage_mb/no-network与Runtime/hybrid_harbor_environment.py:107、137显示实际靠Compose override、额外broker容器和远程A100 worker。将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。

2. 失败评分分类需修复。Task/tests/grade.py:53–62全部异常写-1，Task/trusted/numerical_worker.py:50–52异常写reward=0；未完成评估不能当普通分数。区分质量/内存 Hard Gate 与格式、超时、基础设施异常；只有已完成质量评估的无效候选写 -1。未出分异常记录 score=null 并按 Harness 异常通道失败，不产生正常 reward 哨兵。给出正常有限分数、质量失败、格式错误、队列失联/超时四分支的状态/reward 对照及更新源码；异常不得被收集成普通 0 或 -1 排名。

3. 规范轨迹与最终交付。expert_evidence/trajectory_codex.json第48–102轮多字段，45轮COMPLETED而非ok；两轨全部161条time需统一指定格式，摘要需1–3句说明实际方法而不是长原话/标签。保留原始记录，额外来源放旁表，输出恰八字段规范版，分数/null/失败和best标记按实际证据保真。两条selected可继续作证据，另明确唯一最终方法及公开Dev选择依据，提供best_method或明确等价路径。验收查逐轮映射、最终恢复与移交SHA。

4. 公开Dev Reference证据待核。optimization_evidence的全部正式结果来自formal/Hidden，未定位同一作者Reference公开Dev成对自验。补作者 Reference 同一源码/graph 在公开 Dev 协议下的正式成对结果；明确公开与 Hidden 分别使用的案例、重复数、聚合和 B/R/U，现有 Hidden 全量结果继续保留。公开 Dev 的原始阶段/重复记录、全部声明 seed、结果与日志，绑定当前 Reference；复算归一化 [0.15,0.8] 和 ≥3σ。若已有结果可直接补入，无需为本静态审查自动启动运行。

现有正式数据无需因以上文字/格式修订自动重跑：B=58432.48055013021ms，R=33729.16768900553ms，归一化0.422766800733905，σ_B=180.57194646667463ms，改善136.8059σ。全部3seed、18重复、4608阶段来源完整且静态复算相符。
两轨保守活动下界7.175/7.739小时且满足无训练短迭代例外证据；本次不要求补凑10小时，也不将多日wall span当有效时长。已有NOP48项Task文件与当前ZIP一致，不因NOP/指纹放宽项单独退回。
标准四项指纹、H20/L20换卡影响/耗时/重算及迁移后的NOP关联可按2026-10-21口径后补；架构迁移本身不属于该放宽。官方0.3.10与独立脚本验收待补，本报告不宣称平台通过。
