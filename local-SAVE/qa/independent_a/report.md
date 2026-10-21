# SAVE PBMC IFNB：独立当前 V4 静态复核 A

核心结论：**不通过 / NOT READY**。研究优化面、Baseline、公平五 seed 改善及两轨时长/八字段材料在所交证据范围内通过；阻断来自当前运行拓扑/配置与可达非有限 reward 写入路径。未以指纹、迁卡补证或 NOP 缺口单独判退。

输入：`SAVE_PBMC_IFNB_TASK.zip`；SHA256：`487dfcc7e7dfebdd261ee7f59eb2098f0a89308557c07a9af6ad65d6a0b6aad4`。独立 session_id=`save_a`；configured_model=`gpt-6.1-sol`（派发配置）；runtime_reported_model=`not_exposed`；runtime_version=`not_exposed`。本报告为 `local-v4-content-review`，不是官方 0.3.10 工具产物。

## 优化面和实际调用边界

接口允许在 method.py 实现 build_model/train_model/predict，开放模型模块、条件编码、目标/更新规则和推理；只固定数据、缩放/指标、五 seed、文件/资源/重载门。实际载入候选 Python 可执行方法，未被常量 AST guard 收窄。

题面入口为固定 solve.sh → grader → source-only queue → 可信 evaluator/supervisor → chroot 中候选方法。候选三callable实际被载入，代码不仅可写超参；标签只在候选结束后由可信父进程读取。现有实际执行依赖额外 broker 与远端A100，当前内场兼容性另见实现表。

## Baseline 来源、训练和预算

Starter 与正式 B 为同一 condition-dependent VAE：38.02M 参数、gene-block 条件模型，fresh 初始化，AdamW，batch=1024、lr=1e-4、weight_decay=2.5e-5、200 epoch，控制细胞直接向 stimulated 条件解码后逆缩放。源码来源/commit 由包内 provenance 声明；本次不作包外 upstream 真实性认证。

相同 VAE、同训练配方、每 seed 独立训练；仅推理改为 max(原始控制表达 + stimulated 解码 - control 解码, 0)，各次条件解码重置共同推理 seed，使条件残差替代直接生成。Reference 与最终 Agent 最佳方法是不同角色。

角色：Starter=正式Baseline=评分B来源；sanity只作诊断。十条原始日志均完成200/200，B/R所有方法源SHA、模型字节SHA与result一致，模型各152124106字节；独立reload PID和重载日志已交，重载指标差为0。本次只流式校验权重字节，不反序列化，不复跑。200 epoch对应的约4200次更新是代码/数据推导，不伪称日志逐步计数。

Baseline 专项总评：**在已提供材料范围内合理**。

| 项目 | 状态 | 事实与证据 |
|---|---|---|
| B01 | pass | 正式 Baseline 等于未经修改 Starter，具有可解释的条件 VAE；包内朴素 identity/pooled-delta 诊断均明显更差，未见代表性不足的直接证据。 证据：`workspace/harbor_task/environment/starter/method.py:7-41`；`expert_evidence/source_provenance.json:4-17`；`expert_evidence/implementation_checks/naive_dev_diagnostics.json:5-20` |
| B02 | pass | 所有十条原始 run.log 均到 200/200；训练循环执行真实 backward/optimizer.step，正常结束，checkpoint/reload 记录齐全。未见专门截短 B 或错误加载。 证据：`workspace/harbor_task/environment/starter/conditional_backend/VAE_train.py:164-218`；`workspace/harbor_task/environment/starter/conditional_backend/VAE_train.py:225-311`；`optimization_evidence/baseline_runs/seed_17/run.log`；`optimization_evidence/reference_runs/seed_17/run.log` |
| B03 | pass | B/R 相同数据哈希、五 seed、200 epoch/1024、900 秒上限及 A100/4线程；B 训练359.22–361.26秒，R 362.06–365.99秒。历史科学对照公平；当前部署拓扑另行不通过。 证据：`optimization_evidence/训练证据说明.md:5-13`；`optimization_evidence/baseline_runs/seed_17/result.json#protocol`；`optimization_evidence/reference_runs/seed_17/result.json#protocol` |
| B04 | pass | B 使用披露的默认配方，R 保持所有训练参数；改进变量是推理残差，未见给 B 异常超参或不同选择集的直接证据。 证据：`workspace/harbor_task/environment/starter/method.py:7-29`；`workspace/reference/method.py:7-49`；`optimization_evidence/训练证据说明.md:5-9` |
| B05 | pass | 17/29/43/71/97 完整成对；seed 进入初始化、训练随机过程和 seed+10000 推理。各训练PID/重载PID分离、跨seed模型哈希不同；同seed B/R同配方可得相同权重。 证据：`workspace/harbor_task/environment/starter/conditional_backend/model.py:178-182`；`workspace/harbor_task/environment/starter/conditional_backend/VAE_train.py:164-169`；`optimization_evidence/comparison_summary.json:9-71`；`optimization_evidence/baseline_runs/seed_17/execution/execution_receipt.json` |
| B06 | pass | 包内 Dev sanity：identity MSE .01916361、pooled delta .01087857、equal-context delta .01425165，均差于正式 B .00365682。诊断标非正式，不充当评分锚点。 证据：`expert_evidence/implementation_checks/naive_dev_diagnostics.json`；`optimization_evidence/comparison_summary.json#statistics` |
| B07 | pass | 训练、数据/尺度、指标、完整seed均固定，Reference 源码差异只在条件残差推理；无需为联合推理策略另强制所有单因素消融。 证据：`workspace/reference/method.py:43-49`；`optimization_evidence/训练证据说明.md:5-11`；`optimization_evidence/comparison_summary.json#validity` |
| B08 | pass | 来源、角色、默认配方、研究范围和未复现整篇论文的限制已披露；无证据支持“故意削弱”推断。 证据：`expert_evidence/source_provenance.json:7-17`；`expert_evidence/专家作业说明文档.md:7`；`optimization_evidence/训练证据说明.md:5-17` |

五维事实：
- training_sufficiency：十条日志均显示200/200；无early-stop/异常终止。约21 batch/epoch、约4200 updates仅为依数据形状和DataLoader推导，非独立观察的step计数。
- method_representativeness：条件VAE默认配方；包内三种简单锚点更弱，未要求SOTA。上游官方来源属包内声明。
- budget_comparability：同数据/缩放/五seed/200epoch/单卡900秒；实测成本相近，R未获额外训练预算。
- implementation_correctness：梯度与优化器更新可沿源码定位；权重文件、fresh/reload独立PID、重载指标与原值一致。
- selection_integrity：初始三seed后双方共同追加71/97；完整五seed未抽样；B锚点等于正式B均值。

## Reference 实际方法及全部正式成对数值

| training seed | Baseline MSE | Reference MSE | B−R | 原始结果路径 |
|---|---:|---:|---:|---|
| 17 | 0.0035213459653151711 | 0.0025195494768602486 | 0.0010017964884549225 | `optimization_evidence/baseline_runs/seed_17/result.json`；`optimization_evidence/reference_runs/seed_17/result.json` |
| 29 | 0.0033021915259081455 | 0.0025885801697712648 | 0.00071361135613688068 | `optimization_evidence/baseline_runs/seed_29/result.json`；`optimization_evidence/reference_runs/seed_29/result.json` |
| 43 | 0.0037422385312627805 | 0.0026323001823446378 | 0.0011099383489181426 | `optimization_evidence/baseline_runs/seed_43/result.json`；`optimization_evidence/reference_runs/seed_43/result.json` |
| 71 | 0.0039580559147975088 | 0.0026512386258838146 | 0.0013068172889136942 | `optimization_evidence/baseline_runs/seed_71/result.json`；`optimization_evidence/reference_runs/seed_71/result.json` |
| 97 | 0.0037602680370891666 | 0.0027709962089949437 | 0.00098927182809422287 | `optimization_evidence/baseline_runs/seed_97/result.json`；`optimization_evidence/reference_runs/seed_97/result.json` |

均值 B=0.0036568199948745547，R=0.0026325329327709819；n−1 样本标准差 σB=0.00025142367973954213，σR=9.2492084357402582e-05。Δ=0.0010242870621035727，3σB=0.00075427103921862633；Δ/σB=4.07394825803。U=0（MSE数学下界），归一化改善=0.28010322179905667，即 28.010322180%；剩余到U=10.470505σB。五seed质量门均通过，属于3–5σ可接受证据。

这是五次独立随机训练，非同一训练后模型的重复评估；同seed双方相同训练而权重相同合理，跨seed权重不同。固定seed不使总体评估变成确定性。历史A100设备不改标为H20/L20。

| 角色/seed | 实际训练秒 | 完整执行秒 | 重载秒 | 日志完成 |
|---|---:|---:|---:|---|
| baseline/17 | 360.633858 | 376.296416 | 3.960896 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| reference/17 | 364.140445 | 380.269231 | 4.056548 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| baseline/29 | 359.219803 | 374.224877 | 3.824421 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| reference/29 | 362.062423 | 378.304558 | 4.058240 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| baseline/43 | 359.220861 | 374.256949 | 3.878998 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| reference/43 | 365.994668 | 382.199520 | 4.083935 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| baseline/71 | 361.256119 | 377.410397 | 3.869025 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| reference/71 | 364.401687 | 381.322829 | 4.155123 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| baseline/97 | 360.860195 | 376.322278 | 3.964011 | 200/200，exit/reload=0，权重/方法SHA匹配 |
| reference/97 | 363.524746 | 379.382326 | 4.042847 | 200/200，exit/reload=0，权重/方法SHA匹配 |

## 三项内容门

| 项目 | 状态 | 判断与证据 |
|---|---|---|
| G01 | pass | 接口允许在 method.py 实现 build_model/train_model/predict，开放模型模块、条件编码、目标/更新规则和推理；只固定数据、缩放/指标、五 seed、文件/资源/重载门。实际载入候选 Python 可执行方法，未被常量 AST guard 收窄。 证据：`workspace/harbor_task/instruction.md:16-33`；`expert_evidence/runtime_integration/numerical_sources/gpu_jail_entry.py:138-155`；`workspace/reference/method.py:43-49` |
| G02 | pass | 在已交静态材料范围内 Baseline 合理；五维事实与八项专项均支持健康、公平的正式对照。历史 A100 科学证据保留，迁卡影响不单独阻断。 证据：`workspace/harbor_task/environment/starter/method.py`；`optimization_evidence/训练证据说明.md`；`optimization_evidence/comparison_summary.json` |
| G03 | pass | 五seed均有效且逐seed正向改善；Δ=.00102428706210，σB=.000251423679740，Δ/σB=4.073948；归一化=.280103222。满足3σ与[.15,.8]，属可接受而非≥5σ强证据。 证据：`optimization_evidence/comparison_summary.json:16-103`；`optimization_evidence/baseline_runs/seed_17/result.json`；`optimization_evidence/baseline_runs/seed_29/result.json` |

## 两条轨迹、有效时长和格式

- Codex + gpt-6.1-sol：八字段记录 52 轮；run_summary有效轮次 37；计时表 37 行、正credit Native行 37；有效 36508.691002607 秒（10.141303056小时）。nearest-context transfer逐步加入donor pooling、hurdle/detection与intensity shrinkage、transfer rule及query适配；失败12轮。 run_summary的effective_rounds、Native工作turn数与八字段实验/恢复记录数不是同一口径，建议说明映射。 最终 Public MSE=0.001351913106690589、归一化=0.630303622113。源：`expert_evidence/trajectory_codex.json` 与该轨clock/public-experiments。
- Codex + doubao-seed-2-1-turbo-260628：八字段记录 376 轮；run_summary有效轮次 159；计时表 427 行、正credit Native行 154；有效 36639.615034819 秒（10.177670843小时）。nearest-context shift逐步加入多context ensemble、signed log-ratio与上下调非对称；69次有分Public实验，包含大量工具/传输恢复失败。 run_summary的effective_rounds、Native工作turn数与八字段实验/恢复记录数不是同一口径，建议说明映射。 最终 Public MSE=0.0015912749345400839、归一化=0.564847343656。源：`expert_evidence/trajectory_seed.json` 与该轨clock/public-experiments。

所有成功Public分数都能逐值关联包内实际实验；八字段恰全、失败score=null且保留失败原因；retained_best顺MSE下降。方法独立，最终canonical best_method为GPT轨，另一轨留在独立最终源码附件。各轨最终SHA与新Actor/独立Verifier复验记录相符。私有校准结果不是平台官方Hidden结论，不用于轨内选优。

计时重新逐行求和，无计入区间重叠、无credit超过起止减披露扣除的上界；普通推理/分析/训练计入，排队/下载/阻塞/协议故障扣除。无需另造全调用记录；本次也未做12h压力测试。

格式：三目录、完整正式seed、训练模型/重载、唯一best_method均齐。source task为root-context加明确export/native-context与私有注入；该等价路径不因目录名扣错。建议解释几种轮次数的统计口径；重复字段/原始失败材料可归并但不强制删减。annotation中的审核/运行状态建议归run_summary。

## Harbor 与实现检查

静态检查已覆盖全部要求；本次 runtime=not_run_by_reviewer。已有0.23 NOP与最终复验只支持其真实历史混合拓扑，不冒称当前平台验收。

| 项目 | 状态 | 事实、影响和修复 |
|---|---|---|
| H01 | pass | 任务根、双Dockerfile/test.sh及依赖齐全；Hidden 不为空口号：专家私有供应有controls/targets和contract，注入器仅写导出tests且校验Agent context不变。 证据：`workspace/harbor_task/task.toml`；`workspace/harbor_task/tests/test.sh`；`expert_evidence/platform_adapter/inject_private_assets.py:20-43`；`expert_evidence/private_verifier_provisioning/private_contract.json` |
| H02 | fail | 显式separate正确，但两环境仍声明storage_mb/no-network，provider强制只支持NO_NETWORK；这些不是当前内场支持的任务条件。包内0.23原生schema校验声明不覆盖V4兼容性。 证据：`workspace/harbor_task/task.toml:12-36`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:30-46`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:123-124` 最小修改：删除 task.toml 中两处 storage_mb 与 no-network/network_mode 限制，更新题面和 provider，使任务在当前内场实际联网条件下仍保持私有数据不可见；不能仅删文字而保留强制禁网代码。 验收：新的 task.toml、instruction、provider 与目标平台支持字段一致；提供保留 Hidden/冻结面隔离的源码路径。 |
| H03 | fail | 实际provider将gpus改为0，另启动broker容器并以SSH送A100数值工作；不满足每sandbox单容器、直接使用自身GPU的当前核心要求。源码root-context有明确export适配，不因目录名另判失败。 证据：`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60-62`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:118-153`；`expert_evidence/runtime_integration/portable_runtime_v4/model_relay.py:59-72`；`expert_evidence/platform_adapter/export_native.py:22-32` 最小修改：将 Public/Dev 数值执行放入 Agent 自身分配的 GPU 容器；将私有评分及受限候选执行放入独立 Verifier 自身 GPU 容器。移除额外 broker 容器、任务专用 Compose 依赖和 SSH/远端 GPU 队列；同步构建、入口、资源与 artifacts，保留历史真实 A100 记录。 验收：重交单容器 Agent/Verifier 的 Dockerfile/可信执行调用链及 H20/L20 资源配置，给出公开Dev和最终受限评分入口；已有记录需清楚区分历史混合拓扑与新配置。新卡耗时/重算及新NOP依当前放宽口径可后补。 |
| H04 | fail | test.sh有真实reward入口且原子替换，但仅按COMPLETE/all_seeds写score，未检查有限性。有限大预测可令MSE平方溢出并传播为无限score；该边界不符合有限reward合同。 证据：`workspace/harbor_task/tests/test.sh:3`；`workspace/harbor_task/tests/grader.py:17-30`；`expert_evidence/runtime_integration/numerical_sources/independent_metrics.py:14-22`；`expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:86-94` 最小修改：可信数值父进程对各seed MSE、聚合raw_metric和归一化score逐级 math.isfinite；写reward前再次校验有限标量，并用 allow_nan=False 序列化。溢出归为明确无效/数值失败，score=null，清理旧reward，不标COMPLETE。 验收：提交有限大预测导致平方溢出的边界用例结果、正常候选结果、无reward/失败分类记录；证明 NaN/Inf 不会进入 reward.txt/reward.json。本次仅静态定位，未运行该用例。 |
| H05 | pass | 已有两次NOP/最终复验均选择同题路径、独立provider且Verifier未禁用；job.example明示仅配置示例。当前拓扑不符合V4由环境项裁定，不误写为空选择。 证据：`expert_evidence/nop_trial/config.json`；`expert_evidence/nop_trial/result.json#config.verifier.disable`；`expert_evidence/runtime_integration/job.example.json`；`expert_evidence/final_revalidation/gpt/config.json` |
| H06 | manual | 两次历史NOP均为nop/separate、finished_at存在、exception=null、reward=0并有五seed评估。provider版本不同于当前v4且缺官方四项指纹精确绑定，当前版关联记非阻断待补；不是未跑过NOP。 证据：`expert_evidence/nop_trial/result.json#verifier_environment_mode`；`expert_evidence/nop_trial/verifier/result.json`；`expert_evidence/nop_trial/trial_task_binding.json`；`expert_evidence/portable_runtime_nop/result.json` 最小修改：保留现有历史NOP原值/版本；以后取得当前任务精确版本对应的同一次Trial配置、result、reward及非空日志，或提供可核对的版本绑定。 验收：同次NOP完整链路及真实版本映射；只作为非阻断后补，不要求本次启动新运行。 |

| 检查项 | 状态 | 事实与证据 |
|---|---|---|
| QA01 | pass | 题面具八部分，未直接给论文标题/arXiv/repo或参考解；一般条件模型背景不构成答案泄露。 证据：`workspace/harbor_task/instruction.md:1-39` |
| QA02 | pass | 实际可信父进程按(B-meanMSE)/(B-U)连续、单调、未裁剪；Dev B=正式均值、U=0，私有split独立校准B。有限性边界单列评分入口缺陷。 证据：`expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:86-93`；`workspace/harbor_task/tests/anchors.json`；`expert_evidence/private_verifier_provisioning/private_contract.json#hidden_anchor` |
| QA03 | pass | Reference源码/入口、全部5份result和非空原始训练日志、152124106字节权重/索引/独立reload齐全；流式SHA均匹配，归一化.280103222在范围内。 证据：`workspace/reference/method.py`；`workspace/harbor_task/tests/train_eval.py:21-90`；`optimization_evidence/reference_runs/seed_17/result.json`；`optimization_evidence/reference_runs/seed_97/model/reload.log`；`optimization_evidence/comparison_summary.json` |
| QA04 | pass | 训练随机性不能当确定性；按五个B原始值以n-1复算σB及Δ，4.073948σ满足3σ。不是用标准误/配对差方差替代。 证据：`optimization_evidence/comparison_summary.json#paired_results`；`optimization_evidence/comparison_summary.json#statistics` |
| QA05 | fail | 大小/预测/重载/时间/冻结检查均有实现；但task和provider将storage_mb、强制禁网当环境条件，当前内场不支持，不能以本地Docker限制等同当前可部署约束。 证据：`workspace/harbor_task/task.toml:15-18`；`workspace/harbor_task/task.toml:31-34`；`workspace/harbor_task/instruction.md:19-24`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:30-46` 最小修改：删除 task.toml 中两处 storage_mb 与 no-network/network_mode 限制，更新题面和 provider，使任务在当前内场实际联网条件下仍保持私有数据不可见；不能仅删文字而保留强制禁网代码。 验收：新的 task.toml、instruction、provider 与目标平台支持字段一致；提供保留 Hidden/冻结面隔离的源码路径。 |
| QA06 | fail | 单命令入口和失败分类存在，正常已有分数有效；但无finite检查的COMPLETE可把溢出MSE/score写reward，未保证有限主标量。 证据：`workspace/harbor_task/tests/test.sh`；`workspace/harbor_task/tests/grader.py:26-30`；`expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:86-94` 最小修改：可信数值父进程对各seed MSE、聚合raw_metric和归一化score逐级 math.isfinite；写reward前再次校验有限标量，并用 allow_nan=False 序列化。溢出归为明确无效/数值失败，score=null，清理旧reward，不标COMPLETE。 验收：提交有限大预测导致平方溢出的边界用例结果、正常候选结果、无reward/失败分类记录；证明 NaN/Inf 不会进入 reward.txt/reward.json。本次仅静态定位，未运行该用例。 |
| QA07 | pass | 仅查instruction，未披露Hidden细胞/目标标签/可恢复答案；说明存在私有context并禁止访问合法。 证据：`workspace/harbor_task/instruction.md:7`；`workspace/harbor_task/instruction.md:19-24` |
| QA08 | pass | 仅查instruction，未披露Reference的条件残差代码、具体配置/结果或直接答案。 证据：`workspace/harbor_task/instruction.md:16-39` |
| QA09 | pass | 评分前删除reward.txt/json；result和reward经pending临时文件replace，存在可信侧清理/原子替换实现。 证据：`workspace/harbor_task/tests/grader.py:17-29` |
| QA10 | pass | 公开冻结面Docker权限a-w；可信数值preflight逐项查frozen_assets及私有资产/runner；候选chroot UID/seccomp且不接收标签，父进程退出后读取target。 证据：`workspace/harbor_task/environment/Dockerfile:11`；`expert_evidence/runtime_integration/numerical_sources/gpu_fixed_evaluator.py:12-30`；`expert_evidence/runtime_integration/numerical_sources/gpu_jail_entry.py:48-115`；`expert_evidence/runtime_integration/numerical_sources/gpu_jail_supervisor.py:107-126` |
| QA11 | pass | 八部分都有输入输出、环境、指标公式、可改范围、边界、提交callables、迭代及完成标准等实质内容。 证据：`workspace/harbor_task/instruction.md:3-39` |
| QA12 | pass | ZIP清点没有.git、refs/reflog/tags；未靠清理脚本宣称已清理历史。 证据：`@inventory` |
| QA13 | pass | B/R正式五seed完整同集、原始log与result均有效，每seed MSE降低且所有重载门通过；独立统计与汇总一致。 证据：`optimization_evidence/comparison_summary.json:16-103`；`optimization_evidence/baseline_runs/seed_17/result.json`；`optimization_evidence/reference_runs/seed_97/result.json` |
| QA14 | pass | 开放方法可实现pseudobulk/上下文适配等未在题面具体提示的方向；包内Agent已实做，Reference仅.2801，最终最佳.6303，未饱和。 证据：`workspace/harbor_task/instruction.md:16-33`；`expert_evidence/best_method/method.py:1-5`；`expert_evidence/run_summary.json#agent_execution` |
| QA15 | fail | 已有A100/4线程完整单轮约31–32分钟，未见2h/CPU/显存超限；但当前provider明示硬要求外置A100且把本地GPU设0，尚未配置为本sandbox H20/L20。此为部署选择错误，非仅换卡后耗时缺证。 证据：`optimization_evidence/训练证据说明.md:13`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60-62`；`workspace/harbor_task/instruction.md:24-36`；`workspace/harbor_task/task.toml:16-17` 最小修改：将实际provider与题面设为本sandbox单张H20或L20，删除硬编码外置A100路径，按现有预算保留历史A100实测；新卡性能计时另列待核。 验收：H20/L20实际资源选择与可执行部署配置；新卡时间测量依当前放宽可后补，不能将A100记录改标成目标卡实测。 |
| QA16 | pass | 训练任务按各10h审：逐行求和GPT36508.691s、Seed36639.615s，无计入区间重叠、无超原始跨度扣除上界。下载/等待/传输/故障扣除有表，正常推理/分析可计；未将墙钟或12h存活当有效研发。 证据：`expert_evidence/actual_runtime_evidence/gpt-clock.json#rows`；`expert_evidence/actual_runtime_evidence/seed-clock.json#rows`；`expert_evidence/run_summary.json#agent_execution`；`expert_evidence/专家作业说明文档.md:3` |
| QA17 | fail | separate、候选移交、私有注入与既有NOP存在；当前阻断是额外broker/远程GPU拓扑、禁网/磁盘配置与非有限reward路径。NOP版本待核本身不阻断。 证据：`workspace/harbor_task/task.toml:24-36`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:60-62`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:118-153`；`workspace/harbor_task/tests/grader.py:26-30` 最小修改：将 Public/Dev 数值执行放入 Agent 自身分配的 GPU 容器；将私有评分及受限候选执行放入独立 Verifier 自身 GPU 容器。移除额外 broker 容器、任务专用 Compose 依赖和 SSH/远端 GPU 队列；同步构建、入口、资源与 artifacts，保留历史真实 A100 记录。 删除 task.toml 中两处 storage_mb 与 no-network/network_mode 限制，更新题面和 provider，使任务在当前内场实际联网条件下仍保持私有数据不可见；不能仅删文字而保留强制禁网代码。 可信数值父进程对各seed MSE、聚合raw_metric和归一化score逐级 math.isfinite；写reward前再次校验有限标量，并用 allow_nan=False 序列化。溢出归为明确无效/数值失败，score=null，清理旧reward，不标COMPLETE。 验收：重交单容器 Agent/Verifier 的 Dockerfile/可信执行调用链及 H20/L20 资源配置，给出公开Dev和最终受限评分入口；已有记录需清楚区分历史混合拓扑与新配置。新卡耗时/重算及新NOP依当前放宽口径可后补。 新的 task.toml、instruction、provider 与目标平台支持字段一致；提供保留 Hidden/冻结面隔离的源码路径。 提交有限大预测导致平方溢出的边界用例结果、正常候选结果、无reward/失败分类记录；证明 NaN/Inf 不会进入 reward.txt/reward.json。本次仅静态定位，未运行该用例。 |
| QA18 | pass | 两条轨迹分别52/376轮，每轮恰八字段，40/69条成功分数全部能关联原始Public实验；307条Seed失败保留null/真实恢复原因。最终各自源码SHA和新Actor/Verifier复验可追溯。 证据：`expert_evidence/trajectory_codex.json#rounds`；`expert_evidence/trajectory_seed.json#rounds`；`expert_evidence/actual_runtime_evidence/gpt-public-experiments.json`；`expert_evidence/actual_runtime_evidence/seed-public-experiments.json`；`expert_evidence/final_trajectory_methods/gpt/revalidation.json`；`expert_evidence/final_trajectory_methods/seed/revalidation.json` |
| QA19 | pass | 题面明确训练五seed、推理seed+10000及改善超过Baseline样本3σ；未额外要求题面固定replicate文字。 证据：`workspace/harbor_task/instruction.md:12-14` |
| QA20 | fail | 可改文件与平台coding/terminal工具策略清楚，但网络/外部web禁用是明确任务条件且provider强制禁网，与当前内场网络能力口径冲突。 证据：`workspace/harbor_task/instruction.md:17-24`；`expert_evidence/runtime_integration/portable_runtime_v4/environment.py:39-46` 最小修改：删除 task.toml 中两处 storage_mb 与 no-network/network_mode 限制，更新题面和 provider，使任务在当前内场实际联网条件下仍保持私有数据不可见；不能仅删文字而保留强制禁网代码。 验收：新的 task.toml、instruction、provider 与目标平台支持字段一致；提供保留 Hidden/冻结面隔离的源码路径。 |
| QA21 | manual | run_summary缺task_binding/hash_profile/task_content_sha256。包内有历史自定义冻结指纹，不能猜作官方四项算法；当前缺失单列非阻断待补，不据此重跑或推断模型损坏。 证据：`expert_evidence/run_summary.json`；`workspace/harbor_task/tests/frozen_assets.json`；`expert_evidence/nop_trial/task-source-manifest.json` 最小修改：取得真实发布task_fingerprint.py后在四项范围外生成结果，记录真实版本/profile并填run_summary.task_binding，保留历史指纹原值。 验收：同一最终完整ZIP对应的官方四项指纹、脚本版本/SHA和task_binding；当前允许后补，不以此单独阻断。 |

## 核心修改与验收材料

1. **改造实际运行拓扑并同步当前配置。** `environment.py:60-62` 把gpus改0，`:147-153` 启动额外broker；`model_relay.py:59-72` 以SSH送远端数值工作。将 Public/Dev 数值执行放入 Agent 自身分配的 GPU 容器；将私有评分及受限候选执行放入独立 Verifier 自身 GPU 容器。移除额外 broker 容器、任务专用 Compose 依赖和 SSH/远端 GPU 队列；同步构建、入口、资源与 artifacts，保留历史真实 A100 记录。 重交单容器 Agent/Verifier 的 Dockerfile/可信执行调用链及 H20/L20 资源配置，给出公开Dev和最终受限评分入口；已有记录需清楚区分历史混合拓扑与新配置。新卡耗时/重算及新NOP依当前放宽口径可后补。

2. **删除当前内场不支持的磁盘/禁网任务条件。** `task.toml:15-18,31-34` 与 `environment.py:39-46,123-124` 为明确反证。删除 task.toml 中两处 storage_mb 与 no-network/network_mode 限制，更新题面和 provider，使任务在当前内场实际联网条件下仍保持私有数据不可见；不能仅删文字而保留强制禁网代码。 新的 task.toml、instruction、provider 与目标平台支持字段一致；提供保留 Hidden/冻结面隔离的源码路径。

3. **保证可信主分数有限。** `independent_metrics.py:14-22` 只检查输入有限；例如各预测值为1e200时，均值仍有限但MSE平方可溢出。`gpu_fixed_evaluator.py:86-94` 可继续COMPLETE，`grader.py:26-29` 不检查score有限性即写reward。常量方法能通过同形状/非负和确定性reload；这是静态可达路径，未执行该候选。可信数值父进程对各seed MSE、聚合raw_metric和归一化score逐级 math.isfinite；写reward前再次校验有限标量，并用 allow_nan=False 序列化。溢出归为明确无效/数值失败，score=null，清理旧reward，不标COMPLETE。 提交有限大预测导致平方溢出的边界用例结果、正常候选结果、无reward/失败分类记录；证明 NaN/Inf 不会进入 reward.txt/reward.json。本次仅静态定位，未运行该用例。

## 五类非阻断待补

- fingerprint：缺官方task_binding，历史NOP/provider与当前v4精确绑定未核验。 非阻断后补；不伪造当前哈希。
- hardware_gain：现有B/R真实A100材料；未见H20/L20迁卡影响补证。 已有成对科学结论仍可通过，迁卡影响延期。
- hardware_time：新卡完整单轮耗时未实测。 仅计时缺口非阻断；当前硬要求外置A100的配置错误仍须修。
- nop：已有历史NOP完整链路，不是当前精确版本绑定的独立NOP。 仅版本对应/补全可后补；本次不新跑。
- hardware_rescore：未迁卡重算五seed/锚点。 不强制本次重跑，按已交原值审。

## 工具缺口、运行边界和可见隐私

真实官方0.3.10及独立格式/GPU/指纹工具本次未核验/未执行，工具验收=PENDING；未改版本字符串、未运行旧收集器冒充新版。取得工具后对最终ZIP按真实CLI检查并保存版本/SHA及原始输出。已证实的核心不通过不会被工具缺口遮盖。

本次仅ZIP直接读取文本/配置/指标/日志/成员元数据与权重流式SHA；未导入/执行提交代码、安装依赖、运行Docker/Verifier、训练、调用模型或反序列化模型。日志真实性/目标平台物理隔离与12h稳定性未作独立动态认证；上游来源仅按包内可追溯声明核对。

阅读过的源码发现非官方模型网关端点类型，位置 `expert_evidence/runtime_integration/portable_runtime_v4/model_transport.py:7`；建议外置可信部署配置，确认授权后再外发。此处不回显其值。未做完整附件/二进制隐私扫描，主控单独逐件执行；不以文本未命中代替发布门禁。

风险最多三条：
- 正式Reference为4.074σ证据，满足门槛但未达5σ强证据；保留完整五seed避免夸大普遍收益。
- 无有限标量护栏的MSE/score路径静态可达溢出，需修复后验收。
- 额外broker与远端GPU是现有profile必要依赖，不能仅靠改配置名称满足当前单容器内场。
