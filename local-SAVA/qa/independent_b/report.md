# SAVA 独立本地 V4 内容复核

审查身份：session_id=sava_b；configured_model=gpt-6-astra/high（主控派发凭证）；runtime_reported_model=not_exposed；runtime_version=not_exposed。
输入：SAVA_AutoResearch_submission.zip；ZIP SHA256：`8e950e70c9434f14da4ae64890a90074860a1598d1c4f6f543a9cf69d8165979`。以下路径以ZIP内SAVA_AutoResearch目录为相对根。
只读完整ZIP，未使用旧QA结论、包外作者补充或同伴报告。

## 优化面与实际调用边界
固定图像10维表示上的分层最优传输估值数值加速；以逐例保真约束下的方法耗时为主指标。题面开放可执行方法，AST guard允许函数/类及数值库，实际worker调用value并由父进程计时，非固定常量搜索。
minimize mean_case_method_seconds；逐次NRMSE≤0.01、Spearman≥0.999、top-quarter overlap≥0.98，有限(N,)输出。
边界证据：workspace/harbor_task/instruction.md:15；workspace/harbor_task/instruction.md:23；workspace/harbor_task/tests/evaluator.py:6；workspace/harbor_task/tests/candidate_worker.py:15。函数和类可修改，worker实际调用value；评分器按源快照、AST guard和独立UID运行候选。

## Baseline 专项
总评：合理（在提供材料范围内）
未改Starter=正式Baseline=公开B锚点。以第一对contiguous 1024批建立20类的190个经验Sinkhorn距离，逐批求样本coupling与GeomLoss双势，经中心化校准后按外层batch coupling汇总。共同上游修复早于正式对照；固定encoder仅提供相同输入，无训练。
包内来源声明与实现可核对；未访问包外原论文/仓库，未独立认证官方代码血缘或encoder训练史。

- training_sufficiency：不适用：无训练/微调；10份result均NOT_APPLICABLE，无需checkpoint或学习曲线。冻结encoder训练历史未知已披露，不作为训练充分性证据。
- method_representativeness：包内给出固定官方commit、共同修复与practical算子范围；canonical含可见冗余但作为明确数值算子优化起点合理，不声称效率SOTA。没有同任务同预算更强常用对照证明其不适用。
- budget_comparability：双方同镜像标识、2CPU/2048MiB/0GPU、相同数据/目标/5 seeds、每例9次新进程，同一计时边界。正式单seed wall约57–69秒。
- implementation_correctness：输入、20类标签成本、样本/批层耦合与校准完整；残差检查，540条正式调用均通过保真门。未运行代码，未独立认证外部encoder。
- selection_integrity：全部5 seeds成对、源码哈希各角色恒定、正式源与日志哈希一致；B101干扰试跑排除及串行重跑已披露。无直接挑最差B或异步训练冒充重复的反证。

| 项目 | 状态 | 事实与证据 |
|---|---|---|
| B01 | pass | 正式Baseline是透明canonical/未改Starter；包内说明限定数值效率任务，不能用其年龄或冗余单独判弱化。 证据：expert_evidence/方法与复现说明.md:39；expert_evidence/source_alignment_manifest.json；workspace/harbor_task/environment/starter/method.py |
| B02 | pass | 完整数值路径与540条质量记录正常；无训练，不以缺训练曲线/模型判退。 证据：workspace/harbor_task/environment/starter/method.py:21；optimization_evidence/训练证据说明.md:3；optimization_evidence/baseline_runs/seed_101/run.log |
| B03 | pass | 全部正式双方同资源上限、同镜像、同输入/目标、同计时范围与重复协议；未观察额外预算差。 证据：optimization_evidence/baseline_runs/seed_101/result.json#execution；optimization_evidence/reference_runs/seed_101/result.json#execution |
| B04 | pass | 共同数值修复与参数口径已披露，Reference可改变求解方式；未发现仅Baseline异常参数的直接证据。 证据：expert_evidence/方法与复现说明.md:15；expert_evidence/prerequisites/common_repairs.patch；workspace/harbor_task/environment/starter/method.py；workspace/reference/method.py |
| B05 | pass | 正式101/202/303/404/505全部成对，分别选择六个seed案例，每例9次新进程。算法数值可确定但耗时随机；seed不是训练seed，未强求确定性算法人为使用RNG。 证据：workspace/harbor_task/tests/evaluator.py:83；workspace/harbor_task/tests/candidate_worker.py:15；optimization_evidence/comparison_summary.json |
| B06 | pass | 官方practical透明Starter即基本锚点；NLL/kNN针对旧检测目标，不是当前数值保真效率的同协议更强锚点。 证据：expert_evidence/方法与复现说明.md:17；expert_evidence/方法与复现说明.md:43 |
| B07 | pass | 共同输入/目标/协议/预算一致，改善归属于Reference整体压缩与复用方案；不将共同修复计为独有创新。 证据：optimization_evidence/comparison_summary.json；expert_evidence/方法与复现说明.md:7；workspace/reference/method.py:8 |
| B08 | pass | 来源、角色、参数、限制、encoder未知历史及对照适用范围均披露。 证据：expert_evidence/方法与复现说明.md；expert_evidence/source_alignment_manifest.json；optimization_evidence/训练证据说明.md |

## Reference 与全部正式原始成对数据
带质量权重的farthest-first支撑压缩与3轮Voronoi均值更新；只求跨域100个标签距离，复用批成本矩阵计算双势。既变更分布表示也变更计算图。正式源码为workspace/reference/method.py，不与Agent最终best_method混淆。
这五个seed为固定数据case标识；每例9次独立新进程用于随机耗时测量，不是独立训练，算法数值确定也不使计时确定。

| seed | B原始秒 | R原始秒 | 正向改善秒 |
|---|---:|---:|---:|
| 101 | 0.5309270903332314 | 0.3803967503336025 | 0.1505303399996288 |
| 202 | 0.5180353616645638 | 0.3782085348308707 | 0.139826826833693 |
| 303 | 0.5196528819991121 | 0.3829074791695651 | 0.136745402829547 |
| 404 | 0.523678173830073 | 0.3841831811684339 | 0.1394949926616391 |
| 505 | 0.5151795976635185 | 0.3763024723327059 | 0.1388771253308126 |

B均值=0.5214946210980997，n−1样本标准差=0.006102814476157093；R均值=0.3803996835670356，样本标准差=0.003247775168112226。Δ=0.1410949375310641，3σ_B=0.01830844342847128，Δ/σ_B=23.1196504633；U=0，归一化=0.2705587590413945，满足[0.15,0.8]。R到U的剩余距离为62.331845σ_B。
全部十个result、measurement与原始run.log均逐项读取；每run六例×九次=54条，共540条；每例取九次中位数再对六例求均值，与result完全一致。角色内源码及日志哈希全部吻合，退出码均0、质量全部有效。
质量极值：NRMSE最大=0.007341078344218985；Spearman最小=0.9999418075490896；top-quarter overlap最小=0.9933333333333333。

| 角色/seed | 真实result路径 | 原始日志路径 | log复算值 | wall秒 |
|---|---|---|---:|---:|
| baseline/101 | optimization_evidence/baseline_runs/seed_101/result.json | optimization_evidence/baseline_runs/seed_101/run.log | 0.5309270903332314 | 69.373705 |
| reference/101 | optimization_evidence/reference_runs/seed_101/result.json | optimization_evidence/reference_runs/seed_101/run.log | 0.3803967503336025 | 57.017687 |
| baseline/202 | optimization_evidence/baseline_runs/seed_202/result.json | optimization_evidence/baseline_runs/seed_202/run.log | 0.5180353616645638 | 64.181802 |
| reference/202 | optimization_evidence/reference_runs/seed_202/result.json | optimization_evidence/reference_runs/seed_202/run.log | 0.3782085348308707 | 56.748077 |
| baseline/303 | optimization_evidence/baseline_runs/seed_303/result.json | optimization_evidence/baseline_runs/seed_303/run.log | 0.5196528819991121 | 65.026093 |
| reference/303 | optimization_evidence/reference_runs/seed_303/result.json | optimization_evidence/reference_runs/seed_303/run.log | 0.3829074791695651 | 56.916232 |
| baseline/404 | optimization_evidence/baseline_runs/seed_404/result.json | optimization_evidence/baseline_runs/seed_404/run.log | 0.523678173830073 | 65.347529 |
| reference/404 | optimization_evidence/reference_runs/seed_404/result.json | optimization_evidence/reference_runs/seed_404/run.log | 0.3841831811684339 | 57.365457 |
| baseline/505 | optimization_evidence/baseline_runs/seed_505/result.json | optimization_evidence/baseline_runs/seed_505/run.log | 0.5151795976635185 | 64.690920 |
| reference/505 | optimization_evidence/reference_runs/seed_505/result.json | optimization_evidence/reference_runs/seed_505/run.log | 0.3763024723327059 | 57.373230 |

## 三项内容门
| 项目 | 状态 | 结论及证据 |
|---|---|---|
| G01 | pass | 开放可执行value方法，guard不限制为常量字典；Reference改变支撑表示、跨类求解和成本复用，研究空间成立。 证据：workspace/harbor_task/instruction.md:15；workspace/harbor_task/instruction.md:23；workspace/harbor_task/tests/evaluator.py:6 |
| G02 | pass | 在包内材料范围内Baseline合理；五维与专项均无已证实削弱/不公平。外部训练来源未知不影响本题固定输入数值比较。 证据：workspace/harbor_task/environment/starter/method.py；expert_evidence/方法与复现说明.md；optimization_evidence/comparison_summary.json |
| G03 | pass | 全部正式成对原始日志可复算；改善0.141094937531秒，归一化0.270558759041，23.11965σ_B，540次保真有效且尚有空间。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_101/result.json |

## 双轨有效时长、八字段及格式
两轨各用自己的公开历史；真实请求记录模型/推理档分别为gpt-6.1-sol/xhigh与doubao-seed-2-1-turbo-260628/high。核对所有分段成功API原始耗时和请求切片，汇总相符。各episode窗口无重叠；counted_successful_api_seconds已扣重复回复，不二次扣。

| 分段 | episodes | wall秒 | 扣除秒 | 有效秒 |
|---|---:|---:|---:|---:|
| gpt | 12 | 33583.012910 | 1831.648997 | 31751.363913 |
| seed_original | 14 | 10986.671461 | 3481.596233 | 7505.075228 |
| seed_continuation_v3 | 15 | 15416.845426 | 6652.065126 | 8764.780301 |
| seed_continuation_v4 | 16 | 6791.344907 | 1833.053606 | 4958.291301 |
| seed_continuation_v5 | 5 | 7392.576174 | 1419.942242 | 5972.633932 |
Codex有效31751.363913秒=8.819823小时；Seed四段合计27200.780761秒=7.555772小时，不能相加。排除GPT未收集第13轮、Seed重复卡顿/环境失败/无输出等待、setup-only片段。干净复验及12小时稳定性另列。
非训练例外可接受：源码无训练/微调；Codex三方法窗口1012.348/788.682/1012.304秒，Seed未中断窗口962.327/930.333秒，约13–17分钟；各三闭环24个proposal/implementation/score/next-decision原始会话片段逐段匹配，持续方法演进与结束方向有证据。重叠方法窗口不累计时长，不新增分钟数阈值。
Codex64轮=63完整评分+1运行失败；Seed36轮=16完整评分（9有效、7质量失败）+20中断。全部行恰好八字段；79个完成轮的分数、完成时间和retained_best与protected_reports逐一吻合，未出分null。现有score为归一化奖励，需按V4改为原始raw_metric。
Codex全局best干净复验raw=0.03571169853336566秒、归一化0.9315204853730451；Seed自身best干净复验raw=0.6885859794352048秒、归一化−0.32040859402397；二者各270次有效。Agent best分数不冒充专家Reference的归一化门槛。
best_method仅method.py与solve.sh；其余历史记录允许保留。expert_annotation运行/发布状态建议归run_summary，duration_hours建议明确为wall；time_accounting的历史snapshot状态与最终完成状态应清楚分层。

## 核心结论与 Harbor 部署
**FAIL / NOT READY。静态内容复核已完成。** 方法空间、Baseline合理性、成对改善和七小时非训练例外成立；四项核心返修见下文。官方0.3.10及目标平台动态验收尚未执行。
路径：Agent context=workspace/harbor_task/environment，Verifier context=workspace/harbor_task/tests；本包两Dockerfile COPY来源与入口可解析、各自依赖自备。私有30组cases/targets仅进Verifier，700目录与UID1001候选边界存在；父评分器读gold，子进程仅得到允许数组，评分原始参照/reward由可信侧持有。静态实现不等于穷尽操作系统认证。

| 项目 | 状态 | 结论与证据 |
|---|---|---|
| H01 | pass | 真实任务根、两Dockerfile、测试入口和30组私有cases/targets齐全；评分器实际加载私有目标并调用受限worker。 证据：workspace/harbor_task/task.toml；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/tests/evaluator.py:83 |
| H02 | fail | 已交Harbor0.24.0/schema1.4原生NOP记录且显式separate；但两environment段的storage_mb/network_mode与当前内场合同冲突。 证据：workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:19；workspace/harbor_task/task.toml:33；workspace/harbor_task/task.toml:37；expert_evidence/harbor_nop/current/trial_task_binding.json |
| H03 | fail | 两构建context的COPY和入口可解析，依赖自备；environment直属Compose被V4禁止，artifacts还包含公开结果/历史源码，超出候选面。 证据：workspace/harbor_task/environment/Dockerfile；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/environment/docker-compose.yaml；workspace/harbor_task/task.toml:2 |
| H04 | pass | 可信入口生成单个有限标量reward，原子写入且与原生NOP收集值一致；单次worker超时缺陷另列核心约束项。 证据：workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/verifier.py:24；expert_evidence/harbor_nop/current/verifier/reward.txt |
| H05 | pass | 已有Job选择本任务与nop、docker provider，Verifier未禁用；无空选择或覆盖冲突。 证据：expert_evidence/harbor_nop/current/job/config.json；expert_evidence/harbor_nop/current/result.json#config |
| H06 | pass | 同一次原生NOP的config/result/reward/log/移交文件齐全，separate、结束且无exception；reward均为0.00787694793835168。仅既有记录，非本次实跑。 证据：expert_evidence/harbor_nop/current/config.json；expert_evidence/harbor_nop/current/result.json；expert_evidence/harbor_nop/current/verifier/reward.txt；expert_evidence/harbor_nop/current/trial.log；expert_evidence/harbor_nop/current/artifacts/manifest.json |
既有原生NOP：Harbor0.24.0/docker、separate、结束且exception=null；config/result/reward/log/候选清单一致，reward=0.00787694793835168，30例270次质量有效。只支持历史运行记录，不是本次实跑，0分并非必要条件。

## 21项实现检查
| 检查项 | 状态 | 原因与证据 |
|---|---|---|
| QA01 | pass | 八部分完整，无论文标题/arXiv/repo或Reference方法泄露。 证据：workspace/harbor_task/instruction.md |
| QA02 | pass | 评分(B−x)/B，U=0，连续单调不裁剪；公开B为正式均值，Hidden用独立baseline-only锚点0.968649282300。 证据：workspace/harbor_task/tests/evaluator.py:110；workspace/harbor_task/environment/public_eval/protocol.json#score；workspace/harbor_task/tests/protocol.json#score；expert_evidence/prerequisites/hidden_baseline_anchor.json |
| QA03 | pass | Reference可运行接口、五组结果/原始日志齐全；归一化0.270558759041，非训练无需模型重载。 证据：workspace/reference/method.py；optimization_evidence/reference_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_202/result.json；optimization_evidence/reference_runs/seed_303/result.json |
| QA04 | pass | 评估耗时随机；按全部5个Baseline原始值n−1复算σ=0.006102814476，改善23.11965σ满足3σ。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_101/run.log；optimization_evidence/reference_runs/seed_101/run.log；optimization_evidence/baseline_runs/seed_202/run.log |
| QA05 | fail | 存在guard、UID降权、快照与资源配置，但无换行stdout可使readline阻塞而越过60秒检查；还含当前禁用的磁盘/禁网声明。 证据：workspace/harbor_task/tests/evaluator.py:47；workspace/harbor_task/tests/evaluator.py:49；workspace/harbor_task/tests/evaluator.py:58；workspace/harbor_task/tests/evaluator.py:60 |
| QA06 | pass | test.sh单命令进入可信verifier；标量reward文件原子写出，HARD_GATE为-1，其他故障null并非正常分数。 证据：workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/verifier.py:19；workspace/harbor_task/tests/verifier.py:24 |
| QA07 | pass | 仅查题面：没有Hidden样本、标签或可恢复答案。 证据：workspace/harbor_task/instruction.md |
| QA08 | pass | 仅查题面：定义canonical目标与接口，不披露专家Reference具体压缩方案、代码或分数。 证据：workspace/harbor_task/instruction.md |
| QA09 | pass | 可信verifier先删预写reward/result，再以临时文件replace；评分JSON亦原子replace。 证据：workspace/harbor_task/tests/verifier.py:5；workspace/harbor_task/tests/verifier.py:24；workspace/harbor_task/tests/evaluator.py:77 |
| QA10 | pass | 候选冻结快照444，solve哈希验证；私有数据目录700且worker降至UID1001；未把候选import到root评分进程。仅确认有静态保护实现。 证据：workspace/harbor_task/tests/evaluator.py:29；workspace/harbor_task/tests/evaluator.py:118；workspace/harbor_task/tests/Dockerfile:6；workspace/harbor_task/tests/verifier.py:13 |
| QA11 | pass | 目标、数据、公式、可改面、边界、提交、迭代与完成标准均有实质内容；网络条款的部署冲突另列。 证据：workspace/harbor_task/instruction.md |
| QA12 | pass | ZIP成员无.git目录/历史；Dockerfile仅创建空运行时.git目录。 证据：@inventory；workspace/harbor_task/environment/Dockerfile:9 |
| QA13 | pass | 十个正式result与run.log/measurement逐一对应，540条真实格式测量逐例聚合一致；所有seed改善同向且质量门通过。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_101/result.json；optimization_evidence/baseline_runs/seed_202/result.json |
| QA14 | pass | 方法面开放且Reference未饱和；输入几何因子化、代数消去/核复用等未由题面给出具体解法，Agent轨迹已有演进证据。 证据：workspace/harbor_task/instruction.md:23；workspace/reference/method.py；expert_evidence/method_cycles/codex.json |
| QA15 | pass | CPU题0GPU、2CPU/2048MiB；正式单seed57–69秒，完整best复验206/461秒、NOP评分约513秒，已有记录支持单轮≤2h。无换卡事项。 证据：workspace/harbor_task/task.toml；expert_evidence/best_method_clean_rerun/execution.json；expert_evidence/seed_best_clean_rerun/execution.json；expert_evidence/harbor_nop/current/result.json |
| QA16 | pass | 有效时长31751.363913/27200.780761秒，分别8.8198/7.5558h；非训练、约13–17分钟完整窗口、各3闭环、后续方向和各自best干净复验支持7h例外。 证据：expert_evidence/time_accounting/summary.json；expert_evidence/method_cycles/codex.json；expert_evidence/method_cycles/seed.json；expert_evidence/method_cycles/ending_directions.json |
| QA17 | fail | separate及双context路径存在，既有NOP完整；但environment直属Compose、禁用配置及多余公开结果artifacts不符合当前V4部署/移交合同。 证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:19；workspace/harbor_task/task.toml:33 |
| QA18 | fail | 两轨64/36条均恰好八字段，79份完成报告与时间/最佳标记可匹配；但score写归一化reward而非当前V4原始耗时口径，需静态转换。 证据：expert_evidence/trajectory_codex.json#score_semantics；expert_evidence/trajectory_codex.json#rounds[0].score；expert_evidence/trajectory_seed.json#rounds[0].score；expert_evidence/native_records/codex/protected_reports/2caed5f0a5cb48978475ac0f0c33f046.json#raw_metric |
| QA19 | pass | 题面已说明计时噪声、正式5 seeds与至少3倍Baseline样本标准差改善。 证据：workspace/harbor_task/instruction.md:19 |
| QA20 | pass | 题面含可改范围、网络规则和工具策略；禁网与白名单的当前平台不适用性作为部署失败处理。 证据：workspace/harbor_task/instruction.md:23；workspace/harbor_task/instruction.md:33 |
| QA21 | manual（非阻断） | run_summary缺task_binding.hash_profile/task_content_sha256；存在旧task checksum与逐文件映射，不能冒充V4四项官方指纹。属非阻断待补。 证据：expert_evidence/run_summary.json；expert_evidence/harbor_nop/current/trial_task_binding.json |

## 逐项最小修改与验收材料
### 1. 移除当前V4禁用的Compose、磁盘与禁网配置
事实：environment/直属存在docker-compose.yaml；task.toml的两个environment段声明storage_mb及network_mode=no-network；instruction.md第33行依赖禁网/模型白名单。
位置：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/task.toml:15；workspace/harbor_task/task.toml:19；workspace/harbor_task/task.toml:33；workspace/harbor_task/task.toml:37；workspace/harbor_task/instruction.md:33。
影响：V4当前内场明确不接受直属Compose以及storage_mb、禁网/网络白名单声明；历史Docker实验不能证明这一部署合同符合当前平台。
最小修改：删除直属Compose文件；移除任务及镜像内副本的storage_mb/network_mode声明，更新题面网络/工具策略为平台实际可用能力。将必要的可信服务启动、UID降权与超时清理放入单sandbox单容器且平台支持的实现，不靠额外容器。
验收材料：重交完整ZIP，静态配置与题面一致；目标平台确认服务启动、Actor UID与候选UID、CAP_KILL等必要权限及真实资源落实。保留历史记录而非改标为新平台实测。

### 2. 将artifacts缩至候选所需资产
事实：task.toml第2行除候选目录还收集公开评分结果目录；dev_service.py会在该目录保存评分JSON和方法历史快照。
位置：workspace/harbor_task/task.toml:2；workspace/harbor_task/environment/public_eval/dev_service.py:21；workspace/harbor_task/environment/public_eval/dev_service.py:24；workspace/harbor_task/tests/verifier.py:15。
影响：V4只移交候选代码及必要配置/模型，公开评分结果和历史源码并非Verifier输入。即使现有Verifier不信任这些结果，扩大artifacts仍违反交付边界。
最小修改：从artifacts删除公开结果目录，仅保留题面规定的两文件候选目录；专家过程日志留在expert_evidence，不作为Agent候选移交。
验收材料：更新配置与候选移交清单；新版本Verifier只接收必要候选并产生自身评分。既有NOP可保留，版本关联后补。

### 3. 修复无换行stdout导致的超时失效
事实：两份evaluator.py在select确认可读后调用阻塞readline（第49、60行）；之后还有无deadline的stdout.read（第68行）。候选输出少量不带换行字节并继续计算时，readline可阻塞，后续60秒剩余量检查无法执行。
位置：workspace/harbor_task/tests/evaluator.py:47；workspace/harbor_task/tests/evaluator.py:49；workspace/harbor_task/tests/evaluator.py:58；workspace/harbor_task/tests/evaluator.py:60；workspace/harbor_task/tests/evaluator.py:68；workspace/harbor_task/environment/public_eval/evaluator.py:49。
影响：单次初始化/方法60秒约束是题面核心规则；select只保证有字节可读，不保证完整行到达。7200秒外层Verifier时限不等于单次60秒，正常TIMEOUT探针也不能覆盖该分支。
最小修改：使用非阻塞、有上限的字节读取与累计缓冲解析行，所有read/wait均服从同一monotonic deadline；在finally中终止并回收进程组。公开服务端也应执行完整评测deadline，而非只由客户端放弃等待。
验收材料：授权隔离环境中的负向记录：导入/调用阶段无换行输出后持续运行、超长单行、DONE后不退出均在指定deadline内终止；无遗留worker，分类保留TIMEOUT/RESOURCE。同步两份评分器。

### 4. 八字段score恢复V4原始指标口径
事实：两轨每轮均有八字段且全部79份完成报告可匹配，但score保存归一化奖励，例如Codex第1轮-0.374633743283，而对应原始耗时为0.716864103102秒。Codex score_semantics明确写normalized；Seed也使用同口径。
位置：expert_evidence/trajectory_codex.json#score_semantics；expert_evidence/trajectory_codex.json#rounds[0].score；expert_evidence/trajectory_seed.json#rounds[0].score；expert_evidence/native_records/codex/protected_reports/2caed5f0a5cb48978475ac0f0c33f046.json#raw_metric。
影响：当前V4要求每轮score为实际Public/Dev原始指标或声明代理指标；当前使用主指标的归一化reward，未声明独立代理指标。可追溯但字段语义不符合当前格式。
最小修改：按现成protected_reports将有完整测量轮的score写为raw_metric；未出分保持null，失败原因和质量失败状态保留。归一化reward与质量门用于选优，可在外部映射/说明保存，八字段不扩列。
验收材料：两轨逐轮映射到原报告的raw_metric、时间、状态、方法标识；保留原轨迹原样为历史材料，新轨迹不改实验事实/顺序/有效时间。无需重跑模型或正式B/R。

## 五类非阻断待补与工具缺口
四项指纹：run_summary缺task_binding.hash_profile/task_content_sha256，旧task checksum及文件表原样保留；真实官方task_fingerprint.py可用后计算四项并补字段，输出放范围外。不单凭缺失/差异判退或重跑。
换卡B/R影响、换卡耗时、换卡重算三类对本CPU题不适用。NOP已有完整历史记录，修订后版本关联仍可后补，不单独阻断。上述放宽不覆盖已证实的Compose/配置/移交/超时/字段语义问题。
未获得/执行真实官方0.3.10、check_submission.py、独立GPU/格式脚本或task_fingerprint.py。当前为local-v4-content-review，未运行旧0.3.2工具、更未改版本字符串。官方工具验收待真实发布包可用后进行并记录版本/SHA。

## 本次运行与发布边界
- 未执行或导入任何提交代码、Docker、Verifier、安装脚本；未训练、未调用外部模型、未反序列化模型。
- 已有NOP/clean rerun/稳定性均是包内历史记录，不是本次独立运行；正式平台Hidden、内场权限与资源验收未完成。
- 仅静态可见源码/题面未发现需回显的秘密；完整逐件二进制隐私门禁由主控独立执行，本报告不冒充完成。

## 具体风险（最多三条）
- 候选stdout处理的阻塞读取使60秒deadline存在实际静态缺口，需边界负向复验。
- 历史研究/正常NOP依赖旧Docker配置，不能覆盖移除Compose/禁网配置后的目标平台身份与权限。
- 冻结encoder训练历史未获核验；结论仅适用相同固定表示上的数值保真/效率，不能扩大为新图像泛化。
