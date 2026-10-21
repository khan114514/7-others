# SAVA 独立本地 V4 静态内容复核

审查模型配置：gpt-6.1-sol / high（主控派发声明及工具接受）；runtime精确模型ID未暴露，版本not_exposed；session_id=sava_a。
输入逻辑名：SAVA_AutoResearch_submission.zip；ZIP SHA256：8e950e70c9434f14da4ae64890a90074860a1598d1c4f6f543a9cf69d8165979。以下引用均相对归档外包装 SAVA_AutoResearch/。
采用当前三期V4及2026-10-21五类非阻断口径；只读取指定ZIP非历史QA材料。未执行提交代码，不作官方0.3.10或平台验收声明。

## 优化面及实际执行边界
固定10D图像表示上的分层传输估值数值算子；在高保真门下最小化方法耗时。实际parser接受可执行函数/类与NumPy等数值程序，限制文件/进程/网络/解释器访问。
输入x:(N,10)、y:(N,)、r:(M,10)、ry:(M,)和seed；输出有限浮点(N,)向量。允许目标/内部批次/支撑表示/数值求解/聚合方法更新，固定输出保真门、数据和可信计时。guard仅限制I/O、进程、动态执行和解释器访问，未将接口压缩为常量字典。
证据：workspace/harbor_task/instruction.md:21-31；workspace/harbor_task/tests/evaluator.py:6-27；workspace/reference/method.py:8-86。

## Baseline 来源、实现、预算及公平性
共同修复后官方practical canonical：全20类对经验Sinkhorn标签成本、1024连续分批、样本coupling与点potential、batch coupling及n/(n-1)校准。无训练，固定表示来源限制已披露。Starter/正式Baseline/公开B同一方法。
包内专家说明绑定官方代码commit 7efc8a20cab9898d56fcf32d4f57ae55589152c9；本次不读取外部论文或声称独立复现。共同质量/成本修复在正式B/R前完成并披露，不计Reference创新。冻结encoder训练史不足已披露；任务只比较相同固定表示的数值算子，不能扩写为encoder训练充分或新图像泛化结论。
Baseline总体：在已提供材料范围内合理。以下五维事实：
- training_sufficiency：任务仅对固定10D输入执行数值推理，无optimizer、训练预算、checkpoint或重载训练主张；十份result.training均NOT_APPLICABLE。冻结encoder训练史未知已披露，不能称其已验证充分。 证据：optimization_evidence/训练证据说明.md；expert_evidence/方法与复现说明.md。
- method_representativeness：正式Baseline=共同修复后的官方practical canonical路径=未改Starter=公开B；来源固定官方commit在包内说明。目标是算子数值保真及效率，不主张分类/检测SOTA。190项全类对几何包含未使用项已披露；冗余不等于故障或故意削弱。 证据：workspace/harbor_task/environment/starter/method.py:8-19；expert_evidence/方法与复现说明.md；expert_evidence/source_alignment_manifest.json。
- budget_comparability：双方同固定输入/目标/依赖/图像切分/五seed、每seed六例九次fresh进程，同2CPU/2048MiB/0GPU；正式十次执行记录同一镜像；时间是被优化对象，非要求相同时长。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_101/result.json#protocol；optimization_evidence/reference_runs/seed_101/result.json#execution。
- implementation_correctness：canonical计算标签几何、样本coupling和potential、batch coupling及n/(n-1)校准，检查batch边缘残差；全部正式调用有限并过质量门。未见关闭输入、加载错误checkpoint或训练截短。 证据：workspace/harbor_task/environment/starter/method.py:21-53；optimization_evidence/baseline_runs/seed_101/measurement.json。
- selection_integrity：五seed全交，B101重叠试跑排除及串行重跑已披露；方法源码及log哈希逐份一致；后续Agent搜索与正式Reference区分，没有把重复评估冒充多次训练。 证据：optimization_evidence/训练证据说明.md；optimization_evidence/comparison_summary.json；expert_evidence/best_method_selection.json。

| 专项 | 状态 | 事实与证据 |
|---|---|---|
| B01 | pass | 来源、角色与固定算子适用范围披露完整；合理canonical起点，不据旧方法或可删冗余判弱基线。 expert_evidence/方法与复现说明.md；expert_evidence/source_alignment_manifest.json |
| B02 | pass | 无训练；实际数值路径、终止及540次保真材料健全，Baseline270次全部有效。 workspace/harbor_task/environment/starter/method.py:21-53；optimization_evidence/baseline_runs/seed_101/run.log |
| B03 | pass | 双方同输入、目标、计时范围、镜像、CPU/RAM和正式seed；效率研究允许实际耗时不同。 optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_101/result.json |
| B04 | pass | canonical固定公开起点；Reference压缩/删冗余/成本复用是允许研究变量；参数与选择范围已说明，未见单边异常训练配置。 workspace/harbor_task/environment/starter/method.py；workspace/reference/method.py；expert_evidence/方法与复现说明.md |
| B05 | pass | 计时为随机评估：五数据seed配对、每例九fresh调用；数值算法不使用seed随机采样不使固定数据seed无效，未冒充重复训练。 workspace/harbor_task/tests/evaluator.py:83-115；optimization_evidence/comparison_summary.json |
| B06 | pass | 官方canonical同时为初始方法和正式锚点，实际输出通过高保真门；无适用更强同协议锚点被包内材料证实却忽略。没有编造随机/常数分数。 workspace/harbor_task/environment/starter/method.py；optimization_evidence/baseline_runs/seed_101/measurement.json；expert_evidence/方法与复现说明.md |
| B07 | pass | 共同修复发生在正式B/R前，不计Reference收益；固定表示、数据边界、质量门和计时范围相同。 expert_evidence/方法与复现说明.md；optimization_evidence/训练证据说明.md；workspace/reference/method.py |
| B08 | pass | 来源、冗余限制、未知encoder训练史、选择口径及实际作用范围均披露，不把canonical宣称为当前最快。 expert_evidence/方法与复现说明.md；expert_evidence/来源对齐说明.md |

## Reference 方法与全部正式成对原值
带权farthest-first条件支撑压缩（48支撑、3轮Voronoi均值），只求实际使用的10×10跨域标签成本，复用batch成本矩阵；保留固定输出保真约束。
方向minimize。每个数据seed六例，每例九个fresh独立进程；这是随机计时评估，非独立训练。完整十组result及54条原始log计时分别与measurement逐条一致；十个源码哈希和log哈希均匹配。非训练字段为NOT_APPLICABLE，未加载任何权重。

| seed | Baseline秒 | Reference秒 | 改善秒 | B原始result | R原始result |
|---|---:|---:|---:|---|---|
| 101 | 0.5309270903332314 | 0.38039675033360254 | 0.15053033999962884 | optimization_evidence/baseline_runs/seed_101/result.json | optimization_evidence/reference_runs/seed_101/result.json |
| 202 | 0.5180353616645638 | 0.37820853483087075 | 0.13982682683369302 | optimization_evidence/baseline_runs/seed_202/result.json | optimization_evidence/reference_runs/seed_202/result.json |
| 303 | 0.5196528819991121 | 0.3829074791695651 | 0.136745402829547 | optimization_evidence/baseline_runs/seed_303/result.json | optimization_evidence/reference_runs/seed_303/result.json |
| 404 | 0.523678173830073 | 0.3841831811684339 | 0.13949499266163906 | optimization_evidence/baseline_runs/seed_404/result.json | optimization_evidence/reference_runs/seed_404/result.json |
| 505 | 0.5151795976635185 | 0.3763024723327059 | 0.1388771253308126 | optimization_evidence/baseline_runs/seed_505/result.json | optimization_evidence/reference_runs/seed_505/result.json |

B均值=0.5214946210980997，n−1样本标准差=0.006102814476157093；R均值=0.38039968356703563，n−1样本标准差=0.003247775168112226。Δ=0.1410949375310641秒，3σ_B=0.01830844342847128秒，Δ/σ_B=23.11965046329096。U=0来自非负耗时的数学下确界；归一化Δ/(B−U)=0.27055875904139454，即27.0558759041%，位于[0.15,0.8]。R到U剩余空间=0.38039968356703563秒，为62.331844602717σ_B。
全部540次质量有效；最大NRMSE=.007341078344218985、最低Spearman=.9999418075490896、最低top-quarter overlap=.9933333333333333。Public B=0.5214946210980997；Hidden独立B=.9686492823001269，规模不同，不混算。结果JSON同时保存全部60个case的540项计时、逐次质量值、process wall及复算校验。

## 三项内容门
| 内容门 | 状态 | 结论与证据 |
|---|---|---|
| G01 | pass | 题面与AST guard允许函数、类、求解器/分批/几何/校准/聚合；Reference实际引入带权支撑压缩、跨域几何删冗余和成本复用，非固定常数搜索。 证据：workspace/harbor_task/instruction.md:21-31；workspace/harbor_task/tests/evaluator.py:6-27；workspace/reference/method.py:8-86 |
| G02 | pass | 在包内可提供材料范围内Baseline合理；非训练canonical路径与未改Starter一致，五seed同资源正式材料完整，无已证实削弱/协议不公平。 证据：optimization_evidence/训练证据说明.md；expert_evidence/方法与复现说明.md；workspace/harbor_task/environment/starter/method.py |
| G03 | pass | 全部五组成对/540次调用质量有效；改善0.141094937531秒=23.119650σ_B，归一化0.270558759041，满足3σ及[0.15,0.8]，U=0为非负耗时数学下界。 证据：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/baseline_runs/seed_202/result.json；optimization_evidence/baseline_runs/seed_303/result.json；optimization_evidence/baseline_runs/seed_404/result.json；optimization_evidence/baseline_runs/seed_505/result.json；optimization_evidence/reference_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_202/result.json；optimization_evidence/reference_runs/seed_303/result.json；optimization_evidence/reference_runs/seed_404/result.json；optimization_evidence/reference_runs/seed_505/result.json |

## 两轨、时长、八字段与格式
固定数值推理，无训练/微调；实际完整提案到决策窗口约13–17分钟，每轨三个源码/会话/完整评分/后续决策闭环，持续演进，各有结束方向与自身最佳方法干净复验。两条均超过7h，未将训练任务套用例外。
- Codex + GPT-6.1 Sol：gpt-6.1-sol/xhigh；64行，63份正式Public/Dev报告；有效31751.36391291509秒=8.819823309h，episode墙钟33583.012909626006秒。跨域几何删冗余、Gibbs递推、成本因子化/耦合消去及内存/GEMM优化；63份完整公开报告，最后无新分数的502故障未计有效时间。
  自身best干净复验raw=0.03571169853336566、score=0.9315204853730451，质量有效；证据：expert_evidence/trajectory_codex.json；expert_evidence/method_cycles/codex.json；expert_evidence/best_method_clean_rerun/execution.json。
- Codex + Seed 2.1 Turbo：doubao-seed-2-1-turbo-260628/high；36行，16份正式Public/Dev报告；有效27200.780761258秒=7.555772434h，episode墙钟40587.437968413025秒。canonical重复、失败近似、NumPy Sinkhorn、GeomLoss/refinement至v19；保留20条中断、7条质量失败和9条成功，不将canonical重复计新方法。
  自身best干净复验raw=0.6885859794352048、score=-0.32040859402397，质量有效；证据：expert_evidence/trajectory_seed.json；expert_evidence/method_cycles/seed.json；expert_evidence/seed_best_clean_rerun/execution.json。

| 时间组件 | episode数 | 墙钟秒 | 扣除秒 | 有效秒 | 证据 |
|---|---:|---:|---:|---:|---|
| gpt | 12 | 33583.012909626006 | 1831.6489967109183 | 31751.36391291509 | expert_evidence/time_accounting/gpt.json |
| seed_original | 14 | 10986.671460625017 | 3481.5962328289265 | 7505.075227796091 | expert_evidence/time_accounting/seed_original.json |
| seed_continuation_v3 | 15 | 15416.845426292013 | 6652.065125616979 | 8764.780300675033 | expert_evidence/time_accounting/seed_continuation_v3.json |
| seed_continuation_v4 | 16 | 6791.344907370978 | 1833.0536062927931 | 4958.291301078184 | expert_evidence/time_accounting/seed_continuation_v4.json |
| seed_continuation_v5 | 5 | 7392.576174125017 | 1419.9422424163242 | 5972.633931708693 | expert_evidence/time_accounting/seed_continuation_v5.json |
时间按episode起止及真实扣除项复算，各分段求和与声明吻合。Seed四阶段总墙钟40587.437968413025秒、扣除13386.657207155025秒；两轨各自核算，不相加。setup-only、失败API/卡顿/未收集轮次排除；API或CLI存活本身不当有效研究。六个方法闭环的提案、实施、分数结论、下一决策文本均能在对应原生session解析文本中找到；其窗口重叠不作时间加数。未逐条独立重建全部API/工具片段的联合区间，已有分段材料按静态口径足够支持。
完整窗口GPT=1012.348/788.682/1012.304秒，Seed v16/v17=962.327/930.333秒；v15含中断5039.394秒不充当典型短迭代。两轨各≥3方法闭环、结束方向及各自best干净复验齐全，故应用非训练七小时例外，不把不足10h自动判退。
每行恰round/policy_name/method_summary/status/score/failure_reason/retained_best/time八字段且轮次递增。出分行可匹配63/16份公开受保护报告；成功均有限分数/null失败原因，未出分失败为null。轨迹分数为实际公开evaluator输出的声明归一化指标，原始秒数在受保护报告；不混Hidden。GPT故障末行时间明确为最后实际API回执；Seed时区由专家说明绑定Asia/Shanghai。
全局交付best来自GPT，历史score=.9355238795621499，发布空白变换后干净复验score=.9315204853730451。该best高于.8不使正式Reference门失败；Reference仍为独立专家参考.270558759。Seed自身v19仍慢于Baseline，负分保留，未误写为方法提升。12h稳定性材料记录43815.05229387502秒、6次功能轮，单独列出，不计研究时间，不声称本次复跑。

格式建议（除明确核心偏差外不单独扣错）：
- expert_evidence/native_records（extra_allowed）：保留原始模型/session/报告支撑，可另附索引减少阅读成本，不因此扣错。
- workspace/harbor_task/environment/instruction.md（merge_candidate）：镜像构建确需此副本，保留时与根题面同步生成并检查相同；两份task.toml同理。
- expert_evidence/expert_annotation.json（merge_candidate）：精简为类型/优化面/同split B/R/U和基本隔离声明；best/轨迹/稳定性元数据归run_summary，不因多交单独判错。
- expert_evidence/time_accounting/summary.json（merge_candidate）：实际已封定但state仍INCOMPLETE_LIVE_SNAPSHOT，且run_summary.duration_hours记录墙钟；建议显式wall_hours/effective_hours，避免误读。
- workspace/harbor_task/environment/docker-compose.yaml（obsolete_layout）：属于当前V4明确禁用文件，按核心问题删除并迁移支持的单容器部署参数。

## 核心结论与Harbor
核心结论：FAIL / NOT READY。三项研究内容门通过；静态复核已完成。21项计数：{"pass": 18, "fail": 2, "manual_nonblocking": 1}。确定性核心失败不被工具待补或其他未实跑项遮盖。
路径：Agent context=environment/，Verifier context=tests/，/workspace/solution候选路径一致，test.sh写/logs/verifier/reward.txt；私有材料未进入Agent COPY。包内Harbor版本0.24.0/docker依据现成配置/result/lock；未加载原生schema或认证目标平台。
现成同Trial NOP：nop/separate，结束、exception=null，reward=.00787694793835168与日志/result一致；147个任务文件哈希逐项匹配。该记录支持旧部署分支已走通，不豁免当前V4部署冲突。

| Harness | 状态 | 事实与证据 | 最小修改与验收 |
|---|---|---|---|
| H01 | pass | 任务根、双Dockerfile、test.sh、30个私有case及30个私有target完整；独立Verifier实际读取tests私有材料。 workspace/harbor_task/task.toml；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/private_cases/seed_601_uniform_labels.npz；workspace/harbor_task/tests/private_targets/seed_601_uniform_labels.npy | 无需修改。  |
| H02 | fail | 包内原生NOP记录Harbor0.24.0/docker与separate；但当前TOML声明不支持的storage_mb和禁网设置，与当前V4冲突。 workspace/harbor_task/task.toml:12-37；expert_evidence/harbor_nop/current/trial_task_binding.json | 两份配置同步删除不受支持的磁盘/网络限制字段；题面按平台实际联网与默认工具策略改写，保留候选接口内的文件/网络API禁令和可信评分边界，不以基础设施禁网作为正确性前提。 同步后的题面和配置；目标平台可识别资源配置及真实网络条件说明；允许联网环境中的公开Dev与独立Verifier隔离验收材料。 |
| H03 | fail | 双context COPY/WORKDIR/依赖和候选路径可静态闭合，但Agent context含禁用Compose，且artifacts额外移交公开历史结果目录。 workspace/harbor_task/environment/Dockerfile；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/environment/docker-compose.yaml；workspace/harbor_task/task.toml:2；expert_evidence/harbor_nop/current/artifacts/manifest.json | 删除任务context内的Compose文件；用平台认可的单容器部署参数落实服务所需UID、能力和超时清理。不要直接去掉CAP_KILL而使既有killpg退化。；artifacts仅保留/workspace/solution；公开历史报告留在专家/Agent日志归档，不作为Verifier候选移交。同步两份TOML和任何调用覆盖。 完整重打包；无四种禁用文件的成员清单；平台可识别的单容器部署配置、Actor降权/冻结面与超时清理记录。；新完整包配置与移交manifest，证明仅最终method.py、冻结solve.sh进入Verifier所需候选路径；Verifier重新计算并原子写reward。 |
| H04 | pass | test.sh调用verifier；结果先清理再可信原子写reward.txt；基础设施错误null并非统一0；归一化标量有限。超时缺口另归核心约束项。 workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/verifier.py:4-25；workspace/harbor_task/tests/evaluator.py:77-81 | 无需修改。  |
| H05 | pass | 已有Job/Trial明确task、docker provider与nop；verifier.disable=false，separate真实分支在result/lock记录，无调用空选择反证。 expert_evidence/harbor_nop/current/config.json；expert_evidence/harbor_nop/current/job/config.json；expert_evidence/harbor_nop/current/result.json；expert_evidence/harbor_nop/current/lock.json | 无需修改。  |
| H06 | pass | 已有同Trial原生NOP记录完整：nop/separate、结束且exception=null，reward=0.00787694793835168三处一致；147项文件hash逐项匹配当前任务。不是本次实跑，也不能抵消当前V4部署冲突。 expert_evidence/harbor_nop/current/config.json；expert_evidence/harbor_nop/current/result.json；expert_evidence/harbor_nop/current/verifier/reward.txt；expert_evidence/harbor_nop/current/trial.log；expert_evidence/harbor_nop/current/trial_task_binding.json | 无需修改。  |

| 检查项 | 状态 | 原因与证据 | 最小修改与验收 |
|---|---|---|---|
| QA01 | pass | 八部分齐全，题面未披露论文标题、arXiv或解法repo；canonical数值定义是任务接口说明。 workspace/harbor_task/instruction.md:1-45 | 无需修改。  |
| QA02 | pass | Dev/Hidden各用本split正式B，U=0；(B-x)/B连续单调、不裁剪，B对应0、0秒对应1。 workspace/harbor_task/tests/evaluator.py:109-114；workspace/harbor_task/tests/protocol.json:39-45；workspace/harbor_task/environment/public_eval/protocol.json#score | 无需修改。  |
| QA03 | pass | Reference实现及五个正式result/log完整；非训练无需checkpoint；540次原值复算支持归一化0.270558759。 workspace/reference/method.py；optimization_evidence/reference_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_505/run.log；optimization_evidence/comparison_summary.json | 无需修改。  |
| QA04 | pass | 随机计时按五seed全量配对复算n-1 σ_B，正向改善23.119650σ_B，超过3σ；不是将固定seed当确定性评价。 optimization_evidence/comparison_summary.json；workspace/harbor_task/instruction.md:17-19 | 无需修改。  |
| QA05 | fail | CPU/RAM/源码大小/降权和冻结面有实现，但管道阻塞readline可绕过单次截止；完整公开评测只有客户端等待超时，没有可信整轮终止。 workspace/harbor_task/tests/evaluator.py:47-68；workspace/harbor_task/environment/public_eval/dev_client.py:12-15；workspace/harbor_task/environment/public_eval/dev_service.py:23；workspace/harbor_task/task.toml:15-19 | 使用带剩余deadline的非阻塞/分块管道读取，处理无换行、输出上限、EOF与清理；在可信父侧为整个Dev评价设置独立截止时间，超时杀掉进程组并等待回收，服务返回TIMEOUT且可接受下一请求。；两份配置同步删除不受支持的磁盘/网络限制字段；题面按平台实际联网与默认工具策略改写，保留候选接口内的文件/网络API禁令和可信评分边界，不以基础设施禁网作为正确性前提。 受隔离环境中的无换行import输出、无换行call输出和多个单调用均未超时但整轮超时三类负测；有限时返回、无残留worker及下一次Dev成功记录；同步公开/私有evaluator。；同步后的题面和配置；目标平台可识别资源配置及真实网络条件说明；允许联网环境中的公开Dev与独立Verifier隔离验收材料。 |
| QA06 | pass | solve.sh是完整Dev单命令入口；tests/test.sh正式调用受信verifier，失败分类与标量reward链路可见。 workspace/harbor_task/environment/starter/solve.sh；workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/verifier.py:9-25 | 无需修改。  |
| QA07 | pass | 仅题面检查：无Hidden样本、标签或可恢复答案；提及私有集及边界不构成泄露。 workspace/harbor_task/instruction.md:9；workspace/harbor_task/instruction.md:27-37 | 无需修改。  |
| QA08 | pass | 仅题面检查：未披露专家支撑压缩参考解、具体参数或结果；canonical起始算子是任务规范。 workspace/harbor_task/instruction.md:13-25 | 无需修改。  |
| QA09 | pass | Verifier删除reward.txt/reward.json/result.json，JSON及reward使用临时文件原子replace，候选不写最终reward。 workspace/harbor_task/tests/verifier.py:4-6；workspace/harbor_task/tests/verifier.py:24；workspace/harbor_task/tests/evaluator.py:77-81 | 无需修改。  |
| QA10 | pass | Agent镜像冻结evaluator/targets/solve；方法可信快照只读，子进程降UID1001，私有case/target root专用权限。静态不认证穷尽攻击面。 workspace/harbor_task/environment/Dockerfile:15-21；workspace/harbor_task/tests/Dockerfile:4-7；workspace/harbor_task/tests/evaluator.py:29-46；workspace/harbor_task/tests/evaluator.py:118-124 | 无需修改。  |
| QA11 | pass | 目标/环境/指标/可改范围/边界/提交/迭代/完成条件各有实质说明。 workspace/harbor_task/instruction.md:3-45 | 无需修改。  |
| QA12 | pass | ZIP成员清点无.git历史；镜像mkdir空.git不等于携带remote/tag/reflog。 @inventory；workspace/harbor_task/environment/Dockerfile:9 | 无需修改。  |
| QA13 | pass | 五seed两侧原始result/measurement/log齐全，逐seed均改善、全部质量有效；非训练无模型重载义务。 optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/baseline_runs/seed_202/result.json；optimization_evidence/baseline_runs/seed_303/result.json；optimization_evidence/baseline_runs/seed_404/result.json；optimization_evidence/baseline_runs/seed_505/result.json；optimization_evidence/reference_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_202/result.json；optimization_evidence/reference_runs/seed_303/result.json；optimization_evidence/reference_runs/seed_404/result.json；optimization_evidence/reference_runs/seed_505/result.json | 无需修改。  |
| QA14 | pass | 函数/类接口开放，未直接提示的Gibbs递推、成本因子化与耦合消去等在轨迹中实际可探索；Reference分数0.27056未饱和。 workspace/harbor_task/tests/evaluator.py:6-27；workspace/reference/method.py；expert_evidence/trajectory_codex.json；expert_evidence/best_method/method.py | 无需修改。  |
| QA15 | pass | CPU任务0GPU、2核；正式每seed执行约56.75–69.37秒，完整最佳Dev干净复验206.34/460.74秒，代表完整迭代约13–17分钟，未见已证实单轮资源超限。 optimization_evidence/baseline_runs/seed_101/result.json#execution；expert_evidence/best_method_clean_rerun/execution.json；expert_evidence/seed_best_clean_rerun/execution.json；expert_evidence/method_cycles/codex.json；expert_evidence/method_cycles/seed.json | 无需修改。  |
| QA16 | pass | 两轨有效8.819823h和7.555772h；不达默认10h，但非训练短迭代七小时例外有各三闭环、后续方向、自身best干净复验及扣除材料。 expert_evidence/run_summary.json#duration_exception；expert_evidence/time_accounting/summary.json；expert_evidence/method_cycles/codex.json；expert_evidence/method_cycles/seed.json | 无需修改。  |
| QA17 | fail | separate、Hidden及现有NOP有证据，但当前V4禁用Compose/网络磁盘配置与非候选artifacts使Harbor部署静态不通过。 workspace/harbor_task/task.toml；workspace/harbor_task/environment/docker-compose.yaml；expert_evidence/harbor_nop/current/artifacts/manifest.json | 删除任务context内的Compose文件；用平台认可的单容器部署参数落实服务所需UID、能力和超时清理。不要直接去掉CAP_KILL而使既有killpg退化。；两份配置同步删除不受支持的磁盘/网络限制字段；题面按平台实际联网与默认工具策略改写，保留候选接口内的文件/网络API禁令和可信评分边界，不以基础设施禁网作为正确性前提。；artifacts仅保留/workspace/solution；公开历史报告留在专家/Agent日志归档，不作为Verifier候选移交。同步两份TOML和任何调用覆盖。 完整重打包；无四种禁用文件的成员清单；平台可识别的单容器部署配置、Actor降权/冻结面与超时清理记录。；同步后的题面和配置；目标平台可识别资源配置及真实网络条件说明；允许联网环境中的公开Dev与独立Verifier隔离验收材料。；新完整包配置与移交manifest，证明仅最终method.py、冻结solve.sh进入Verifier所需候选路径；Verifier重新计算并原子写reward。 |
| QA18 | pass | 两条当前任务轨迹64/36行，每行恰八字段且轮次递增；成功有限分数与null失败原因、失败无分为null。所有出分行可匹配63/16份真实受保护报告，六个方法闭环链接可回查原session。 expert_evidence/trajectory_codex.json；expert_evidence/trajectory_seed.json；expert_evidence/method_cycles/codex.json；expert_evidence/method_cycles/seed.json | 无需修改。  |
| QA19 | pass | 题面明确时间有噪声、五个正式seed配对、至少3倍Baseline样本标准差，符合随机评价seed及有效阈值说明。 workspace/harbor_task/instruction.md:17-19 | 无需修改。  |
| QA20 | pass | 题面可改method、候选API限制与模型传输/网页/安装工具策略明确。策略与当前平台不符已另列修复，不能把文字完整性当落实成功。 workspace/harbor_task/instruction.md:21-33 | 无需修改。  |
| QA21 | manual_nonblocking | run_summary未含task_binding.hash_profile/task_content_sha256；已有147项文件hash和NOP checksum不等于官方四项指纹。按当前放宽口径非阻断待补，不自拟算法。 expert_evidence/run_summary.json；expert_evidence/harbor_nop/current/trial_task_binding.json#task_files_sha256 | 获得真实发布task_fingerprint.py后记录版本/哈希，对四项全递归计算并补run_summary.task_binding；保持历史值来源，不给旧结果补造当前hash。 真实官方指纹脚本版本/SHA与输出，四项范围及run_summary绑定；此项当前不作为返修通过前提。 |

## 逐问题退回说明
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

## 五类非阻断待补
- fingerprint_hash / pending：run_summary.task_binding缺失；NOP147项hash逐项匹配，但未替代真实官方四项指纹。 修改/验收：待获得真实官方脚本后补版本、范围与task_binding。 真实task_fingerprint.py版本/SHA、完整四项输出与绑定。
- hardware_migration_BR_gain / not_applicable：本题CPU/0GPU，不存在换H20/L20后的G02/G03补证要求。
- hardware_migration_timing / not_applicable：本题CPU/0GPU，现有耗时保持原设备记录，不改标目标GPU实测。
- NOP / existing_evidence_supported：当前包已有同Trial完整NOP，不需要为了放宽项启动新运行。
- hardware_migration_rescore / not_applicable：本题CPU/0GPU，正式分数复核按已有五seed，不要求换卡重算。

## 工具缺口与本次运行边界
未获得/运行真实官方0.3.10发布包、check_submission.py、独立GPU前检与task_fingerprint.py。本报告为local-v4-content-review，不是官方PASS。 后补真实工具版本/SHA、原始输出及退出状态；不得改旧工具版本号或覆盖本次首轮报告。
- 未执行、导入或反序列化提交代码/模型；未运行Docker、Verifier、训练或外部模型。
- 现成NOP、干净复验与12h稳定性只按包内记录核对，未冒充本次实跑或平台通过。
- 源码/题面可见作者home类路径未发现；本次不替代主控的逐件二进制完整隐私扫描，也不授权外发。

风险最多三条：
- Compose依赖与当前不支持的网络/磁盘配置会影响目标平台部署；本地NOP不能豁免。
- 阻塞管道读取及缺少可信整轮Dev终止可能使候选或公开评分服务超出题面时限。
- artifacts额外包含历史公开结果/源码快照，扩大最终候选移交面；未证实已操纵分数。
