# auto0505 独立只读内容复核（reviewer B）

输入 ZIP：auto0505-uploaded-source-snapshot.zip
ZIP SHA256：afa8cc2bfa825049b87ca3bd73909ba76d4300441638a865d72bcc0a6cfcfc77
配置模型：gpt-6-astra；runtime_reported_model/version：not_exposed；session_id：rf_b。
审查类型：local-v4-content-review，V4 + 2026-10-21五类放宽；非官方0.3.10报告。

## 优化面与实际执行边界
冻结生成模型与二阶更新下，设计确定性26节点分配函数；方法在受限CPU子进程生成节点，可信模型进程只接收验证后数值。
开放完整算法而不是仅常量搜索。固定模型、二阶更新、25步双方向/100NFE、512×512、12 Public+4 Hidden。候选函数在降权且受Landlock/seccomp的CPU进程内执行，可信父进程重新验证数值后构造桥接函数；高权限模型进程不导入候选源码。
证据：workspace/harbor_task/instruction.md, workspace/harbor_task/tests/candidate_worker.py:149, workspace/harbor_task/tests/local_backend.py:80

## Baseline 来源、实现与公平性
总评：在已提供历史正式材料范围内合理。
Starter=正式Baseline=归一化B来源；采用默认logit time_shift，mu=0.63。Reference采用sin((1-i/25)*π/2)节点。两侧均冻结预训练模型推理，无训练/微调；训练充分性、checkpoint重训与训练seed不适用。
历史共同预算：16图、100NFE、空提示、guidance=1、inject=0、bf16；A100 40GB、8核。Baseline/Reference numeric elapsed为305.4678/306.6705秒，完整worker receipt为392.4930/393.9862秒。

- training_sufficiency：不适用：无优化器、epochs或训练；冻结模型推理，真实加载/16图输出见成对日志。
- method_representativeness：默认time_shift公式有包内来源，合适的朴素正式对照；未发现需强制升级SOTA的证据。
- budget_comparability：同16图、512×512、100NFE、bf16、空提示、guidance=1、inject=0、相同A100/8核。
- implementation_correctness：B/R记录完成、无异常、实际schedule合法；一般接口重复节点问题与该两次有效节点分开处理。
- selection_integrity：当前正式成对记录为一次确定性评估，无需随机seed抽样；Baseline默认来源明确；Reference历史筛选记录并非重新认证。

| 项 | 状态 | 事实与证据 | 修改及验收 |
|---|---|---|---|
| B01 | pass | Starter、正式Baseline和评分B均为冻结logit time_shift默认节点，包内原实现给出同一mu规则；朴素方法适合固定模型/NFE节点研究，不凭年代要求SOTA。 证据：workspace/harbor_task/environment/starter/method.py:16；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/sampling.py:46；workspace/harbor_task/tests/anchors.json:9 | — |
| B02 | pass | 无需训练；成对日志显示模型加载及16图完成，回执exit_code=0。Baseline节点有26项且严格下降，没有观察到未训练或损坏对照。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/scorer-stdout.txt；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt.exit_code | — |
| B03 | pass | 两侧回执共享模型/数据/评分器/环境与100NFE；均为A100 40GB、8核亲和性，完整worker总耗时392.4930/393.9862秒。预算可比。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt | — |
| B04 | pass | Baseline用来源代码默认mu=0.63；Reference改变节点函数，属于公开优化面。未见只对Baseline设异常参数或截短预算的直接证据。 证据：workspace/harbor_task/environment/starter/method.py:18；workspace/reference/method.py:9；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/sampling.py:58 | — |
| B05 | pass | 无训练seed或随机评估seed协议；同一冻结模型做确定性反演重建。单次成对正式评估不冒充重复训练；样本图像不能当seed。 证据：workspace/harbor_task/tests/run_eval.py:196；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/modules/autoencoder.py:268；expert_evidence/historical_v341/training-applicability.json | — |
| B06 | pass | 评分B逐图均值等于Baseline正式全16图值。anchors列uniform/square等仅作辅助自述，未将未核验的sanity分数当作独立实证；未发现Baseline故障性低于基本对照的证据。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；workspace/harbor_task/tests/anchors.json:49 | — |
| B07 | pass | B/R唯一研究差异为时间节点；成对回执的协议、资产、权重、依赖字段一致，源码与实际schedule符合logit/sine公式。收益限于固定任务，未证明单个启发式的普遍效益。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；workspace/reference/method.py:10；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/schedule.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/schedule.json | — |
| B08 | pass | 专家说明给出默认偏移Baseline与正弦Reference的角色、预算和历史版本；与源码及回执一致。支持在已交材料范围内合理，不等于新部署已验收。 证据：expert_evidence/专家作业说明文档.md:3；expert_evidence/historical_v341/METHOD-EXPLANATION.md:3 | — |

## Reference 方法与全部正式原始值
正式协议无training_seed/evaluation_seed；每侧一份actual_v341完整确定性评分。实际数值来源为下述metrics.json，Harbor result是接口回执，不能把它伪装为规范seed_<n>/result.json：
- Baseline原始指标：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json
- Reference原始指标：optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json
- Baseline Trial result：optimization_evidence/baseline_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__QTvtVoh/result.json
- Reference Trial result：optimization_evidence/reference_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__DcGEtFn/result.json

| split / image | Baseline PSNR | Reference PSNR | R−B |
|---|---:|---:|---:|
| public / coco_000000000139 | 19.6263629270993 | 19.3246472055012 | -0.301715721598 |
| public / coco_000000000285 | 23.328605655905 | 23.9169236932625 | 0.588318037357 |
| public / coco_000000000632 | 14.7541131073316 | 15.1451834323552 | 0.391070325024 |
| public / coco_000000001000 | 18.8657586511116 | 15.9500419623975 | -2.91571668871 |
| public / coco_000000001268 | 16.6129604697789 | 16.6673694147707 | 0.0544089449918 |
| public / coco_000000001425 | 20.5132555519788 | 20.2334245273218 | -0.279831024657 |
| public / coco_000000001490 | 18.4359770621044 | 19.451136176094 | 1.01515911399 |
| public / coco_000000039769 | 17.2834052929981 | 17.2290643329844 | -0.0543409600136 |
| public / repo_art | 27.0465485797876 | 24.1486882191292 | -2.89786036066 |
| public / repo_boy | 17.6791304244775 | 23.8199043154625 | 6.14077389099 |
| public / repo_horse | 25.0213060953549 | 25.2561906287136 | 0.234884533359 |
| public / repo_nobel | 13.1715018320852 | 21.5554048808243 | 8.38390304874 |
| hidden / coco_000000000802 | 16.95575608451 | 18.5775746264274 | 1.62181854192 |
| hidden / coco_000000001296 | 19.8041780242965 | 19.632209614218 | -0.171968410078 |
| hidden / repo_cartoon | 15.2632398315979 | 23.1149589341345 | 7.85171910254 |
| hidden / repo_hiking | 28.1000189615828 | 23.2948665839468 | -4.80515237764 |

| 聚合 | B均值 | R均值 | Δ | U | 条件归一化 | B逐图样本std | R逐图样本std |
|---|---:|---:|---:|---:|---:|---:|---:|
| public | 19.3615771375011 | 20.2248315657347 | 0.863254428234 | 21.2197429818949 | 0.464573402228 | 4.08520340193 | 3.51818086536 |
| full | 19.5288824095 | 20.4573492842215 | 0.928466874721 | 21.5356233090307 | 0.462674017826 | 4.33797574138 | 3.22625828895 |
| hidden | 20.0307982254968 | 21.1549024396817 | 1.12410421418 | 22.4832642904382 | 0.458356684422 | 5.69646607443 | 2.40710615317 |

以上逐图std用n−1复算，只描述样本图像差异，不是随机评估σ_B。运行级每侧n=1，std=null；本题确定性、不训练，3σ检验不适用，也不恢复统一5%门槛。Public U有12图原始结果支持；full/hidden U只见锚点及转抄，所指源文件未随包交付，故其归一化为条件计算。
每图变化并非全部为正：不能将平均改善表述为每张图都改善。两侧质量门材料显示26项严格下降合法节点、16图有限结果、NFE100、COMPLETE/exit0。

## 三项内容门

| 项 | 状态 | 事实与证据 | 修改及验收 |
|---|---|---|---|
| G01 | pass | 优化面开放可执行get_timesteps函数；工作进程导入并调用函数，只对来源、资源和输出节点施加限制。可实现密度映射、分段网格、离散节点预算分配等方法，不是只允许常量字典。 证据：workspace/harbor_task/instruction.md:13；workspace/harbor_task/tests/candidate_worker.py:149；workspace/reference/method.py:9 | — |
| G02 | pass | 在现有静态与历史正式记录范围内Baseline合理：默认节点、正常推理、共同数据/NFE/设备预算，无直接不公平反证。新设备影响补证不作为否决条件。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；workspace/harbor_task/environment/starter/method.py | — |
| G03 | manual | 历史Public Δ=0.8632544282 dB、归一化0.4645734022，满足范围且正向；full Δ=0.9284668747、归一化0.4626740178。但full/hidden U仅见锚点及转抄，所引用原始full上限文件缺失，正式评分上限依据尚未闭环。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；workspace/harbor_task/tests/anchors.json:33；expert_evidence/historical_v341/final-method-clean/gpt-historical-metrics.json | 补回full/hidden上限所引用的原始逐图评分、对应方法与日志，并修正anchors.upper_bound.source为包内实际路径；如该值仅为估计，准确说明估计依据及预先标定规则。 验收：能从独立原始材料复算full U=21.535623309030747及hidden U=22.483264290438154，或审核一个有依据且未为通过倒推的新标定；保留原B/R数值与历史身份。 |

## 双轨与格式
当前Codex（gpt-6.1-sol/xhigh）和Seed（doubao-seed-2-1-turbo-260628/high）轨迹各0行；实际调用、有效时长未知，摘要明确未启动。历史轨迹102/47行的八字段齐全，但属于历史版本且没有可据以认证当前有效时长的材料。本审查不把墙钟首末跨度当有效时长，不对未知区间补造扣除。
best_method为空；未建立7h例外。正常训练虽可计时，但本题根本没有训练；容器12h稳定性另列当前未跑。
缺optimization_evidence/训练证据说明.md、comparison_summary.json；可用现有非训练说明与actual_v341原始材料整理，不捏造seed。历史目录、重复公开图像、部署源码仅提出精简/归位建议，不因多交判退。

## 核心总评、完成度与Harbor
核心总评：FAIL / NOT READY。安全静态检查已完成；未完成的证据和运行事项明确列出。决定性问题是当前独立轨迹/最终方法缺失、合法重复节点与积分器不相容、运行依赖不可交付；另有正式U来源、当前版本/provider和GPU配置适用性待核。
有两份Dockerfile，不因双镜像形式判错；Agent context=environment/，Verifier context=tests/，artifacts=/workspace/solution。实际隐藏材料在tests/assets/hidden，formal读取full；reward接口存在。
当前运行状态not_run；旧A100/RPC证据仅用于历史成对数值。没有因NOP未跑单独否决。

| 项 | 状态 | 事实与证据 | 修改及验收 |
|---|---|---|---|
| H01 | pass | 任务根完整，tests实际含4张Hidden+12张Public；formal链固定full，公开入口固定public，Agent context未含Hidden。 证据：workspace/harbor_task/task.toml；workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/asset_manifest.json；workspace/harbor_task/tests/local_backend.py:87 | — |
| H02 | manual | TOML显式separate及顶层artifacts，结构可读；包内仅历史摘要标Harbor0.23.0，当前部署未提供锁定版本/schema加载或provider配置凭据。不能将静态键名等同平台生效。 证据：workspace/harbor_task/task.toml；expert_evidence/historical_v341/run_summary.json#Harbor_installed_version；expert_evidence/deployment_source/README.md:34 | 声明本次目标Harbor版本和provider，按该版本核对task.toml及GPU/timeout/artifacts字段。 验收：目标版本配置解析记录或可核验schema依据，证明separate及资源/移交字段被读取。 |
| H03 | fail | 两任务Dockerfile上下文内COPY路径正确，artifacts覆盖评分读取的solution；但FROM依赖本地未发布tag，运行镜像COPY weights来源不在ZIP，19项资产仅哈希清单，无平台可获取入口。 证据：workspace/harbor_task/environment/Dockerfile:1；workspace/harbor_task/tests/Dockerfile:1；expert_evidence/deployment_source/runtime-image/Dockerfile:14；expert_evidence/deployment_source/prepare_weights.py:18；expert_evidence/deployment_source/runtime-image/weight_manifest.json | 提供平台可获取的不可变运行镜像（准确入口及digest），或提供19项冻结模型资产的批准获取入口、版本、大小、哈希和准备步骤；修正构建文档使包内路径可直接定位。 验收：接收方无需作者原机器目录即可准备依赖，environment/与tests/各自构建路径闭合，公开Dev及独立Verifier可从规定候选目录评分；保留目标环境构建/入口记录。 |
| H04 | pass | 正式入口绝对路径正确；成功由可信父进程将有限数字对象写/logs/verifier/reward.json；无分数失败不伪装正常0分。 证据：workspace/harbor_task/tests/test.sh:4；workspace/harbor_task/tests/evaluate_candidate.py:265；workspace/harbor_task/tests/evaluate_candidate.py:44 | — |
| H05 | not_applicable | 当前包未提供3.5.0实际Job/Trial调用配置；旧actual_v341配置仅供历史数值追溯，不当作当前平台提交。无强制job.yaml要求。 证据：expert_evidence/run_summary.json:39；expert_evidence/deployment_source/README.md:34 | — |
| H06 | pending_nonblocking | 当前NOP明确NOT_RUN。旧v3.4.1 Trial有reward0和正常结束，只说明历史分支；版本关联与本次新的本地GPU链尚待补证，缺NOP不单独否决。 证据：expert_evidence/run_summary.json:39；optimization_evidence/baseline_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__QTvtVoh/result.json | 在授权且依赖就绪的目标Harbor环境提供已有同版本NOP记录或后续运行记录；无需为本审查自动启动。 验收：同一Trial的config、result、reward、非空日志及版本映射，表明separate和artifacts移交；仅初始无效候选分支须如实限缩覆盖。 |

## 21项实现检查

| 项 | 状态 | 事实与证据 | 修改及验收 |
|---|---|---|---|
| QA01 | pass | 题面有八部分且未直接给出正弦Reference、论文标题、arXiv或仓库链接。 证据：workspace/harbor_task/instruction.md | — |
| QA02 | pass | score=(PSNR-B)/(U-B)未裁剪，实际reward使用score；score_clipped只是显示字段。B映射0、U映射1；full U来源缺口单列内容门。 证据：workspace/harbor_task/tests/run_eval.py:255；workspace/harbor_task/tests/local_backend.py:127；workspace/harbor_task/tests/evaluate_candidate.py:265 | — |
| QA03 | manual | Reference实现与原始非空评分日志、正式metrics和Trial result均存在；无需训练checkpoint。历史Public归一化达标，但full U来源尚未核实。不是因缺新卡复跑判退。 证据：workspace/reference/method.py；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/scorer-stdout.txt；workspace/harbor_task/tests/anchors.json:44 | 补回full/hidden上限所引用的原始逐图评分、对应方法与日志，并修正anchors.upper_bound.source为包内实际路径；如该值仅为估计，准确说明估计依据及预先标定规则。 验收：能从独立原始材料复算full U=21.535623309030747及hidden U=22.483264290438154，或审核一个有依据且未为通过倒推的新标定；保留原B/R数值与历史身份。 |
| QA04 | not_applicable | 固定推理协议无随机采样；AE返回mean，节点确定。运行级重复不足2时样本标准差为null；逐图标准差不是σ_B，不能据此施加3σ。 证据：workspace/harbor_task/tests/run_eval.py:196；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/modules/autoencoder.py:268 | — |
| QA05 | fail | 约束实现存在，但题面和guard允许相邻相等节点；冻结denoise以(t_prev-t_curr)/2为除数，零步长会产生非有限值，合法边界输入与数值调用不一致。 证据：workspace/harbor_task/instruction.md:20；workspace/harbor_task/tests/evaluate_candidate.py:76；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/sampling.py:133 | 把题面与两套节点验证器统一为严格递减节点，明确零步长不合法；或在可信积分器实现并验证零步长语义。不能仅把异常改成有效分数。 验收：提交重复相邻节点、正常严格递减节点及端点边界的验证结果；重复节点在模型调用前被正确拒绝或按明确协议处理，正常候选保持冻结NFE和有限结果。 |
| QA06 | pass | test.sh单命令调用可信评分器；成功有限标量写reward.json；格式、硬门、超时、资源、基础设施分开处理，非作弊硬门−1。 证据：workspace/harbor_task/tests/test.sh:4；workspace/harbor_task/tests/evaluate_candidate.py:44；workspace/harbor_task/tests/evaluate_candidate.py:265 | — |
| QA07 | pass | 只检查题面：说明私有评测边界，无Hidden图像、标签或可恢复答案。 证据：workspace/harbor_task/instruction.md:22 | — |
| QA08 | pass | 只检查题面：无正弦Reference代码/配置/结果，无直接解题答案。 证据：workspace/harbor_task/instruction.md | — |
| QA09 | pass | 评分前清理reward/metrics等旧结果；最终可信父进程经os.replace写出。 证据：workspace/harbor_task/tests/evaluate_candidate.py:20；workspace/harbor_task/tests/evaluate_candidate.py:239；workspace/harbor_task/tests/evaluate_candidate.py:270 | — |
| QA10 | pass | 冻结公开材料和weights设为不可写，私有tests为root私有；formal候选降至65534并受Landlock/seccomp，可信侧校验资产和仅接受26数值节点。 证据：workspace/harbor_task/environment/Dockerfile:15；workspace/harbor_task/tests/Dockerfile:5；workspace/harbor_task/tests/evaluate_candidate.py:81；workspace/harbor_task/tests/isolation.py:28；workspace/harbor_task/tests/local_backend.py:48 | — |
| QA11 | pass | 八部分实质覆盖目标、环境、指标、修改面、限制、提交、迭代和结束条件；节点重复的语义缺陷已另列。 证据：workspace/harbor_task/instruction.md | — |
| QA12 | pass | ZIP成员清点无.git目录；未把历史专家材料COPY到Agent上下文。 证据：@inventory；workspace/harbor_task/environment/Dockerfile | — |
| QA13 | pass | 已有actual_v341成对metrics、日志、同协议回执和Trial result；完整16图均可逐项复算，均值正向。目录非seed命名适合确定性非训练任务，不要求捏造seed或重训。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json | — |
| QA14 | pass | 可构造分段密度映射或离散预算交换等合法新节点函数，题面未直接提示这些方向。Reference Public归一化约0.465，尚有可识别方法空间。 证据：workspace/harbor_task/instruction.md:4；workspace/harbor_task/tests/candidate_worker.py:157；expert_evidence/historical_v341/final-method-clean/gpt-historical-metrics.json | — |
| QA15 | manual | 历史A100完整worker耗时392.49/393.99秒、8核/1GPU；目标配置仍仅A100。H20/L20迁移计时可后补，但包内自述用户授权不等于本审查获得授权证明；当前平台配置适配仍待确认。 证据：workspace/harbor_task/task.toml:17；workspace/harbor_task/task.toml:27；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt；expert_evidence/run_summary.json#hardware_choice | 保持历史A100真值；在目标提交前将GPU配置与实际获准范围一致，提供主控可核实的适用授权或H20/L20配置。换卡实测耗时可非阻断后补。 验收：目标资源配置可被平台识别且≤8GPU/64核；补充设备范围依据，迁移耗时单列待核，不改标历史计时。 |
| QA16 | fail | 当前两轨rounds均为0，run_summary明确not started且effective_seconds=null；两条有效时长要求未实现。历史轨迹不能以名称或日历跨度补足，7h例外也未建立。 证据：expert_evidence/run_summary.json#agent_execution；expert_evidence/trajectory_codex.json#rounds；expert_evidence/trajectory_seed.json#rounds | 从同一公开Starter分别开展两条干净、互不可见的实际模型迭代，提交八字段轨迹、有效时间区间及扣除依据、唯一最终方法和公开干净复验；保留旧记录的真实历史身份。 验收：两条轨迹与run_summary相符，默认各≥10h；非训练短迭代7h例外须逐条满足≥3闭环、下一方向、最佳复验及完整时长依据。不得把空轨迹或历史墙钟跨度当有效时间。 |
| QA17 | fail | separate、公开Dev、私有Hidden与候选移交路径已实现；但共享运行镜像仅有本地tag，源码构建所需weights目录缺失且无可用不可变获取入口，完整运行依赖未交付。NOP本身可后补。 证据：workspace/harbor_task/environment/Dockerfile:1；workspace/harbor_task/tests/Dockerfile:1；expert_evidence/deployment_source/runtime-image/Dockerfile:14；expert_evidence/deployment_source/README.md:9 | 提供平台可获取的不可变运行镜像（准确入口及digest），或提供19项冻结模型资产的批准获取入口、版本、大小、哈希和准备步骤；修正构建文档使包内路径可直接定位。 验收：接收方无需作者原机器目录即可准备依赖，environment/与tests/各自构建路径闭合，公开Dev及独立Verifier可从规定候选目录评分；保留目标环境构建/入口记录。 |
| QA18 | fail | 当前两份轨迹为空数组，best_method=null且缺唯一最终方法目录；无可审的当前八字段闭环/选优复验。历史102/47行只是历史版本，不能冒充本版独立轨迹。 证据：expert_evidence/trajectory_codex.json:2；expert_evidence/trajectory_seed.json:2；expert_evidence/run_summary.json:38 | 从同一公开Starter分别开展两条干净、互不可见的实际模型迭代，提交八字段轨迹、有效时间区间及扣除依据、唯一最终方法和公开干净复验；保留旧记录的真实历史身份。 验收：两条轨迹与run_summary相符，默认各≥10h；非训练短迭代7h例外须逐条满足≥3闭环、下一方向、最佳复验及完整时长依据。不得把空轨迹或历史墙钟跨度当有效时间。 |
| QA19 | not_applicable | 确定性固定推理协议，无随机训练或评估seed要求；不凭空要求多seed或统一5%阈值。 证据：workspace/harbor_task/instruction.md:4；workspace/harbor_task/tests/run_eval.py；expert_evidence/historical_v341/training-applicability.json | — |
| QA20 | pass | 明确唯一可改文件，允许公开文献/公开Dev，限定候选标准库及网络/子进程策略，禁止外部计算和额外容器。 证据：workspace/harbor_task/instruction.md:13；workspace/harbor_task/instruction.md:20；workspace/harbor_task/instruction.md:22 | — |
| QA21 | pending_nonblocking | task_binding有hash_profile而task_content_sha256=null；真实官方脚本待取得。本次只绑定输入ZIP SHA256，不猜四项指纹算法。 证据：expert_evidence/run_summary.json:33 | 取得真实官方task_fingerprint.py后按四项全量范围计算，保留原历史标识并记录版本映射。 验收：真实工具版本/哈希及其原输出，覆盖environment/、task.toml、tests/、instruction.md；输出放范围外。 |

## 五类非阻断待补
- 指纹/哈希：task_content_sha256=null；本次ZIP SHA256已独立计算；不猜官方算法。
- 换卡对Baseline/Reference和提升影响：已交A100历史成对值照实审查；H20/L20数值补验可延期。
- 换卡后耗时：历史worker约393秒，目标卡未实测；不改标旧卡计时。
- NOP：当前未跑；旧v3.4.1记录不自动覆盖新本地GPU链。
- 换卡引出的重算分数：无需为了本轮内容复核重做旧成对数据；正式上限来源缺口与换卡无关。

## 工具缺口与未实跑边界
真实官方0.3.10、独立格式/GPU/四项指纹脚本均未取得或未运行。没有修改旧工具版本号，没有把本地报告称为官方通过。本次仅ZIP、文本/JSON、文件哈希和统计计算；未执行或导入提交代码、未运行Docker/Verifier/模型/训练，未反序列化权重。完整逐附件隐私扫描由主控另做，本报告不冒充该门禁。

## 具体风险（最多三条）
- 当前空轨迹与未交最佳方法是确定核心缺口，不可由历史时长或旧QA替代。
- 重复节点在guard下合法而固定积分器除零，需统一接口域后复验。
- 运行基础资产无接收方可用入口；full上限原始来源也缺失，不能把有限源码快照标为完整可验收部署。

## 可复制给专家的退回说明
本包当前结论：NOT READY（静态核心不通过，非官方工具或平台结果）。

1. 两条当前轨迹rounds为空，run_summary表示尚未开始、effective_seconds=null、best_method=null。请提交两条真实独立模型闭环与有效时间材料、唯一最终方法和干净Public复验。默认各10h；符合完整条件的非训练短迭代才可各7h。保留历史记录但不要补造秒数或把历史QA转写成新证据。

2. 题面及evaluate_candidate.py允许相邻相等节点，但deployment_source/runtime-image/repo/.../flux/sampling.py:133除以相邻步长的一半。请统一严格递减域或实现明确零步长处理，补重复节点/正常节点边界复验；在模型评分前明确无效状态，不能把NaN包装成基础设施失败后当作完成。

3. 两任务Dockerfile依赖本地rfsteps运行tag；部署Dockerfile COPY weights的来源不在ZIP，prepare_weights要求作者原权重目录。请提供平台可获取不可变镜像digest或19项冻结资产的精确合规入口/版本/大小/哈希，修正文档包内路径，补目标环境构建与公开/私有入口调用证据。

4. 历史Public配对提升与归一化已核实达标，Baseline在现有材料内合理。full/hidden U的源文件引用不存在，仅有锚点转抄；请补原始上限逐图指标、方法与日志或明确可审计的估计依据，补comparison_summary和非训练适用性说明。无需为了换卡而自动重算已有成对结果。

5. 当前Harbor版本/provider和A100-only配置的适用范围需可核查依据。包内自述授权不作为本审查收到人类授权的替代。保持历史A100设备和耗时真值；目标GPU迁移耗时/分数、NOP、官方四项指纹等按现行口径非阻断后补。

验收边界：本次只静态读取ZIP；不执行提交代码、模型、Docker或训练。真实0.3.10及格式/GPU/指纹工具尚缺，不声称官方PASS。

