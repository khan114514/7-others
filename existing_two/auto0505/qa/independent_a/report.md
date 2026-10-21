auto0505 当前V4独立只读复核（reviewer A）
结论：FAIL / NOT READY；安全静态内容复核已完成。ZIP SHA256：afa8cc2bfa825049b87ca3bd73909ba76d4300441638a865d72bcc0a6cfcfc77
审查身份：session_id=rf_a；configured_model=gpt-6.1-sol；runtime_reported_model=not_exposed；runtime_version=not_exposed。
真实报告类型：local-v4-content-review；不是官方0.3.10工具输出，也不是本次动态复跑。

优化面及实际调用边界
冻结模型和二阶数值更新，只改变get_timesteps产生的26节点；实际worker调用完整函数，guard只限制合法节点与资源/来源边界，不限制函数算法为常量参数。
题面见 workspace/harbor_task/instruction.md:4-25；工作进程在tests/candidate_worker.py:149-158调用真实函数；tests/evaluate_candidate.py:67-78核验26节点。候选不作为高权限模块导入数值评分器，只移交经验证的26个有限数值。

Baseline来源、实现与公平预算
非训练的默认time_shift schedule：mu=0.5+(1024−256)/(4096−256)×0.65=0.63，均匀t经exp(mu)/(exp(mu)+1/t−1)映射；两方向共享。Starter、正式B、评分锚点同角色。
非训练正弦节点t=sin((1−i/25)π/2)，高t端更密、低t端更疏；只改变节点分配，模型、更新式、数据和NFE不变。
非训练任务：训练epochs、updates、曲线与训练checkpoint不适用；不将缺训练记录等同少训练。固定模型获取问题单列部署待核。历史两组receipt记录相同协议、数据/评分器/权重身份、A100-PCIE-40GB、8核、100 NFE；数值评分308.0607366/309.2353661秒，总receipt 392.4930/393.9862秒。完整Trial含平台准备约428.145/430.963秒，均低于2h。
Baseline五维事实：
- training_sufficiency：not_applicable；固定模型非训练推理，无epochs/updates/checkpoint选择或重训要求；不是少训练问题。 证据：workspace/harbor_task/instruction.md:7；workspace/harbor_task/tests/run_eval.py:164-239
- method_representativeness：pass；默认time_shift作为合理朴素基线与该任务匹配；源码含相同默认mu映射，未见必须替换它的可比更强锚点证据。 证据：workspace/harbor_task/environment/starter/method.py:16-28；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/sampling.py:46-74
- budget_comparability：pass；同16图、512²、空提示、bf16、25步双向100NFE、同A100与8核；历史数值评分308.0607/309.2354秒。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt
- implementation_correctness：pass；正常加载与16图原始指标；可信数值层只接收节点；静态检查无直接实现故障反证，非本次动态复现。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/scorer-stdout.txt；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/scorer-stdout.txt；workspace/harbor_task/tests/local_backend.py:80-88
- selection_integrity：pass；确定性正式B/R单次完整集均有保存，没有挑训练seed；历史探索不作当前独立轨迹认可。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；expert_evidence/run_summary.json:26-31

Baseline八项专项：
项目 | 状态 | 事实与证据
--- | --- | ---
B01 | 通过 | Starter=正式Baseline=评分B，基于包内默认time_shift实现，有任务适配性；不要求换SOTA。 证据：workspace/harbor_task/environment/starter/method.py:16-28；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/sampling.py:46-74；workspace/harbor_task/tests/anchors.json:9-19
B02 | 通过 | 非训练固定模型；编码返回mean，没有采样噪声；已有原始加载和逐图结果，无实现故障反证。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/scorer-stdout.txt:4-24；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/modules/autoencoder.py:268-275
B03 | 通过 | B/R同模型/数据/评分器/100 NFE，历史同A100-40GB、8核，数值成本约308/309秒。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt
B04 | 通过 | 默认偏移映射固定，Reference换节点生成；共同冻结设置一致。未见只对B异常配置或不同选择集反证；不将Agent搜索时长等同单次B/R预算。 证据：workspace/harbor_task/environment/starter/method.py:18-28；workspace/reference/method.py:9-13；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt.protocol；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt.protocol
B05 | 不适用 | 无训练seed或随机评价重复；固定模型确定性单次正式测量合法，不虚构五seed或把16图当重复。 证据：workspace/harbor_task/tests/run_eval.py:196-239；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/modules/autoencoder.py:268-275
B06 | 通过 | 正式B就是官方Starter角色；anchors另列uniform=17.8567344417，低于B，但无原始同协议uniform记录，不把该自述当独立验证。 证据：workspace/harbor_task/tests/anchors.json:49-55；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json#overall；workspace/harbor_task/environment/starter/method.py
B07 | 通过 | 对比只改变26节点；固定模型、数据、指标、NFE一致，实际节点JSON可读；未见额外训练数据或资源反证。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/schedule.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/schedule.json；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt.protocol；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt.protocol
B08 | 通过 | 说明披露默认偏移B、正弦R及固定模型/100 NFE；锚点有角色映射。当前大模型获取和镜像仍待交付闭环，属于部署限制。 证据：expert_evidence/专家作业说明文档.md:3-7；workspace/harbor_task/tests/anchors.json:9-45
总评：在已提供历史同协议材料范围内合理；未发现Baseline实现故障、无依据削弱或挑seed的直接红旗。

Reference方法与全部正式原始数值
正式seed：无（不适用）；actual_v341各1次确定性完整16图运行，不虚构训练seed或随机重复。两份真正Trial result：
optimization_evidence/baseline_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__QTvtVoh/result.json
optimization_evidence/reference_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__DcGEtFn/result.json
原始逐图指标来源：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json#per_image；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json#per_image。原始run stdout均为相邻scorer-stdout.txt，含模型加载、warning及逐图结果；不是人工占位日志。
分布 | B均值 | R均值 | Δ(dB) | 相对Δ | U | 归一化R | 重复运行n−1标准差
--- | --- | --- | --- | --- | --- | --- | ---
overall | 19.52888240950000 | 20.45734928422148 | 0.92846687472148 | 4.7543267211% | 21.53562330903075 | 0.46267401782591 | B/R均null（各1次）
public | 19.36157713750107 | 20.22483156573475 | 0.86325442823368 | 4.4585956098% | 21.21974298189495 | 0.46457340222787 | B/R均null（各1次）
hidden | 20.03079822549678 | 21.15490243968167 | 1.12410421418489 | 5.6118792748% | 22.48326429043815 | 0.45835668442236 | B/R均null（各1次）
Public的U=21.21974298189495可对应包内historical_v341/final-method-clean/gpt-historical-metrics.json#overall.psnr；full U=21.535623309030747仅能核到anchors声明，anchors.source引用的原始full指标文件不在ZIP，应补已有真实来源。上述full归一化是按已声明U复算，不冒充其原始来源已核验。
确定性依据：autoencoder.py:268-275在sample分支也直接return mean，run_eval.py没有采样噪声；单次B/R没有可估计的运行间σ_B，不硬填0、不套3σ、不恢复统一5%。图片间样本标准差仅描述数据差异：full B=4.3379757414、R=3.2262582889，绝不能用作随机评估显著性。训练seed与Seed模型名、16图与随机重复均已区分。
全部16图原值：
split/image | B PSNR | R PSNR | R−B
--- | --- | --- | ---
public/coco_000000000139 | 19.62636292709934 | 19.32464720550117 | -0.30171572159817
public/coco_000000000285 | 23.32860565590497 | 23.91692369326247 | 0.58831803735750
public/coco_000000000632 | 14.75411310733161 | 15.14518343235521 | 0.39107032502360
public/coco_000000001000 | 18.86575865111160 | 15.95004196239751 | -2.91571668871409
public/coco_000000001268 | 16.61296046977890 | 16.66736941477066 | 0.05440894499176
public/coco_000000001425 | 20.51325555197879 | 20.23342452732179 | -0.27983102465699
public/coco_000000001490 | 18.43597706210436 | 19.45113617609398 | 1.01515911398962
public/coco_000000039769 | 17.28340529299807 | 17.22906433298442 | -0.05434096001365
public/repo_art | 27.04654857978757 | 24.14868821912924 | -2.89786036065833
public/repo_boy | 17.67913042447747 | 23.81990431546254 | 6.14077389098507
public/repo_horse | 25.02130609535487 | 25.25619062871364 | 0.23488453335878
public/repo_nobel | 13.17150183208523 | 21.55540488082434 | 8.38390304873911
hidden/coco_000000000802 | 16.95575608450997 | 18.57757462642744 | 1.62181854191747
hidden/coco_000000001296 | 19.80417802429648 | 19.63220961421799 | -0.17196841007848
hidden/repo_cartoon | 15.26323983159789 | 23.11495893413448 | 7.85171910253660
hidden/repo_hiking | 28.10001896158280 | 23.29486658394676 | -4.80515237763605

三项内容门：
项目 | 状态 | 结论与证据
--- | --- | ---
G01 | pass | 函数接口与guard允许任意确定性26节点算法；Reference将默认偏移映射改为正弦映射，属于方法变化。 证据：workspace/harbor_task/instruction.md:4,13,20；workspace/harbor_task/tests/candidate_worker.py:149-158；workspace/harbor_task/tests/evaluate_candidate.py:67-78；workspace/reference/method.py:9-13
G02 | pass | 在现有历史同协议材料范围内合理：正式B沿用Starter默认偏移节点，固定模型非训练、同100 NFE及同16图；未发现故障基线或不公平削弱的直接反证。 证据：workspace/harbor_task/environment/starter/method.py:16-28；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt.protocol；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt.protocol；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/schedule.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/schedule.json
G03 | pass | 历史Public/Dev改善0.8632544282 dB，归一化0.4645734022在范围内；完整集改善0.9284668747 dB。确定性评估不套3σ或统一5%门槛。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json#public；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json#public；expert_evidence/historical_v341/final-method-clean/gpt-historical-metrics.json#overall；workspace/harbor_task/tests/anchors.json:33-45

两轨时长、八字段与格式
当前trajectory_codex.json与trajectory_seed.json均为{"rounds":[]}；run_summary.json:14-23明确未启动且effective_seconds=null，不能算已完成10h或7h例外。任务虽非训练，仍缺短迭代例外、各≥3闭环、真实起止/扣除区间、下一方向与best_method干净复验。
历史原始轨迹分别102轮和47轮，记录的八字段齐全，但不等于当前合格轨迹。专家作业说明文档.md:9披露历史Seed访问pilot/正弦Reference；本次不读历史QA，不把历史墙钟时间或旧报告接受值折算成当前有效秒数。
workspace/、expert_evidence/、optimization_evidence/职责可识别。当前缺唯一best_method及轨迹，证据目录缺训练证据说明.md（本题应说明非训练）和comparison_summary.json；原始actual_v341记录可保留并用汇总映射，无需为确定性任务造seed目录、训练模型或空model/。
deployment_source/README.md:12-15的deployment/、package/示例非ZIP真实目录，建议改为实际路径。公开图双份是可合并建议，不单独扣错。历史原件可保留但须与当前材料分层。

核心结论与Harbor边界
确定失败：两条当前独立轨迹为空且未启动，缺有效研究时长和独立演进；这些不是五类放宽事项。其余核心待核：不可变镜像/模型获取入口、目标Harbor原生配置兼容性。
显式separate、各自单容器/GPU、公开Dev、4张私有Hidden、/workspace/solution移交和/logs/verifier/reward.json静态链路存在。Agent context不含私有材料；受限候选Landlock/seccomp、降权、资源限额实现可识别。当前manifest中runtime/worker/repo/images逐项存在并与SHA256一致。
包内最终CPU测试日志显示9项边界和9项协议检查通过及权限fixture通过；本次只读这些日志，不能称GPU镜像、Harbor移交、正式score或12h稳定性已验收。
Harbor项目 | 状态 | 事实与证据
--- | --- | ---
H01 | pass | 任务三目录清楚；正式/tests包含4张私有图和12张公开图，评分走full；Agent context只含公开材料，不含Reference或专家证据。 证据：workspace/harbor_task/task.toml；workspace/harbor_task/tests/asset_manifest.json#images；workspace/harbor_task/tests/local_backend.py:116-124；workspace/harbor_task/environment/Dockerfile:4-9
H02 | manual | 显式separate与artifacts存在；包内未锁定目标Harbor版本，也无当前原生schema加载记录。未将TOML文本可读当作平台接受。 证据：workspace/harbor_task/task.toml:1-28；expert_evidence/deployment_source/README.md:34
修改：锁定实际Harbor版本/provider，按该版本原生配置解析核对schema_version、verifier.environment和GPU字段。 验收：目标Harbor版本/provider、当前task.toml的原生解析输出或同版本成功Trial配置。
H03 | manual | 双Dockerfile COPY均在各自context内，候选路径移交一致；共同基镜像仅本地tag，源码COPY weights所需文件不在ZIP，也没有平台不可变资产入口，部署依赖不闭环。 证据：workspace/harbor_task/environment/Dockerfile:1-9；workspace/harbor_task/tests/Dockerfile:1-7；workspace/harbor_task/task.toml:2；expert_evidence/deployment_source/runtime-image/Dockerfile:13-16；expert_evidence/deployment_source/README.md:9-20
修改：补平台可读取的不可变运行镜像入口/digest或合法模型资产入口；按包内实际目录修正文档构建命令，锁定目标Harbor版本，完成原生配置解析与单容器GPU评分验证。 验收：镜像版本/digest及可获取依据、权重提供方/精确安全入口/版本/大小/SHA256、实际Harbor版本和配置解析输出、公开Dev及有效候选独立Verifier原始日志/结果/reward。无需向候选提供秘密。
H04 | pass | 正式入口使用绝对/tests路径；可信输出写/logs/verifier/reward.json数值对象；清理、原子写和失败分类可识别。 证据：workspace/harbor_task/tests/test.sh:1-4；workspace/harbor_task/tests/evaluate_candidate.py:20-25,44-64,237-240,265-270
H05 | not_applicable | 当前版本未交Job/Trial调用配置；历史v3.4.1配置仅作历史评分来源，不充当当前部署覆盖。 证据：expert_evidence/run_summary.json:39-43
H06 | manual | 当前版本明确NOT_RUN_CURRENT_VERSION；已有历史v3.4.1 NOP与本版不同，不能当作本版验收；依当前口径非阻断待补。 证据：expert_evidence/run_summary.json:39；optimization_evidence/baseline_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__QTvtVoh/result.json
修改：后续具备授权目标环境时复用同版本已有记录或跑NOP；保持未运行声明，不自动启动。 验收：同一当前Trial的config/result、reward、非空日志、任务版本和separate/实际移交对应依据。

21项实现：
检查项 | 状态 | 原因与证据
--- | --- | ---
QA01 | pass | 题面八部分齐全；没有论文标题、论文编号、解法仓库或参考方法提示。 证据：workspace/harbor_task/instruction.md:1-33
QA02 | pass | 主分数为未裁剪(PSNR-B)/(U-B)，B映射0、U映射1；score_clipped仅展示，reward使用score。 证据：workspace/harbor_task/tests/run_eval.py:255-269；workspace/harbor_task/tests/local_backend.py:127-133；workspace/harbor_task/tests/evaluate_candidate.py:265-270
QA03 | pass | Reference源码及入口存在；已有确定性历史正式结果、原始stdout和Trial result，Public归一化0.4645734022。目录为等价actual_v341，不虚构训练seed。 证据：workspace/reference/method.py；workspace/reference/solve.sh；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/scorer-stdout.txt；optimization_evidence/reference_runs/actual_v341/harbor-job/rfsteps-v341-build-check-2026100__DcGEtFn/result.json
QA04 | not_applicable | 固定模型、均值编码、无采样噪声的确定性评估；单次正式运行的样本标准差不可估计，不将图片方差或锚点重复数当作σ_B。 证据：expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/modules/autoencoder.py:268-275；workspace/harbor_task/tests/run_eval.py:196-239
QA05 | pass | 节点形状/端点/范围/顺序、标准库来源、CPU/内存/墙钟与文件/网络/子进程限制均有生效调用实现；未支持内核隔离时失败关闭。 证据：workspace/harbor_task/tests/evaluate_candidate.py:67-95,122-224；workspace/harbor_task/tests/candidate_worker.py:139-174；workspace/harbor_task/tests/isolation.py:28-106
QA06 | pass | 单命令test.sh调用可信入口；正常有限标量写reward.json；Hard Gate与格式/超时/资源/基础设施分类清楚。 证据：workspace/harbor_task/tests/test.sh:1-4；workspace/harbor_task/tests/evaluate_candidate.py:44-64,265-283
QA07 | pass | 仅题面检查：没有Hidden样本、标签或可恢复答案。 证据：workspace/harbor_task/instruction.md:1-33
QA08 | pass | 仅题面检查：没有Reference代码、具体参考配置或参考分数。 证据：workspace/harbor_task/instruction.md:1-33
QA09 | pass | 入口清理旧reward/metrics；可信侧用同目录临时文件os.replace写最终结果。 证据：workspace/harbor_task/tests/evaluate_candidate.py:20-25,237-240,268-270
QA10 | pass | Agent为普通UID、冻结目录去写；Verifier的/tests为root私有；数值侧验证评分器、资产和权重清单哈希，候选只移交节点。 证据：workspace/harbor_task/environment/Dockerfile:10-18；workspace/harbor_task/tests/Dockerfile:3-5；workspace/harbor_task/tests/local_backend.py:22-58,80-88
QA11 | pass | 目标、环境、指标、可改范围、硬边界、提交、迭代与完成条件均有实质内容。 证据：workspace/harbor_task/instruction.md:3-33
QA12 | pass | ZIP成员清点无.git元数据；不声称做过不可达Git对象分析。 证据：@inventory
QA13 | pass | 已有B/R完整确定性同协议16图正式原值及有效状态；同100 NFE、同评分器/模型/数据哈希，正向改善。没有训练seed，不要求重复训练。 证据：optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/output/metrics.json；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/output/metrics.json
QA14 | pass | 26节点非均匀分配可表达分段/误差平衡/嵌入量化感知等题面未提示方向；公开Reference归一化约0.465，仍有空间。 证据：workspace/harbor_task/instruction.md:13-22；workspace/harbor_task/tests/candidate_worker.py:149-158；workspace/reference/method.py:9-13；workspace/harbor_task/tests/anchors.json:33-45
QA15 | manual | 历史A100完整Trial约428/431秒、8核1卡；当前配置仍A100，目标H20/L20耗时未实测且包内说明已暂缓。未发现已证实超限。 证据：workspace/harbor_task/task.toml:10-28；optimization_evidence/baseline_runs/actual_v341/gpu-owned-job/receipt.json#receipt.scoring_seconds；optimization_evidence/reference_runs/actual_v341/gpu-owned-job/receipt.json#receipt.scoring_seconds；expert_evidence/run_summary.json:5-9
修改：保留A100历史设备与时间；后续平台确认硬件配置，需迁移时补H20/L20估算或已有计时，不将旧卡改标新卡。 验收：平台卡型确认，或后续H20/L20设备/资源、完整单轮计时范围和日志。
QA16 | fail | 当前两轨rounds均为空，摘要明确current isolated run not started、有效秒数null；没有满足各10h或7h例外的运行。历史墙钟不能替代当前独立有效时长。 证据：expert_evidence/trajectory_codex.json:2；expert_evidence/trajectory_seed.json:2；expert_evidence/run_summary.json:10-31
修改：在不接触Reference、Hidden、pilot和另一轨的独立上下文中，从同一冻结题包各自启动两组模型；保存真实公开方法→评分→保留/回退闭环，按起止及扣除区间核算有效时长。 验收：两份非空八字段轨迹、实际模型/档位、起止和排队/安装/故障扣除记录、各轨最终方法和干净公开复验。默认各≥10h；使用各≥7h例外时补齐非训练、迭代很短、各≥3闭环、后续方向和最佳方法复验全部依据。
QA17 | manual | separate、公开Dev、Hidden、artifacts和reward静态链路存在；当前不可变镜像/权重入口及目标Harbor schema仍未核验。NOP缺口单列非阻断。 证据：workspace/harbor_task/task.toml；workspace/harbor_task/tests/test.sh；expert_evidence/deployment_source/README.md:7-20,34-42
修改：补平台可读取的不可变运行镜像入口/digest或合法模型资产入口；按包内实际目录修正文档构建命令，锁定目标Harbor版本，完成原生配置解析与单容器GPU评分验证。 验收：镜像版本/digest及可获取依据、权重提供方/精确安全入口/版本/大小/SHA256、实际Harbor版本和配置解析输出、公开Dev及有效候选独立Verifier原始日志/结果/reward。无需向候选提供秘密。
QA18 | fail | 当前两条轨迹0轮，未交当前best_method；历史轨迹不能替代，专家说明承认历史Seed访问pilot和正弦Reference。缺合格独立方法演进证据。 证据：expert_evidence/trajectory_codex.json:2；expert_evidence/trajectory_seed.json:2；expert_evidence/run_summary.json:14-23,38；expert_evidence/专家作业说明文档.md:9
修改：在不接触Reference、Hidden、pilot和另一轨的独立上下文中，从同一冻结题包各自启动两组模型；保存真实公开方法→评分→保留/回退闭环，按起止及扣除区间核算有效时长。 验收：两份非空八字段轨迹、实际模型/档位、起止和排队/安装/故障扣除记录、各轨最终方法和干净公开复验。默认各≥10h；使用各≥7h例外时补齐非训练、迭代很短、各≥3闭环、后续方向和最佳方法复验全部依据。
QA19 | not_applicable | 确定性评价，无训练随机seed或随机评估有效提升阈值需要在题面说明。 证据：workspace/harbor_task/instruction.md:4,7；expert_evidence/deployment_source/runtime-image/repo/RF-Solver-Edit/FLUX_Image_Edit/src/flux/modules/autoencoder.py:268-275
QA20 | pass | 可改method.py、可查公开文献；候选禁止联网/子进程，公开迭代在自己的单容器/GPU内，工具边界清楚。 证据：workspace/harbor_task/instruction.md:13-22
QA21 | manual | task_binding有profile但task_content_sha256=null；官方指纹工具未取得。不猜算法、不为历史结果补造当前指纹。 证据：expert_evidence/run_summary.json:33-37
修改：获得真实发布task_fingerprint.py后，对四项完整范围生成指纹，输出置于四项之外，保留历史版本原值。 验收：真实脚本版本/哈希、运行输出及当前run_summary.task_binding；仅此缺口不阻断当前内容结论。

五类放宽待补（不会单独导致返修）
- fingerprint_hash：当前task_content_sha256=null；官方工具未取得；历史bridge/source哈希角色不同保留原值。 建议：真实指纹工具到位后补四项范围指纹，不编造当前哈希。
- G02_G03_card_migration：历史真实卡型A100；当前包继续请求A100，H20/L20补证暂缓。 建议：后续迁移如影响数值，补已有可比性/锚点证据；不把换卡补跑作为当前内容通过的唯一门槛。
- QA15_target_card_timing：历史A100耗时存在，H20/L20耗时未测。 建议：后续据实际卡型补计时/估算来源。
- NOP：本版NOP未运行；历史版本NOP不替代当前。 建议：目标环境具备后补同版本同Trial链路记录；仅此缺口非阻断。
- QA03_QA13_card_rescore：新部署/迁移未重算，历史同协议原始B/R可追溯。 建议：保留历史分数和真实设备，后续迁移补证不强制当前全seed重跑。

工具缺口
真实官方0.3.10发布包、独立格式/GPU/指纹工具未取得/未运行；没有改版本、没有旧工具改标PASS。后续取得真实工具后记录版本/SHA256，按真实CLI验收。目标Harbor版本/provider未暴露，未做原生schema加载。

本次运行边界
本次仅zipfile读取源码、文本、配置、指标及元数据，并用可信自写算术/哈希复算。未执行或导入提交代码，未运行Docker、Verifier、安装、训练、模型调用或权重反序列化，未做GPU/12h压力测试。当前版本运行状态保持not_run。
未作完整二进制隐私扫描，主控另行逐附件扫描。当前workspace/deployment_source可读文本未见作者home/邮箱/私网地址；这不是完整发布隐私PASS。

最多三条具体风险
- 两条当前轨迹完全空，历史Seed访问pilot/Reference的披露意味着旧轨不能证明独立探索。
- 运行基镜像与权重缺可用不可变入口；源码和CPU回归日志不能证明GPU部署已可用。
- full上限U的source原始路径不在ZIP；应补已有原始来源，避免将可复算分数误当成锚点已完整核验。

可直接复制给专家的说明
本轮指定完整ZIP静态复核结论：FAIL / NOT READY。方法空间与历史Baseline/Reference比较可接受，但当前两条独立轨迹均为空，不能作为完成提交。

1. 补当前合格独立双轨与真实有效时长。
事实：expert_evidence/trajectory_codex.json:2、trajectory_seed.json:2均为空；run_summary.json:14-23明确未启动、秒数null、轮数0，best_method为null。专家作业说明文档.md:9披露历史Seed访问pilot与正弦Reference，不能用旧轨替代干净当前轨。
原因：缺少当前独立方法→公开评分→决策及有效时长，属于核心问题，不在五类放宽内。
最小修改：从同一冻结公开题包、隔离上下文各自重新形成独立轨迹，禁止看到另一轨、Reference、Hidden或pilot；按实际起止和扣除区间核算。
验收材料：两份非空八字段轨迹、模型/档位与时间核算、各轨唯一最佳方法与干净公开复验。默认各10h；若选非训练各7h例外，完整说明迭代很短、各≥3闭环、下一方向、复验与时长扣除依据。

2. 闭合部署与Harbor配置证据。
事实：两份Dockerfile依赖rfsteps-numerical:3.5.0-a100本地tag；deployment_source/runtime-image/Dockerfile:14 COPY weights，但ZIP没有weights；README:9-20要求外部原权重，仅列准备脚本/哈希，无平台认可不可变入口/digest。当前Harbor版本/provider及原生配置解析未提供。README:12-15构建示例目录也不对应ZIP真实路径。
原因：源码包不能证明目标平台可获得必需资产并接收配置；这是未核验，不是已确认构建失败。
最小修改：提供可拉取不可变运行镜像入口/digest或合法资产安全入口，记提供者、精确入口、操作、版本、大小、哈希；修正文档实际路径，锁定目标Harbor并做原生解析。
验收材料：版本/provider、原生配置解析、实际镜像/资产身份和有效候选公开Dev与独立Verifier的原始结果/reward/log。保持历史A100原值，不把CPU fixture称为GPU实跑。

3. 整理已有数值证据，勿改分数凑范围。
历史原始完整集B=19.528882409499996、R=20.45734928422148，差0.928466874721483 dB；Public归一化0.4645734022278672已在要求范围。保留actual_v341原始metrics/stdout/Trial result，新增非训练适用性说明和comparison_summary，逐一指向真实result路径。训练seed与训练checkpoint不适用，不造空目录。full U在tests/anchors.json声明21.535623309030747，但source原始路径不在ZIP；请补已有同协议上限原始full指标与方法身份，或更正为真实包内来源，不倒推U。

非阻断待补：官方四项指纹/哈希；H20/L20迁移影响；目标卡耗时；当前NOP及版本关联；仅由换卡引出的重算/标定。保留未运行/待核，不单独因此强制全量重跑。官方0.3.10及独立格式/GPU/指纹工具尚未取得，取得后按真实版本CLI验收，不改旧工具版本冒充。

边界：本轮仅静态阅读ZIP和可信算术/哈希核对，未执行任何提交代码/Docker/训练/外部模型/Verifier或加载权重。未作当前Harbor、GPU、稳定性或完整二进制隐私验收。没有把上述待核隐藏成PASS，也没有根据缺证据推断作者动机。
