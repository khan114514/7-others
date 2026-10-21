# Auto2912 独立静态内容复审

结论：FAIL / NOT READY。研究三门通过，但当前 V4 的构建、资源网络条件与逐轮轨迹格式存在明确失败。

审查类型：local-v4-content-review；session=stssdl_a。
审查模型：GPT-6 (system identity; exact deployment variant unavailable)。
ZIP SHA256：`18ee42736dcb71d492cf47a8f83a3d863d148cdb4b1a3d4a9488a8c8087e3dba`。
未使用包内或包外旧QA结论；未执行或导入提交代码。

## 优化面与方法
Executable fitted correction of frozen ST-SSDL upstream forecasts,with fresh fitting and independent reload per seed.

Baseline: Unchanged Starter identity returns the shared trained predictor output. fit produces48-byte state; no SGD required. Upstream log records63epochs and early stop.

Reference: Nine conditional features: base,recent history,slope,anchor,sin/cos clock,horizon,node. HistGradientBoostingRegressor fits residuals for100iterations with seed-specific initialization; adds0.5times predicted residual.

## 全部正式成对跑分
| seed | Baseline MAE | Reference MAE | 改善 | 质量门 |
|---|---:|---:|---:|---|
| 17 | 15.421857156583 | 15.094631864779 | 0.327225291804 | 双方通过 |
| 29 | 15.421857156583 | 15.100112152135 | 0.321745004449 | 双方通过 |
| 43 | 15.421857156583 | 15.093119894249 | 0.328737262335 | 双方通过 |

B mean=15.421857156583; B sample std(n-1)=0.000000000000; R mean=15.095954637054; R sample std=0.003679023302.
Positive improvement=0.325902519529; relative=2.113251%; normalized R=0.688520079296; U=14.948519443815; remaining headroom=0.147435193239.
Baseline 是确定性 identity；Reference 是随机拟合，不能因 σ_B=0 宣称整体评价确定。严格正改善、3σ 及归一化门槛均满足。六个模型的流式 hash、大小与索引匹配，fit/reload 原始日志齐全，reload 差为0。worker 日志的简短 JSON 与代码实际输出一致，不能仅因日志短而认定伪造。

## Baseline 专项
总评：在已提供材料范围内合理。Starter、正式 Baseline 和评分 B 一致，未发现直接弱化、错误训练或挑 seed 反证。
| 项目 | 状态 | 事实和证据 |
|---|---|---|
| B01 | 通过 | Formal B is unchanged identity Starter on a shared trained predictor; appropriate for a fixed-predictor correction task. Evidence: workspace/harbor_task/environment/engine/starter.py:5; workspace/harbor_task/environment/engine/actor_worker.py:33; expert_evidence/provenance/shared-backbone-training.run.log:251; optimization_evidence/baseline_runs/seed_17/result.json; expert_evidence/专家作业说明文档.md:13 |
| B02 | 通过 | Identity requires no SGD. Shared backbone actually stopped at epoch63/200 with 42147 batches; all formal fit/reload exits are0. No wrong-checkpoint or label-misalignment evidence. Evidence: workspace/harbor_task/environment/engine/starter.py:5; workspace/harbor_task/environment/engine/actor_worker.py:33; expert_evidence/provenance/shared-backbone-training.run.log:251; optimization_evidence/baseline_runs/seed_17/result.json; expert_evidence/专家作业说明文档.md:13; expert_evidence/provenance/shared-backbone-training.run.log:248 |
| B03 | 通过 | Same calibration/Dev data, seeds17/29/43,2CPU,180s fit and60s reload limits. Fitting-cost differences are allowed method variables. Evidence: workspace/harbor_task/environment/engine/grader.py:104; workspace/harbor_task/environment/engine/grader.py:127; optimization_evidence/comparison_summary.json |
| B04 | 通过 | Identity has no anomalous hyperparameters. Reference fixes a100-iteration residual tree and0.5 mixture. Pilot/frozen-anchor records precede formal R. Evidence: workspace/harbor_task/environment/engine/starter.py; workspace/reference/method.py:22; expert_evidence/provenance/anchors-frozen-before-reference.json; expert_evidence/provenance/pilot-upper-estimate.json |
| B05 | 通过 | Complete3 paired seeds enter random/numpy seeds and Reference random_state. Same method/protocol hashes across each role. Identical identity states are semantically expected. Evidence: workspace/harbor_task/environment/engine/actor_worker.py:33; workspace/reference/method.py:23; optimization_evidence/comparison_summary.json |
| B06 | 通过 | Identity is the trained upstream forecast, not an untrained/random predictor. No unexplained stronger same-protocol sanity anchor is shown. Evidence: workspace/harbor_task/environment/engine/starter.py:9; expert_evidence/专家作业说明文档.md:3 |
| B07 | 通过 | Only correction method changes; data,mask,shared predictor and evaluation remain fixed. No claim of independent neural retraining or use of Hidden for formal Dev comparison. Evidence: workspace/reference/method.py; workspace/harbor_task/environment/engine/grader.py:68; expert_evidence/run_summary.json#independent_neural_training_replicates |
| B08 | 通过 | Expert discloses identity role,shared initialization,fresh correction fits and raw-only to frozen-protocol B reconciliation. Evidence: expert_evidence/专家作业说明文档.md:3; expert_evidence/专家作业说明文档.md:29; expert_evidence/provenance/baseline-frozen-protocol-recheck.json |

五维事实：
- training_sufficiency: Identity does not require training; upstream early stop and correction completion are documented.
- method_representativeness: Reasonable naive fixed-predictor correction anchor; SOTA is not required.
- budget_comparability: Same full seeds/data/scorer/2CPU cap. B and R take approximately0.31-0.38s and5.87-7.12s per seed.
- implementation_correctness: Identity and residual-tree execution paths are coherent; shape,finite and independent reload gates pass in existing records.
- selection_integrity: All3 seeds; pre-anchor raw identity equals frozen-protocol recheck. B/U frozen before formal R.

## 三项内容门
| 项目 | 状态 | 理由与证据 |
|---|---|---|
| G01 | pass | Executable fit/predict permits structural methods. Guard limits source/resources rather than constants; Reference learns residuals. Evidence: workspace/harbor_task/instruction.md:13; workspace/harbor_task/environment/engine/grade_server.py:44; workspace/harbor_task/environment/engine/actor_worker.py:24; workspace/reference/method.py:18 |
| G02 | pass | Baseline is reasonable within submitted evidence: identity shares the trained predictor,Starter and scoring anchor; no direct weakening/seed-selection evidence. Evidence: workspace/harbor_task/environment/engine/starter.py:5; workspace/harbor_task/environment/engine/actor_worker.py:33; expert_evidence/provenance/shared-backbone-training.run.log:251; optimization_evidence/baseline_runs/seed_17/result.json; expert_evidence/专家作业说明文档.md:13 |
| G03 | pass | Complete3 formal pairs valid. Improvement0.3259025195, sigma_B0 and normalized0.6885200793 satisfy positive improvement and[0.15,0.8]. Evidence: optimization_evidence/comparison_summary.json; optimization_evidence/baseline_runs/seed_17/result.json; optimization_evidence/reference_runs/seed_17/result.json; expert_evidence/provenance/pilot-upper-estimate.json; expert_evidence/provenance/anchors-frozen-before-reference.json |

## 双轨与有效时长

GPT track: gpt-6.1-sol/xhigh; packaged rounds=27; effective=36003.790908s (10.001053030h).
Round elapsed=36641.069331s; union exclusions=688.296345s; initial actual task work=51.017922s.
Residual trees, temporal/spatial calibration, neural mixtures and confidence diffusion. 11 controller scores succeeded; 16 failed with HTTP409.
Final Dev MAE=13.902094246542; normalized=3.210737004563. Evidence: expert_evidence/trajectory_codex.json.

Seed track: doubao-seed-2-1-turbo-260628/high; packaged rounds=79; effective=36008.387279s (10.002329800h).
Round elapsed=39019.004448s; union exclusions=3046.633039s; initial actual task work=36.015869s.
Node/horizon bias, anchor mixtures, ridge and clock features. 78 controller scores succeeded; 53 zero-effective ledger rounds are not 53 method cycles.
Final Dev MAE=15.165965823367; normalized=0.540610490806. Evidence: expert_evidence/trajectory_seed.json.

两轨各自均超过10h，不相加，不使用7h例外。全部27/79个账本的 elapsed-excluded=effective 与排除区间 union 均复算一致；原始事件区间无跨轮重叠。无模型最终复验与独立43239.755秒容器稳定记录不计入模型时间。现有成功分数逐项对应控制器原始输出，retained_best 与当时最好 MAE 一致。
逐轮必需八字段均存在，但正式 V4 要求恰八字段：GPT16轮多 original_status；Seed全部79轮多 method_summary_evidence，另1轮多 original_status。失败命令末行均为 HTTP409 Conflict，属于评分器 busy，不能统一归为算法失败。

## 格式对齐建议
Three top-level roles and complete formal paired evidence exist. Extra evidence is allowed;Compose and exactly-eight-key deviations are substantive current-contract issues.
- obsolete_layout: workspace/harbor_task/environment/Dockerfile — Use environment build context and a public Starter contained inside it.
- misplaced: expert_evidence/trajectory_codex.json — Move per-round extra metadata to a sidecar.
- misplaced: expert_evidence/trajectory_seed.json — Move79 per-round evidence objects to sidecar; keep concise1-3 sentence summaries.
- extra_allowed: expert_evidence/official_source/ — Optional upstream material may remain; consider separate expert-only attachment with license/privacy review.
- merge_candidate: optimization_evidence/训练证据说明.md — Reduce repeated expert narrative without losing formal evidence mapping.
- merge_candidate: optimization_evidence/baseline_runs/seed_17/result.json — Optionally add explicit schema/role/task_type and grouped method/protocol/execution indexes. Equivalent existing fields are not alone a rejection. The checkpoint.path is run-relative; disclose that base or use package-relative paths.

## 21项实现检查
| 项目 | 状态 | 原因与证据 |
|---|---|---|
| QA01 | pass | Eight substantive sections; no paper title,arXiv,repository or reference-answer clues in instruction. Evidence: workspace/harbor_task/instruction.md |
| QA02 | pass | Actual score=(B-mean)/(B-U),continuous,monotone and unclipped. B is formal identity aggregate; U is preregistered pilot estimate. Evidence: workspace/harbor_task/tests/engine/grader.py:187; workspace/harbor_task/tests/verifier_assets/protocol.json:15 |
| QA03 | pass | Reference source,entry,all3 results/raw logs,and actual model/reload records exist. Normalized Reference0.68852. Evidence: workspace/reference/method.py; optimization_evidence/reference_runs/result.json; optimization_evidence/reference_runs/seed_43/model/artifact.json |
| QA04 | pass | Reference is stochastic. All3 pairs were recalculated with n-1 Baseline sample standard deviation: delta>0 and delta>=3sigma_B=0. Evidence: optimization_evidence/comparison_summary.json; optimization_evidence/baseline_runs/seed_17/result.json; optimization_evidence/reference_runs/seed_29/result.json |
| QA05 | fail | Source,memory,CPU,wall-time and prediction gates exist; instruction/TOML still require unsupported no-network and storage_mb conditions. Evidence: workspace/harbor_task/instruction.md:16; workspace/harbor_task/task.toml:18; workspace/harbor_task/tests/engine/grader.py:39 |
| QA06 | pass | Single test.sh->grader entry; scalar reward and classified failure handling exist. Evidence: workspace/harbor_task/tests/test.sh; workspace/harbor_task/tests/engine/grader.py:217 |
| QA07 | pass | Instruction-only check: no Hidden examples,labels or recoverable answers. Evidence: workspace/harbor_task/instruction.md |
| QA08 | pass | Instruction-only check: no Reference code,configuration or result. Generic executable interface is not an answer. Evidence: workspace/harbor_task/instruction.md |
| QA09 | pass | Formal entry deletes prewritten rewards; successful trusted reward uses pending file and atomic replace. Evidence: workspace/harbor_task/tests/test.sh:4; workspace/harbor_task/tests/engine/grader.py:220 |
| QA10 | pass | Assets/engine are frozen; UID1000+bwrap sees read-only source and allowed inputs. Labels are read after candidate exits; source hash is rechecked. Evidence: workspace/harbor_task/tests/Dockerfile:14; workspace/harbor_task/tests/engine/grader.py:111; workspace/harbor_task/tests/engine/grader.py:132; workspace/harbor_task/tests/engine/grader.py:157 |
| QA11 | pass | Goal,setting,metric,scope,boundaries,submission,iteration and completion contain substantive instructions. Evidence: workspace/harbor_task/instruction.md:3; workspace/harbor_task/instruction.md:24 |
| QA12 | pass | Inventory contains no .git directory; no claim of exhaustive unreachable-object audit. Evidence: @inventory |
| QA13 | pass | All3 formal paired seeds share protocol/metric; model records and raw logs exist. Every pair improves; gates true and fit/reload exits0. Evidence: optimization_evidence/comparison_summary.json; optimization_evidence/baseline_runs/result.json; optimization_evidence/reference_runs/result.json |
| QA14 | pass | Executable structural method space and unsaturated Reference. Submitted trajectories show legal conditional/spatiotemporal directions. Best score>1 is allowed for unclipped estimated U; U was not moved. Evidence: workspace/harbor_task/instruction.md:13; workspace/reference/method.py; expert_evidence/best_method/conditional_calibration.py; expert_evidence/trajectory_codex.json |
| QA15 | pass | Scoring task uses0GPU/2CPU. Full formal B/R take about1.07/19.19s; selected Dev348.12s and local Hidden382.68s are below2h. A100 belongs only to upstream preparation,not relabeled target-card scoring. Evidence: workspace/harbor_task/task.toml:16; optimization_evidence/baseline_runs/result.json; optimization_evidence/reference_runs/result.json; expert_evidence/provenance/native_harbor/gpt/forecast-final-dev-gpt-d8cbab87/verifier/test-stdout.txt; expert_evidence/provenance/forecast-export-manifest.json#backbone_training_device |
| QA16 | pass | Separate clock recomputation: GPT36003.790908s and Seed36008.387279s,each>=10h. Exclusion unions checked; no sum across tracks or credit for final no-model replay/stability. Evidence: expert_evidence/trajectory_codex.json#duration_seconds; expert_evidence/trajectory_seed.json#duration_seconds; expert_evidence/provenance/model_execution/gpt/initial-normal-model-work.json; expert_evidence/provenance/model_execution/seed/initial-normal-model-work.json |
| QA17 | fail | Separate,public Dev,private Hidden,handoff,reward and local NOP exist. Current package still relies on forbidden Compose/root context and unsupported network/resource conditions. Evidence: workspace/harbor_task/task.toml; workspace/harbor_task/environment/docker-compose.yaml; workspace/harbor_task/tests/docker-compose.yaml |
| QA18 | fail | Eight keys exist and scores/retention match. GPT adds original_status in16 rounds; Seed adds method_summary_evidence in79 and original_status in1. Extra keys violate the contract;409 failures need classification. Evidence: expert_evidence/trajectory_codex.json#rounds[1].original_status; expert_evidence/trajectory_seed.json#rounds[0].method_summary_evidence; expert_evidence/provenance/model_execution/gpt/round_0002/trusted-evaluation-command.json |
| QA19 | pass | Instruction specifies seeds17/29/43 and gates+strict positive improvement. No extra repeat-count prose was imposed. Evidence: workspace/harbor_task/instruction.md:10 |
| QA20 | pass | Scope,network/host rules and public evaluation tool strategy are stated. Current network incompatibility is handled under constraints. Evidence: workspace/harbor_task/instruction.md:13; workspace/harbor_task/instruction.md:16; workspace/harbor_task/instruction.md:22 |
| QA21 | manual | run_summary lacks task_binding/hash_profile/task_content_sha256. Trial member manifests are not substituted for official four-input fingerprint. Non-blocking under2026-10-21 policy. Evidence: expert_evidence/run_summary.json; expert_evidence/provenance/native_harbor/nop/task-source-manifest.json |

## Harbor 静态与已有记录
版本依据仅为包内0.23.0本地Trial和适配源码，未联网核验官方schema。已有四份当前任务成员清单均55项且与当前ZIP对应；各Trial的result与reward文件一致。NOP=0符合identity行为。现有适配分支成功不等于当前平台或本次动态验收。
| 子项 | 状态 | 原因与证据 |
|---|---|---|
| H01 | pass | Complete task root and Verifier entry/dependencies exist. Real Hidden injection assets and call chain exist; candidate receives features without private labels. Evidence: workspace/harbor_task/task.toml; workspace/harbor_task/tests/test.sh:5; expert_evidence/held_out_assets/INJECTION.md; expert_evidence/provenance/harbor_adapter/post_selection_harbor.py:40 |
| H02 | fail | Explicit separate is correct, but storage_mb and multiple no-network settings conflict with current V4 platform conditions. No official schema was loaded. Evidence: workspace/harbor_task/task.toml:9; workspace/harbor_task/task.toml:18; workspace/harbor_task/task.toml:22 |
| H03 | fail | Direct Compose files violate V4. Agent COPY requires task-root context; environment context cannot resolve prefixed sources or root files. Historical local adapter success does not waive V4. Evidence: workspace/harbor_task/environment/docker-compose.yaml:4; workspace/harbor_task/tests/docker-compose.yaml:1; workspace/harbor_task/environment/Dockerfile:5; workspace/harbor_task/environment/Dockerfile:10; expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:69 |
| H04 | pass | test.sh clears old rewards; trusted grader writes scalar score through pending+replace. Format/timeout/execution errors and hard gates are classified. Evidence: workspace/harbor_task/tests/test.sh:4; workspace/harbor_task/tests/engine/grader.py:217; workspace/harbor_task/tests/engine/grader.py:224 |
| H05 | pass | Existing local Trial configs select the intended task and explicit adapter; Verifier is enabled. Local version0.23.0. This does not certify target-platform adapter installation. Evidence: expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/config.json; expert_evidence/provenance/harbor_adapter/post_selection_harbor.py; expert_evidence/provenance/native_harbor/PATH_BINDING.md |
| H06 | pass | Same NOP Trial config/result/reward/log exist: nop,separate,finished,no exception,reward0 agreement. All55 task-source members match current package. Evidence supports the submitted local adapter branch only. Evidence: expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/config.json; expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/result.json; expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/verifier/reward.txt; expert_evidence/provenance/native_harbor/nop/task-source-manifest.json |

公开Dev：Agent通过solve.sh调用同环境受保护评分器，可反复公开反馈；公开校正数据可拟合，Dev标签不可用于fit。候选在UID1000和bwrap中执行，只读源码与允许输入；可信侧在候选结束后读取标签，pickle仅在候选隔离进程内重载。冻结资产和评分目录受权限保护，reward由可信侧写。
Hidden：真实特征/标签注入材料在专家侧，适配器仅在Verifier且Dev选优冻结后注入，test.sh按--hidden-assets选择，候选只获无标签输入。已有本地Hidden reward=3.0009272267；这不是平台正式Hidden。artifacts仅移交 /workspace/solution。
当前Agent Dockerfile在历史任务根context可以由旧Compose适配器构建；本报告没有把历史成功写成构建失败。问题是当前environment context不能解析这些根路径，且两个直属Compose明确违反当前V4。

## 非阻断待补
- QA21 fingerprint: Missing official four-input task_binding fingerprint is non-blocking. No official algorithm was guessed.
- G02/G03/QA03/QA13/QA15 migration: Current correction scoring isCPU;historical upstream A100 is separate. No automatic full retraining/recalibration solely for device migration.
- QA17/H06 NOP: Complete matching local NOP exists;not independently rerun. Future version association after migration may be supplemented under relaxation,but does not waive confirmed Compose failures.
- 指纹补证：获得真实官方脚本后提交脚本版本/SHA、四项全递归原始输出和run_summary.task_binding。结果放四项范围之外，不臆造算法，不因缺指纹要求重跑训练。

## 工具缺口与本次未实跑
官方0.3.10发布包及独立格式/GPU/指纹工具不可用，工具验收待补；本地旧0.3.2未被执行或改标为新版。内容审查已安全完成，不称官方PASS。
本次未执行提交Python/shell、Docker/Verifier/NOP/安装、训练或模型调用；未反序列化权重。目标平台Hidden、schema、资源落实和12h稳定性未本次实测。

## 隐私与外发边界
Author-machine absolute paths remain. Only types and member locations are reported; values are not copied. Runtime container paths must be distinguished from author-machine paths. No whole-binary/nested-archive strict certification or external release was performed.
- AUTHOR_ABSOLUTE_PATH: expert_evidence/provenance/forecast-export-manifest.json:5
- AUTHOR_ABSOLUTE_PATH: expert_evidence/provenance/shared-backbone-training.run.log:254
- AUTHOR_ABSOLUTE_PATH: expert_evidence/provenance/preparation-complete.json:3
Remediation: Regenerate author-machine paths as relative paths or stable logical identifiers. Preserve raw originals outside release attachments. Rebuild the complete ZIP and strictly scan each final artifact separately.
Acceptance evidence: Zero-blocker strict scan for final ZIP and every attachment,rebuilt member/hash manifests,and relative-only report references.

## 专家退回说明
本轮独立静态内容审查为 NOT READY。研究三门通过；以下当前V4核心问题需修订。

C1 Unsupported network/storage conditions
事实/定位：workspace/harbor_task/instruction.md:16; workspace/harbor_task/task.toml:9; workspace/harbor_task/task.toml:18; expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:51
最小修改：Remove unsupported storage_mb and no-network/allowlist requirements from current submission. Synchronize instruction,TOML and offline-only adapter; retain private-data/scoring confinement under actual platform networking.
复验验收：New complete ZIP with supported network/resource declarations and actual platform mapping; constraint call chain remains valid. Do not automatically rerun all training solely for this edit.

C2 Compose-dependent formal build/runtime and old task-root context
事实/定位：workspace/harbor_task/environment/docker-compose.yaml:1; workspace/harbor_task/tests/docker-compose.yaml:1; workspace/harbor_task/environment/Dockerfile:5; expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:72
最小修改：Remove direct Compose files and formal Compose/extra broker-container dependencies. Put public Starter inside environment and rewrite Agent COPY for environment build context; preserve separate Verifier and candidate artifacts handoff.
复验验收：Complete package,one-container startup contract,valid COPY/dependency/entry mappings for both contexts and platform adapter evidence. Dynamic build/handoff/reward evidence only in an authorized target environment.

C3 Trajectory rounds have extra keys and vague scorer-busy classification
事实/定位：expert_evidence/trajectory_codex.json#rounds[1]; expert_evidence/trajectory_seed.json#rounds[0]
最小修改：Keep exactly8 keys per round. Move original_status and method_summary_evidence into a sidecar with original records preserved. Classify409 scorer-busy separately from invalid methods and clarify ledger rounds versus effective method cycles.
复验验收：Both trajectory JSONs have exactly8 round keys,unchanged original scores/times/retention,and relative sidecar references. Failure classification traces to original trusted evaluation command.

外发门禁：Author-machine absolute paths remain. Only types and member locations are reported; values are not copied. Runtime container paths must be distinguished from author-machine paths. No whole-binary/nested-archive strict certification or external release was performed.
仅报命中位置：AUTHOR_ABSOLUTE_PATH expert_evidence/provenance/forecast-export-manifest.json:5; AUTHOR_ABSOLUTE_PATH expert_evidence/provenance/shared-backbone-training.run.log:254; AUTHOR_ABSOLUTE_PATH expert_evidence/provenance/preparation-complete.json:3
修改与验收：Regenerate author-machine paths as relative paths or stable logical identifiers. Preserve raw originals outside release attachments. Rebuild the complete ZIP and strictly scan each final artifact separately. Zero-blocker strict scan for final ZIP and every attachment,rebuilt member/hash manifests,and relative-only report references.

非阻断待补：缺官方四项指纹可后补；不因换卡自动全量重算重训；本包已有当前本地NOP，后续版本关联可后补，不豁免确定Compose问题。
工具缺口：官方0.3.10及格式/GPU/指纹工具验收待补，不冒充官方PASS。
本次只静态读取唯一完整ZIP，未执行未知代码、容器、评分、模型或外发。
