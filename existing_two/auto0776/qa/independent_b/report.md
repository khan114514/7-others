# auto0776 独立 V4 静态质检（reviewer B）

输入：`auto0776-uploaded-source-snapshot.zip`；842个成员。
SHA256：`df27ff7cd6fff4a7b10d199ce35daec4c8ed65df8aac6e07b50957163525cc7c`。

session_id=onedc_b；configured_model=gpt-6-astra（可信派发）；runtime_reported_model=not_exposed，runtime_version=not_exposed。此为 local-v4-content-review，不冒充官方0.3.10工具、平台验收或本次动态复跑。

## 优化面与实际执行边界

冻结图像 codec 的测试时量化方法设计。 minimize DISTS；public/hidden bpp 各≤B×1.005；full 24图均值；整数形状、有限值、确定性。

允许method.py内实现函数/辅助方法；solution_guard只核两文件、大小和冻结solve.sh。candidate_worker在UID25001、seccomp限制后导入候选，返回同形有限整数张量；run_eval双调用/输入指纹检查并在codec真实量化点挂接。真实rANS码流决定bpp，冻结decoder独立解码，可信侧算DISTS。证据：`workspace/harbor_task/tests/{solution_guard,candidate_client,candidate_worker,run_eval,score}.py`。

## Baseline合理性与Reference方法

τ=0.6 naive dead-zone：sign(y)*round(clamp(abs(y)-0.1))，冻结模型不训练；正式B与Starter一致。

逐 latent RDOQ：比较 round(y)±1，最小化平方误差+0.055×Gaussian 概率比特成本，依 scales 自适应选择。

Baseline总评：在已提供材料范围内合理。Starter、正式B和归一化锚点一致，经典/朴素方法及低分不是失败理由。

- training_sufficiency：不适用：冻结预训练模型、测试时量化，全部正式记录training=NOT_APPLICABLE。
- method_representativeness：Starter与正式B相同，τ=0.6位于目标低码率工作点；官方round的full bpp=0.024381849500868028超门，不能仅凭其DISTS更低否定本B。
- budget_comparability：同final-v8镜像、checkpoint、16+8图顺序及A10单卡，B/R各三次新评分进程。
- implementation_correctness：dead-zone函数接入真实codec量化点；真实rANS编码、独立解码；原始日志正常加载和完成，未发现功能故障。
- selection_integrity：全部三组B/R保留并与raw per_image均值一致；参数探索与正式final-v8统计分开。

| 项目 | 结论 | 事实与证据 | 最小修改与验收材料 |
|---|---|---|---|
| B01 | 通过 | 目标冻结模型/低码率门内，dead-zone是可解释朴素对照，不以年代或分数低判弱化。 `workspace/harbor_task/environment/starter/method.py:21`；`workspace/harbor_task/instruction.md:9`；`optimization_evidence/official-paper-reproduction.json#metrics` | 无需修改；仅限本次静态证据范围。  |
| B02 | 通过 | 无训练，量化/熵编码/解码流程正常；六份原始日志非空且指标一致。 `workspace/harbor_task/environment/starter/method.py:24`；`workspace/harbor_task/tests/run_eval.py:224`；`optimization_evidence/baseline_runs/seed_eval_replicate_1/run.log` | 无需修改；仅限本次静态证据范围。  |
| B03 | 通过 | B/R同数据、镜像、checkpoint、顺序和资源，各三次新进程。 `optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json#protocol`；`optimization_evidence/reference_runs/seed_eval_replicate_1/result.json#protocol` | 无需修改；仅限本次静态证据范围。  |
| B04 | 通过 | 未见只给B异常配置或不公平截短；固定τ/λ的正式方法包含不同决策规则。 `optimization_evidence/candidates/baseline_dz/method.py`；`optimization_evidence/candidates/reference_rdoq/method.py`；`optimization_evidence/lambda_sweep/README.md` | 无需修改；仅限本次静态证据范围。  |
| B05 | 通过 | eval_replicate为新评估进程，全部seed0，不冒充独立训练；方法与非seed配置固定。 `optimization_evidence/native_calibration_v8/original_records/raw/B_1/execution.json`；`optimization_evidence/native_calibration_v8/original_records/raw/B_2/execution.json`；`optimization_evidence/native_calibration_v8/original_records/raw/R_1/execution.json`；`workspace/harbor_task/environment/repo/onedc/src/inference.py:28` | 无需修改；仅限本次静态证据范围。  |
| B06 | 通过 | 已有官方round作为sanity；其码率超任务门，不是同门内可替代强锚点。 `optimization_evidence/official-paper-reproduction.json#metrics`；`workspace/harbor_task/tests/anchors.json` | 无需修改；仅限本次静态证据范围。  |
| B07 | 通过 | B→R只改quantize，冻结模型/熵模型/数据/评分相同。 `workspace/harbor_task/tests/run_eval.py:224`；`workspace/reference/method.py:24`；`workspace/harbor_task/environment/starter/method.py:24` | 无需修改；仅限本次静态证据范围。  |
| B08 | 通过 | 源码、工作点、方法、正式协议、三次结果与探索材料有披露；不采纳作者自评PASS标签。 `optimization_evidence/训练证据说明.md`；`optimization_evidence/comparison_summary.json`；`optimization_evidence/lambda_sweep/README.md` | 无需修改；仅限本次静态证据范围。  |

## 全部正式数值、来源与复算

固定模型确定性推理；三次新评分进程，实际seed0；不是三次训练。 所有正式B/R result的DISTS、bpp和原始24图均值一致；execution退出码均0，原始日志非空。

| 角色/重复 | full DISTS | full bpp | Public DISTS | Hidden DISTS | elapsed秒 |
|---|---:|---:|---:|---:|---:|
| B_1 | 0.11711087326208751 | 0.0204925537109375 | 0.12342559173703194 | 0.10448143631219864 | 37.6141786513 |
| R_1 | 0.11463754375775655 | 0.020459493001302086 | 0.12056996300816536 | 0.10277270525693893 | 37.6827918012 |
| B_2 | 0.11711087326208751 | 0.0204925537109375 | 0.12342559173703194 | 0.10448143631219864 | 83.1657996817 |
| R_2 | 0.11463754375775655 | 0.020459493001302086 | 0.12056996300816536 | 0.10277270525693893 | 37.4995034996 |
| B_3 | 0.11711087326208751 | 0.0204925537109375 | 0.12342559173703194 | 0.10448143631219864 | 37.099205035 |
| R_3 | 0.11463754375775655 | 0.020459493001302086 | 0.12056996300816536 | 0.10277270525693893 | 37.5208677916 |
| U_1 | 0.11314917107423146 | 0.02051628960503472 | 0.118447195738554 | 0.1025531217455864 | 59.7432863071 |
| U_2 | 0.11314917107423146 | 0.02051628960503472 | 0.118447195738554 | 0.1025531217455864 | 37.9552083419 |
| U_3 | 0.11314917107423146 | 0.02051628960503472 | 0.118447195738554 | 0.1025531217455864 | 127.30790794 |

逐项真实路径（JSON另保存完整metrics、execution、run.log、方法哈希）：

- B_1：`optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/native_calibration_v8/original_records/raw/B_1/out/metrics.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_1/run.log`。
- R_1：`optimization_evidence/reference_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/native_calibration_v8/original_records/raw/R_1/out/metrics.json`；`optimization_evidence/reference_runs/seed_eval_replicate_1/run.log`。
- B_2：`optimization_evidence/baseline_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/native_calibration_v8/original_records/raw/B_2/out/metrics.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_2/run.log`。
- R_2：`optimization_evidence/reference_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/native_calibration_v8/original_records/raw/R_2/out/metrics.json`；`optimization_evidence/reference_runs/seed_eval_replicate_2/run.log`。
- B_3：`optimization_evidence/baseline_runs/seed_eval_replicate_3/result.json`；`optimization_evidence/native_calibration_v8/original_records/raw/B_3/out/metrics.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_3/run.log`。
- R_3：`optimization_evidence/reference_runs/seed_eval_replicate_3/result.json`；`optimization_evidence/native_calibration_v8/original_records/raw/R_3/out/metrics.json`；`optimization_evidence/reference_runs/seed_eval_replicate_3/run.log`。
- U_1：`optimization_evidence/native_calibration_v8/original_records/raw/U_1/out/metrics.json`；`optimization_evidence/native_calibration_v8/original_records/raw/U_1/run.log`。
- U_2：`optimization_evidence/native_calibration_v8/original_records/raw/U_2/out/metrics.json`；`optimization_evidence/native_calibration_v8/original_records/raw/U_2/run.log`。
- U_3：`optimization_evidence/native_calibration_v8/original_records/raw/U_3/out/metrics.json`；`optimization_evidence/native_calibration_v8/original_records/raw/U_3/run.log`。

| split | B均值 | R均值 | U均值 | B/R样本标准差(n−1) | 改善 | 相对改善 | R归一化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| public | 0.12342559173703194 | 0.12056996300816536 | 0.118447195738554 | 0 / 0 | 0.0028556287288665771 | 2.313643944% | 0.5736041748666919 |
| hidden | 0.10448143631219864 | 0.10277270525693893 | 0.1025531217455864 | 0 / 0 | 0.0017087310552597046 | 1.63543986% | 0.8861267164836524 |
| full | 0.11711087326208751 | 0.11463754375775655 | 0.11314917107423146 | 0 / 0 | 0.0024733295043309622 | 2.111955479% | 0.6243098009518612 |

Public cap=0.020115795135498046；Hidden cap=0.021553459167480468。B public/hidden bpp=0.020015716552734375 / 0.02144622802734375；R=0.02001317342122396 / 0.021352132161458336；U=0.020081837972005207 / 0.02138519287109375。九次全部过门。

Public R→U余量0.0021227672696113586，full余量0.0014883726835250854。σ_B=0时不计算无定义比值，仍要求严格正向改善；不恢复统一5%门槛。Hidden单独归一化0.8861267164836524，不是本题Public/Dev范围门；公开0.573604和全量0.624310均在[0.15,0.8]。

## 三项内容门

| 项目 | 结论 | 事实与证据 | 最小修改与验收材料 |
|---|---|---|---|
| G01 | pass | 开放可执行quantize函数；guard限制文件/张量/资源而不限制常量。Reference为尺度感知RDOQ，非纯调参。 `workspace/harbor_task/instruction.md:25`；`workspace/harbor_task/tests/solution_guard.py:11`；`workspace/harbor_task/tests/candidate_worker.py:98`；`workspace/harbor_task/tests/run_eval.py:224`；`workspace/reference/method.py:24` | 无需修改；仅限本次静态证据范围。  |
| G02 | pass | 已有正式记录范围内Baseline合理；未发现故障/少训练/弱化反证。硬件迁移影响与交付缺口单列。 `workspace/harbor_task/environment/starter/method.py`；`optimization_evidence/comparison_summary.json` | 无需修改；仅限本次静态证据范围。  |
| G03 | pass | 全部三组B/R与raw指标相符；Public归一化0.5736041748666919，full0.6243098009518612，正向改善且全部门通过。hidden单独0.8861不是Public/Dev范围门。 `optimization_evidence/comparison_summary.json`；`workspace/harbor_task/tests/anchors.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_3/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_3/result.json` | 无需修改；仅限本次静态证据范围。  |

## 双轨时长、八字段与格式

GPT48条：19 ok、28 failed、1 not_run；Seed11条：1 ok、6 failed、4 quality_gate_failed。GPT从逐点分配推进到精确CDF、通道归一化和局部多尺度统计；Seed尝试greedy/Lagrangian并保留真实CLI和码率失败，失败本身不是研究违规。

自报有效时长GPT14.5915594583h、Seed14.9449116158h（`run_summary.json#time_policy`、`time-accounting.json`），计入共享构建、排队、故障、修复和暂停后审查。包内旧作者授权文字不覆盖本次可信主控要求的当前V4。

合格正式区间（Asia/Shanghai）：GPT16:03:16→21:11:43.022668，墙钟上限18507.022668秒=5.14083963h；Seed16:01:33→21:15:34.694233，上限18841.694233秒=5.2338039536h。扣除前已经各不足最低7h。更早轨迹在`time-accounting.json#legacy_exploration`明确被列为隔离缺陷exploratory，不能合入同一合格包。

逐轮audit-projection旧strict effective求和GPT5446.318355324736秒，Seed0秒。这只是旧严格闭环算法局部证据；V4可计真实失败方法闭环，不能把0等同Seed没有研究。完整V4净时长未核验，但合格墙钟上限已确定失败。

canonical每轮实际18字段；GPT缺attempt23、Seed缺3/7。应投影为恰好八字段，顺序round与原attempt在sidecar关联，不能编造缺轮。round_audit_evidence使用rounds/...，应明确包根位置。

唯一最终GPT源码在`expert_evidence/closed_original_workspace/gpt/solution/method.py`与`expert_evidence/final_native_replays/gpt/method.py`相同SHA。最终full DISTS=0.11237433056036632、reward=1.1955827260919034，门内有效；超过U合法不应裁剪。Seed最终full DISTS=0.11519672721624374，hidden bpp=0.02158355712890625高于cap，reward=-1，已保留。不能把Seed最终质量失败自动推断为Hidden调参或其他违规。

已有保留容器12h存活/健康记录，与有效研究时长独立，本次未压力测试。三目录职责可辨；没有exact best_method目录名不单独扣错，补唯一最终方法映射即可。

- `expert_evidence/trajectory_codex.json`（misplaced）：扩展字段移sidecar，canonical八字段；保留原attempt映射。
- `expert_evidence/trajectory_seed.json`（misplaced）：同上，保留全部真实失败/质量门失败。
- `expert_evidence/expert_annotation.json`（misplaced）：缺失：补题型/方向/优化面/指标方向及同split B/R/U基本标注，不填官方Hidden验收结论。
- `expert_evidence/closed_original_workspace/gpt/solution`（extra_allowed）：保留真实最终工作区，明确唯一最终源码/入口映射或补轻量best_method。
- `optimization_evidence/lambda_sweep`（extra_allowed）：可保留参数选择材料，不混入正式主统计。
- `expert_evidence/final_research_history`（extra_allowed）：保留原始方法/失败证据，不能以旧副本覆盖当前规范轨迹。
- `optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json#artifacts`（misplaced）：method.source_path/artifacts缺optimization_evidence前缀，统一相对包根引用；可就近解析但应消除歧义。

## 核心结论、Harbor与21项

**FAIL / NOT READY。** 完成可安全静态审查；确定失败为当前平台配置/部署不匹配、完整资产交付不闭合、有效研究时长不足及canonical轨迹格式。方法/Baseline/提升通过不能抵消这些问题。

Agent context=environment/；Verifier context=tests/。普通COPY基本一致，Agent未COPY私有tests。verifier.collect把候选送到默认/logs/artifacts/solution，test.sh使用同路径，不能仅因顶层artifacts列public-results判移交缺失。大资产缺失和本地镜像不可获取才是交付问题。

Harbor静态失败；本次runtime=not_run。已有同次NOP reward0、separate、无exception和结束记录，TOML/两Dockerfile哈希与当前ZIP一致，可保留为历史链路证据。

| 项目 | 结论 | 事实与证据 | 最小修改与验收材料 |
|---|---|---|---|
| H01 | fail | ZIP hidden_assets仅.gitkeep，无八张图或manifest；有历史注入/构建回执，但无可获取安全资产入口。评分器明确要求manifest，完整交付未闭合。 `workspace/harbor_task/task.toml`；`workspace/harbor_task/tests/hidden_assets/.gitkeep`；`workspace/harbor_task/tests/run_eval.py:164`；`expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt:3` | 提供可获取的不可变模型/工具资产入口及校验步骤，或平台认可预置镜像；补Hidden私有材料安全入口、版本/清单/注入操作，不向Agent暴露。 接收方可定位所有named build contexts、私有manifest，验证大小/哈希并获得目标平台构建与准备回执；本机镜像ID不是获取入口。 |
| H02 | fail | separate明确、有0.23旧schema回执；但TOML仍声明storage_mb和no-network，违反当前内场口径。 `workspace/harbor_task/task.toml:16`；`workspace/harbor_task/task.toml:26`；`workspace/harbor_task/task.toml:34`；`expert_evidence/platform_adapter/harbor-schema-validation.json` | 删除task.toml的storage_mb及平台禁网配置依赖，将extra_docker_compose改为当前内场支持的单sandbox单容器配置；保留worker最小权限。明确H20/L20，保留A10历史。 更新TOML、镜像/资源/隔离说明、目标版本格式/schema检查和artifact移交链；不因换卡强制重做历史B/R。 |
| H03 | fail | 普通COPY和collect移交路径可解释；named模型/工具上下文未交，只有本地镜像ID/tag；adapter显式extra_docker_compose依赖。 `workspace/harbor_task/environment/Dockerfile:15`；`workspace/harbor_task/tests/Dockerfile:3`；`workspace/harbor_task/task.toml:19`；`expert_evidence/platform_adapter/onedc_harbor.py:49`；`expert_evidence/platform_adapter/offline-assets-manifest.json` | 提供可获取的不可变模型/工具资产入口及校验步骤，或平台认可预置镜像；补Hidden私有材料安全入口、版本/清单/注入操作，不向Agent暴露。 删除task.toml的storage_mb及平台禁网配置依赖，将extra_docker_compose改为当前内场支持的单sandbox单容器配置；保留worker最小权限。明确H20/L20，保留A10历史。 接收方可定位所有named build contexts、私有manifest，验证大小/哈希并获得目标平台构建与准备回执；本机镜像ID不是获取入口。 更新TOML、镜像/资源/隔离说明、目标版本格式/schema检查和artifact移交链；不因换卡强制重做历史B/R。 |
| H04 | pass | 默认artifacts接收候选，清理旧分并原子写有限reward.json；格式/超时异常退出，非作弊质量门-1。 `workspace/harbor_task/tests/test.sh:9`；`workspace/harbor_task/tests/test.sh:11`；`workspace/harbor_task/tests/test.sh:44`；`workspace/harbor_task/tests/score.py:28` | 无需修改；仅限本次静态证据范围。  |
| H05 | pass | 历史NOP配置定位任务/nop/adapter/GPU，result中verifier.disable=false且separate；不将其称当前内场验收。 `expert_evidence/harbor_nop_current/trial/config.json`；`expert_evidence/harbor_nop_current/trial/result.json` | 无需修改；仅限本次静态证据范围。  |
| H06 | pass | 同次NOP config/result/reward/脱敏stdout齐，reward0、无exception、separate/结束时间；TOML与两Dockerfile SHA匹配当前包。本次未执行。 `expert_evidence/harbor_nop_current/task-binding.json`；`expert_evidence/harbor_nop_current/trial/config.json`；`expert_evidence/harbor_nop_current/trial/result.json`；`expert_evidence/harbor_nop_current/trial/verifier/reward.json`；`expert_evidence/harbor_nop_current/trial/verifier/test-stdout.txt` | 无需修改；仅限本次静态证据范围。  |

| 项目 | 结论 | 事实与证据 | 最小修改与验收材料 |
|---|---|---|---|
| QA01 | pass | 八部分齐全，无论文标题/仓库/参考算法泄露。 `workspace/harbor_task/instruction.md` | 无需修改；仅限本次静态证据范围。  |
| QA02 | pass | score=(B-DISTS)/(B-U)，B>U和有限数检查；reward不取score_clipped，连续单调未裁剪。 `workspace/harbor_task/tests/score.py:9`；`workspace/harbor_task/tests/test.sh:55` | 无需修改；仅限本次静态证据范围。  |
| QA03 | pass | Reference实现/入口及三组result/log齐全；Public0.5736、full0.6243范围内，非训练不要求训练产物。 `workspace/reference/method.py`；`optimization_evidence/reference_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_3/result.json` | 无需修改；仅限本次静态证据范围。  |
| QA04 | not_applicable | 固定环境确定性：seed0、同输入双调用及三次原值一致；已复算n-1标准差0和正向改善，不将零方差外推到跨设备。 `workspace/harbor_task/tests/run_eval.py:104`；`workspace/harbor_task/environment/repo/onedc/src/inference.py:28`；`optimization_evidence/comparison_summary.json` | 无需修改；仅限本次静态证据范围。  |
| QA05 | fail | 码率/张量/文件/超时有实现，但当前环境仍依赖storage_mb、平台no-network及Compose，不能认定当前内场限制已落实。 `workspace/harbor_task/task.toml:16`；`workspace/harbor_task/task.toml:34`；`expert_evidence/platform_adapter/onedc_harbor.py:49` | 删除task.toml的storage_mb及平台禁网配置依赖，将extra_docker_compose改为当前内场支持的单sandbox单容器配置；保留worker最小权限。明确H20/L20，保留A10历史。 更新TOML、镜像/资源/隔离说明、目标版本格式/schema检查和artifact移交链；不因换卡强制重做历史B/R。 |
| QA06 | pass | 正式单命令test.sh计算并写标量reward；失败分型明确。 `workspace/harbor_task/tests/test.sh:44`；`workspace/harbor_task/tests/score.py:28` | 无需修改；仅限本次静态证据范围。  |
| QA07 | pass | 仅题面：未给Hidden样本/像素/标签/可恢复答案；提eight hidden不构成泄露。 `workspace/harbor_task/instruction.md` | 无需修改；仅限本次静态证据范围。  |
| QA08 | pass | 仅题面：无Reference代码/配置/成绩；公开Starter Baseline分数不是参考解泄露。 `workspace/harbor_task/instruction.md` | 无需修改；仅限本次静态证据范围。  |
| QA09 | pass | 清理reward.json/txt和out，再以os.replace原子替换最终分数。 `workspace/harbor_task/tests/test.sh:9`；`workspace/harbor_task/tests/test.sh:58` | 无需修改；仅限本次静态证据范围。  |
| QA10 | pass | 冻结清单、评分前后校验、root权限保护及降权tensor worker存在；不在root grader导入候选。 `workspace/harbor_task/environment/grader/build_integrity.py`；`workspace/harbor_task/environment/grader/protect_runtime.sh`；`workspace/harbor_task/tests/run_eval.py:74`；`workspace/harbor_task/tests/candidate_client.py:47` | 无需修改；仅限本次静态证据范围。  |
| QA11 | pass | 八部分各有目标/环境/指标/接口/禁令/提交/迭代/完成实质内容。 `workspace/harbor_task/instruction.md` | 无需修改；仅限本次静态证据范围。  |
| QA12 | pass | ZIP无.git对象/remotes/tags/reflog；.gitignore和构建时空.git不构成历史泄露。 `@inventory` | 无需修改；仅限本次静态证据范围。  |
| QA13 | pass | 全部三组B/R同协议；result/log/raw逐图值可追溯，每组full改善0.002473329504330962且双split过门。 `optimization_evidence/comparison_summary.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/baseline_runs/seed_eval_replicate_3/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_1/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_2/result.json`；`optimization_evidence/reference_runs/seed_eval_replicate_3/result.json` | 无需修改；仅限本次静态证据范围。  |
| QA14 | pass | 可做联合码率分配和感知统计代理；GPT最终方法给出方法空间证据，Reference未饱和。 `expert_evidence/final_native_replays/gpt/method.py`；`optimization_evidence/candidates/upper_bound_v8/method.py`；`workspace/harbor_task/instruction.md:25` | 无需修改；仅限本次静态证据范围。  |
| QA15 | manual（非阻断） | 历史A10九次B/R/U elapsed37.099–127.308秒，1GPU/8CPU/32GiB；目标H20/L20耗时未实测，仅迁移补证非阻断。 `workspace/harbor_task/task.toml:23`；`optimization_evidence/native_calibration_v8/summary.json` | 明确目标H20/L20配置，区分历史实测和目标估算；有新记录后追加，不改A10标签。 目标设备耗时依据或平台确认；不作为本次退回单独原因。 |
| QA16 | fail | 自报14.59/14.94h计入构建/队列/故障；合格正式起止上限GPT5.14084h、Seed5.23380h，扣除前已低于7h。包内旧授权不覆盖本次V4。 `expert_evidence/run_summary.json#time_policy`；`expert_evidence/time-accounting.json#final_v8_formal`；`expert_evidence/time-accounting.json#user_authorized_recalculation` | 按当前V4扣除构建/排队/故障/阻塞；隔离失效旧轨不得混入。保留暂停状态，另行授权后补真实迭代，不补造时间。 两轨独立起止、扣除区间、方法→Public评分→决策；默认各≥10h，短非训练例外各≥7h且完整例外材料/最佳干净复验。 |
| QA17 | fail | 历史NOP与独立可信评分可追溯；完整交付缺可用模型/工具/Hidden入口，旧storage/network/Compose与现行内场契约冲突。 `workspace/harbor_task/task.toml`；`workspace/harbor_task/environment/Dockerfile`；`expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt`；`expert_evidence/platform_adapter/onedc_harbor.py` | 提供可获取的不可变模型/工具资产入口及校验步骤，或平台认可预置镜像；补Hidden私有材料安全入口、版本/清单/注入操作，不向Agent暴露。 删除task.toml的storage_mb及平台禁网配置依赖，将extra_docker_compose改为当前内场支持的单sandbox单容器配置；保留worker最小权限。明确H20/L20，保留A10历史。 接收方可定位所有named build contexts、私有manifest，验证大小/哈希并获得目标平台构建与准备回执；本机镜像ID不是获取入口。 更新TOML、镜像/资源/隔离说明、目标版本格式/schema检查和artifact移交链；不因换卡强制重做历史B/R。 |
| QA18 | fail | canonical每轮18字段；GPT48条缺23、Seed11条缺3/7；round_audit_evidence=rounds/...未解析包根路径。摘要计时口径不符合当前合同。 `expert_evidence/trajectory_codex.json#rounds`；`expert_evidence/trajectory_seed.json#rounds`；`expert_evidence/run_summary.json` | 保留原始sidecar，canonical每轮恰八字段；顺序round与原attempt建立映射，缺号不造实验；引用改包根路径，解释模型失败和控制器采纳差别。 两份八字段rounds，顺序编号、原尝试映射、逐轮Public指标/质量门/method hash/最佳决策可回溯，摘要同步。 |
| QA19 | not_applicable | 无训练、固定seed0确定性推理，不强加随机评估阈值文字；正式三次重复已逐一审查。 `workspace/harbor_task/instruction.md:4`；`workspace/harbor_task/instruction.md:11`；`workspace/harbor_task/tests/run_eval.py:104` | 无需修改；仅限本次静态证据范围。  |
| QA20 | pass | 题面明确可改范围、worker网络/执行限制、公开文献平台工具策略和数据外传禁令。 `workspace/harbor_task/instruction.md:25`；`workspace/harbor_task/instruction.md:33` | 无需修改；仅限本次静态证据范围。  |
| QA21 | manual（非阻断） | run_summary缺task_binding.hash_profile/task_content_sha256；NOP局部SHA相符不能代替官方四项指纹。非阻断。 `expert_evidence/run_summary.json`；`expert_evidence/harbor_nop_current/task-binding.json` | 取得真实task_fingerprint.py后计算四项全递归指纹，输出放四项范围外，保留历史原值。 官方工具版本/SHA、输出及task_binding；不改旧结果制造一致。 |

## 五类放宽待补与工具边界

- 指纹/哈希：待补，不阻断。缺四项task_binding；NOP主要文件SHA匹配。 用真实官方工具保留历史差异。 验收：工具版本/SHA与四项输出。
- 换卡B/R影响：待补，不阻断。正式A10记录未改标，基于原值完成方法与提升审查。 目标资料可得后另列影响。 验收：目标环境说明/已有比较记录。
- 换卡耗时：待补，不阻断。A10实测存在，无H20/L20实测。 区分目标估算，不自动补跑。 验收：目标耗时依据/平台确认。
- NOP：已有历史核对通过；补充非阻断。同次config/result/reward/脱敏日志与主要文件绑定一致。 需要时补原始可信侧日志与完整绑定。 验收：同次Trial原始回执。
- 换卡重算：可延期，不阻断。完整正式A10分数已复算。 不仅因换卡重跑全部B/R或调U。 验收：有新数据再另列协议来源，不覆盖旧记录。

官方0.3.10、独立格式/GPU/指纹工具未运行；真实工具验收待补，不能手改旧工具版本号或声称官方PASS。本次不执行题包代码/导入模块、Docker/Verifier、训练/模型调用或反序列化权重，不外发。完整附件/二进制严格隐私扫描由主控另做。

可见隐私待核：`workspace/harbor_task/environment/repo/onedc/readme.md:64,99`含capability-style分享入口；只报告类型/位置，不复写链接值，需确认公开必要性或替换稳定入口。依赖版本号造成的私网IP外形命中已排除。

## 可复制给专家的退回说明

当前完整源码ZIP按V4为NOT READY。方法空间、Baseline合理性及三次正式成对提升可以保留，不要削弱B或调整U修复。

完整交付：tests/hidden_assets只有占位；run_eval.py:164要求manifest。environment/Dockerfile:15–17要求两个named contexts，TOML:28/tests Dockerfile:3只有本地镜像ID/tag。 提供可获取的不可变模型/工具资产入口及校验步骤，或平台认可预置镜像；补Hidden私有材料安全入口、版本/清单/注入操作，不向Agent暴露。 验收材料：接收方可定位所有named build contexts、私有manifest，验证大小/哈希并获得目标平台构建与准备回执；本机镜像ID不是获取入口。

内场配置：task.toml:16,26,34保留no-network/storage_mb；platform_adapter/onedc_harbor.py:49–60依赖extra_docker_compose。 删除task.toml的storage_mb及平台禁网配置依赖，将extra_docker_compose改为当前内场支持的单sandbox单容器配置；保留worker最小权限。明确H20/L20，保留A10历史。 验收材料：更新TOML、镜像/资源/隔离说明、目标版本格式/schema检查和artifact移交链；不因换卡强制重做历史B/R。

有效时长：time-accounting.json合格正式区间GPT5.14083963h、Seed5.2338039536h，扣除前已不足7h；自报14h不符当前口径。 按当前V4扣除构建/排队/故障/阻塞；隔离失效旧轨不得混入。保留暂停状态，另行授权后补真实迭代，不补造时间。 验收材料：两轨独立起止、扣除区间、方法→Public评分→决策；默认各≥10h，短非训练例外各≥7h且完整例外材料/最佳干净复验。

轨迹格式：两canonical每轮18字段、编号缺口和sidecar路径未明确；需补expert_annotation基本标注、唯一最终方法路径。 保留原始sidecar，canonical每轮恰八字段；顺序round与原attempt建立映射，缺号不造实验；引用改包根路径，解释模型失败和控制器采纳差别。 验收材料：两份八字段rounds，顺序编号、原尝试映射、逐轮Public指标/质量门/method hash/最佳决策可回溯，摘要同步。

非阻断后补：四项指纹、H20/L20换卡分数/耗时/锚点影响、NOP原始补充链路。历史NOP reward0且主要文件绑定一致；不因这些项强制新运行。真实官方工具验收另列待补。

发布前确认源码README第64/99行capability-style入口的公开必要性，逐件严格隐私扫描；不在报告复写入口值。
