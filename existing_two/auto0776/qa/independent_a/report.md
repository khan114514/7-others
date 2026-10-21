# auto0776 独立只读 V4 内容复核 A

结论：FAIL / NOT READY。ZIP SHA256：df27ff7cd6fff4a7b10d199ce35daec4c8ed65df8aac6e07b50957163525cc7c。
configured_model=gpt-6.1-sol；runtime_reported_model=not_exposed；runtime_version=not_exposed；session_id=onedc_a。

## 优化面及实际边界

冻结codec/熵模型/解码/数据，仅改残差测试时量化函数。
未经修改Starter=dead-zone τ0.6，sign(y)*round(clamp(abs(y)-0.1,0))；固定预训练模型，无训练。
逐latent在round(y)及±1中最小化平方量化误差+0.055×高斯熵bits；σ自适应方法更新。
允许可执行方法函数；实际guard只限制文件合同与张量合法性，R不是仅调整已有参数。

## Baseline专项

在已提供材料范围内合理。Starter=正式Baseline=B锚点；官方round为不同码率额外复现对照。
- training_sufficiency：无训练；固定权重，评估重复不算训练seed。
- method_representativeness：合理naive dead-zone及工作点说明，未见足以证明不合理的可比强锚点。
- budget_comparability：同A10/权重/24图顺序/门槛/3次进程；换卡影响后补。
- implementation_correctness：源码与原始编码日志支持功能健全；未独立运行。
- selection_integrity：全部3对原值及日志相符，参数选择有披露，无挑最差B直接反证。

## Baseline八项

| 项目 | 状态 | 事实和证据 |
|---|---|---|
| B01 | pass | Starter、正式Baseline与B锚点同为τ0.6 dead-zone；固定checkpoint低码率工作点有选择说明，不能用超码率官方round直接否定。 来源：expert_evidence/03-baseline-quality.md:5-16；expert_evidence/02-optimization-surface.md |
| B02 | pass | 无训练；量化公式非空，原始日志有真实24图编码、指标和正常退出，无故障模型冒充对照的直接反证。 来源：workspace/harbor_task/environment/starter/method.py:24-29；optimization_evidence/native_calibration_v8/original_records/raw/B_1/run.log |
| B03 | pass | B/R各3个新评分进程，同final-v8 A10、权重、图序、码率门和资源；B/R约37–83秒。 来源：optimization_evidence/训练证据说明.md；optimization_evidence/native_calibration_v8/summary.json |
| B04 | pass | τ与λ选参工作点有披露；正式比较未见针对B的异常配置反证。 来源：expert_evidence/02-optimization-surface.md；optimization_evidence/lambda_sweep/README.md |
| B05 | pass | 明确评估重复而非训练seed；真实命令seed0，六组正式结果完整。 来源：optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json:5-7；optimization_evidence/native_calibration_v8/original_records/raw/B_1/execution.json |
| B06 | pass | 官方uniform复现为不同码率额外对照；naive量化器是任务内工作点锚点。 来源：optimization_evidence/official-paper-reproduction.json；expert_evidence/02-optimization-surface.md |
| B07 | pass | B到R仅变测试时量化函数；同residual/scales、固定codec、数据、评测及预算。 来源：workspace/harbor_task/environment/starter/method.py；workspace/reference/method.py；optimization_evidence/训练证据说明.md |
| B08 | pass | 来源、限制、参数选择、哈希、三次正式值已披露；不直接采信作者自述PASS。 来源：expert_evidence/03-baseline-quality.md；optimization_evidence/comparison_summary.json |

## 正式全部成对原值与复算

无训练；eval_replicate表示新评分进程，固定执行seed=0，不冒充训练seed。

| replicate | B full DISTS | R full DISTS | B−R | B/R有效 | B/R耗时秒 |
|---|---:|---:|---:|---|---|
| eval_replicate_1 | 0.11711087326208751 | 0.11463754375775655 | 0.0024733295043309622 | true/true | 37.61417865/37.68279180 |
| eval_replicate_2 | 0.11711087326208751 | 0.11463754375775655 | 0.0024733295043309622 | true/true | 83.16579968/37.49950350 |
| eval_replicate_3 | 0.11711087326208751 | 0.11463754375775655 | 0.0024733295043309622 | true/true | 37.09920503/37.52086779 |

baseline replicate 1：`optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json` → `optimization_evidence/native_calibration_v8/original_records/raw/B_1/out/metrics.json`，日志`optimization_evidence/baseline_runs/seed_eval_replicate_1/run.log`。主值相等，标准run.log与原始run.log字节相等。

reference replicate 1：`optimization_evidence/reference_runs/seed_eval_replicate_1/result.json` → `optimization_evidence/native_calibration_v8/original_records/raw/R_1/out/metrics.json`，日志`optimization_evidence/reference_runs/seed_eval_replicate_1/run.log`。主值相等，标准run.log与原始run.log字节相等。

baseline replicate 2：`optimization_evidence/baseline_runs/seed_eval_replicate_2/result.json` → `optimization_evidence/native_calibration_v8/original_records/raw/B_2/out/metrics.json`，日志`optimization_evidence/baseline_runs/seed_eval_replicate_2/run.log`。主值相等，标准run.log与原始run.log字节相等。

reference replicate 2：`optimization_evidence/reference_runs/seed_eval_replicate_2/result.json` → `optimization_evidence/native_calibration_v8/original_records/raw/R_2/out/metrics.json`，日志`optimization_evidence/reference_runs/seed_eval_replicate_2/run.log`。主值相等，标准run.log与原始run.log字节相等。

baseline replicate 3：`optimization_evidence/baseline_runs/seed_eval_replicate_3/result.json` → `optimization_evidence/native_calibration_v8/original_records/raw/B_3/out/metrics.json`，日志`optimization_evidence/baseline_runs/seed_eval_replicate_3/run.log`。主值相等，标准run.log与原始run.log字节相等。

reference replicate 3：`optimization_evidence/reference_runs/seed_eval_replicate_3/result.json` → `optimization_evidence/native_calibration_v8/original_records/raw/R_3/out/metrics.json`，日志`optimization_evidence/reference_runs/seed_eval_replicate_3/run.log`。主值相等，标准run.log与原始run.log字节相等。

B均值=0.11711087326208751，R均值=0.11463754375775655；n−1样本标准差B=0、R=0。Δ=0.002473329504330962，相对改善2.111955478972299%。U=0.11314917107423146；full归一化=0.6243098009518612。public B=0.12342559173703194、R=0.12056996300816536、U=0.118447195738554，归一化=0.5736041748666919。确定性正向改善/归一化均满足；3σ_B=0且Δ>0，但不以零样本方差单独证明一般确定性，不恢复统一5%门槛。

三次各角色以下指标完全相同：

| 指标 | B | R |
|---|---:|---:|
| bpp_mean | 0.0204925537109375 | 0.020459493001302086 |
| lpips | 0.17629042702416578 | 0.17375360118846098 |
| psnr | 21.74609302673294 | 21.82438685391236 |
| msssim | 0.7513415142893791 | 0.7578616241614023 |
| public.dists | 0.12342559173703194 | 0.12056996300816536 |
| public.bpp_mean | 0.020015716552734375 | 0.02001317342122396 |
| hidden.dists | 0.10448143631219864 | 0.10277270525693893 |
| hidden.bpp_mean | 0.02144622802734375 | 0.021352132161458336 |

码率上限public=0.020115795135498046、hidden=0.021553459167480468；全部B/R均通过。U三次DISTS=0.11314917107423146、bpp=0.02051628960503472，样本标准差0；R到U剩余0.001488372683525085。U原始记录：optimization_evidence/native_calibration_v8/original_records/raw/U_1..3/。

## 内容三门

| 项目 | 状态 | 事实和证据 |
|---|---|---|
| G01 | pass | 实际quantize执行接口允许方法算法；guard仅核文件合同及张量，非字面量超参限制。R从dead-zone变为逐latent率失真候选选择。 来源：workspace/harbor_task/instruction.md:25-40；workspace/harbor_task/tests/solution_guard.py:11-23；workspace/harbor_task/tests/candidate_worker.py:98-129；workspace/reference/method.py:24-33 |
| G02 | pass | 在已交材料范围内Baseline合理；无训练不足、故障或不公平的直接反证。 来源：expert_evidence/03-baseline-quality.md；optimization_evidence/native_calibration_v8/original_records/raw/B_1/run.log |
| G03 | pass | 3对正式主值有效、过门且正改善；full归一化0.6243098，public0.5736042，均达[0.15,0.8]。 来源：optimization_evidence/comparison_summary.json；optimization_evidence/native_calibration_v8/original_records/raw/B_1/out/metrics.json；optimization_evidence/native_calibration_v8/original_records/raw/R_1/out/metrics.json |

## 双轨时长、格式与最终结果

GPT 48条：19 ok、28 failed、1 not_run；原round缺23。合格16:03:16→21:11:43，墙钟上界18507秒=5.14083h。Seed 11条：1 ok、6 failed、4 quality_gate_failed；原round缺3/7。合格16:01:33→21:15:34，墙钟上界18841秒=5.23361h。按V4还需扣除排队/故障等，均不可能达到7h下限。
run_summary声称14.59156h/14.94491h，来自共享06:53:11起全投入时钟，明确含安装、构建、排队、故障及修复。包内历史“用户授权”是作者证据，不是本次可信主控直接授权，不能覆盖当前V4合同。旧物理隔离不合格探索不能并入合格轨迹。
规范轨迹每轮有controller_original_status等额外字段及跳号。Seed没有retained_best=true；其摘要best来自失败CLI后的独立控制器采纳，应与模型有效闭环分开。真实失败/null应保留，不能补成成功。
GPT最终full DISTS=0.11237433056036632、reward=1.1955827260919034，有效；Seed full DISTS=0.11519672721624374、reward=-1，无效，hidden bpp=0.02158355712890625超0.021553459167480468。选GPT有据，>1未裁剪分合法；Seed失败不改变正式Reference的提升结论。

## 格式与Harbor说明

三目录职责基本齐；expert_annotation.json缺失。唯一最终法可从final_native_replays/gpt/method.py映射，建议归位，不能把多交本身扣错。result.source_path/artifacts应统一提交包根相对路径。
Agent context=environment/，Verifier context=tests/；本地COPY存在。named model_assets/agent_tools、基础镜像及Hidden只有摘要/manifest/历史回执，当前可操作安全取得入口待核，不假定包外存在。Hidden并非只有一句平台注入：部署/NOP/构建回执支持构建前注入；但交付入口仍须补明。
verifier.collect将/workspace/solution复制到/logs/artifacts/solution，test.sh同路径读取，采用默认artifacts转交；不能仅因顶层artifacts收public-results而误判未移交。

## Harbor六项

| 项目 | 状态 | 事实和证据 |
|---|---|---|
| H01 | manual | 任务根/入口齐。Hidden有组织者构建前注入和历史回执；ZIP只含占位，无checkpoint实体。manifest有大小/hash，但缺可用镜像/资产注册入口，不能假定包外可取。 来源：workspace/harbor_task/task.toml；workspace/harbor_task/tests/hidden_assets/.gitkeep；expert_evidence/platform_adapter/offline-assets-manifest.json；expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt:2-3 |
| H02 | fail | separate显式存在；但storage_mb和no-network不符合本批内场要求。0.23.0为包内声明，本次未原生schema加载。 来源：workspace/harbor_task/task.toml:14-34；expert_evidence/platform_adapter/harbor-schema-validation.json |
| H03 | fail | 本地COPY来源和collect→默认artifacts→test.sh路径有效；实际provider通过extra_docker_compose执行。外部named contexts和基础镜像也缺可操作入口。 来源：workspace/harbor_task/environment/Dockerfile:15-17；workspace/harbor_task/tests/Dockerfile:3-11；workspace/harbor_task/task.toml:19-20；workspace/harbor_task/tests/test.sh:11；expert_evidence/platform_adapter/onedc_harbor.py:39-60 |
| H04 | pass | 可信侧清旧结果并原子写/logs/verifier/reward.json有限reward，不以stdout替代。 来源：workspace/harbor_task/tests/test.sh:9-10；workspace/harbor_task/tests/test.sh:44-65 |
| H05 | fail | NOP/回放配置走自定义OfflineOneDCDockerEnvironment，底层仍依赖Compose，未迁移当前支持环境。 来源：expert_evidence/harbor_nop_current/trial/config.json:11-16；expert_evidence/platform_adapter/onedc_offline_harbor.py；expert_evidence/platform_adapter/onedc_harbor.py:49-60 |
| H06 | manual（非阻断） | 同Trial nop/separate正常结束reward0，与Baseline一致；task/Dockerfile四个绑定hash匹配本ZIP。test-stdout仅脱敏摘要，完整链路待补，非阻断。 来源：expert_evidence/harbor_nop_current/trial/config.json；expert_evidence/harbor_nop_current/trial/result.json:92-115；expert_evidence/harbor_nop_current/trial/verifier/reward.json；expert_evidence/harbor_nop_current/trial/verifier/test-stdout.txt；expert_evidence/harbor_nop_current/task-binding.json |

## 21项实现检查

| 项目 | 状态 | 事实和证据 |
|---|---|---|
| QA01 | pass | 八部分有实质内容，题面无论文标题/ID、Reference算法或仓库线索。 来源：workspace/harbor_task/instruction.md:1-64 |
| QA02 | pass | score=(B-p)/(B-U)，检查B>U；reward使用未裁剪score，clipped字段仅展示。 来源：workspace/harbor_task/tests/score.py:9-38 |
| QA03 | pass | R源码和执行入口可识别；3个正式result及非空原始日志齐全；public/full归一化均达标。非训练不要求训练checkpoint。 来源：workspace/reference/method.py；optimization_evidence/reference_runs/seed_eval_replicate_1/result.json；optimization_evidence/reference_runs/seed_eval_replicate_2/result.json；optimization_evidence/reference_runs/seed_eval_replicate_3/result.json |
| QA04 | not_applicable | 固定权重、seed0、顺序和确定性量化协议；3次新进程一致。已复算sigma_B=0与严格正改善，不恢复5%门槛。 来源：optimization_evidence/native_calibration_v8/original_records/raw/B_1/execution.json；optimization_evidence/comparison_summary.json |
| QA05 | fail | worker/file/timeout/码率保护存在；但TOML声明storage_mb及no-network，实际provider用extra_docker_compose构造环境，违背当前内场约束。 来源：workspace/harbor_task/task.toml:14-34；expert_evidence/platform_adapter/onedc_harbor.py:39-60 |
| QA06 | pass | test.sh贯通guard、受限评分和有限标量reward；格式/超时/资源故障不伪装有效0分。 来源：workspace/harbor_task/tests/test.sh:27-65；workspace/harbor_task/tests/run_eval.py:340-364 |
| QA07 | pass | 只读题面未见Hidden样本、标签或可恢复答案；仅声明8张及隔离。 来源：workspace/harbor_task/instruction.md:8-43 |
| QA08 | pass | 只读题面未披露R算法/参数/结果；公开Baseline值和接口不等于参考答案。 来源：workspace/harbor_task/instruction.md:1-64 |
| QA09 | pass | 清理旧reward/out，结果临时写后os.replace。 来源：workspace/harbor_task/tests/test.sh:9-10；workspace/harbor_task/tests/test.sh:58-63；workspace/harbor_task/tests/run_eval.py:64-71 |
| QA10 | pass | 有root权限保护、前后冻结哈希、独立worker UID和IPC；仅确认实现，不认证完整物理隔离。 来源：workspace/harbor_task/tests/run_eval.py:74-94；workspace/harbor_task/tests/run_eval.py:145；workspace/harbor_task/tests/run_eval.py:322；workspace/harbor_task/environment/grader/protect_runtime.sh；workspace/harbor_task/tests/candidate_client.py:27-77 |
| QA11 | pass | 目标、环境、公式、范围、边界、提交、流程、完成条件均具体。 来源：workspace/harbor_task/instruction.md:3-64 |
| QA12 | pass | ZIP清点无.git历史/remote/tag/reflog，.gitignore不是.git。 来源：@inventory |
| QA13 | pass | 全部3对正式主值与原始metrics相等，正式run.log与raw字节相等；同协议，逐对正改善、质量门全过。 来源：optimization_evidence/comparison_summary.json；optimization_evidence/baseline_runs/seed_eval_replicate_1/result.json；optimization_evidence/reference_runs/seed_eval_replicate_1/result.json |
| QA14 | pass | 可研究局部/通道误差分配和感知代理等题面未提示的方法；Reference未饱和，GPT最终有效值也有余量证据。 来源：workspace/harbor_task/instruction.md:25-30；expert_evidence/02-optimization-surface.md；expert_evidence/final_native_replays/summary.json |
| QA15 | manual（非阻断） | A10正式B/R约37–83秒，U最长127.31秒；配置1GPU/8CPU/32GiB。H20/L20迁移耗时与成绩未核，非阻断待补。 来源：optimization_evidence/native_calibration_v8/summary.json；workspace/harbor_task/task.toml:21-35 |
| QA16 | fail | 主时钟计安装、排队、故障及共享准备。合格GPT16:03:16→21:11:43、Seed16:01:33→21:15:34，墙钟上界5.14/5.23h，均不足7h。 来源：expert_evidence/time-accounting.json:3-8；expert_evidence/time-accounting.json:87-109；expert_evidence/run_summary.json:17-47；expert_evidence/05-run-isolation.md |
| QA17 | fail | separate及collect/reward链存在；但当前provider依赖Compose且有不支持配置，必要资产安全交付入口待核。NOP缺口单列非阻断。 来源：workspace/harbor_task/task.toml；expert_evidence/platform_adapter/onedc_harbor.py；expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt |
| QA18 | fail | GPT48条/Seed11条都有8核心字段但多附加字段且round跳号；Seed仅1条ok且无retained_best=true，摘要best来自失败CLI后的控制器采纳，呈现未对齐。 来源：expert_evidence/trajectory_codex.json#rounds；expert_evidence/trajectory_seed.json#rounds；expert_evidence/CANONICAL-TRAJECTORY-PROJECTION.json#seed_best_source；expert_evidence/run_summary.json:17-47 |
| QA19 | not_applicable | 固定seed0确定性评测，不另增训练seed/重复次数题面要求；正式3次重复完整性已核。 来源：workspace/harbor_task/instruction.md:4-6；workspace/harbor_task/instruction.md:40；optimization_evidence/训练证据说明.md:3 |
| QA20 | pass | 题面有可改范围、外发/候选网络边界和公共文献平台工具策略；不支持禁网配置的实质问题另列。 来源：workspace/harbor_task/instruction.md:25-43 |
| QA21 | manual（非阻断） | run_summary缺task_binding.hash_profile/task_content_sha256，不猜官方哈希，不拿旧NOP checksum冒充当前四项指纹。 来源：expert_evidence/run_summary.json |

## 每项不通过及待核的修改/验收材料


**QA05：worker/file/timeout/码率保护存在；但TOML声明storage_mb及no-network，实际provider用extra_docker_compose构造环境，违背当前内场约束。**
事实位置：workspace/harbor_task/task.toml:14-34；expert_evidence/platform_adapter/onedc_harbor.py:39-60
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。
复验/验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。

**QA15：A10正式B/R约37–83秒，U最长127.31秒；配置1GPU/8CPU/32GiB。H20/L20迁移耗时与成绩未核，非阻断待补。**
事实位置：optimization_evidence/native_calibration_v8/summary.json；workspace/harbor_task/task.toml:21-35
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：保留旧A10实测，明确目标H20/L20与估算依据，有现成记录后补。
复验/验收材料：目标卡调度/资源、完整单轮计时依据；不强制因换卡补跑。

**QA16：主时钟计安装、排队、故障及共享准备。合格GPT16:03:16→21:11:43、Seed16:01:33→21:15:34，墙钟上界5.14/5.23h，均不足7h。**
事实位置：expert_evidence/time-accounting.json:3-8；expert_evidence/time-accounting.json:87-109；expert_evidence/run_summary.json:17-47；expert_evidence/05-run-isolation.md
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：按V4从各自合格独立轨迹计时，扣除安装、排队、故障和阻塞，保留旧记录；补足实际有效迭代，默认各10h，或完整7h例外且各至少7h。
复验/验收材料：两轨各自起止及扣除区间并集，真实方法→Public评分→决策记录；例外还需短迭代、每轨至少3闭环、后续方向及最佳方法干净有效复验。

**QA17：separate及collect/reward链存在；但当前provider依赖Compose且有不支持配置，必要资产安全交付入口待核。NOP缺口单列非阻断。**
事实位置：workspace/harbor_task/task.toml；expert_evidence/platform_adapter/onedc_harbor.py；expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。补必要模型及Hidden的可操作安全入口。
复验/验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。模型/Hidden版本、大小、哈希及组织者可用取得方式。

**QA18：GPT48条/Seed11条都有8核心字段但多附加字段且round跳号；Seed仅1条ok且无retained_best=true，摘要best来自失败CLI后的控制器采纳，呈现未对齐。**
事实位置：expert_evidence/trajectory_codex.json#rounds；expert_evidence/trajectory_seed.json#rounds；expert_evidence/CANONICAL-TRAJECTORY-PROJECTION.json#seed_best_source；expert_evidence/run_summary.json:17-47
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：每轮规范八字段单独输出，附加来源移到旁表；真实记录连续编号并保留旧attempt映射，不能补造实验。明确控制器采纳和模型闭环、最终最佳方法的关系。
复验/验收材料：两份可解析八字段轨迹、原attempt映射、每轮公开metrics/源码关联、与run_summary一致的轮数/时长及唯一最终法。

**QA21：run_summary缺task_binding.hash_profile/task_content_sha256，不猜官方哈希，不拿旧NOP checksum冒充当前四项指纹。**
事实位置：expert_evidence/run_summary.json
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：取得真实task_fingerprint.py后计算四项递归指纹，报告放四项外，补task_binding。
复验/验收材料：真实工具版本/SHA、原始指纹报告及同ZIP关联；非阻断待补。

**H01：任务根/入口齐。Hidden有组织者构建前注入和历史回执；ZIP只含占位，无checkpoint实体。manifest有大小/hash，但缺可用镜像/资产注册入口，不能假定包外可取。**
事实位置：workspace/harbor_task/task.toml；workspace/harbor_task/tests/hidden_assets/.gitkeep；expert_evidence/platform_adapter/offline-assets-manifest.json；expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt:2-3
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：补平台认可不可变模型/镜像安全入口及Hidden准备来源、操作和manifest版本。
复验/验收材料：组织者可用准确取得方式、版本/大小/哈希、Verifier准备及调用记录；无需将Hidden公开给Agent。

**H02：separate显式存在；但storage_mb和no-network不符合本批内场要求。0.23.0为包内声明，本次未原生schema加载。**
事实位置：workspace/harbor_task/task.toml:14-34；expert_evidence/platform_adapter/harbor-schema-validation.json
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。
复验/验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。

**H03：本地COPY来源和collect→默认artifacts→test.sh路径有效；实际provider通过extra_docker_compose执行。外部named contexts和基础镜像也缺可操作入口。**
事实位置：workspace/harbor_task/environment/Dockerfile:15-17；workspace/harbor_task/tests/Dockerfile:3-11；workspace/harbor_task/task.toml:19-20；workspace/harbor_task/tests/test.sh:11；expert_evidence/platform_adapter/onedc_harbor.py:39-60
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。补named contexts与镜像可获取交付。
复验/验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。各context构建及候选转交证据、不可变资产操作。

**H05：NOP/回放配置走自定义OfflineOneDCDockerEnvironment，底层仍依赖Compose，未迁移当前支持环境。**
事实位置：expert_evidence/harbor_nop_current/trial/config.json:11-16；expert_evidence/platform_adapter/onedc_offline_harbor.py；expert_evidence/platform_adapter/onedc_harbor.py:49-60
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。
复验/验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。

**H06：同Trial nop/separate正常结束reward0，与Baseline一致；task/Dockerfile四个绑定hash匹配本ZIP。test-stdout仅脱敏摘要，完整链路待补，非阻断。**
事实位置：expert_evidence/harbor_nop_current/trial/config.json；expert_evidence/harbor_nop_current/trial/result.json:92-115；expert_evidence/harbor_nop_current/trial/verifier/reward.json；expert_evidence/harbor_nop_current/trial/verifier/test-stdout.txt；expert_evidence/harbor_nop_current/task-binding.json
影响：当前核心要求未满足或事实未核；标非阻断的事项可后补。
最小修改：保留现有记录，经安全渠道补同Trial完整原始日志与版本关联，不自动重跑。
复验/验收材料：同Trial配置、结果、reward、原始非空日志及任务版本关联。

## 核心结论、五类放宽、工具缺口与运行边界

核心FAIL来自有效时长、当前配置/Compose、轨迹呈现；方法空间/Baseline/成对提升通过。静态审查completed=true；缺资产入口和原生接入事实仍为manual，不遮盖已知失败。
五类非阻断待补：四项指纹/hash；换卡B/R和提升影响；换卡耗时；NOP完整记录/版本关联；换卡重算分数。A10历史结果保真，不单凭这些缺口判退，不自动启动复跑。
真实官方0.3.10及独立格式/GPU/指纹脚本未提供，工具验收待补。本报告为local-v4-content-review，未改版本或伪称官方/平台PASS。
本次只读唯一ZIP，不执行/导入提交代码、不运行Docker/Verifier/训练/模型、不反序列化权重。只确认既有材料支持程度，不认证独立复现/平台通过。包内>12h容器生命周期/健康证据不替代有效研究时间或本次压力测试。
源码/题面可见读取未见需回显的私密值；固定镜像账户home路径不当作作者home。完整逐件二进制隐私门禁由主控另做。报告只用相对证据。

风险最多三条：
- 主时间计V4排除项，合格轨迹墙钟上界均不足7h。
- 正式provider依赖Compose且声明storage_mb/no-network，历史运行不等于当前内场兼容。
- 必要模型/Hidden不在ZIP且缺可操作安全取得入口；摘要/hash不证明接收方可运行。

## 可复制专家退回说明

本轮FAIL / NOT READY；不是官方0.3.10或平台验收。

可保留：方法级量化接口开放，naive dead-zone Baseline合理；全部3对B/R原值和原始日志对齐，Δ=0.002473329504330962，full归一化0.6243098009518612/public0.5736041748666919，码率门均通过。

QA05：worker/file/timeout/码率保护存在；但TOML声明storage_mb及no-network，实际provider用extra_docker_compose构造环境，违背当前内场约束。
事实位置：workspace/harbor_task/task.toml:14-34；expert_evidence/platform_adapter/onedc_harbor.py:39-60
影响：当前核心条件未满足/事实未核，不能由其他项通过抵消。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。
验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。

QA16：主时钟计安装、排队、故障及共享准备。合格GPT16:03:16→21:11:43、Seed16:01:33→21:15:34，墙钟上界5.14/5.23h，均不足7h。
事实位置：expert_evidence/time-accounting.json:3-8；expert_evidence/time-accounting.json:87-109；expert_evidence/run_summary.json:17-47；expert_evidence/05-run-isolation.md
影响：当前核心条件未满足/事实未核，不能由其他项通过抵消。
最小修改：按V4从各自合格独立轨迹计时，扣除安装、排队、故障和阻塞，保留旧记录；补足实际有效迭代，默认各10h，或完整7h例外且各至少7h。
验收材料：两轨各自起止及扣除区间并集，真实方法→Public评分→决策记录；例外还需短迭代、每轨至少3闭环、后续方向及最佳方法干净有效复验。

QA18：GPT48条/Seed11条都有8核心字段但多附加字段且round跳号；Seed仅1条ok且无retained_best=true，摘要best来自失败CLI后的控制器采纳，呈现未对齐。
事实位置：expert_evidence/trajectory_codex.json#rounds；expert_evidence/trajectory_seed.json#rounds；expert_evidence/CANONICAL-TRAJECTORY-PROJECTION.json#seed_best_source；expert_evidence/run_summary.json:17-47
影响：当前核心条件未满足/事实未核，不能由其他项通过抵消。
最小修改：每轮规范八字段单独输出，附加来源移到旁表；真实记录连续编号并保留旧attempt映射，不能补造实验。明确控制器采纳和模型闭环、最终最佳方法的关系。
验收材料：两份可解析八字段轨迹、原attempt映射、每轮公开metrics/源码关联、与run_summary一致的轮数/时长及唯一最终法。

H01：任务根/入口齐。Hidden有组织者构建前注入和历史回执；ZIP只含占位，无checkpoint实体。manifest有大小/hash，但缺可用镜像/资产注册入口，不能假定包外可取。
事实位置：workspace/harbor_task/task.toml；workspace/harbor_task/tests/hidden_assets/.gitkeep；expert_evidence/platform_adapter/offline-assets-manifest.json；expert_evidence/platform_adapter/DEPLOYMENT-PROFILE.txt:2-3
影响：当前核心条件未满足/事实未核，不能由其他项通过抵消。
最小修改：补平台认可不可变模型/镜像安全入口及Hidden准备来源、操作和manifest版本。
验收材料：组织者可用准确取得方式、版本/大小/哈希、Verifier准备及调用记录；无需将Hidden公开给Agent。

H03：本地COPY来源和collect→默认artifacts→test.sh路径有效；实际provider通过extra_docker_compose执行。外部named contexts和基础镜像也缺可操作入口。
事实位置：workspace/harbor_task/environment/Dockerfile:15-17；workspace/harbor_task/tests/Dockerfile:3-11；workspace/harbor_task/task.toml:19-20；workspace/harbor_task/tests/test.sh:11；expert_evidence/platform_adapter/onedc_harbor.py:39-60
影响：当前核心条件未满足/事实未核，不能由其他项通过抵消。
最小修改：删除storage_mb和禁网/白名单声明，按当前联网条件实现边界；将正式执行迁移为每sandbox单容器直接使用自身GPU的支持路径，去掉对Compose适配的依赖。补named contexts与镜像可获取交付。
验收材料：更新题面/TOML/provider配置；目标环境原生schema、构建、候选移交及reward记录。各context构建及候选转交证据、不可变资产操作。

格式补齐：expert_evidence/expert_annotation.json；明确已选GPT唯一最终法/入口映射，保留真实失败、无效Seed结果、旧attempt与来源。统一result.source_path/artifacts为提交包根相对路径。

非阻断后补：四项指纹/task_binding；A10迁移H20/L20的成绩、锚点、时间、重算；NOP原始完整日志与版本关联。本次不要求为这些自动复跑。工具缺口：需真实官方0.3.10和独立脚本，不能改旧版本冒充。

本次未执行题包代码、Docker、模型或训练，不代表平台已运行/通过。

