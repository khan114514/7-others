# Heima 独立只读内容复核

报告类型：local-v4-content-review；会话：heima_a；日期：2026-10-22。
ZIP SHA256：ce7d24591a81f080954e9910b1e02d77e093260ba205fa3a524fbfb693dcf017
审查者：OpenAI Codex；主控配置 gpt-6.1-sol / high，运行时自述 GPT-6，精确后端版本未暴露。
仅输入完整 ZIP；未读取包外旧报告、其他审查结论或包内旧QA结论。使用当前 V4 与 2026-10-21 五类放宽规则。

## 优化面与方法
冻结视觉推理目标的推理时草稿策略优化，降低端到端latency ratio；不是训练任务。
普通AR；draft返回[]，目标逐token采样与forward。Starter=正式Baseline=Dev归一化锚点B；独立calibration为另一个普通AR测量角色。
预先从固定tokenizer取得36 token结构前缀，精确匹配已生成前缀后每次最多提议16 token；不匹配或越过前缀则退回AR。目标验证所有提议，R非最终Agent最佳。
固定模型/数据/采样与预算，Agent可实现真实Python策略。训练为NOT_APPLICABLE；五seed为固定模型的随机评估，非五次训练。

## 全部正式 B/R 数值
| seed | Baseline | Reference | B-R |
|---|---:|---:|---:|
| 17 | 1.000103380176573 | 0.373456636115598 | 0.626646744060975 |
| 29 | 0.999924867484366 | 0.407006021007487 | 0.592918846476879 |
| 43 | 1.000907990960543 | 0.412585396817574 | 0.588322594142969 |
| 71 | 0.999648023906308 | 0.389729277890587 | 0.609918746015721 |
| 101 | 1.000101210766072 | 0.398433284779739 | 0.601667925986333 |

B均值=1.000137094658772，样本σ_B=0.000469290955224；R均值=0.396242123322197，样本σ_R=0.015405849970995（n=5，分母n-1=4）。
改善=0.603894971336576；相对改善/归一化R=0.603812191910163（U=0）；3σ_B=0.001407872865673；改善=1286.824228σ_B；到U剩余0.396242123322197。
来源：optimization_evidence/{baseline,reference}_runs/seed_{17,29,43,71,101}/result.json#metrics.primary.value及各自per-sample.jsonl；逐题延迟/校准比重算全部960角色记录，与result一致。十个run.log为已披露同一五seed三角色进程的完整日志副本。所有方法/日志/逐样本hash与索引匹配。
质量：480/480成对整段token和答案一致，逐seed准确率相同；seed43解析率95/96，其余100%，全部门槛通过。

## Baseline 合理性
结论：在已提供的既有 Dev 材料范围内合理；未发现直接不公平红旗。全局平台架构/预算问题单列，不把旧卡或哈希差异当作弱化证据。
- training_sufficiency: not_applicable；冻结模型推理；seed为evaluation_seed，不冒充独立训练。 证据：optimization_evidence/训练证据说明.md:3; optimization_evidence/baseline_runs/seed_17/result.json:7
- method_representativeness: pass；普通AR是无需草稿策略的直接对照，并非随机/常数预测；返回空草稿仍真实目标逐token推理。 证据：optimization_evidence/baseline_runs/method.py:4; workspace/backend/policy_decoder_v1.py:97
- budget_comparability: pass；同批反平衡顺序、同每位置RNG、相同冻结模型与128/256预算，启动和IPC计入；整体平台资源违规另列。 证据：workspace/backend/run_policy_score_v2.py:104; workspace/backend/policy_decoder_v1.py:16; workspace/backend/policy_decoder_v1.py:147
- implementation_correctness: pass；AR真实执行目标forward和采样；五个seed两侧480条整段token相同、准确率相同；未发现对照关闭关键输入。 证据：workspace/backend/policy_decoder_v1.py:21; workspace/backend/policy_decoder_v1.py:51; workspace/backend/policy_decoder_v1.py:97
- selection_integrity: pass；五个正式seed全交，均在一个声明的三角色进程测量；完整日志重复存于各角色/seed是已披露共享进程，不是五次训练。 证据：optimization_evidence/baseline_runs/seed_17/result.json:84; optimization_evidence/comparison_summary.json:10

| 项目 | 状态 | 理由/证据 |
|---|---|---|
| B01 | pass | 普通 AR 是合法目标推理的基准，并兼任 Starter 与主评分 B；选择理由明确。 optimization_evidence/训练证据说明.md:5；optimization_evidence/baseline_runs/method.py |
| B02 | pass | 不训练；目标输入、采样和forward路径存在且原始tokens/质量一致，未见功能故障。 workspace/backend/policy_decoder_v1.py:16；workspace/backend/policy_decoder_v1.py:97 |
| B03 | pass | 成对数据、目标与预算相同，三角色反平衡且计启动/IPC；平台资源设计另见C01/C03。 workspace/backend/run_policy_score_v2.py:104；workspace/backend/run_policy_score_v2.py:136 |
| B04 | pass | Baseline无异常学习率等训练参数；R研究草稿生成策略，固定采样配置与选择协议披露。 optimization_evidence/baseline_runs/seed_17/result.json#protocol；optimization_evidence/reference_runs/method.py |
| B05 | pass | 全部五个evaluation seed成对，真实seed经hash转入torch RNG；不宣称独立训练。 workspace/backend/run_policy_score_v2.py:105；workspace/backend/policy_decoder_v1.py:17 |
| B06 | pass | 普通AR即适用基本锚点；无需编造随机输出对照，无包内简单锚点直接优于它的反证。 optimization_evidence/baseline_runs/method.py；workspace/harbor_task/environment/starter.py |
| B07 | pass | 两侧除草稿方法外固定；原始480对整段token相同，改进归因于减少forward及策略开销。 optimization_evidence/reference_runs/method.py:5；workspace/backend/policy_decoder_v1.py:105 |
| B08 | pass | 来源、方法、冻结权重、协议、五seed及历史漂移修正规则有披露。 optimization_evidence/训练证据说明.md:3；optimization_evidence/训练证据说明.md:9 |

## G01–G03
| 项目 | 状态 | 语义结论与证据 |
|---|---|---|
| G01 | pass | draft(state) 可执行任意合法标准库策略，真实调用进入独立候选进程；Reference 实现结构前缀提议，属于方法而非仅常量搜索。 workspace/harbor_task/instruction.md:17；workspace/backend/policy_jail_runner.py:38；workspace/backend/policy_decoder_v1.py:80；optimization_evidence/reference_runs/method.py:5 |
| G02 | pass | 在包内既有公开 Dev 协议范围内 Baseline 合理：普通 AR 与校准独立测量；与 R 同模型、数据、seed、采样及预算，不训练，无实现故障或挑 seed 直接反证。 optimization_evidence/baseline_runs/method.py:4；optimization_evidence/训练证据说明.md:3；workspace/backend/run_policy_score_v2.py:96；optimization_evidence/baseline_runs/seed_17/result.json#protocol |
| G03 | pass | 全部五组原始逐样本与 result 一致；R归一化0.603812191910，改善1286.824σ_B，所有质量门通过，剩余空间充足。A100迁移影响另列待补。 optimization_evidence/comparison_summary.json:17；optimization_evidence/baseline_runs/seed_17/per-sample.jsonl；optimization_evidence/reference_runs/seed_17/per-sample.jsonl |

## 双轨与时间
模型路由投影均为指定模型/high；每条各自核算，未相加。时区UTC见score-and-time-derivation。区间并、差、交由本审查独立代码重算，没有执行包内clock source。
- gpt：19轮，成功FULL 5轮，其余真实失败保留null；保守有效36749.336764秒（10.208149101小时）。最佳原始Dev=0.423135843763386，归一化=0.576922157949004。全部轮恰八字段；2816条投影事件与包内节选逐项对上，3条缺原日志时段计零。
- seed：8轮，成功FULL 5轮，其余真实失败保留null；保守有效41497.018063秒（11.526949462小时）。最佳原始Dev=0.321564551439560，归一化=0.678479527299933。全部轮恰八字段；3510条投影事件与包内节选逐项对上，2条缺原日志时段计零。
GPT探索保守历史复用后转向结构前缀/闭合，后续同方法多次测量及失败；Seed探索结构模板、答案猜测、动态抽取等，最终最佳采用a0a4bb…，与best_method/method.py SHA一致。重复测量不作为新的方法贡献。
独立性静态依据：Agent镜像仅含公开Dev/Starter/CLI；Reference与专家证据在构建context外，主控gpt/seed独立队列及按物理lane绑定评分。包内未见直接交叉泄露反证；公开活动是节选，不能认证未公开提示及完整历史绝无交叉可见，未做本次物理隔离实测。
起止、失败退避、排队、GPU/CLI重叠、原时段cap均核对；正常分析/推理可计，后补完成查询与主控工作不计。公开活动为明确节选，本结论不声称独立认证日志真实性。

## 核心结论
FAIL / NOT READY。21项：{'pass': 17, 'fail': 3, 'non_blocking_pending': 1}；审查已完成。G01/G02/G03均通过，但核心平台架构、内场配置和完整单轮预算不符合当前合同。
本报告不是官方0.3.10输出，不是本次独立复跑或平台验收。

### C01 正式运行依赖 Compose、两个额外 broker 容器和 SSH 外部 GPU 后端；task.toml 却声明 gpus=0，不符合每 sandbox 单容器并直接使用自身 GPU 的当前合同。
证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:217；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:39；workspace/host_adapter/host_paths.py#gpu_command；workspace/harbor_task/task.toml:24
最小修改：删除 environment/ 和 tests/ 直属 Compose 文件及外部 GPU/额外 broker 容器依赖；将可信推理、私有评分及候选隔离部署进各自单容器，用本 sandbox 分配的 GPU。
验收材料：重交完整三目录包；给出单容器 Dockerfile、直接 GPU 配置、公开 Dev/私有 Verifier 实际调用、候选移交和有限 reward 记录。

### C02 任务声明并强制 no-network，配置 storage_mb=512；这些是当前内场明确不声明的约束，且适配器仅接受禁网策略。
证据：workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:23；workspace/harbor_task/task.toml:27；workspace/host_adapter/heima_harbor_environment.py:74
最小修改：移除 task.toml 的 storage_mb 与 no-network 声明，并按内场实际联网/资源能力同步题面、镜像和约束实现。
验收材料：更新后的 TOML、题面和约束执行点；真实格式/GPU 脚本检查结果，以及目标平台接受的资源配置。

### C03 完整私有协议现有 NOP 的 GPU 活动为64740.009秒，独立 Verifier为66585.789秒；支持入口明确取消外层超时。该实际运行设计尚未满足完整单轮≤7200秒。
证据：expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch,completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:17；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/task.toml:17
最小修改：将完整评分设计纳入单轮≤7200秒预算并保留可中断的总超时；不要用 NoWallPrivateTrial 取消时限作为正常入口。若无法满足，向平台提交实测与明确例外申请。
验收材料：同一最终入口的完整训练/推理、重载和评分耗时范围与资源说明；可中断时限配置及完整有限 reward。仅补换卡计时不要求当前自动重跑。

## 放宽待补（非阻断）
- QA21：task_binding四项官方指纹缺失。已有依据：ZIP整体SHA已独立计算；包内有多种源码清单，但不冒充官方四项指纹。建议：真实工具版本/源码SHA与输出；最终run_summary.task_binding及指纹覆盖明细。
- G02/G03/QA03/QA13/QA15：现有实测GPU为A100，未提供H20/L20上的分数/耗时重新核验。已有依据：expert_evidence/current-runtime-anchor-source-binding.json#actual_gpu；保留正式Dev五组真实原数。建议：后续需要时提交真实目标设备与资源配置、迁移影响说明；不以换卡补跑作为本次核心结论前提。C01/C03实质架构/预算问题另列。
- H06/QA17：NOP支持无外层时限入口，文档后改；普通默认入口同版本链路尚未本次验证。已有依据：expert_evidence/supported_invocation/README.md:17-32。建议：同一次Trial配置、result、reward、日志与版本关联；不得把无时限旧记录改写为新普通入口验证。

## 格式与安全
三顶层职责齐全；轻量题包不内嵌巨型模型，外部资产有repository revision/size/hash及安装/私有注入路径。两直属Compose与资源配置是实质不兼容，不只是格式建议。
- expert_evidence/historical_dev_anchors/ [merge_candidate]：与optimization_evidence重复的历史Dev证据可通过明确索引/哈希引用减重；不得删除原始证据唯一副本。
- workspace/backend/ [extra_allowed]：保留可信评分源码/资产索引，但按C01合入目标sandbox执行路径；目录多交本身不扣错。
- expert_evidence/clock_proof/ [extra_allowed]：保留区间投影和出处，明确节选并非完整原始会话。
遍历3054个成员和元数据，针对作者home/邮箱/私网IP/常见凭据定向检查；无作者home/私网IP/常见真实凭据命中，文本邮箱为example保留域，图像随机二进制regex命中不作为邮箱证据；无绝对/..成员或symlink。
不是官方严格扫描；未认证二进制全部编码/嵌套对象与许可，公开发布须独立严格门禁。

## QA01–QA21
| 检查项 | 状态 | 原因与证据 |
|---|---|---|
| QA01 | pass | 八部分齐全，无论文标题、arXiv、参考源码或作者机器路径直接泄露。 workspace/harbor_task/instruction.md:1；workspace/harbor_task/instruction.md:3；workspace/harbor_task/instruction.md:44 |
| QA02 | pass | Dev与私有评分均为(B-M)/(B-U)，U=0，未clip；质量无效另返-1。 workspace/backend/run_policy_score_v2.py:147；workspace/backend/run_policy_score_v2.py:154；workspace/backend/run_private_hidden_score_v2.py:186 |
| QA03 | pass | Reference源码与入口、五个正式result/log/per-sample齐全，归一化0.603812满足范围；非训练不需新训练模型。 optimization_evidence/reference_runs/method.py；optimization_evidence/comparison_summary.json:71；optimization_evidence/reference_runs/seed_17/result.json:84 |
| QA04 | pass | 按全部五组原始值复算n-1标准差；Δ=0.603894971337 ≥3σ=0.001407872866。 optimization_evidence/comparison_summary.json:44；workspace/backend/run_policy_score_v2.py:149 |
| QA05 | fail | token/forward/候选大小和IPC等有真实限制；正式环境仍依赖额外容器/外部GPU且强制当前内场不接受的禁网/磁盘配置。 workspace/backend/isolated_policy.py:16；workspace/backend/policy_decoder_v1.py:169；workspace/harbor_task/task.toml:12；workspace/host_adapter/heima_harbor_environment.py:217 |
| QA06 | pass | test.sh单命令调用grade，有限标量写Harness reward，基础设施错误退出不伪装正常0分。 workspace/harbor_task/tests/test.sh:3；workspace/harbor_task/tests/grade.py:111；workspace/harbor_task/tests/grade.py:125；workspace/harbor_task/tests/grade.py:131 |
| QA07 | pass | 仅检查题面：提及Hidden边界和样本量，无私有样本、标签或可恢复答案。 workspace/harbor_task/instruction.md:9；workspace/harbor_task/instruction.md:32；workspace/harbor_task/instruction.md:46 |
| QA08 | pass | 仅检查题面：未给Reference代码/配置/得分或具体答案；Devσ阈值属于公开评价协议。 workspace/harbor_task/instruction.md:13；workspace/harbor_task/instruction.md:15；workspace/harbor_task/instruction.md:36 |
| QA09 | pass | 评分前删除reward.txt/json及pending，可信最终值通过pending.replace原子写入。 workspace/harbor_task/tests/grade.py:54；workspace/harbor_task/tests/grade.py:131 |
| QA10 | pass | 候选在chroot降权/seccomp进程，可信源码/私有数据以独立不可变清单核验，候选分数不被采信。静态内容不等于本次物理认证。 workspace/backend/policy_jail_launcher.py:10；workspace/backend/policy_jail_runner.py:21；workspace/backend/trusted_private_backend_support.py:59；workspace/harbor_task/tests/grade.py:111 |
| QA11 | pass | 目标、环境、公式、可改范围、边界、提交格式、迭代与完成条件均有实质内容。 workspace/harbor_task/instruction.md:3；workspace/harbor_task/instruction.md:7；workspace/harbor_task/instruction.md:11；workspace/harbor_task/instruction.md:17；workspace/harbor_task/instruction.md:28；workspace/harbor_task/instruction.md:34；workspace/harbor_task/instruction.md:40；workspace/harbor_task/instruction.md:44 |
| QA12 | pass | ZIP清点无.git目录或Git历史；未以私有报告替代核验。 @inventory:3054_members_no_.git |
| QA13 | pass | 全部正式五seed成对有效、每seed同向改善；原始ratio、质量门、日志与method hash复算一致。 optimization_evidence/comparison_summary.json:17；optimization_evidence/训练证据说明.md:5；optimization_evidence/baseline_runs/seed_17/result.json#artifacts |
| QA14 | pass | 代码级策略空间开放；可研究基于实测提议成本和接受概率的动态预算控制等未规定具体实现，R尚余0.39624到U。 workspace/harbor_task/instruction.md:36；workspace/backend/policy_decoder_v1.py:80；expert_evidence/best_method/method.py:583 |
| QA15 | fail | 当前完整私有单轮实测GPU64740秒、Verifier66586秒，支持入口取消外层超时，尚无≤7200秒完整方案。A100→H20/L20迁移计时单独待补。 expert_evidence/native_harbor/current-nop/scorer-result.json:10；expert_evidence/native_harbor/current-nop/result.json:112；expert_evidence/supported_invocation/README.md:17；workspace/operator/run_recover_private_native.py:173 |
| QA16 | pass | 独立复算起止、失败退避/队列扣除、GPU/CLI并集及原时段上限：GPT36749.337秒、Seed41497.018秒，各≥10h；五条缺日志全排除。 expert_evidence/clock_proof/clock-source-projections.json#roles；expert_evidence/research_activity_originals/；expert_evidence/clock_proof/README.md:10 |
| QA17 | fail | separate、artifacts及可信reward实现存在；H02/H03/H05受C01-C03核心不兼容阻断。H06版本关联单列非阻断。 workspace/harbor_task/task.toml:2；workspace/harbor_task/task.toml:15；workspace/host_adapter/heima_harbor_environment.py:217；workspace/operator/run_recover_private_native.py:173 |
| QA18 | pass | 两轨19/8轮，每轮恰八字段、时间可解析；成功各5轮。全部score与声明转换记录代数一致，失败为null，保留标记顺序一致；Seed最佳源码hash吻合。 expert_evidence/trajectory_codex.json#rounds；expert_evidence/trajectory_seed.json#rounds；expert_evidence/score-and-time-derivation.json；expert_evidence/best_method/method.py |
| QA19 | pass | 明确随机评估、五个固定seed、n-1σ及3σ有效提升阈值；非确定性不豁免。 workspace/harbor_task/instruction.md:13；workspace/harbor_task/instruction.md:15；workspace/harbor_task/instruction.md:26 |
| QA20 | pass | 题面明确可改solution、工具实际命令、标准库及网络规则。网络规则与内场不符另在QA05退回。 workspace/harbor_task/instruction.md:17；workspace/harbor_task/instruction.md:21；workspace/harbor_task/instruction.md:30；workspace/harbor_task/instruction.md:32 |
| QA21 | non_blocking_pending | run_summary缺task_binding.hash_profile/task_content_sha256；不自创官方四项指纹算法，不单独判退。 expert_evidence/run_summary.json |

## H01–H06
目标版本为包内声明Harbor0.23.0；provider为自定义HeimaDockerEnvironment。COPY静态路径按environment/和tests/各自context审，TOML仅语法解析；未原生加载schema。
| 检查项 | 状态 | 原因与证据 |
|---|---|---|
| H01 | pass | 真实根、两Dockerfile和test.sh齐全；私有1380行走root-only注入，按固定样本/图像hash校验并由私有评分实际读取，非空目录占位。 workspace/harbor_task/task.toml；workspace/backend/trusted_private_backend_support.py:91；workspace/backend/run_private_hidden_score_v2.py:77；expert_evidence/native_harbor/current-nop/scorer-result.json |
| H02 | fail | TOML语法可解析、声明Harbor0.23.0与separate；storage_mb/no-network违反当前内场配置合同。包内schema仅作待核版本证据，不冒称原生加载。 workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:23；expert_evidence/harbor-0.23.0-native-schema.json |
| H03 | fail | COPY均相对正确context、入口/移交一致且不复制私有材料；实际运行依赖Compose、额外broker和SSH外部GPU，违反单容器自身GPU边界。 workspace/harbor_task/environment/Dockerfile；workspace/harbor_task/tests/Dockerfile；workspace/host_adapter/heima_harbor_environment.py:217；workspace/host_adapter/evaluator_bridge.py:51 |
| H04 | pass | grade清理旧分数、核验全私有协议/源hash与有限值，可信原子写reward；已有NOP的result/scorer/reward一致。 workspace/harbor_task/tests/grade.py:54；workspace/harbor_task/tests/grade.py:111；workspace/harbor_task/tests/grade.py:131；expert_evidence/native_harbor/current-nop/reward.txt |
| H05 | fail | 所交实际支持入口NoWallPrivateTrial覆盖外层超时为None，完整私有约18h；普通10800秒配置本身也高于7200秒单轮预算，无法按当前预算宣称可用。 workspace/operator/run_recover_private_native.py:173；workspace/operator/run_recover_private_native.py:205；workspace/harbor_task/task.toml:17；expert_evidence/supported_invocation/README.md:17 |
| H06 | non_blocking_pending | 已有同一Trial NOP配置/结果/reward/log、separate及完整6900对，支持其无外层时限调用；题面文档后改，普通默认入口同版本链路待核，不因NOP版本缺口单独阻断。 expert_evidence/native_harbor/current-nop/config.json；expert_evidence/native_harbor/current-nop/result.json；expert_evidence/native_harbor/current-nop/reward.txt；expert_evidence/native_harbor/current-nop/trial.log；expert_evidence/supported_invocation/README.md:22 |

## 工具缺口
- official QA Skill 0.3.10/check_submission.py/GPU precheck/task_fingerprint.py：unavailable。取得真实发布包后按真实CLI验收并记录工具SHA；本报告不冒充其输出。
- strict privacy-check：not_run。本次完成本地只读定向安全复核；公开发布前由可信安装的privacy-check --strict逐件扫描主ZIP和报告附件。本包内release_tools未执行。

## 目标环境未本次实跑
- 未构建镜像或原生Harbor加载/Trial
- 未实跑H20/L20与完整Public/Hidden
- 未对候选进行本次动态攻击/物理隔离认证
- 未做本次12小时稳定性压测，包内既有43233秒713观测仅保留其历史范围

## 可复制给专家
本次只读复核结论：NOT READY。公开Dev正式B/R五组证据和提升通过；Baseline在已交范围内合理，双轨保守有效时长各≥10h。请处理以下核心问题：

C01：正式运行依赖 Compose、两个额外 broker 容器和 SSH 外部 GPU 后端；task.toml 却声明 gpus=0，不符合每 sandbox 单容器并直接使用自身 GPU 的当前合同。
证据：workspace/harbor_task/environment/docker-compose.yaml:1；workspace/harbor_task/tests/docker-compose.yaml:1；workspace/host_adapter/heima_harbor_environment.py:217；workspace/host_adapter/heima_harbor_environment.py:223；workspace/host_adapter/heima_harbor_environment.py:228；workspace/host_adapter/evaluator_bridge.py:39；workspace/host_adapter/host_paths.py#gpu_command；workspace/harbor_task/task.toml:24
修改：删除 environment/ 和 tests/ 直属 Compose 文件及外部 GPU/额外 broker 容器依赖；将可信推理、私有评分及候选隔离部署进各自单容器，用本 sandbox 分配的 GPU。
验收：重交完整三目录包；给出单容器 Dockerfile、直接 GPU 配置、公开 Dev/私有 Verifier 实际调用、候选移交和有限 reward 记录。

C02：任务声明并强制 no-network，配置 storage_mb=512；这些是当前内场明确不声明的约束，且适配器仅接受禁网策略。
证据：workspace/harbor_task/task.toml:12；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:23；workspace/harbor_task/task.toml:27；workspace/host_adapter/heima_harbor_environment.py:74
修改：移除 task.toml 的 storage_mb 与 no-network 声明，并按内场实际联网/资源能力同步题面、镜像和约束实现。
验收：更新后的 TOML、题面和约束执行点；真实格式/GPU 脚本检查结果，以及目标平台接受的资源配置。

C03：完整私有协议现有 NOP 的 GPU 活动为64740.009秒，独立 Verifier为66585.789秒；支持入口明确取消外层超时。该实际运行设计尚未满足完整单轮≤7200秒。
证据：expert_evidence/native_harbor/current-nop/scorer-result.json#gpu_started_epoch,completed_epoch；expert_evidence/native_harbor/current-nop/result.json#verifier；expert_evidence/supported_invocation/README.md:17；workspace/operator/run_recover_private_native.py:173；workspace/harbor_task/task.toml:17
修改：将完整评分设计纳入单轮≤7200秒预算并保留可中断的总超时；不要用 NoWallPrivateTrial 取消时限作为正常入口。若无法满足，向平台提交实测与明确例外申请。
验收：同一最终入口的完整训练/推理、重载和评分耗时范围与资源说明；可中断时限配置及完整有限 reward。仅补换卡计时不要求当前自动重跑。

非阻断待补：官方四项指纹、A100向H20/L20迁移影响及目标耗时、NOP文档/默认入口版本关联可后补；不单凭这些要求全量重跑。实际额外容器/远程GPU和取消时限的问题不属于放宽范围。
本机无真实官方0.3.10工具；内容复核不冒称官方新版自检或平台通过。静态材料一致不等于独立复现；本次未运行提交代码、模型、Docker或训练。
