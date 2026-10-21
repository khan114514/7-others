# Auto2912 完整 ZIP 独立静态复核

优化面：固定训练好的上游交通预测器输出上，学习可执行条件校正方法；三seed完整Dev MAE最小化。
实际guard允许任意fit/predict方法，限制16个Python文件、源大小、fit/reload时间、状态和输出。冻结数据/协议/可信评分。Reference为新增残差学习，非单纯参数搜索。证据：workspace/harbor_task/instruction.md；workspace/harbor_task/tests/engine/grader.py；workspace/harbor_task/tests/engine/actor_worker.py。

Baseline：正式Baseline=未经改动Starter identity，直接返回固定骨干预测；共享骨干在PEMS08上真实训练63epoch后按patience20早停并选择best-val模型，后处理fit返回标识状态。
Reference：9维base/近期历史/斜率/锚点/周期/步长/节点特征→HistGradientBoostingRegressor拟合残差，absolute_error、100迭代上限、15叶、learning_rate0.08/l2=10，预测加0.5倍校正。

Baseline五维事实：
- training_sufficiency：共享骨干实际63epoch，best-val epoch43，正常patience20早停；无差别截短的直接证据。正式identity无参数更新需求。
- method_representativeness：对固定预测后处理任务，原预测不变是可辩护的朴素正式对照；未用包外SOTA推断。
- budget_comparability：同公开校正数据、Dev、每seed180+60秒与2CPU/4GiB；校正模型结构允许改变。
- implementation_correctness：fit/predict调用生效，标签在候选退出后才由root读取；六个checkpoint索引大小与SHA一致，已有reload退出0/预测差0。
- selection_integrity：完整三seed；共同骨干明确共享，U在Reference开始前约3秒冻结；三组B/R原值可追溯。

| Baseline专项 | 结论 | 事实与证据 |
|---|---|---|
| B01 | 通过 | Starter、正式Baseline与B均为同一固定上游预测器的identity输出；它是固定预测后处理任务的自然对照，不是随机/故障网络。 workspace/harbor_task/environment/engine/starter.py；expert_evidence/provenance/shared-backbone-training.run.log；expert_evidence/provenance/forecast-export-manifest.json；optimization_evidence/训练证据说明.md |
| B02 | 通过 | 共享上游计划200epoch，日志实际63epoch、42147步、patience20正常早停；最优验证epoch43。identity无需SGD；三次保存48字节状态并独立重载。 workspace/harbor_task/environment/engine/starter.py；expert_evidence/provenance/shared-backbone-training.run.log；expert_evidence/provenance/forecast-export-manifest.json；optimization_evidence/训练证据说明.md；optimization_evidence/baseline_runs/seed_17/run.log |
| B03 | 通过 | 双方相同固定预测、74个校正origin、148个Dev origin、170节点及12步预测；每seed同180秒fit/60秒reload、2CPU/4GiB。任务允许新增校正训练。 workspace/harbor_task/instruction.md:13；workspace/harbor_task/tests/engine/grader.py:104；expert_evidence/provenance/forecast-export-manifest.json |
| B04 | 通过 | identity无搜索参数；Reference显式树模型与固定0.5混合，结构/校正训练为允许变量。未见只削弱Baseline的异常配置证据。 workspace/harbor_task/environment/engine/starter.py；workspace/reference/method.py；optimization_evidence/训练证据说明.md |
| B05 | 通过 | 17/29/43全交；worker将seed传入fit且设置Python/NumPy随机源，Reference传入random_state；同方法跨seed哈希不变。共享神经骨干未冒充三次神经重训。 workspace/harbor_task/tests/engine/actor_worker.py:33；workspace/reference/method.py:18；optimization_evidence/comparison_summary.json |
| B06 | 通过 | identity本身即合理恒等锚点；其分数等于B，归一化0不意味着无效。未发现包内同协议的更弱/异常基线被选作B。 workspace/harbor_task/environment/engine/starter.py；workspace/harbor_task/tests/verifier_assets/protocol.json |
| B07 | 通过 | 收益归因于固定预测上的条件残差校正整体方案；同Dev、种子、评分器和资源上限，未声称各组件独立消融贡献。 workspace/harbor_task/tests/engine/grader.py；workspace/reference/method.py；expert_evidence/provenance/forecast-export-manifest.json |
| B08 | 通过 | 专家说明清楚披露固定骨干、非独立神经重训、CPU后处理边界及先验pilot的U；未见挑最差Baseline/最好Reference seed。 expert_evidence/专家作业说明文档.md:3；optimization_evidence/训练证据说明.md:15；expert_evidence/provenance/anchors-frozen-before-reference.json |

Baseline总评：在已提供材料范围内合理。共享骨干非每seed重训，identity确定性重复合法；未发现直接削弱红旗。

| 正式seed | Baseline MAE | Reference MAE | B−R | 原始来源 |
|---|---:|---:|---:|---|
| 17 | 15.421857156583444 | 15.094631864779133 | 0.327225291804311 | optimization_evidence/baseline_runs/seed_17/result.json#metrics.mae；optimization_evidence/reference_runs/seed_17/result.json#metrics.mae |
| 29 | 15.421857156583444 | 15.100112152134592 | 0.321745004448852 | optimization_evidence/baseline_runs/seed_29/result.json#metrics.mae；optimization_evidence/reference_runs/seed_29/result.json#metrics.mae |
| 43 | 15.421857156583444 | 15.093119894248630 | 0.328737262334814 | optimization_evidence/baseline_runs/seed_43/result.json#metrics.mae；optimization_evidence/reference_runs/seed_43/result.json#metrics.mae |

均值 B=15.421857156583444；R=15.095954637054119。n−1样本标准差 σ_B=0.0，σ_R=0.0036790233017135。改善=0.325902519529325（2.11325080%）；U=14.948519443814908；归一化=0.688520079296307；剩余R−U=0.147435193239211。
三seed全部有效且逐项改善。B确定性，R随机fresh-fit；3σ_B=0但仍要求真实正改善，不能把σ=0称为全评估确定或无限显著性。6个checkpoint大小/SHA逐件与索引一致，已有独立reload exit0、最大预测差0；未加载模型。完整模型路径、哈希与逐seed成本见JSON baseline_quality.model_evidence。
Reference拟合日志为worker实际stdout的一行结构记录，结合源码fit路径、独立状态文件与重载证据，可支持实际拟合；不将其误称逐迭代学习曲线。上游实际训练日志为63epoch、42147步、早停20轮，best val epoch43。

| 内容门 | 结论 | 证据与理由 |
|---|---|---|
| G01 | pass | 16个Python文件内开放可执行fit/predict，guard仅限制源文件/资源/输出；Reference新增条件残差树学习，非只改标量参数。 workspace/harbor_task/instruction.md:13；workspace/harbor_task/tests/engine/grader.py:56；workspace/harbor_task/tests/engine/actor_worker.py:24；workspace/reference/method.py |
| G02 | pass | 在已提供静态材料范围内，固定预测identity正式Baseline合理；共享骨干和校正训练角色披露，未发现不公平削弱。 workspace/harbor_task/environment/engine/starter.py；expert_evidence/provenance/shared-backbone-training.run.log；expert_evidence/provenance/forecast-export-manifest.json；optimization_evidence/训练证据说明.md |
| G03 | pass | 全部三seed成对有效；均值改善0.3259025195293255，归一化0.6885200792963072在范围内。B为确定性identity，R拟合有波动，按随机口径复算3σ_B=0仍有真实正向改善。 optimization_evidence/comparison_summary.json；expert_evidence/provenance/anchors-frozen-before-reference.json；expert_evidence/provenance/pilot-upper-estimate.json；optimization_evidence/baseline_runs/seed_17/result.json#metrics.mae；optimization_evidence/baseline_runs/seed_29/result.json#metrics.mae；optimization_evidence/baseline_runs/seed_43/result.json#metrics.mae；optimization_evidence/reference_runs/seed_17/result.json#metrics.mae；optimization_evidence/reference_runs/seed_29/result.json#metrics.mae；optimization_evidence/reference_runs/seed_43/result.json#metrics.mae |

双轨有效时长与八字段：

| 轨迹 | 模型/档位 | 打包轮数 | 正工作时间轮数 | 轮内elapsed | 扣除并集 | 初始实际CLI | 有效秒/小时 | 最终Dev |
|---|---|---:|---:|---:|---:|---:|---|---|
| gpt | gpt-6.1-sol/xhigh | 27 | 22 | 36641.069331 | 688.296345 | 51.017922 | 36003.790908 / 10.001053030h | MAE=13.902094246542；score=3.210737004563 |
| seed | doubao-seed-2-1-turbo-260628/high | 79 | 26 | 39019.004448 | 3046.633039 | 36.015869 | 36008.387279 / 10.002329800h | MAE=15.165965823367；score=0.540610490806 |

全部成功轨迹score逐轮与trusted-evaluation-command.json原始raw_mean相符，17个空分失败全部可定位HTTP409评分busy。GPT 5轮、Seed 53轮有效秒为0，均已从时长扣除，但run_summary仍把全部27/79条写为effective_rounds；这些是记录数量而非有效方法闭环数量。
八字段均出现，但GPT16条另加original_status，Seed79条另加method_summary_evidence（其中1条又加original_status）；按V4八字段导出应移入侧表。GPT末轮描述同时写“retained”历史模型文字和当前failed，需区分当前轮实际评分与引述历史。既有best_method在冻结后独立本地Trial复验成功，最终Hidden仅在冻结后注入；不计入长轨时长。
初始真实读题与工具检查分别51.017922/36.015869秒可计；本次没有把容器12h uptime、下载或冻结后复验补入10h。含拟合任务不适用7h例外。

格式：三大目录齐全；B/R可追溯但result为旧字段布局，缺显式role/task_type等结构字段；轨迹有扩展字段与错误有效轮数。Compose部署属于核心问题。
- obsolete_layout — optimization_evidence/baseline_runs/seed_17/result.json：将现有真实字段归入schema_version/role/seed/task_type/method/protocol/training/execution/metrics/quality_gate/artifacts；B无新增训练写NOT_APPLICABLE，R写fresh fit。补方法源码相对路径和依赖标识，不编造旧记录缺失值。
- merge_candidate — optimization_evidence/comparison_summary.json：补逐seed result相对来源路径；当前数值已由本次直接读取原result验证。
- misplaced — expert_evidence/trajectory_seed.json：method_summary_evidence与original_status移到独立证据映射，单轮保留八字段；原始事件继续保存。
- extra_allowed — expert_evidence/provenance：保留原始训练、CLI、Trial与失败证据；移出无需随公开任务发布的历史重复候选与大CLI二进制，按专家附件职责归位。
- merge_candidate — workspace/harbor_task/tests/instruction.md：同步任务根与Verifier副本，避免网络/资源口径修复时只改一份。

核心结论：NOT READY：方法空间、Baseline、正式Reference提升和双轨有效时长均有支持；当前内场环境配置/构建/单容器要求明确不符，轨迹有效轮数与故障分类也需修正。不是由五类放宽缺口造成。
检查完成度：已完成安全静态内容复核；目标平台动态验收未执行。

| Harbor项 | 结论 | 事实与证据 |
|---|---|---|
| H01 | pass | 任务根与入口完整；公开Dev经受保护local scorer可迭代，专家侧真实Hidden资产、注入实现及冻结后Hidden Trial齐全，未把仅有目录当完成。 workspace/harbor_task/task.toml；workspace/harbor_task/tests/test.sh；expert_evidence/held_out_assets/INJECTION.md；expert_evidence/provenance/harbor_adapter/post_selection_harbor.py:32；expert_evidence/provenance/native_harbor/selected_hidden_injection/returned-result.json |
| H02 | fail | TOML可静态解析，separate显式；但当前storage_mb/no-network与内场合同冲突。已有Harbor0.23私有适配记录不能证明当前平台接受。 workspace/harbor_task/task.toml；expert_evidence/专家作业说明文档.md:15 |
| H03 | fail | Verifier tests/ context自洽；Agent COPY要求environment/...及根solution，无法直接以environment/构建。两处Compose与额外broker依赖不满足当前单容器规则。 workspace/harbor_task/environment/Dockerfile:5；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/environment/docker-compose.yaml；workspace/harbor_task/tests/docker-compose.yaml；expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:69 |
| H04 | pass | test.sh→grader写/logs/verifier/reward.txt，已有Trial reward与结果一致；成功原子写，Hard Gate=-1，执行/超时错误不伪造正常分数。 workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/engine/grader.py:207；expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/verifier/reward.txt |
| H05 | pass | 已有Trial选择本题、nop与自定义FrozenEvaluationEnvironment，verifier未禁用；任务artifacts移交solution，候选源码有manifest。此项只认可本地调用记录。 expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/config.json；expert_evidence/provenance/native_harbor/nop/returned-result.json；workspace/harbor_task/task.toml:2 |
| H06 | pass | 同一次NOP config/result/reward/log齐全：nop、separate、exception=null、reward=0，55个任务文件逐件hash与包中一致。为既有本地0.23记录，本次未运行。 expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/config.json；expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/result.json；expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/trial.log；expert_evidence/provenance/native_harbor/nop/forecast-final-nop-gpt-43a9918f/verifier/reward.txt；expert_evidence/provenance/native_harbor/nop/task-source-manifest.json |

Harbor边界：既有本地0.23.0自定义Docker适配器；55个文件与NOP及最终Trial source-manifest完全一致。CPU0GPU；记录2CPU/4GiB，无GPU请求、无特权容器。Agent/Verifier不同sandbox与候选bwrap隔离均有静态实现，但模型轨另需broker容器且有Compose依赖，不能宣称当前单容器部署合格。

| 检查项 | 结论 | 具体事实与证据 |
|---|---|---|
| QA01 | pass | 题面有八部分且未披露论文标题、arXiv、参考源码或答案线索。 workspace/harbor_task/instruction.md |
| QA02 | pass | 可信评分使用(B-mean)/(B-U)，未裁剪；B映射0、U映射1，minimize方向连续单调。 workspace/harbor_task/tests/engine/grader.py:187；workspace/harbor_task/tests/verifier_assets/protocol.json:15 |
| QA03 | pass | Reference入口、三seed原始结果/日志/模型/reload齐全，字节元数据一致；归一化0.6885200793。 workspace/reference/method.py；optimization_evidence/reference_runs/result.json；optimization_evidence/reference_runs/seed_17/result.json；optimization_evidence/reference_runs/seed_29/result.json；optimization_evidence/reference_runs/seed_43/result.json |
| QA04 | pass | 随机Reference拟合不能按固定seed视为全确定；三次B样本标准差0，改善>0且≥3σ_B。该检验不意味着统计泛化已独立证实。 optimization_evidence/comparison_summary.json；workspace/harbor_task/tests/engine/actor_worker.py:33 |
| QA05 | fail | 虽有UID1000/bwrap、180/60秒超时、源/状态限额与形状检查，但task.toml声明storage_mb及no-network，题面承诺无Internet，不符合当前内场合同。 workspace/harbor_task/task.toml:9；workspace/harbor_task/task.toml:18；workspace/harbor_task/instruction.md:16；workspace/harbor_task/tests/engine/grader.py:111 |
| QA06 | pass | 固定solve.sh调用公开评分；test.sh调用可信grader，成功写有限reward，格式/超时/Hard Gate分开处理；Hard Gate=-1。 workspace/harbor_task/solution/solve.sh；workspace/harbor_task/tests/test.sh；workspace/harbor_task/tests/engine/grader.py:207 |
| QA07 | pass | 仅按题面检查：没有Hidden样本、标签或可恢复答案。 workspace/harbor_task/instruction.md |
| QA08 | pass | 仅按题面检查：未披露Reference代码、配置或数值答案。 workspace/harbor_task/instruction.md |
| QA09 | pass | test.sh先删reward.txt/json，成功写reward.pending.txt并原子replace；候选无reward访问面。 workspace/harbor_task/tests/test.sh:4；workspace/harbor_task/tests/engine/grader.py:220 |
| QA10 | pass | 公开冻结资产只读、Dev标签root-only；源快照只读并核源码hash，solve固定hash；候选在bwrap受限身份中执行，root不加载pickle。 workspace/harbor_task/tests/Dockerfile:14；workspace/harbor_task/tests/engine/grader.py:100；workspace/harbor_task/tests/engine/grader.py:111；workspace/harbor_task/tests/engine/grader.py:132；workspace/harbor_task/tests/engine/actor_worker.py:20 |
| QA11 | pass | 八部分均有任务输入输出、公式、可改范围、边界、提交、迭代与完成标准；不只空标题。 workspace/harbor_task/instruction.md |
| QA12 | pass | ZIP成员清点无.git目录，未见版本库历史暴露。 @inventory |
| QA13 | pass | 三组17/29/43成对原始结果齐全，逐seed均改善，全部质量门true；六份checkpoint与reload记录可追溯。 optimization_evidence/baseline_runs/result.json；optimization_evidence/reference_runs/result.json；optimization_evidence/comparison_summary.json |
| QA14 | pass | 可实现新的条件表示、稳健校正或空间组合；题面不直接给出参考树方法。正式Reference未达到估计U，R-U=0.1474351932。 workspace/reference/method.py；workspace/harbor_task/instruction.md:13；optimization_evidence/comparison_summary.json |
| QA15 | pass | 正式CPU任务gpus=0，现有inspect为2CPU/4GiB；B完整fit/reload约1.067s，R约19.189s，最终GPT复验约348.119s，低于2h。A100仅共享上游准备，不冒充H20/L20实测。 workspace/harbor_task/task.toml:15；optimization_evidence/baseline_runs/result.json；optimization_evidence/reference_runs/result.json；expert_evidence/provenance/physical_isolation/forecast-final-nop-gpt-43a9918f__verifier__trial/container-inspect.json |
| QA16 | pass | 独立账本复算GPT36003.790908秒、Seed36008.387279秒；扣除并集一致，初始真实读题计入，冻结复验/稳定性不计。任务含拟合，不使用7h例外。 expert_evidence/trajectory_codex.json#duration_seconds；expert_evidence/trajectory_seed.json#duration_seconds；expert_evidence/provenance/model_execution/gpt；expert_evidence/provenance/model_execution/seed |
| QA17 | fail | 已有separate/NOP/Hidden注入记录，但正式包保留禁用Compose，Agent构建依赖task-root context；模型轨适配器创建辅助broker容器，当前内场部署尚不满足。 workspace/harbor_task/environment/docker-compose.yaml；workspace/harbor_task/tests/docker-compose.yaml；workspace/harbor_task/environment/Dockerfile:5；expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:67 |
| QA18 | fail | 分数逐轮与原始评分一致，但摘要把27/79全记录称effective_rounds，账本中5/53轮为0秒基础设施阻塞；另有八字段之外的字段及HTTP409统称方法失败，未忠实区分有效方法与评分故障。 expert_evidence/run_summary.json:10；expert_evidence/run_summary.json:42；expert_evidence/trajectory_codex.json；expert_evidence/trajectory_seed.json；expert_evidence/provenance/model_execution/gpt/round_0017/time-accounting.json；expert_evidence/provenance/model_execution/seed/round_0035/time-accounting.json |
| QA19 | pass | 题面明确seed17/29/43、完整均值、全部门通过及严格正向MAE改善，无额外统一百分比门槛。 workspace/harbor_task/instruction.md:10 |
| QA20 | pass | 题面明确可编辑Python源、可用numpy/scipy/sklearn与固定solve迭代工具、禁止外网及评分篡改；网络口径与内场冲突另列环境配置失败。 workspace/harbor_task/instruction.md:13；workspace/harbor_task/instruction.md:16；workspace/harbor_task/instruction.md:22 |
| QA21 | manual（非阻断） | run_summary无task_binding.hash_profile/task_content_sha256；已有NOP逐文件manifest与当前55个任务文件一致，但不能替代官方四项指纹。按五类放宽记非阻断待补。 expert_evidence/run_summary.json；expert_evidence/provenance/native_harbor/nop/task-source-manifest.json |

具体修改与验收材料：
请按以下动作返修，保留原始B/R、失败和长轨证据：
1. 删除environment/、tests/直属Compose文件；将Agent所需公开文件与starter归入environment/，改COPY为该context内相对来源；使用平台每sandbox单容器能力，移除任务对额外broker容器的依赖。 事实：两份直属Compose文件、environment/Dockerfile第5–12行，以及私有适配器第69–74行。原因：当前要求environment/ context、每sandbox单容器、禁止正式Compose。验收：重交完整ZIP、平台采用的Agent environment/与Verifier tests/构建定义及单容器部署配置；在目标Harbor上补构建、Public Dev、artifacts移交与separate评分记录。
2. 移除task.toml中的storage_mb和任务级禁网/网络白名单声明；题面改用平台实际网络与工具策略，隔离继续依赖文件权限/受限候选/冻结面而不是未支持的禁网承诺。 事实：task.toml第9/14/18/22行及题面第16行。原因：当前内场不接受这些配置承诺。验收：同步task.toml及其tests副本、instruction及适配器；提交目标平台可解析的配置和隔离验证记录。
3. 将run_summary的effective_rounds改为经过方法闭环定义核对的数值，并另列总记录数；把0秒API阻塞/旧候选重评标为基础设施记录；把17个HTTP409失败明确写为评分忙。每轮只保留规定八字段，辅助证据移到独立映射，摘要限定1–3句。 事实：run_summary effective_rounds=27/79，但原始账本中5/53轮为0秒；17个失败原输出是HTTP409。原因：有效方法闭环与基础设施记录混淆。验收：提交归一后的两条轨迹、逐轮到原始CLI/评分/时间账本的映射；不得改变原始分数、时间或删失败历史。复核八字段、状态与零工作轮分类。
4. 最终Hidden入口应要求私有注入材料存在并校验，不要在正式评分时静默退回Dev；保留现有冻结后Hidden注入实现与回执，提供平台安全注入配置。
5. 逐seed结果补齐可追溯schema和比较表source paths；原缺失值填null并解释。可见训练execution/forecast-export索引含作者环境绝对路径，脱敏后逐件做发布隐私扫描。

五类放宽项（不单独阻断）：
- 四项指纹/哈希：待补。run_summary未含task_binding；已有逐文件manifest55项一致，不冒充官方profile。 建议：取得真实官方脚本后补四项指纹；无需仅为此全量重跑。
- 换卡对Baseline/Reference影响：不适用当前核心。正式任务CPU0GPU，共享骨干历史A100保真，未声称已迁移H20/L20。 建议：若平台后续要求迁移上游准备，保留旧证据并补影响说明；本次不强制补跑。
- 换卡耗时：不适用当前核心。正式CPU实测已列，当前没有H20/L20迁移耗时主张。 建议：若将来改卡，实测与估算分列，迁移计时可后补。
- NOP：已有本地记录。同一当前文件版本的nop/separate/reward0材料齐全；新平台验收未做。 建议：部署修复后建议补目标平台同版本NOP；该记录本身不单独阻断。
- 换卡重算分数：不适用当前核心。已直接核对全部CPU正式原值，无仅因换卡要求重算。 建议：后续如换卡影响标定可单列复核，不将该项混作当前Compose与轨迹失败豁免。

工具缺口：真实官方0.3.10、独立格式/GPU/指纹脚本未执行；当前是local-v4-content-review，不是官方自检或平台PASS。
运行边界：本次未执行提交代码、模型、Docker、Verifier或训练；原始长轨/NOP/重载记录仅做静态交叉核对。
在部署和轨迹核心问题修复前不应发布为就绪；不要只提交补丁，应重交三目录完整ZIP并更新SHA256。

风险（最多三条）：
- 正式Hidden依赖专家适配器注入，test.sh缺资产时默默评分Dev；迁移到平台必须将最终Hidden模式设为显式必需并给注入回执。
- 现有U是先验pilot估计，最终GPT候选分数已达3.2107；未裁剪本身正确，但应保留U估计局限并避免把>1解释成异常或反调U。
- 可见训练provenance仍含作者环境绝对路径；发布前逐件脱敏与隐私扫描，不在本报告复制原值。

审查身份与输入：
ZIP SHA256：18ee42736dcb71d492cf47a8f83a3d863d148cdb4b1a3d4a9488a8c8087e3dba
session_id=stssdl_b；configured_model=gpt-6-astra；reasoning_effort=high；fork_turns=none（主控派发声明）；runtime_reported_model=not_exposed；runtime_version=not_exposed。
仅使用该完整ZIP、技能和通用流程合同；没有引用包外报告/同伴结论或包内历史QA。
