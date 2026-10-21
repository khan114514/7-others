# 独立 V4 内容复核：Exact Multihead Retrieval Kernel

本地 schema：local-v4-content-review。核心结论：**NOT READY / FAIL**。静态检查完成；含核心待核事实，未进行动态复跑、官方0.3.10或平台验收。

ZIP：AutoResearch_attention_v023_final_20261009.zip；SHA256：`bd75ad937f2e1ba807bd8db95f569e7cbac350e1258dac27465129a2dff28e42`；745个成员。
审查会话：attention_b；配置模型 gpt-6-astra / high / fork_turns=none（主控派发凭证）；runtime_reported_model=not_exposed；runtime_version=not_exposed。仅接收指定完整ZIP和流程规则，未读同伴或历史QA结论。

## 优化面与实际执行边界

固定数值语义的多头检索算子CPU精确推理加速；开放attention及纯helper、布局/代数/分块。guard允许方法级变化但存在缓存限制绕过。
输入是x、wq/wk/wv、alpha/beta；仅method.py可改，固定数值算子及float32/单线程/输入不变要求。公开Dev与最终私有输入分离。真实入口允许attention和helper，不是只接收常量参数。Reference采用Gram恒等式，属于方法变化。

证据：workspace/harbor_task/instruction.md:10；workspace/harbor_task/tests/grader.py:28；workspace/harbor_task/tests/worker.py:24

但赋值guard未递归检查解包目标，模块属性可成为跨调用缓存；worker连续使用同值输入且最终只克隆同值张量。该评分边界缺陷必须修复，不能以已有数值改善抵消。

## Baseline 专项

总评：在已提供材料范围内合理。Starter=正式Baseline=主评分B来源；健康控制为同一正确Starter。B均值映射0是正常归一化定义。非训练，不需要checkpoint。

未改Starter：q/k投影→原始及扰动softmax→完整通道检索/混合/范数缩放→value投影；alpha=1直接等价简化。非训练。

Reference：按L/C形状选择token Gram恒等式恢复范数，将value投影提前减少完整通道中间量；其他形状退回直接路径。非超参搜索。

双方同输入/5seed/float32/1CPU/2GiB/3warmup/11计时，正式Dev单seed约46–60秒。源码与原始日志支持现有合理性，未发现少训练、故障对照、挑seed或额外模型资源。外部来源未独立验证，不能以简短upstream-parity自述当外部真实性认证。

| 专项 | 状态 | 证据与理由 |
|---|---|---|
| B01 | 通过 | 直接精确算子实现作为朴素正式对照，Starter、正式B与零分锚点映射清晰；未声称SOTA。 workspace/harbor_task/environment/starter/method.py:7；expert_evidence/专家作业说明文档.md:11；optimization_evidence/comparison_summary.json#score_contract |
| B02 | 通过 | 无训练；直接投影/softmax/范数路径与题面一致，全部正式输出通过数值门，未发现初始化模型冒充训练或故障弱化。 workspace/harbor_task/environment/starter/method.py:8；workspace/harbor_task/tests/grader.py:105；optimization_evidence/baseline_runs/seed_17/result.json#quality_gate |
| B03 | 通过 | 双方同五seed、输入生成、1CPU/2GiB、float32、3warmup和11次计时；实际单seed约46–60秒（Dev）。 optimization_evidence/训练证据说明.md:5；optimization_evidence/baseline_runs/seed_17/result.json#execution；optimization_evidence/reference_runs/seed_17/result.json#execution |
| B04 | 通过 | alpha/beta由协议冻结；Baseline直接计算，Reference改变合法代数路径，无Baseline异常参数或额外训练预算反证。 workspace/harbor_task/environment/public_assets/dev/protocol.json；workspace/harbor_task/environment/starter/method.py；workspace/reference/method.py |
| B05 | 通过 | 完整五组17/29/43/71/101，seed实际进入输入Generator；相同方法哈希跨seed一致。计时为随机评估，非独立训练。 workspace/harbor_task/tests/grader.py:95；workspace/harbor_task/tests/grader.py:219；optimization_evidence/comparison_summary.json#source_results |
| B06 | 通过 | 健康控制就是同一正确Starter；每case同次重新测量且数值与oracle比较，随机/常数输出不适合作本题精确算子锚点。 workspace/harbor_task/tests/grader.py:225；workspace/harbor_task/tests/baseline_method.py；optimization_evidence/baseline_runs/seed_17/result.json#records |
| B07 | 通过 | Reference以token Gram恒等式和结合律改变方法，未改变输入/数据/精度门；不把各子变化单独因果化。 workspace/reference/method.py:18；optimization_evidence/训练证据说明.md:7 |
| B08 | 通过 | 专家说明披露直接Baseline来源、角色、局限、公平预算与非训练属性；外部源对照仅作辅助，未独立核验外部来源。 expert_evidence/专家作业说明文档.md:9；optimization_evidence/训练证据说明.md |

五维事实：
- training_sufficiency：纯推理，无训练/微调/checkpoint；训练证据明确NOT_APPLICABLE。（optimization_evidence/baseline_runs/seed_17/result.json#training）
- method_representativeness：固定算子的正确朴素直接实现，存在清晰Starter映射；未发现包内同预算常用锚点证明其无代表性。（workspace/harbor_task/environment/starter/method.py；expert_evidence/专家作业说明文档.md:11）
- budget_comparability：同协议/数据/资源和计时次数；训练预算不适用。（optimization_evidence/训练证据说明.md:5）
- implementation_correctness：源码与数学算子一致，十个正式Dev结果及日志均记录质量门通过。（workspace/harbor_task/environment/starter/method.py；optimization_evidence/comparison_summary.json#source_results）
- selection_integrity：全五seed、统一方法哈希，逐组原始日志可追溯，未见挑最差B/最好R。（optimization_evidence/comparison_summary.json#paired_results）

## 全部正式成对数值与来源

指标：relative_process_cpu_time，越低越好。seed是生成输入/随机计时评估seed，非训练seed。每个原值已对照自己的result.json#raw_value和run.log末行，并从11个样本中位数/健康控制比逐case复算。10个Dev和10个专家Hidden预校准运行status均ok、exit_code均0、质量门通过。

| split | seed | Baseline | Reference | 改善B−R | B/R秒数 |
|---|---:|---:|---:|---:|---:|
| Dev | 17 | 1.140405400356479 | 0.5116721784753299 | 0.628733221881 | 54.561/45.960 |
| Dev | 29 | 1.216448073683598 | 0.4600538051925116 | 0.756394268491 | 52.674/48.446 |
| Dev | 43 | 0.9824788978125498 | 0.5293415824776744 | 0.453137315335 | 54.244/51.869 |
| Dev | 71 | 1.086612204224258 | 0.4969746581497158 | 0.589637546075 | 59.308/54.555 |
| Dev | 101 | 0.9123103439763108 | 0.4965188206302016 | 0.415791523346 | 60.226/45.622 |
| 专家Hidden预校准 | 17 | 1.212927949145425 | 0.5258901879801482 | 0.687037761165 | 52.028/48.832 |
| 专家Hidden预校准 | 29 | 1.191040684021493 | 0.450082740436716 | 0.740957943585 | 53.517/49.018 |
| 专家Hidden预校准 | 43 | 1.032956388041047 | 0.5296006741709839 | 0.50335571387 | 59.462/100.206 |
| 专家Hidden预校准 | 71 | 0.9720198731539528 | 0.4340622480917604 | 0.537957625062 | 57.702/53.935 |
| 专家Hidden预校准 | 101 | 0.9153536438216223 | 0.3888142755423572 | 0.526539368279 | 53.718/40.969 |

逐seed真实来源（每行均含两角色result与其同目录run.log）：
- seed 17：optimization_evidence/baseline_runs/seed_17/result.json#raw_value；optimization_evidence/reference_runs/seed_17/result.json#raw_value；日志：optimization_evidence/baseline_runs/seed_17/run.log；optimization_evidence/reference_runs/seed_17/run.log
- seed 29：optimization_evidence/baseline_runs/seed_29/result.json#raw_value；optimization_evidence/reference_runs/seed_29/result.json#raw_value；日志：optimization_evidence/baseline_runs/seed_29/run.log；optimization_evidence/reference_runs/seed_29/run.log
- seed 43：optimization_evidence/baseline_runs/seed_43/result.json#raw_value；optimization_evidence/reference_runs/seed_43/result.json#raw_value；日志：optimization_evidence/baseline_runs/seed_43/run.log；optimization_evidence/reference_runs/seed_43/run.log
- seed 71：optimization_evidence/baseline_runs/seed_71/result.json#raw_value；optimization_evidence/reference_runs/seed_71/result.json#raw_value；日志：optimization_evidence/baseline_runs/seed_71/run.log；optimization_evidence/reference_runs/seed_71/run.log
- seed 101：optimization_evidence/baseline_runs/seed_101/result.json#raw_value；optimization_evidence/reference_runs/seed_101/result.json#raw_value；日志：optimization_evidence/baseline_runs/seed_101/run.log；optimization_evidence/reference_runs/seed_101/run.log
- seed 17：expert_evidence/hidden_calibration/baseline_runs/seed_17/result.json#raw_value；expert_evidence/hidden_calibration/reference_runs/seed_17/result.json#raw_value；日志：expert_evidence/hidden_calibration/baseline_runs/seed_17/run.log；expert_evidence/hidden_calibration/reference_runs/seed_17/run.log
- seed 29：expert_evidence/hidden_calibration/baseline_runs/seed_29/result.json#raw_value；expert_evidence/hidden_calibration/reference_runs/seed_29/result.json#raw_value；日志：expert_evidence/hidden_calibration/baseline_runs/seed_29/run.log；expert_evidence/hidden_calibration/reference_runs/seed_29/run.log
- seed 43：expert_evidence/hidden_calibration/baseline_runs/seed_43/result.json#raw_value；expert_evidence/hidden_calibration/reference_runs/seed_43/result.json#raw_value；日志：expert_evidence/hidden_calibration/baseline_runs/seed_43/run.log；expert_evidence/hidden_calibration/reference_runs/seed_43/run.log
- seed 71：expert_evidence/hidden_calibration/baseline_runs/seed_71/result.json#raw_value；expert_evidence/hidden_calibration/reference_runs/seed_71/result.json#raw_value；日志：expert_evidence/hidden_calibration/baseline_runs/seed_71/run.log；expert_evidence/hidden_calibration/reference_runs/seed_71/run.log
- seed 101：expert_evidence/hidden_calibration/baseline_runs/seed_101/result.json#raw_value；expert_evidence/hidden_calibration/reference_runs/seed_101/result.json#raw_value；日志：expert_evidence/hidden_calibration/baseline_runs/seed_101/run.log；expert_evidence/hidden_calibration/reference_runs/seed_101/run.log

| 统计量 | Dev | 专家Hidden预校准 |
|---|---:|---:|
| baseline_mean | 1.06765098401064 | 1.06485970763671 |
| baseline_sample_std | 0.121632957682699 | 0.132131527965878 |
| reference_mean | 0.498912208985087 | 0.465690025244393 |
| reference_sample_std | 0.0255353752121792 | 0.0609557527198294 |
| positive_improvement | 0.568738775025553 | 0.599169682392315 |
| three_sigma | 0.364898873048098 | 0.396394583897633 |
| improvement_sigma_multiple | 4.67586076883213 | 4.53464583068355 |
| normalized_reference | 0.532701026405728 | 0.562674761844525 |
| remaining_headroom_sigma_multiple | 4.1017847340898 | 3.52444289726714 |

标准差均为n−1样本标准差。U=0，归一化(B−R)/(B−0)。两split均达到3σ且未达到5σ；Reference在[0.15,0.8]，到U的剩余空间均≥3σ。Hidden表只是作者已有预校准，不是平台Hidden或本次复测。旧绝对CPU时间v1材料未混入当前ratio统计。

Baseline源码SHA256：dfe9194dc9754a70324890ecd603bf860e7c0266f4ba8bb7dca973263def36e5；Reference源码SHA256：a44453728d853b705b8cac46f17925b4c6029d44d99629204efd7d26529bad8e。已按ZIP字节核对，与全部正式记录一致。

## 三项内容门

| 内容门 | 状态 | 结论及证据 |
|---|---|---|
| G01 | pass | 接口允许任意纯CPU张量方法、helper和代数重排；实际guard允许函数，Reference用Gram范数和投影重排，非纯调参。 workspace/harbor_task/instruction.md:16；workspace/harbor_task/tests/grader.py:73；workspace/reference/method.py:18 |
| G02 | pass | 在已提供材料范围内合理；正确朴素Starter为正式Baseline及B锚点，非训练，预算和全seed公平。 workspace/harbor_task/environment/starter/method.py；optimization_evidence/训练证据说明.md；optimization_evidence/comparison_summary.json |
| G03 | pass | 全部五组成对原始值可追溯；Dev归一化0.5327010264，改善4.6758607688σ_B；质量门全通过，达到3σ但未达5σ。 optimization_evidence/comparison_summary.json#source_results；workspace/harbor_task/environment/public_assets/dev/protocol.json:19 |

## 双轨、有效时长与选优

| 轨迹 | 打包轮数/声明闭环 | 声明下界h | 扣除配置失败后下界h | 宽松估算h |
|---|---:|---:|---:|---:|
| Codex + GPT-6.1 Sol (xhigh) | 64/59 | 10.102225980594 | 10.022689525178 | 14.650887705151 |
| Codex + Doubao Seed 2.1 Turbo (high) | 35/8 | 10.007259767539 | 9.927637316925 | 18.061537755900 |

GPT 71条、Seed 42条时长区间求和与原汇总一致，区间没有重叠。原始runtime和评分日志路径均存在。下界剔除了部分缺日志/失联模型阶段，但仍包含配置拒绝后Starter评分；Seed至少需再核实260.505659秒合格时间才能证明默认10h。修正下界不是实际总时长上界，故此处是核心待核，不武断断言实际研究一定不足。未自动适用7h例外，也不要求新增独立计时器。

两轨分别使用gpt-6.1-sol/xhigh和doubao-seed-2-1-turbo-260628/high；已有Trial记录为Codex memory-codex、Docker、separate且无异常。Seed凭据未在本次读取/调用；本次没有任何模型请求。有效时间、模型runtime和稳定性分开：已有12.002825h是Starter环境存活工作负载。

GPT探索投影/Gram/Cholesky面板和共享工作区；Seed探索投影次序、Gram范数、3D bmm。两者继续方向在各自runtime最后session中。交付唯一最佳为GPT方案，公共全五seed干净选优均值0.4279666746653184，Seed为0.4803020135338603。最终方法再次Dev原值0.4187087604867304、分数0.607822437521813；专家Hidden原值0.40314299028368433、分数0.6214121096023081。方法哈希837fb14fcfc1ff60a6ef2be6e1b0d73a5bc8b9137d672e20721f446d50b2adf0与选优/复验相符。

标准轨迹有64/35轮，但各轮含额外字段，Seed多轮为文本残片。所有有数值的轨迹score可回溯对应grader；Seed第19/30/31轮实际grader出分而交付score为null。GPT第14/52等轮摘要中的内轮观察与最终控制器复测/保留结论混在一起，必须明确改写。原始日志不得删除。

## 格式与部署隔离

三顶层目录齐全。Agent context=workspace/harbor_task/environment，Verifier context=workspace/harbor_task/tests；所有COPY来源在context内且存在。Agent未COPY私有tests/Reference/专家证据。各sandbox单容器，无四种Compose直属文件。artifacts移交/workspace/solution，入口/tests/test.sh实际使用/tests/grader.py，依赖在各自镜像安装。

私有protocol包含6种case，通过固定生成器准备输入；候选子进程UID1002，私有protocol权限600。输出oracle由可信父进程计算，reward清理并原子写入。上述文件部署路径成立，但进程内缓存/torch.equal可变状态缺陷使受限执行不能通过；独立容器并未自动解决该缺陷。

- expert_evidence/best_method/（misplaced）：此目录保留唯一方法源码、solve.sh及必要配置；将rerun_dev/rerun_hidden/verification/build log移到专家复验子目录，更新引用。不因多交本身判退。
- expert_evidence/absolute_cpu_time_protocol_v1/（extra_allowed）：明确为旧绝对CPU时间协议辅助证据，勿并入当前ratio正式统计。
- expert_evidence/raw_evidence/（extra_allowed）：原始会话/回执保留以支持追溯；交付轨迹按八字段整理，不以原始文本代替摘要。
- workspace/harbor_task/environment/instruction.md（merge_candidate）：环境副本因Docker COPY有用途；修订时自动同步根题面和task.toml，避免漂移。
- expert_evidence/专家作业说明文档.md:41（merge_candidate）：该段仍称等待GPT收尾/29轮Seed；用最终64/35轮、净时长和选优事实更新，保留历史仅放时间线。

## Harbor 六项

| 项 | 状态 | 结论及证据 |
|---|---|---|
| H01 | pass | 任务根及入口齐全；非空私有protocol定义6种形状，经generate(seed*1009+i)生成Hidden输入，只在Verifier准备。 workspace/harbor_task/task.toml；workspace/harbor_task/tests/protocol.json；workspace/harbor_task/tests/grader.py:95；workspace/harbor_task/tests/grader.py:219 |
| H02 | fail | 显式separate；目标包述Harbor0.23.0，本地旧schema记录不是当前内场验收；storage_mb/禁网/白名单冲突已确认。 workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:24；workspace/harbor_task/task.toml:25；workspace/harbor_task/task.toml:37；workspace/harbor_task/task.toml:38；workspace/harbor_task/instruction.md:20；workspace/harbor_task/environment/task.toml；expert_evidence/prerequisite_evidence/harbor-schema-validation.json#harbor_version |
| H03 | fail | 双context COPY与移交路径静态对齐、无Compose；候选以UID1002子进程运行，但进程内持久缓存/完整性检查存在可定位绕过。 workspace/harbor_task/tests/grader.py:55；workspace/harbor_task/tests/grader.py:71；workspace/harbor_task/tests/worker.py:24；workspace/harbor_task/tests/worker.py:33；workspace/harbor_task/environment/public_eval/grader.py:55；workspace/harbor_task/environment/Dockerfile；workspace/harbor_task/tests/Dockerfile；workspace/harbor_task/task.toml:2 |
| H04 | pass | test.sh写有限标量reward，清理后原子替换；正常负分和>1未被裁剪。已交NOP负分对应有效计时波动。 workspace/harbor_task/tests/test.sh；expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/verifier/reward.txt |
| H05 | pass | 已有Job选定本题与独立模型轨迹，Docker provider、separate、verifier未禁用；resume仅来自各自历史。自定义memory-codex为已有运行配置，不冒充本次可直接部署。 expert_evidence/raw_evidence/campaigns/asd-research-1791486012/config.json；expert_evidence/raw_evidence/campaigns/asd-research-1791486043/config.json；expert_evidence/raw_evidence/campaigns/asd-research-1791486012/harbor_task__VqWxhgD/result.json#config |
| H06 | nonblocking_pending | 同一NOP Trial的config/result/reward/日志齐全，separate且无异常，reward=-0.04457318804105793；当前四项版本绑定仍未核验。 expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/config.json；expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/result.json；expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/verifier/reward.txt；expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/trial.log；expert_evidence/harbor_nop_v023/portable_export.json |

已有NOP的config/result/reward/trial.log属于同一Trial；separate=true、无异常，reward=-0.04457318804105793。负分是未裁剪计时比正常可能结果，不据此判失败。当前版本对应关系仍可后补，本次未运行Harbor。旧schema记录版本0.1.0，当前task版本0.2.3；不把旧schema自验当当前官方工具通过。

## 全部21项实现检查

| 项 | 状态 | 原因与证据 |
|---|---|---|
| QA01 | pass | 题面八部分齐全，无论文标题、arXiv/repo或Reference方法泄露。 workspace/harbor_task/instruction.md:1 |
| QA02 | pass | 实现(anchor-latency)/(anchor-U)，未裁剪；B映射0、U映射1，lower-is-better单调。 workspace/harbor_task/tests/grader.py:239；workspace/harbor_task/tests/protocol.json:19；workspace/harbor_task/environment/public_assets/dev/protocol.json:21 |
| QA03 | pass | Reference源码/入口和五seed result/run.log完整；非训练无需checkpoint；归一化0.5327010264。 workspace/reference/method.py；workspace/reference/solve.sh；optimization_evidence/comparison_summary.json#source_results |
| QA04 | pass | 计时随机；从所有正式值按n−1复算σ_B=0.1216329577，改善0.5687387750≥3σ=0.3648988730。 optimization_evidence/comparison_summary.json#paired_results |
| QA05 | fail | 存在数值/超时/源代码限制，但元组模块属性写入可绕过持久缓存禁令；网络/磁盘约束也与当前V4冲突。 workspace/harbor_task/tests/grader.py:55；workspace/harbor_task/tests/grader.py:71；workspace/harbor_task/tests/worker.py:24；workspace/harbor_task/tests/worker.py:33；workspace/harbor_task/environment/public_eval/grader.py:55；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:24；workspace/harbor_task/task.toml:25；workspace/harbor_task/task.toml:37；workspace/harbor_task/task.toml:38；workspace/harbor_task/instruction.md:20；workspace/harbor_task/environment/task.toml |
| QA06 | pass | 单命令test.sh调用grader，成功/数值违规有标量分数并写reward；格式/资源/基础设施失败明确分类且不伪装0。 workspace/harbor_task/tests/test.sh:3；workspace/harbor_task/tests/grader.py:254 |
| QA07 | pass | 仅审题面：未披露私有case、标签或可恢复私有答案。 workspace/harbor_task/instruction.md |
| QA08 | pass | 仅审题面：未披露Reference代码、Gram解法或私有参考成绩。 workspace/harbor_task/instruction.md |
| QA09 | pass | 入口删除旧reward/result；grader和reward用临时文件原子替换。 workspace/harbor_task/tests/test.sh:3；workspace/harbor_task/tests/test.sh:12；workspace/harbor_task/tests/grader.py:180；workspace/harbor_task/tests/grader.py:195 |
| QA10 | pass | 存在文件冻结实现：Agent公共文件root所有且只读，Verifier协议600；solve.sh字节比较，候选复制只读。此项不等同缓存/进程内状态隔离通过。 workspace/harbor_task/environment/Dockerfile:13；workspace/harbor_task/tests/Dockerfile:6；workspace/harbor_task/tests/grader.py:198 |
| QA11 | pass | 目标、输入输出、环境、指标、可改/禁止边界、提交、迭代和完成条件均有实质说明。 workspace/harbor_task/instruction.md:3；workspace/harbor_task/instruction.md:28 |
| QA12 | pass | ZIP清点无.git目录/元数据，不需执行清理脚本。 @inventory |
| QA13 | pass | 正式Dev十个运行全部有效、同五seed，每seed Reference均优于Baseline；原始计时样本复算与日志一致。 optimization_evidence/comparison_summary.json#source_results |
| QA14 | pass | 公开接口还可探索分块/布局/共享投影与范数分解；Reference未饱和，已有最佳方法进一步降低计时比。 workspace/reference/method.py；expert_evidence/best_method/method.py；expert_evidence/best_method/rerun_dev/result.json#raw_value |
| QA15 | pass | CPU题0GPU、1CPU/2GiB；正式Dev单seed45.62–60.23秒，最佳五seed干净评测56.83秒；每case120秒和Verifier1800秒上限，未见单轮>2h证据。 workspace/harbor_task/task.toml:21；workspace/harbor_task/tests/grader.py:147；expert_evidence/best_method/rerun_dev/result.json#clean_execution；optimization_evidence/comparison_summary.json#source_results |
| QA16 | manual | 下界求和可复现，但Seed含配置拒绝后286.641秒Starter评分；扣后下界9.927637h，尚不能确认10h；不等于已证明真实总时长不足。 expert_evidence/auditable_time_lower_bound.json:1401；expert_evidence/auditable_time_lower_bound.json:2234；expert_evidence/raw_evidence/campaigns/asd-research-1791370972/harbor_task__d9XHobK/agent/round_00001.events.jsonl:1；expert_evidence/raw_evidence/campaigns/asd-research-1791370972/harbor_task__d9XHobK/agent/runtime.json:20；expert_evidence/run_summary.json:27 |
| QA17 | fail | 目录/context/artifacts/reward链条存在，但当前配置不符内场口径，候选缓存限制可绕过；NOP版本补证本身不阻断。 workspace/harbor_task/task.toml；workspace/harbor_task/tests/grader.py:55；workspace/harbor_task/tests/worker.py:24 |
| QA18 | fail | 两轨存在且大多数原始分数可回溯；非恰八字段，Seed摘要残片、已出分轮记null及GPT摘要/最终保留记录矛盾。 expert_evidence/trajectory_codex.json:567；expert_evidence/trajectory_codex.json:149；expert_evidence/trajectory_seed.json:6；expert_evidence/trajectory_seed.json:205；expert_evidence/trajectory_seed.json:336；expert_evidence/trajectory_seed.json:348；expert_evidence/raw_evidence/campaigns/asd-research-1791486012/harbor_task__VqWxhgD/agent/round_00052.grader.log |
| QA19 | pass | 题面明确五个seed与计时随机性，声明改善须超过Baseline样本标准差3倍。 workspace/harbor_task/instruction.md:12 |
| QA20 | pass | 题面有可改范围、网络规则和本地工具策略；网络规则的当前适用性缺陷已计入约束与配置项。 workspace/harbor_task/instruction.md:16；workspace/harbor_task/instruction.md:20 |
| QA21 | nonblocking_pending | run_summary缺task_binding.hash_profile/task_content_sha256；保留既有源码/协议哈希，未猜测官方四项算法。 expert_evidence/run_summary.json |

## 可直接复制给专家的逐问题修改与验收

P1 跨调用缓存可绕过源码 guard，破坏计时评分边界（confirmed_failure）
事实：赋值检查仅判断最外层 target 是否为 ast.Attribute；函数内元组解包目标中的模块属性未递归检查。hasattr 未被禁止，torch 的普通新属性名也允许读取。因此可用元组目标把输出持久化在 torch 属性中，warmup 后重复返回缓存。worker 全部计时调用使用同一输入，所谓 fresh call 只是克隆同值输入，不能识别这种缓存。相同赋值缺口还能改写 worker 所用的 torch.equal。
定位：workspace/harbor_task/tests/grader.py:55；workspace/harbor_task/tests/grader.py:71；workspace/harbor_task/tests/worker.py:24；workspace/harbor_task/tests/worker.py:33；workspace/harbor_task/environment/public_eval/grader.py:55
影响：题面明确禁止持久状态和结果缓存，而实际 guard/worker 不保证这些边界；缓存值仍能通过父进程 oracle 的数值检查。此为静态可定位的执行边界缺陷，不表示现有 Reference 或已交最佳候选用了该方式。
最小修改：递归检查所有赋值、解包、增量赋值及删除目标，限制模块/函数属性变更及间接模块状态；对可调用对象采用明确安全计算面。把输入完整性/计时可信状态与候选隔离，并用变化输入验证反缓存，避免只重传相同克隆输入。同步公开与私有 worker/guard。
验收材料：提交修订源码及在授权隔离环境中针对元组属性写入、模块缓存、torch.equal 修改的拒绝记录；合法纯张量实现仍可运行，变换输入和数值质量门有效；不接受只新增文字禁令。

P2 配置与题面仍依赖当前内场不支持的网络和磁盘约束（confirmed_failure）
事实：Agent 设置 allowlist/allowed_hosts，environment 和 verifier.environment 声明 storage_mb=4096、network_mode=no-network；题面将禁网/模型主机白名单列为保证。
定位：workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:24；workspace/harbor_task/task.toml:25；workspace/harbor_task/task.toml:37；workspace/harbor_task/task.toml:38；workspace/harbor_task/instruction.md:20；workspace/harbor_task/environment/task.toml
影响：当前 V4 明确本批不声明 storage_mb、禁网和网络白名单。这不是换卡耗时、NOP 或指纹放宽项；旧 Harbor 本地接受字段不代表当前平台能力。
最小修改：删除两份 task.toml 中不支持的存储与网络字段，按当前平台实际联网条件改写两份题面；候选纯计算/禁止外部通信由受限执行边界落实，不能依赖平台白名单承诺。
验收材料：完整新 ZIP 中题面、task.toml 及环境副本一致；以目标平台真实版本的格式/schema 检查结果确认字段支持，并提供候选边界验证材料。无需仅为此要求旧正式 CPU 分数换卡重跑。

P3 轨迹摘要、状态与实际评分/保留结果未对齐，且不是恰八字段（confirmed_failure）
事实：64 条 GPT 和 35 条 Seed 每轮含 packaged_source，13 条 Seed 失败轮另含 original_status。Seed 第1/14/19/30/31等轮 method_summary 是截断代码/XML或重复输出，非1–3句方法摘要。GPT 第52轮摘要称0.421919084且劣于最佳，最终 score=0.40776985324241416、retained_best=true，原 grader 支持后者。Seed 第19/30/31轮 score=null，但各自 grader 已出 raw=0.524809612381887、0.4892946021797527、0.5282904292793187。
定位：expert_evidence/trajectory_codex.json:567；expert_evidence/trajectory_codex.json:149；expert_evidence/trajectory_seed.json:6；expert_evidence/trajectory_seed.json:205；expert_evidence/trajectory_seed.json:336；expert_evidence/trajectory_seed.json:348；expert_evidence/raw_evidence/campaigns/asd-research-1791486012/harbor_task__VqWxhgD/agent/round_00052.grader.log
影响：无法从标准轨迹准确区分模型失败、实际方法评分和最终保留决策；原始日志存在不能使不一致的交付轨迹自动通过。
最小修改：将来源/原始状态索引移至顶层旁表，逐轮只保留规定八字段。保留原始事件不改写，依据实际生效源码、最终 grader 与保留行为重写简洁摘要；明确模型中断与评分结果不同，有真实已出分则记录该值并说明失败类型。同步过时的专家说明。
验收材料：两条严格八字段 JSON、逐轮外置来源映射；核对全部99轮数值、失败原因、保留标记，尤其上述轮次；不得把模型自报分数替代最终公共评分或删除失败记录。

P4 Seed 十小时有效时长尚不能由现有下界确认（core_evidence_gap）
事实：逐会话求和确为 Seed 36026.1351631398秒，但其中286.64082220802084秒是 CLI 配置立即拒绝后、工具数0且方法未改的 Starter 评分；这不是有效方法研究。扣除此项，已可审计下界为35739.49434093178秒（9.927637316925494小时），比10小时少260.50565906822秒。GPT 同类扣除后下界为36081.68229063977秒（10.022689525177714小时）。
定位：expert_evidence/auditable_time_lower_bound.json:1401；expert_evidence/auditable_time_lower_bound.json:2234；expert_evidence/raw_evidence/campaigns/asd-research-1791370972/harbor_task__d9XHobK/agent/round_00001.events.jsonl:1；expert_evidence/raw_evidence/campaigns/asd-research-1791370972/harbor_task__d9XHobK/agent/runtime.json:20；expert_evidence/run_summary.json:27
影响：现有摘要不能证明两条均≥10h。修正下界不足不证明实际总时长一定不足；另有被保守扣除的未知时间，不能直接当有效时间加回。该问题不属于五类放宽。
最小修改：从下界中扣除纯配置/环境故障与无方法研究的自动评分，并核对其余API失联/阻塞区间。优先用已有原始记录补足可核验有效研究时间；不足再在另行授权后补真实迭代。若采用非训练短迭代7h例外，须独立提供全部适用依据，不能把包内旧用户表述当成本会话授权。
验收材料：每轨起止、扣除区间与去重算式，Seed 至少还需260.505659秒已有有效时间的证据或合法补足；或完整7h例外材料（短迭代、≥3闭环、持续演进、下一方向、最佳方法干净复验）。未知区间保持未知。

## 五类放宽事项（非阻断）

- 指纹/哈希：nonblocking_pending。task_binding两个字段缺失；历史协议哈希与校准后协议可能不同，未用哈希差异否定正式数据。 后补建议：真实脚本补四项指纹与版本映射，不全量重跑。 依据：expert_evidence/run_summary.json；optimization_evidence/baseline_runs/seed_17/result.json#protocol_sha256
- 换卡后B/R影响：not_applicable。CPU算子；保留linux/aarch64历史设备，未将其标为H20/L20实测。 后补建议：若平台更换CPU环境，可后补计时比影响说明。 依据：expert_evidence/专家作业说明文档.md:27
- 换卡耗时：not_applicable。CPU0GPU，已有计时日志；本次未验证目标平台耗时。 后补建议：目标平台时间验证另列，不单独阻断。 依据：expert_evidence/best_method/rerun_dev/result.json#clean_execution
- NOP记录/版本：nonblocking_pending。既有NOP链路完整，当前版本绑定未核验。 后补建议：复用并补版本关联，不自动补跑。 依据：expert_evidence/harbor_nop_v023/portable_export.json
- 换卡后重算分数：not_applicable。当前审查使用已有真实CPU正式全seed数据；不要求因换卡重做。 后补建议：仅设备变化引起的重算可延期。 依据：optimization_evidence/comparison_summary.json

工具缺口：真实官方0.3.10发布包及独立格式/GPU/指纹脚本未取得，完整官方工具验收待补；未调用旧0.3.2收集器或伪改版本。取得后记录真实版本/SHA和原始报告。
运行边界：仅静态读取ZIP的文本/JSON/源码AST与元数据及自行复算统计；未导入/执行提交代码，未运行Docker、Verifier、训练、模型、安装脚本或权重反序列化。已有运行记录不等于本次独立复跑、平台正式Hidden或目标环境隔离认证。
可见隐私：已读workspace源码/题面未见个人home、邮箱或私有端点命中；未实施完整二进制严格扫描，该工作由主控另行完成。报告未复制秘密或本机个人路径。
风险仅列三项：可绕过缓存限制；旧网络/磁盘配置与当前内场不兼容；轨迹及有效时间摘要与原始记录未完全对齐。
