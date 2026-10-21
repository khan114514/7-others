# attention_v023 独立完整ZIP静态复核

核心结论：**FAIL / NOT READY**。方法优化面、Baseline合理性和Reference提升三门通过；当前内场配置与正式轨迹规范不通过，Seed有效时长证明另待重算。五类放宽项单列，不因指纹或NOP缺口判退。

schema=`local-v4-content-review` v4；policy=`research-quality-v3`，当前V4及2026-10-21五类非阻断口径。此为本地内容复核，非官方0.3.10、平台验收或本次独立动态复现。

输入：`AutoResearch_attention_v023_final_20261009.zip`；SHA256：`bd75ad937f2e1ba807bd8db95f569e7cbac350e1258dac27465129a2dff28e42`；745成员，展开总字节22,346,573。session_id=`attention_a`；configured_model=`gpt-6.1-sol`，configured_reasoning_effort=`high`（主控派发）；runtime_reported_model=`not_exposed`，runtime_version=`not_exposed`。新会话只读同一完整ZIP；未读包外历史QA、同伴报告或包内旧QA。

## 优化面及实际调用边界

固定精确多头检索算子的CPU推理延迟优化；候选纯函数实现与helper开放，实际AST guard允许方法更新。

题面允许仅改method.py的attention与纯helpers。guard检查imports、AST、持久状态、反射/FS等，而保留任意CPU张量代数重排；grader复制只读源码，worker在kernel用户下执行候选，仅向其提供输入。固定目标、alpha/beta、allclose数值门、单线程、数据种子和评分规则。见 `workspace/harbor_task/instruction.md:10–20`、`workspace/harbor_task/tests/grader.py:28–92,119–177,198–238`。不存在只接受学习率/权重字面量的纯HPO限制。

## Baseline来源、实现、实际训练与公平预算

未改Starter直接计算q/k与两组softmax，在完整通道空间检索/混合/范数恢复后做value投影；健康控制同源码，B取正式五seed均值。

角色映射：Starter=正式Baseline=健康直接实现；主B=正式五seed均值1.0676509840106392，未另设弱零点。其SHA与tests/baseline_method.py相同。此题不训练，所有训练/checkpoint/reload项NOT_APPLICABLE；五个seed是输入生成与计时评测seed，不是训练重复。来源/限制见 `expert_evidence/专家作业说明文档.md:11,27,33`。

共同条件：同6个Dev case、同5seed、float32 CPU、单线程、3次warmup/11次测量、1CPU/2GiB。正式单seed实际端到端45.62–60.23秒。两侧的实现方法差异合法；没有单边缩减训练/数据或挑差Baseline的直接证据。

| Baseline项 | 状态 | 五维事实及证据 |
|---|---|---|
| B01 | pass | Starter、正式Baseline和评分B均绑定正确直接实现，来自固定算子语义；经典直接法是可辩护的naive对照。 `workspace/harbor_task/environment/starter/method.py:7`；`expert_evidence/专家作业说明文档.md:11` |
| B02 | pass | 无训练；q/k、双softmax、完整通道范数与value投影实现完整，正式10次质量门通过。没有训练不足或故障注入反证。 `workspace/harbor_task/environment/starter/method.py:8`；`workspace/harbor_task/tests/grader.py:105`；`optimization_evidence/baseline_runs/seed_17/result.json#training` |
| B03 | pass | 同一5 seed、6 case、float32、1线程、3 warmup/11计时、1 CPU/2GiB；R改变计算方法，未增加数据或预算。 `optimization_evidence/训练证据说明.md:5`；`workspace/harbor_task/environment/public_assets/dev/protocol.json:6`；`optimization_evidence/baseline_runs/seed_17/result.json#execution`；`optimization_evidence/reference_runs/seed_17/result.json#execution` |
| B04 | pass | alpha/beta由评分器固定；Baseline源码与Starter逐字节哈希一致，双方同计时规则，无异常单边超参数的直接证据。 `workspace/harbor_task/tests/protocol.json:22`；`workspace/harbor_task/environment/starter/method.py`；`optimization_evidence/baseline_runs/seed_17/result.json#method` |
| B05 | pass | 全部正式seed成对完整；种子传入生成器，随机性来自计时，5组是输入/评测seed而非训练seed；未冒充独立训练。 `workspace/harbor_task/tests/grader.py:95`；`workspace/harbor_task/tests/grader.py:219`；`optimization_evidence/comparison_summary.json:9`；`optimization_evidence/baseline_runs/seed_17/result.json#protocol.data_generation` |
| B06 | pass | 健康Starter同时充当每次测量控制，评分Baseline本身是独立测量同一健康实现；无随机/常数弱锚点。无额外sanity结果并非失败。 `workspace/harbor_task/tests/grader.py:225`；`workspace/harbor_task/tests/grader.py:234`；`workspace/harbor_task/tests/baseline_method.py:7` |
| B07 | pass | Reference以Gram二次型恢复范数并提前value投影、按形状回退；其余数据、评分与资源一致，收益归因于整体合法计算重排。 `workspace/reference/method.py:18`；`workspace/reference/method.py:21`；`optimization_evidence/训练证据说明.md:7` |
| B08 | pass | 专家披露来源、角色、限制及非SOTA定位；说明B来自未改Starter的全seed均值。 `expert_evidence/专家作业说明文档.md:11`；`expert_evidence/专家作业说明文档.md:33` |

专项结论：**在已提供材料范围内合理**。五维事实：training_sufficiency=无训练；method_representativeness=正确朴素直接法；budget_comparability=同协议/资源/数据；implementation_correctness=健康控制与质量门支持；selection_integrity=全正式seed及历史协议明确分离。未发现直接红旗；静态一致不等于独立复现。

## Reference实际方法、全部正式成对原值及复算

先用密集线性投影，L相对C较小时构造XX^T，用二次型恢复检索前范数并在较窄value空间检索；其他形状直接回退，alpha=1采用等价路径。

正式Dev v2主分数单位是无量纲候选/control进程CPU时间比。U=0为预声明数学非负floor；B/R均来自全五组均值。逐一读取全部正式result和run.log，重算每case11样本中位数、candidate/control、6case均值，与result/log一致；日志虽单行JSON，包含完整samples/stdout，与评分器print路径一致，不能因单行而视为人工摘要。

### 正式 Dev v2

| seed | Baseline | Reference | B−R | 原始来源 |
|---|---:|---:|---:|---|
| 17 | 1.14040540035648 | 0.51167217847533 | 0.62873322188115 | `optimization_evidence/baseline_runs/seed_17/result.json`；`optimization_evidence/reference_runs/seed_17/result.json`；同目录`run.log:1` |
| 29 | 1.2164480736836 | 0.460053805192512 | 0.756394268491086 | `optimization_evidence/baseline_runs/seed_29/result.json`；`optimization_evidence/reference_runs/seed_29/result.json`；同目录`run.log:1` |
| 43 | 0.98247889781255 | 0.529341582477674 | 0.453137315334875 | `optimization_evidence/baseline_runs/seed_43/result.json`；`optimization_evidence/reference_runs/seed_43/result.json`；同目录`run.log:1` |
| 71 | 1.08661220422426 | 0.496974658149716 | 0.589637546074542 | `optimization_evidence/baseline_runs/seed_71/result.json`；`optimization_evidence/reference_runs/seed_71/result.json`；同目录`run.log:1` |
| 101 | 0.912310343976311 | 0.496518820630202 | 0.415791523346109 | `optimization_evidence/baseline_runs/seed_101/result.json`；`optimization_evidence/reference_runs/seed_101/result.json`；同目录`run.log:1` |

n=5；B均值=1.06765098401064，σB(n−1)=0.121632957682699；R均值=0.498912208985087，σR=0.0255353752121792；Δ=0.568738775025553；3σB=0.364898873048098；Δ/σB=4.675860769；归一化Δ/B=0.532701026406。全部质量门ok/true。
剩余至U空间R−0=0.498912208985=4.1017847σB；主门槛通过，属于3–5σ可接受证据，建议后续复核。

### 专家私有预校准 v2

| seed | Baseline | Reference | B−R | 原始来源 |
|---|---:|---:|---:|---|
| 17 | 1.21292794914543 | 0.525890187980148 | 0.687037761165277 | `expert_evidence/hidden_calibration/baseline_runs/seed_17/result.json`；`expert_evidence/hidden_calibration/reference_runs/seed_17/result.json`；同目录`run.log:1` |
| 29 | 1.19104068402149 | 0.450082740436716 | 0.740957943584777 | `expert_evidence/hidden_calibration/baseline_runs/seed_29/result.json`；`expert_evidence/hidden_calibration/reference_runs/seed_29/result.json`；同目录`run.log:1` |
| 43 | 1.03295638804105 | 0.529600674170984 | 0.503355713870063 | `expert_evidence/hidden_calibration/baseline_runs/seed_43/result.json`；`expert_evidence/hidden_calibration/reference_runs/seed_43/result.json`；同目录`run.log:1` |
| 71 | 0.972019873153953 | 0.43406224809176 | 0.537957625062192 | `expert_evidence/hidden_calibration/baseline_runs/seed_71/result.json`；`expert_evidence/hidden_calibration/reference_runs/seed_71/result.json`；同目录`run.log:1` |
| 101 | 0.915353643821622 | 0.388814275542357 | 0.526539368279265 | `expert_evidence/hidden_calibration/baseline_runs/seed_101/result.json`；`expert_evidence/hidden_calibration/reference_runs/seed_101/result.json`；同目录`run.log:1` |

n=5；B均值=1.06485970763671，σB(n−1)=0.132131527965878；R均值=0.465690025244393，σR=0.0609557527198294；Δ=0.599169682392315；3σB=0.396394583897633；Δ/σB=4.534645831；归一化Δ/B=0.562674761845。全部质量门ok/true。
专家私有预校准使用独立私有case，满足当前统计门槛；这是包内专家记录，不称官方Hidden验收。

### 历史绝对时间 Dev v1

| seed | Baseline | Reference | B−R | 原始来源 |
|---|---:|---:|---:|---|
| 17 | 108.970902333333 | 50.3155553333333 | 58.655347 | `expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/baseline_runs/seed_17/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/reference_runs/seed_17/result.json`；同目录`run.log:1` |
| 29 | 121.7637145 | 53.8823123333333 | 67.8814021666667 | `expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/baseline_runs/seed_29/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/reference_runs/seed_29/result.json`；同目录`run.log:1` |
| 43 | 106.0392705 | 51.0542638333333 | 54.9850066666667 | `expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/baseline_runs/seed_43/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/reference_runs/seed_43/result.json`；同目录`run.log:1` |
| 71 | 105.811061666667 | 48.1754441666667 | 57.6356175 | `expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/baseline_runs/seed_71/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/reference_runs/seed_71/result.json`；同目录`run.log:1` |
| 101 | 113.957770333333 | 53.5256038333333 | 60.4321665 | `expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/baseline_runs/seed_101/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/optimization_evidence/reference_runs/seed_101/result.json`；同目录`run.log:1` |

n=5；B均值=111.308543866667，σB(n−1)=6.70412399449301；R均值=51.3906359，σR=2.36500220924544；Δ=59.9179079666667；3σB=20.112371983479；Δ/σB=8.937470133；归一化Δ/B=0.538304660947。全部质量门ok/true。
历史辅助协议以毫秒为单位，单独保存且不混入当前v2主均值；此处复算仅用于核对全部已交成对材料。

### 历史绝对时间私有 v1

| seed | Baseline | Reference | B−R | 原始来源 |
|---|---:|---:|---:|---|
| 17 | 117.101791333333 | 45.9773263333333 | 71.124465 | `expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/baseline_runs/seed_17/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/reference_runs/seed_17/result.json`；同目录`run.log:1` |
| 29 | 132.088436833333 | 41.1700203333333 | 90.9184165 | `expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/baseline_runs/seed_29/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/reference_runs/seed_29/result.json`；同目录`run.log:1` |
| 43 | 112.774096833333 | 49.9267361666667 | 62.8473606666667 | `expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/baseline_runs/seed_43/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/reference_runs/seed_43/result.json`；同目录`run.log:1` |
| 71 | 94.2110758333333 | 40.842986 | 53.3680898333333 | `expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/baseline_runs/seed_71/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/reference_runs/seed_71/result.json`；同目录`run.log:1` |
| 101 | 107.283881333333 | 39.6376805 | 67.6462008333333 | `expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/baseline_runs/seed_101/result.json`；`expert_evidence/absolute_cpu_time_protocol_v1/hidden_calibration/reference_runs/seed_101/result.json`；同目录`run.log:1` |

n=5；B均值=112.691856433333，σB(n−1)=13.8425708124146；R均值=43.5109498666667，σR=4.32566539919653；Δ=69.1809065666667；3σB=41.5277124372438；Δ/σB=4.997692084；归一化Δ/B=0.613894461909。全部质量门ok/true。
历史辅助协议以毫秒为单位，单独保存且不混入当前v2主均值；此处复算仅用于核对全部已交成对材料。

## 三项内容门

| 项目 | 状态 | 结论与证据 |
|---|---|---|
| G01 | pass | 接口允许纯函数helpers、CPU PyTorch算法重排和形状分支；实际guard限制危险能力而非只接受数值。Reference包含Gram范数/提前投影方法更新。 `workspace/harbor_task/instruction.md:16`；`workspace/harbor_task/tests/grader.py:28`；`workspace/reference/method.py:18` |
| G02 | pass | 在已提供材料范围内Baseline合理：健康直接Starter、角色/锚点统一、非训练、同协议公平、全部正式seed完整。 `optimization_evidence/训练证据说明.md:5`；`optimization_evidence/comparison_summary.json`；`workspace/harbor_task/environment/starter/method.py` |
| G03 | pass | 正式Dev全5组有效：Δ=.5687387750=4.67586077σB，归一化=.5327010264；达到3σ和[.15,.8]。私有预校准亦支持，非官方Hidden验收。 `optimization_evidence/comparison_summary.json:16`；`optimization_evidence/baseline_runs/seed_17/run.log:1`；`optimization_evidence/reference_runs/seed_17/run.log:1` |

## 双轨有效时长、八字段与最终方法

| 模型轨 | 打包轮数 / 声明完整闭环 | 原下界 | 本轮核对 |
|---|---|---:|---|
| Codex + GPT-6.1 Sol | 64 / 59 | 36368.013530140s | GPT下界36368.013530s=10.10222598h，逐原始session算术一致。 |
| Codex + Doubao Seed 2.1 Turbo | 35 / 8 | 36026.135163140s | Seed原下界36026.135163s=10.00725977h；已定位纯连接故障残余29.838178s，修正该下界35996.296985s=9.99897138h，≥10h证据待核。 |

逐段算术：GPT71个计时条目、已观察单调总量49795.917229s、模型不确定量扣除13427.903698s；Seed42个条目、62858.170780s、扣除26826.771622s；区间无重叠。已与各runtime.json的session起止/observed/model_seconds、原始评分状态核对。包内宽松估算GPT14.6509h、Seed18.0615h仅保留来源，不能替代当前认证。

Seed round20–23原始事件没有模型方法工作，只有错误条目与连接失败；runtime均0成功工具、未完成、transport_failed=true，24请求全URLError。包内计时代码只扣5–15秒重试暂停（`expert_evidence/raw_evidence/tools/recalculate_active_time.py:60–65`），把短重试间隙算入“有效模型阶段”，与当前V4扣除API失联/故障要求不符。扣除后是当前证据下界不足，不是已证明真实总时长不足；可用已有其他正常工作区间补核，不要求独立计时器或全量新日志。最终声明不使用7h例外，暂不凭“非训练”自动授予例外。

两轨轮号连续、时间格式可解析、score逐轮与所指原始trajectory.jsonl一致，成功/失败分数null处理保留。但正式每轮额外packaged_source（Seed部分另有original_status），Seed摘要含工具代码/DSML；GPT round14摘要称0.500445后回退，正式score=.3816908283且retained_best=true。内部自测与外部可信评分可以不同，必须明确两者并忠实记录决策，当前摘要未做到。

最终唯一方法以全冻结公开seed比较：GPT均值.4279666746653184胜Seed .4803020135338603，选GPT；干净Dev raw=.4187087604867304、score=.607822437521813，私有复验raw=.40314299028368433、score=.6214121096023081。见 `expert_evidence/clean_selection_1791505706/selection.json`、`expert_evidence/best_method/verification.json`。私有复验晚于公开选优；未见将每轮Hidden回填为Dev分数的直接证据。12.0028h稳定性仅已有Starter镜像工作负载记录，不计研究时长。

## 格式、Harbor部署与评分隔离

三个顶层目录齐全，非训练无空model目录。Agent=context environment/，COPY仅公开Starter/Dev；Verifier=context tests/，私有protocol、grader、worker及依赖自备。候选路径/workspace/solution由artifacts保持原路径；separate明确。regular/size/AST guard、只读候选源、kernel降权、私有protocol600、可信oracle父进程和原子reward均有实现；本次未动态验证权限/镜像或对抗隔离。

| Harbor项 | 状态 | 证据与结论 |
|---|---|---|
| H01 | pass | 完整任务根、公开Dev与双Dockerfile均存在；Hidden由私有非空protocol定义确定性生成，grader在Verifier读取。 `workspace/harbor_task/task.toml`；`workspace/harbor_task/tests/Dockerfile:5`；`workspace/harbor_task/tests/protocol.json:22`；`workspace/harbor_task/tests/grader.py:95` |
| H02 | fail | TOML文本结构及separate明确，但storage_mb、禁网、白名单与当前内场合同冲突；包内旧native schema记录不是当前部署验收。 `workspace/harbor_task/task.toml:18`；`workspace/harbor_task/task.toml:24`；`workspace/harbor_task/task.toml:30`；`workspace/harbor_task/task.toml:37` |
| H03 | pass | Agent context=environment，仅COPY公开文件；Verifier context=tests，依赖自备，入口/tests/test.sh；artifacts移交/workspace/solution并保持路径。 `workspace/harbor_task/environment/Dockerfile:6`；`workspace/harbor_task/tests/Dockerfile:2`；`workspace/harbor_task/tests/Dockerfile:5`；`workspace/harbor_task/task.toml:2`；`workspace/harbor_task/tests/test.sh:4` |
| H04 | pass | test.sh清理预写reward，调用可信评分，有限标量经临时文件原子replace写reward.txt；格式/超时/基础设施错误不伪装零分。 `workspace/harbor_task/tests/test.sh:3`；`workspace/harbor_task/tests/test.sh:10`；`workspace/harbor_task/tests/grader.py:254` |
| H05 | pass | 已有Trial为docker provider，nop/研究Agent配置可识别，verifier.disable=false；结果记录separate。未执行当前原生schema加载。 `expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/result.json#config`；`expert_evidence/raw_evidence/campaigns/asd-research-1791486012/harbor_task__VqWxhgD/config.json` |
| H06 | manual | 已有同Trial config/result/reward/日志，nop、separate、无异常且reward一致；当前精确任务版本关联仅有导出说明，未用真实新版指纹独立确认，非阻断待补。 `expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/config.json`；`expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/result.json:89`；`expert_evidence/harbor_nop_v023/harbor_task__aZLUUS4/verifier/reward.txt:1`；`expert_evidence/harbor_nop_v023/portable_export.json:18` |

路径静态通过与当前配置接收性分开：包内目标Harbor0.23.0、provider=docker；旧schema-validation记录指0.1.0，不作为0.2.3本次独立schema加载。已有NOP同Trial的reward=-.04457318804105793、separate、无异常且已结束；负分本身不失败，精确版本关联非阻断后补。未执行任何NOP/Harbor/Docker。

格式建议（额外材料不单独扣错）：
- `expert_evidence/best_method/` [misplaced]：目录含rerun_dev/、rerun_hidden/、verification.json与verifier-build.log。将复验日志和私有评分材料移到expert_evidence独立validation/，best_method仅留唯一method.py/solve.sh及必要配置；不因多交单独扣错。
- `expert_evidence/absolute_cpu_time_protocol_v1/` [extra_allowed]：保留为明确标注的历史协议证据，继续与正式v2相分离。
- `expert_evidence/专家作业说明文档.md:41` [merge_candidate]：该段仍称GPT安全收尾/唯一方法复验待完成、Seed29轮，与最终run_summary及64/35轮轨迹不同；更新最终状态并保留历史说明出处。
- `expert_evidence/run_summary.json#guard_repair_evidence` [misplaced]：指向包外相对位置且相应材料未在该位置。若作为当前证据使用，应改为包内实际路径或明确历史缺失；不用其证明当前guard验收。
- `expert_evidence/trajectory_codex.json#rounds[*].packaged_source` [misplaced]：出处有用，移至顶层轮次映射以满足正式每轮恰八字段；不丢原始证据。

## 实现检查21项

| 检查项 | 状态 | 原因 / 证据 |
|---|---|---|
| QA01 | pass | 题面八部分齐全，无论文标题、arXiv、解法仓库或Reference直接提示。 `workspace/harbor_task/instruction.md:1`；`workspace/harbor_task/instruction.md:3`；`workspace/harbor_task/instruction.md:28` |
| QA02 | pass | score=(B-x)/(B-0)，B>0，连续单调不裁剪；B为正式5seed均值，数学floor U=0。 `workspace/harbor_task/tests/grader.py:239`；`workspace/harbor_task/tests/grader.py:243`；`workspace/harbor_task/environment/public_assets/dev/protocol.json:19` |
| QA03 | pass | Reference源码/入口及五seed结果原始日志完整，非训练无需checkpoint；归一化.532701在要求区间。 `workspace/reference/method.py`；`workspace/reference/solve.sh`；`optimization_evidence/comparison_summary.json` |
| QA04 | pass | 计时随机性明确；基于全部5对原始值复算n−1样本标准差，改善4.67586077σ≥3σ；3–5σ为可接受并建议复核。 `workspace/harbor_task/instruction.md:12`；`optimization_evidence/comparison_summary.json:43` |
| QA05 | fail | 数值/源码/超时/资源限制有实现，但题面与TOML仍以禁网/白名单/storage_mb声明当前内场不接收的环境约束。 `workspace/harbor_task/instruction.md:20`；`workspace/harbor_task/task.toml:18`；`workspace/harbor_task/task.toml:24`；`workspace/harbor_task/task.toml:37`；`workspace/harbor_task/tests/grader.py:119` |
| QA06 | pass | 标准solve.sh提供Dev单命令；最终test.sh调用可信grader并原子写有限reward；失败类别有独立诊断。 `workspace/harbor_task/environment/starter/solve.sh:3`；`workspace/harbor_task/tests/test.sh:4`；`workspace/harbor_task/tests/test.sh:10`；`workspace/harbor_task/tests/grader.py:254` |
| QA07 | pass | 仅检查题面：没有Hidden具体形状/标签或可恢复答案；描述独立私有输入合法。 `workspace/harbor_task/instruction.md:7`；`workspace/harbor_task/instruction.md:20` |
| QA08 | pass | 仅检查题面：未披露Reference代码、Gram计算方案、具体Reference配置或分数。 `workspace/harbor_task/instruction.md:10`；`workspace/harbor_task/instruction.md:16`；`workspace/harbor_task/instruction.md:28` |
| QA09 | pass | 可信入口清理预写reward/result；result与reward均经临时文件replace写入。 `workspace/harbor_task/tests/test.sh:3`；`workspace/harbor_task/tests/test.sh:12`；`workspace/harbor_task/tests/grader.py:180`；`workspace/harbor_task/tests/grader.py:195` |
| QA10 | pass | 公开冻结目录root所有/去写权限，冻结solve.sh逐字节检查；Verifier候选复制只读并降权kernel，protocol仅root可读。 `workspace/harbor_task/environment/Dockerfile:13`；`workspace/harbor_task/tests/Dockerfile:6`；`workspace/harbor_task/tests/grader.py:124`；`workspace/harbor_task/tests/grader.py:198`；`workspace/harbor_task/tests/grader.py:211` |
| QA11 | pass | 目标输入输出、环境、公式、开放范围、禁令、提交、迭代及完成标准均有实质内容；部署策略冲突已由对应项列出。 `workspace/harbor_task/instruction.md:3`；`workspace/harbor_task/instruction.md:6`；`workspace/harbor_task/instruction.md:9`；`workspace/harbor_task/instruction.md:16`；`workspace/harbor_task/instruction.md:19`；`workspace/harbor_task/instruction.md:22`；`workspace/harbor_task/instruction.md:25`；`workspace/harbor_task/instruction.md:28` |
| QA12 | pass | 清点745个成员未发现.git目录、remotes/tags/reflog。 `@inventory` |
| QA13 | pass | 正式Baseline/Reference五seed集一致，10份result及原始stdout完整，逐seed均改善、质量门通过；历史v1不混入正式统计。 `optimization_evidence/comparison_summary.json:16`；`optimization_evidence/训练证据说明.md:11` |
| QA14 | pass | 函数接口可进行融合投影、Gram/Cholesky、布局/分块/等价alpha路径等方法更新，题面未直接提示Gram解；Reference分数未饱和。 `workspace/harbor_task/instruction.md:16`；`workspace/reference/method.py:18`；`expert_evidence/best_method/method.py:86` |
| QA15 | pass | CPU题0 GPU，正式单seed端到端45.62–60.23秒；完整五seedDev/私有复测均远低2h，已有1CPU/2GiB日志支持。历史设备为linux/aarch64，未冒充H20/L20。 `optimization_evidence/baseline_runs/seed_101/result.json#execution`；`expert_evidence/best_method/rerun_dev/result.json#start_unix`；`expert_evidence/专家作业说明文档.md:27` |
| QA16 | manual | 算术复核原下界GPT36368.014s、Seed36026.135s；但Seed纯URLError轮20–23仍计29.838s，修正后该下界35996.297s，当前≥10h证据待重算。 `expert_evidence/auditable_time_lower_bound.json:1875`；`expert_evidence/raw_evidence/tools/recalculate_active_time.py:60`；`expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/runtime.json#sessions[round=20..23]` |
| QA17 | fail | 目录、路径、separate、移交和reward静态成立；H02当前内场配置冲突使兼容性不通过。NOP精确关联为非阻断待补，不是本项失败原因。 `workspace/harbor_task/task.toml:18`；`workspace/harbor_task/task.toml:24`；`workspace/harbor_task/task.toml:30`；`workspace/harbor_task/tests/test.sh:4` |
| QA18 | fail | 两轨完整留存且score可回溯原始记录，但每轮多字段、Seed摘要含DSML/代码、GPT round14摘要/可信分数/保留标记矛盾，未满足当前八字段及语义规范。 `expert_evidence/trajectory_codex.json:151`；`expert_evidence/trajectory_codex.json:155`；`expert_evidence/trajectory_seed.json:8` |
| QA19 | pass | 题面明确计时随机、正式seed与3σ有效改善阈值；不额外要求重复次数文字，但已声明5seed全部提交。 `workspace/harbor_task/instruction.md:12` |
| QA20 | pass | 题面明确可改范围、本地工具与研究网络策略；其当前内场禁网/白名单冲突由部署检查判退，不把本项存在性重复判缺失。 `workspace/harbor_task/instruction.md:16`；`workspace/harbor_task/instruction.md:20` |
| QA21 | manual | run_summary缺task_binding/hash_profile/task_content_sha256；真实发布指纹脚本不可用，未猜算法，按当前口径非阻断待补。 `expert_evidence/run_summary.json#task_binding (missing)` |

静态覆盖完成：全部G/B/H/QA项均已检查；manual项表示尚未闭环的事实，不掩盖已确定的fail。QA计数：{"pass": 16, "fail": 3, "manual": 2}。

## 核心问题、最小修订与验收材料

1. **deployment_policy — fail**：题面第20行声明禁网和模型主机白名单；TOML agent.network_mode/allowed_hosts、environment.storage_mb/network_mode、verifier.environment.storage_mb/network_mode 仍声明当前内场不接收的限制。公开镜像内重复题面/配置相同。

   定位：`workspace/harbor_task/instruction.md:20`；`workspace/harbor_task/task.toml:18`；`workspace/harbor_task/task.toml:19`；`workspace/harbor_task/task.toml:24`；`workspace/harbor_task/task.toml:25`；`workspace/harbor_task/task.toml:37`；`workspace/harbor_task/task.toml:38`；`workspace/harbor_task/environment/instruction.md:20`；`workspace/harbor_task/environment/task.toml:18`。

   影响：当前V4明确内场不声明storage_mb、禁网或网络白名单。这是确定的配置/部署合同冲突，属于核心配置问题，不能归入换卡或NOP放宽。

   最小修改：删除 task.toml 及 environment/task.toml 中 storage_mb、network_mode、allowed_hosts；同步两份 instruction.md 的禁网/白名单承诺，改为当前平台默认联网/工具策略。保留候选纯函数与禁止外部数据的可执行计算边界。

   验收：重交同步的题面与配置、真实格式/资源前检结果或平台可接收配置说明；目标环境核对单容器、1 CPU/2 GiB 与 separate。不得将历史禁网实测改标为当前部署；本修订本身不要求全 seed 重跑。

2. **trajectory_semantics — fail**：GPT 64轮、Seed 35轮全部有第9字段packaged_source；Seed失败轮另有original_status。Seed round1/4/5等摘要直接含Python/DSML片段。GPT round14摘要写0.500445且已回退，实际score=0.3816908283且retained_best=true，未区分内外两次评测；这是摘要/决策语义不一致，并非正式B/R伪造证据。

   定位：`expert_evidence/trajectory_codex.json:14`；`expert_evidence/trajectory_codex.json:151`；`expert_evidence/trajectory_codex.json:153`；`expert_evidence/trajectory_codex.json:155`；`expert_evidence/trajectory_seed.json:8`；`expert_evidence/trajectory_seed.json#rounds[13].original_status`。

   影响：V4要求每轮恰八字段、摘要1–3句忠实于生效改动及结果，并可追溯评分和最佳标记。原始来源分数一致不消除摘要和决策矛盾。

   最小修改：保留原始轨迹不改写；重新生成正式两条 rounds，每轮恰好八字段，把 packaged_source/original_status 移到顶层轮次映射；将工具调用片段整理为1–3句实际生效方法摘要，明确内部测量与可信外部测量，按实际保留决策同步摘要/retained_best。

   验收：两份可解析八字段轨迹、顶层原始出处映射；逐轮与对应 grader.log 的 raw_value/status/源码版本及恢复或保留记录一致，特别核对 GPT round14 和 Seed round1、2、4、5。无需补造实验。

3. **effective_time — manual**：Seed声称可审计下界36026.1351631398秒，但round20–23分别计入9.1474644737、6.5681648317、6.6756276535、7.4469207369秒。对应原始事件只有配置警告、连接错误与turn.failed，runtime记录成功工具0、model_transport_failed=true、model_completed=false，所有24次请求为URLError。扣除这29.8381776959秒，现有该下界变为35996.2969854439秒（9.9989714h）。

   定位：`expert_evidence/auditable_time_lower_bound.json:1875`；`expert_evidence/auditable_time_lower_bound.json:1894`；`expert_evidence/auditable_time_lower_bound.json:1913`；`expert_evidence/auditable_time_lower_bound.json:1932`；`expert_evidence/raw_evidence/tools/recalculate_active_time.py:60`；`expert_evidence/raw_evidence/tools/recalculate_active_time.py:62`；`expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/runtime.json#sessions[round=20..23]`；`expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/round_00020.events.jsonl:3`；`expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/coding-plan.audit.jsonl#started_unix[1791422341..1791422372]`。

   影响：API失联和故障重试不能计有效研究时间；包内历史用户的宽松计时叙述不能替代本次当前V4。修正的是已认证下界，不能据此断言真实全部研究时间必然不足10h，其他未认证区间仍可能包含有效工作。当前10h证明尚不成立。

   最小修改：按当前V4从已有原始事件重新核算有效时间，扣除纯连接故障/重试区间，不能用短于5秒为由继续计入；保留正常失败方法工作、模型思考及真实评分，重叠只扣一次。优先从现有记录中核实其他实际有效区间。

   验收：修订的逐段起止、故障扣除、并集及合计表，引用原始 runtime/events/request audit；每条有效时长≥36000秒，或满足当前7h例外的完整依据。现有证据不足时如实标未完成，不冒称实跑不足或伪造新时间。

## 五类放宽待补（单列、非阻断）
- QA21指纹/哈希：run_summary.task_binding缺失；保留方法/协议/ZIP真实哈希，未推算官方四项指纹。 后补材料/口径：真实发布脚本及四项指纹输出，非当前通过前提。
- G02/G03换卡影响：CPU题，现有aarch64真实记录；没有H20/L20迁移补证要求。 后补材料/口径：不适用，不要求补跑。
- QA15换卡耗时：CPU0 GPU；已有端到端耗时，未提出换卡后重计时。 后补材料/口径：不适用。
- QA17/H06 NOP：已有完整NOP同Trial且reward一致，精确当前版本关联待核。 后补材料/口径：后补关联即可，不自动运行。
- QA03/QA13换卡重算：未因换卡要求重算；正式历史CPU分数保真使用。 后补材料/口径：不适用。

附：协议hash差异保真记录（非阻断待核）：正式Dev result.protocol_sha256=`d1a9010296bbbcc4bef8723ba1a3eadd6e6c443c33fd9bb304b2d83aebab55d5`，当前Dev成员SHA256=`e22d4f05e7e41efb171a5d6cd6ca3c1e3ca1ae76acaf33ef04aacccde9944d59`；正式私有校准protocol_sha256=`06d343a312dde5454ea5f680c0b2dccef24dbe50ef50a92fe67025f054240c6c`，当前tests/protocol.json SHA256=`179ffb8c60d0e85b0018016eed07c9cdfd7483d33e9d08193ce2d2232cc15813`。正式raw-only校准与当前含B快照不同，保留差异并后补版本映射；单凭hash差异不推出篡改、损坏或造假。最终五seed干净复验分别56.0913s/57.9164s。

## 工具缺口与本次运行边界

真实官方0.3.10、独立check_submission/GPU/task_fingerprint工具不可用，完整工具验收待补；没有改版本字符串、猜官方哈希或用旧工具PASS冒充新版。仅运行可信自编ZIP解析/数值复算脚本；未执行或导入提交代码、构建镜像、运行Verifier/安装/训练/模型调用、反序列化权重、动态重跑及12h测试。当前官方separate文档只作2026-10-22静态规范对照，0.23.0精确源码未成功取得，不声称加载其schema。

workspace可见源码/题面文本的home/邮箱/私网IP扫描未命中；未代替主控执行全二进制发布隐私门禁。报告仅列包内相对路径/逻辑artifact，不回显秘密或私有端点。

风险（最多3条）：
- 随机计时改善处3–5σ区间，支持当前最低门槛；不能称≥5σ强证据或硬件无关不变成绩。
- Seed有效时间认证余量很小，已查出纯连接故障被计入；需重核现有有效区间，不能把历史宽松估算当当前证明。
- 本次静态审查与已有本地Harbor记录不等于当前平台部署、最终Hidden或独立动态复现。

## 可复制给专家的退回说明

本轮当前V4静态复核结论为不通过（NOT READY），G01/G02/G03与正式五seed成对分数通过，不需要为此主动削弱Baseline或重做全部实验。

1. [fail] 题面第20行声明禁网和模型主机白名单；TOML agent.network_mode/allowed_hosts、environment.storage_mb/network_mode、verifier.environment.storage_mb/network_mode 仍声明当前内场不接收的限制。公开镜像内重复题面/配置相同。
定位：workspace/harbor_task/instruction.md:20；workspace/harbor_task/task.toml:18；workspace/harbor_task/task.toml:19；workspace/harbor_task/task.toml:24；workspace/harbor_task/task.toml:25；workspace/harbor_task/task.toml:37；workspace/harbor_task/task.toml:38；workspace/harbor_task/environment/instruction.md:20；workspace/harbor_task/environment/task.toml:18。
影响：当前V4明确内场不声明storage_mb、禁网或网络白名单。这是确定的配置/部署合同冲突，属于核心配置问题，不能归入换卡或NOP放宽。
最小修改：删除 task.toml 及 environment/task.toml 中 storage_mb、network_mode、allowed_hosts；同步两份 instruction.md 的禁网/白名单承诺，改为当前平台默认联网/工具策略。保留候选纯函数与禁止外部数据的可执行计算边界。
验收材料：重交同步的题面与配置、真实格式/资源前检结果或平台可接收配置说明；目标环境核对单容器、1 CPU/2 GiB 与 separate。不得将历史禁网实测改标为当前部署；本修订本身不要求全 seed 重跑。

2. [fail] GPT 64轮、Seed 35轮全部有第9字段packaged_source；Seed失败轮另有original_status。Seed round1/4/5等摘要直接含Python/DSML片段。GPT round14摘要写0.500445且已回退，实际score=0.3816908283且retained_best=true，未区分内外两次评测；这是摘要/决策语义不一致，并非正式B/R伪造证据。
定位：expert_evidence/trajectory_codex.json:14；expert_evidence/trajectory_codex.json:151；expert_evidence/trajectory_codex.json:153；expert_evidence/trajectory_codex.json:155；expert_evidence/trajectory_seed.json:8；expert_evidence/trajectory_seed.json#rounds[13].original_status。
影响：V4要求每轮恰八字段、摘要1–3句忠实于生效改动及结果，并可追溯评分和最佳标记。原始来源分数一致不消除摘要和决策矛盾。
最小修改：保留原始轨迹不改写；重新生成正式两条 rounds，每轮恰好八字段，把 packaged_source/original_status 移到顶层轮次映射；将工具调用片段整理为1–3句实际生效方法摘要，明确内部测量与可信外部测量，按实际保留决策同步摘要/retained_best。
验收材料：两份可解析八字段轨迹、顶层原始出处映射；逐轮与对应 grader.log 的 raw_value/status/源码版本及恢复或保留记录一致，特别核对 GPT round14 和 Seed round1、2、4、5。无需补造实验。

3. [manual] Seed声称可审计下界36026.1351631398秒，但round20–23分别计入9.1474644737、6.5681648317、6.6756276535、7.4469207369秒。对应原始事件只有配置警告、连接错误与turn.failed，runtime记录成功工具0、model_transport_failed=true、model_completed=false，所有24次请求为URLError。扣除这29.8381776959秒，现有该下界变为35996.2969854439秒（9.9989714h）。
定位：expert_evidence/auditable_time_lower_bound.json:1875；expert_evidence/auditable_time_lower_bound.json:1894；expert_evidence/auditable_time_lower_bound.json:1913；expert_evidence/auditable_time_lower_bound.json:1932；expert_evidence/raw_evidence/tools/recalculate_active_time.py:60；expert_evidence/raw_evidence/tools/recalculate_active_time.py:62；expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/runtime.json#sessions[round=20..23]；expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/round_00020.events.jsonl:3；expert_evidence/raw_evidence/campaigns/asd-research-1791409298/harbor_task__rwoVqpw/agent/coding-plan.audit.jsonl#started_unix[1791422341..1791422372]。
影响：API失联和故障重试不能计有效研究时间；包内历史用户的宽松计时叙述不能替代本次当前V4。修正的是已认证下界，不能据此断言真实全部研究时间必然不足10h，其他未认证区间仍可能包含有效工作。当前10h证明尚不成立。
最小修改：按当前V4从已有原始事件重新核算有效时间，扣除纯连接故障/重试区间，不能用短于5秒为由继续计入；保留正常失败方法工作、模型思考及真实评分，重叠只扣一次。优先从现有记录中核实其他实际有效区间。
验收材料：修订的逐段起止、故障扣除、并集及合计表，引用原始 runtime/events/request audit；每条有效时长≥36000秒，或满足当前7h例外的完整依据。现有证据不足时如实标未完成，不冒称实跑不足或伪造新时间。

非阻断待补：run_summary.task_binding及官方四项指纹缺失、现有NOP与当前精确版本关联待核；按2026-10-21口径不因这些缺口单独阻断。换卡成绩/计时/重算对本CPU题不适用，不要求补跑。真实官方0.3.10及独立格式/GPU/指纹工具仍待取得；本报告不是官方工具或平台PASS。

保留全部原始数值、失败和旧日志。请重交包含workspace、expert_evidence、optimization_evidence的完整ZIP及新SHA256，不只补丁。此为静态审查及现有证据核对，未独立运行提交程序。
