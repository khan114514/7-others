# Bifrost-1：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `c97e90c1e23768b61b992fa0b03b0894a4a7d425b33bafe077160ff911aa2e3c`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA06, QA15, QA17, QA18, H02, H03, H04。待核项：G03, QA03, QA16。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

核心结论：NOT READY（静态内容复核FAIL；另有核心未核验项）。
1. 当前执行拓扑必须改。Task/task.toml:18/24/39声明禁网和storage_mb，Agent/Verifier均0GPU；Runtime/hybrid_harbor_environment.py:107/137/166生成Compose覆盖、另启broker并依赖远端A100。按当前V4改为各sandbox单容器直接使用自身H20/L20，删除额外工作容器/远端计算及不支持的配置，保持Dev与Hidden边界。重交目标provider配置、完整依赖、单容器/自身GPU及候选移交/reward链路材料。换卡成绩与耗时可后补，不要求为此重做已交正式seed。
2. 两轨时长须据已有材料闭环。保守区间复算GPT25828.664409399033秒、Seed27860.4020447731秒；包内canonical为null。默认10h未证，不能把估计10h写实测。可优先完整说明7h例外：无训练、短轮次依据、每轨≥3真实方法→Dev→决策、持续演进/后续方向、干净最佳复验及排队/下载/故障区间扣除，不要求仅因未知总时长就盲目延长运行。
3. 规范轨迹。expert_evidence/trajectory_codex.json#rounds[47:102]含额外字段、成功COMPLETED和纯实验名摘要；两轨时间均非指定秒级格式。原记录保持不变另存sidecar，主轨迹恰八字段、成功ok、失败/null规则正确、统一秒级时间及声明时区，补生效方法/结果摘要与原反馈映射。指定唯一最终best_method及真实恢复位置，不只保留两套候选路径。
4. 作者Reference正式结果很好，但缺当前公开Dev资格证据。optimization_evidence/及oracle/result.json均为formal。补同Reference图artifact的Dev profile原始结果/log和归一化/质量门，不把formal标签改成Dev。原GPU stdout缺失已披露；能恢复则原样补，不能恢复明确缺失并保留原4608stage、18重复、native Trial及CPU审计，不生成替代日志。
正式B/R无需因分差大而削弱或换SOTA：B58432.48055013021ms、R33729.16768900553ms、σB180.57194646667463ms、改善24703.31286112468ms、归一化0.422766800733905；18×256阶段复算一致、质量误差0。三个seed是采样seed，六份result来自同一成对推理job，并非独立训练。
非阻断待补：官方四项指纹与run_summary.task_binding缺失；A100到H20/L20的成绩影响/锚点与计时补证；目标换卡重算可延期。已有历史NOP可读且一致，不要求因NOP额外开跑。真实0.3.10格式/GPU/指纹工具未提供，完整工具验收待补；本报告不是该官方工具输出，也不是平台通过。
额外路由提醒：Runtime/hybrid_relay.py:24仍保留非Coding Plan的旧Seed路由。若继续执行涉及豆包的工作，须在可信侧按用户当前共享配置切至Coding Plan Chat Completions并保持指定模型；历史证据不改标，本次未读凭据也未调用模型。


## 独立审查 B 的具体建议

Bifrost 完整题包退回说明（独立静态复核）
结论：NOT READY。研究优化面成立，Baseline在材料范围内合理，正式Hidden全量B/R改善强；以下核心交付问题仍须处理。

1. 执行架构与当前平台契约不符。Task/task.toml的gpus=0/storage_mb/no-network与Runtime/hybrid_harbor_environment.py:107、137显示实际靠Compose override、额外broker容器和远程A100 worker。将公开 Dev 和私有 Verifier 的数值执行各自放入对应单容器，直接使用该 sandbox 的 H20/L20；移除远程 worker、额外 broker 容器、Compose override 与数值 SSH/controller 依赖。保留 data-only DAG 校验、候选移交与独立私有评分。提交更新后的题面、TOML、镜像依赖和调用链；提供目标 Harbor 对自身 GPU、单容器及候选→独立 Verifier 的配置/验收材料。历史 A100 数值保真保留，换卡成绩/耗时补验可后补。

2. 失败评分分类需修复。Task/tests/grade.py:53–62全部异常写-1，Task/trusted/numerical_worker.py:50–52异常写reward=0；未完成评估不能当普通分数。区分质量/内存 Hard Gate 与格式、超时、基础设施异常；只有已完成质量评估的无效候选写 -1。未出分异常记录 score=null 并按 Harness 异常通道失败，不产生正常 reward 哨兵。给出正常有限分数、质量失败、格式错误、队列失联/超时四分支的状态/reward 对照及更新源码；异常不得被收集成普通 0 或 -1 排名。

3. 规范轨迹与最终交付。expert_evidence/trajectory_codex.json第48–102轮多字段，45轮COMPLETED而非ok；两轨全部161条time需统一指定格式，摘要需1–3句说明实际方法而不是长原话/标签。保留原始记录，额外来源放旁表，输出恰八字段规范版，分数/null/失败和best标记按实际证据保真。两条selected可继续作证据，另明确唯一最终方法及公开Dev选择依据，提供best_method或明确等价路径。验收查逐轮映射、最终恢复与移交SHA。

4. 公开Dev Reference证据待核。optimization_evidence的全部正式结果来自formal/Hidden，未定位同一作者Reference公开Dev成对自验。补作者 Reference 同一源码/graph 在公开 Dev 协议下的正式成对结果；明确公开与 Hidden 分别使用的案例、重复数、聚合和 B/R/U，现有 Hidden 全量结果继续保留。公开 Dev 的原始阶段/重复记录、全部声明 seed、结果与日志，绑定当前 Reference；复算归一化 [0.15,0.8] 和 ≥3σ。若已有结果可直接补入，无需为本静态审查自动启动运行。

现有正式数据无需因以上文字/格式修订自动重跑：B=58432.48055013021ms，R=33729.16768900553ms，归一化0.422766800733905，σ_B=180.57194646667463ms，改善136.8059σ。全部3seed、18重复、4608阶段来源完整且静态复算相符。
两轨保守活动下界7.175/7.739小时且满足无训练短迭代例外证据；本次不要求补凑10小时，也不将多日wall span当有效时长。已有NOP48项Task文件与当前ZIP一致，不因NOP/指纹放宽项单独退回。
标准四项指纹、H20/L20换卡影响/耗时/重算及迁移后的NOP关联可按2026-10-21口径后补；架构迁移本身不属于该放宽。官方0.3.10与独立脚本验收待补，本报告不宣称平台通过。

