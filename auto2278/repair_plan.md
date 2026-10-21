# Symmetric Attention Decomposition / 注意力效率：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `bd75ad937f2e1ba807bd8db95f569e7cbac350e1258dac27465129a2dff28e42`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA17, QA18, H02, H03。待核项：QA16。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

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


## 独立审查 B 的具体建议

本次独立V4静态复核结论：NOT READY。现有优化面、正式Baseline和Reference提升数值符合门槛；请修复以下核心问题并重交完整ZIP。未执行题包代码或动态复跑，不是官方0.3.10/平台验收。

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

五类放宽事项（单独不阻断）：
- 指纹/哈希：nonblocking_pending。task_binding两个字段缺失；历史协议哈希与校准后协议可能不同，未用哈希差异否定正式数据。 后补建议：真实脚本补四项指纹与版本映射，不全量重跑。 依据：expert_evidence/run_summary.json；optimization_evidence/baseline_runs/seed_17/result.json#protocol_sha256
- 换卡后B/R影响：not_applicable。CPU算子；保留linux/aarch64历史设备，未将其标为H20/L20实测。 后补建议：若平台更换CPU环境，可后补计时比影响说明。 依据：expert_evidence/专家作业说明文档.md:27
- 换卡耗时：not_applicable。CPU0GPU，已有计时日志；本次未验证目标平台耗时。 后补建议：目标平台时间验证另列，不单独阻断。 依据：expert_evidence/best_method/rerun_dev/result.json#clean_execution
- NOP记录/版本：nonblocking_pending。既有NOP链路完整，当前版本绑定未核验。 后补建议：复用并补版本关联，不自动补跑。 依据：expert_evidence/harbor_nop_v023/portable_export.json
- 换卡后重算分数：not_applicable。当前审查使用已有真实CPU正式全seed数据；不要求因换卡重做。 后补建议：仅设备变化引起的重算可延期。 依据：optimization_evidence/comparison_summary.json

工具缺口：真实官方0.3.10发布包及独立格式/GPU/指纹脚本未取得，完整官方工具验收待补；未调用旧0.3.2收集器或伪改版本。取得后记录真实版本/SHA和原始报告。
运行边界：仅静态读取ZIP的文本/JSON/源码AST与元数据及自行复算统计；未导入/执行提交代码，未运行Docker、Verifier、训练、模型、安装脚本或权重反序列化。已有运行记录不等于本次独立复跑、平台正式Hidden或目标环境隔离认证。
可见隐私：已读workspace源码/题面未见个人home、邮箱或私有端点命中；未实施完整二进制严格扫描，该工作由主控另行完成。报告未复制秘密或本机个人路径。
风险仅列三项：可绕过缓存限制；旧网络/磁盘配置与当前内场不兼容；轨迹及有效时间摘要与原始记录未完全对齐。

