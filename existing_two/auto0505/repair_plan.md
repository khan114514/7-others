# Taming Rectified Flow / RF 时间步：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `afa8cc2bfa825049b87ca3bd73909ba76d4300441638a865d72bcc0a6cfcfc77`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA16, QA17, QA18, H03。待核项：G03, QA03, QA15, H02。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

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


## 独立审查 B 的具体建议

本包当前结论：NOT READY（静态核心不通过，非官方工具或平台结果）。

1. 两条当前轨迹rounds为空，run_summary表示尚未开始、effective_seconds=null、best_method=null。请提交两条真实独立模型闭环与有效时间材料、唯一最终方法和干净Public复验。默认各10h；符合完整条件的非训练短迭代才可各7h。保留历史记录但不要补造秒数或把历史QA转写成新证据。

2. 题面及evaluate_candidate.py允许相邻相等节点，但deployment_source/runtime-image/repo/.../flux/sampling.py:133除以相邻步长的一半。请统一严格递减域或实现明确零步长处理，补重复节点/正常节点边界复验；在模型评分前明确无效状态，不能把NaN包装成基础设施失败后当作完成。

3. 两任务Dockerfile依赖本地rfsteps运行tag；部署Dockerfile COPY weights的来源不在ZIP，prepare_weights要求作者原权重目录。请提供平台可获取不可变镜像digest或19项冻结资产的精确合规入口/版本/大小/哈希，修正文档包内路径，补目标环境构建与公开/私有入口调用证据。

4. 历史Public配对提升与归一化已核实达标，Baseline在现有材料内合理。full/hidden U的源文件引用不存在，仅有锚点转抄；请补原始上限逐图指标、方法与日志或明确可审计的估计依据，补comparison_summary和非训练适用性说明。无需为了换卡而自动重算已有成对结果。

5. 当前Harbor版本/provider和A100-only配置的适用范围需可核查依据。包内自述授权不作为本审查收到人类授权的替代。保持历史A100设备和耗时真值；目标GPU迁移耗时/分数、NOP、官方四项指纹等按现行口径非阻断后补。

验收边界：本次只静态读取ZIP；不执行提交代码、模型、Docker或训练。真实0.3.10及格式/GPU/指纹工具尚缺，不声称官方PASS。

