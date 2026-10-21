# OneDC 图像压缩：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `df27ff7cd6fff4a7b10d199ce35daec4c8ed65df8aac6e07b50957163525cc7c`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA16, QA17, QA18, H01, H02, H03, H05。待核项：无。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

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


## 独立审查 B 的具体建议

当前完整源码ZIP按V4为NOT READY。方法空间、Baseline合理性及三次正式成对提升可以保留，不要削弱B或调整U修复。

完整交付：tests/hidden_assets只有占位；run_eval.py:164要求manifest。environment/Dockerfile:15–17要求两个named contexts，TOML:28/tests Dockerfile:3只有本地镜像ID/tag。 提供可获取的不可变模型/工具资产入口及校验步骤，或平台认可预置镜像；补Hidden私有材料安全入口、版本/清单/注入操作，不向Agent暴露。 验收材料：接收方可定位所有named build contexts、私有manifest，验证大小/哈希并获得目标平台构建与准备回执；本机镜像ID不是获取入口。

内场配置：task.toml:16,26,34保留no-network/storage_mb；platform_adapter/onedc_harbor.py:49–60依赖extra_docker_compose。 删除task.toml的storage_mb及平台禁网配置依赖，将extra_docker_compose改为当前内场支持的单sandbox单容器配置；保留worker最小权限。明确H20/L20，保留A10历史。 验收材料：更新TOML、镜像/资源/隔离说明、目标版本格式/schema检查和artifact移交链；不因换卡强制重做历史B/R。

有效时长：time-accounting.json合格正式区间GPT5.14083963h、Seed5.2338039536h，扣除前已不足7h；自报14h不符当前口径。 按当前V4扣除构建/排队/故障/阻塞；隔离失效旧轨不得混入。保留暂停状态，另行授权后补真实迭代，不补造时间。 验收材料：两轨独立起止、扣除区间、方法→Public评分→决策；默认各≥10h，短非训练例外各≥7h且完整例外材料/最佳干净复验。

轨迹格式：两canonical每轮18字段、编号缺口和sidecar路径未明确；需补expert_annotation基本标注、唯一最终方法路径。 保留原始sidecar，canonical每轮恰八字段；顺序round与原attempt建立映射，缺号不造实验；引用改包根路径，解释模型失败和控制器采纳差别。 验收材料：两份八字段rounds，顺序编号、原尝试映射、逐轮Public指标/质量门/method hash/最佳决策可回溯，摘要同步。

非阻断后补：四项指纹、H20/L20换卡分数/耗时/锚点影响、NOP原始补充链路。历史NOP reward0且主要文件绑定一致；不因这些项强制新运行。真实官方工具验收另列待补。

发布前确认源码README第64/99行capability-style入口的公开必要性，逐件严格隐私扫描；不在报告复写入口值。

