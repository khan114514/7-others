# ST-SSDL 时空序列预测：修改建议

本次结论：**NOT_READY**。两次独立静态复核绑定同一个 ZIP，SHA256 `18ee42736dcb71d492cf47a8f83a3d863d148cdb4b1a3d4a9488a8c8087e3dba`。官方 0.3.10 工具及目标环境动态验收未执行。

报告不通过项：QA05, QA17, QA18, H02, H03。待核项：无。五类放宽待补项按原报告单列，不要求为这些缺口自动重跑研究。

## 独立审查 A 的具体建议

本轮独立静态内容审查为 NOT READY。研究三门通过；以下当前V4核心问题需修订。

C1 Unsupported network/storage conditions
事实/定位：workspace/harbor_task/instruction.md:16; workspace/harbor_task/task.toml:9; workspace/harbor_task/task.toml:18; expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:51
最小修改：Remove unsupported storage_mb and no-network/allowlist requirements from current submission. Synchronize instruction,TOML and offline-only adapter; retain private-data/scoring confinement under actual platform networking.
复验验收：New complete ZIP with supported network/resource declarations and actual platform mapping; constraint call chain remains valid. Do not automatically rerun all training solely for this edit.

C2 Compose-dependent formal build/runtime and old task-root context
事实/定位：workspace/harbor_task/environment/docker-compose.yaml:1; workspace/harbor_task/tests/docker-compose.yaml:1; workspace/harbor_task/environment/Dockerfile:5; expert_evidence/provenance/harbor_adapter/forecast_harbor_environment.py:72
最小修改：Remove direct Compose files and formal Compose/extra broker-container dependencies. Put public Starter inside environment and rewrite Agent COPY for environment build context; preserve separate Verifier and candidate artifacts handoff.
复验验收：Complete package,one-container startup contract,valid COPY/dependency/entry mappings for both contexts and platform adapter evidence. Dynamic build/handoff/reward evidence only in an authorized target environment.

C3 Trajectory rounds have extra keys and vague scorer-busy classification
事实/定位：expert_evidence/trajectory_codex.json#rounds[1]; expert_evidence/trajectory_seed.json#rounds[0]
最小修改：Keep exactly8 keys per round. Move original_status and method_summary_evidence into a sidecar with original records preserved. Classify409 scorer-busy separately from invalid methods and clarify ledger rounds versus effective method cycles.
复验验收：Both trajectory JSONs have exactly8 round keys,unchanged original scores/times/retention,and relative sidecar references. Failure classification traces to original trusted evaluation command.

外发门禁：Author-machine absolute paths remain. Only types and member locations are reported; values are not copied. Runtime container paths must be distinguished from author-machine paths. No whole-binary/nested-archive strict certification or external release was performed.
仅报命中位置：AUTHOR_ABSOLUTE_PATH expert_evidence/provenance/forecast-export-manifest.json:5; AUTHOR_ABSOLUTE_PATH expert_evidence/provenance/shared-backbone-training.run.log:254; AUTHOR_ABSOLUTE_PATH expert_evidence/provenance/preparation-complete.json:3
修改与验收：Regenerate author-machine paths as relative paths or stable logical identifiers. Preserve raw originals outside release attachments. Rebuild the complete ZIP and strictly scan each final artifact separately. Zero-blocker strict scan for final ZIP and every attachment,rebuilt member/hash manifests,and relative-only report references.

非阻断待补：缺官方四项指纹可后补；不因换卡自动全量重算重训；本包已有当前本地NOP，后续版本关联可后补，不豁免确定Compose问题。
工具缺口：官方0.3.10及格式/GPU/指纹工具验收待补，不冒充官方PASS。
本次只静态读取唯一完整ZIP，未执行未知代码、容器、评分、模型或外发。


## 独立审查 B 的具体建议

核心结论：NOT READY：方法空间、Baseline、正式Reference提升和双轨有效时长均有支持；当前内场环境配置/构建/单容器要求明确不符，轨迹有效轮数与故障分类也需修正。不是由五类放宽缺口造成。

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

