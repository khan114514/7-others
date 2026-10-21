# 七题复核交接

先阅读根目录 README 和目标题目的 repair_plan.md、qa/consensus.json，再决定返修范围。另两题 auto0505、auto0776 的源码入口在原仓库，当前复核报告位于 existing_two/。

本仓库保存静态复核和待修改材料。历史 PASS 与本次 V4 内容结论分别记录；不得改写原始分数、旧报告或方法证据来制造通过。官方 0.3.10 工具与动态目标环境验收未执行时必须如实注明。

按具体报告处理核心问题。指纹/哈希差异、换卡影响补证、换卡耗时、NOP待补和仅由换卡引出的重算分数，按本次用户口径作为非阻断待补；不能据此忽略真实方法、Baseline、质量门、隔离和运行故障。

修复后交完整题包并重新绑定 SHA256。不得把整个专家材料包挂载给 Actor；只提供题目声明的公开输入。继续研究、付费资源、训练或远端运行须有当前用户授权，不自动恢复历史暂停任务。

如调用 Doubao/Ark/Seed，使用火山 Coding Plan 的 OpenAI 兼容 base_url https://ark.cn-beijing.volces.com/api/coding/v3 ，Chat Completions 路径 /chat/completions；不得自动回退按量接口，不将任何密钥、Cookie 或私有配置写入仓库。
