# 运行方补充说明（2026-10-03 21:0x CST）

本轨道 agent 的 run_summary.md/.json 冻结于 v1.0 收尾（2026-10-03 13:57），覆盖 round 1-4。
v1.1（12:30 放行暖启动）后的 round 5-7 已由 agent 完成闭环并写入 trajectory.jsonl，但截至运行方硬截止（21:30），
agent 尚未将其并入 run_summary 与 best_method（会话在收尾审计循环中延迟）。运行方做如下机械补记，内容全部来自 agent 自己的记录，未新增任何实验结论：

- 正式轨迹：trajectory_codex.json 由运行方按规范格式（{"rounds":[...]}）从 agent 的 trajectory.jsonl 原样转换（旧 v1.0 草稿备份为 trajectory_codex.v10draft.bak）。
- round5 warmstart_official_dd_reuse8192_onepass_uniform：FID-50k=79.1768（quality_gate_failed，教师复用已按 v1.1 计费 1077.57s）。
- round6 warmstart_official_dd_teacher64_onepass_uniform：FID-50k=9.65872744080815，**低于 Baseline 9.663262058820692（-0.05%）**，retained_best=true。归因文件 v11_attribution.json。
- round7 为 round6 的干净复验：FID 逐位一致（冻结 seed 集下确定性复现）。
- 注意：round6 相对 Baseline 的提升幅度 0.0045（0.05%），远小于 3σ 强证据标准；按 G03 属「协议内正向但未达充分性」的边缘结果，结论以 QA 复核为准。
- best_method：agent 的 best_method/ 仍为 v1.0 版本（=官方 Baseline 权重引用）。round6 候选权重在 historical-codex-models/round06/training/checkpoint.pth，评测证据在 agent-output/rounds/round06/ 与 validation/round06_v11_clean_validation/。
- 有效时长：agent 核算 13.696 小时（v11_progress.json，含 v1.1 时段），不低于 10h 要求。
