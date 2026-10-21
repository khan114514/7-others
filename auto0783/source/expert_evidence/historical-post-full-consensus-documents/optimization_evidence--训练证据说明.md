# 训练证据说明

本任务不训练或微调。固定发布权重的 28 层 BF16 decoder，101、202、303 是推理采样随机种子，不是训练种子。每个种子有两个 Hidden 输入案例，每个案例完整执行三次 256 步成对重复。两者共享同一合法预训练起点、同一输入、同一 CUDA 子进程、交替执行顺序及相同质量门，不能把这些测量称为三次独立训练。

本目录六个 result.json 从新版题包同一次真实 Oracle 成对工作提取，不新增运行，不拼接历史分数。18 条重复记录和全部 4608 阶段原始数据、六份最终 latent、原 CPU 独立复算报告、真实命令和日志保存在 expert_evidence/release-full-validation/oracle。每个 run.log 是同一工作对应的原独立 CPU 审计 stdout，以原始内容显式重复；不是伪造的逐种子 GPU stdout。原 worker/GPU stdout journal 已通过真实 SSH 读取并持久备份，公开隐私派生副本位于 oracle/actual-worker.stdout.jsonl；同一父进程 journal 也保留该父执行的其他 release 作业，不能冒充逐种子独立日志。真实起止时间、耗时和 CUDA 退出码属于整个成对工作，每个种子未另造起止时间。

不创建空 model 目录；checkpoint 描述及校验在 dependencies/checkpoint，可通过固定公开源材料化。所有角色 result 的 model 状态为 NOT_APPLICABLE。此处 COMPLETE 只指 Baseline/Reference 全部正式成对测量与原独立 CPU 审计齐全，不代表专家研究时长、容器连续性、整包隐私或双 AI 质检通过。
