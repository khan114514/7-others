# 当前私有协议的 Native 复跑

本入口为 `run_recover_private_native.py`，使用真实 Harbor 0.23.0、Python 3.12、
Docker Compose v2、离线 Actor 与独立 RootVerifier。运行入口、任务和可信适配器
必须来自同一源码清单。运行中不可修改它们。全部候选 Python 文件只作为字节
移交到已验证 GPU jail，主控不导入候选模块。

本版本完整评分固定 seed 17、29、43、71、101，每 seed 1380 题，共 6900 对
私有记录；每对分别执行 calibration、baseline、candidate。Dev 的 96 题锚点
证据具有自己的协议范围，不能把 Dev 成绩写成这个完整私有协议的成绩。
质量条件和评分由后端冻结协议执行：token 与 answer agreement 至少 0.99、
候选准确率下降不超过 0.01、各角色 parsed answer rate 至少 0.9，calibration
与 baseline token 必须一致。B 是各 seed baseline/calibration 逐题延迟比均值，
M 是 candidate/calibration 逐题延迟比均值，U=0；score=(B-M)/(B-U)，不裁剪。
质量失败的实际返回值为 -1。角色启动耗时已按评测题数摊销计入延迟。

## 安装、私有接线与配置

先按 `../host_adapter/base_image/README.md` 核验和构建规定 ARM64 Actor 基础
镜像，按 `../backend/README.md` 部署 Linux x86_64 GPU 后端及全部外部资产。
控制器架构与 GPU jail 架构是两项独立约束；不能将 ARM64 CLI binary manifest
当作 x86_64 binary 验证。配置仅放可信主控目录、权限 0600；证据目录独立 0700，
不能位于提交包或 Actor 可见目录。按 `../host_adapter/README.md` 填写完整八字段
配置，两个 GPU argv 分别指向 Dev 服务和私有 RootVerifier 服务。

正式运行前，由专家或平台通过私有通道注入冻结的 1380 行及图像。公开包不含
私有问题、答案或图像。完成 full installer、实际 jail audit、完整资产核验和
私有注入后，按 `../../expert_evidence/private_runtime_bootstrap/README.md`
运行 bootstrap；它只处理 root0700 runtime parent，不代表完整评分通过。
安装或完整资产核验只在新部署或安全的评分结束边界进行，不碰在途 GPU 后端。
保留至少 2 GiB 实际部署卷空间，并包含所有共享 GPU 锁；不删除其他任务数据。

本复跑使用 FrozenMethodAgent，不调用模型。配置示例禁用模型凭据获取命令，
运行前移除示例中的两个禁用变量。冻结适配器中的旧 Seed provider 实现不是本
复跑的授权模型入口。实际研究及独立 QA 必须使用经验证的独立 Coding Plan
Chat→Responses 主控桥接及自己的历史、候选和队列：GPT 为 ModCon gpt-6.1-sol/high，
Seed 为 doubao-seed-2-1-turbo-260628/high，Seed 仅
`https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions`，逐请求在可信
主控 RAM 读取当前 0600 配置，禁止回退按量路由。不要为复评分提供模型密钥。

## 命令与模式

以下命令从三目录包根运行；`HEIMA_HOST_CONFIG` 必须是真实绝对配置路径。
`--role` 选择物理评分队列，不能据此把复跑时间计为该模型的研究时间。

```sh
export HEIMA_HOST_CONFIG=/absolute/trusted/operator-config.json
unset HEIMA_REPRO_DISABLED_GPT_KEY HEIMA_REPRO_DISABLED_SEED_KEY
python3.12 workspace/operator/run_recover_private_native.py --policy baseline --role gpt --preflight-only
python3.12 workspace/operator/run_recover_private_native.py --policy best --role seed --preflight-only
python3.12 workspace/operator/run_recover_private_native.py --policy reference --role seed --preflight-only
```

预检读取完整 task/adapter/method 字节及元数据，不启动 Docker、GPU 或模型。
`--probe-only` 构建离线 Actor 并采集 UID1000 的文件视图、四项私有目录拒绝、
物理 mount/network/GPU 隔离及方法字节导出；它不运行 Verifier、不产生分数。
只在目标部署已有评分/资产/隔离门槛且没有本角色在途 ticket 时做新探针。

```sh
python3.12 workspace/operator/run_recover_private_native.py --policy best --role seed --probe-only
python3.12 workspace/operator/run_recover_private_native.py --policy baseline --role gpt
python3.12 workspace/operator/run_recover_private_native.py --policy best --role seed
```

移除模式参数后才执行完整 1380×5 的 separate Verifier。使用实际
NoWallPrivateTrial，显式覆盖外层 Verifier timeout 为 null。task.toml 中 10800 秒是
普通 Harbor Trial 的默认总时限；上述完整私有入口不使用该默认时限，也不证明
普通 Trial 能在 10800 秒内完成。受支持的完整私有模式保留全部 6900 对、冻结
质量条件、推理预算及 0.5 秒 policy IPC 限额，评分协议未缩减。当前记录的
有效配置与源码对应关系见 ../../expert_evidence/supported_invocation/。
`baseline` 使用 Starter，`best` 默认读 `expert_evidence/best_method`，`reference`
默认读 `workspace/reference`；后两者需要完整可运行源码，best 还需非空最多
65536 字节 method_notes.json。显式 `--task`、`--adapter`、`--method-dir` 可选。
运行前检查所属队列 active ticket 和进程；在途 ticket 复用，不重复提交。

原始日志在私有证据目录 `portable-operator-runs/<实际ID>`。只有原始 Native
成功、separate Verifier、五 seed×1380、全部质量、源码/配置/policy/notes 固定、
实际 reward 与 scorer 完全一致，才返回 ACTUAL_PORTABLE_FULL_NATIVE_REPRODUCTION_PASS。
失败保留原始记录，不补写分数。完整逐题重算及数值汇总发布另设门槛，私有
原始载荷只保留于主控侧。未变任务与容器已完成单次 43233 秒、713 次观察的稳定性验证；宿主 SSH
传输重试修复另有实际回归证据。范围见 current_source_stability/，不声称新宿主
适配器已跑满 12 小时。当前已有同一可执行运行版本的完整 NOP 终态；文档修订对应关系见
supported_invocation。独立双模型 HIGH 0.3.2 QA 与共识仍须真实完成。

交付目录还包含未变的独立 base_image/provision.py，用于基础镜像供给。
它不在当前在途评分使用的 11 文件运行适配器中。干净目录静态预检分别记录
这 11 文件与附加供给脚本的完整清单，不把静态预检计为新的 Native 评分。
model-disabled-config-runtime.json 保留此前 CPU 原件；其范围仅为未变的配置
解析及禁用凭据逻辑，不宣称新的网络请求、GPU 评分或传输稳定性。
