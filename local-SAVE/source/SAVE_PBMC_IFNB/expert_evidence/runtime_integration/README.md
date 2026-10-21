# 可信运行适配代码

实际评分、隔离、双构建上下文、私有注入与原生结果字段的源码定位见 [实际执行代码与原生结果索引](execution-contract.md) 和 `execution-contract-index.json`。该索引从已供源码、已安装的 Harbor 0.23 原生类型与既有运行记录生成；不添加新运行或平台验收结论。

此目录是专家/平台可信主控代码，不进入 Agent 构建上下文。`portable_runtime_v4.environment:SaveHybridDockerEnvironment` 为 Harbor 0.23 的自定义 provider；它创建独立无网络 Docker Actor/Verifier、只读 CLI 挂载和各角色独占的队列，并将源代码送往可信 A100 数值服务。它不声明本地 Docker 具有 NVIDIA GPU。

部署值来自独立的 0600 私有 JSON。按 deployment.example.json 准备实际值，再在可信主控设置 SAVE_DEPLOYMENT_FILE。配置、SSH 私钥、Ark 共享凭据和 macOS Keychain 均留在交付包及 Actor 之外；不得把该配置作为容器环境变量或挂载交给 Actor。模型端点、模型标识和档位在代码中固定，禁止按量路由与重定向回退。

数值宿主使用固定 Python 3.10.21 / PyTorch 2.5.1+cu124 和当前 GPU driver。numerical_sources/ 提供实际使用的隔离、评分、锁和冻结实现。gpu_service.py.in 的唯一参数槽是可信 GPU UUID/全部既有共享锁的映射；在可信服务目录把该槽替换为部署映射并保存为 gpu_service.py。原执行源与模板哈希在 source_manifest.json 中分开记录；源重建校验在内存中完成，作者机器路径不随包交付。部署到其他映射后必须重新冻结并实跑，不能沿用本机的运行结论。

数值服务根目录名使用稳定任务标识 save_autoresearch_20261009。按实现准备 preparation/remote_venv、submission/workspace/harbor_task、private/hidden_calibration_assets、private/private_verifier_contract_v1.json 和 private/gpu-jail-runtime-v2；私有资产仅进入可信评分父进程和独立 Verifier。构建脚本先准备只读 CUDA jail，再通过 freeze_gpu_binding.py 冻结源码和输入指纹。existing GPU locks 必须全部存在，UUID 相符且 VRAM 占用不超过 64 MiB 才能启动；不能删除他人锁或抢占任务。

CLI 原生二进制不重复分发。runtime-build/native-binaries.json 给出实际验证的三个 Linux arm64 文件、字节数、SHA256 与版本。部署者从其授权的相同 CLI 安装中提供 codex、rg、codex-code-mode-host 到 runtime-build/agent，逐文件核对哈希；该目录最终恰好五个只读普通文件。其他版本或架构须重新验证。随附 queue_client.py、local_model_proxy.py 与 queue_broker.py 为完整源码。

先用平台适配目录中的 export_native.py 和 inject_private_assets.py 生成原生双上下文，再从新输出目录执行 launch_nop.py。该命令走真实 Harbor Trial 和独立 Verifier。缺配置、依赖、私有注入或 GPU runtime 时不能产出成功 reward。

Seed 的 DSML 响应不会被解释为已执行调用。新版会对指定 Coding Plan 模型发起实际函数调用格式修复请求；只有模型真正返回的 API function call 才交付 Native CLI。原始响应、修复请求、成功/失败及时间分别保留。此修复不改变方法、数据、指标或推理档位。

## 原生配置与 provider 映射证据

`native_schema/existing_config_validation.json` 记录使用实际安装的 Harbor 0.23.0 原生 `TrialConfig`、`TaskConfig`、`JobConfig` 完成的类型校验。对应 JSON Schema 和原生模块的源文件 SHA 一并保留。校验没有启动新的 Trial、模型或候选程序。

历史初次 NOP 的 provider 符号是 `save_hybrid_environment:SaveHybridDockerEnvironment`；portable NOP 使用 `portable_runtime.environment:SaveHybridDockerEnvironment`；当前最终方法复验使用 `portable_runtime_v4.environment:SaveHybridDockerEnvironment`。这些名称是实际配置中的不同版本，不能互相改名冒充。每个配置的有效 kwargs、Agent、未禁用的 Verifier 与原始结果中的 separate 状态均单独列出。`native_schema/*.source.py.txt` 是对应可信模块的可读源码快照，标明当前观察到的源 SHA、导出 SHA 和路径脱敏；它们不是待导入的部署脚本，也不声称新平台已运行。

当前 v4 的可部署源码仍位于 `portable_runtime_v4/`。将此目录放入可信主控的 `preparation/`，以该目录作为 Python import root；由 Harbor 原生 `environment.import_path` 加载 `environment.py` 中继承 `DockerEnvironment` 的类。原生 `agent.name=nop` 使用 Harbor 自带 NOP；最终方法复验配置中的独立 Artifact Agent 是已有运行记录的一部分，不是 NOP 身份。双 Docker 构建上下文与私有注入仍按平台适配目录的实际实现准备。

`job.example.json` 是已由原生 `JobConfig` 校验、但没有执行的单任务 NOP 示例。所有尖括号路径必须由部署者在可信主控侧显式实例化；禁止把示例误称为已有 Job 运行证据。`launch_nop.py` 同样调用原生 `Trial.create`，所以已有单任务 Trial 不依赖额外 job.yaml。已有真实 NOP/最终复验的配置、结果和 reward 在各自证据目录；新平台的配置、凭据、CLI 二进制与 GPU 外部部署不能由本地历史成功代替。
