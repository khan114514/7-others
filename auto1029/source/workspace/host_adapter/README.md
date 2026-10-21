# 当前独立 RootVerifier 的可信主控配置

此目录始终在 Actor 与 Verifier 构建上下文之外。FrozenMethodAgent 复跑不发
模型请求；Dev 评分和 RootVerifier 使用物理分离队列。Candidate 只以 source
bytes 传递，GPU 主控禁止直接 import。Harbor 固定 0.23.0，Python 3.12，
PyYAML 固定 6.0.3。完整依赖和真实 binary/build 证据须与当前源码清单绑定。

`operator-config.example.json` 是非秘密模板，不能直接启动实际评分。用实际
绝对路径和已认证 SSH alias 替换占位值；不做 shell 变量展开。配置必须属于
当前可信主控用户、权限 0600；private_root 为包外同用户所有的 0700 目录。
全部八字段均必填，包括 verifier_gpu_service_argv。字段之外内容会拒绝。

- gpu_service_argv 调用已完整验证安装的 private/gpu_evaluator_service.py，Dev-only。
- verifier_gpu_service_argv 调用同一部署的 private/private_verifier_service.py，
  Root-only1380×5。两个 argv 都不加 --role，由 bridge 固定添加物理角色。
- 两个 credential_commands 都保持空。credential_environment_names 的两个
  禁用变量在复跑进程中必须 unset；配置中不得放密钥、获密钥命令或旧 Ark profile。
- http_proxy 可为 null 或不含用户名、密码、query/fragment 的非秘密 HTTP(S) URL。
  GPU 后端不调用 GPT 或 Seed，配置和凭据不得进入任一容器或候选环境。

本配置用于无模型复评分；不能用旧 provider relay 做新的 Seed 模型请求。
已授权的模型研究及独立 QA 有自己的主控 Coding Plan bridge，必须固定模型和
high。模型路由的真实 CLI 证明、历史隔离和逐请求刷新凭据单独验收。

按 operator/README.md 执行静态 preflight 和只在安全边界执行实际 probe/full。
一个成功的 static preflight 证明字节/布局可解析，不证明新机器安装、GPU
成绩、Native PASS、12h稳定性或 AI QA。基础镜像 binary_manifest 的 ARM64
测试范围与 Linuxx86_64 GPU runtime 完整安装证据分开核验；外部 CLI 二进制
不随轻量题包转发，也不能凭相同文件名替代逐字节 SHA/版本/架构证据。
