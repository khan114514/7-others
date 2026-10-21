# Bifrost 当前V4独立静态复核 A

ZIP SHA256：c97e90c1e23768b61b992fa0b03b0894a4a7d425b33bafe077160ff911aa2e3c
模型：configured_model=gpt-6.1-sol；runtime_reported_model=not_exposed；runtime_version=not_exposed；session_id=bifrost_a。

## 优化面与实际边界
固定BF16发布权重28层decoder；可提交1–256节点data-only DAG，改依赖、切片/重组、attention计算、prefix生命周期及受限编译region。实际validate不是只收参数。候选Python builder不被Verifier导入。固定采样/质量门及可信算子解释器；延迟越低越好，(B-T)/B连续不裁剪。
证据：Task/instruction.md:17–21、Task/environment/public_eval/tensor_dag.py:26–84、Task/tests/grade.py:28–41。

## Baseline来源、实现与公平预算
发布eager dense双分支decoder，文本/图像两分支全形状投影/注意力/MLP；无训练，固定权重28层。Starter与正式B同语义。
角色：Starter=正式Baseline=B锚点同语义；NOP为额外sanity。无训练/微调，101/202/303是推理采样种子。正式18次重复×256阶段，同job同CUDA子进程、同权重/输入/质量/交替顺序，cache每次重置且首次构建计时。训练曲线/optimizer/checkpoint训练充分性不适用。实际使用A100历史环境不改标成H20/L20。
五维事实：
- training_sufficiency：NOT_APPLICABLE：固定发布权重推理。18重复×256阶段有原始行；不创造训练曲线/epochs。
- method_representativeness：发布eager解码图是固定预测执行加速的可解释合理朴素对照；不要求SOTA。
- budget_comparability：同job、同权重/输入/质量门/配对顺序；首次prefix构建计时。A100到目标卡影响可后补。
- implementation_correctness：源码对齐检查及正式source_alignment=0，全部4608阶段与6最终latent记录通过。未加载权重或独立运行。
- selection_integrity：正式seed全集101/202/303均交，六份result与原stage复算一致；方法源码hash静态匹配。历史搜索完备性未独立认证。

Baseline专项总评：在已提供材料范围内合理。

| 项目 | 状态 | 证据与结论 |
|---|---|---|
| B01 | pass | 正式对照为发布形状的BF16 eager双分支28层decoder；Starter、正式B和归一化锚点语义一致。研究的是固定预测下执行加速，朴素eager对照适用。；Task/environment/starter/method.py:9、Task/environment/public_eval/protocol/benchmark.json#baseline、Task/environment/public_eval/published/baseline_ops.py |
| B02 | pass | 不训练/微调，训练量、优化器、曲线不适用。正式记录含18次完整256步、源实现对齐0误差，且每次重置prefix缓存；没有把未训练模型冒充已训练模型。；optimization_evidence/训练证据说明.md:3、Task/environment/public_eval/full_decoder_evaluator.py:138、expert_evidence/release-full-validation/oracle/result.json#source_alignment |
| B03 | pass | 两侧同一预训练权重、输入/seed、同CUDA子进程和交替计时，首次缓存构建计入；同质量/34GiB上限，缓存是允许研究变量。A100历史设备保真；目标卡补证单列。；Task/environment/public_eval/full_decoder_evaluator.py:147、optimization_evidence/baseline_runs/seed_101/result.json#protocol |
| B04 | pass | 任务冻结dtype/权重/采样/质量门；Reference只增prefix复用，未发现B专属异常参数或额外数据。该结论限已有源码及正式job，不认证所有历史搜索预算。；Task/solution/reference_method.py:34、Task/environment/public_eval/protocol/benchmark.json |
| B05 | pass | 101/202/303为采样seed，实际进入torch.manual_seed和randperm；每seed两个案例、各3重复。六份result是同一个成对job按角色/seed拆分，合法且披露，不是三次独立训练。；Task/environment/public_eval/full_decoder_evaluator.py:126、Task/environment/public_eval/full_decoder_evaluator.py:149、optimization_evidence/训练证据说明.md:3 |
| B06 | pass | 包内NOP以未改Starter测得B58274.88579305013ms、候选58294.56111907959ms，reward -0.000337629593978607，符合计时噪声；正式B未明显低于此适用sanity锚点。；expert_evidence/release-full-validation/nop/result.json |
| B07 | pass | 作者Reference保留同形状dense算子，仅文本分支7个节点改prefix生命周期；正式误差为0。联合GPT/Seed方法另列，没有用其分数代替作者Reference或声称单因素因果。；Task/solution/reference_method.py:34、expert_evidence/release-full-validation/oracle/result.json#quality_pass、expert_evidence/expert_annotation.json#current_method_descriptions |
| B08 | pass | 公开协议给出Baseline来源、共享角色、成本范围；训练证据说明披露无训练、同job拆分、CPU日志重复及GPU stdout缺失。没有把声明当独立复现。；Task/environment/public_eval/protocol/benchmark.json#baseline、optimization_evidence/训练证据说明.md |

## Reference方法与全部正式成对原始值
保持相同算子和形状，将文本norm/q/k/v/out/postnorm/MLP七节点生命周期设为prefix；每prompt/重复重置，首次构建计入计时。
正式记录来自expert_evidence/release-full-validation/oracle/result.json；三seed六份结果各为同一个job的角色/seed视图。run.log为同job原CPU审计stdout，原GPU worker stdout未随包提供且已明确false；不认定为独立重训，也不认定复制日志造假。

| sampling seed | Baseline ms | Reference ms | 正向改善 ms | 原始result路径 |
|---|---:|---:|---:|---|
| 101 | 58263.60826110839844 | 33606.80603027343750 | 24656.80223083496094 | optimization_evidence/baseline_runs/seed_101/result.json；optimization_evidence/reference_runs/seed_101/result.json |
| 202 | 58411.00238800048828 | 33728.20019531250000 | 24682.80219268798828 | optimization_evidence/baseline_runs/seed_202/result.json；optimization_evidence/reference_runs/seed_202/result.json |
| 303 | 58622.83100128173828 | 33852.49684143066406 | 24770.33415985107422 | optimization_evidence/baseline_runs/seed_303/result.json；optimization_evidence/reference_runs/seed_303/result.json |

B均值=58432.48055013021ms；B样本标准差(n−1)=180.57194646667463ms。R均值=33729.16768900553ms；R样本标准差=122.8482629295528ms。改善=24703.31286112468ms；3σB=541.7158394000239ms；改善/σB=136.80592885276224；U=0ms，归一化=0.422766800733905。剩余延迟headroom=33729.16768900553ms。
上述formal质量门通过，4608行每组256步，阶段两误差均0；6个最终latent结果记录通过。延迟有计时噪声，不能视为确定性；全固定权重不意味着评估确定。原阶段→各case三重复中位数→每seed两case均值→三seed均值，独立复算完全一致。

18条完整正式重复（由原raw-stage-times.json逐阶段求和；每条256步）：

| case | seed | repeat | B ms | R ms |
|---:|---:|---:|---:|---:|
| 0 | 101 | 0 | 57705.82113647460938 | 33308.66484069824219 |
| 0 | 101 | 1 | 57946.47552490234375 | 33426.11553955078125 |
| 0 | 101 | 2 | 58105.80786132812500 | 33530.78276062011719 |
| 1 | 101 | 0 | 58428.64637756347656 | 33679.11723327636719 |
| 1 | 101 | 1 | 58580.74099731445312 | 33787.49652099609375 |
| 1 | 101 | 2 | 58696.24633789062500 | 33925.67602539062500 |
| 2 | 202 | 0 | 58668.26139831542969 | 33851.17993164062500 |
| 2 | 202 | 1 | 58763.35719299316406 | 33945.01222229003906 |
| 2 | 202 | 2 | 58811.39796447753906 | 34008.93847656250000 |
| 3 | 202 | 0 | 58277.29808044433594 | 33616.94012451171875 |
| 3 | 202 | 1 | 57982.10556030273438 | 33472.25196838378906 |
| 3 | 202 | 2 | 58058.64758300781250 | 33511.38816833496094 |
| 4 | 303 | 0 | 58285.94079589843750 | 33607.76196289062500 |
| 4 | 303 | 1 | 58554.48667907714844 | 33812.86805725097656 |
| 4 | 303 | 2 | 58707.54910278320312 | 33936.20474243164062 |
| 5 | 303 | 0 | 58549.04533386230469 | 33782.75941467285156 |
| 5 | 303 | 1 | 58691.17532348632812 | 33892.12562561035156 |
| 5 | 303 | 2 | 58713.41368103027344 | 33944.01988220214844 |

其他当前正式验证分数（各自与自己的同GPU Baseline配对，不能作为GPT对Seed直接配对显著性）：

| 方法 | B均值ms | 方法均值ms | 归一化 | 完整数值job秒 |
|---|---:|---:|---:|---:|
| oracle | 58432.480550130211 | 33729.167689005531 | 0.422766800733905 | 1881.432218 |
| nop | 58274.885793050133 | 58294.561119079590 | -0.000337629593979 | 2466.968354 |
| gpt-selected | 58570.950492858887 | 14484.337151845297 | 0.752704420366009 | 1431.889635 |
| seed-selected | 58277.173444112144 | 15967.340546290079 | 0.726010381035333 | 1452.279783 |
每种方法全3seed原值、标准差和质量统计见report.json additional_formal_validations。作者Reference的formal成绩不得冒充Public/Dev自验；当前同artifact未找到Dev原结果。

## 内容门
- G01 pass：可组合1–256个算子节点、改变依赖/切片/注意力/生命周期并选择编译region；实际guard允许结构方法，非固定超参搜索。
  证据：Task/instruction.md:17；Task/environment/public_eval/tensor_dag.py:26；Task/environment/public_eval/bounded_graph_compiler.py:14；Task/solution/reference_method.py:34
- G02 pass：在已提供材料范围内Baseline合理；B01–B08支持健康的固定权重推理比较。没有发现无依据少训练、错误checkpoint或挑seed的直接反证。
  证据：optimization_evidence/comparison_summary.json；optimization_evidence/训练证据说明.md；expert_evidence/release-full-validation/oracle/result.json；expert_evidence/release-full-validation/oracle/raw-stage-times.json；Task/environment/starter/method.py
- G03 manual：正式formal协议提升强且完整：归一化0.422766800733905、改善136.806σB。但作者Reference同一artifact只找到formal原始结果，未核到当前Public/Dev自验分数；不能用formal结果冒充Dev门槛。
  证据：optimization_evidence/comparison_summary.json；optimization_evidence/训练证据说明.md；expert_evidence/release-full-validation/oracle/result.json；expert_evidence/release-full-validation/oracle/raw-stage-times.json；Task/environment/public_eval/full_decoder_evaluator.py:21
  最小修改：为作者Reference补当前公开Dev协议原始反馈并与同一源码/图artifact绑定；保留已交formal数值，目标卡影响只作后补。
  验收材料：同Reference artifact的Dev result/log、profile=dev、三公开seed、质量门及归一化复算；不接受改写formal profile。

## 两轨时长、方法演进与格式
GPT102轮（45条COMPLETED、29条ok、10质量失败、9failed、7not_run、2incomplete）；Seed59轮（23ok、10failed、24not_run、2incomplete）。GPT最终115图：107节点、67prefix、四编译region，主要改完整post-softmax执行分区；Seed最终91图：65节点、23prefix、flattened 2D image MLP。两轨经历切片数值失败、保持形状复用、布局/分区、编译区域探索，已有原反馈与最终完整复验。
GPT保守160个不重叠区间合计25828.664409399033秒（7.174629002611h）；Seed363个区间27860.4020447731秒（7.739000567993h）。更宽观察合计为27098.01613020897秒与31740.479091826594秒；practical estimate 36823.34645545215/36053.12410325161秒不能改标为净实测。canonical_effective_seconds两者均null。
无训练、闭环反馈中位153.186/168.496秒支持短迭代；≥3真实方法闭环已有候选材料。完整7h资格需要研究区间分类、后续方向及best恢复/复验逐轨闭环；本次QA16为manual，未认定已少于7h，也未声称通过默认10h。
主轨迹GPT48–102多于八字段；45成功status为COMPLETED；GPT102条和Seed59条time都不符YYYY-MM-DD HH:mm:ss。多条method_summary仅实验名，另一些整段原作者报告；需规范摘要，原文保sidecar。两轨最终方法均保存，但尚需明确唯一最终best_method和恢复位置。
布局Task/有workspace/README.md映射不单独扣错；重复历史自述建议归历史索引；expert_annotation应精简；多交原始证据不因数量判错。

## 核心结论、Harbor与运行边界
结论：NOT READY；静态内容复核FAIL。明确核心失败为QA05、QA17/H03及QA18；G03/QA03、QA16仍有核心待核。其余项目的通过仅为静态实现与已有材料支持。
Docker COPY源和绝对路径可对齐；Agent context=Task/environment、Verifier context=Task/tests；artifacts含program.json/method.py；Hidden分块在独立Verifier拼接且hash校验；候选只有受限JSON。实际adapter另起broker容器、Compose覆盖和远端A100已有controller，当前单sandbox单容器/自身GPU不成立。
目标版本包内锁定Harbor0.23.0；历史NOP config/result、separate、reward与日志一致且有Task48绑定。本次未执行Docker、Verifier、任何提交代码或权重加载；不认证目标平台原生schema、动态资源/构建、12h稳定性或外部提交。

Harbor六项：

| 项目 | 状态 | 事实/证据及修复 |
|---|---|---|
| H01 | pass | 实际任务根Task/，由workspace/README映射；两Dockerfile和test.sh存在，Hidden三块非空并由prepare_hidden校验拼接，grade核SHA。；证据：workspace/README.md:5、Task/task.toml、Task/tests/prepare_hidden.py:2、Task/tests/grade.py:24 |
| H02 | pass | 包内目标Harbor0.23.0，TOML显式separate和顶层artifacts；自定义provider有实现及历史Trial。此为包内静态支持，不声称官方schema已加载或当前平台接受。；证据：Runtime/requirements.txt、Task/task.toml:2、Task/task.toml:33、expert_evidence/current-package-nop/trial/config.json |
| H03 | fail | Docker COPY上下文和提交路径可对齐，但实际adapter额外docker run broker、生成Compose覆盖，并依赖远端A100/已有controller；Agent/Verifier均0GPU，违当前单sandbox单容器及自身GPU契约。；证据：Task/environment/Dockerfile:3、Task/tests/Dockerfile:3、Runtime/hybrid_harbor_environment.py:107、Runtime/hybrid_harbor_environment.py:137、Runtime/hybrid_harbor_environment.py:166、Task/task.toml:25、Task/task.toml:40；修改：将数值执行依赖装入各自sandbox镜像并直接使用分配的H20/L20；删除额外broker/远端worker启动依赖及任务自定义Compose覆盖；同步公开Dev与正式Verifier路径。；验收：目标provider配置、完整构建依赖/模型资产入口、实际单容器及自身GPU分配记录、公开Dev与独立Verifier候选移交/reward链路；换卡成绩/耗时可后补。 |
| H04 | pass | test.sh调用grade；预清reward，有限标量写pending后replace到标准reward.txt；校验artifact/profile/输入hash/退出码。基础设施错误写失败信息并抛异常，不能视作有效质量失败成绩。；证据：Task/tests/test.sh:3、Task/tests/grade.py:12、Task/tests/grade.py:34、Task/tests/grade.py:53 |
| H05 | pass | 已有NOP及选中候选Trial选择Task、custom Docker provider，Verifier未禁用；selected使用冻结artifact Agent，未冒充研究模型轨迹。配置自洽只支持历史adapter，当前拓扑问题见H03。；证据：expert_evidence/current-package-nop/trial/config.json、expert_evidence/current-package-nop/trial/result.json#config.verifier、expert_evidence/release-full-validation/gpt-selected/trial/config.json |
| H06 | pass | 已有当前Task48绑定NOP：config/result均nop、separate、无exception、reward -0.000337629593978607与文件一致；非空stdout及artifact移交记录。仅证明其历史自定义环境链路，非本次实跑或当前平台验收。；证据：expert_evidence/current-package-nop/trial/config.json、expert_evidence/current-package-nop/trial/result.json、expert_evidence/current-package-nop/trial/verifier/reward.txt、expert_evidence/current-package-nop/trial/verifier/test-stdout.txt、expert_evidence/current-package-nop/trial_task_binding.json |

## QA实现检查

| 项目 | 状态 | 事实/证据 | 最小修改与验收材料 |
|---|---|---|---|
| QA01 | pass | 题面八部分齐全；目标模型身份是执行任务对象，未给作者Reference图/分数或完整论文标题/解法仓库。；Task/instruction.md:1、Task/instruction.md:42 | 无需核心修改；本次未动态复跑。 |
| QA02 | pass | 有效结果(B-T)/B连续、随延迟下降单调、不clip，U=0ms为数学下界；Baseline语义锚点0，NOP真实噪声保留。；Task/instruction.md:13、Task/environment/public_eval/full_decoder_evaluator.py:212 | 无需核心修改；本次未动态复跑。 |
| QA03 | manual | Reference代码、入口及三seed实际formal结果/非空CPU审计日志齐，formal归一化合格；GPU原stdout未附，且当前作者Reference Public/Dev原始成绩未核验。；Task/solution/solve.sh、Task/solution/reference_method.py、optimization_evidence/训练证据说明.md:5、optimization_evidence/comparison_summary.json、optimization_evidence/训练证据说明.md、expert_evidence/release-full-validation/oracle/result.json、expert_evidence/release-full-validation/oracle/raw-stage-times.json | 修改：补同Reference artifact的当前Dev反馈；能恢复原GPU stdout则按原样补，不能恢复明确缺失并保留现有stage/Trial/CPU证据。；验收：Dev profile及源码/图绑定、三seed原始反馈/质量/归一化；日志来源须保持CPU审计与GPU worker区分。 |
| QA04 | pass | 延迟与采样评估有波动；正式3seed全集复算n−1 σB180.57194646667463ms，Δ24703.31286112468ms，≥3σ且≥5σ。；optimization_evidence/comparison_summary.json、optimization_evidence/训练证据说明.md、expert_evidence/release-full-validation/oracle/result.json、expert_evidence/release-full-validation/oracle/raw-stage-times.json | 无需核心修改；本次未动态复跑。 |
| QA05 | fail | 算子/图/质量/时间guard存在，但当前配置声明storage_mb、no-network；实际求解依赖额外broker及远端GPU，V4内场约束不匹配。；Task/task.toml:18、Task/task.toml:24、Task/task.toml:39、Task/instruction.md:27、Runtime/hybrid_harbor_environment.py:137 | 修改：删除平台不支持的storage_mb/禁网配置，按实际联网能力改题面；把GPU计算迁入sandbox自身，去除额外工作容器/远端计算依赖。；验收：最终TOML/题面与目标provider支持字段一致；单容器/自身GPU的可审配置和实际链路材料。 |
| QA06 | pass | 单命令bash /tests/test.sh、标准reward写入、finite检查和错误抛出链路可识别；原生Trial记录异常与reward分开。；Task/tests/test.sh、Task/tests/grade.py、expert_evidence/current-package-nop/trial/result.json | 无需核心修改；本次未动态复跑。 |
| QA07 | pass | 只审instruction：只披露Hidden数量/协议，未给样本embedding、标签或可恢复答案。；Task/instruction.md:7 | 无需核心修改；本次未动态复跑。 |
| QA08 | pass | 只审instruction：没有作者Reference代码/具体图/成绩；公开操作目录和允许方向属于接口。；Task/instruction.md:15、Task/instruction.md:34 | 无需核心修改；本次未动态复跑。 |
| QA09 | pass | 清理预写reward及错误文件，并由可信grade pending.replace原子提交最终标量。；Task/tests/test.sh:3、Task/tests/grade.py:12、Task/tests/grade.py:21 | 无需核心修改；本次未动态复跑。 |
| QA10 | pass | Agent镜像受read_only策略；Verifier仅解析受限JSON不导入builder，固定Hidden SHA及提交artifact绑定，候选不能指定路径/命令/后端。静态实现不认证全部物理隔离。；Runtime/hybrid_harbor_environment.py:95、Task/tests/grade.py:24、Task/tests/queue_client.py:13、Task/environment/public_eval/tensor_dag.py:26 | 无需核心修改；本次未动态复跑。 |
| QA11 | pass | 目标/环境/指标/开放范围/硬边界/提交/流程/完成条件均有实质内容；联网边界的当前平台不适配另由QA05处理。；Task/instruction.md:1、Task/instruction.md:5、Task/instruction.md:11、Task/instruction.md:15、Task/instruction.md:23、Task/instruction.md:32、Task/instruction.md:36、Task/instruction.md:42 | 无需核心修改；本次未动态复跑。 |
| QA12 | pass | ZIP成员清点无.git目录；不声称作者原仓库全部历史已清。；@inventory | 无需核心修改；本次未动态复跑。 |
| QA13 | pass | 三个正式采样seed两侧同集合、六份result对应同job逐seed值与原stage复算一致；均正改善且质量门通过。非训练任务不造model/reload。；optimization_evidence/comparison_summary.json、optimization_evidence/训练证据说明.md、expert_evidence/release-full-validation/oracle/result.json、expert_evidence/release-full-validation/oracle/raw-stage-times.json、optimization_evidence/baseline_runs/seed_202/result.json、optimization_evidence/reference_runs/seed_303/result.json | 无需核心修改；本次未动态复跑。 |
| QA14 | pass | 图结构/中间复用/算子组合/编译区域均可变化；文本prefix复用之外，可优化图消活与布局、区域边界成本，题面未给具体实现；作者R0.42277未饱和。；Task/environment/public_eval/tensor_dag.py、Task/environment/public_eval/bounded_graph_compiler.py、Task/solution/reference_method.py、Trajectories/GPT/selected-method/program.json | 无需核心修改；本次未动态复跑。 |
| QA15 | manual（非阻断） | 四个历史A100完整数值job分别2466.97/1881.43/1431.89/1452.28秒，均<2h；当前没有H20/L20实测。仅换卡耗时待核非阻断，拓扑不适配由核心项处理。；expert_evidence/release-full-validation/oracle/result.json、expert_evidence/release-full-validation/nop/result.json、expert_evidence/release-full-validation/gpt-selected/result.json、expert_evidence/release-full-validation/seed-selected/result.json | 修改：保留A100历史设备与真实计时范围，后补目标H20/L20估算或实测；不为此重写旧成绩或强制重跑。；验收：目标资源规划及有来源的耗时估算/后续实际记录；注明当前未核验。 |
| QA16 | manual | 保守不重叠区间复算GPT25828.664409399033秒、Seed27860.4020447731秒，均7–10h；默认10h未证。无训练且反馈中位数153/168秒支持短轮次，但未完成完整7h例外资格与净研究区间分类闭环。；expert_evidence/run_summary.json#agent_execution、expert_evidence/original-source-method-loops/conservative-clock-source-accounting.json、expert_evidence/recovered-original-duration/gpt-duration-reconstruction.json、expert_evidence/recovered-original-duration/seed-duration-reconstruction.json、expert_evidence/original-source-method-loops/gpt-actual-method-feedback-windows.json、expert_evidence/original-source-method-loops/seed-actual-method-feedback-windows.json | 修改：优先整理已有起止/扣除区间，核实保守区间的研究活动属性；明确采用默认10h或完整7h例外，并对每轨给≥3方法→Dev→决策、后续合法方向及干净最佳复验映射。不要按墙钟填时长，也无需盲目补跑。；验收：两轨各自净时长/并集合并及排除依据；若采用7h，完整非训练/短迭代例外说明、至少3真实闭环、后续方向和选中方法干净复验来源。包内历史用户放宽文本不是本次新授权。 |
| QA17 | fail | separate、镜像COPY、Hidden、标准reward和历史NOP可追溯；H03明确额外broker/远端GPU及Compose依赖与当前运行契约冲突，历史成功不消除该核心失败。；Task/task.toml:33、Runtime/hybrid_harbor_environment.py:107、Runtime/hybrid_harbor_environment.py:137、Runtime/hybrid_harbor_environment.py:166 | 修改：按H03迁移单容器自身GPU执行，保持公开Dev、私有Hidden、仅artifact移交和可信reward。；验收：目标平台可审配置/镜像依赖/候选移交/独立Verifier出分材料；NOP仅待补不单独阻断。 |
| QA18 | fail | GPT102轮、Seed59轮均有演进与原反馈；GPT48–102轮多出字段且45轮成功status=COMPLETED；两轨全部time不符指定秒级格式，若干摘要只有实验名。unique最终best_method选择映射也未闭环。；expert_evidence/trajectory_codex.json#rounds[47:102]、expert_evidence/trajectory_seed.json#rounds、expert_evidence/expert_annotation.json#current_selected_methods | 修改：保留原记录作sidecar；规范主轨迹恰八字段、成功ok、失败真实分类/null规则、统一YYYY-MM-DD HH:mm:ss并说明时区，补1–3句生效改动/结果摘要；披露保留标记含义并指定唯一最终方法/恢复位置。；验收：两条可解析八字段主JSON与原结果逐轮映射、分数不改、时间原始值到规范值映射、唯一最终源码/入口/必要配置和恢复复测记录。 |
| QA19 | pass | 题面给101/202/303采样seed和有效改善≥3σB，说明样本标准差n−1；无固定训练seed即确定性误述。；Task/instruction.md:7、Task/instruction.md:13 | 无需核心修改；本次未动态复跑。 |
| QA20 | pass | 题面列可改范围、网络规则及shell/Python/Dev工具策略；网络策略本身与内场不匹配已由QA05判退。；Task/instruction.md:15、Task/instruction.md:25、Task/instruction.md:27 | 无需核心修改；本次未动态复跑。 |
| QA21 | manual（非阻断） | run_summary无task_binding.hash_profile/task_content_sha256；另有Task48 manifest及历史原生checksum，不能冒充官方四项指纹。本项单独非阻断待补。；expert_evidence/run_summary.json、dependencies/task/task-files.json、expert_evidence/current-package-nop/trial/result.json#task_checksum | 修改：获得真实发布task_fingerprint.py后按四项完整范围生成，报告放范围之外；保留历史值与版本，不猜哈希算法。；验收：真实脚本版本/哈希及四项指纹、run_summary.task_binding；仅此缺口不要求全量重跑。 |

## 当前五类放宽待补
指纹/哈希及缺字段：非阻断。G02/G03换卡影响、QA03/QA13仅由换卡产生的成绩重算、QA15目标卡耗时：可延期，历史A100不改标。本包有完整可读NOP，NOP缺失/关联待核本来也不单独判退。放宽不覆盖额外容器/远端GPU的明确部署结构冲突，也不覆盖Public/Dev证据及主轨迹格式。

## 工具缺口与可见隐私边界
真实官方0.3.10发布工具及独立格式/GPU/指纹脚本未提供，完整工具验收待补；没有改版本字符串、调用旧收集器或冒充官方通过。完整ZIP逐件二进制隐私扫描交主控；本报告仅源码/文本可见面，没有读取凭据，没有外发。

## 具体风险（至多3条）
- 原GPU worker stdout在当前ZIP缺失，CPU审计stdout明确重复，不应被描述为逐seed GPU原日志。
- 保守活动下界与10h估计不同；包装中旧用户计时口径不是本次审查权限来源。
- Runtime/hybrid_relay.py:24为旧Seed非Coding Plan路由；后续模型调用需依当前授权路由迁移，历史证据保持原样。

## 可复制给专家的退回说明
核心结论：NOT READY（静态内容复核FAIL；另有核心未核验项）。
1. 当前执行拓扑必须改。Task/task.toml:18/24/39声明禁网和storage_mb，Agent/Verifier均0GPU；Runtime/hybrid_harbor_environment.py:107/137/166生成Compose覆盖、另启broker并依赖远端A100。按当前V4改为各sandbox单容器直接使用自身H20/L20，删除额外工作容器/远端计算及不支持的配置，保持Dev与Hidden边界。重交目标provider配置、完整依赖、单容器/自身GPU及候选移交/reward链路材料。换卡成绩与耗时可后补，不要求为此重做已交正式seed。
2. 两轨时长须据已有材料闭环。保守区间复算GPT25828.664409399033秒、Seed27860.4020447731秒；包内canonical为null。默认10h未证，不能把估计10h写实测。可优先完整说明7h例外：无训练、短轮次依据、每轨≥3真实方法→Dev→决策、持续演进/后续方向、干净最佳复验及排队/下载/故障区间扣除，不要求仅因未知总时长就盲目延长运行。
3. 规范轨迹。expert_evidence/trajectory_codex.json#rounds[47:102]含额外字段、成功COMPLETED和纯实验名摘要；两轨时间均非指定秒级格式。原记录保持不变另存sidecar，主轨迹恰八字段、成功ok、失败/null规则正确、统一秒级时间及声明时区，补生效方法/结果摘要与原反馈映射。指定唯一最终best_method及真实恢复位置，不只保留两套候选路径。
4. 作者Reference正式结果很好，但缺当前公开Dev资格证据。optimization_evidence/及oracle/result.json均为formal。补同Reference图artifact的Dev profile原始结果/log和归一化/质量门，不把formal标签改成Dev。原GPU stdout缺失已披露；能恢复则原样补，不能恢复明确缺失并保留原4608stage、18重复、native Trial及CPU审计，不生成替代日志。
正式B/R无需因分差大而削弱或换SOTA：B58432.48055013021ms、R33729.16768900553ms、σB180.57194646667463ms、改善24703.31286112468ms、归一化0.422766800733905；18×256阶段复算一致、质量误差0。三个seed是采样seed，六份result来自同一成对推理job，并非独立训练。
非阻断待补：官方四项指纹与run_summary.task_binding缺失；A100到H20/L20的成绩影响/锚点与计时补证；目标换卡重算可延期。已有历史NOP可读且一致，不要求因NOP额外开跑。真实0.3.10格式/GPU/指纹工具未提供，完整工具验收待补；本报告不是该官方工具输出，也不是平台通过。
额外路由提醒：Runtime/hybrid_relay.py:24仍保留非Coding Plan的旧Seed路由。若继续执行涉及豆包的工作，须在可信侧按用户当前共享配置切至Coding Plan Chat Completions并保持指定模型；历史证据不改标，本次未读凭据也未调用模型。

