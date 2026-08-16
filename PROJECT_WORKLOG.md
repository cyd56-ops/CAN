# 1. Project goal and non-goals

## Current goal

项目名称为 `CAN: Lattice-Based Neural Model Access Control`。目标是研究如何把格密码验证关系编译为固定的定点/量化神经网络，并用其结果对 CNN/DNN 模型能力实施 fail-closed 访问控制；后续再扩展到 MoE/多智能体中的认证证据、授权策略、短期 capability 和工具网关。

项目面向科研论文发表和可复现实验。阶段 A 是当前主线，首先区分并逐步实现 toy LWE 数值解锁、
challenge-response 请求绑定/新鲜性、格身份协议和后续标准公钥格签名验证；阶段 B 是后续系统扩展。

长期研究路线固定为 `V0 -> V1-prep -> V1 -> V2`，并与既有工程编号并行而不重命名现有模块。
各路线必须以独立模块、协议标识、registry、adapter 和测试持续保留：V1 不得通过重命名、改写或
替换 V0/A0 代码实现，V2 也不得覆盖 V0 或 V1；跨路线只允许复用明确无协议语义的通用 helper，且
必须保持入口和接受集合隔离。具体映射如下：
V0 对应已闭合的 A0/A1/A2 toy LWE 数值解锁、神经等价和模型硬门控；V1-prep 对应 A3 请求绑定/
新鲜性壳与 A4 canonical `(y,z)` 公开格关系编译，目的只是隔离并闭合神经代数内核；V1-P1 已记录
普通矩阵 SIS Sigma baseline；当前 V1-P2 主路线已选择 Boudgoust--Takahashi `FSwA-S` 的底层
交互式 Module-SIS Sigma protocol，使用 `R_q`、`Abar=[A|I]`、`t=Abar*s`、`u=Abar*y`、
`z=y+c*s` 和 exact `Abar*z=u+c*t`；V2 最后以标准 ML-DSA verifier 作为标准兼容和工程比较
目标。V1 与 V2 的业务模型实验都选择 CIFAR-100 与 CIFAR-style ResNet-18；V1 使用独立
`CAN-V1-CIFAR100-RESNET18-v1` model/input profile，未来 V2 必须使用 V2-local 协议 adapter、
registry 和入口绑定同一业务 benchmark，不能复用 V1 的认证接受入口。已验收的
Fashion-MNIST/MLP（以及 V0 路线内可能的 LeNet 对照）继续只属于 V0/V1-prep 小型回归和开销
对照，不删除、不改写，也不作为 V1/V2 认证失败时的回退。V1-prep 不是身份认证，V2 也不进入
首篇论文 MVP。

阶段 A0 精确协议、Python 技术 bootstrap、精确整数 oracle、A1 构造无关数值/算子规格、首个
固定 ReLU 构造决定、dependency-free exact-integer conformance backend 和 A1-B1 PyTorch
CPU exact-integer backend 已完成并通过全域与防御性安全测试；A2-E1 已核验固定 CPU ML 环境和
Fashion-MNIST 数据，实现严格输入校验、MLP 与确定性无门控 baseline，两次十 epoch 运行均得到
`88.08%` test accuracy 和相同预测/模型摘要。A2-E1 单一协调器、固定响应和二元前置硬门控现已
实现；gate 重训保持 `88.08%`，10,000 个 allow 标签全部匹配 baseline，所测拒绝路径均为零
protected-model calls。A2-E2 能力分级实验规格现已固定独立 public coarse model、默认关闭的可信
entry binding、单一协调器三态语义和隔离验收矩阵；独立 public model 无门控 baseline 与三态
协调器现已实现。独立 trusted materializer 已按 D-024 的固定协议重建两个 accepted state，严格
校验本地 manifest、文件摘要与 canonical state digest 后，由 no-training evaluator 完成真实
10,000-image 三态报告；两个预测摘要均与 baseline 一致，完整调用计数和三路 latency 已记录。
A2-E2 现已闭合；A3-v1 challenge-response/request-binding 规格和默认关闭的单进程运行时协议壳均已
完成，固定 133 字节 canonical message、规范输入摘要、60 秒 challenge、可信 nonce 生命周期、原子
consume、evidence-only verifier 边界和安全游戏。A4 已选择 GPV PFDH 短原像公开验证关系，冻结
非生产 `A4-GPV-PFDH-TOY-v1` 的 105 字节 proof、公开 profile、hash-to-syndrome、exact relation 和
神经证明义务，并实现无私钥 reference 与 A3 evidence adapter。A4-C1 已进一步冻结并实现
`CAN-RELU-A4-PFDH-TOY-v1` dependency-free exact graph、全输入 `V_nn==V_ref` 证明和 A3 neural
adapter，V1-prep 因此闭合。V1-P1 的矩阵 SIS 决策保留为历史 baseline；V1-P2 的商环、
key/transcript、bounded-uniform rejection、canonical polynomial encoding、安全游戏、A3-v2 绑定和
neural relation 现已冻结。非生产 V1-P2 conformance profile、canonical polynomial parser、公开
registry、coefficient-domain exact reference、A3-v2 commit-first 单次终态协调器和 evidence adapter
已实现并通过差分、集成和防御性安全测试；V1-M1 也已冻结 CIFAR-100/CIFAR-style ResNet-18
headline model 路线，但尚未下载或训练。独立非生产 `V1-P2-PSR-E1` prover/sampler/rejection
实验契约现已冻结，并已实现临时 generated-key fixture、确定性 SHAKE256 sampler、commit-first
single attempt、A3-v2 fresh-transcript retry/exhaustion harness、公开 vector manifest 和 exact
differential。`V1-C1-MSIS` 已实现固定 dependency-free coefficient-domain graph、全 canonical input
`V_nn==V_ref` 证明、独立 valid/tamper differential 和 A3-v2 neural evidence route；accept 最多一次
protected call，relation reject 与 foreign route 均为零 protected calls。V1-M1 GPU/software tuple、
CIFAR-100 data/training protocol 及 isolated model/archive parser/adapter/baseline runner 已冻结并通过本机
unit/security tests；尚未下载数据或训练。当前唯一下一步是 `SERVER_REQUIRED` 的首次正式 archive 下载与
两次 CIFAR-100/ResNet-18 baseline；不开始 V2 或 Stage B。

## Success conditions

- 为选定格验证关系定义规范输入域、精确整数 oracle 和神经编译语义。
- 证明或明确限定 completeness、`V_nn = 1 -> V_ref = 1` 的 soundness-preservation，以及逐层数值误差预算。
- 验证失败、畸形输入、replay 和 tamper 产生零受保护模型/工具副作用。
- 保留只属于 V0/V1-prep 的 Fashion-MNIST/MLP 小型回归基线，并在
  CIFAR-100/CIFAR-style ResNet-18 上建立 V1 主实验及后续 V2 标准 verifier 对照的可复现准确率、
  预测摘要、延迟、认证开销和 fail-closed 安全测试结果。
- 论文主张与已有 DNN 密码实现和格签名神经网络工作相比具有明确、经文献检索支持的差异。
- 后续每个 checkpoint 同步 Git 状态、准确命令、结果、风险和唯一下一步。

## Explicit non-goals

- 首版不实现完整 ML-DSA 神经验证，也不把它作为 MVP 依赖。
- A0 toy LWE 解密不称为数字签名、身份认证或已证明不可伪造的 token。
- 首篇论文不解决白盒权重读取/修改、删层、TEE、安全启动、远程证明或完整侧信道。
- 阶段 A 未闭合前不把阶段 B 的 MoE 系统扩展为当前实现范围。
- 不实现攻击性后门或未授权访问利用；参考后门论文仅用于防御性边界分析。
- 不把 toy、单机、小参数、有限测试或实验性结果描述为生产保证。

# 2. Architecture and security invariants

## Current architecture

当前实现了非生产 A0 精确 reference、A1 dependency-free exact-integer conformance verifier、
A1-B1 PyTorch CPU exact-integer backend，以及 A2-E1 Fashion-MNIST MLP、无门控 baseline、单一
协调器与二元前置硬门控，以及 A2-E2 独立 public MLP/baseline、本地三态协调器、trusted
materializer 和已完成的 accepted-state 三态报告；A3-v1 默认关闭的 codec/parser、trusted nonce
store 和 coordinator 协议壳已实现；A4 GPV toy exact reference、公开 profile/proof parser 与 A3
exact/neural adapters 及 A4-C1 固定 neural verifier 已实现。当前 V1-P2 FSwA-S Module-SIS 协议
已经选定；非生产 concrete profile、exact reference、A3-v2、adapter 和 `V1-P2-PSR-E1`
generated-key/sampler/single-attempt/retry experiment 已实现；生产 prover、密码安全参数和 neural
verifier 尚未实现。V1 主业务模型路线已经冻结为独立
CIFAR-100/CIFAR-style ResNet-18 profile，但其数据、模型、训练环境、accepted artifact、A3-v2
input adapter 和门控实验尚未实现；现有 Fashion-MNIST/MLP 路线保持已验收状态。
项目与计划模块边界为：

- `AGENTS.md`：长期稳定的工作、工程、安全、测试和 Git 约束。
- `PROJECT_WORKLOG.md`：当前动态事实、状态、任务、决定和 checkpoint 记录。
- `docs/RESEARCH_DESIGN.md`：研究问题、术语、阶段路线、形式化义务、实验和论文定位。
- `docs/A0_PROTOCOL_SPEC.md`：A0-v1 的 toy profile、23 字节编码、精确 relation、oracle 伪代码、误差契约和攻击边界。
- `docs/A1_NUMERICAL_SPEC.md`：A1-v1 的 credential 隔离、可信 profile 编译、tensor/算子语义、误差预算、证明义务和延期假设。
- `docs/A1_CONSTRUCTION_DECISION.md`：A1-C1 的固定整数 ReLU 主构造、exact-ops 对照、全域证明、参数范围和实现契约。
- `docs/A1_BACKEND_DECISION.md`：A1-B1 的 Linux x86_64 PyTorch CPU exact-integer 物理映射、安装渠道、复测和禁用契约。
- `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`：A2-E1 的 Fashion-MNIST/MLP baseline、确定性训练、指标、artifact 和二元硬门控契约。
- `docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`：A2-E2 的独立 public coarse model、可信 entry binding、三态提交语义和隔离验收矩阵。
- `docs/A3_CHALLENGE_RESPONSE_SPEC.md`：A3-v1 canonical message/input digest、challenge/nonce 生命周期、原子 consume、安全游戏和验收矩阵。
- `docs/A4_GPV_RELATION_SPEC.md`：A4 GPV-PFDH toy 公开 profile、105 字节 proof、exact relation 和神经证明义务。
- `docs/A4_NEURAL_CONSTRUCTION_DECISION.md`：A4-C1 point-pulse ReLU graph、范围账本、证明和实现契约。
- `docs/V1_PROTOCOL_SELECTION_DECISION.md`：V1-P1 reviewed 矩阵 SIS 身份协议、commit-first
  transcript、canonical 编码、安全游戏和路线隔离的历史 baseline。
- `docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`：当前 V1-P2 FSwA-S Module-SIS Sigma、商环、
  polynomial transcript/encoding、M-LWE/M-SIS 安全边界和 neural construction 契约。
- `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md`：V1-P2 非生产 generated-key、deterministic sampler、
  emit/abort、fresh retry、secret lifecycle、测试向量和统计指标契约。
- `docs/V1_MODEL_EXPERIMENT_DECISION.md`：V1-M1 CIFAR-100/CIFAR-style ResNet-18 input、模型、
  reproducibility、gate 和 artifact 边界。
- `SECURITY.md`：信任模型、受保护资产、输入验证、密钥/replay 生命周期和明确不保证的性质。
- `README.md`、`pyproject.toml`、`requirements-dev.lock`、`requirements-ml.lock`：项目入口、
  Python 3.11 配置、精确直接依赖及开发/ML 环境解析锁。
- `src/can/reference/`：已实现 A0-v1 oracle、A4 public relation，以及 V1-P2 公开
  profile/parser/registry 和 coefficient-domain exact relation。
- `src/can/verifier/`：已实现 A1-C1 不可变 compiled profile/registry、固定三层 affine/ReLU graph、
  dependency-free 与 A1-B1 PyTorch CPU exact-integer backends，以及 A4-C1 dependency-free sparse
  exact graph 和 evidence-only adapters。
- `src/can/model/`：已实现 A2-E1 protected MLP、A2-E2 独立 public MLP，以及 V1-M1 独立
  CIFAR-style ResNet-18。
- `src/can/access/`：已实现 A2-E1/A2-E2 协调器、A3-v1 协议壳、A4 evidence adapter、V1-P2
  A3-v2 commit-first 单次终态状态机和 exact evidence adapter，以及 V1-M1 raw CIFAR adapter/route。
- `src/can/experiments/`：已实现 A2-E1/A2-E2 数据核验、确定性训练/评估、gate 标签等价、trusted
  state materialization/加载、三态只评估、latency 和报告入口，以及 V1-P2 generated-key、SHAKE256
  sampler、single-attempt prover、无 secret manifest 和 V1-M1 无下载 baseline runner。
- `tests/`：已有 A0/A1、A2、A3、A4、V1-P2/A3-v2、V1-P2-PSR-E1 和 V1-M1 的单元、差分、集成和
  防御性安全测试。
- `scripts/check_governance_docs.sh`：对必需文档、bootstrap 文件、唯一下一步和任务状态枚举执行确定性检查。
- `paper/`：两篇本地相关工作，仅作为研究参考资料。

包边界已经创建；A0 reference、A1 两个 verifier backends、A2-E1 MLP、无门控与 gate 实验入口、
访问协调器和硬门控，以及 A2-E2 独立 public MLP/baseline、三态协调器、trusted materializer、已验收
权重的三态报告、A3-v1 默认关闭协议壳、A4 toy exact adapter 和 A4-C1 dependency-free neural
verifier 均已实现；V1-P2 exact/A3-v2 implementation package 也已闭合，但 prover/sampler、neural
verifier、生产安全参数与 qint8/CUDA/export 仍未完成；非生产 generated-key/sampler/single-attempt
experiment 与 A3-v2 fresh retry 已闭合。
V0/A0 与 A4/V1-prep 代码必须原样保留为
独立复现路线。

## Trust boundaries and data flow

阶段 A 计划数据流：

```text
不可信业务输入和凭据字节
-> 规范化解析与精确类型检查
-> 本地可信 profile
-> 确定性神经验证器
-> 无授权能力的结构化证据
-> 唯一访问协调器
-> 已提交的 deny 或内部 protected-model decision
-> allow 后调用受保护 CNN/DNN
```

阶段 B 在协调器之后加入短期 capability、Router/专家和最终工具网关。Router、LLM 和业务专家均不可信；只有协调器提交权限，只有网关执行受保护工具副作用。

阶段 A 的能力分级已先闭合 `DENY`/protected model 二元硬门控，并选择独立
`784->64->2` public MLP 作为 A2-E2 主路线。public entry 默认关闭并由本地可信部署配置绑定，
public 功能/输出与 protected path 独立且不可升级；protected 验证失败固定 deny，不能把 public
作为更弱验证路线。独立 head、共享 trunk、浅层/深层和 MASK 只保留为后续比较。

## Invariants currently established

- A0 是非生产数值解锁实验，不具有签名或身份认证语义。
- A0-v1 固定 `n=32`、`m=8`、`q=257`、参考半径 12、神经阈值 8 和目标误差上界 4；请求只携带 slot 和 `b`，不携带 `A` 或安全参数。
- A0 parser 只接受精确 `bytes` 和 23 字节唯一编码；非规范 `b`、未知版本/profile/slot、错误长度和类型混淆均返回结构化拒绝证据。
- A0 slot/registry 在构造时拒绝错误 shape/range、bool 系数、全零行、重复 slot 和错误 entry 类型，加载后保持不可变且无 profile/slot 回退。
- `V_ref` 使用有界精确整数语义，只返回无 gate/capability 的证据；issuer-core、reference-guard 和 reject 由最大逐分量距离精确区分。
- credential 与业务输入 `x` 使用独立 schema、tensor 和调用边界；任何候选 “Secret Trigger” 术语在形成后续决定前只指显式 credential，不能指隐藏业务输入模式。
- A1 共同 profile 固定 credential tensor 为 shape `(8,)`、`int32`、scale `1` 和范围 `[0,256]`；本地编译器以 A0 精确 `int64` 语义生成规范相位锚点，运行时共同语义按线性残差、规范模距离、阈值 8、八路 AND 和 evidence 流水线组织。
- A1 modulo 不连续边界必须逐分支证明、穷尽完整有限残差域或由形式方法直接证明；未经证明的普通 Lipschitz 传播和有限随机差分不能支持全域 soundness。
- A1-C1 主构造固定为 `CAN-RELU-EXACT-v1`：`8->40->16->1` 三个 affine+ReLU blocks，五个 ReLU/分量计算 exact modular distance，两个 ReLU/分量计算整数阈值，一个 ReLU 计算八路 AND，语义误差为 `0`。
- A1-C1 把本地规范锚点 `t` 折叠进第一层 bias；主 core 禁止 `%`、Floor、`abs`、比较、Sigmoid、MASK 和任何 `V_ref`/exact-ops fallback，普通整数路线只作互斥测试基线。
- A1-C1 dependency-free backend 已按固定 graph 实现；unit/differential/security tests 穷尽 513 个残差、129 个距离、9 个 AND 和值、每个分量全部 `b_i=0..256` 和 A0 向量族，得到零 false accept 与 issuer-core 零 false reject。
- A1 evidence 只含稳定结果码；compiled profile/registry 全部不可变，主 adapter 只接受原始 23 字节 credential，内部异常返回配置拒绝且 instrumentation 证明不调用 `V_ref` 或 exact-ops fallback。
- A1-B1 固定为 `CAN-TORCH-CPU-EXACT-v1`：Linux x86_64、CPython 3.11、官方 PyTorch
  `2.13.0+cpu` CPU wheel、`int32` weight/bias/activation、`int64` reduction、scale `1`、
  zero-point `0` 和 eager `mul/sum/add/clamp`；qint8、CUDA、export 和所有 runtime fallback 均不
  属于该路线。
- A1-B1 已实现 explicit optional-dependency module；startup gate 核验环境、buffer、range、
  operator micro-probe 和实际 profile 有限域分解，每次 raw adapter 调用复核环境与 module
  contract，失败禁用实例。完整差分得到零 false accept 与 issuer-core 零 false reject。
- A2-E1 固定为 `CAN-A2-FMNIST-MLP-v1`：Fashion-MNIST、`784->256->128->10` float32 CPU MLP、
  55,000/5,000 deterministic train/validation split、十 epoch Adam 和 test accuracy/latency baseline；
  `torchvision==0.28.0+cpu` 已从官方 CPU index 单独安装和核验。
- A2-E1 MLP 只接受 contiguous CPU float32 `(N,1,28,28)` finite `[0,1]` 输入，labels 只接受 CPU
  int64 `(N,)` 的 `[0,9]`；固定数据 hash/split 和两次十 epoch baseline 已复现为 `88.08%` test
  accuracy、相同 ordered predictions 和 canonical state hash。
- A2-E1 只允许 raw business input + raw credential -> A1 evidence -> 唯一协调器 -> protected MLP；
  请求方不能提交 evidence/decision/backend/policy，所有非 `NUMERIC_ACCEPT` 路径必须返回固定 deny
  envelope 且 protected-model 调用计数为零。该契约已由 unit/integration/security tests 和真实 gate
  report 核验；10,000 个 allow 请求各提交一次并调用一次模型，rejected probe 为零模型调用。
- A2 二元 protected-model gate 已闭合；A2-E2 规格固定独立 `784->64->2` public MLP 的
  footwear/non-footwear 输出、默认关闭且本地绑定的 public entry、同一协调器互斥
  `DENY`/`PUBLIC`/`PROTECTED`、version-2 envelopes、零 protected calls/features 和不可升级/
  重标记/复用边界。
- A2-E2 `CAN-A2-FMNIST-PUBLIC-MLP-v1` 已独立实现，不导入 protected model/baseline/gate；只接受
  contiguous CPU float32 `(N,1,28,28)` finite `[0,1]` 图像，source labels 只接受规范 Fashion-MNIST
  int64 `[0,9]` 并精确映射为 `NON_FOOTWEAR={0,1,2,3,4,6,8}` 与
  `FOOTWEAR={5,7,9}`。
- 两次固定 public 训练均得到 test loss `0.007989783663357957`、accuracy `99.85%`
  (`9985/10000`)、confusion `[[6989,11],[4,2996]]`、相同 prediction SHA
  `f54b2351...6f0a`、state SHA `b71980eb...122be` 和 fingerprint `e4fbf9c0...c14f`；该结果只是
  coarse 分类 baseline，不具有授权或安全语义。
- A2-E2 `A2CapabilityCoordinator` 已实现默认关闭的精确本地 policy、启动审计事件、互斥
  `DENY`/`PUBLIC`/`PROTECTED` 提交、固定 version-2 envelopes、独立 storage 检查和线程安全
  计数/计时。public 成功为零 verifier/protected calls，protected reject 为零 public/protected calls；
  请求字段不能选择路由/策略/模型/backend/evidence/decision，模型或 verifier 异常没有 fallback。
- A2-E2 trusted materializer 只在 ignored `artifacts/a2/local-states/` 保存两个 CPU float32
  `state_dict` 与 canonical manifest；加载边界校验固定文件名、协议/数据摘要、文件 SHA-256、
  dtype/device/layout/finiteness、拓扑和 canonical state digest。只评估入口不训练，已完成真实
  10,000-image 逐标签三态等价、完整计数和 latency 报告。
- 客户端不能选择 `A`、`q`、算法、阈值、噪声 profile 或量化 scale；所有安全参数来自本地可信 registry。
- 安全承载 verifier 只持有公开验证信息；真实私钥不进入模型、verifier、日志或 checkpoint。
- 所有外部输入先规范化并严格验证；未知/重复字段、类型混淆、非有限值、溢出和非规范编码默认拒绝。
- 最低神经 soundness 目标是对全部可表示输入满足 `V_nn(a) = 1 -> V_ref(canonical(a)) = 1`。
- 落入数值模糊区或证明覆盖范围之外的输入默认拒绝。
- 验证器只产生证据，唯一协调器提交权限；失败产生零受保护模型或工具副作用。
- replay 防护允许并要求使用模型外部的可信 nonce/challenge 状态。
- A3-v1 proof message 固定为 133 字节，绑定 domain/version、本地 `model_id=1`、32 字节
  `identity_id`、`scope_id=1`、服务端签发/到期时间、32 字节 nonce 和 canonical image SHA-256；
  challenge TTL 固定 60 秒，客户端不能提交时间、nonce、key、算法、profile、evidence 或 decision。
- A3-v1 canonical image 是 detached/cloned CPU contiguous float32 `(1,1,28,28)`，拒绝 non-finite、
  越界和 negative zero，并以固定 big-endian IEEE-754 编码计算摘要；未来模型必须使用同一快照。
- A3-v1 verifier 只产生绑定 exact message digest/identity 的 evidence；A0/A1 evidence 和 A2 public
  response 不能进入该路径，没有本地 A4 profile 时入口默认关闭。
- A3-v1 nonce store 只承诺单进程线程安全线性化语义；exact accept 后原子 `PENDING -> CONSUMED`
  并在临界区复核 binding/expiry，只有唯一成功者可进入 protected model，consume 后不回滚。
- A4 首个关系固定为 reviewed GPV PFDH 的非生产 conformance profile：`q=257`、`8 x 72` 公开矩阵、
  `||z||_inf <= 1`、32 字节 salt、105 字节 proof 和 SHAKE256 hash-to-syndrome；客户端不能提交 key、
  profile、矩阵、hash、norm bound 或 decision。
- A4 public profile 在构造时复制并校验 exact shape/range/mod-257 满行秩；reference 与 adapter 只持有
  公开信息并产生 evidence，不含 private key、trapdoor、signer、nonce 状态、模型或授权能力。
- A4-C1 固定为 `CAN-RELU-A4-PFDH-TOY-v1`：canonical input `y||z` 为 80 个 scale-1 `int32`，
  graph 为 `80->3600->1153->1` 三个 sparse affine+ReLU blocks，固定倍数集合 `K=-72..71`；第一层
  计算 norm violations 与 residual hinges，第二层计算 1152 个 exact point pulses 和 norm sum，最后
  以 `rho(sum(p)-v-7)` 输出 exact bit。
- A4-C1 使用 `int64` reduction 与已证明安全的 `int32` activation storage；完整 residual point-pulse
  标量域、全部 signed-int8 norm 标量域、canonical relation differential 和 A3 neural adapter 已测试，
  代数证明覆盖全部 canonical `(y,z)` 并给出 `V_nn==V_ref`，运行时不调用 reference fallback。
- A4 测试 gadget 矩阵使 valid proof 可公开构造，只支持 relation/neural conformance 与 A3 组合测试；
  当前仍不提供 GPV random-oracle/SIS 不可伪造性、生产参数或 V1 身份认证结论。
- V1-P2 conformance profile 固定 `N=8,q=257,k_mod=2,ell_mod=2,eta=1,gamma=8,kappa=2,B=6`；
  commitment/public residues 使用 canonical u32，challenge 使用 fixed-weight ternary i8，response 使用
  signed i32 编码并在 relation 中精确限制到 `[-6,6]`，系数域 oracle 在模约减前使用 Python exact int。
- `V1-P2-PSR-E1` 固定 32-byte toy seed、SHAKE256 role/counter stream、无偏 byte rejection、
  `s in {-1,0,1}^32`、`y in [-8,8]^32`、112 个 fixed-weight challenge、fresh-transcript retry 和
  generated-key fixture 隔离。理论单次 emit probability 为 `(13/17)^32`，约 `0.0001869914674`，
  期望约 5348 次尝试；这只解释 toy abort rate，不支持性能或安全主张。
- V1-P2 exact verifier 只持有公开 `Abar,t` 并产生稳定 evidence；A3-v2 独立绑定 verification/input
  profile、identity、model、scope、133-byte message、commitment、server challenge、transcript 和
  opaque snapshot。一个 parsed response attempt、abort 或 expiry 原子终结 transcript；并发 replay
  最多一次 verifier/allow/protected call，pre-commit reject 均为零 protected calls。
- V1-M1 固定独立 `CAN-V1-CIFAR100-RESNET18-v1` headline route 和 `(1,3,32,32)` uint8/RGB input
  contract；archive/source digest、训练环境、超参数、CIFAR-style ResNet-18、strict adapter 和无下载
  runner 已实现。数据、weights、正式 baseline/gate 与性能结果均尚未产生。
- MoE Router 和自然语言输出不能创建权限，工具网关必须用真实调用参数重新验证 capability。
- 未经代码、证明和测试支持，不声称任何密码或系统安全性质。

# 3. Current state

- **Current branch:** `main`；2026-08-12 已在 `/home/kali/CAN` 初始化本地 Git repository，并配置 `origin` 为 `https://github.com/cyd56-ops/CAN.git`。
- **Last published source checkpoint:** `dc8f209`（`Add V1-M1 epoch progress reporting`）已推送至
  `origin/main`；其前序 V1-M1 artifact/training-boundary checkpoint 为 `c6c38df`。
- **Worktree state:** 唯一 worktree 为 `/home/kali/CAN`；当前 `main` 跟踪 `origin/main`。以
  `git status --short` 和 `git rev-parse HEAD` 复核本机工作树状态；`.gitignore` 排除了 `.venv/`、
  `data/`、`artifacts/`、checkpoints、模型权重和 `paper/*.pdf`。
- **Compute resource:** `SERVER_REQUIRED`。batch 进度可观测性已在本机实现并通过完整质量检查；已授权的
  AutoDL A4000 环境仍是正式 archive 下载和两次训练唯一允许的环境。本机不得下载数据、初始化 CUDA 训练
  或产生正式 V1-M1 artifact。
- **Filesystem state:** 当前包含治理/研究/安全与 A0--A4/V1 协议、规格和构造决定文档、Python 项目及开发/ML 依赖锁、A0/A4/V1 exact reference、A1 两个 verifier backends、A4-C1/V1-C1 dependency-free verifiers、A2-E1/A2-E2 模型与协调器、A3-v1/A3-v2 state/coordinator、A4/V1 exact/neural evidence adapters、单元/差分/集成/安全测试和治理检查脚本；ignored `data/a2/` 与 `artifacts/a2/` 保存数据、local states、manifest、license 和报告，`paper/` 保存两份 ignored PDF；`.git` 现为本地 Git metadata，`.agents` 与 `.codex` 目录仍为空。
- **Completed modules:** A0 parser/reference evidence、A1 exact backends、A2-E1/A2-E2 模型/协调器/实验、A3-v1 默认关闭协议壳、A4 exact/neural relation 与 adapters，以及 V1-P2 non-production public profile/parser/registry、coefficient-domain exact relation、A3-v2 commit-first single-terminal coordinator、exact/neural evidence adapters、`V1-C1-MSIS` graph、`V1-P2-PSR-E1` generated-key/sampler/single-attempt experiment、fresh-transcript retry/exhaustion harness 和 V1-M1 CIFAR model/archive parser/adapter/runner/artifact writer 已实现。V1-P2 生产 prover、安全参数、NTT 和加速/量化 neural backend 尚未实现。
- **Completed specifications:** A1/A2/A3 规格保持闭合；A4 已选择 reviewed GPV PFDH 并固定非生产 public profile、proof/message 编码、exact reference relation、A3 adapter，以及 A4-C1 `80->3600->1153->1` point-pulse graph、范围账本和全部 canonical `(y,z)` 上的 `V_nn==V_ref` 证明；V1-P1 普通矩阵方案保留为 baseline，V1-P2 已冻结 reviewed FSwA-S Module-SIS protocol、non-production exact profile/range ledger、polynomial transcript/encoding、M-LWE/M-SIS 安全边界、A3-v2、direct-convolution neural relation，以及 `V1-P2-PSR-E1` toy prover/sampler/rejection 实验契约；V1-M1 已冻结 CIFAR-100/CIFAR-style ResNet-18 route、数据供应链/许可边界、fine-label order、split、preprocessing、two-run deterministic training and acceptance protocol；长期 `V0 -> V1-prep -> V1 -> V2` 路线及独立代码保留约束已同步。
- **Tests passed:** A0/A1 正负向、全域 differential 和 no-fallback 测试保持通过；V1-P2-PSR-E1 retry focused 58 项通过。V1-C1 focused unit/differential/security/A3-v2 route suite 47 项通过；V1-M1 batch-progress focused unit/security suite 30 项通过；完整 pytest 625 项通过。
- **Configured stack:** 本机 A1/A2 继续使用 Python `==3.11.*`（3.11.9）、官方 torch `2.13.0+cpu`、torchvision `0.28.0+cpu`、NumPy `2.4.4`、Pillow `12.2.0` 和 `requirements-ml.lock`。V1-M1 另冻结 AutoDL `can-v1`：Python 3.11.9、RTX A4000 16,376 MiB、driver 580.82.07、torch `2.13.0+cu126`、torchvision `0.28.0+cu126`，两个 primary wheel SHA-256 见 V1-M1 决策 section 8；不得将其写回 A2 CPU lock。
- **Baseline result:** 两次同种子十 epoch运行均得到 test loss `0.33665058851242063`、accuracy `88.08%`、prediction SHA `e5b48d60...e4a7`、state SHA `88062fee...8613` 和 determinism fingerprint `a59a9a9a...7d53`；模型 235,146 parameters/940,584 bytes。本机 batch-1 median 为 110.8/104.9 us，batch-256 median 为 2987.4/2790.8 us。
- **Gate result:** gate 重训得到同一 `88.08%`、prediction/state SHA；10,000 个 gated labels 全部匹配，accepted/rejected end-to-end median 为 `1849.2/1570.5 us`，verifier-only median `1245.5 us`，accepted coordinator median `85.4 us`，accepted overhead `1750.2 us`/`1767.88%`。10,000 allow 各一次模型调用，rejected probe 与 1,100 次 rejected latency 请求均为零模型调用。
- **Public baseline result:** 两次固定十 epoch 运行均得到 test loss `0.007989783663357957`、accuracy `99.85%`、prediction SHA `f54b2351...6f0a`、state SHA `b71980eb...122be` 和 fingerprint `e4fbf9c0...c14f`；模型 50,370 parameters/201,480 bytes。两次本机 batch-1 median 为 `70.2/65.9 us`，batch-256 median 为 `1464.0/1340.001 us`。
- **Capability report result:** accepted-state 评估的 protected/public prediction SHA 分别为 `e5b48d60...e4a7` 与 `f54b2351...6f0a`；10,000 次 public、10,000 次 protected 调用精确互斥，单次拒绝探针和默认关闭探针均为零模型调用。最终复核运行的 public/protected/deny end-to-end median 为 `238.7/1540.9/1324.4 us`。
- **Known limitations:** A0、A4 与 V1 conformance 均为非安全 toy profile；A1/A2 只支持当前 CPU tuple；A4-C1/V1-C1 只有 dependency-free sparse exact backend，没有生产参数、NTT、PyTorch/qint8/CUDA/export、系统 related-work 检索或白盒保证。V1-P2 sampler/single-attempt/retry 与 V1-C1 只支持 toy reproducibility、compiled arithmetic conformance 和 coordinator state testing；V1-M1 archive 已在服务器完成校验与解压、R1 正在运行，但尚无完成的 weights、baseline/gate/性能结果、验收或协议安全结论。
- **Documentation/code consistency:** 研究设计、安全文档、V1-P2/PSR/V1-M1 决策、README、治理脚本和本日志已同步 non-production exact/neural/A3-v2、`V1-P2-PSR-E1` generated-key/single-attempt/retry、V1-C1 graph/A3-v2 route、CIFAR/ResNet route、已验证 V1 GPU tuple、训练协议、artifact/report writer、本机 V1-M1 implementation、batch-level stdout progress 与服务器 formal-baseline 操作手册；`dc8f209` 已同步至 `origin/main`，本次 batch-progress 改动尚未形成或发布 Git checkpoint，唯一下一步为 `SERVER_REQUIRED` 的完成 R1 与一次 R2。
- **Server execution status:** 2026-08-16 项目负责人报告：已在冻结的 AutoDL A4000 环境从唯一首方 URL 完成 CIFAR-100 archive 的 size/SHA-256/MD5 校验并显式解压；预注册 R1（seed `1729`）已经启动。尚未向本工作树报告 focused-test 输出、R1 terminal output、artifact、metrics 或 R2 状态，故不得声称 baseline 已完成、验收或产生性能结果。
- **Security relevance:** A1/A2/A3/A4 既有边界保持不变；V1 测试支持 canonical polynomial encodings、公开 immutable profile、exact negacyclic relation、A3-v2 binding、单次终态 response、abort/expiry、route confusion、内部错误、replay/concurrency 和 pre-commit reject 零 protected calls。公开 fixtures 可直接构造 valid relation，只支持 conformance，不证明私钥持有、M-LWE/M-SIS 安全、主动冒充安全或授权安全。

## Verified command status

| Purpose | Exact command | Result |
| --- | --- | --- |
| Git status | `git status --short` | 无法运行：不是 Git 仓库（exit 128） |
| Current branch | `git branch --show-current` | 无法运行：不是 Git 仓库（exit 128） |
| Worktree list | `git worktree list --porcelain` | 无法运行：不是 Git 仓库（exit 128） |
| Full HEAD | `git rev-parse HEAD` | 无法运行：不是 Git 仓库（exit 128） |
| A2 focused tests | `.venv/bin/python -m pytest tests/unit/test_a2_mlp.py tests/unit/test_a2_baseline.py tests/unit/test_a2_gate.py tests/unit/test_a2_gate_experiment.py tests/integration/test_a2_gate_integration.py tests/security/test_a2_baseline_security.py tests/security/test_a2_gate_security.py tests/security/test_a2_gate_experiment_security.py` | 通过，73 tests |
| A2 public focused tests | `.venv/bin/python -m pytest tests/unit/test_a2_public_mlp.py tests/unit/test_a2_public_baseline.py tests/security/test_a2_public_baseline_security.py` | 通过，41 tests |
| A2 capability/materializer focused tests | `.venv/bin/python -m pytest tests/unit/test_a2_capability.py tests/unit/test_a2_capability_experiment.py tests/unit/test_a2_materialize.py tests/integration/test_a2_capability_integration.py tests/security/test_a2_capability_security.py tests/security/test_a2_capability_experiment_security.py tests/security/test_a2_materialize_security.py` | 通过，93 tests |
| A3 protocol focused tests | `.venv/bin/python -m pytest tests/unit/test_a3_protocol.py tests/security/test_a3_protocol_security.py` | 通过，29 tests |
| A4 relation focused tests | `.venv/bin/python -m pytest tests/unit/test_a4_reference.py tests/integration/test_a4_a3_integration.py tests/security/test_a4_reference_security.py` | 通过，39 tests |
| A4 neural focused tests | `.venv/bin/python -m pytest tests/unit/test_a4_neural.py tests/differential/test_a4_neural_differential.py tests/security/test_a4_neural_security.py tests/integration/test_a4_a3_integration.py` | 通过，21 tests |
| V1-P2 exact/A3-v2 focused tests | `.venv/bin/python -m pytest tests/unit/test_v1_reference.py tests/differential/test_v1_reference_differential.py tests/unit/test_a3_v2.py tests/integration/test_v1_a3_v2_integration.py tests/security/test_v1_a3_v2_security.py` | 通过，52 tests |
| V1-P2-PSR-E1 focused tests | `.venv/bin/python -m pytest tests/unit/test_v1_psr.py tests/differential/test_v1_psr_differential.py tests/security/test_v1_psr_security.py tests/integration/test_v1_a3_v2_integration.py` | 通过，58 tests |
| Unit tests | `.venv/bin/python -m pytest tests/unit` | 通过，372 tests |
| Differential tests | `.venv/bin/python -m pytest tests/differential` | 通过，32 tests |
| Integration tests | `.venv/bin/python -m pytest tests/integration` | 通过，13 tests |
| Security tests | `.venv/bin/python -m pytest tests/security` | 通过，168 tests |
| V1-M1 epoch-progress focused tests | `.venv/bin/python -m pytest tests/unit/test_v1_cifar100_resnet.py tests/unit/test_v1_m1_adapter.py tests/unit/test_v1_m1_baseline.py tests/security/test_v1_m1_route_security.py tests/security/test_v1_m1_artifact_security.py` | 通过，28 tests |
| Full tests | `.venv/bin/python -m pytest` | 通过，623 tests |
| Lint | `.venv/bin/ruff check .` | 通过 |
| Format | `.venv/bin/ruff format --check .` | 通过，86 files |
| Type checking | `.venv/bin/mypy src tests` | 通过，86 source files |
| Dependency consistency | `.venv/bin/python -m pip check` | 通过；pip 另提示用户 cache 目录不可写并禁用 cache |
| ML runtime probe | `.venv/bin/python -c 'import importlib.metadata as m, numpy, PIL, torch, torchvision; ...'` | torch `2.13.0+cpu`、torchvision `0.28.0+cpu`、NumPy `2.4.4`、Pillow `12.2.0`；CPU-only ops available |
| Torch wheel hash | `sha256sum /home/kali/.cache/pip/http-v2/4/3/7/f/7/437f741e1b68cb39af20cff8f6c1b059fa9e16f2465507823eab7667.body` | `6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b` |
| Torchvision wheel hash | `sha256sum /home/kali/.cache/pip/http-v2/0/e/5/4/f/0e54f4303d860a30f153423c1eb75bca4f925a10ab84608719f9b129.body` | `1dad604dfc0177ecebe0891bd9701fe2c62ec3f7819a247be541b3fb6effee99` |
| V1 GPU inventory (remote, user-provided) | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` | NVIDIA RTX A4000, 16,376 MiB, driver 580.82.07 |
| V1 host/conda probe (remote, user-provided) | `nvidia-smi \| sed -n '1,4p'; python --version; cat /etc/os-release; free -h; df -h /` | driver CUDA compatibility 13.0; Python 3.11.9 in `can-v1`; Ubuntu 22.04.1 LTS; 6 vCPU; 251 GiB RAM total/227 GiB available; root overlay 30 GiB/18 GiB free |
| V1 package consistency (remote, user-provided) | `python -m pip check` | `No broken requirements found.` |
| V1 Torch wheel hash (remote, user-provided) | `sha256sum /root/.cache/pip/http-v2/1/c/b/d/7/1cbd782261bb3ae356a3b815ba1896fc05173a48118ea775d7fde7ce.body` | `0f4e49e334e24b552f694f6315e0676fb3f816fb0f727871b9c6d1f73784cc25` |
| V1 Torchvision wheel hash (remote, user-provided) | `sha256sum /root/.cache/pip/http-v2/1/9/9/7/3/19973bc26893d997a3dafc41954d9a61e22e077dedeb62ef18686f14.body` | `92f53415dd68e56b6f912441997ab0e78fcd6245b1706ee6e88ce2df917248fa`; archive listing contains `torchvision/` |
| A2 repeat 1 | `PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_baseline --repeat 1` | 通过，test `88.08%`，报告写入 ignored root |
| A2 repeat 2 | `PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_baseline --repeat 2` | 通过，与 repeat 1 的 metrics/predictions/state 相同 |
| A2 compare | `.venv/bin/python -m can.experiments.a2_baseline --compare` | 通过，fingerprint `a59a9a9ac2797261eb824af564d6fa64a3c3e19fa43886b2349aa48bccaf7d53` |
| A2 gate experiment | `PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_gate --run` | 通过，10,000 labels match，rejected probe zero calls，报告写入 ignored root |
| A2 public repeat 1 | `PYTHONHASHSEED=20260730 .venv/bin/python -m can.experiments.a2_public_baseline --repeat 1` | 通过，test `99.85%`，报告写入 ignored root |
| A2 public repeat 2 | `PYTHONHASHSEED=20260730 .venv/bin/python -m can.experiments.a2_public_baseline --repeat 2` | 通过，与 repeat 1 的 metrics/predictions/state 相同 |
| A2 public compare | `.venv/bin/python -m can.experiments.a2_public_baseline --compare` | 通过，fingerprint `e4fbf9c09afc3aaada32dd60f7368346a64138497178618c78ac0b1baeb4c14f` |
| A2 materialization/report | `.venv/bin/python -m can.experiments.a2_materialize --run` | 通过；两个 accepted state/manifest 校验通过，真实 10,000-image `capability.json` 已生成 |
| A2 no-training report replay | `PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_materialize --report` | 通过；复用 local states 重跑报告，未训练或改写 state |
| Check script syntax | `bash -n scripts/check_governance_docs.sh` | 通过 |
| Governance docs | `./scripts/check_governance_docs.sh` | 通过 |
| V0/V1-prep/V1/V2 route consistency | `./scripts/check_governance_docs.sh` | 通过；十一份路线文档均保留 `V1-prep` 非认证边界和独立代码路线 |
| Research/security docs | `./scripts/check_governance_docs.sh` | 通过 |
| A0 protocol spec | `./scripts/check_governance_docs.sh` | 通过 |
| A1 numerical spec | `./scripts/check_governance_docs.sh` | 通过 |
| A1 construction decision | `./scripts/check_governance_docs.sh` | 通过 |
| A1 backend decision | `./scripts/check_governance_docs.sh` | 通过 |
| A2 model experiment protocol | `./scripts/check_governance_docs.sh` | 通过 |
| A2 capability experiment spec | `./scripts/check_governance_docs.sh` | 通过 |
| A3 challenge-response protocol spec | `./scripts/check_governance_docs.sh` | 通过；18 个必需标题、唯一下一步和任务状态枚举已检查 |
| A4 GPV relation spec | `./scripts/check_governance_docs.sh` | 通过；15 个必需标题、bootstrap 文件、唯一下一步和任务状态枚举已检查 |
| A4 neural construction decision | `./scripts/check_governance_docs.sh` | 通过；15 个必需标题、实现/测试文件和路线边界已检查 |
| V1 protocol selection decision | `./scripts/check_governance_docs.sh` | 通过；18 个必需标题、route isolation、唯一下一步和状态枚举已检查 |
| V1 Module-SIS protocol decision | `./scripts/check_governance_docs.sh` | 通过；18 个必需标题、V1-prep/route isolation、唯一下一步和状态枚举已检查 |
| V1 prover/sampler/rejection spec | `./scripts/check_governance_docs.sh` | 通过；18 个必需标题、toy/secret/theorem boundary、唯一下一步和状态枚举已检查 |
| V1 model experiment decision | `./scripts/check_governance_docs.sh` | 通过；18 个必需标题、route isolation、bootstrap 文件、唯一下一步和状态枚举已检查 |
| A3 canonical message length probe | `.venv/bin/python -c 'domain=b"CAN-A3-MSG-v1\x00"; total=len(domain)+1+4+32+2+8+8+32+32; print(len(domain), total)'` | 通过，domain 14 bytes、message 133 bytes |

# 4. Milestones

| Phase | Goal | Status | Acceptance criteria | Associated commit |
| --- | --- | --- | --- | --- |
| M0 Governance baseline | 建立长期约束与动态事实源 | completed | 治理文档和检查脚本存在且通过 | 不适用，尚无 Git 仓库 |
| M1 Research and security baseline | 确认目标、阶段、威胁模型、非目标和论文边界 | completed | 研究设计、`SECURITY.md`、长期规则和工作日志一致 | 不适用，尚无 Git 仓库 |
| M2 A0 protocol specification | 固定 toy LWE 精确关系、输入域和 token issuer 语义 | completed | A0-v1 可直接指导 oracle，并从 wire format 排除 client-chosen `A`/参数 | 不适用，尚无 Git 仓库 |
| M3 Technical bootstrap | 建立 Python/PyTorch 项目和质量配置 | completed | 版本锁定，单元/集成/安全/lint/类型命令可运行 | 不适用，尚无 Git 仓库 |
| M4 Exact and neural verifier | 实现 oracle、编译网络和误差分析 | completed | dependency-free/PyTorch CPU 全域差分通过，接受集合主张有证明和明确限制 | 不适用，尚无 Git 仓库 |
| M5 Model access control | 集成最小 MLP 和 fail-closed 协调器 | completed | 拒绝路径零受保护模型调用，准确率/延迟/开销可复现 | 不适用，尚无 Git 仓库 |
| M6 Request binding and freshness | 固定并实现 challenge-response、请求绑定和 replay 状态 | completed | tamper/replay/过期/并发/post-commit 测试通过，且不把 proof stub 称为认证 | 不适用，尚无 Git 仓库 |
| M7 V1-prep public-relation compiler | 闭合 A3 binding 与 A4 canonical `(y,z)` 神经代数内核 | completed | 全输入神经 soundness、evidence-only 组合和 toy 非认证边界明确 | 不适用，尚无 Git 仓库 |
| M8 V1 lattice authentication | 选择并实现已有安全分析支持的格 challenge-response/身份协议 | in_progress | 协议安全与神经编译证明分离，身份、输入、scope 和 replay 测试闭合 | 待确认 |
| M9 V2 ML-DSA verifier | 评估标准 ML-DSA 精确 reference 与分模块神经验证 | pending | 标准向量、规范解析和获准模块的 exact equivalence 通过 | 待确认 |
| M10 Stage B tool authorization | 实现 capability 和强制工具网关 | pending | Router/提示注入/直接调用不能绕过网关 | 待确认 |

# 5. Task board

| Task | Owner or route | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- |
| 创建长期 Agent 约束 | Codex / documentation | completed | 当前用户要求 | 人工核对规则，并运行 `./scripts/check_governance_docs.sh` |
| 创建动态工作事实源 | Codex / documentation | completed | 当前目录只读盘点 | 核对本文件与探测结果，并运行 `./scripts/check_governance_docs.sh` |
| 导入两篇参考论文 | Codex / reference assets | completed | `/mnt/e/CAN/paper` 中的两个 PDF 可读 | 源/目标大小、`cmp` 和 SHA-256 均一致 |
| 确认项目研究基线 | 项目负责人 / research design | completed | 用户提供完整方案和科研定位 | 第 1、2、9 节及研究设计已同步 |
| 建立安全文档 | Codex / security design | completed | 已确认安全敏感范围 | `SECURITY.md` 明确事实、计划和不保证的性质 |
| 固定 A0 协议规格 | Research / cryptography | completed | 研究与安全基线 | `docs/A0_PROTOCOL_SPEC.md` 固定 relation、issuer、编码、误差契约和攻击边界 |
| 完成系统文献检索 | Research / related work | pending | 可用的论文数据库或网络检索 | related-work 矩阵覆盖神经密码、格签名网络和模型访问控制 |
| 初始化实现与质量工具 | Engineering | completed | A0 协议规格完成 | 入口配置存在，准确质量命令已运行并记录 |
| 实现精确 oracle | Research / reference | completed | A0 协议规格与技术 bootstrap | 71 个单元测试和 7 个 A0 安全测试覆盖正负向、边界和确定性向量 |
| 吸收认证神经元初步设想 | Research / design | completed | A0 oracle 与现有安全架构 | 保留独立验证、模关系、margin 和 capability 分级方向；争议构造进入延期假设清单 |
| 固定 A1 数值/算子规格 | Research / verifier | completed | 精确 oracle | `docs/A1_NUMERICAL_SPEC.md` 固定共同语义、tensor 范围/误差和全输入 soundness 义务，治理检查通过 |
| 选择 A1 神经构造和证明方法 | Research / verifier | completed | A1 数值/算子规格 | `docs/A1_CONSTRUCTION_DECISION.md` 固定 exact ReLU 主候选、exact-ops 对照、三层范围和全域证明路线 |
| 实现固定神经验证器 | Research / verifier | completed | oracle、数值语义和构造决定 | dependency-free backend 与 oracle 全域差分一致，134 tests 通过且无 reference/baseline fallback |
| 固定 A1 部署 backend | Research / verifier | completed | dependency-free verifier | `docs/A1_BACKEND_DECISION.md` 固定 CPU exact 目标、wheel/hash、算子/dtype、artifact 和复测要求 |
| 实现 A1-B1 PyTorch CPU backend | Engineering / verifier | completed | A1-B1 决策与 CPU wheel | exact operator mapping、全域差分、no-fallback 和 artifact tests 全部通过 |
| 固定 A2 最小业务模型实验协议 | Research / model access | completed | A1-B1 验收完成 | `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md` 固定唯一数据集/模型、版本与 artifact、确定性训练、指标和硬门控边界 |
| 实现 A2-E1 无门控 MLP baseline | Engineering / model | completed | A2-E1 协议 | 数据/hash/license、严格输入校验、两次同种子十 epoch 结果和 32 项 focused tests 已核验 |
| 集成二元硬门控 | Research / model access | completed | A2-E1 baseline 验收完成 | 唯一协调器、固定响应、拒绝零 protected-model calls、全测试集 allow labels 一致 |
| 固定 A2-E2 capability 分级实验规格 | Research / model access | completed | A2-E1 二元门控验收完成 | `docs/A2_CAPABILITY_EXPERIMENT_SPEC.md` 固定独立 public 能力、默认关闭策略、三态语义与隔离矩阵 |
| 实现 A2-E2 独立 public model baseline | Engineering / model | completed | A2-E2 规格完成 | 两次固定种子十 epoch 运行的 metrics/predictions/state 一致，严格输入与 artifact 测试通过 |
| 实现 A2-E2 三态协调器与只评估 runner | Research / model access | completed | public baseline 单独验收完成 | 79 项 focused 调用矩阵、固定 envelope、不可升级、并发、异常、报告与无 fallback 测试通过 |
| 运行 A2-E2 已验收权重三态报告 | Research / model access | completed | 按 D-024 确定性物化并校验两个 accepted model states | 10,000 protected/public 标签摘要、完整计数和三路 latency 已写入固定 ignored report |
| 固定 A3 challenge-response 协议规格 | Research / protocol security | completed | A2-E2 报告验收完成 | `docs/A3_CHALLENGE_RESPONSE_SPEC.md` 固定 133-byte message、nonce 生命周期、原子 consume、安全游戏和拒绝零副作用 |
| 实现 A3-v1 请求绑定与 freshness 协议壳 | Engineering / protocol security | completed | A3-v1 规格完成 | canonical codec/hash、in-memory store、默认关闭 coordinator 和 proof-stub-only tests 通过 tamper/replay/expiry/concurrency/post-commit 矩阵 |
| 固定 A4 GPV 公钥关系与 reference adapter | Research / cryptography | completed | A3-v1 协议壳完成 | reviewed relation、105-byte proof、公开 profile、exact `V_ref`、A3 adapter 和 37 项 focused tests |
| 固定 V0/V1-prep/V1/V2 长期路线 | Research / roadmap | completed | 项目负责人确认路线调整 | 工作日志、研究设计、安全边界、A3/A4 规格和 README 一致，治理检查通过 |
| 选择并实现 A4 固定神经 verifier | Research / verifier | completed | A4 exact relation/reference 验收完成 | A4-C1 graph、全输入证明、21 项 focused tests 和 A3 neural adapter 完成且无 reference fallback |
| 完成 V1-P1 普通矩阵 SIS 协议评估 | Research / cryptography | completed | A4 neural core 与 A3 composition 闭合 | `docs/V1_PROTOCOL_SELECTION_DECISION.md` 固定可复现的历史 baseline、commit-first transcript、canonical encoding、rejection/abort 和安全游戏 |
| 保留 V1-P1 普通矩阵 SIS 设计基线 | Research / cryptography | completed | V1-P1 决策完成 | 历史设计、来源和取舍保留，但不作为当前 V1 实现目标 |
| 选择 V1-P2 Module-SIS Sigma 主协议 | Research / cryptography | completed | 项目负责人决定切换当前 V1 主路线 | `docs/V1_MODULE_SIS_PROTOCOL_DECISION.md` 冻结 FSwA-S、商环、key/transcript、encoding、rejection、安全游戏和神经关系 |
| 冻结 V1 CIFAR-100/ResNet-18 主实验路线 | Research / model experiment | completed | 项目负责人确认新增主实验并保留 A2 baseline | `docs/V1_MODEL_EXPERIMENT_DECISION.md` 固定独立 input/model profile、CIFAR-style architecture、复现/门控验收和 route isolation；未下载或训练 |
| 实现 V1-P2 exact reference 与 A3-v2 协议壳 | Engineering / protocol security | completed | V1-P2 决策完成 | non-production profile、canonical parser、公开 registry、exact relation、单次终态状态机及 52 项 focused 测试通过 |
| 冻结 V1-P2 prover/sampler/rejection 实验契约 | Research / cryptography | completed | exact/A3-v2 checkpoint 闭合 | `V1-P2-PSR-E1` 固定 toy domain、SHAKE256 deterministic sampler、emit/abort、fresh retry、secret lifecycle、向量与指标；治理检查通过 |
| 实现 V1-P2 generated-key fixture 与 deterministic vectors | Engineering / cryptography | completed | `V1-P2-PSR-E1` 规格闭合 | 临时 `s` 生成公开 `t`，三个 SHAKE256 sampler、single attempt 与公开 manifest 可复现；51 项 focused tests 覆盖 byte rejection、边界/计数、exact differential 和 lifecycle |
| 实现 V1-P2 A3-v2 fresh-transcript retry harness | Engineering / protocol security | completed | generated-key/sampler/single-attempt checkpoint 闭合 | 每次 abort 使用新 nonce/transcript/y/u/c，forced abort、first success、expiry、exhaustion、replay/concurrency 和零 protected/verifier calls 通过 |
| 实现 V1-C1-MSIS neural verifier | Research / verifier | completed | prover/retry 与 exact range ledger 闭合 | `56 -> 11056 -> 17 -> 1` fixed affine/ReLU graph、全 canonical input `V_nn==V_ref`、独立 differential/no-fallback 和 A3-v2 route-level zero-call 通过 |
| 冻结 V1-M1 GPU、数据与训练协议 | Engineering / model experiment | completed | 已触发 `SERVER_REQUIRED`、通知项目负责人并完成 A4000/CUDA/wheel/hash/determinism smoke；随后完成 `LOCAL_OK` data/training protocol | 仅冻结官方 source/hash、许可边界、split、preprocessing、two-run SGD/cosine protocol、validation-only checkpoint rule 和预注册 acceptance threshold；未下载或训练 |
| 实现 V1-M1 isolated model/archive parser/adapter/baseline runner | Engineering / model experiment | completed | V1-M1 data/training protocol 已冻结 | CIFAR-style ResNet-18、verified archive-to-extraction parser、fixed split/preprocessing/training selection、raw-input adapter 和 A3-v2 route isolation；25 项 focused tests 通过且未下载或训练 |
| 实现 V1-M1 ignored artifact/report writer | Engineering / model experiment | completed | isolated baseline runner 已完成 | 保存选定 state、结构化 manifest/report，拒绝覆盖和 symlink；14 项 focused artifact/route tests 通过且未下载或训练 |
| 实现 V1-M1 epoch 进度可观测性 | Engineering / model experiment | completed | R1 已在旧 runner 进程启动；输出不得影响训练、随机性或 artifact | 每 epoch 输出稳定的公共训练/验证摘要并 flush；单元测试覆盖格式与内容；R2 前记录实际 source HEAD |
| 扩展 V1-M1 batch 进度可观测性 | Engineering / model experiment | completed | 项目负责人要求开始提示、同步进度条/百分比和结束提示；输出不得影响训练、随机性、选模或 artifact | 无依赖 reporter 已完成；focused 30 项、完整 625 项与质量/治理检查通过，项目负责人已授权提交并直推 |
| 发布 V1-M1 batch-progress checkpoint | Engineering / publication | in_progress | 项目负责人明确授权提交当前 5 个已验证 source/test/documentation 文件并直推 `origin/main` | 提交前运行治理检查，推送后记录 commit、远端 ref 与干净工作树 |
| 同步 V1-M1 checkpoint 到 `origin/main` | Engineering / publication | completed | writer、progress 文档/测试和完整质量检查闭合，项目负责人明确授权直推 | `c6c38df` 与 `dc8f209` 已由 `main` 直推，内容不含数据、weights 或 artifacts |
| 执行 V1 CIFAR-100/ResNet-18 baseline 与认证门控实验 | Engineering / model experiment | in_progress | V1-M1 environment、data/training protocol、isolated implementation 和 artifact/report writer 闭合；R1 已启动 | 两次可复现 baseline、accepted artifact、allow prediction equivalence、reject zero calls 和认证/模型 latency 报告通过 |
| 建立 V2 ML-DSA 标准 reference 基线 | Research / cryptography | pending | V1 exact/neural/authentication 闭合 | 标准测试向量与规范 parser/reference 通过；不要求首版神经化全部 hash/encoding |
| 绑定 V2 CIFAR-100/ResNet-18 对照实验 | Engineering / model experiment | pending | V2 ML-DSA reference 与独立 V2 adapter 闭合 | 复用同一业务 benchmark 但保持 V2-local registry/adapter/入口，V1/V2 route confusion 和无 fallback 测试通过 |
| 设计阶段 B capability 网关 | Research / authorization | pending | V1 阶段验收；V2 不是前置依赖 | 工具参数绑定和不可绕过测试通过 |

# 6. Current next step

**计算资源：`LOCAL_OK`。当前只提交并直推已验证的 V1-M1 batch-progress source checkpoint；不得下载
CIFAR-100、训练或生成真实 V1-M1 artifact。**

**唯一下一步：`LOCAL_OK` 运行治理检查后提交并直推当前 batch-progress checkpoint 至 `origin/main`，记录
commit、远端 ref 与干净工作树。随后恢复 `SERVER_REQUIRED`：保留既有 R1 的实际 source HEAD、terminal output、
state/manifest/report，并在开始 R2 前记录新 source HEAD，将与 R1 的差异限定为 observability-only。不得重试、
增加 run、改变数据、split、预处理、超参数或阈值，也不得进入 gate、性能、V2、Fiat--Shamir、ML-DSA 或
Stage B。**

# 7. Blockers and residual risks

- **Git publication status:** `origin` 已使用 SSH URL `git@github.com:cyd56-ops/CAN.git`；项目负责人明确授权直推
  `main` 后，`git push -u origin main` 成功将 `ae79db1..c6c38df` 推送至远端，并建立
  `main -> origin/main` upstream。推送内容经 staged diff 检查与敏感模式扫描，不含数据、weights、artifact、
  private key、GitHub token 或 AWS access key。
- **V1-M1 experiment status:** 2026-08-15 已在项目负责人提供的已授权 AutoDL 单 GPU 容器完成 A4000/CUDA/wheel/hash 与 deterministic smoke；CIFAR-100 official archive/source/hash、许可边界、训练/验证切分、预处理、超参数、阈值与两次完整训练协议已冻结，详见 `docs/V1_MODEL_EXPERIMENT_DECISION.md` sections 4, 7--10。2026-08-16 项目负责人报告已完成 archive 三项校验与显式解压，并已启动 R1；R1 结果、R2、accepted artifact、gate 与性能测量均尚未报告或验收。
- **Artifact lifecycle:** D-024 重新物化的两个 `state_dict`、manifest 和 `capability.json` 只位于
  ignored `artifacts/a2/`；它们不得提交、上传或进入发布包。任何删除后的再次生成都必须重跑固定
  materializer、摘要校验和 no-training evaluator。
- **Environment limitation:** 系统 Python 缺少 `ensurepip/python3.11-venv`，标准 `python3 -m venv .venv` 不能自行安装 pip；已使用 `python3 -m pip --python .venv install 'pip==24.0'` 建立可用隔离环境，并在 `README.md` 记录 fallback。
- **Research risk:** 本 checkpoint 定向核验了 GPV STOC 2008、Lyubashevsky 2009/2012、Liu--Zhandry 2019 与 Devevey et al. 2023/2024，但尚未完成系统 related-work 检索，最终新颖性边界仍可能变化。
- **Novelty risk:** 通用密码 DNN 编译、签名控制模型输出和格签名神经网络已有直接先例；必须证明 LWE 专用量化 soundness 和访问控制组合的增量。
- **Security risk:** 任意 `A`、chosen `b` 或把公开加密误当认证可造成直接解锁或判决 oracle。
- **A0 limitation:** A0-v1 从 wire format 排除了 client-chosen `A`，但小参数下 adaptive chosen-`b`、replay、输入替换和白盒读取仍明确不受保护。
- **A3/A4 composition risk:** A3-v1 与 A4 exact adapter 已通过 toy gadget proof 的组合测试，但
  gadget matrix 允许公开构造 proof，deterministic stub/gadget 均不能形成不可伪造性主张。in-memory
  store 重启会使 outstanding challenge 失效，也不支持分布式原子性或 durable consume。
- **A4 cryptographic risk:** `q=257,n=8,m=72,beta_inf=1` 未经 security estimator 选择，具体
  SHAKE256 hash-to-syndrome 不自动满足 GPV random-oracle 证明；当前只支持 relation conformance。
- **A4 neural backend risk:** A4-C1 已对全部 canonical `(y,z)` 证明并测试 exact
  `V_nn==V_ref`，但 hash-to-syndrome 仍在可信 canonical preprocessing，且当前只有 dependency-free
  sparse exact backend；任何 PyTorch/qint8/CUDA/export/剪枝映射都必须重新做范围证明与全域差分，
  不能继承当前等价结论。
- **V1 protocol risk:** V1-P2 non-production conformance parameters、exact/A3-v2、generated-key/
  sampler/single-attempt 与 fresh retry/exhaustion measurement 已闭合，但生产 prover 尚未实现，密码
  安全参数、M-LWE/M-SIS theorem conditions 和主动冒充安全尚未闭合。当前 toy profile 理论 emit probability 只有
  `(13/17)^32`，高 abort rate 不适合性能或生产结论；引用 FSwA-S/Dilithium 不能替代 estimator、
  条件核对和实现审查。
- **V1 neural risk:** V1-C1 已在固定 toy profile 上以 direct coefficient-domain convolution、point
  pulses 和 exact integer range ledger 证明 `V_nn==V_ref`；其 `11056` 第一层仅是该 toy ledger 的结果，
  不能外推到更大参数。NTT、PyTorch、qint8、CUDA、export、剪枝或微调均须独立语义/范围证明和差分，
  不继承当前结论。
- **A3-v2 state risk:** commit-first 单进程状态机已实现在 challenge 前保存 commitment/input snapshot，
  并让一个 parsed response attempt 终结 transcript；该选择减少多次验证 oracle，但会放大
  malformed-response 与 DoS 的可用性权衡，不支持 durable/distributed consume。
- **V1 model-experiment risk:** V1-M1 已冻结域隔离的 CIFAR-100/CIFAR-style ResNet-18 input/model
  profile，并已实现 strict adapter、archive parser、training runner 与 artifact writer；当前 R1 的正式
  结果尚未报告。完整训练的 determinism、BatchNorm state、数据增强、decoded dataset 摘要、accepted
  artifact 生命周期与两次结果差异仍须依据已冻结协议验收，不能继承 A2 的复现结论。
- **Fiat--Shamir proof risk:** abort loop 的 ROM/QROM 证明存在已发表的技术修正；交互式 V1-P2 不能
  自动升级为非交互签名或强不可伪造结论。
- **Route isolation risk:** V0、V1、V2 必须长期并存；V1 exact/A3-v2 已使用新增 parser、registry、
  evidence 和 route-confusion tests，后续 prover/neural/model 路径仍不得共享协议入口或形成 fallback。
- **Roadmap composition risk:** V1-prep 只证明 canonical 格代数谓词的神经编译与 A3 组合边界；它不
  证明 V1 的知识可靠性、不可伪造性或身份授权。V1-P2 的 M-LWE/M-SIS、transcript/rejection 安全与
  neural core soundness 必须分开论证；V2 标准一致性也不能反推 V1 安全。
- **Definition risk:** A1-C1 已把主 core 限定为固定 affine/ReLU，并把普通 `%`/Floor/比较隔离为对照；后续实现若偏离该 graph 或把 adapter 操作算作神经层，论文主张即失效。
- **Proof risk:** dependency-free 和 A1-B1 PyTorch CPU exact backends 已验证完整 toy 域；任何 framework upgrade、kernel、device、storage、accumulator、rounding、saturation 或 export 差异仍可能破坏 error `0`，必须逐 backend 重证和复测。
- **Deployment risk:** 黑盒可信入口是假设而非实现保证；白盒持有者可删层、改权重或直接调用业务网络。
- **Compatibility risk:** torch/torchvision 官方 CPU wheels、hash 和当前 WSL2 kernels 已实测，NumPy/Pillow 已锁定；没有原生 Linux、其他 CPU、framework upgrade 或 accelerator 证据。本机 latency 不能外推。
- **Experimental limitation:** 当前已有 A0/A1 toy 域、A2 模型/门控、A3 freshness 壳、A4 exact/neural
  toy relation，以及 V1-P2 non-production exact/neural/A3-v2 conformance 和 toy single-attempt prover；
  尚无安全参数、生产 prover、不可伪造 V1 认证、已完成并验收的 V1 CIFAR 模型实验、量化 artifact 或白盒不可绕过结论。`88.08%` protected accuracy 与
  `99.85%` coarse public accuracy 都不是安全指标。
- **Performance observation:** 当前 gate accepted median overhead 为 `1750.2 us`/`1767.88%`，主要来自逐请求 A1-B1 verifier contract 复核；这是本机测量，不是跨平台结论，后续优化不能削弱 startup/runtime gate 或引入 fallback。
- **Environment observation:** ML runtime 当前固定 torch `2.13.0+cpu`、torchvision `0.28.0+cpu`、NumPy `2.4.4` 和 Pillow `12.2.0`，580 项测试通过；`pip check` 无 broken requirements，但报告用户 cache 目录不可写并禁用 cache。任何版本变化必须重新核验数据、确定性和结果。
- **Dataset supply-chain risk:** Fashion-MNIST loader 只配置 HTTP mirror 和 MD5；本次另存四个 SHA-256 并通过 GitHub contents API 核验 MIT license blob/hash，但这些 reproducibility identifiers 仍不能提供传输级现代来源认证。数据与许可不得提交或再分发。
- **Concept hypothesis risk:** A1-C1 已处置层数、bounded ReLU、Floor/Sigmoid/MASK 和显式 `A*s`；A2-E2 已实现并测量独立模型三态主路线，但通用 sawtooth、LWE/SIS 神经兼容性、替代对照和形式机器证书仍缺少实验，不能用于当前论文主张。
- **Reference asset risk:** 两份 PDF 的版权、再分发许可和是否属于论文私有材料尚未确认；按 `AGENTS.md` 约束，不得自动提交或推送这些二进制文件。

# 8. Recent work log

## 2026-08-16 - Expand V1-M1 batch progress observability

- **Decision:** 项目负责人要求训练器在训练开始时提示、以 train/validation/test 的每个已完成 batch 同步更新
  进度条和百分比，并在 artifact 写入完成后输出训练结束提示。
- **Scope:** 此工作仅修改 `src/can/experiments/v1_m1_baseline.py` 的 stdout 可观测性及其测试和 V1-M1 操作
  文档。进度只能读取公开 run/epoch/stage/batch 计数；不得记录样本、预测、权重、secret 或逐 batch metric，
  也不得影响随机性、优化、scheduler、validation-only checkpoint 选择或 artifact 内容。
- **Resource / next step:** `LOCAL_OK`。先完成实现、focused/full 测试、质量检查和治理文档检查；不下载或训练。
- **Implementation:** 在 runner 内加入无依赖 private reporter。它在 data loading/deterministic setup 后输出
  `training started`，按每个已完成的 train/validation/test batch 以 `flush=True` 重绘固定 30-column ASCII
  progress bar 与百分比，保留每 epoch 汇总；selected state、manifest 与 report 成功原子写入后才输出
  `training completed`。它只读取公开 run/epoch/stage/batch 计数。
- **Verification:** `.venv/bin/python -m pytest tests/unit/test_v1_m1_baseline.py tests/unit/test_v1_cifar100_resnet.py tests/unit/test_v1_m1_adapter.py tests/security/test_v1_m1_route_security.py tests/security/test_v1_m1_artifact_security.py` -> `30 passed`；`.venv/bin/python -m pytest -q` -> `625 passed in 34.18s`；`.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`、`.venv/bin/mypy src tests`、`.venv/bin/python -m pip check`、`bash -n scripts/check_governance_docs.sh` 与 `./scripts/check_governance_docs.sh` 均通过。`pip check` 仅提示用户 cache 目录不可写并禁用 cache。
- **Publication authorization:** 项目负责人已明确授权将当前 5 个 source/test/documentation 文件提交并直推
  `origin/main`；不包含 data、weights、artifact、secret 或其他生成输出。
- **Resource / next step:** `LOCAL_OK`。运行治理检查后提交并推送；推送完成后记录 commit/remote ref，并恢复
  `SERVER_REQUIRED` 的 R1/R2 交接。

## 2026-08-16 - Publish the V1-M1 progress checkpoint on main

- **Decision:** 按项目负责人明确授权，将 `dc8f209`（`Add V1-M1 epoch progress reporting`）直接推送到
  `origin/main`，不创建 backup branch。
- **Scope:** 发布每 epoch stdout progress、其单元测试、V1-M1 protocol/server documentation 和 R1/R2
  source provenance 记录；不包含 CIFAR data、解压内容、weights、state、manifest、report、secret 或其他
  generated artifact。
- **Verification:** 推送输出为 `4ef2ee1..dc8f209  main -> main`；推送后本地 `main` 与 `origin/main`
  同步，HEAD 为 `dc8f209d999c8400dfe98acba68c218400772594`，且 `git status --short` 为空。
- **Next step / resource:** `SERVER_REQUIRED`。完成旧 runner 的 R1，随后在记录 source HEAD 后仅运行一次
  R2；不进入 gate、性能、V2、Fiat--Shamir、ML-DSA 或 Stage B。

## 2026-08-16 - V1-M1 epoch progress observability checkpoint

- **Scope:** 在 `src/can/experiments/v1_m1_baseline.py` 的每个完整 train/validation epoch 后，以
  `flush=True` 输出一条稳定、仅含公共聚合指标的进度行：run、seed、epoch、train/validation loss、
  validation top-1 和当前 best-validation checkpoint。该输出不读取或写入样本、权重、secret、
  checkpoint selection、artifact 或 latency 数据。
- **Server boundary:** 项目负责人已报告 R1（seed `1729`）在旧进程中运行；该进程不会加载本 checkpoint。
  R1 完成后必须保留其 source HEAD、terminal output 和 artifact。若 R2 使用本 checkpoint，则必须记录其
  不同 source HEAD，并将差异限定为 observability-only；不得据此重跑 R1 或修改已冻结训练语义。
- **Documentation:** `docs/V1_MODEL_EXPERIMENT_DECISION.md` 与
  `docs/V1_AUTODL_ENVIRONMENT_SETUP.md` 已同步 stdout 进度行、artifact 边界与 R1/R2 source provenance。
- **Verification:** V1-M1 focused unit/security suite `28 passed`；完整
  `.venv/bin/python -m pytest` 收集并通过 `623` 项；`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`、`.venv/bin/mypy src tests` 和
  `.venv/bin/python -m pip check` 通过。`pip check` 仅提示用户 cache 目录不可写并禁用 cache。
- **Next step / resource:** `SERVER_REQUIRED`。完成旧 runner 的 R1，随后在记录 source HEAD 后仅运行一次
  R2；不进入 gate、性能、V2、Fiat--Shamir、ML-DSA 或 Stage B。

## 2026-08-12 - Git bootstrap for initial GitHub publication

- **Authorization and remote check:** 项目负责人要求将当前项目上传至 `https://github.com/cyd56-ops/CAN`。经授权的 `git ls-remote --heads https://github.com/cyd56-ops/CAN.git` 返回 exit `0` 且无 refs，远端没有现有分支，不存在需合并或覆盖的远端历史。
- **Local repository:** 在 `/home/kali/CAN` 执行 `git init -b main`，并将 `origin` 配置为该 HTTPS URL。提交身份仅写入 repository-local config：`user.name=cyd56-ops`、`user.email=yandachen56@gmail.com`。
- **Initial staging boundary:** `git add .` 后精确暂存 103 个 source、test、configuration 和 documentation 文件，共 27,304 insertions；忽略规则保留 `.venv/`、`data/`、`artifacts/`、`checkpoints/`、`*.pt`、`*.pth`、`*.ckpt` 与 `paper/*.pdf`。常见 private key、GitHub token 与 AWS access-key 模式扫描无匹配；`git diff --cached --check` 仅报告 `.gitignore` 与 `requirements-dev.lock` 的既有 EOF blank-line 提示。
- **Boundary held:** 未将训练数据、artifacts、checkpoint、local state、secret、credential 或 reference PDF 暂存。此 Git bootstrap 不修改 V1-M1 的 `SERVER_REQUIRED` 状态，未下载数据或运行训练。
- **Verification:** `bash -n scripts/check_governance_docs.sh` -> passed; `./scripts/check_governance_docs.sh` -> `governance documentation check: PASS`.
- **Initial commit:** `git commit -m 'Initial import of CAN research project'` created root commit `bdca23417a7b9382051463ffff80a6659c7ca339` on `main`, authored as `cyd56-ops <yandachen56@gmail.com>`; it contains the audited 103-file initial import.
- **Push attempt:** `git push -u origin main` returned `fatal: could not read Username for 'https://github.com': No such device or address`. `origin` remains the configured HTTPS URL, and `main` has no upstream. A follow-up probe found no configured credential helper, no installed `gh` CLI, and no running SSH agent; no credential, token, private key, or remote write was created.
- **Next operation:** after the project负责人 completes GitHub authentication in this environment, push `main` to `origin`. The local commit history is retained and no history rewrite is needed.

## 2026-08-12 - GitHub authentication blocker checkpoint

- Verified the post-push local state: `main` is clean at local commit `bdca23417a7b9382051463ffff80a6659c7ca339`, `origin` is `https://github.com/cyd56-ops/CAN.git`, and no upstream tracking branch exists.
- The repository cannot use HTTPS push without an authenticated credential, and no non-interactive credential helper is configured. `ssh-add -l` reports no authentication agent, so an SSH remote cannot be used without separately authorizing and loading a GitHub SSH key.
- Boundary held: no access token, password, private key, data, artifact, checkpoint, or local reference PDF was printed, stored, committed, or uploaded. GitHub has not received project content in this session.
- Commit-ready candidate file list: `PROJECT_WORKLOG.md` only. This local checkpoint records the authentic publication state and does not change V1-M1 `SERVER_REQUIRED`.

## 2026-08-12 - V1-M1 server availability probe

- Re-read the governing state, V1 model experiment decision, security boundary, and current V1-C1 implementation checkpoint. The current global next step remains V1-M1 and remains marked `SERVER_REQUIRED`.
- Re-ran required Git metadata checks. This workspace still has an empty/unusable `.git` directory; `git status --short --branch`, `git branch --show-current`, `git worktree list --porcelain`, and `git rev-parse HEAD` all exit with `fatal: not a git repository`. No branch, commit, or worktree state is claimed.
- Performed read-only local environment probes only. The host is `Linux 6.18.33.2-microsoft-standard-WSL2`; `nvidia-smi` is unavailable; PCI reports only `Microsoft Corporation Basic Render Driver`; `.venv/bin/python` reports Python `3.11.9`, `torch=2.13.0+cpu`, `torchvision=0.28.0+cpu`, `torch.cuda.is_available()=False`, `torch.version.cuda=None`, and CUDA device count `0`.
- Searched repository code/configuration for an existing server endpoint, connection configuration, or V1-M1 remote probe helper. None exists. This local CPU environment cannot be recorded as the required GPU tuple.
- Boundary held: no accelerator package installation, data download, formal data preparation, training, checkpoint generation, or new model artifact occurred. V1-M1 remains pending until an authorized GPU server environment is available; its tuple and deterministic policy must then be recorded before a GPU smoke check.
- Documentation verification: `bash -n scripts/check_governance_docs.sh` -> passed; `./scripts/check_governance_docs.sh` -> `governance documentation check: PASS`.
- Commit-ready candidate file list for this checkpoint: `PROJECT_WORKLOG.md` only. No Git commit is possible because this workspace is not a usable Git repository.

## 2026-08-12 - V1-C1-MSIS neural graph and A3-v2 route checkpoint

- **Branch/worktree:** `/home/kali/CAN` 不是 Git repository；`git status --short --branch`、`git branch
  --show-current`、`git worktree list --porcelain` 和 `git rev-parse HEAD` 都返回 exit 128，因此没有分支、
  完整 HEAD、worktree 状态或 staged file list。
- **Implementation:** 新增 `CAN-RELU-V1-MSIS-COEFF-v1` 的固定 public-profile compiler 与 dependency-free
  exact integer sparse affine/ReLU evaluator。canonical `(u,c,z)` 输入依序通过 exact negacyclic
  coefficient residual、完整 residual-multiple 三点 point pulses、coefficient norm violations 与 final hard
  conjunction，固定 topology 为 `56 -> 11056 -> 17 -> 1`。该 graph 使用 `int64` reduction、`int32`
  activations，编译 profile 不可变，运行时不导入或调用 exact reference、access 或 model fallback。
- **A3-v2 composition:** `V1NeuralAdapter` 仅将 neural evidence 映射为 profile/message/commitment/
  challenge/response/transcript-bound A3-v2 evidence；只有 coordinator 对 `NEURAL_ACCEPT` 提交后才调用
  protected operation。route-level tests 验收一次 accept/replay 最多一次 protected call、equation tamper
  reject 终结 transcript 且为零 protected calls、V0 foreign wire 不进入 neural route 且 verifier/protected
  calls 均为零。
- **Proof and claim boundary:** 对固定非生产 `N=8,q=257,k_mod=2,ell_mod=2,eta=1,gamma=8,kappa=2,B=6`
  profile，完整 canonical input range ledger 与 ReLU point-pulse identity 给出 `V_nn==V_ref`，从而包含
  `V_nn=1 -> V_ref=1`。这只是 compiled arithmetic conformance；不证明私钥持有、M-LWE/M-SIS、HVZK、
  主动冒充、Fiat--Shamir、不可伪造性、生产认证、NTT/PyTorch/qint8/CUDA/export 等价性或性能。
- **Exact checks run:**
  - `.venv/bin/python -m pytest tests/unit/test_v1_neural.py tests/differential/test_v1_neural_differential.py tests/security/test_v1_neural_security.py tests/integration/test_v1_neural_a3_v2_integration.py tests/unit/test_a3_v2.py tests/integration/test_v1_a3_v2_integration.py tests/security/test_v1_a3_v2_security.py -q` -> 47 passed.
  - `.venv/bin/python -m pytest` -> 595 passed in 37.68s.
  - `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` -> passed; 78 files already formatted.
  - `.venv/bin/mypy src tests` -> Success: no issues found in 78 source files.
  - `.venv/bin/python -m pip check` -> No broken requirements found; pip disabled its unwritable user cache.
  - `bash -n scripts/check_governance_docs.sh` and `./scripts/check_governance_docs.sh` -> passed.
- **Resource transition:** V1-C1 closes the final `LOCAL_OK` protocol/neural checkpoint. The unique next step is now
  V1-M1 GPU/software tuple freeze, marked `SERVER_REQUIRED`; project负责人已获通知。此 checkpoint 未下载
  CIFAR-100、未安装服务器环境、未运行训练，亦未生成模型、数据或 secret artifacts。
- **Commit-ready filesystem candidates:** `src/can/verifier/v1.py`, `src/can/verifier/__init__.py`,
  `src/can/access/v1_adapter.py`, `src/can/access/__init__.py`, `tests/unit/test_v1_neural.py`,
  `tests/differential/test_v1_neural_differential.py`, `tests/security/test_v1_neural_security.py`,
  `tests/integration/test_v1_neural_a3_v2_integration.py`, `README.md`, `SECURITY.md`,
  `docs/RESEARCH_DESIGN.md`, `docs/A3_CHALLENGE_RESPONSE_SPEC.md`, `docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`,
  `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md`, `docs/V1_MODEL_EXPERIMENT_DECISION.md`,
  `scripts/check_governance_docs.sh` and `PROJECT_WORKLOG.md`. 因目录不是 Git repository，这是 filesystem
  candidate list 而非 staged list。

## 2026-08-12 - V1-P2 A3-v2 fresh-transcript retry harness checkpoint

- **Branch/worktree:** `/home/kali/CAN` 仍不是 Git repository；`git status --short`、`git branch
  --show-current`、`git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128，因此无
  分支、commit 或 staged file list 可展示。
- **Implementation:** 扩展 `src/can/experiments/v1_psr.py`，增加 commitment-before-challenge 的
  prepared attempt、trusted challenge finish/abort、严格正整数 `max_attempts`、A3-v2
  fresh-transcript retry/exhaustion harness、nonce/transcript freshness digest、以及 sampler/response/
  exact/A3/total 分阶段 latency 记录。A3-v2 access state machine 本身未修改。
- **Protocol behavior:** 每次 retry 由 coordinator 新建 nonce、message/transcript、commitment、mask 和
  server challenge；abort、expiry 或终态 response 都只终结当前 transcript。旧 terminal wire object 的
  replay 和并发竞争均被拒绝；honest emitted response 经过 `verify_v1_ref` differential 后最多触发一次
  protected callback；retry exhaustion 的 verifier/protected calls 均为零。
- **Tests:** 新增 retry report 单元测试、forced-abort/first-success、exhaustion、expiry/freshness 集成
  测试和 budget/identity fail-closed 安全测试。focused command
  `.venv/bin/python -m pytest tests/unit/test_v1_psr.py tests/differential/test_v1_psr_differential.py
  tests/security/test_v1_psr_security.py tests/integration/test_v1_a3_v2_integration.py` 通过 58 项；
  `.venv/bin/python -m pytest` 通过 585 项；unit 372、differential 32、integration 13、security 168；
  `.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`、`.venv/bin/mypy src tests`、`.venv/bin/python
  -m pip check`、`bash -n scripts/check_governance_docs.sh` 和 `./scripts/check_governance_docs.sh` 均通过。
  `pip check` 仅提示用户 cache 目录不可写并禁用 cache。
- **Security/claim boundary:** 仍为本机非生产 toy experiment；retry 结果只支持 fresh state、有限域
  arithmetic completeness、协调器终态和零副作用 conformance，不支持生产 keygen/prover、M-LWE/M-SIS
  安全、HVZK、主动冒充、Fiat--Shamir 或不可伪造性结论。seed、secret、mask、response 和 transcript
  原文不进入公开报告。
- **Next step / resource:** 唯一下一步改为冻结并实现 `V1-C1-MSIS` coefficient-domain neural
  construction，仍为 `LOCAL_OK`；在该 neural checkpoint 闭合前不进入 CIFAR-100/ResNet-18、V2、
  Fiat--Shamir 或 Stage B。首次进入 V1-M1 GPU 环境冻结或正式 CIFAR 训练前，必须先把资源标记改为
  `SERVER_REQUIRED` 并通知项目负责人。
- **Commit-ready filesystem candidates:** `src/can/experiments/v1_psr.py`、
  `tests/unit/test_v1_psr.py`、`tests/integration/test_v1_a3_v2_integration.py`、
  `tests/security/test_v1_psr_security.py`、`README.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、
  `docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`、`docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md` 和
  `PROJECT_WORKLOG.md`。当前无 Git staged list；这些是文件系统候选。

## 2026-08-12 - V1-P2 generated-key/sampler/single-attempt implementation checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 仍返回 exit 128。当前目录不是 Git repository，
  无分支、worktree、有效 `HEAD` 或 staged file list。
- **Implementation:** 新增 `src/can/experiments/v1_psr.py`，实现 32-byte seed 的三个 domain-separated
  SHAKE256 role/counter streams、255/255/224 byte rejection、32-coordinate secret/mask、规范 112 个
  challenge、临时 `t=Abar*s` public profile、`u=Abar*y`、commit-first single attempt、未约减
  `z=y+c*s`、B=6 emit/abort、公开摘要 manifest 和拒绝覆盖写入策略。
- **Secret boundary:** `V1GeneratedKeyFixture` 只在 experiment 进程中持有临时 seed/secret，可用上下文
  管理或 `close()` 覆盖并释放 Python 对象；公开 profile、attempt `repr` 和 manifest 不含 seed、secret、
  mask、response 或 transcript 原文。该做法只是 toy hygiene，不是生产内存清零或真实 keygen。
- **Verification:** focused command
  `.venv/bin/python -m pytest tests/unit/test_v1_psr.py tests/differential/test_v1_psr_differential.py tests/security/test_v1_psr_security.py`
  51 项通过；覆盖固定跨实现 vectors、三个 byte-rejection 路线、完全均匀有限域计数、secret/mask
  边界、112 challenges、`X^8=-1`、5/6/7 emit、13-to-13 引理、commit-before-challenge、全部 challenge
  exact completeness、固定 sampler 首个 emit 和 artifact/lifecycle 负向测试。分组 unit 371、
  differential 32、integration 10、security 167 项通过；`.venv/bin/python -m pytest` 580 项通过；
  `.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`（73 files）、
  `.venv/bin/mypy src tests`（73 source files）、`.venv/bin/python -m pip check`、
  `bash -n scripts/check_governance_docs.sh` 和 `./scripts/check_governance_docs.sh` 通过。`pip check`
  另提示用户 cache 目录不可写并禁用 cache。
- **Scope:** 未修改 V1 exact verifier、A3-v2 coordinator、V0/A0、V1-P1、A3-v1 或 A4-C1；未实现
  A3-v2 retry、生产 keygen/prover、security estimator、neural/NTT、CIFAR、Fiat--Shamir、ML-DSA
  或 Stage B。
- **Next step / resource:** 唯一下一步推进到 A3-v2 fresh-transcript retry harness，仍为 `LOCAL_OK`，
  当前不需要服务器。
- **Commit-ready filesystem candidates:** `src/can/experiments/v1_psr.py`、
  `tests/unit/test_v1_psr.py`、`tests/differential/test_v1_psr_differential.py`、
  `tests/security/test_v1_psr_security.py`、`README.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`、
  `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md`、`PROJECT_WORKLOG.md` 和
  `scripts/check_governance_docs.sh`。当前目录不是 Git repository，因此这是本 checkpoint 的文件系统
  候选而非 staged list。

## 2026-08-12 - Server-resource trigger annotation checkpoint

- **Resource status:** 当前唯一下一步及随后 toy prover/A3-v2 retry、首个 dependency-free
  `V1-C1-MSIS` 均标记 `LOCAL_OK`，不需要服务器。
- **Trigger:** 当工作日志的唯一下一步首次进入 V1-M1 GPU tuple 冻结或 CIFAR-100/ResNet-18 正式
  训练时，必须先改为 `SERVER_REQUIRED` 并通知项目负责人；通知前不得安装正式服务器训练环境、
  下载正式训练数据或启动论文训练。
- **Expected server scope:** 单 NVIDIA GPU、至少 8 GiB VRAM/16 GiB RAM 为最低实用目标，优先
  12--16 GiB VRAM/32 GiB RAM；准确 GPU、driver/CUDA、PyTorch wheel、batch、epoch 和 deterministic
  policy 仍必须在 V1-M1 环境 checkpoint 单独冻结，当前不预先绑定具体服务器。
- **Changes:** 同步 `AGENTS.md`、V1-M1 决策、task board、唯一下一步资源标记和治理检查；不修改
  runtime、模型、数据、依赖或已验收 artifact。
- **Verification:** `bash -n scripts/check_governance_docs.sh` 和
  `./scripts/check_governance_docs.sh` 通过；`.venv/bin/python -m pytest` 529 项通过；
  `.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`（69 files）、
  `.venv/bin/mypy src tests`（69 source files）和 `.venv/bin/python -m pip check` 通过。`pip check`
  另提示用户 cache 目录不可写并禁用 cache。
- **Commit-ready filesystem candidates:** `AGENTS.md`、`PROJECT_WORKLOG.md`、
  `docs/V1_MODEL_EXPERIMENT_DECISION.md` 和 `scripts/check_governance_docs.sh`。当前目录不是 Git
  repository，因此这是本 checkpoint 的文件系统候选而非 staged list。

## 2026-08-12 - V1-P2 prover/sampler/rejection specification checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree、有效 `HEAD` 或 staged file list。
- **Specification:** 新增 `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md`，冻结非生产
  `V1-P2-PSR-E1`：`s in {-1,0,1}^32`、`y in [-8,8]^32`、112 个 fixed-weight ternary challenge、
  32-byte toy seed、SHAKE256 role/counter streams、无偏 byte rejection、generated-key/public-profile
  分离、`u=Abar*y`、`z=y+c*s`、`||z||_inf<=6` emit/abort、fresh retry 和临时 secret 生命周期。
- **Distribution lemma:** 对任意固定 toy `s,c`，每个 `(c*s)_i` 在 `[-2,2]`，truncation 恰把 13 个
  mask 值双射到 `[-6,6]`；因此 `p_emit=(13/17)^32=0.00018699146739962278`，期望
  `5347.837598722525` 次尝试。该高 abort rate 只解释 toy experiment，不支持性能、HVZK、M-LWE/
  M-SIS、Fiat--Shamir 或不可伪造性结论。
- **Vector/measurement plan:** 固定 secret/mask/challenge domain、convolution、norm 5/6/7、freshness、
  forced-abort/retry/exhaustion、exact differential 和 lifecycle families；manifest 只允许公开 profile/
  seed/challenge/commitment 摘要、计数和统计，不允许 secret、mask、response collection 或 replay token。
- **Governance:** README、SECURITY、RESEARCH_DESIGN、V1-P2 决策、工作日志和治理脚本已同步；治理
  脚本新增该规格文件和 18 个必需标题检查。没有修改 V0/A0、V1-P1、A3-v1、A4-C1、exact/A3-v2
  runtime、CIFAR、neural、Fiat--Shamir、ML-DSA 或 Stage B。
- **Verification:** `.venv/bin/python -m pytest` 529 项通过；`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`（69 files）、`.venv/bin/mypy src tests`（69 source files）、
  `.venv/bin/python -m pip check`、`bash -n scripts/check_governance_docs.sh` 和
  `./scripts/check_governance_docs.sh` 全部通过。`pip check` 仅提示用户 cache 不可写并禁用 cache。
- **Claim boundary:** 本 checkpoint 只冻结非生产实验契约和测试向量计划，没有实现 keygen、prover、
  sampler、rejection loop、A3 retry、security estimator、neural verifier 或生产密码库 adapter。
- **Artifacts:** 未生成或保存 secret、mask、response/transcript collection、random stream state、CIFAR
  data、model state、checkpoint、database 或报告；既有 ignored A2 artifacts 与 PDF 未修改。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`、
  `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md` 和 `scripts/check_governance_docs.sh`。目录不是 Git
  repository，因此这是准确 filesystem candidate list 而非 staged list。
- **Incomplete work:** generated-key fixture、deterministic sampler/vector manifest、prover/rejection
  runtime、A3-v2 retry、密码安全参数、`V1-C1-MSIS`、V1-M1、V2、Stage B 和系统 related-work 检索。
- **Next step:** 只执行第 6 节的 generated-key fixture/deterministic vector checkpoint；不在同一
  checkpoint 接入 coordinator retry、neural、CIFAR、Fiat--Shamir、ML-DSA 或 Stage B。

## 2026-08-11 - V1-P2 exact/A3-v2 and V1-M1 decision checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree、有效 `HEAD` 或 staged file list。
- **Exact profile/reference:** 冻结非生产 `N=8,q=257,k_mod=2,ell_mod=2,eta=1,gamma=8,kappa=2,B=6`
  profile、u32 public/commitment residues、fixed-weight ternary challenge、signed-i32 response 和 Python
  exact-int accumulator；新增 immutable public profile/registry、四类 canonical wire parsers、固定公开
  matrix/target、negacyclic convolution 与 exact `Abar*z=u+c*t`/norm evidence。
- **A3-v2 composition:** 新增独立 133-byte message domain、verification/input-profile binding digest、
  server challenge、transcript identifier、单进程 store、commit-first coordinator 和 V1 exact adapter。
  一个 parsed response、abort 或 expiry 原子终结 transcript；evidence 只含公开摘要，唯一 coordinator
  才提交 allow 并执行 opaque protected callback。
- **Security fix/tests:** `abort()` 现在与 `begin()`/`respond()` 一样把内部时钟等异常统一映射为固定
  deny。新增 integration/security tests 覆盖 exact accept/reject 组合、route/profile/type confusion、
  V0/V1-P1/A3-v1/A4 bytes 无 fallback、内部时钟失败、并发 32 次重复 response 仅一次
  verifier/allow/protected call，以及 public runtime objects 无 secret/authority fields。
- **Model decision:** `docs/V1_MODEL_EXPERIMENT_DECISION.md` 冻结 V1-M1 的独立
  `CAN-V1-CIFAR100-RESNET18-v1`、canonical `(1,3,32,32)` uint8/RGB input digest、CIFAR-style
  ResNet-18、A3-v2 adapter、baseline-before-gate、reproducibility、artifact 和性能验收边界；未下载
  CIFAR-100、未安装新 package、未实现模型或训练。
- **Verification:** V1 focused 命令 52 项通过；unit 333、differential 30、integration 10、security 156，
  完整 `.venv/bin/python -m pytest` 529 项通过；`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`（69 files）、`.venv/bin/mypy src tests`（69 source files）、
  `.venv/bin/python -m pip check`、`bash -n scripts/check_governance_docs.sh` 和治理检查均通过。
- **Claim boundary:** 当前 public fixture 可直接构造 valid relation，只证明 parser/exact arithmetic、
  request binding、单次终态和所测 fail-closed 性质；不证明 private-key possession、M-LWE/M-SIS
  concrete security、HVZK/special soundness、主动冒充安全、Fiat--Shamir、neural soundness 或授权安全。
- **Artifacts:** 未生成 secret、mask/rejection state、transcript collection、CIFAR data、model state、
  checkpoint、database 或报告；既有 ignored A2 artifacts 与 PDF 未修改。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`、
  `docs/V1_MODEL_EXPERIMENT_DECISION.md`、`scripts/check_governance_docs.sh`、
  `src/can/access/__init__.py`、`src/can/access/a3_v2.py`、`src/can/access/v1_adapter.py`、
  `src/can/reference/__init__.py`、`src/can/reference/v1.py`、`tests/_v1_support.py`、
  `tests/differential/test_v1_reference_differential.py`、`tests/unit/test_a3_v2.py`、
  `tests/unit/test_v1_reference.py`、`tests/integration/test_v1_a3_v2_integration.py` 和
  `tests/security/test_v1_a3_v2_security.py`。目录不是 Git repository，因此这是文件系统候选而非
  staged list。
- **Incomplete work:** V1-P2 prover/sampler/rejection loop、密码安全参数与 theorem-condition 核对、
  `V1-C1-MSIS`、V1-M1 data/input/model/training/gate、V2 ML-DSA、Stage B 和系统 related-work 检索均
  未完成。
- **Next step:** 只执行第 6 节的 V1-P2 non-production prover/sampler/rejection specification
  checkpoint；不在同一 checkpoint 实现 neural、CIFAR、Fiat--Shamir、ML-DSA 或 Stage B。

## 2026-08-11 - V1-P2 FSwA-S Module-SIS protocol-selection checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree、有效 `HEAD` 或 staged file list。
- **Route decision:** 按项目负责人决定，当前 V1 唯一实现目标切换为 V1-P2
  `CAN-V1-FSWA-MSIS-ID-v1`；V1-P1 普通矩阵 SIS 作为历史/简单实验 baseline 原样保留，V0/A0、
  A3-v1、A4-C1 和后续 V2 均不改写、不重命名，也不作为弱回退。
- **Reviewed protocol:** 新增 `docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`，选择 Boudgoust--Takahashi
  FSwA-S/vanilla-Dilithium baseline 的底层交互式 commit--challenge--response protocol。冻结
  `R_q=Z_q[X]/(X^N+1)`、`Abar=[A|I]`、`t=Abar*s`、`u=Abar*y`、server-sampled sparse ternary
  `c`、`z=y+c*s`、bounded-uniform rejection `||z||_inf<=B` 和 exact
  `Abar*z=u+c*t`。引用来源同时包括 Dilithium specification、Kiltz--Lyubashevsky--Schaffner 的
  canonical identification/QROM 分析和 Devevey et al. 的 Fiat--Shamir-with-aborts 修正分析。
- **Claim separation:** `t=A*s1+s2` 的 public-key distribution 是 M-LWE 问题，accepting-transcript
  soundness 是 M-SIS 问题，A3-v2 提供 message/input/scope/nonce/replay binding，未来 neural core 只
  证明 arithmetic soundness-preservation；交互式 V1-P2 不自动取得 Fiat--Shamir ROM/QROM 或
  ML-DSA 签名结论。
- **Composition/neural decision:** V1-P2 使用新的 A3-v2 commit-first transcript、固定 wire domain 和
  单次终态 response。首个 exact semantic backend 必须使用 coefficient-domain negacyclic
  convolution；未来 `V1-C1-MSIS` 才编译 `Abar*z-u-c*t` 的 modular-zero checks、系数范数和 final
  AND。NTT、HighBits/LowBits、compression、hints、SHAKE encoding 和完整 ML-DSA 保留到后续。
- **Documentation/governance:** 同步 `README.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、
  `docs/A3_CHALLENGE_RESPONSE_SPEC.md`、`docs/A4_GPV_RELATION_SPEC.md`、
  `docs/A4_NEURAL_CONSTRUCTION_DECISION.md`、V1-P1 baseline、本日志和治理脚本；路线一致性现覆盖
  九份文档，V1-P2 决策文档具有独立 18-heading 检查。
- **Verification:** `bash -n scripts/check_governance_docs.sh`、`./scripts/check_governance_docs.sh`、
  `.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`（60 files）、
  `.venv/bin/mypy src tests`（60 source files）、`.venv/bin/python -m pip check` 和完整
  `.venv/bin/python -m pytest`（477 passed in 44.64s）全部通过。`pip check` 只有已知 cache 不可写
  警告，无 broken requirements。
- **Security boundary:** 本 checkpoint 只冻结 reviewed protocol、编码/组合边界和证明责任，不实现
  concrete parameters、keygen、prover、uniform sampler、rejection loop、exact/neural verifier、
  Fiat--Shamir signature 或生产认证保证。引用原方案不表示尚未选择的 CAN 参数继承其安全级别。
- **Artifacts:** 未生成或保存 secret polynomial、private key、mask/rejection state、transcript
  collection、model state、checkpoint、database、data 或 report；现有 ignored A2 artifacts/PDF 未修改。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A3_CHALLENGE_RESPONSE_SPEC.md`、
  `docs/A4_GPV_RELATION_SPEC.md`、`docs/A4_NEURAL_CONSTRUCTION_DECISION.md`、
  `docs/V1_PROTOCOL_SELECTION_DECISION.md`、`docs/V1_MODULE_SIS_PROTOCOL_DECISION.md` 和
  `scripts/check_governance_docs.sh`。目录不是 Git repository，因此这是准确 filesystem 候选而非
  staged list；没有运行时代码或测试文件改动。
- **Incomplete work:** V1-P2 concrete conformance profile/range ledger/public fixtures、exact reference、
  A3-v2、prover/sampler、`V1-C1-MSIS`、V2 ML-DSA、Stage B 和系统 related-work 检索均未完成。
- **Next step:** 只执行第 6 节的 V1-P2 concrete-profile/exact-reference/A3-v2 checkpoint；不得修改
  V0/A0、V1-P1、A3-v1 或 A4-C1，也不在同一 checkpoint 实现 sampler、neural verifier、NTT、
  Fiat--Shamir、ML-DSA 或 Stage B。

## 2026-08-11 - V1-P1 protocol selection and route-preservation checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Protocol decision:** 新增 `docs/V1_PROTOCOL_SELECTION_DECISION.md`，选择 Lyubashevsky 2012
  标准矩阵方案经 Liu--Zhandry 2019 抽取的交互式 SIS Sigma protocol，协议标识为
  `CAN-V1-LYU12-SIS-ID-v1`。冻结短秘密 `S`、公开 `A,T=A*S mod q`、commitment `a=A*y`、稀疏
  ternary challenge、response `r=y+S*c`、rejection sampling/abort 与 exact relation
  `A*r=T*c+a mod q`、`sum(r_j^2)<=B2`。
- **Composition decision:** V1-P1 要求 commitment 先于 server challenge，因此现有 A3-v1 不直接
  复用为 wire protocol；后续新增 A3-v2，绑定既有 133-byte A3 message、profile/commitment digest、
  challenge 和 transcript ID，并让一个 parsed response attempt 原子终结。Fiat--Shamir 签名转换继续
  延期，不能由交互式协议自动推出 ROM/QROM 或不可伪造结论。
- **Neural compatibility:** V1 的模等式可借鉴 A4-C1 point-pulse 证明模式，但当前 A4-C1 固定
  `q=257`、`8x72`、signed-int8 与无穷范数，不能直接承载 V1 profile-sized matrix、signed-int32
  response 和 squared `L2` bound；必须新增 V1 exact relation 与后续 `V1-C1`。
- **Route preservation:** 按项目负责人要求，V0、V1、V2 作为独立可复现路线长期共存。V1/V2 不得
  重命名、改写、替换或覆盖 V0/A0、A3-v1 或 A4/V1-prep；后续只新增独立协议标识、registry、parser、
  adapter、evidence 类型和测试，并显式覆盖 route confusion/no-fallback。
- **Documentation/governance:** 同步 `README.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、
  `docs/A3_CHALLENGE_RESPONSE_SPEC.md`、`docs/A4_GPV_RELATION_SPEC.md`、
  `docs/A4_NEURAL_CONSTRUCTION_DECISION.md` 和本日志；治理脚本新增 V1 文档及 18 个标题检查，路线
  一致性现覆盖八份文档。
- **Verification:** `bash -n scripts/check_governance_docs.sh`、`./scripts/check_governance_docs.sh`、
  `.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`（60 files）、
  `.venv/bin/mypy src tests`（60 source files）、`.venv/bin/python -m pip check` 和完整
  `.venv/bin/python -m pytest`（477 passed in 37.76s）全部通过。`pip check` 只有已知 cache 不可写
  警告，无 broken requirements。
- **Security boundary:** 本 checkpoint 只完成协议选择、组合接口和安全主张分离，不实现 concrete
  parameters、keygen、prover、Gaussian/rejection sampler、exact/neural verifier、Fiat--Shamir 签名
  或认证安全结论。引用原论文不表示 CAN 参数或实现继承其定理。
- **Artifacts:** 没有生成或保存 secret matrix、private key、sampler state、transcript collection、
  model state、checkpoint、database、data 或 report；现有 ignored A2 artifacts 和 PDF 未修改。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A3_CHALLENGE_RESPONSE_SPEC.md`、
  `docs/A4_GPV_RELATION_SPEC.md`、`docs/A4_NEURAL_CONSTRUCTION_DECISION.md`、
  `docs/V1_PROTOCOL_SELECTION_DECISION.md` 和 `scripts/check_governance_docs.sh`。目录不是 Git
  repository，因此这是准确 filesystem 候选而非 staged list；没有运行时代码或测试文件改动。
- **Incomplete work:** V1-P1 concrete conformance profile、range ledger、public fixtures、exact reference、
  A3-v2、prover/sampler、`V1-C1`、V2 ML-DSA、Stage B 和系统 related-work 检索均未完成。
- **Next step:** 只执行第 6 节的独立 V1 exact-reference/A3-v2 checkpoint；不修改 V0/A0、A3-v1、
  A4-C1，不在同一 checkpoint 实现 sampler、neural verifier、Fiat--Shamir、ML-DSA 或 Stage B。

## 2026-08-11 - A4-C1 fixed neural graph and V1-prep closure checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Construction decision:** 新增 `docs/A4_NEURAL_CONSTRUCTION_DECISION.md` 并冻结 A4-C1
  `CAN-RELU-A4-PFDH-TOY-v1`。canonical `y||z` 为 80 个 scale-1 整数，固定三层 topology 为
  `80->3600->1153->1`，使用 `K=-72..71` 的 point pulses、norm violation accumulator 和最终 ReLU
  conjunction；dense slot count 为 4,444,707，sparse nonzero upper bound 为 257,185。
- **Proof and implementation:** 对全部 canonical `(y,z)` 证明 `V_nn==V_ref`，并在
  `src/can/verifier/a4.py` 实现 dependency-free sparse affine/ReLU graph。固定 profile 只由本地可信
  A4 public profile 编译，运行时没有 `%`、Floor、比较、float、客户端 profile 或 reference fallback；
  `src/can/access/a4_adapter.py` 新增 neural evidence adapter，仍只由 A3 coordinator 提交授权。
- **Tests:** 新增 9 unit、4 differential、4 security 和 2 integration tests，覆盖全部 residual
  point-pulse 标量域、全部 signed-int8 norm 标量域、canonical relation differential、profile 冻结、
  client key/profile/decision 注入拒绝、reference fallback 禁止和 A3 tamper/replay/零额外模型调用。
- **Existing-test isolation:** 完整套件首次运行得到 476 passed/1 failed；失败是既有 A2 tiny-training
  测试在多线程 CPU 下未固定 deterministic algorithms，权重摘要偶发漂移。按同文件 public-baseline
  已有模式，仅在测试内保存/恢复全局设置并固定单线程 deterministic execution；该 isolated test 三次
  独立运行均通过，未修改 A2 模型、训练实现、参数或已验收 artifact。
- **Verification:** A4 neural focused 21、A4 exact focused 39、unit 302、differential 28、integration 8、
  security 139 和完整 pytest 477 项通过；Ruff lint、Ruff format（60 files）和 mypy（60 source files）
  通过。shell syntax、治理文档和 `pip check` 由本 checkpoint 最终复核。
- **Security boundary:** 本 checkpoint 闭合的是 toy canonical relation 的 exact neural compilation 与
  A3 evidence composition，不是 GPV signer/keygen、知识证明、身份认证或不可伪造性。公开 gadget proof
  仍可任意构造；V1 必须另选 reviewed protocol 并独立分析 Fiat-Shamir/challenge/rejection 安全。
- **Artifacts:** 没有生成或保存 private key、trapdoor、proof collection、model state、checkpoint、
  database、data 或 report；现有 ignored A2 artifacts 与 PDF 未修改。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A3_CHALLENGE_RESPONSE_SPEC.md`、
  `docs/A4_GPV_RELATION_SPEC.md`、`docs/A4_NEURAL_CONSTRUCTION_DECISION.md`、
  `scripts/check_governance_docs.sh`、`src/can/access/__init__.py`、
  `src/can/access/a4_adapter.py`、`src/can/verifier/__init__.py`、`src/can/verifier/a4.py`、
  `tests/conftest.py`、`tests/unit/test_a2_baseline.py`、`tests/unit/test_a4_neural.py`、
  `tests/differential/test_a4_neural_differential.py`、`tests/integration/test_a4_a3_integration.py` 和
  `tests/security/test_a4_neural_security.py`。目录不是 Git repository，因此这是准确 filesystem 候选
  而非 staged list。
- **Next step:** 只执行第 6 节的 V1 protocol-selection 决策；不在同一 checkpoint 实现 signer/keygen、
  ML-DSA、Stage B 或第二 neural backend。

## 2026-08-11 - V0/V1-prep/V1/V2 roadmap alignment checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Route change:** 按项目负责人决定固定 `V0 -> V1-prep -> V1 -> V2` 长期路线，同时保留现有
  A0--A4 工程编号。V0 映射到 A0/A1/A2；V1-prep 映射到 A3 binding/freshness 与 A4 canonical
  `(y,z)` 神经代数内核；V1 后续另选 reviewed challenge-response/身份协议；V2 最后研究 ML-DSA。
- **Claim boundary:** A4 neural completeness/soundness 只构成 V1 可复用编译内核，不证明知识可靠性、
  不可伪造性、身份授权或 replay 安全。若 V1 采用 noisy LWE、rounding、decomposition 或 hint，必须
  扩展 exact relation，不能把 `A*z=y mod q` 无条件复用。V2 不属于首篇论文 MVP，也不承诺任意剪枝、
  微调或未验证 export 后保持等价。
- **Documentation:** 同步 `README.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、
  `docs/A3_CHALLENGE_RESPONSE_SPEC.md` 和 `docs/A4_GPV_RELATION_SPEC.md`；治理脚本新增六份路线文档的
  `V1-prep` 边界检查。没有修改运行时代码、协议编码、测试逻辑或已验收实验 artifact。
- **Exact checks run:** `bash -n scripts/check_governance_docs.sh`、
  `./scripts/check_governance_docs.sh`、`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`、`.venv/bin/mypy src tests`、
  `.venv/bin/python -m pip check` 和 `.venv/bin/python -m pytest` 均通过；完整 pytest 为 458 passed，
  format 为 56 files，mypy 为 56 source files。`pip check` 仅提示用户 cache 目录不可写并禁用 cache。
- **Incomplete work:** A4 固定神经构造与全输入证明、V1 具体协议选择/实现、安全参数和 V2 ML-DSA
  reference/neural modules 均未完成；本 checkpoint 只调整研究路线与文档约束。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A3_CHALLENGE_RESPONSE_SPEC.md`、
  `docs/A4_GPV_RELATION_SPEC.md` 和 `scripts/check_governance_docs.sh`。目录不是 Git repository，因此
  这是准确 filesystem 候选而非 staged list。
- **Next step:** 仍只执行第 6 节的 A4/V1-prep fixed neural construction/proof 决策；V1 协议选择、
  V2 ML-DSA 和 Stage B 均等待其闭合。

## 2026-08-11 - A4 GPV exact public-relation and A3 adapter checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Reviewed relation:** 定向核验 GPV STOC 2008 原始论文第 5 节及 NIST FIPS 204 比较边界，选择
  GPV probabilistic full-domain-hash 的短原像公开验证谓词；没有把 targeted review 描述为完整
  related-work 检索。
- **Specification:** 新增 `docs/A4_GPV_RELATION_SPEC.md`，固定非生产
  `A4-GPV-PFDH-TOY-v1`：`q=257`、`8 x 72` 满行秩公开矩阵、32 字节 salt、72 个 signed-int8
  系数、`||z||_inf <= 1`、105 字节 proof、SHAKE256 rejection-sampled syndrome、exact relation、A3
  evidence 边界和 canonical `(y,z)` neural completeness/soundness 义务。
- **Implementation:** 新增 `src/can/reference/a4.py` 和 `src/can/access/a4_adapter.py`；公开 profile 在
  构造期复制并校验 shape/range/mod-257 满秩，reference 严格解析 message/proof 并精确检查 norm 与
  `A*z mod q`，adapter 只产生 A3 message/identity-bound evidence。没有 signer、keygen、trapdoor、
  私钥、A0/A1 fallback、模型调用或全局授权状态；A3 旧占位码由 `AUTHENTIC_ACCEPT` 收紧为中性的
  `PROOF_ACCEPT`，避免 toy gadget relation 在代码层面先行声称认证。
- **Tests:** 新增 22 unit、2 integration 和 13 security tests，覆盖 signed-int8/105-byte parsing、
  fixed hash vector、公开配置不可变/满秩、norm/equation/message tamper、key/profile/decision 注入、
  no-authority/no-secret AST 边界，以及 A3 invalid-proof retry、atomic consume 和 replay 单次模型调用。
  valid fixture 使用公开 gadget binary decomposition，刻意无需私钥且可公开伪造。
- **Verification:** A4 focused 37、unit 293、integration 6、security 135 和 full pytest 458 项通过；
  Ruff lint、Ruff format（56 files）、mypy（56 source files）、`pip check`、shell syntax 和治理文档
  检查通过。一次将 unit 与 security 并行运行时，既有 A2 tiny-training state hash 测试出现漂移；按
  规定独立串行复跑 unit 为 293 passed，完整套件独立运行为 458 passed。
- **Security boundary:** 当前结果只支持 exact toy public relation、无私钥 adapter 和所测 A3
  组合；小参数未经 estimator 选择，具体 SHAKE256 映射不自动继承 GPV random-oracle 归约，公开
  gadget fixture 不证明私钥持有、不可伪造、公钥认证、neural soundness、白盒或生产安全。
- **Artifacts:** 没有生成或保存 private key、trapdoor、proof collection、model state、checkpoint、
  database、data 或 report；现有 ignored A2 artifacts 与 PDF 未修改。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A3_CHALLENGE_RESPONSE_SPEC.md`、
  `docs/A4_GPV_RELATION_SPEC.md`、`scripts/check_governance_docs.sh`、
  `src/can/access/__init__.py`、`src/can/access/a2_capability.py`、
  `src/can/access/a3_protocol.py`、`src/can/access/a4_adapter.py`、
  `src/can/reference/__init__.py`、`src/can/reference/a4.py`、`tests/conftest.py`、
  `tests/unit/test_a3_protocol.py`、`tests/unit/test_a4_reference.py`、
  `tests/integration/test_a4_a3_integration.py`、`tests/security/test_a3_protocol_security.py` 和
  `tests/security/test_a4_reference_security.py`。目录不是 Git repository，因此这是准确 filesystem
  候选而非 staged list。
- **Next step:** 只执行第 6 节的 A4 fixed neural construction/proof 决策，不实现完整 ML-DSA、
  signer/keygen、Stage B 或第二 backend。

## 2026-08-08 - A3-v1 runtime protocol shell checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Implementation:** 新增 `src/can/access/a3_protocol.py` 并从 `src/can/access/__init__.py` 导出。
  实现 A3-v1 exact 133-byte message/parser、canonical detached image snapshot/hash、可信单进程
  `A3NonceStore`、version-3 challenge/protected/deny envelopes、默认关闭的
  `A3ProtocolCoordinator` 和只产生 evidence 的 `A3VerificationProfile`/`A3Evidence`。
- **Coordinator boundary:** challenge 只从本地 profile、trusted model、trusted clocks/CSPRNG 创建；
  response 在 verifier 前复核 message/profile/input/expiry/state，exact evidence 后在线性化临界区
  原子执行 `PENDING -> CONSUMED`，仅唯一成功者调用 protected model。A4 缺失时固定 deny；不接入
  A0/A1 evidence、A2 public response、客户端 route/policy/decision 或任何 fallback。
- **Tests:** 新增 `tests/unit/test_a3_protocol.py` 和 `tests/security/test_a3_protocol_security.py`，
  覆盖 canonical round-trip/tamper、negative zero、默认关闭、invalid-proof retry、expiry/clock rollback、输入替换、
  A1 evidence 注入、并发 replay、nonce collision、未知 identity、非规范 proof、独立 store、请求字段
  注入和 post-commit model failure。deterministic `b"valid"` proof 只用于协议壳测试，不代表认证。
- **Verification:** A3 focused `.venv/bin/python -m pytest tests/unit/test_a3_protocol.py tests/security/test_a3_protocol_security.py` 为 29 passed；完整 `.venv/bin/python -m pytest` 为 421 passed；`.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`（51 files）和 `.venv/bin/mypy src tests`（51 source files）通过；`bash -n scripts/check_governance_docs.sh`、`./scripts/check_governance_docs.sh` 和 `.venv/bin/python -m pip check` 均通过，后者仅有已知用户 cache 不可写警告。
- **Security boundary:** A3 只闭合单进程 request binding/freshness/at-most-once 壳；没有 A4 proof
  不可伪造性、公钥认证、分布式/durable state、TLS/channel binding、DoS、白盒或生产保证。
- **Commit-ready filesystem candidates:** `PROJECT_WORKLOG.md`、`README.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`scripts/check_governance_docs.sh`、`src/can/access/__init__.py`、
  `src/can/access/a3_protocol.py`、`tests/unit/test_a3_protocol.py` 和
  `tests/security/test_a3_protocol_security.py`。目录不是 Git repository，因此这是准确 filesystem
  候选而非 staged list；ignored A2 data/states/reports 与 PDF 未修改。
- **Next step:** 进入 A4，先选择 reviewed public-key lattice relation 并冻结 exact reference/neural
  verifier 契约；不在同一 checkpoint 实现完整 ML-DSA 或 Stage B。

## 2026-08-08 - A3-v1 challenge-response protocol specification checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Specification:** 新增 `docs/A3_CHALLENGE_RESPONSE_SPEC.md`，固定 `CAN-A3-BOUND-CHALLENGE-v1`：
  133-byte `CAN-A3-MSG-v1` 编码绑定 version、local model、32-byte identity、scope、issued/expiry、
  32-byte nonce 和 canonical input SHA-256；TTL 固定 60 秒，proof 保持 opaque 并由未来 A4 本地
  profile 进一步收紧。
- **Input/state boundary:** canonical image 使用 detached/cloned CPU contiguous float32
  `(1,1,28,28)`，拒绝 non-finite、range drift 和 negative zero，再以固定 big-endian IEEE-754
  编码 hash；nonce store 固定单进程 thread-safe linearizable `PENDING -> CONSUMED`，原子 consume
  内复核 message/profile/input binding 与 monotonic expiry，consume 后不回滚。
- **Authorization boundary:** verifier 只产生绑定 exact message digest/identity 的 evidence；唯一
  coordinator 只有在 exact accept 和 atomic consume 唯一成功后才能进入 protected model。A0/A1
  numeric evidence、A2 public response、client key/profile/boolean/decision 和 fallback 均被排除；A4
  未激活前 A3 protected entry 默认关闭。
- **Security games:** 固定 canonical binding、per-nonce at-most-once invocation、expiry linearization、
  tamper、no-downgrade 和 18-row acceptance matrix；区分 pre-commit 零调用拒绝与 consume 后 model
  failure 已进入一次模型的事实。deterministic proof stub 仅可用于下一 checkpoint 的协议测试，不能
  被描述为认证或不可伪造性证据。
- **Documentation:** 同步 `README.md`、`SECURITY.md` 和 `docs/RESEARCH_DESIGN.md`，修正
  `SECURITY.md` 中已过时的 accepted-state/report 状态；治理脚本现将 A3 规格和 18 个标题作为必需
  文档检查。
- **Verification:** `bash -n scripts/check_governance_docs.sh` 和
  `./scripts/check_governance_docs.sh` 通过；canonical message probe 得到 domain `14`/total `133`
  bytes；`.venv/bin/python -m pytest` 为 392 passed；`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`（48 files）和 `.venv/bin/mypy src tests`（48 source files）通过；
  `.venv/bin/python -m pip check` 无 broken requirements，仅有已知用户 cache 不可写警告。
- **Artifacts:** 本 checkpoint 没有生成 key、credential、nonce database、model state、checkpoint、
  data 或实验 report；现有 ignored A2 local states/reports 未修改。
- **Security boundary:** 本 checkpoint 只闭合单进程协议规格，不实现 proof verifier、请求绑定运行时或
  replay 防护，不提供身份认证、签名不可伪造、分布式原子性、durable consume、TLS/channel binding、
  DoS、白盒或生产保证。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A3_CHALLENGE_RESPONSE_SPEC.md` 和
  `scripts/check_governance_docs.sh`。目录不是 Git repository，因此这是准确 filesystem 候选而非
  staged list；`artifacts/`、`data/`、`paper/*.pdf` 和其他 ignored 文件不在列表中。
- **Incomplete work:** A3 runtime codec/parser/store/coordinator shell、A4 公钥格签名 relation/neural
  verifier、qint8/CUDA/export、系统 related-work 检索和 Stage B 均未实现。
- **Next step:** 只执行第 6 节的 A3-v1 默认关闭协议壳实现，不在同一 checkpoint 选择 A4 relation 或
  扩大到 Stage B。

## 2026-08-08 - A2-E2 accepted-state materialization and empirical report checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short`、`git branch --show-current`、
  `git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128。当前目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`；环境只提供空的只读 `.git` 目录。
- **Implementation:** 完成 `src/can/experiments/a2_materialize.py`。trusted materializer 在两个固定
  `PYTHONHASHSEED` 子进程中分别重建 protected/public accepted baseline，只在 canonical state 与
  prediction digest 均匹配后保存 CPU float32 `state_dict`；随后生成严格 manifest，再由独立
  no-training 子进程加载 state 并调用既有 `run_a2_capability_experiment`。修复了 `Path` 默认实例被
  exact-class 判断误拒绝、`python -m` 子进程错误使用 `__main__` 和受限加载异常未导入 `pickle`。
- **State boundary:** manifest 使用 duplicate-key/non-finite 拒绝和 canonical JSON，固定校验 runtime、
  seed、epoch、数据根/资源摘要、文件名、拓扑、参数量、canonical state 与文件 SHA-256。loader
  使用 `weights_only=True`，拒绝 symlink、超限文件、非 `OrderedDict`、错误 key/dtype/device/layout、
  非有限 tensor、拓扑或 state digest 漂移；state/manifest 均为 `0600` local ignored artifacts。
- **Empirical result:** protected/public canonical state SHA 分别精确匹配
  `88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613` 与
  `b71980ebd3fb6e1a729b77109c98d3b4580e9e9cf8d3a28296cf6c18d1c122be`。10,000-image 三态报告的
  prediction SHA 分别匹配 `e5b48d60...e4a7` 与 `f54b2351...6f0a`；完整计数为 10,000 public calls、
  10,000 protected calls 和 1 次零模型调用 reject。默认 public-off probe 同样零模型调用。
- **Latency:** 最终 no-training 复核运行的 public/protected/deny end-to-end median 为
  `238.7/1540.9/1324.4 us`，p95 为 `339.7/1985.6/1826.8 us`，verifier-only median 为
  `1101.7 us`。这些是当前 WSL2/单 CPU thread 的实验观察，不是跨平台或生产保证。
- **Tests:** 新增 9 项 materializer unit tests 和 5 项 security tests，覆盖 state/manifest round trip、
  入口子进程/seed 顺序、重复/非规范 JSON、协议/数据/state/file tamper、覆盖保护、dtype/device/layout/
  finiteness、symlink、restricted loading 和 AST no-training 边界。A1 artifact 测试保持“backend 调用
  前后无新增 artifact”，同时适配 D-024 已允许存在的 ignored A2 local states。
- **Verification:** capability/materializer focused 93、unit 252、differential 24、integration 4、
  security 112 和 full pytest 392 项通过；Ruff lint、Ruff format（48 files）、mypy（48 source files）、
  `pip check`、shell syntax 和治理文档检查通过。`pip check` 仅有已知 cache 目录不可写警告，无 broken
  requirements。
- **Artifacts:** `artifacts/a2/local-states/protected-state.pt` 为 943,381 bytes/file SHA
  `b0cd10fd...4f84f`，`public-state.pt` 为 203,869 bytes/file SHA `e020b45c...b671`，manifest 为
  1,818 bytes，最终 `capability.json` 为 10,630 bytes。它们全部被 `.gitignore` 排除，不是提交候选；
  optimizer、credential、图像、logits、feature 或完整模型 pickle 均未保存。
- **Security boundary:** 结果只支持当前固定黑盒入口上的模型调用隔离、预测等价和无 fallback；A0
  credential 仍可 replay 且未绑定业务输入，不支持认证、不可伪造、白盒不可绕过、模型保密或生产
  访问控制结论。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`、
  `src/can/experiments/a2_materialize.py`、`tests/unit/test_a2_materialize.py`、
  `tests/security/test_a2_materialize_security.py` 和
  `tests/security/test_a1_torch_backend_security.py`。目录不是 Git repository，因此这是准确 filesystem
  候选而非 staged list；`artifacts/`、`data/`、`paper/*.pdf` 和其他 ignored 文件不在列表中。
- **Incomplete work:** A3 challenge-response/request binding/replay 状态、A4 公钥格签名、
  qint8/CUDA/export、系统 related-work 检索和 Stage B 均未实现。
- **Next step:** 只执行第 6 节的 A3 协议规格冻结，不在同一 checkpoint 实现认证或扩大到 Stage B。

## 2026-07-29 - A2-E2 three-state coordinator implementation checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、
  `git branch --show-current`、`git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128；
  当前目录不是 Git repository，无分支、worktree 或有效 `HEAD`。
- **Coordinator:** 新增 `src/can/access/a2_capability.py` 和导出，保持 A2-E1 version-1 协调器不变。
  A2-E2 使用默认关闭且精确 bool 的本地 policy、固定启动审计事件、一个实例内互斥三态提交、
  version-2 deny/public/protected envelopes、线程安全独立计数和有界 timing。public entry 不接收
  credential；protected entry 只信任固定 A1-B1 evidence，reject 不降级到 public。
- **Model boundary:** 构造和每次调用复核 exact model class/topology/parameter metadata/eval/hooks，
  public/protected 参数以底层 storage identity 保持分离；协调器独立复核 logits 的 exact tensor、
  dtype/device/shape/layout/finiteness。图像在验证后复制为 contiguous snapshot，request 额外字段统一
  deny。所选模型异常返回固定 deny，但保持一次已提交的 selected decision 和准确 entered-call 计数，
  不二次提交或调用另一模型。
- **Experiment runner:** 新增 `src/can/experiments/a2_capability.py`。它严格解析并交叉核对两次
  protected/public baseline reports，拒绝重复 JSON 字段、类型混淆、非有限或不一致指标，只接受
  state SHA 等于 D-019/D-022 已验收值的两个内存模型；随后才允许执行 10,000 张直接/三态标签
  等价、default-off probe、完整计数、response size 和三路 latency。入口没有训练调用、CLI 路由、
  credential、policy、backend 或 report-path override。
- **Tests:** 新增/扩展 79 项 focused unit/integration/security tests，覆盖默认关闭、三态成功/拒绝、
  malformed images/credentials、inactive backend、policy/model drift、unknown route/policy/model/head/
  backend/evidence/decision fields、底层 storage overlap、非规范 logits、双向模型异常、verifier/reference
  无 fallback、public response 重放/重标记、三态并发精确计数、public 对 protected state/output 零影响、
  report duplicate/tamper/path/schema 和 AST no-training 边界。
- **Verification:** `.venv/bin/python -m pytest tests/unit/test_a2_capability.py
  tests/unit/test_a2_capability_experiment.py tests/integration/test_a2_capability_integration.py
  tests/security/test_a2_capability_security.py tests/security/test_a2_capability_experiment_security.py` 为
  79 passed；完整 `.venv/bin/python -m pytest` 为 378 passed；`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`（45 files）和 `.venv/bin/mypy src tests`（45 source files）通过。
  分组 unit 243、differential 24、integration 4、security 107 均通过；`pip check` 无 broken
  requirements，`bash -n scripts/check_governance_docs.sh` 和治理文档检查通过。
- **Empirical blocker:** baseline reports 的真实摘要已由新读取器核验为 protected `88.08%`、public
  `99.85%`，但 earlier checkpoints 删除了临时 state files，workspace 也没有 `.pt/.pth/ckpt/
  safetensors/ONNX` 模型 artifact。D-022 禁止本 checkpoint 重训，因此未运行真实已验收权重的
  10,000-image 三态评估，也未生成 `artifacts/a2/capability.json`；随机初始化模型只用于结构测试。
- **Security boundary:** 当前证据支持所测黑盒入口的调用隔离、输出范围和无 fallback，不支持 A0
  credential 的认证、不可伪造、新鲜性或输入绑定，也不支持 endpoint authorization、白盒不可绕过、
  模型保密或生产部署。public 响应只是数据，不是 bearer capability。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`、
  `scripts/check_governance_docs.sh`、`src/can/access/__init__.py`、
  `src/can/access/a2_capability.py`、`src/can/experiments/a2_capability.py`、
  `tests/unit/test_a2_capability.py`、`tests/unit/test_a2_capability_experiment.py`、
  `tests/integration/test_a2_capability_integration.py`、
  `tests/security/test_a2_capability_security.py` 和
  `tests/security/test_a2_capability_experiment_security.py`。目录不是 Git repository，故这是准确
  filesystem 候选而非 staged list；ignored data/reports/license/PDF 不在列表中。
- **Incomplete work:** 只剩已验收权重的三态完整报告被 state materialization 阻塞；MASK/共享
  head/trunk、qint8/CUDA/export、A3、Stage B 和安全承载格关系继续明确延期。
- **Next step:** 只执行第 6 节的 accepted-state 三态报告；没有可信 states 时先取得项目负责人决定。

## 2026-07-29 - A2-E2 independent public coarse-model baseline checkpoint

- **Branch/worktree:** `/home/kali/CAN`；四个 Git 只读命令均返回 exit 128，目录不是 Git
  repository，无分支、worktree 或有效 `HEAD`。
- **Changes:** 新增 `src/can/model/a2_public_mlp.py`，实现不导入 protected model 的
  `CAN-A2-FMNIST-PUBLIC-MLP-v1`（`784->64->2`、50,370 float32 parameters）、严格图像/public
  label 校验与固定 Fashion-MNIST source-to-coarse 映射。新增
  `src/can/experiments/a2_public_baseline.py`，独立固定数据摘要/split、public seed/loader、Adam 十
  epoch、二分类评估、model/state digest、临时序列化、latency、ignored report 和 repeat compare；
  未修改 `A2AccessCoordinator` 或 A2-E1 protected 模型/训练入口。
- **Tests:** 新增 41 项 focused unit/security tests，覆盖独立拓扑/参数 storage、finite logits、
  image/label type/dtype/shape/layout/finiteness/range、精确 coarse mapping、数据 resource tamper、固定
  split/hash、单线程 tiny determinism、report identity、repeat 类型混淆、空 latency、AST import
  隔离、CLI route/training/authorization 注入和固定 artifact 路径。
- **Public baseline:** 两个独立进程均得到 test loss `0.007989783663357957`、accuracy `99.85%`
  (`9985/10000`)、class support `7000/3000`、confusion `[[6989,11],[4,2996]]`、prediction SHA
  `f54b2351606f21ff31fc7c23ed394c4dbe13ccb9b150a7fe10b6b27076926f0a`、state SHA
  `b71980ebd3fb6e1a729b77109c98d3b4580e9e9cf8d3a28296cf6c18d1c122be` 和 fingerprint
  `e4fbf9c09afc3aaada32dd60f7368346a64138497178618c78ac0b1baeb4c14f`。
- **Resource/latency:** public 模型为 201,480 parameter bytes，临时序列化 203,849 bytes 后删除；
  两次 batch-1 median 为 `70.2/65.9 us`，batch-256 median 为 `1464.0/1340.001 us`，peak RSS
  与 latency 不属于 deterministic fingerprint。这些只适用于当前 WSL2/i7-1260P/CPU tuple。
- **Verification:** public focused 41、unit 203、differential 24、integration 2、security 70、full
  pytest 299 项通过；Ruff lint/format（38 files）、mypy（38 source files）、`pip check`、shell
  syntax 和治理脚本通过。`pip check` 只有已知用户 cache 目录不可写提示，无 broken requirements。
- **Artifacts:** ignored `artifacts/a2/public-baseline-repeat-{1,2}.json` 为 5,528/5,530 bytes；字段
  扫描不含 credential、secret、原始 image、logits、features、evidence 或 capability。ignored roots
  外没有模型/checkpoint/pickle/NumPy artifact。
- **Security boundary:** public baseline 是无门控 coarse 分类器，不产生 decision、context 或
  capability，不证明模型隔离、认证、不可伪造、replay/输入绑定、endpoint authorization、白盒
  不可绕过或生产安全；`99.85%` 不是安全指标。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`、
  `docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`、`scripts/check_governance_docs.sh`、
  `src/can/model/a2_public_mlp.py`、`src/can/experiments/a2_public_baseline.py`、
  `tests/unit/test_a2_public_mlp.py`、`tests/unit/test_a2_public_baseline.py`、
  `tests/security/test_a2_public_baseline_security.py`。目录不是 Git repository，故这是本 checkpoint
  的准确文件系统候选而非 staged list；ignored data/report/license/PDF 不在列表中。
- **Incomplete work:** public entry/三态 coordinator、完整调用矩阵、MASK/共享 head/trunk、
  qint8/CUDA/export、A3、Stage B 和安全承载格关系均未实现。
- **Next step:** 只执行第 6 节的 A2-E2 本地绑定三态协调器实验 checkpoint。

## 2026-07-29 - A2-E2 capability-tier experiment specification checkpoint

- **Branch/worktree:** `/home/kali/CAN`；`git status --short --branch`、
  `git branch --show-current`、`git worktree list --porcelain` 和 `git rev-parse HEAD` 均返回 exit 128；
  目录不是 Git repository，无分支、worktree 或有效 `HEAD`。
- **Decision:** A2-E2 主路线固定为独立 `784->64->2` public float32 CPU MLP，输出仅为
  footwear/non-footwear coarse class，与 protected MLP 不共享权重、head、feature 或 artifact。
  public entry 默认关闭并由本地可信部署配置绑定；同一协调器是 `DENY`/`PUBLIC`/`PROTECTED`
  唯一提交点，protected 验证失败只能 deny，禁止 fallback 到 public。
- **Specification:** 新增 `docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`，固定 public/protected 功能与
  version-2 envelope、entry/request 边界、不可升级/重标记/复用规则、模型/artifact 分离、public
  训练常量、完整调用计数验收矩阵、延迟/指标和两阶段实现顺序。该文档明确 A2-E2 capability 只是
  不可转移的单请求内部决定，不是 Stage B bearer token。
- **Documentation/governance:** 同步 `README.md`、`docs/RESEARCH_DESIGN.md`、
  `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md` 和 `SECURITY.md`；治理脚本现强制新规格及 17 个标题存在。
- **Verification:** `.venv/bin/python -m pytest` 为 258 passed；`.venv/bin/ruff check .`、
  `.venv/bin/ruff format --check .`（33 files）、`.venv/bin/mypy src tests`（33 source files）、
  `.venv/bin/python -m pip check`、`bash -n scripts/check_governance_docs.sh` 和
  `./scripts/check_governance_docs.sh` 全部通过。`pip check` 仅报告用户 cache 目录不可写并禁用
  cache，没有 broken requirements。
- **Artifacts:** 本 checkpoint 没有运行训练、生成模型、checkpoint、数据或实验 report，也没有修改
  运行时代码。
- **Security boundary:** 规格只定义固定黑盒入口的可测调用/输出隔离，不支持认证、不可伪造、
  replay/输入绑定、endpoint authorization、白盒不可绕过或生产安全主张。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`、
  `docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`、`scripts/check_governance_docs.sh`。目录不是 Git
  repository，故这是本 checkpoint 的准确文件系统候选而非 staged list；ignored data/report/PDF
  不在列表中。
- **Incomplete work:** public model/baseline、三态协调器、public entry、MASK/共享 head/trunk、
  qint8/CUDA/export、A3、Stage B 和安全承载格关系均未实现。
- **Next step:** 只执行第 6 节的 A2-E2 独立 public coarse-model 无门控 baseline checkpoint。

## 2026-07-29 - A2-E1 single coordinator and binary hard gate checkpoint

- **Branch/worktree:** `/home/kali/CAN`；四个 Git 只读命令均返回 exit 128，目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Changes:** 新增 `src/can/access/a2_gate.py`，实现只接受单张规范业务图像与原始 credential 的
  `A2AccessCoordinator`、固定 allow/deny envelopes、精确 evidence 类型检查、唯一提交点、线程安全
  调用计数与有界无敏感计时；本地配置固定 exact A1-B1 backend 和 A2 MLP，任何非精确
  `NUMERIC_ACCEPT`、输入错误、配置漂移或异常均 deny，不调用 reference/dependency-free fallback。
  新增 `src/can/experiments/a2_gate.py`，复用确定性训练并执行全测试集标签等价、拒绝探针和固定延迟
  方法；baseline 仅提取共享私有训练 helper，不改变模型、数据、optimizer 或已接受结果。
- **Tests:** 新增 41 项 A2 gate unit/integration/security tests，覆盖合法 allow 的提交/调用顺序、
  parse/profile/config/numeric 拒绝、业务输入类型/shape/finiteness、evidence/decision/backend/policy
  注入、错误 evidence 类型、verifier 异常、backend 失活、模型配置漂移、无 fallback、accepted
  replay、并发 rejected replay、固定响应、计数/计时不可回写、gate report 和 CLI 参数边界。
- **Gate experiment:** `PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_gate --run`
  重训得到 test accuracy `88.08%`、prediction SHA
  `e5b48d60c19304e54c412416abd0201e9c747afd00830b93af9122a738a2e4a7` 和 state SHA
  `88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613`；全部 10,000 个 gated
  labels 匹配 baseline，计数为 10,000 verifier/commit/protected calls。rejected probe 为一次 deny、
  零 protected calls。
- **Latency:** 本机 model-only、accepted、rejected 和 verifier-only batch-1 median 分别为
  `99.0/1849.2/1570.5/1245.5 us`，accepted coordinator median `85.4 us`，accepted overhead
  `1750.2 us`/`1767.88%`；1,100 次 rejected latency 请求均为零 protected calls。结果仅适用于当前
  WSL2/i7-1260P/CPU tuple，不声称 constant-time 或跨平台性能。
- **Verification:** A2 focused 73、unit 165、differential 24、integration 2、security 67、full pytest
  258 项通过；Ruff lint/format（33 files）、mypy（33 source files）、`pip check` 和治理脚本通过。
  `pip check` 另提示用户 cache 目录不可写并禁用 cache，但没有 broken requirements。
- **Artifacts:** ignored `artifacts/a2/gate.json` 为 6,423 bytes，不含 credential、secret、图像、
  logits、features 或 evidence；ignored roots 外没有模型/checkpoint/pickle/NumPy artifact。
- **Security boundary:** 该 checkpoint 支持固定黑盒入口上所测拒绝路径零模型调用和响应不泄露，
  不支持身份认证、不可伪造、replay/业务输入绑定、白盒不可绕过、生产部署或安全承载密码主张。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`、
  `scripts/check_governance_docs.sh`、`src/can/access/__init__.py`、`src/can/access/a2_gate.py`、
  `src/can/experiments/a2_baseline.py`、`src/can/experiments/a2_gate.py`、`tests/conftest.py`、
  `tests/unit/test_a2_gate.py`、`tests/unit/test_a2_gate_experiment.py`、
  `tests/integration/test_a2_gate_integration.py`、`tests/security/test_a2_gate_security.py`、
  `tests/security/test_a2_gate_experiment_security.py`。目录不是 Git repository，故这是准确文件系统
  候选而非 staged list；ignored data/reports/license/PDF 不在列表中。
- **Incomplete work:** A0 credential 仍可 replay 且不绑定业务输入；没有 public capability、MASK、
  LeNet/MNIST、qint8/CUDA/export、A3 challenge-response、阶段 B 或安全承载格关系。
- **Next step:** 只执行第 6 节的 A2-E2 public/protected capability 分级实验规格 checkpoint。

## 2026-07-23 - A2-E1 deterministic MLP baseline checkpoint

- **Branch/worktree:** `/home/kali/CAN`；四个 Git 只读命令均返回 exit 128，目录不是 Git repository，
  无分支、worktree 或有效 `HEAD`。
- **Environment/data:** 从官方 CPU index 安装并核验
  `torchvision-0.28.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`，SHA-256
  `1dad604dfc0177ecebe0891bd9701fe2c62ec3f7819a247be541b3fb6effee99`；固定 torch
  `2.13.0+cpu`、NumPy `2.4.4`、Pillow `12.2.0` 并新增 `requirements-ml.lock`。四个
  Fashion-MNIST resource MD5 与协议一致，另记录 SHA-256；decoded train/test 为 60,000/10,000、
  uint8 images/int64 labels，train/validation index hashes 已固定。上游 MIT license blob 为
  `6bc221fc3a933bbcba0161c926d69e4897382eda`，ignored snapshot SHA-256 为
  `13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1`。
- **Changes:** 新增 `src/can/model/a2_mlp.py`，实现 235,146-parameter
  `784->256->128->10` float32 CPU MLP，并严格拒绝错误 type/dtype/device/shape/layout、NaN/Inf、
  越界 image/label。新增 `src/can/experiments/a2_baseline.py`，固定环境、数据/split hash、seeds、
  Adam 十 epoch、evaluation、canonical model digest、临时序列化和 100 warm-up + 1,000 latency 方法；
  CLI 不允许选择 hyperparameter、device、data root、credential 或授权参数。
- **Baseline:** 两次独立同种子进程均得到 test loss `0.33665058851242063`、accuracy `88.08%`
  (`8808/10000`)、prediction SHA `e5b48d60c19304e54c412416abd0201e9c747afd00830b93af9122a738a2e4a7`、
  state SHA `88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613` 和 fingerprint
  `a59a9a9ac2797261eb824af564d6fa64a3c3e19fa43886b2349aa48bccaf7d53`。batch-1 median 为
  110.8/104.9 us，batch-256 median 为 2987.4/2790.8 us；这些只是在当前 WSL2 i7-1260P 上的
  model-only 结果。
- **Verification:** focused A2 32、unit 144、differential 24、integration 1、security 48、full pytest
  217 项通过；Ruff lint/format（25 files）、mypy（25 source files）、`pip check` 均通过。最终治理、
  shell syntax、尾随空白、artifact 和 Git 状态检查记录在本节关闭前的 verified command table。
- **Artifacts:** ignored `data/a2/`、license 和两个 JSON reports 保留用于复现；临时 943,357-byte
  state file 已删除，项目 ignored roots 外没有模型/checkpoint/pickle/NumPy artifact。
- **Security boundary:** 本 checkpoint 没有 coordinator、gate、capability 或安全承载密码；分类
  accuracy 不证明认证或访问控制。A0 credential 仍可 replay。
- **Commit-ready filesystem candidates:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A1_BACKEND_DECISION.md`、
  `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`、`pyproject.toml`、`scripts/check_governance_docs.sh`、
  `requirements-ml.lock`、`src/can/model/a2_mlp.py`、`src/can/experiments/a2_baseline.py`、
  `tests/unit/test_a2_mlp.py`、`tests/unit/test_a2_baseline.py`、
  `tests/security/test_a2_baseline_security.py`。目录不是 Git repository，故这是准确文件系统候选而非
  staged list；ignored data/reports/license/PDF 不在列表中。
- **Next step:** 只执行第 6 节的 A2-E1 单一协调器与二元硬门控 checkpoint。

## 2026-07-23 - A2-E1 minimum model experiment protocol checkpoint

- **Branch/worktree:** `/home/kali/CAN`；四个 Git 只读命令均返回 exit 128，目录不是 Git
  repository，无分支、worktree 或有效 `HEAD`。
- **Decision:** 新增 `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`，唯一选择
  `CAN-A2-FMNIST-MLP-v1`：Fashion-MNIST、`784->256->128->10` float32 CPU MLP、固定
  55,000/5,000 split seeds、十 epoch Adam、`>=85%` smoke floor、两次同种子复现、准确率/延迟/
  参数指标、固定响应 envelope、evidence -> 唯一协调器 -> protected model 和拒绝零调用测试。
- **Package/data probe:** 官方 CPU index 的只读 `pip index` 查询列出
  `torchvision 0.28.0+cpu`；本地已存在但未安装的 CPython 3.11 torchvision 0.28.0 wheel metadata
  声明 `torch==2.13.0`、Python `>=3.10,!=3.14.1`、NumPy/Pillow，并给出 Fashion-MNIST 四个
  resource MD5。`torchvision`、NumPy 和 Pillow module spec 仍为 `None`，没有安装 package 或下载
  dataset。实时获取上游 LICENSE 时外部 GitHub 连接失败，因此协议要求数据获取 checkpoint 重新
  核验 MIT license、保存来源/hash 且不再分发。
- **Security and scope:** 请求方只能提交 business input 和 raw credential，不能提交 evidence、
  decision、backend、profile 或 policy；只有 exact `NUMERIC_ACCEPT` 可进入内部 allow，所有其他
  情况固定 deny 且要求 protected call count 为零。A0 replay、黑盒假设和 toy/非生产限制保留；
  public capability、MASK、qint8/CUDA/export、Stage B 和安全承载密码均未进入实现。
- **Exact checks:** unit 114、differential 24、integration 1、security 46、full pytest 185 全部通过；
  full run 为 `185 passed, 1 warning`，warning 仍是未安装可选 NumPy。Ruff lint 通过，format check
  为 20 files，mypy 20 source files 通过，`pip check`、`bash -n scripts/check_governance_docs.sh`
  和治理检查通过；尾随空白检查无匹配。项目 `.venv`/`paper` 外没有 `.pt`、`.pth`、`.ckpt`、
  ONNX、safetensors、pickle 或 NumPy artifact。
- **Checkpoint files:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、
  `docs/RESEARCH_DESIGN.md`、`docs/A1_BACKEND_DECISION.md`、
  `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`、`pyproject.toml`、`scripts/check_governance_docs.sh`。当前
  不是 Git 仓库，这是准确文件系统候选而非 staged list。
- **Incomplete work:** 没有安装 torchvision/NumPy/Pillow、获取 Fashion-MNIST、训练 MLP、生成 ML
  环境锁、实现协调器/硬门控或取得业务零调用/性能结论。
- **Next step:** 只执行第 6 节的 A2-E1 无门控 Fashion-MNIST MLP baseline checkpoint；baseline
  验收后才实现二元硬门控。

## 2026-07-23 - A1-B1 PyTorch CPU exact backend implementation checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支、worktree 或有效 `HEAD`。
- **Environment:** 按决定命令从官方 CPU index 安装
  `torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`；缓存 wheel body SHA-256 为
  `6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b`。metadata/runtime 均为
  `2.13.0+cpu`，CUDA/HIP 均为 `None`，CUDA unavailable，torchvision absent；setuptools 已从
  CPU index 解析的 78.1.0 恢复为项目固定的 83.0.0。
- **Changes:** 新增 `src/can/verifier/a1_torch.py`，只暴露可信 registry backend 构造和 raw
  23-byte evidence adapter；三层只执行 `mul -> sum(dtype=int64) -> add -> clamp(min=0) ->
  int32 cast`，weights/biases 为 CPU non-persistent `int32` buffers，零 Parameters、空
  `state_dict()`。startup gate 核验支持 tuple、operator micro-probe、range ledger、buffer
  content/layout 和实际 profile 的逐分量全域分解；每次调用复核环境/module contract，异常禁用
  实例且不回退。
- **Tests added:** 新增 `tests/unit/test_a1_torch_backend.py`、
  `tests/differential/test_a1_torch_differential.py` 和
  `tests/security/test_a1_torch_backend_security.py`，覆盖 513 residual、129 distance、9 AND
  sums、每 slot/component 全部 `b_i=0..256`、A0 core/guard/reject/wrap/bit-zero/mixed/malformed
  向量、environment/device/dtype/shape/stride/content/persistence/hook 漂移、operator exception、
  concurrency/replay、optional import、no-fallback 和无 artifact。
- **Verification:** 旧 134 项 baseline 通过；unit 114、differential 24、integration 1、security
  46、full 185 项通过；Ruff lint/format、mypy 20 source files、`pip check`、脚本语法和治理检查
  均通过。PyTorch import 报告一次缺少可选 NumPy 互操作 warning；整数 backend 不使用 NumPy。
- **Artifacts and scope:** 项目目录未生成 `.pt`/`.pth`/`.ckpt`/ONNX/safetensors/pickle；没有
  安装 torchvision、下载数据、实现业务模型/协调器/capability 或进入 qint8/CUDA/export。
- **Checkpoint files:** `src/can/verifier/a1_torch.py`；三个上述测试文件；`README.md`、
  `SECURITY.md`、`docs/RESEARCH_DESIGN.md`、`docs/A1_NUMERICAL_SPEC.md`、
  `docs/A1_CONSTRUCTION_DECISION.md`、`docs/A1_BACKEND_DECISION.md` 和本工作日志。目录不是 Git
  仓库，因此这是准确文件清单而非 Git staged list。
- **Residual risk:** 结论只覆盖指定 WSL2/Linux x86_64/CPython 3.11/PyTorch CPU wheel 和 toy
  relation；不提供认证、不可伪造、白盒、跨平台、业务零调用或生产安全保证。
- **Next step:** 仅固定 A2 最小数据集/模型/训练/指标/artifact/硬门控实验协议；再下一 checkpoint
  才开始模型 baseline 实验。

## 2026-07-23 - A1 PyTorch backend decision checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `docs/A1_BACKEND_DECISION.md`，选择 A1-B1
  `CAN-TORCH-CPU-EXACT-v1`；固定 Linux x86_64/CPython 3.11/PyTorch `2.13.0+cpu` 官方 CPU
  wheel、三层 tensor layout、`int32` storage/product、`int64` reduction/pre-activation、scale `1`、
  zero-point `0`、eager `mul/sum/add/clamp` 映射、startup activation gate、no-fallback、artifact、
  禁用/迁移和完整复测契约。同步 README、研究设计、安全文档、A1 数值/构造文档和治理脚本。
  没有安装 PyTorch、下载数据集、实现 backend/模型/协调器或写出 compiled artifact。
- **Environment result:** 只读探测得到 WSL2 Linux `6.18.33.2`、x86_64、glibc 2.38、Intel
  i7-1260P/16 logical CPUs/AVX2、CPython 3.11.9；`torch`/`torchvision` 未安装，
  `nvidia-smi`/`rocminfo` 不可用，因此首目标只支持 CPU。PyTorch 官方页面在探测时标记 2.13.0
  stable；官方 CPU index 存在
  `torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`，SHA-256 为
  `6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b`。
- **Security invariants:** 请求方不能选择 tensor/device/dtype/scale/candidate/operator；qint8、CUDA、
  float、fusion 和 export 不进入 A1-B1；compiled buffers 只在内存/pytest 临时目录生成并
  non-persistent；任何版本、device、layout、range、trace 或 artifact gate 失败都返回
  `CONFIG_REJECT` 并禁用 backend，绝不调用 dependency-free、exact-ops 或 `V_ref` fallback。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `.venv/bin/python -c 'import platform, sys; print(platform.platform()); print(platform.machine()); print(platform.libc_ver()); print(sys.version.split()[0])'` -> Linux WSL2 x86_64、glibc 2.38、Python 3.11.9。
  - `.venv/bin/python -c 'import importlib.util; print(importlib.util.find_spec("torch")); print(importlib.util.find_spec("torchvision"))'` -> 两项均为 `None`。
  - `command -v nvidia-smi` 与 `command -v rocminfo` -> 均无匹配（exit 1）。
  - `lscpu`、`free -h`、`df -h .` -> 只读探测通过；结果记录在 backend 决策第 3 节。
  - `.venv/bin/python -m pytest tests/unit` -> 通过，100 tests。
  - `.venv/bin/python -m pytest tests/differential` -> 通过，12 tests。
  - `.venv/bin/python -m pytest tests/integration` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/security` -> 通过，21 tests。
  - `.venv/bin/python -m pytest` -> 通过，134 tests。
  - `.venv/bin/ruff check .` -> 通过。
  - `.venv/bin/ruff format --check .` -> 通过，16 files already formatted。
  - `.venv/bin/mypy src tests` -> 通过，16 source files。
  - `.venv/bin/python -m pip check` -> 通过，No broken requirements found；pip cache 权限 warning
    不影响结果。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
- **Test result:** 当前 134 项 dependency-free 基线无回归，Ruff、格式、mypy、依赖、shell 语法和
  治理检查全部通过。由于 PyTorch 未安装，本 checkpoint 没有执行 A1-B1 kernel、dtype、trace、
  artifact 或性能测试，不能声称 backend 等价性。
- **Incomplete work:** A1-B1 安装/实现/全域复测、torchvision/数据集、qint8/CUDA/export、
  LeNet/MLP、协调器、系统 related-work 检索和安全承载格关系尚未完成。
- **Commit-ready candidate files:** 本 checkpoint 准确修改
  `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、
  `docs/A1_NUMERICAL_SPEC.md`、`docs/A1_CONSTRUCTION_DECISION.md`、
  `docs/A1_BACKEND_DECISION.md`、`scripts/check_governance_docs.sh`。连同上一 checkpoint 尚无 Git
  基线可区分的实现候选，累计文件系统候选还包括 `src/can/verifier/__init__.py`、
  `src/can/verifier/a1.py`、`tests/differential/README.md`、
  `tests/differential/test_a1_differential.py`、`tests/security/test_a1_verifier_security.py` 和
  `tests/unit/test_a1_verifier.py`；这不是 `git status` 输出。
- **Next step:** 执行第 6 节定义的 A1-B1 CPU backend 安装、实现和全域性质复测，不进入 A2。

## 2026-07-23 - A1 dependency-free verifier implementation checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `src/can/verifier/a1.py` 并从包入口导出 A1-C1 常量、不可变 affine/ReLU layer、compiled profile/compiler、无回退 compiled registry、稳定 evidence 和公共 `verify_a1` adapter；私有 core 只执行固定三层 affine/ReLU，测试 trace 与授权 evidence 分离。新增 29 个 A1 单元测试、12 个差分测试和 13 个 A1 安全测试，更新 differential README、README、研究/安全/A1 文档和治理脚本。没有安装 PyTorch、写出 compiled artifact 或实现业务模型/协调器。
- **Numerical result:** 实现穷尽全部 `u=-256..256`、`d=0..128`、AND 和值 `0..8` 和八个位置的 `b_i=0..256`；ReLU trace 与 `V_ref` 逐分量距离一致，issuer-core 零 false reject、全域零 false accept，reference guard 9--12 按设计保守拒绝。
- **Security invariants:** 只接受 A0 exact bytes parser 的规范 credential；compiled profile/registry 拷贝并冻结所有输入且无 profile/slot 回退；请求方不能提交 anchor、weight、threshold、scale、candidate、evidence 或 gate；公开 evidence 只含结果码；内部异常返回 `CONFIG_REJECT`，instrumentation 证明不调用 `V_ref`/exact-ops fallback；无全局可变 secret/profile/授权状态。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `.venv/bin/python -m pytest tests/unit/test_a1_verifier.py` -> 通过，29 tests。
  - `.venv/bin/python -m pytest tests/differential/test_a1_differential.py` -> 通过，12 tests。
  - `.venv/bin/python -m pytest tests/security/test_a1_verifier_security.py` -> 通过，13 tests。
  - `.venv/bin/python -m pytest tests/unit` -> 通过，100 tests。
  - `.venv/bin/python -m pytest tests/differential` -> 通过，12 tests。
  - `.venv/bin/python -m pytest tests/integration` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/security` -> 通过，21 tests。
  - `.venv/bin/python -m pytest` -> 通过，134 tests。
  - `.venv/bin/ruff check .` -> 通过。
  - `.venv/bin/ruff format --check .` -> 通过，16 files already formatted。
  - `.venv/bin/mypy src tests` -> 通过，16 source files。
  - `.venv/bin/python -m pip check` -> 通过，No broken requirements found；pip cache 权限 warning 不影响结果。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
  - `rg -n '[[:blank:]]+$' README.md PROJECT_WORKLOG.md SECURITY.md docs scripts src tests` -> 无匹配，exit 1 表示尾随空白检查通过。
- **Test result:** A1 focused 54 tests 和完整 134 项套件通过，Ruff、格式、mypy、依赖和治理检查全部通过；当前结论限于 Python exact-integer conformance backend，不证明 PyTorch/量化等价、认证或业务模型零调用。
- **Incomplete work:** A1 PyTorch/量化 backend 决策与实现、设备/安装渠道、转换后性质测试、LeNet/MLP、协调器、系统 related-work 检索和安全承载格关系尚未完成。
- **Commit-ready candidate files:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、`docs/A1_NUMERICAL_SPEC.md`、`docs/A1_CONSTRUCTION_DECISION.md`、`scripts/check_governance_docs.sh`、`src/can/verifier/__init__.py`、`src/can/verifier/a1.py`、`tests/differential/README.md`、`tests/differential/test_a1_differential.py`、`tests/security/test_a1_verifier_security.py`、`tests/unit/test_a1_verifier.py`。当前无 Git 仓库，故这是文件系统候选列表而非 `git status` 输出。
- **Next step:** 执行第 6 节定义的 A1 deployment backend 决策，不安装 PyTorch 或实现业务模型。

## 2026-07-23 - A1 fixed ReLU construction decision checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `docs/A1_CONSTRUCTION_DECISION.md`，选择 `CAN-RELU-EXACT-v1` 固定整数 ReLU 主构造和互斥 `A1-EXACT-OPS-v1` 对照；固定允许算子、`t` bias 折叠、`8->40->16->1` 拓扑、逐层公式/range、dense/sparse 参数计数、手工分段证明、完整有限域穷举和下一实现契约。同步 README、A1 数值规格、研究设计、安全文档和治理脚本；没有安装 PyTorch 或实现代码。
- **Construction result:** 每个分量用 5 个 ReLU 精确计算 bounded modular distance、2 个 ReLU 精确计算整数阈值，最终 1 个 ReLU 计算八路 AND，共 57 个 ReLU；所有主图数值为 `int32`/scale `1`，语义误差为 `0`。普通 `%`/Floor/`abs`/compare 只作测试基线，Sigmoid 不进入主路径，MASK 延期到 A2，显式 runtime `A*s` 只作 compiler audit。
- **Security invariants:** 主 core 只允许固定 affine/ReLU；compiled `t` 和 bias 按 toy secret-bearing artifact 管理；任何 input/profile/config 异常 fail closed；主 adapter 禁止 `V_ref`/exact-ops fallback；网络输出仍只是 evidence 来源，不具有 gate、decision、authorization 或 capability。A0 chosen-`b`、replay、输入替换和白盒风险仍未解决。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `.venv/bin/python -m pytest tests/unit` -> 通过，71 tests。
  - `.venv/bin/python -m pytest tests/integration` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/security` -> 通过，8 tests。
  - `.venv/bin/python -m pytest` -> 通过，80 tests。
  - `.venv/bin/ruff check .` -> 通过。
  - `.venv/bin/ruff format --check .` -> 通过，12 files already formatted。
  - `.venv/bin/mypy src tests` -> 通过，12 source files。
  - `.venv/bin/python -m pip check` -> 通过，No broken requirements found；pip cache 权限 warning 不影响结果。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
  - `.venv/bin/python -c 'r=lambda x:max(0,x); f=lambda u:-129+r(-u)+2*r(u+129)-r(u+1)-2*r(u)+2*r(u-128); exact=lambda u:abs((u%257)-128); assert all(f(u)==exact(u) for u in range(-256,257)); assert all(r(9-d)-r(8-d)==int(d<=8) for d in range(129)); assert all(r(s-7)==int(s==8) for s in range(9)); assert all(f(b-t)==abs(((b-t)%257)-128) for b in range(257) for t in range(257)); print("PASS: 513 residuals, 129 distances, 9 conjunction sums, 66049 b/t pairs")'` -> 通过。
  - `.venv/bin/python -c 'r=lambda x:max(0,x); rows=[(r(-u),r(u+129),r(u+1),r(u),r(u-128)) for u in range(-256,257)]; assert tuple(max(row[i] for row in rows) for i in range(5))==(256,385,257,256,128); d=[-129+h0+2*h1-h2-2*h3+2*h4 for h0,h1,h2,h3,h4 in rows]; assert (min(d),max(d))==(0,128); assert 8*40+40*16+16+40+16+1==1033; assert 40+16*5+16+40+16+1==193; print("PASS: layer ranges and dense/sparse parameter counts")'` -> 通过。
  - `rg -n '[[:blank:]]+$' README.md PROJECT_WORKLOG.md SECURITY.md docs scripts/check_governance_docs.sh` -> 无匹配，exit 1 表示尾随空白检查通过。
- **Test result:** 现有 80 项代码测试无回归，构造公式与完整有限标量域、层范围和参数计数独立检查通过；Ruff、格式、mypy、依赖和治理检查全部通过。本 checkpoint 只新增构造决定和治理约束，没有新增代码模块或已执行 verifier 性质。
- **Incomplete work:** A1 compiled profile/compiler、affine/ReLU evaluator、adapter/evidence、全域代码测试、backend 语义验证、LeNet/MLP、协调器、PyTorch 设备配置和系统 related-work 检索尚未完成。
- **Commit-ready candidate files:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、`docs/A1_NUMERICAL_SPEC.md`、`docs/A1_CONSTRUCTION_DECISION.md`、`scripts/check_governance_docs.sh`。当前无 Git 仓库，故这是文件系统候选列表而非 `git status` 输出。
- **Next step:** 执行第 6 节定义的 dependency-free A1 verifier conformance backend，不安装 PyTorch 或实现模型。

## 2026-07-23 - A1 numerical and operator specification checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `docs/A1_NUMERICAL_SPEC.md`，固定 credential/业务输入隔离、`A1-INT32-S1` API、可信 compiled profile、规范相位锚点、共同 tensor/算子范围、round-to-nearest ties-to-even 扩展规则、溢出拒绝、误差组合、全输入 soundness、evidence-only 边界和未来差分测试；同步 README、研究设计、安全文档和治理脚本。没有安装 PyTorch，也没有实现 verifier、LeNet/MLP 或协调器。
- **Numerical decision:** 对 A0 本地相位使用规范证明接口 `t=(A_slot*s_test) mod 257`，使共同运行时残差位于 `[-256,256]`；显式 `A*s` 与常量折叠仍为候选选择。modulo 不连续边界不得套用未经分支证明的普通 Lipschitz 界，必须逐分支、穷尽 513 个残差或使用形式方法直接证明组合误差。
- **Security invariants:** 非规范 shape/dtype/scale/range 在 core 前拒绝；阈值和 AND 精确执行且不占用误差预算；候选必须满足逐分量距离误差 `<=4` 和 `V_nn=1 -> V_ref=1`；evidence 不含授权能力，失败不得调用 `V_ref` 回退。该规格不解决 A0 的 chosen-`b`、replay、输入替换或白盒风险。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `.venv/bin/python -m pytest tests/unit` -> 通过，71 tests。
  - `.venv/bin/python -m pytest tests/integration` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/security` -> 通过，8 tests。
  - `.venv/bin/python -m pytest` -> 通过，80 tests。
  - `.venv/bin/ruff check .` -> 通过。
  - `.venv/bin/ruff format --check .` -> 通过，12 files already formatted。
  - `.venv/bin/mypy src tests` -> 通过，12 source files。
  - `.venv/bin/python -m pip check` -> 通过，No broken requirements found；pip cache 权限 warning 不影响结果。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
  - `.venv/bin/python -c 'from can.reference import A0_CENTER, A0_MODULUS, center_q, mod_q; assert all((u := b - t) >= -256 and u <= 256 and (k := u // A0_MODULUS) in (-1, 0) and 0 <= (p := u - A0_MODULUS * k) < A0_MODULUS and abs(p - A0_CENTER) == abs(center_q(mod_q(b - t) - A0_CENTER)) for b in range(A0_MODULUS) for t in range(A0_MODULUS))'` -> 通过，穷尽全部 66,049 个规范 `b/t` 组合。
  - `test $((4 + 4)) -le 8 && test $((8 + 4)) -le 12 && test $((13 - 4)) -gt 8` -> 通过，issuer-core completeness、reference-radius soundness 和 first-reject 边界关系成立。
  - `rg -n '[[:blank:]]+$' README.md PROJECT_WORKLOG.md SECURITY.md docs scripts/check_governance_docs.sh` -> 无匹配，exit 1 表示尾随空白检查通过。
- **Test result:** 现有 80 项代码测试无回归，Ruff、格式、mypy、依赖和治理检查全部通过；本 checkpoint 只新增规格和治理约束，没有新增代码模块或实现性质。
- **Incomplete work:** A1 主构造/允许算子/证明方法选择、神经 verifier、全域误差证书、LeNet/MLP、协调器、PyTorch 设备配置和系统 related-work 检索尚未完成。
- **Commit-ready candidate files:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、`docs/A1_NUMERICAL_SPEC.md`、`scripts/check_governance_docs.sh`。当前无 Git 仓库，故这是文件系统候选列表而非 `git status` 输出。
- **Next step:** 执行第 6 节定义的 A1 构造决策，不安装 PyTorch 或实现模型。

## 2026-07-23 - Authentication-neuron concept route refinement checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 根据外部初步方案材料完善 `docs/RESEARCH_DESIGN.md`、`SECURITY.md` 和本工作日志；采纳独立验证模块、credential/业务输入隔离、可信参数编译、线性/模距离/阈值/evidence 流水线、显式 margin 和 public/protected capability 分级方向；A2 路线固定为先闭合二元 protected-model gate，再研究分级 capability。没有修改 A0 relation、代码或测试。
- **Deferred hypotheses:** Secret Trigger 术语、层深与能力映射、LWE/SIS 神经兼容性、2–3 层规模、少量 ReLU modulo、Floor 算子边界、Sigmoid 容错、MASK 对照路线和显式 `A*s`/常量折叠均保留意见，待 A1/A2 基础闭合后评估，不进入当前安全或论文主张。
- **Mandatory boundary:** 根据项目强制架构，verifier 仍只产生 evidence，协调器仍是唯一权限提交点；MASK/层内零化只能作为输出遮蔽对照，不能替代 protected-model 调用之前的硬门控或成为 OR 弱回退。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `.venv/bin/python -m pytest` -> 通过，80 tests。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
  - `test "$(rg --count '^\*\*唯一下一步：' PROJECT_WORKLOG.md)" -eq 1` -> 通过。
  - `rg -n '[[:blank:]]+$' PROJECT_WORKLOG.md SECURITY.md docs/RESEARCH_DESIGN.md` -> 无匹配，exit 1 表示尾随空白检查通过。
- **Test result:** 现有 80 项代码测试无回归；治理标题、唯一下一步和任务状态检查通过。此次只调整设计与路线，没有新增实现性质。
- **Incomplete work:** A1 数值规格、延期假设的文献/复杂度/实验评估、神经 verifier、二元协调器、public/protected capability 实验和 LeNet/MLP 均未实现。
- **Commit-ready candidate files:** `PROJECT_WORKLOG.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`。当前无 Git 仓库，故这是文件系统候选列表而非 `git status` 输出；外部 PPT 未复制到项目。
- **Next step:** 执行第 6 节定义的共同基础优先 A1 数值/算子规格，不安装 PyTorch 或实现模型。

## 2026-07-23 - A0 exact reference oracle checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `src/can/reference/a0.py` 并从包入口导出 A0-v1 固定常量、严格 23 字节 parser、不可变 slot/registry、`mod_q`/`center_q`、结构化 evidence 和精确 `verify_ref`；新增协议单元测试与防御性安全测试；同步 README、研究设计、安全状态、治理文件清单和工作日志。没有实现神经 verifier、LeNet/MLP 或访问协调器。
- **Security invariants:** raw credential 仅接受精确 `bytes`；registry 构造拒绝错误 shape/range、bool、全零行、重复/错误 entry；未知/禁用 profile/slot 无回退；reference evidence 不含 gate、decision、authorization 或 capability。A0 replay、adaptive chosen-`b`、输入替换和白盒风险仍未解决。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `.venv/bin/python -m pytest tests/unit/test_a0_reference.py tests/security/test_a0_reference_security.py` -> 通过，77 tests。
  - `.venv/bin/python -m pytest tests/unit` -> 通过，71 tests。
  - `.venv/bin/python -m pytest tests/integration` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/security` -> 通过，8 tests。
  - `.venv/bin/python -m pytest` -> 通过，80 tests。
  - `.venv/bin/ruff check .` -> 通过。
  - `.venv/bin/ruff format --check .` -> 通过，12 files already formatted。
  - `.venv/bin/mypy src tests` -> 通过，12 source files。
  - `.venv/bin/python -m pip check` -> 通过，No broken requirements found；pip cache 权限 warning 不影响结果。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
  - `rg -n '[[:blank:]]+$' AGENTS.md PROJECT_WORKLOG.md README.md SECURITY.md docs pyproject.toml requirements-dev.lock scripts src tests` -> 无匹配，exit 1 表示尾随空白检查通过。
- **Test result:** 精确 relation 的 core/guard/reject、bit-zero、wrap、八路 AND、全部距离 `0..128` 和八个位置的全部 `b_i=0..256` 均通过；畸形输入、类型混淆、client-supplied `A`、零矩阵、chosen `b=h` 和 profile 降级均 fail closed。有限测试不证明不可伪造性或神经 soundness。
- **Incomplete work:** A1 数值/算子规格、神经 verifier、误差证明、LeNet/MLP、协调器、PyTorch 设备配置和系统 related-work 检索尚未完成。
- **Commit-ready candidate files:** `README.md`、`PROJECT_WORKLOG.md`、`SECURITY.md`、`docs/RESEARCH_DESIGN.md`、`scripts/check_governance_docs.sh`、`src/can/reference/__init__.py`、`src/can/reference/a0.py`、`tests/unit/test_a0_reference.py`、`tests/security/test_a0_reference_security.py`。当前无 Git 仓库，故这是文件系统候选列表而非 `git status` 输出。
- **Next step:** 执行第 6 节定义的 A1 数值/算子规格，不安装 PyTorch 或实现模型。

## 2026-07-21 - Python technical bootstrap checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `README.md`、`.gitignore`、`pyproject.toml`、`requirements-dev.lock`、`src/can/` 五个计划包边界和单元/集成/安全 bootstrap 测试；扩展治理脚本并同步长期规则、研究设计、安全状态和工作日志。没有实现 LWE、神经 verifier、LeNet 或访问协调器。
- **Environment setup:** 当前解释器为 Python 3.11.9。`python3 -m venv .venv` 因系统缺少 `ensurepip` 失败；随后使用 `python3 -m pip --python .venv install 'pip==24.0'` 补足项目内 pip，并以 `.venv/bin/python -m pip install -e '.[dev]'` 成功安装锁定的轻量开发依赖。
- **Dependency resolution:** 包索引显示 PyTorch 2.13.0 与 torchvision 0.28.0；pip 元数据解析确认 torchvision 要求该 torch 版本。由于 dry-run 仍开始下载大型 CUDA wheel，命令被主动取消；`ml` 依赖没有安装或执行。
- **Exact checks run:**
  - `.venv/bin/python -m pytest tests/unit` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/integration` -> 通过，1 test。
  - `.venv/bin/python -m pytest tests/security` -> 通过，1 个敏感产物忽略配置测试。
  - `.venv/bin/python -m pytest` -> 通过，3 tests。
  - `.venv/bin/ruff check .` -> 通过。
  - `.venv/bin/ruff format --check .` -> 通过，9 files already formatted。
  - `.venv/bin/mypy src tests` -> 通过，9 source files。
  - `.venv/bin/python -m pip check` -> 通过，No broken requirements found。
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过。
  - `rg -n '[[:blank:]]+$' AGENTS.md PROJECT_WORKLOG.md README.md SECURITY.md docs pyproject.toml requirements-dev.lock scripts src tests` -> 无匹配，exit 1 表示尾随空白检查通过。
- **Test result:** 技术 bootstrap 的包、配置和质量门槛可执行且通过；这些测试不覆盖任何密码、模型或访问控制行为。
- **Incomplete work:** A0 精确 oracle、神经 verifier、LeNet/MLP、误差证明、PyTorch 设备配置和系统 related-work 检索尚未完成。
- **Next step:** 执行第 6 节定义的 A0-v1 精确 oracle，不实现神经 verifier 或业务模型。

## 2026-07-21 - A0 protocol specification checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增 `docs/A0_PROTOCOL_SPEC.md`；固定 A0-v1 toy profile、23 字节 wire encoding、可信 slot registry、精确 oracle、未来神经误差契约和测试向量族；同步研究、安全、长期约束、检查脚本和工作日志。
- **Exact checks run:**
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过，检查五份受管文档、唯一下一步和状态枚举。
  - `test $((1 + 2 + 4 + 8 * 2)) -eq 23` -> 通过，wire credential 为 23 字节。
  - `awk 'BEGIN { p=(25/257)^8; if (p >= 1e-8) exit 1 }'` -> 通过，指标约为 `8.017797488e-09`。
  - `test $((4 + 4)) -le 8` 和 `test $((8 + 4)) -le 12` -> 通过，completeness/soundness 常量关系成立。
- **Test result:** A0 文档结构、wire 长度和已声明的算术关系通过；没有业务代码或业务测试。
- **Incomplete work:** A0 oracle、神经 verifier、LeNet、误差证明、技术配置和系统 related-work 检索均未实现。
- **Next step:** 执行第 6 节定义的 Python/PyTorch 技术 bootstrap，不实现业务逻辑。

## 2026-07-21 - Research and security baseline checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 根据项目负责人提供的研究方案和可行性评审，新增 `docs/RESEARCH_DESIGN.md` 与 `SECURITY.md`；更新 `AGENTS.md`、本工作日志和治理文档检查范围。
- **Exact checks run:**
  - `bash -n scripts/check_governance_docs.sh` -> 通过。
  - `./scripts/check_governance_docs.sh` -> 通过，检查四份受管文档、唯一下一步和状态枚举。
  - `rg -n '[[:blank:]]+$' AGENTS.md PROJECT_WORKLOG.md SECURITY.md docs/RESEARCH_DESIGN.md scripts/check_governance_docs.sh` -> 无匹配，尾随空白检查通过。
- **Test result:** 检查脚本语法和治理/研究/安全文档结构通过；没有业务代码或业务测试。
- **Incomplete work:** A0 精确协议、系统 related-work 检索、Git/项目初始化、实现、证明和实验均未开始。
- **Next step:** 创建并评审第 6 节定义的 `docs/A0_PROTOCOL_SPEC.md`。

## 2026-07-21 - Reference paper import checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 创建 `paper/`，从 `/mnt/e/CAN/paper` 原样复制两份 PDF，并更新本工作日志。
- **Exact checks run:**
  - `cmp -s '/mnt/e/CAN/paper/How to Securely Implement Cryptography in Deep Neural Networks.pdf' 'paper/How to Securely Implement Cryptography in Deep Neural Networks.pdf'` -> 通过。
  - `cmp -s '/mnt/e/CAN/paper/Planting Undetectable Backdoors in Machine Learning Models.pdf' 'paper/Planting Undetectable Backdoors in Machine Learning Models.pdf'` -> 通过。
  - `sha256sum '/mnt/e/CAN/paper/How to Securely Implement Cryptography in Deep Neural Networks.pdf' 'paper/How to Securely Implement Cryptography in Deep Neural Networks.pdf'` -> 源/目标均为 `505b197590038e18d23685bacc3e945ec79fff8e653fe9d34ef8c4b1f15706ca`。
  - `sha256sum '/mnt/e/CAN/paper/Planting Undetectable Backdoors in Machine Learning Models.pdf' 'paper/Planting Undetectable Backdoors in Machine Learning Models.pdf'` -> 源/目标均为 `6716671e0a4af370b9d5ad34ed8a5f0302b4f314af8a601dfc772dd49bb2052b`。
  - `file paper/*.pdf` -> 两个文件均识别为 PDF 1.5。
  - `./scripts/check_governance_docs.sh` -> 通过。
- **Test result:** 两份目标 PDF 与源文件逐字节一致；治理文档检查通过。
- **Incomplete work:** 论文版权/再分发许可和项目基线仍待确认；PDF 不得自动提交或推送。
- **Next step:** 由项目负责人确认并记录第 6 节所列项目基线。

## 2026-07-21 - Initial governance checkpoint

- **Branch/worktree:** `/home/kali/CAN`；不是 Git 仓库，无分支或 worktree 身份。
- **Full commit SHA:** 不适用；没有有效 `HEAD`。
- **Changes:** 新增长期约束 `AGENTS.md`、动态事实源 `PROJECT_WORKLOG.md` 和治理文档检查脚本 `scripts/check_governance_docs.sh`；因安全相关性无法由现有内容确认，未创建 `SECURITY.md`。
- **Exact checks run:**
  - `git status --short --branch` -> 失败（exit 128）：不是 Git 仓库。
  - `git branch --show-current` -> 失败（exit 128）：不是 Git 仓库。
  - `git worktree list --porcelain` -> 失败（exit 128）：不是 Git 仓库。
  - `git rev-parse HEAD` -> 失败（exit 128）：不是 Git 仓库。
  - `rg --files -g '!**/.git/**'` -> 创建文档前无项目文件。
  - `find . -maxdepth 3 -mindepth 1 -print` -> 创建文档前只发现空 `.git`、`.agents`、`.codex` 目录。
- **Test result:** 未运行单元、集成、安全、lint 或类型检查；仓库没有代码和对应配置。
- **Script syntax check:** `bash -n scripts/check_governance_docs.sh` -> 通过。
- **Documentation check:** `./scripts/check_governance_docs.sh` -> 通过。
- **Incomplete work:** 项目基线、Git 初始化、技术栈、架构、安全相关性和质量命令均待确认。
- **Next step:** 由项目负责人确认并记录第 6 节所列项目基线。

# 9. Decisions

## D-001 - Use an evidence-only initial baseline

- **Date:** 2026-07-21
- **Decision:** 对无法从代码、配置或 Git 验证的信息统一标记为“待确认”，不选择或虚构技术栈与项目能力。
- **Reason:** 当前目录没有项目内容，任何具体业务或工程判断都缺乏证据。
- **Alternatives:** 根据目录名推测项目用途；套用某种默认语言模板。
- **Consequences:** 文档可以安全作为后续事实源，但项目实现必须等待基线确认。

## D-002 - Do not create SECURITY.md yet

- **Date:** 2026-07-21
- **Decision:** 本 checkpoint 不创建 `SECURITY.md`；当需求或实现首次出现认证、密码学、权限、敏感数据或网络安全边界时再创建。
- **Reason:** 当前没有可供描述的信任模型、资产、攻击者能力、输入、认证流程或密钥生命周期；现在创建会把模板误当成已实现安全设计。
- **Alternatives:** 创建全为“待确认”的安全模板。
- **Consequences:** `AGENTS.md` 中的条件性安全规则立即有效；安全相关能力进入范围时，创建 `SECURITY.md` 是同一 checkpoint 的强制工作。

## D-003 - Treat the directory as non-Git until Git proves otherwise

- **Date:** 2026-07-21
- **Decision:** 不把环境提供的空 `.git` 目录视为有效仓库，不记录虚构分支、HEAD 或 worktree。
- **Reason:** 四个 Git 只读命令均返回“not a git repository”。
- **Alternatives:** 自动执行 `git init`。
- **Consequences:** 当前 checkpoint 无 commit SHA，且只能列出文件系统层面的待提交候选；是否初始化版本库留待项目负责人或后续明确任务决定。

## D-004 - Confirm the research and security scope

- **Date:** 2026-07-21
- **Decision:** 项目正式定位为格密码神经验证网络的模型访问控制研究，先完成阶段 A，再扩展阶段 B；该决定使 D-002 的“安全相关性待确认”失效并触发创建 `SECURITY.md`。
- **Reason:** 项目负责人明确给出了 LWE 验证、模型门控、认证授权、capability 和工具访问控制目标。
- **Alternatives:** 继续保留通用空仓库基线；同时实现 A/B 全部模块。
- **Consequences:** 后续工作受安全架构和密码规则约束；首篇论文范围收敛到阶段 A。

## D-005 - Separate numeric unlock, authentication and signature claims

- **Date:** 2026-07-21
- **Decision:** A0 称为 toy LWE 数值解锁；A3 只有在协议实现后才能声称请求绑定与 freshness，仍不能
  单独称为认证；只有 A4 公钥签名关系实现、组合进 A3 并满足相应安全定义后才能称为公钥认证或验签。
  D-025 对原始阶段术语作出这一收紧澄清。
- **Reason:** LWE 解密不自动提供不可伪造性；公开加密可能允许任何人生成“授权”密文，任意 `(A,b)` 还可能直接解锁或形成判决 oracle。
- **Alternatives:** 把所有阶段统一称为 LWE 验签层。
- **Consequences:** API、文档、实验和论文必须使用与已证明性质相符的名称；A0 不承担安全认证结论。

## D-006 - Use one-sided soundness preservation as the minimum neural security goal

- **Date:** 2026-07-21
- **Decision:** 神经验证器的最低安全目标是 `V_nn(a) = 1 -> V_ref(canonical(a)) = 1`，并对边界模糊区 fail closed。
- **Reason:** 仅证明合法样本在 `epsilon < margin` 时不翻转，不能排除攻击者构造靠近边界的非法输入造成假接受。
- **Alternatives:** 只报告随机差分准确率；使用对称容差扩大接受集合。
- **Consequences:** A0/A1 规格必须覆盖全部可表示输入或明确缩小主张，有限实验不能替代证明。

## D-007 - Keep private keys outside security-bearing verifiers

- **Date:** 2026-07-21
- **Decision:** toy 数值实验只使用临时非生产测试参数；安全承载 verifier 只嵌入公开验证信息，真实签名私钥留在模型之外。
- **Reason:** 模型权重中的 secret 在白盒场景可直接提取，也会把解密 oracle 风险带入访问控制接口。
- **Alternatives:** 把长期 LWE secret 固定编码在可分发模型权重中。
- **Consequences:** 白盒秘密保护不作为 A0 结论；后续真正认证路线转向挑战响应或公钥格签名验证。

## D-008 - Make authorization and tool enforcement mandatory, not routed

- **Date:** 2026-07-21
- **Decision:** 阶段 B 的验证器只产生证据，唯一协调器提交授权，工具网关按实际调用参数强制验证 capability；Router 和专家不能绕过这条链。
- **Reason:** 普通 MoE Router 和自然语言输出不适合作为安全决策或最终强制点。
- **Alternatives:** 把认证专家作为 Router 可选专家；信任业务专家输出的 `allow`。
- **Consequences:** 阶段 B 的系统测试必须覆盖 Router 绕过、提示注入、直接工具调用和 capability 参数错配。

## D-009 - Position prior work as a novelty constraint

- **Date:** 2026-07-21
- **Decision:** 不把通用密码 DNN 编译、签名控制分类器或格签名神经网络本身列为创新；主张聚焦 LWE 专用量化 soundness、误差证明和防御性访问控制组合。
- **Reason:** 两篇本地论文已直接覆盖上述通用能力和格签名神经网络先例。
- **Alternatives:** 继续把“验签神经元”本身作为主要创新。
- **Consequences:** 在完成系统文献检索前，论文新颖性仍是残余风险；实现应优先产生可证明的差异化结果。

## D-010 - Fix the A0-v1 toy relation and remove matrices from the wire format

- **Date:** 2026-07-21
- **Decision:** A0-v1 使用 `n=32`、`m=8`、`q=257`、center 128、bounded centered-binomial noise 4、reference radius 12、neural threshold 8 和 error target 4；credential 固定为 23 字节，`A_slot` 只由本地 registry 根据 slot 解析。
- **Reason:** 单方程 `A=0,b=h` 可被请求方直接解锁；八分量 AND 降低随机落入接受包络的概率，并提供组件级误差分析，而本地 registry 从结构上阻止客户端替换 `A` 或参数。
- **Alternatives:** 允许请求直接携带 `(A,b)`；只使用一个 LWE 方程；从请求中的 profile 接受不同阈值。
- **Consequences:** client-chosen `A` 和参数降级不再是 wire-level 输入，但 adaptive chosen-`b`、replay、输入替换和白盒风险仍不受 A0 保护；这些限制不能用约 (8.02e-9) 的随机命中率掩盖。

## D-011 - Pin the Python bootstrap and keep ML dependencies optional

- **Date:** 2026-07-21
- **Decision:** 项目固定 Python `==3.11.*`，当前开发工具固定为 pytest 9.1.1、Ruff 0.15.22 和 mypy 2.3.0，并记录解析后的开发依赖锁；PyTorch 2.13.0 与 torchvision 0.28.0 放入可选 `ml` 依赖组，在目标 CPU/CUDA 环境确定前不安装。
- **Reason:** 精确 oracle 不依赖 PyTorch；把大型、设备相关依赖与基础质量工具隔离，可以先建立可复现的整数语义和测试，同时避免无意下载 CUDA artifacts。
- **Alternatives:** 启动时安装未锁定的最新包；立刻安装默认 CUDA PyTorch；把所有依赖混入基础环境。
- **Consequences:** 当前 `.venv` 可运行 bootstrap 和后续纯 Python oracle 测试，但不能训练或执行 LeNet；进入神经 verifier/业务模型阶段前必须选择安装渠道、目标设备并验证 PyTorch 运行时。

## D-012 - Make the A0 reference boundary immutable and evidence-only

- **Date:** 2026-07-23
- **Decision:** A0 parser 只接受唯一 23 字节 `bytes` 编码；slot/registry 在构造期完成 shape、范围、全零行、重复项和精确类型校验后冻结；`verify_ref` 只返回由证据码与内部距离组成的不可变 `ReferenceEvidence`，不返回或创建授权原语。
- **Reason:** reference oracle 必须成为 A1 差分测试的确定性事实源，同时不能让请求方替换矩阵/profile、利用 bool/int 混淆或把 oracle 结果误当成已提交权限。A0 中间量由输入边界限制在 `int64` 范围内，Python 精确整数可直接实现无溢出的数学语义。
- **Alternatives:** 接受 JSON/tensor 并隐式转换；在请求中携带矩阵或阈值；让 oracle 直接返回 gate；依赖固定宽度整数静默 wraparound。
- **Consequences:** A0 的结构与算术边界可被单元/安全测试直接审阅；协调器和业务零副作用仍未实现，adaptive chosen-`b`、replay、输入替换、白盒暴露和任何密码不可伪造性质仍明确不受保证。

## D-013 - Adopt the common authentication-neuron foundation and defer construction claims

- **Date:** 2026-07-23
- **Decision:** 阶段 A 采纳独立 credential 验证模块、可信参数本地编译、线性残差/规范模距离/阈值/证据流水线、显式数值 margin 和 capability 分级方向；先完成共同 A1 语义与二元 A2 protected-model gate，再评估 public/protected 分级及具体网络构造。
- **Reason:** 初步方案提供了有价值的能力门控与神经模块直觉，但部分术语、激活函数、层数、层级映射和 MASK 路线缺少当前定义、证明或实验。先闭合共同基础可以保留研究空间，同时避免候选假设提前改变接受集合或授权边界。
- **Alternatives:** 立即选定 Secret Trigger、浅层/深层映射、2–3 层 ReLU/Sigmoid 或 MASK 作为主路线；完全忽略初步方案中的能力分级构想。
- **Consequences:** Secret Trigger、层级映射、LWE/SIS 兼容性、层数、ReLU/Floor/Sigmoid/MASK 和显式 `A*s`/常量折叠进入延期假设清单；其中任何候选若后续实现，都必须通过现有 parser、evidence-only verifier、唯一协调器和零受保护副作用约束，不能形成弱回退。

## D-014 - Fix A1 integer semantics before selecting a neural construction

- **Date:** 2026-07-23
- **Decision:** A1-v1 先固定 `A1-INT32-S1` 规范输入、由本地 `int64` 编译产生的相位锚点、`b-t`/欧几里得模距离/阈值 8/八路 AND 语义、逐分量误差 `<=4` 和全输入单向 soundness；具体激活函数、神经算子边界、层数、显式矩阵计算或常量折叠留给下一构造决策。
- **Reason:** A0 常量可把共同运行时残差紧化到 `[-256,256]`，足以定义精确有限语义；但 modulo 在分支边界不连续，若在选择构造前混用普通实数误差传播，会把有限随机一致性误写成全域 soundness。
- **Alternatives:** 直接实现框架 `%`/比较并称为神经网络；立即选择 ReLU、Sigmoid 或 MASK；在没有逐算子范围时只设置总体容差。
- **Consequences:** 所有候选共享同一输入、距离、阈值、evidence 和证明目标，并必须逐分支、穷尽有限残差域或用形式方法处理 modulo；完成本规格不代表 verifier 已实现，下一 checkpoint 必须先选择并论证主构造。

## D-015 - Select an exact bounded ReLU verifier and isolate ordinary operators

- **Date:** 2026-07-23
- **Decision:** A1-C1 选择 `CAN-RELU-EXACT-v1` 作为首个主构造：把本地锚点 `t` 折叠进 bias，以 `8->40->16->1` 的三个整数 affine+ReLU blocks 精确实现 modular distance、阈值和八路 AND；普通 Floor/modulo/abs/compare 实现只作为互斥 exact-ops 测试基线。
- **Reason:** A1 紧残差域只包含 `[-256,256]`，五个 ReLU/分量的短 CPWL 恒等式即可覆盖完整 modular distance；整数阈值和 AND 也有精确 ReLU 恒等式，因此无需浮点容差、Sigmoid、动态分支或通用深 sawtooth，逐分量误差可固定为 `0`。
- **Alternatives:** 在主 core 直接调用 `%`/Floor/compare；使用 Sigmoid 软门；用 MASK 遮蔽业务输出；显式保留固定 `A*s` runtime 子图；实现通用多周期 sawtooth。
- **Consequences:** 主神经声明严格限于固定 integer affine/ReLU graph，exact-ops 不得 fallback；常量折叠收缩为固定 toy relation 主张，不能外推到通用 LWE 或签名验证；下一 checkpoint 可在不安装 PyTorch 的情况下先实现 exact-integer conformance backend 和全域代码测试。

## D-016 - Implement A1 as an immutable evidence-only conformance backend

- **Date:** 2026-07-23
- **Decision:** dependency-free A1 实现使用不可变 compiled profile/registry 和 Python exact integers 承载已证明范围内的 `int32`/scale `1` 三层 affine/ReLU graph；公共 adapter 复用 A0 raw parser，只返回单字段 evidence，内部 trace 仅供测试，主模块不定义 exact-ops 或导入 `verify_ref`。
- **Reason:** 该边界可在不引入设备相关依赖前把构造公式连接到可执行代码和完整 toy 域测试，同时保持 verifier/evidence/authorization 分离，并让后续 PyTorch backend 具有明确差分事实源。
- **Alternatives:** 直接安装 PyTorch 后同时实现模型；把 trace 或距离暴露在 evidence；允许异常时调用 `V_ref`；使用全局 compiled profile 或 secret；在主 core 调用普通 modulo/compare。
- **Consequences:** 当前 134 项测试支持 dependency-free graph 的数值等价和 no-fallback 边界，但不支持 PyTorch/量化、业务零调用或认证主张；M4 保持进行中，下一 checkpoint 必须先固定 deployment backend。

## D-017 - Select a CPU exact-integer PyTorch backend before qint8

- **Date:** 2026-07-23
- **Decision:** A1-B1 选择 `CAN-TORCH-CPU-EXACT-v1`：Linux x86_64、CPython 3.11、官方
  PyTorch `2.13.0+cpu` wheel、CPU eager mode；weight/bias/activation 为 `int32`，product 为
  已证明安全的 `int32`，reduction/pre-activation 为 `int64`，ReLU 后精确转回 `int32`。首版
  affine 只用固定 `mul/sum/add`，ReLU 用 `clamp(min=0)`；qint8、CUDA、export 和 runtime
  fallback 均不进入该 backend。
- **Reason:** 当前环境没有可验证 GPU 工具链，A1-C1 又具有 error `0` 的精确整数证明。官方基础
  tensor API 明确支持整数 mul/add 和显式 sum dtype，而 quantized Linear 引入 quantized
  input/output、scale 和 zero-point；先建立 CPU exact baseline 能保持 accumulator 与转换语义
  可审计，不把未证明的重标定当作零误差。
- **Alternatives:** 直接采用 qint8/FBGEMM；安装 CUDA wheel；用 float `nn.Linear`；依赖
  `torch.mv`/fused kernel 的未固定整数 accumulator；不支持时切回 Python/reference 路线。
- **Consequences:** 下一 checkpoint 必须核验指定 wheel/hash、实现显式 optional torch 模块并
  重跑全部有限域和安全矩阵。任何环境或性质 gate 失败都禁用 backend 并返回 `CONFIG_REJECT`，
  不切换到较弱路线；A2 模型、qint8 和 accelerator 继续延期。

## D-018 - Select Fashion-MNIST and a deterministic MLP before gating

- **Date:** 2026-07-23
- **Decision:** A2-E1 选择 `CAN-A2-FMNIST-MLP-v1`：官方 CPU package tuple 上的
  Fashion-MNIST 与 `784->256->128->10` float32 MLP，先完成确定性无门控 baseline，再由唯一
  协调器实现 `DENY`/protected-model 二元前置硬门控。请求方不能提交 evidence 或 decision，
  所有拒绝要求零 protected-model 调用。
- **Reason:** Fashion-MNIST 保持小型固定输入和 split，同时比 MNIST 更不饱和；MLP 比 LeNet
  引入更少算子与性能变量，适合作为验证门控语义的首个业务网络。把 baseline 与协调器拆成相邻
  checkpoint，可先闭合训练/数据复现，再把零调用性质归因于门控实现。
- **Alternatives:** MNIST + MLP；Fashion-MNIST + LeNet；同时实现两个数据集/模型；先计算 logits
  后 MASK；在同一 checkpoint 加 public capability 或 Stage B token。
- **Consequences:** `pyproject.toml` 的 ML direct pins 收紧为 `torch==2.13.0+cpu` 和
  `torchvision==0.28.0+cpu`，只允许官方 CPU index。下一 checkpoint 安装/锁定 transitive ML
  环境、获取并核验数据、实现和运行无门控 baseline；协调器、LeNet/MNIST、MASK、量化/导出、
  capability 和安全承载密码继续延期。

## D-019 - Accept the deterministic A2 baseline before adding one hard gate

- **Date:** 2026-07-23
- **Decision:** 接受指定 CPU tuple 上两次完全一致、test accuracy `88.08%` 的 Fashion-MNIST MLP
  作为 A2-E1 唯一无门控 baseline；下一 checkpoint 只增加一个固定 A1-B1 verifier、一个协调器和
  model-before-call 二元硬门控，不改变模型、数据、训练结果或进入 capability 分级。
- **Reason:** 固定数据、输入契约和模型结果已经独立复现，使下一步可以把 label 等价、拒绝零调用和
  latency 差异明确归因于访问控制路径，而不是训练或模型变化。
- **Alternatives:** 继续调参提高准确率；先换 CNN/LeNet；同时加入 MASK、public capability 或 Stage B；
  在 logits 计算后再遮蔽输出。
- **Consequences:** `88.08%` 只是单机分类 baseline，不是安全阈值。硬门控必须在调用前提交决定，所有
  reject 保持零 protected-model calls，且 accepted 10,000 个 test labels 必须逐项等于当前 baseline；
  未满足这些条件前 M5 保持 pending。

## D-020 - Close A2-E1 with one pre-call coordinator gate

- **Date:** 2026-07-29
- **Decision:** A2-E1 只使用一个本地 `A2AccessCoordinator` 作为权限提交点。公共入口只接收单张
  规范业务图像和原始 23-byte credential；固定 A1-B1 verifier 仍只产生 evidence，只有 exact
  `A1EvidenceCode.NUMERIC_ACCEPT` 可提交 allow，提交后才执行一次 protected MLP。所有其他情况
  返回固定 deny，不能由请求方注入 evidence、decision、backend、model 或 policy。
- **Reason:** 该结构让业务标签等价、协调器顺序和拒绝零调用都能用调用计数与完整测试集直接核验，
  同时保持 verifier/evidence/authorization 三个边界分离。
- **Alternatives:** 在 logits 计算后乘 gate 或 MASK；允许请求方传 evidence；异常时回退到
  dependency-free/reference verifier；同时加入 public capability、LeNet 或 Stage B token。
- **Consequences:** A2-E1 在 toy、单机、黑盒范围内闭合，M5 最小 MLP gate 完成；该结果不增加
  A0 的认证、不可伪造、replay 或输入绑定性质。下一 checkpoint 只能先固定 A2-E2 capability 分级
  实验规格，public path 不能成为弱 verifier fallback 或调用/泄露 protected path。

## D-021 - Select an independent public model for A2-E2 capability tiers

- **Date:** 2026-07-29
- **Decision:** A2-E2 选择独立 `CAN-A2-FMNIST-PUBLIC-MLP-v1`（`784->64->2`）作为 public
  主路线，只输出固定 footwear/non-footwear coarse class。public entry 默认关闭，由本地可信部署
  配置绑定；同一协调器互斥提交 `DENY`/`PUBLIC`/`PROTECTED`。protected 验证失败保持 A2-E1
  deny，不能 fallback 到 public。
- **Reason:** 独立模型让 public path 的零 protected-model/head calls、零 protected features 和独立
  artifact 能通过直接计数/身份检查审计，同时避免共享 trunk/head 把“未释放输出”误写为“未使用
  protected 能力”。
- **Alternatives:** 独立 public head、共享 trunk 的浅层/深层 exit、独立非模型服务、MASK 或验证
  失败后降级到 public。前三者保留为后续对照；MASK/共享计算不能替代前置边界；失败后降级因
  扩大弱路径而被禁止。
- **Consequences:** 先单独实现并复现 public 无门控 baseline，再扩展协调器；public 响应不具有
  bearer authority，不能升级/重标记/复用为 protected。A0 replay、业务输入未绑定、黑盒和非生产
  限制不变。

## D-022 - Accept the deterministic A2-E2 public baseline before three-state integration

- **Date:** 2026-07-29
- **Decision:** 接受两次独立进程完全一致、coarse test accuracy `99.85%` 的
  `CAN-A2-FMNIST-PUBLIC-MLP-v1` 作为 A2-E2 唯一 public baseline；下一 checkpoint 只集成本地绑定
  public/protected entries、一个三态协调器和 version-2 envelopes，不重训或改变两个模型。
- **Reason:** 固定 public 功能、输入/label contract、预测/state digest 与独立 module/storage 边界后，
  下一步可把 public/protected/deny 调用矩阵和无 fallback 结果归因于协调器，而不是模型变化。
- **Alternatives:** 继续调参；更换 coarse mapping；复用 protected trunk/head；在同一 checkpoint
  重训模型并集成 gate；把 protected 验证失败降级为 public。它们会破坏固定对照、共享 protected
  computation 或扩大弱路径，均不采用。
- **Consequences:** `99.85%` 只说明固定二分类任务在当前单机上的 baseline 性能，不是访问控制或
  密码安全指标。三态集成必须保持 public 零 verifier/protected calls、protected reject 零 public/
  protected calls、互斥提交和不可升级/复用边界。

## D-023 - Require accepted model states for the A2-E2 empirical report

- **Date:** 2026-07-29
- **Decision:** 三态协调器与报告 runner 可以在不重训的 checkpoint 中实现和测试，但真实
  `capability.json` 只能由 state SHA 精确等于 D-019/D-022 已验收摘要的两个本地内存模型生成。
  随机初始化模型、仅引用旧 JSON 指标或放宽 digest 检查不能替代 accepted-state 评估。
- **Reason:** earlier baseline checkpoints 按 artifact policy 删除了临时序列化 state，只留下指标与
  摘要；摘要不能恢复权重。若用随机模型测调用矩阵却报告 accepted baseline 标签/latency，会把结构
  测试误写成经验复现；若在本 checkpoint 自动重训，则直接违反 D-022 的固定对照约束。
- **Alternatives:** 自动重训两次 baseline 并恢复完全相同摘要；保存或提交 pickle/checkpoint；用随机
  模型生成“contract probe”并把旧指标拼入报告；跳过 state digest。除非项目负责人明确修改决定，
  这些路线均不采用。
- **Consequences:** runner 强制校验四份 baseline JSON 和两个 model-state digests，且源码/测试证明
  不调用训练 helper；完整 empirical report 当前被 accepted-state materialization 阻塞。项目负责人
  可提供可信本地 states，或另行明确允许确定性重新物化且重新记录 artifact 生命周期。

## D-024 - Permit deterministic local state rematerialization

- **Date:** 2026-08-08
- **Decision:** 项目负责人明确允许在固定 Linux x86_64/CPython 3.11/CPU wheel、数据摘要、种子、
  split、优化器和十 epoch 协议下重新物化 A2-E1 protected 与 A2-E2 public baseline。materializer
  可以训练并保存仅含 `state_dict` 的本地 ignored checkpoint；随后必须重新计算 canonical state digest
  和文件 SHA-256，只有两者与已验收摘要匹配才可交给不训练的三态 evaluator。
- **Reason:** 原有临时 state 已按 artifact policy 删除，项目负责人选择可审计的确定性重建路线以完成
  已阻塞的经验报告，同时避免把随机模型或旧 JSON 指标拼接成 accepted-state 结果。
- **Artifact lifecycle:** state 文件只位于 `artifacts/a2/local-states/`，manifest 记录实验 ID、拓扑、
  runtime/data/seed 摘要、canonical state digest 和文件 digest；不保存 optimizer、credential、图像、
  logits 或 feature，不纳入提交、发布或外部响应。加载时严格校验 manifest、`state_dict`、CPU/float32
  contract、拓扑和 canonical digest，任何漂移 fail closed。
- **Consequences:** D-023 的 no-training evaluator 边界保持不变；完成 materialization 后必须运行真实
  三态报告、完整质量检查并在下一 checkpoint 记录 state 生命周期和残余风险。

## D-025 - Freeze A3 as a stateful binding protocol before A4

- **Date:** 2026-08-08
- **Decision:** A3-v1 固定 133 字节 `CAN-A3-MSG-v1` proof message、规范 float32 image SHA-256、
  本地 `model_id=1`/`scope_id=1`、32 字节 identity/nonce、服务端时间、60 秒 TTL、单进程 trusted
  nonce store 和 exact accept 后的原子 `PENDING -> CONSUMED`。A3 verifier 只返回绑定 exact message
  digest/identity 的 evidence；A0/A1 numeric evidence 不得作为认证路线，A4 未激活前入口默认关闭。
- **Reason:** 未来公钥格签名必须验证唯一且与真实模型请求一致的消息；单纯正确验签不能阻止跨输入/
  模型/scope 替换或 replay。把 freshness state 与 verifier 分离可保持神经验证研究边界，同时让并发
  at-most-once 模型调用成为可测试协议性质。
- **Alternatives:** 给 A0 23-byte credential 外包一层 nonce；让客户端提交算法/key/profile；验证前
  consume nonce；验证成功后非原子地标记已用；立即进入 A4 并临时使用未固定字符串消息。前两者会
  产生错误认证/降级语义，后两者不能支持并发 at-most-once 或稳定的 request-binding 主张。
- **Consequences:** 下一 checkpoint 只实现 canonical codec/hash、in-memory store、固定 envelopes 和
  默认关闭 coordinator shell，以 deterministic proof stub 完成协议测试但不声称认证。A4 随后选择
  reviewed public-key lattice relation 并实例化 exact proof verifier；分布式状态、durable consume、
  TLS/channel binding、DoS、Stage B 和生产保证继续延期。

## D-026 - Keep A3 runtime proof-stub-only and return to A4 relation selection

- **Date:** 2026-08-08
- **Decision:** A3 runtime 只实现冻结规格要求的 canonical codec/parser、trusted in-memory nonce
  lifecycle、request/input binding、freshness、evidence-only boundary 和唯一协调器提交顺序。测试可
  使用 deterministic proof stub，但运行时没有 A4 profile 时固定配置拒绝，且不把 stub、A0/A1
  evidence 或 A2 public output 解释为认证。
- **Reason:** 这些状态与输入边界是 A4 验签神经网络可安全组合的必要协议前提；先冻结并测试它们能让
  后续公钥关系的接受集合、消息摘要和一次性模型调用具有可审计的组合契约。
- **Consequences:** M6 已闭合；唯一下一步回到 A4 的 reviewed public-key lattice relation、exact
  reference verifier 和 neural soundness/completeness 规格。分布式状态、durable consume、TLS/channel
  binding、DoS、完整 ML-DSA、Stage B、qint8/CUDA/export 和白盒保证继续延期。

## D-027 - Select GPV PFDH as the first exact A4 public relation

- **Date:** 2026-08-11
- **Decision:** A4 首个关系选择 GPV STOC 2008 probabilistic full-domain-hash 的短原像公开验证谓词。
  非生产 `A4-GPV-PFDH-TOY-v1` 固定 `q=257,n=8,m=72,beta_inf=1`、32 字节 salt、105 字节 proof、
  SHAKE256 hash-to-syndrome 和 `A*z mod q = y`；本 checkpoint 只实现无私钥 exact reference 与 A3
  evidence adapter。
- **Reason:** 该 reviewed relation 的公开验证核心可分解为固定公开 affine、规范模等式、短向量检查
  和逻辑 AND，适合下一步研究完整有限输入域的 neural soundness。相比首版直接实现 ML-DSA，它不
  要求在同一 checkpoint 同时闭合 NTT、hint、多项式编码和完整标准 hash 流水线。
- **Alternatives:** 首版直接实现 ML-DSA；选择 Falcon/FN-DSA 完整标准 relation；继续扩展 A0 自定义
  LWE margin；设计新签名方案。前两者超出首个关系 checkpoint，后两者不能提供已有审查关系边界。
- **Consequences:** profile、proof、hash 和 exact relation 已冻结并可与 A3 组合，但 toy 参数、具体
  SHAKE256 映射和公开 gadget fixture 不继承 GPV 不可伪造性结论。M7 进入 `in_progress`；下一步必须
  先选择固定神经构造和全域证明方法，生产参数、signer/keygen、完整 ML-DSA 和 Stage B 继续延期。

## D-028 - Adopt the V0/V1-prep/V1/V2 research route

- **Date:** 2026-08-11
- **Decision:** 长期研究路线固定为 `V0 -> V1-prep -> V1 -> V2`，同时保留 A0--A4 作为实现与
  文档编号。V0 是已闭合的 toy LWE 数值解锁/神经等价/模型门控；V1-prep 是 A3 请求绑定与 A4
  canonical `(y,z)` 神经代数内核；V1 在其后选择已有安全分析支持的格 challenge-response/身份
  协议；V2 最后研究标准 ML-DSA reference 和逐模块神经等价。
- **Reason:** canonical `(y,z)` 内核可以把固定整数矩阵、模等式、范数、累加器和全输入
  `V_nn=1 -> V_ref=1` 证明从具体认证协议中隔离出来。后续精确 SIS/Schnorr-like V1 可把
  transcript 规范化为目标 `y=t+c*u mod q` 后复用该内核；若所选协议包含 LWE noise、rounding、
  decomposition 或 hint，则必须显式扩展 relation，不能假定无条件复用。
- **Alternatives:** 把当前公开可构造的 GPV gadget 直接称为 V1 认证；跳过 V1 直接实现完整 ML-DSA；
  把 A0 LWE 解密成功解释为公开验签；立即重命名现有 A0--A4 模块。前三者混淆数值、协议和标准
  安全主张，最后一项制造无研究收益的代码/文档迁移，均不采用。
- **Consequences:** 当前唯一下一步不变，仍先闭合 A4 canonical `(y,z)` 固定神经构造与全域证明；
  该成果只作为 V1 可复用编译内核，不提供身份认证。V1 必须另行冻结具体协议、key/transcript/
  challenge/rejection 语义和安全游戏；V2 不属于首篇论文 MVP，也不要求任意剪枝、微调或未验证
  export 后保持等价。Stage B 依赖 V1 安全承载认证闭合，不依赖 V2 完成。

## D-029 - Select the A4-C1 exact point-pulse ReLU graph

- **Date:** 2026-08-11
- **Decision:** A4 canonical `(y,z)` compiler 固定为 `CAN-RELU-A4-PFDH-TOY-v1`。该 graph 使用
  scale-1 整数、`int64` reduction、`int32` activation storage、norm violation ReLUs，以及覆盖
  `K=-72..71` 的 residual point pulses，形成 `80->3600->1153->1` 三层 affine/ReLU topology；
  dependency-free sparse evaluator 是当前唯一获准 backend。
- **Reason:** 在 `||z||_inf<=1` 时每个 residual 的有限范围只包含 144 个模 257 倍数，整数三点
  second-difference ReLU pulse 可精确判定每个倍数，八个 residual pulse 与 norm accumulator 的最终
  ReLU conjunction 因而对全部 canonical 输入实现 `V_nn==V_ref`，无需模运算、比较或 reference
  fallback。
- **Alternatives:** 把 `%`/Floor/比较包装成“神经层”；只做随机 differential；使用近似 sawtooth、
  sigmoid 或 margin；立即实现 PyTorch/qint8/CUDA/export。前三者不能支持当前全输入 exact claim，
  最后一项需要独立 backend 范围证明和复测，均不进入本 checkpoint。
- **Consequences:** M7/V1-prep 闭合，但只得到 toy public relation 的神经编译结论，不得到私钥持有、
  知识可靠性、不可伪造性或身份认证。唯一下一步转为冻结一个 reviewed V1 格身份协议及其 transcript、
  Fiat-Shamir/challenge/rejection 语义、安全游戏和 A3 binding；若其 verifier 不能规范化为 A4-C1
  `(y,z)`，必须定义并重新证明新的 exact/neural relation。

## D-030 - Select V1-P1 interactive matrix-SIS identification and preserve every route

- **Date:** 2026-08-11
- **Decision:** V1 首个协议选择 Lyubashevsky 2012 标准矩阵方案经 Liu--Zhandry 2019 抽取的
  交互式 Sigma protocol，标识为 `CAN-V1-LYU12-SIS-ID-v1`。公开关系为
  `A*r=T*c+a mod q` 与 `sum(r_j^2)<=B2`，prover 使用 Gaussian mask、稀疏 ternary challenge、
  `r=y+S*c` 和 rejection sampling。V0、V1、V2 必须以独立模块、协议标识、registry、adapter 与
  测试长期共存；后续版本不得重命名、改写或覆盖前序代码。
- **Reason:** 该 reviewed 协议把私钥留在 prover，verifier 只持公开 `A,T`，并提供清晰的
  commit--challenge--response 关系，适合作为首个身份协议。它避免首版同时引入 noisy LWE
  rounding、Stern commitments/permutations 或 ML-DSA decomposition/hints，同时保留可审查的安全
  来源。路线隔离则保证 V0 数值实验、V1 身份协议和 V2 标准验签都可单独复现和比较。
- **Alternatives:** 直接把 A4 gadget relation 称为认证；采用 noisy LWE Stern-style ID；立即做
  Fiat--Shamir 签名或 ML-DSA；把 A0/V0 文件改造成 V1。第一项没有不可伪造性，第二项显著扩大首个
  neural relation，第三项超出当前证明范围，最后一项破坏复现、比较和 route isolation，均不采用。
- **Consequences:** V1 需要新的 A3-v2 commit-first 状态机、canonical commitment/challenge/response
  编码、公开 registry、exact reference 和后续 `V1-C1`。A4-C1 的模等式 point-pulse 思路可借鉴，
  但固定 `q=257`/int8/无穷范数 graph 不能直接复用。Fiat--Shamir with aborts 的 ROM/QROM 证明、
  concrete security parameters、prover/sampler 和 neural verifier 继续延期。下一实现只能新增 V1
  模块，并必须用 route-confusion/no-fallback 回归测试证明 V0/A3-v1/A4 行为不变。

## D-031 - Adopt FSwA-S Module-SIS as the current V1 protocol

- **Date:** 2026-08-11
- **Decision:** 按项目负责人决定，当前 V1 主路线从 V1-P1 普通矩阵 SIS baseline 切换为 V1-P2
  `CAN-V1-FSWA-MSIS-ID-v1`。V1-P2 采用 Boudgoust--Takahashi FSwA-S 的底层交互式协议：在
  `R_q=Z_q[X]/(X^N+1)` 上固定 `Abar=[A|I]`、`t=Abar*s`，prover 先发送 `u=Abar*y`，再响应
  server challenge `c` 得到 `z=y+c*s`；verifier 精确检查 `||z||_inf<=B` 和
  `Abar*z=u+c*t`。V1-P1 文档与未来实现命名空间继续保留，只是不再是当前实现目标。
- **Reason:** 该 reviewed relation 同时保留简洁的公开 module equation、bounded-uniform masking 和
  coefficient norm，并引入与 Dilithium/ML-DSA 更接近的商环与 module 结构；第一版 exact oracle 可用
  coefficient-domain negacyclic convolution，暂不引入 HighBits、compression、hint、SHAKE encoding
  或 NTT 表示语义。`t=A*s1+s2` 的公开密钥分布和 transcript soundness 也可分别以 M-LWE 与 M-SIS
  分析，不再把“基于 LWE”误写成 verifier 直接执行 LWE 解密。
- **Alternatives:** 仅把 V1-P1 的矩阵换成多项式而不采用已有协议；直接实现完整 ML-DSA；删除或改写
  V1-P1/V0；把交互式 identification 直接称为 Fiat--Shamir 签名。第一项缺少冻结的协议与证明条件，
  第二项扩大到 V2 范围，第三项破坏路线复现，第四项混淆 ROM/QROM 与交互式安全主张，均不采用。
- **Consequences:** 下一 checkpoint 只能新增独立 V1-P2 profile、registry、parser、exact ring relation
  和 A3-v2 状态机，并覆盖 ring wraparound、canonical encoding、challenge/norm boundary、tamper、
  replay、并发、route confusion 和拒绝零 protected calls。M-LWE、M-SIS、A3 binding、neural
  soundness 与未来 Fiat--Shamir/ML-DSA 结论必须分别验证；V0/A0、V1-P1、A3-v1 和 A4-C1 不改写。

## D-032 - Keep V0 on Fashion-MNIST and use CIFAR-100 for V1 and V2

- **Date:** 2026-08-11
- **Decision:** 按项目负责人澄清，V0 保持现有 Fashion-MNIST + MLP 路线及该路线内可能的 LeNet
  对照，不修改既有 parser、模型、artifact 或实验。V1 与 V2 的业务模型实验均使用 CIFAR-100 与
  CIFAR-style ResNet-18；V2 可以使用同一业务 benchmark 做标准 verifier 对照，但必须具有
  V2-local 协议 adapter、registry、入口和测试，不能复用 V1 的认证接受入口。
- **Reason:** 固定业务模型分界可以保留 V0 的低成本全回归价值，同时让 V1/V2 的认证开销在同一
  更现实 CNN 上可比较；独立认证入口则防止共享业务模型被误解为共享 verifier 或弱回退。
- **Alternatives:** 把 V0 迁移到 CIFAR-100；让 V2 回到 Fashion-MNIST；让 V1/V2 共用一个认证入口。
  第一项破坏已验收基线，第二项失去跨协议模型对照，第三项破坏路线隔离，均不采用。
- **Consequences:** 当前 V1-P2 exact/A3-v2 唯一下一步不变，仍不实现或下载 CIFAR-100/ResNet-18。
  后续 V1-M1 先闭合 CIFAR baseline 与 V1 gate；V2 在 ML-DSA reference/adapter 闭合后绑定同一业务
  benchmark，并单独覆盖 V1/V2 route confusion、拒绝零模型调用和无 fallback。

## D-033 - Accept the non-production V1-P2 exact relation and A3-v2 shell

- **Date:** 2026-08-11
- **Decision:** 接受固定 `N=8,q=257,k_mod=2,ell_mod=2,eta=1,gamma=8,kappa=2,B=6` 的
  `CAN-V1-FSWA-MSIS-ID-v1` 非生产 conformance profile、canonical polynomial encodings、公开
  registry、coefficient-domain exact relation、A3-v2 commit-first 单次终态协调器和 evidence adapter。
- **Reason:** 该 checkpoint 已把商环 coefficient order/wraparound、公开 module equation、norm、
  transcript binding、atomic terminal state 和零 protected-call 拒绝路径连接为可执行且相互隔离的
  reference boundary，为后续 prover 与 neural construction 提供唯一事实源。
- **Alternatives:** 复用 A3-v1/Fashion-MNIST parser；让 response 携带参数、evidence 或 decision；
  invalid relation 后允许同 transcript 重试；用 A4-C1 或 V0 verifier fallback；直接进入 NTT/neural。
  这些路线会破坏 commit-first、route isolation、单次终态或 exact semantic oracle，均不采用。
- **Consequences:** 当前实现只证明非生产 relation conformance 与单进程 A3-v2 状态边界，不证明
  secret possession、HVZK、M-LWE/M-SIS concrete security、主动冒充安全或 neural soundness。下一
  checkpoint 先冻结 prover/sampler/rejection 实验契约；任何安全参数或 neural backend另行验收。

## D-034 - Freeze CIFAR-100 and CIFAR-style ResNet-18 as V1-M1

- **Date:** 2026-08-11
- **Decision:** V1 headline protected-model experiment 固定为
  `CAN-V1-CIFAR100-RESNET18-v1`：CIFAR-100、canonical `(1,3,32,32)` uint8/RGB snapshot、独立
  input digest/profile、CIFAR-style ResNet-18，以及 baseline-before-gate 的 exact/neural 对照顺序。
- **Reason:** 独立 V1 model/input/adapter namespace 保留 V0 Fashion-MNIST/MLP 的已验收回归价值，
  同时使 request binding、zero-call 和认证开销能在更现实 CNN 上独立测量。
- **Alternatives:** 修改 A3-v1 接受 CIFAR；把 A2 MLP 重命名为 V1；复用 V2 认证入口；先下载/训练再
  回填数据摘要、环境和阈值。它们会破坏路线隔离或产生 post-hoc 实验选择，均不采用。
- **Consequences:** 本决定不下载数据、不训练模型。后续必须先冻结 source/digest/license、split、
  preprocessing、训练环境、超参数和预注册阈值，再独立验收 baseline；V1-P2 reject 不得 fallback 到
  A2/V0，未来 V2 只复用业务 benchmark 而不复用认证入口。

## D-035 - Freeze the V1-P2 non-production prover/sampler/rejection contract

- **Date:** 2026-08-12
- **Decision:** 接受 `V1-P2-PSR-E1` 作为 V1-P2 唯一非生产 prover 实验契约。它固定 32-byte toy
  seed 的 SHAKE256 role/counter stream、无偏 secret/mask/challenge sampling、临时 generated-key
  profile、exact `u=Abar*y`/`z=y+c*s`、norm-bound emit/abort、fresh-transcript retry、secret hygiene、
  test-vector families 和分离统计指标。
- **Reason:** exact/A3-v2 checkpoint 已冻结 verifier 接受关系，却没有 honest prover distribution。
  在 neural 或模型实验之前显式固定 sampler/rejection，可避免后续根据结果调整 mask、retry 或参数，
  也把 arithmetic completeness、abort distribution、A3 binding 和密码安全定理保持为独立义务。
- **Alternatives:** 直接用 Python `random`；对 byte 无条件取余；把固定公开 conformance target 当作
  generated key；abort 后保留 commitment/challenge；先实现 neural 或生产 keygen。它们分别破坏跨
  实现复现、引入采样偏差、缺少 witness relation、违反单次终态或越过当前安全边界，均不采用。
- **Consequences:** 当前 toy profile 的理论 emit probability 只有 `(13/17)^32`，期望约 5348 次尝试，
  因此不能作为性能或生产参数。下一 checkpoint 只实现临时 generated-key fixture、deterministic
  samplers 和无 secret vector manifest；A3 retry、security estimator、production library、neural、
  CIFAR、Fiat--Shamir、ML-DSA 与 Stage B 继续延期。

## D-036 - Notify before the first server-required V1-M1 checkpoint

- **Date:** 2026-08-12
- **Decision:** 工作日志的唯一下一步必须同时具有唯一计算资源标记。当前 generated-key/sampler/vector、
  toy prover/A3-v2 retry 和首个 dependency-free `V1-C1-MSIS` checkpoint 使用 `LOCAL_OK`；当唯一
  下一步首次进入 V1-M1 GPU 环境冻结或 CIFAR-100/ResNet-18 正式训练时，先改为
  `SERVER_REQUIRED` 并通知项目负责人，再安装服务器训练环境、下载正式训练数据或启动论文训练。
- **Reason:** 当前 CPU-only 本机足以完成协议、解析、exact relation、神经构造与 smoke test，但不适合
  承担 CIFAR-100/ResNet-18 的正式训练、重复种子和消融。显式触发器可避免过早占用服务器，也避免
  在未冻结 GPU/software tuple 时产生不可复现的论文结果。
- **Alternatives:** 现在立即迁移全部工作到服务器；等训练运行缓慢后再临时决定；只在对话中约定而不
  写入事实源。它们分别增加环境切换成本、允许 post-hoc 实验条件，或无法跨会话强制通知，均不采用。
- **Consequences:** 最低实用服务器估计为单 NVIDIA GPU、至少 8 GiB VRAM 和 16 GiB RAM，优先
  12--16 GiB VRAM 与 32 GiB RAM；准确 GPU、driver/CUDA、PyTorch wheel/hash、batch、epoch 和
  deterministic policy 延迟到 `SERVER_REQUIRED` checkpoint 冻结。在该标记出现并通知前，不产生
  CIFAR-100/ResNet-18 正式服务器实验结果。

## D-037 - Accept the V1-P2 toy generated-key and single-attempt implementation

- **Date:** 2026-08-12
- **Decision:** 接受 `src/can/experiments/v1_psr.py` 作为 `V1-P2-PSR-E1` 步骤 1--2 的唯一非生产
  experiment implementation。它复用现有 V1 public profile、wire objects 与 coefficient-domain
  convolution，并固定 SHAKE256 role/counters、无偏 byte rejection、112-challenge order、临时
  generated target、commit-first single attempt、B=6 emit/abort 和无 secret public manifest。
- **Reason:** exact verifier 已提供唯一 arithmetic oracle，但固定 conformance target 不是由 witness
  生成的 public key。独立 experiment 模块把 secret 限制在短生命周期 harness，同时让 honest
  emitted response 可与现有 verifier 做 differential，不把 secret 或采样逻辑引入 verifier 请求路径。
- **Alternatives:** 在 `reference/v1.py` 中加入 secret/keygen；用 Python `random` 或直接 byte modulo；
  把 response/transcript 写入 manifest；在同一 patch 修改 A3-v2 retry。它们分别污染公开 verifier
  边界、破坏跨实现无偏复现、违反 artifact policy，或扩大当前 checkpoint，均不采用。
- **Consequences:** 当前结果只支持 toy sampler、有限域计数、single-attempt completeness 和
  implementation conformance，不证明 key secrecy、M-LWE/M-SIS security、HVZK、主动冒充安全或
  不可伪造性。下一 checkpoint 必须用全新 A3-v2 transcript 实现 retry/exhaustion，不能让 abort 后的
  commitment、challenge、mask、nonce 或 transcript 复用；之后才进入 `V1-C1-MSIS`。

## D-038 - Freeze the V1-M1 GPU software tuple and defer the data/training protocol

- **Date:** 2026-08-15
- **Decision:** 接受项目负责人在已授权 AutoDL 单 GPU 容器中完成的 V1-M1 GPU/software tuple：Ubuntu
  22.04.1、`can-v1` CPython 3.11.9、NVIDIA RTX A4000 16,376 MiB、driver 580.82.07、PyTorch
  `2.13.0+cu126` 和 torchvision `0.28.0+cu126`。两个 primary wheel 的 SHA-256、CUDA runtime、
  环境安装/激活命令、CUDA forward/backward smoke 和同进程 deterministic smoke 记录在
  `docs/V1_MODEL_EXPERIMENT_DECISION.md` section 8。A2 CPU lock 不随此决定修改。
- **Reason:** V1-M1 已进入 `SERVER_REQUIRED` 并完成通知；实际 probe 显示该 environment 可运行 CUDA
  PyTorch，且 `pip check` 无依赖断裂。将 driver compatibility CUDA 13.0 与 wheel runtime CUDA 12.6
  分开记录，避免将 `nvidia-smi` 的兼容性读数误当为训练 runtime。以两个 wheel 的哈希锁定 primary
  binary provenance，同时不把随机 micro-smoke 夸大为训练或性能结论。
- **Alternatives:** 继续把 V1 假定为 CPU-only；立即下载/训练再回填数据或超参数；把 GPU wheel 写入
  A2 CPU lock；由 `nvidia-smi` 的 13.0 读数替代 torch CUDA runtime。前三者分别失去已验证资源、允许
  post-hoc protocol 选择或破坏既有 A2 可复现性，最后一项混淆 driver 与 wheel runtime，均不采用。
- **Consequences:** GPU/software 子步骤完成，但 V1-M1 仍为 in-progress：CIFAR-100 source/digest/license、
  split、preprocessing、batch/data-loader、optimizer/scheduler、epochs、checkpoint rule、seeds 和
  baseline threshold 必须先作为一个 `LOCAL_OK` 决策冻结。此前不下载数据、不安装预训练权重、不训练、
  不产生论文准确率或吞吐量结论；首次正式训练重新标记 `SERVER_REQUIRED` 并只使用此 tuple。
- **Verification:** `bash -n scripts/check_governance_docs.sh`、`./scripts/check_governance_docs.sh` 和
  `git diff --check` 于本 checkpoint 通过。提交候选仅为 `PROJECT_WORKLOG.md`、
  `docs/V1_MODEL_EXPERIMENT_DECISION.md` 和 `README.md`；未修改 runtime、依赖锁、数据或模型代码。

## D-039 - Record the V1 AutoDL environment restart and migration procedure

- **Date:** 2026-08-15
- **Decision:** 新增 `docs/V1_AUTODL_ENVIRONMENT_SETUP.md`，作为已冻结 V1-M1 AutoDL A4000 tuple 的
  操作手册。它规定同一实例重启只做 environment verification；任何换服、释放后重建、系统重置或弹性
  部署切换均按全新 environment 重建，重新记录 hardware/software provenance 并比较 primary wheel
  SHA-256。
- **Reason:** `/root` 的 Conda environment 可在同一实例关机重开后保存，但它不是可迁移 artifact；将
  这两种情形分开可避免把不同 GPU/driver/runtime 的运行误当作一次已冻结实验的直接复现。
- **Consequences:** 当前唯一下一步仍是 `LOCAL_OK` 的 CIFAR-100 数据与训练协议冻结。该手册不下载数据、
  不安装预训练权重、不启动训练，也不修改 A2 CPU lock 或 runtime code。
- **Verification:** `bash -n scripts/check_governance_docs.sh`、`./scripts/check_governance_docs.sh` 和
  `git diff --check` 通过。当前提交候选为 `PROJECT_WORKLOG.md`、`README.md`、
  `docs/V1_MODEL_EXPERIMENT_DECISION.md` 和 `docs/V1_AUTODL_ENVIRONMENT_SETUP.md`。

## D-040 - Freeze the V1-M1 CIFAR-100 data and training protocol

- **Date:** 2026-08-15
- **Decision:** 固定首方 CIFAR-100 Python archive URL、`169001437` byte size、SHA-256
  `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`、首方 MD5、许可证未声明时的
  no-redistribution boundary、100 个 fine-label order、`45,000/5,000/10,000` train/validation/test split、
  trusted preprocessing、train-only crop/flip、two-run seeds `1729/1730`、batch/data-loader policy、200 epoch
  SGD/cosine training、validation-only checkpoint rule 以及 `70.00%`/`2.00 pp` baseline acceptance rules。
  完整值与下载后的 decoded-digest 算法见 `docs/V1_MODEL_EXPERIMENT_DECISION.md` sections 4, 7, 9 and 10。
- **Reason:** 在下载前预注册数据身份、切分、模型输入语义和所有优化选择，使 test split 不会驱动
  checkpoint、超参数、重试或 acceptance threshold。首方页面只公布 archive、大小、MD5 和 citation；
  SHA-256/byte size 由 Hugging Face Datasets 对相同 URL 的 source record 交叉记录。未找到首方许可证或
  redistribution grant，故不能把可下载性误称为发布授权。
- **Alternatives:** 使用非首方镜像作为下载入口；以 test accuracy 选择 checkpoint/seed；下载后再选择
  augment、batch、epochs 或 acceptance threshold；把 archive、图像或 weights 放进仓库。它们分别破坏
  source provenance、引入 test leakage 或 post-hoc selection、损害可复现性，或违反 artifact/许可边界，
  均不采用。
- **Consequences:** V1-M1 GPU、data 与 training protocol 已闭合，但不产生任何模型、数据或实验结果。
  当前唯一下一步转为 `LOCAL_OK` 的 isolated V1-M1 implementation 与 unit/security tests；该步骤不得下载
  CIFAR-100。implementation 完成后，首次 archive 下载或两次 baseline training 必须先改为
  `SERVER_REQUIRED`、通知项目负责人，并仅使用已冻结 AutoDL A4000 tuple。V0/A0、V1-P1、A3-v1、
  A4-C1、V1-C1、Fiat--Shamir、ML-DSA 与 Stage B 不在本 checkpoint 范围内。
- **Verification:** `bash -n scripts/check_governance_docs.sh`、`./scripts/check_governance_docs.sh` 和
  `git diff --check` 已于本 checkpoint 的文档修改后通过；未运行任何 download、training、GPU benchmark 或
  dataset parser。提交候选为 `PROJECT_WORKLOG.md`、`README.md`、
  `docs/V1_MODEL_EXPERIMENT_DECISION.md` 和 `docs/V1_AUTODL_ENVIRONMENT_SETUP.md`。

## D-041 - Implement the local V1-M1 isolated baseline boundary

- **Date:** 2026-08-15
- **Decision:** 新增独立 V1-M1 CIFAR-style ResNet-18（11,220,132 parameters）、严格 raw uint8
  adapter/`V1M1AccessCoordinator`、仅接受已经存在并通过固定 size/SHA-256/MD5 校验 archive 的
  CIFAR-100 parser，以及固定 split、augmentation、SGD/cosine、validation-only model selection 的
  CUDA baseline runner。parser 在读取 pickle 前验证解压的 `train`、`test`、`meta` 与 archive 成员逐字节
  对应，并拒绝未知字段。runner 不下载、解压、写 artifact 或由本机调用。
- **Reason:** 使冻结的 V1-M1 protocol 在数据或 GPU 结果产生前具有可审查的实现边界；将不可信 raw
  image 规范化、A3-v2 evidence-only authorization 和模型调用保持为独立 V1 route，同时防止手工更改的
  解压文件绕过 archive identity。
- **Alternatives:** 复用 A2 Fashion-MNIST model/parser；让调用方提交 `A3V2TrustedInput`；在 unit test
  中下载 CIFAR-100；先用 CPU 运行 200 epochs；忽略 archive 与 extracted tree 的对应关系。它们分别破坏
  route isolation、trusted-input ownership、资源边界、正式训练协议或数据供应链绑定，均不采用。
- **Consequences:** isolated implementation 已完成；没有 archive、解压数据、weights、checkpoint、正式
  accuracy、prediction digest、吞吐量或 latency 结果。唯一下一步变为 `SERVER_REQUIRED` 的首次 archive
  下载与两次训练；该服务器 run 还必须实现 ignored artifact/report persistence，并在 baseline 验收后才进入
  gate 与性能报告。
- **Verification:** V1-M1 focused unit/security suite `25 passed`；`.venv/bin/python -m pytest` 收集并
  通过 620 项；`.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`、`.venv/bin/mypy src tests` 与
  `.venv/bin/python -m pip check` 通过；`bash -n scripts/check_governance_docs.sh`、
  `./scripts/check_governance_docs.sh` 和 `git diff --check` 通过。提交候选为
  `PROJECT_WORKLOG.md`、`README.md`、`docs/V1_MODEL_EXPERIMENT_DECISION.md`、
  `docs/V1_AUTODL_ENVIRONMENT_SETUP.md`、`src/can/access/__init__.py`、
  `src/can/access/v1_m1_adapter.py`、`src/can/experiments/v1_m1_baseline.py`、
  `src/can/model/__init__.py`、`src/can/model/v1_cifar100_resnet.py`、
  `tests/security/test_v1_m1_route_security.py`、`tests/unit/test_v1_cifar100_resnet.py`、
  `tests/unit/test_v1_m1_adapter.py` 和 `tests/unit/test_v1_m1_baseline.py`。

## D-042 - Close the V1-M1 artifact writer before server execution

- **Date:** 2026-08-15
- **Decision:** `run_v1_m1_baseline` 现在在 validation-only selection 和单次 test 后，将选定 CPU
  `state_dict`、`manifest.json` 和 `report.json` 原子写入 ignored `artifacts/v1-m1/run-{1,2}/`。路径
  预检、创建和文件写入均拒绝覆盖及 symlink；state 文件只保存 tensor state，不保存 optimizer state，
  report/manifest 只保存公开数据、训练、模型、环境和摘要信息。项目负责人同时更新仓库规则：默认仍不推送，
  但有明确授权时可直接推送当前 `main`，只有明确要求备份时才使用 `backup/<current-branch>`。
- **Reason:** 两次服务器训练必须留下可审计且可供后续 gate 物化的 selected state，不能只返回临时内存
  结果；明确授权的 main push 使服务器能检出同一源码 checkpoint，而不把数据或权重传入 Git。
- **Alternatives:** 服务器临时手工序列化模型；总是创建 backup branch；在测试中生成真实 CIFAR artifact；
  覆盖同一 run 的既有结果。它们分别削弱可复现接口、违背项目负责人当前发布策略、越过资源边界或破坏
  artifact provenance，均不采用。
- **Consequences:** 当前唯一下一步是 `LOCAL_OK` 的 commit 和授权直推 `origin/main`；成功后恢复为
  `SERVER_REQUIRED` 的正式 CIFAR-100 下载与两次 baseline。没有下载数据、没有训练、没有生成真实 state、
  manifest、report 或模型结果。
- **Verification:** V1-M1 artifact focused unit/security subset `14 passed`；完整 `.venv/bin/python -m pytest`
  收集并通过 622 项；`.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`、
  `.venv/bin/mypy src tests` 和 `.venv/bin/python -m pip check` 通过。治理文档检查、最终候选清单、commit 和
  push 结果在本条之后复核。

## D-043 - Publish the V1-M1 source checkpoint on main

- **Date:** 2026-08-15
- **Decision:** 按项目负责人明确授权，不创建 `backup/main`，将 `c6c38df`（`Implement V1-M1 artifacts and
  training boundary`）直接推送到 `origin/main`，并将本地分支设置为跟踪 `origin/main`。项目规则同时改为：
  默认不推送；有负责人明确授权的 checkpoint 可直推当前 `main`；只有明确要求备份时才使用
  `backup/<current-branch>`。
- **Reason:** 服务器必须检出与本机测试相同的 source/test/configuration/documentation checkpoint；单独
  backup 分支既非项目负责人当前要求，也不能替代服务器所需的 `main` commit。
- **Consequences:** 服务器现在可检出 `c6c38df`；远端不含 CIFAR data、weights、state、manifest、report、
  secret 或其他 generated artifact。唯一下一步已重新标为 `SERVER_REQUIRED` 的正式数据准备与两次训练。
- **Verification:** `git push -u origin main` 输出 `ae79db1..c6c38df  main -> main`，并确认 upstream 为
  `origin/main`。本条发布记录将在单独的 documentation commit 中推送；除 `PROJECT_WORKLOG.md` 外不新增
  发布内容。
