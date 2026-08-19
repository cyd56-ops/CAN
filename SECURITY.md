# Security

## Status and scope

本项目是防御性科研原型，研究格密码验证网络如何控制模型和工具能力。当前已实现非生产 A0-v1
toy 精确整数 oracle、`CAN-RELU-EXACT-v1` dependency-free conformance backend 和 A1-B1
`CAN-TORCH-CPU-EXACT-v1`，并具有协议、全域差分和防御性安全测试。A2-E1 已实现并复现
Fashion-MNIST/MLP baseline、单一协调器和二元前置硬门控，包括严格业务 tensor 校验、固定响应、
拒绝零 protected-model calls 与 10,000 个 allow 标签等价。A2-E2 独立 public coarse-model
无门控 baseline 已实现并两次复现；默认关闭且本地绑定的三态协调器也已实现，固定 version-2
响应、调用计数、并发、异常、不可升级和无 fallback 测试通过。trusted materializer 已重新物化并
校验两个已验收 state，no-training evaluator 已完成真实 10,000-image 三态报告。A3-v1 协议规格现已
固定 canonical message/input digest、60 秒 challenge、可信 nonce 状态、原子 consume 和安全验收矩阵；
A3 默认关闭的运行时壳已实现。A4 已选择 GPV PFDH 短原像公开验证关系，冻结
`A4-GPV-PFDH-TOY-v1` 并实现无私钥 exact reference；A4-C1 又冻结并实现
`CAN-RELU-A4-PFDH-TOY-v1` dependency-free exact graph 和 A3 neural evidence adapter。生产参数与
不可伪造性证明尚未实现。当前 V1-P2 已选择 reviewed FSwA-S 交互式 Module-SIS Sigma protocol，
V1-P1 普通矩阵方案保留为 baseline；非生产 V1-P2 conformance profile、canonical polynomial
parser、公开 registry、coefficient-domain exact reference、A3-v2 commit-first 单次终态协调器和
evidence adapter 已实现；`V1-P2-PSR-E1` 也已实现临时 generated-key fixture、确定性 bounded-uniform
sampler、single-attempt emit/abort、A3-v2 fresh-transcript retry/exhaustion harness 和无 secret manifest。
`CAN-RELU-V1-MSIS-COEFF-v1` dependency-free integer graph、neural evidence adapter 及其 A3-v2
route-level zero-call tests 也已实现。生产 keygen/prover、密码安全参数、NTT、PyTorch/qint8/CUDA/export
和性能结论仍未实现。上述结论只适用于本机 toy
credential/分类实验，不是安全承载
密码、服务部署或生产访问控制保证。

本文档描述计划中的信任模型与必须保持的安全不变量，不构成生产安全保证。任何实现与实验结果都必须明确标注 toy、研究性、黑盒假设和未证明部分。

### Route boundary

路线按 `V0 -> V1-prep -> V1 -> V2` 推进，并与 A0--A4 工程编号并行。V0 是 toy LWE 数值解锁；
V1-prep 是 A3 请求绑定/新鲜性与 A4 canonical `(y,z)` 代数关系的神经编译；V1-P2 已选择 reviewed
交互式 Module-SIS challenge-response/身份协议，并已实现独立 exact/A3-v2 conformance 路径；V2 才
研究 ML-DSA 标准 verifier。V1-prep 的 evidence 不能被解释为身份认证，V2 也不属于当前首版安全
承载范围。

V0、V1 和 V2 的代码路径必须独立保留。不得把 A0/V0 模块重命名、改写或替换为 V1，也不得让 V2
覆盖 V0/V1。每条路线的协议标识、registry、parser、adapter 和测试相互隔离，任何跨路线 fallback、
route confusion 或让旧凭据进入新 verifier 的行为都必须 fail closed。

## Trust model

阶段 A 的初始安全边界是假定的受控黑盒推理服务：

- 模型结构、权重、推理代码、规范解析器和访问协调器可信且不可被请求方修改；
- 请求方只能通过规定入口提交业务输入和认证材料；
- 请求方不能读取受保护中间特征或直接调用验证层之后的业务网络；
- 本地 profile registry、公开验证参数和策略配置可信；
- 在需要 replay 防护的阶段，nonce/challenge 状态存储可信；
- 业务网络本身不单独提交认证或授权；V1-M1-C1 的受保护业务分支只能作为组合模型内部、由 Gate
  Layer evidence 和协调器决定控制的分支调用。M1-C2 只在同一假设下增加显式 public expert 与
  protected expert 二路硬分派；后续 M2 才增加多个 protected experts 和权限掩码。唯一协调器始终
  提交 public/protected/deny，未经 protected 提交不得调用 suffix 或受保护专家。

业务输入 `x` 与 credential 必须使用独立 schema 和解析路径。V1-M1/M2 中 credential 是独立的 commitment/challenge/response transcript，绑定 image digest、model/profile、nonce 和 expiry；客户端私有材料不得进入 Gate Layer、R2、日志或 checkpoint。V1 主路线不把该 transcript 称为 Secret Trigger。若未来增加静态 trigger 对照，只能在独立 protocol identifier 下作为可重放 bearer-gate，并明确不提供身份、不可伪造性或 anti-replay；它不得依赖业务输入中的隐藏像素模式、纹理、提示词或普通特征，不得通过后门训练实现，也不得成为 V1 fallback。

阶段 B 额外信任唯一授权协调器、capability issuer 和工具网关。Router、LLM、规划专家、业务专家及其自然语言输出均不可信，不能提交权限。

## Protected assets

- CNN/DNN 的正常预测、logits、中间特征和其他受保护模型能力；
- 外部数据库、文件、API、执行环境和工具副作用；
- 本地授权策略、profile registry 和 capability 签发状态；
- challenge/nonce 使用状态和审计记录完整性；
- 签名私钥及任何真实凭据；
- 尚未公开的论文材料、实验数据和模型 artifacts。

## Attacker capabilities

首阶段攻击者可以：

- 构造任意业务输入和认证字节；
- 提交未知字段、重复字段、错误长度、错误 dtype 和非规范编码；
- 使用 NaN、Inf、极值、溢出候选、负数模表示和边界值；
- 自适应重复查询并观察最终响应及可见延迟；
- replay 历史请求，替换身份、scope、模型、业务输入或凭据字段；
- 尝试指定更弱算法、参数、矩阵、模数、阈值或验证 profile；
- 在阶段 B 尝试通过 Router、提示注入或业务专家直接调用工具。

首阶段不声称抵抗读取或修改模型权重、修改推理代码、删层、剪枝、微调、模型蒸馏、直接调用中间层、宿主机控制或物理侧信道。这些能力只有在后续明确扩展威胁模型并提供对应保护后才能纳入保证。V1-M2 可研究受信 state reload、module refactor、quantization、pruning、fine-tuning 或 export 后是否仍满足重新验收条件，但不得将这种 transform evaluation 写成对恶意模型改写者的防御。

## Trust boundaries

```text
untrusted bytes
  -> canonical parser
  -> typed request
  -> deterministic verifier
  -> evidence without authority
  -> single trusted authorization coordinator
  -> committed capability/context
  -> protected model or gateway
  -> protected side effect
```

每个箭头都是需要独立测试的边界。任何旁路入口、调试接口、模型 hook 或直接工具端点都视为安全缺陷。

## Input validation rules

- 先解析并规范化，再进行密码验证；解析失败直接拒绝。
- schema 默认封闭：未知字段和重复字段拒绝。
- 字段类型、shape、长度、字节序、整数范围和编码必须精确匹配。
- `bool` 不得作为安全关键整数接受；不得依赖 truthy/falsy 转换。
- 非有限浮点值、负零歧义、越界值和会触发整数溢出的输入拒绝。
- 安全关键认证字段优先使用规范字节串或有界整数，不接受任意实数插值。
- credential tensor 与业务输入 tensor 在解析、shape、dtype 和调用接口上保持分离，不能通过拼接业务特征绕过 credential parser。
- A1-v1 credential tensor 只接受精确 shape `(8,)`、有符号 `int32`、scale `1` 和逐元素 `[0,256]`；隐式 batch、广播、reshape 或类型转换均拒绝。
- 模数负值和等价剩余必须转换为唯一规范表示；非规范输入默认拒绝而不是静默修复。
- 算法、`q`、矩阵、阈值、量化 scale、profile 和密钥用途由本地可信 registry 选择。
- 请求方提交的 `A`、参数集或算法标识不能覆盖本地选择，也不能触发弱回退。
- 落入数值模糊区或证明覆盖范围外的输入一律拒绝。

## Authentication and authorization flow

### A0 toy LWE unlock

A0 只验证数值关系和门控机制，不提供身份认证或数字签名保证。A0-v1 只接受 `docs/A0_PROTOCOL_SPEC.md` 定义的 23 字节凭据；请求方不能提交 `A`、`q` 或阈值，`A_slot` 由本地 registry 解析。必须显式测试 chosen `A` 编码尝试、chosen `b`、零矩阵 registry 和 decryption-oracle 风险。任何 toy secret 都只能在测试临时目录或进程内生成，不能作为真实凭据使用或提交。

当前 `src/can/reference/` 严格拒绝非 `bytes`、错误长度、未知版本、非规范 `b`、未知/禁用 slot、错误矩阵 shape/range、全零行和 bool/int 混淆。registry 与 slot 在加载后不可变；`V_ref` 只返回内部结构化证据码、逐分量距离和最大距离，不返回 gate、decision 或 capability。Python 精确整数在已验证的 A0 范围内与规格要求的 `int64` 结果一致，不依赖溢出或负余数行为。

当前独立业务 MLP 已通过 `src/can/access/a2_gate.py` 连接到固定 A1-B1 verifier 和唯一协调器。
公共入口只接收单张规范业务图像与原始 credential；拒绝响应固定为 `version/status`，allow 只增加
top-1 `class_id`。调用计数、安全测试和真实 rejected probe 支持所测拒绝路径零 protected-model
calls，但不阻止黑盒入口之外的直接模型调用，也不提供 credential 不可伪造性。

### Security-bearing Stage A

安全承载版本应使用公开验证信息。验证器只返回结构化证据，例如 profile、绑定消息摘要和验证状态；它不能直接铸造 gate、capability 或权限 context。唯一协调器根据本地策略提交 allow/deny，并在 allow 后才调用业务网络。V1-M1 已实现的组合对象 `AuthenticatedR2=(V_phi,C,f_theta)` 将固定神经 `V_phi` 作为模型内部 Gate Layer、将 `C` 作为内部可信协调器、将 `PROJECT_WORKLOG.md` 记录的 R2 `f_theta` 作为受保护分支；这不把 verifier 变成授权主体，也不允许请求方通过组合对象的公开入口直接选择 R2。

“认证神经元/层”只表示固定验证网络模块，不是新的授权主体，也不要求它只有一个物理层。
V1-C1 的 `V_phi` 由 exact relation、canonical domain 和可信公开 profile 确定性编译为多层
affine/ReLU graph；topology、weights、bias 和阈值来自关系构造，不从 credential 样本训练。
线性映射、模距离、阈值和逻辑聚合可以在该模块中研究；本地 registry 仍固定算法、矩阵、阈值、
量化 scale 和 profile，任何候选激活函数或网络构造都不能由请求方选择。当前 V1-M1 主路线冻结
Gate Layer 与 R2，不进行联合训练或 learned/soft gate；拒绝必须在 R2 forward 前返回固定 deny，
不能使用 `gate * logits` 或输出遮蔽冒充零调用。

NNAES 只提供“密码逻辑可确定性映射为固定神经图”的方法类比。其 AES round key 在构建时嵌入
key-specific 执行网络；CAN 安全承载 verifier 不采用该密钥布局，只保存公开验证参数，签名私钥和
长期 secret 必须留在模型之外。NNAES 类工作对非标准连续输入的攻击也意味着 canonical parser 是
当前安全边界的组成部分：只有规范定长字节/有界整数进入 `V_phi`，域外实值不能继承域内
`V_nn==V_ref` 证明。

V1-M1-C1 保留为内部最小 reference/evaluator，不作为长期 public/protected 能力。C1 的 accepted-R2
报告闭合后，M1-C2 才研究二专家 capability tiering：设冻结 R2 为
`f_theta=d_theta o s_theta`，显式 public entry 只能执行 `E0=g_psi(s_theta(x))` 并返回预注册的
粗粒度 public label；protected entry 仅在 A3-v2 coordinator 提交后执行
`E1=d_theta(s_theta(x))`，且其 logits 必须与直接 R2 完全相同。public/reject path 不得执行 suffix、
返回 prefix feature、protected logits 或可升级 token；protected protocol 失败、scope mismatch 或
空授权结果必须 deny，不能回退至 public。`g_psi` 可独立训练，但 R2 `theta`、`V_phi` 和 credential
relation 必须冻结。

V1-M2 只在 C2 闭合后研究多受保护专家路由。固定 verifier 仍只产生 evidence；canonical claims
由严格 parser 产生，唯一协调器依据可信本地 policy/registry 提交不可变 `RouteContext` 与
`allowed_mask`。请求方、verifier、普通 MoE router 和业务专家均不能提交或扩大该 mask；task router
只可在允许集合中选择。任何 `j > 0` 的专家实际执行都必须蕴含 reference verifier 接受、协调器已
提交 `PROTECTED` 且 `j` 属于 credential 绑定 scope。上述性质只是在可信组合入口内的路由安全，
不改变本文件对模型权重/推理图可改写者的不保证。

A1 共同数值契约要求本地编译器以 A0 精确 `int64` 语义生成规范相位锚点，并固定 `int32`/scale `1` 的输入、残差、模距离、阈值和八路 AND 语义。modulo 的不连续边界必须逐分支证明或在完整有限域上直接证明，不能用随机差分准确率或未经证明的普通 Lipschitz 界替代。候选实现只有在证明逐分量总距离误差 `<=4`、阈值/AND 精确且全部非规范输入 fail closed 后，才能支持 `V_nn=1 -> V_ref=1` 主张。

A1-C1 选择 `8->40->16->1` 的固定整数 affine/ReLU graph：五个 ReLU/分量实现有界 exact modular distance，两个 ReLU/分量实现整数阈值，最终一个 ReLU 实现八路 AND。规范锚点 `t` 折叠进第一层 bias，所有主图数值为 `int32`/scale `1`，理论误差为 `0`。当前 dependency-free backend 已实现不可变 compiled profile/registry、严格 adapter、私有 core 和无授权能力 evidence；完整 toy 域差分为零 false accept、issuer-core 零 false reject，安全测试证明内部异常不触发 reference fallback。普通 Floor/modulo/compare 不进入主 core。

这些测试证明 dependency-free graph 与 A0 toy relation 的单向包含，不证明 credential 不可伪造。
A2-E1 另行测试了固定公共入口上的协调器和拒绝零模型调用，但不证明进程所有者无法绕过入口。
任何目标 backend 都必须重新执行完整性质测试。

A1-B1 固定 Linux x86_64、PyTorch `2.13.0+cpu`、CPU eager mode、`int32`
weight/bias/activation、`int64` reduction、scale `1` 和 zero-point `0`。该 backend 只允许固定
`mul/sum/add/clamp` 映射；qint8、CUDA、export 和 float 均不属于支持路线。版本、device、buffer、
range 或差分 gate 失败时只产生 `CONFIG_REJECT` 并禁用 backend，不得调用 dependency-free
evaluator、exact-ops 或 `V_ref` 兜底。compiled buffers 只允许在内存/测试临时目录生成且禁止
序列化。当前实现逐次验证环境与 buffer contract，startup gate 重算 range ledger 并完成实际
profile 的逐分量全域差分；security tests 覆盖 buffer content/shape/stride/dtype/device 漂移、
training/persistence/hook 篡改、operator exception、错误输出、并发 replay、无 fallback 和无
artifact。结果只支持指定单机 CPU wheel 上的 toy exact-integer 路线，不支持 qint8、CUDA、
export、认证或业务授权主张。

A2-E1 固定业务请求只能提交规范业务输入与原始 23 字节 credential。请求方不能提交 evidence、
allow/deny、model/backend、profile 或策略；唯一协调器必须调用本地 A1-B1 verifier，并且只有精确
`NUMERIC_ACCEPT` evidence 才能提交内部 allow。所有其他 evidence、错误类型、解析/配置/算子异常
和并发重复拒绝都必须返回相同 deny envelope，受保护 MLP 调用计数为零。allow 只返回 top-1
类别，不返回 logits、特征或 evidence。上述 coordinator/gate 已实现；unit/integration/security
测试覆盖注入、类型混淆、异常、backend 失活、并发拒绝和无 fallback，真实 gate 实验验证全部
10,000 个 allow 标签与 baseline 一致，并以 rejected probe 观测零 protected-model calls。A0
credential 仍可 replay，业务输入未绑定到 credential，不能据此声称认证、新鲜性或 tamper binding。

### Tiered model capability

第一阶段已实现 fail-closed 的二元 protected-model gate：没有已提交的 protected capability 就不调用
受保护模型。A2-E2 独立 public model baseline、默认关闭的本地 policy 和单一三态协调器现已实现。
当前实现保持以下固定边界：

- public capability 只返回独立 `784 -> 64 -> 2` public MLP 的 footwear/non-footwear coarse class，
  不共享 protected 权重、head、feature 或 artifact；
- public entry 由本地可信部署配置绑定且默认不存在；request payload 不能选择 entry、policy、
  capability、model/head、backend、evidence 或 decision；
- protected 验证失败始终返回 deny，不能选择 public；public 不是 verifier fallback；
- public path 不调用 verifier 或 protected model/head，不计算或释放 protected logits/中间特征；
- 请求方不能把 public capability 升级、重标记或复用为 protected capability；
- `DENY`/`PUBLIC`/`PROTECTED` 只由同一协调器提交，三者互斥且没有可转移 bearer token；
- 独立 head、共享 trunk、浅层/深层和 MASK 只保留为后续比较，不属于主路线或 fallback。

协调器在提交 `PUBLIC` 或 `PROTECTED` 后才进入对应模型边界。若所选模型抛出异常或返回非规范
logits，外部结果固定为 deny，但不会二次提交另一决定，也不会调用另一模型；计数仍准确记录已经
进入的所选模型。unit/integration/security 测试覆盖默认关闭、三态成功/拒绝、严格图像与 credential
边界、request 字段注入、底层参数 storage 分离、异常、并发和响应复用。受信 materializer 只保存
ignored CPU float32 `state_dict`，加载时严格校验 canonical manifest、文件摘要、dtype/device/layout、
拓扑和 canonical state digest。真实已验收权重的 10,000-image 报告已验证两个预测摘要与 baseline
一致；完整计数为 10,000 次 public model、10,000 次 protected model，拒绝探针为零模型调用。

### A3 request binding and freshness

`docs/A3_CHALLENGE_RESPONSE_SPEC.md` 已固定 A3-v1 协议语义，运行时壳已按该规格实现。A3 只允许服务端
签发的 133 字节规范 message，绑定版本、本地模型、32 字节 identity、scope、签发/到期时间、32 字节
nonce 和规范业务输入 SHA-256。图像在 hash 前必须按 A2 contract 严格验证、拒绝 negative zero 并
detach/clone；同一快照用于后续 protected call，禁止验证后替换。

当前 A4 exact/neural adapter 只产生绑定 exact message digest 与 identity 的不可变 evidence。V1
exact/neural verifier 使用独立 evidence 类型并保持 evidence-only 契约。唯一协调器必须在接受 evidence
后原子执行 `PENDING -> CONSUMED`，并在同一原子操作中复核 binding 与 expiry；
只有唯一成功者可以提交 protected decision。A0/A1 evidence、A2 public response、客户端 key/profile、
布尔值或 decision 均不能进入 A3 接受路径。没有本地 A4 profile 时 A3 protected entry 默认关闭。

### A4 GPV public-verification relation

`docs/A4_GPV_RELATION_SPEC.md` 固定非生产 `A4-GPV-PFDH-TOY-v1`。proof 是 exact 105 bytes：
版本、32 字节 salt 和 72 个 signed-int8 系数。local profile 只包含 32 字节 identity、`q=257` 的
`8 x 72` 满行秩公开矩阵及公开摘要；请求方不能提交矩阵、profile、模数、norm bound、hash 或
decision。reference relation 重新计算 SHAKE256 hash-to-syndrome，要求 `||z||_inf <= 1` 且
`A*z mod 257` 与八分量 target 完全相等。

`src/can/reference/a4.py` 不导入 signer、trapdoor、A0/A1 fallback 或模型；
`src/can/access/a4_adapter.py` 只把 exact result 转换为 A3 evidence。测试中的 gadget 矩阵可让任何人
公开构造短原像，故只验证编码、关系与协调器组合，不能证明私钥持有或不可伪造性。GPV 论文的
random-oracle/SIS 归约不自动适用于当前 toy 参数和具体 SHAKE256 映射。

当前 A4 是 V1-prep 的 neural algebraic-core checkpoint。它只证明 canonical `(y,z)` 的
exact relation 与神经实现边界；A4-C1 当前已用固定 point-pulse ReLU graph 证明并实现该边界。它不
证明 V1 challenge-response 的知识可靠性、不可伪造性或身份授权。
V1-P2 已选择具体 reviewed protocol，仍必须把 M-LWE public-key pseudorandomness、M-SIS
knowledge soundness、transcript/rejection 和 neural core soundness 分开论证。V1-C1 已以独立
negacyclic-convolution coefficient graph 实现全部 canonical input 的 `V_nn==V_ref`，但该算术结论不
提供 M-LWE/M-SIS、HVZK、主动冒充或身份授权结论。

### V1-P2 interactive Module-SIS identification decision

V1-P2 在 `R_q=Z_q[X]/(X^N+1)` 上使用公开 `Abar=[A|I]`、`t=Abar*s`，prover 单独持有短 module
vector `s`。协议按 `u=Abar*y`、服务端 fixed-weight ternary polynomial challenge `c`、response
`z=y+c*s` 运行，并以 bounded-uniform rejection 控制输出。exact verifier 只检查
`Abar*z=u+c*t` 和 `||z||_inf<=B`，不得持有 `s`、mask 状态或授权能力。

现有 A3-v1 是先签发 challenge 的壳，不能直接承载该 Sigma protocol。独立 A3-v2 现已实现：先
持久化 commitment 与业务输入快照，再采样 challenge；同一 commitment 的一个 parsed response
attempt 原子终结，显式 abort 和 expiry 也终结。并发重复 response 最多一次 verifier/allow/protected
call，解析、route、relation、expiry、replay 与内部错误均固定拒绝且无 pre-commit protected call。
该实现尚不提供 concrete-parameter security、主动冒充安全或 Fiat--Shamir 签名安全。V1-C1 对固定
toy profile 的 compiled arithmetic 已证明 `V_nn==V_ref`，其结论与上述协议安全主张保持分离。

A3-v1 只要求单进程 in-memory store 的线程安全线性化语义。invalid proof 不消费 pending challenge；
valid proof 的并发提交最多一次 consume 和一次 protected call。consume 后不回滚，因此 model/response
失败可能已进入一次模型，不能误记为零调用或用同一 nonce 重试。重启使所有 outstanding challenge
失效；分布式一致性、持久化恢复、TLS/channel binding、速率限制和拒绝服务防护仍不受保证。

### Deferred neural construction hypotheses

A1-C1 已决定主路线不使用 Secret Trigger 术语、普通 Floor/modulo/compare、Sigmoid、MASK 或显式 runtime `A*s`：主 relation 使用有界 exact ReLU 构造和折叠锚点，普通整数路线只作互斥测试基线。通用 sawtooth、LWE/SIS 兼容性和形式证明工具仍待后续研究，MASK 与层深能力映射仍延期到 A2；任何未实现候选都不允许进入当前安全主张或默认执行路径。静态高熵 trigger 若未来实现，只能作为隔离的非密码 bearer-gate 对照，使用独立 parser、identifier 和测试，不与任一 exact/neural verifier 进行 `OR` 组合。

无论延期评估结果如何，验证器只产生 evidence、协调器提交权限以及验证失败零受保护副作用是不可降级架构约束。MASK 或层内零化可以作为输出遮蔽对照实验，但不能与主验证路线进行 OR 回退，也不能替代调用 protected model 之前的协调器决定。

### Stage B tool authorization

身份认证证据先进入本地授权策略，随后由唯一协调器创建短期 capability。Router 和业务专家只能携带 capability 提议工具调用。工具网关必须把 capability 与真实工具、资源和参数重新匹配，验证期限、nonce 和次数后才能执行副作用。

请求方、Router、LLM 或业务专家提供的 `decision: allow` 一律不具有权限语义。

## Key lifecycle

- 真实签名私钥、认证 secret 和长期凭据不得进入 verifier、模型权重、客户端输入、日志、仓库或模型 checkpoint。
- 安全承载 verifier 只保存公开验证参数和本地可信 profile 标识。
- A4 toy reference profile 只保存公开矩阵、identity 和公开摘要；没有 key generation 或 signing API。
- V1-P2 的 short module vector `s=(s1,s2)` 只能存在于独立非生产 experiment fixture；verifier、模型、
  registry、日志、manifest 和请求均只接触公开 `Abar,t` 与绑定摘要。fixture 通过上下文管理或显式
  `close()` 覆盖并释放其 Python 对象中的 toy seed/secret；这是测试 hygiene，不是生产内存清零。
- toy 参数必须明确标记非生产，并在测试临时目录或内存中按需生成和销毁。
- 测试向量不得复用真实密钥；确定性种子只能用于 toy/reproducibility，不能用于生产密钥生成。
- 密钥轮换、吊销、profile 版本和 capability 签发密钥方案尚未设计，因此当前不提供相关保证。

## Replay and tamper protection

需要新鲜性的消息至少绑定：协议版本、模型 ID、身份、scope、时间、nonce、业务输入摘要，以及阶段 B 的工具、资源和参数摘要。A3-v1 已把前述 Stage A 字段固定为 133 字节唯一编码；该规格本身尚不提供 proof 不可伪造性。

- 修改任一绑定字段必须使验证失败；
- nonce 必须具有明确作用域、到期时间和原子 consume 语义；
- 并发重复提交只能有一个成功提交权限；
- A3-v1 在原子 consume 的线性化点使用可信 monotonic deadline，`now == deadline` 时已经过期；
- nonce 状态读取、binding 或原子 consume 不确定时必须拒绝且不得调用 protected model；
- 完全无状态的前馈网络不能独立阻止 replay；可信外部状态是允许且必要的协议组件；
- 固定 A0 token 可 replay 是已知限制，不能被描述为已解决。

## Fail-closed behavior

以下情况在权限提交或受保护调用前出现时统一进入 pre-commit 拒绝路径：解析失败、验证异常、未知
profile、算法不匹配、边界模糊、过期、replay、tamper、策略缺失、capability 不匹配和内部错误。

pre-commit 拒绝路径必须：

- 不调用受保护业务模型或工具；
- 不返回 logits、中间特征、部分结果或可复用 capability；
- 不进行受保护写入、网络调用或其他副作用；
- 返回稳定的固定 envelope；
- 仅在服务端产生不含秘密的结构化审计事件。

权限已提交后的模型/工具异常仍返回固定 deny 且禁止 fallback 或二次提交，但必须准确记录已经发生的
受保护调用；不能把 post-commit 失败错误描述为零调用。A3 nonce 一旦 consume 也不得因该失败回滚。

如果原型受框架限制而仍计算受保护业务网络，只能声明“未释放受保护输出”，不能声明“未使用受保护模型能力”，并必须把该差异记录为残余风险。

显式启用 tiered capability 后，public capability 是受信 public entry 的独立结果，而不是上述错误
的弱回退。protected entry 的任何失败仍固定 deny 并产生零 public/protected-model 调用；public
entry 的任何输入或配置失败同样 deny，且不得调用 protected path。public、protected 与 deny
必须具有互斥、稳定且可测试的 version-2 envelope，响应本身不具有授权能力。

V1-M1-C2 沿用该三态语义作为二专家硬路由：`PUBLIC` 只调用 E0，`PROTECTED` 只调用 E1，
`DENY` 不调用二者。V1-M2 在此基础上增加多个受保护 experts，但只能消费协调器已提交的
`allowed_mask`；mask 为空、unknown expert、scope confusion、router tie/exception 或授权集合不足
均必须在任何受保护 expert forward 前 fail closed，且不能自动降级到 E0。

## Audit and observability

审计事件应使用稳定 schema，至少包含事件版本、时间、请求关联 ID、本地 profile、阶段、结构化结果码和 capability ID 摘要。不得记录私钥、完整凭据、原始签名、敏感业务输入、模型中间特征或可重放 token。

外部响应不得暴露详细密码失败原因。内部审计可以区分解析、验证、replay、策略和网关拒绝，但结果码必须稳定且可测试。

## Required security tests

- 合法、非法、边界、模糊区和畸形编码；
- chosen `A`、零矩阵、chosen `b` 和参数降级；
- NaN、Inf、类型混淆、错误 shape、溢出和非规范模表示；
- 输入、身份、scope、模型、工具和参数篡改；
- replay、过期、并发 nonce 复用和重复 capability consume；
- 验证失败时受保护模型和受保护工具调用计数均为零；
- public capability 不能调用 protected model/head、泄露 protected 输出或升级/重标记/复用为
  protected capability，protected 验证失败也不能 fallback 到 public；
- A4 exact proof 编码、message/salt/vector tamper、超界 signed-int8、退化公开矩阵、client key/profile
  注入以及 A3 invalid-proof retry/atomic consume；
- A4-C1 topology、全部 signed-int8 norm 标量域、完整 residual point-pulse 标量域、canonical
  `(y,z)` exact/neural differential、无 reference fallback 以及 A3 neural adapter 单次 consume；
- V1-P2 exact/A3-v2 已覆盖 ring coefficient order、negacyclic wraparound、commit-first 次序、
  challenge weight、abort、coefficient norm boundary、一个终态 response attempt、route confusion、
  并发 replay、内部错误和拒绝零 protected calls；`V1-P2-PSR-E1` 进一步覆盖三个 SHAKE256 role/
  counter stream、byte rejection 的精确有限域计数、112 challenges、generated target、commit-first
  single attempt、5/6/7 emit 边界、13-to-13 引理、exact differential、公开 manifest 和临时 secret
  生命周期；retry harness 已覆盖 fresh nonce/transcript/mask/commitment/challenge、expiry、exhaustion、
  replay/concurrency 和零 protected/verifier calls；
- `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md` 进一步冻结非生产 `s`/`y` domain、无偏 bounded-uniform
  sampler 语义、domain-separated toy seed、`u=Abar*y`、`z=y+c*s`、emit/abort、fresh retry 和
  generated-key fixture 边界。该契约要求 secret/mask 只存在于临时 prover/harness 生命周期，并以
  理论 emit probability、observed abort rate、exact false-reject 和 retry exhaustion 分开报告；这些
  统计不构成 M-LWE/M-SIS、HVZK、Fiat--Shamir 或不可伪造性证明；
- V1-C1 已覆盖固定 `56 -> 11056 -> 17 -> 1` graph/range ledger、独立 valid/tamper differential、
  foreign-route/type-confusion reject、无 reference fallback，以及 A3-v2 neural accept 的单次
  protected call 和 relation reject/foreign route 的零 protected calls；
- Router 绕过、自然语言伪造授权和直接工具调用；
- 量化、导出、剪枝或微调后重新执行全部验证性质测试。

随机伪造测试没有发现假接受，只能作为实验结果，不能证明不可伪造性。

## Explicitly unsupported guarantees

当前不保证：

- A0 LWE unlock 的身份认证、不可伪造性或 replay 防护；
- 任何静态 Secret Trigger、密码字符串或 bearer-gate 对照的身份认证、不可伪造性、anti-replay 或
  与数字签名等价的安全性；
- `A4-GPV-PFDH-TOY-v1` 的不可伪造性、生产安全参数或 GPV random-oracle 具体实例化证明；
- V1-P2 格身份协议的 M-LWE/M-SIS concrete security、知识可靠性、主动冒充安全或授权安全；当前只
  实现非生产 exact/neural/A3-v2 conformance 与 toy generated-key/sampler/single-attempt/retry experiment；
  尚无生产 prover、密码安全参数或上述安全证明；
- 任何自定义签名方案的密码安全；
- 完整 ML-DSA 兼容性或安全性；
- 白盒模型保密、权重抗提取或本地持有者不可绕过；
- 对模型删层、修改代码、任意微调、蒸馏或宿主机控制的抵抗；
- TEE、安全启动、远程证明或供应链完整性；
- 完整 timing、cache、功耗或其他侧信道防护；
- 拒绝服务、资源耗尽或分布式一致性；
- 生产部署、合规认证或第三方安全审计结论。

## Toy and experimental components

toy LWE/Module-SIS conformance 参数、确定性测试密钥、小输入域穷举、MNIST/Fashion-MNIST、
MLP/LeNet、单机实验和模拟工具网关都属于非生产研究组件。论文和代码必须在首次出现处标注其限制，
不得用“secure”“verified”或“signature”掩盖尚未证明的性质。
