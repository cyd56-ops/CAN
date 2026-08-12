# Lattice-Based Neural Model Access Control: Research Design

## 1. Document status

本文档定义项目当前认可的研究问题、术语、阶段边界、实验路线和论文主张。项目面向科研论文与可复现实验，不是生产访问控制产品。任何实现和论文表述都不得超过代码、证明和测试已经支持的结论。

项目暂定名称为 **CAN: Lattice-Based Neural Model Access Control**。研究对象是把格密码验证关系编译为模型内部的固定神经计算网络，并用其结果控制 CNN/DNN 能力；后续再将同一信任链扩展到 MoE 或多智能体工具调用。

安全边界和不提供的保证见根目录 `SECURITY.md`，动态进度和唯一下一步见根目录 `PROJECT_WORKLOG.md`。

## 2. Core research question

核心问题是：

> 能否把一个定义明确的格密码验证关系编译为定点或量化神经网络，并证明在全部合法编码和对抗性输入上，数值误差不会造成假接受，同时使该结果能够不可绕过地控制黑盒模型能力？

阶段 A 研究模型内部验证和业务模型门控；阶段 B 研究认证证据、授权决策、短期 capability 和工具网关的组合。

## 3. Claims taxonomy

下列概念必须严格区分：

| Term | Meaning in this project | Claim limit |
| --- | --- | --- |
| Exact oracle | 使用精确整数语义实现的参考关系 `V_ref` | 只用于测试与证明，不承担最终访问控制 |
| Neural verifier | 从固定密码参数编译、训练期间冻结的网络 `V_nn` | 必须说明输入域、数值语义和误差界 |
| LWE unlock | 使用 toy LWE 解密关系恢复授权比特 | 不是数字签名，不自动提供身份或不可伪造性 |
| Authentication evidence | 验证器输出的结构化、无授权能力的证据 | 不能直接执行副作用或铸造 capability |
| Authorization | 本地策略对身份、scope、资源和约束的决定 | 只能由唯一可信协调器提交 |
| Capability | 与主体、模型、工具、参数、期限和 nonce 绑定的短期权限 | 必须由工具网关重新验证 |
| Authentication neuron/layer | 对规范 credential 执行固定验证计算的研究性网络模块 | 是系统组织术语；A0 不据此声称身份认证，单个神经元也不能提交权限 |
| Public/protected capability | 由本地策略定义的公共与受保护模型能力 | 不预设它们必然对应浅层/深层神经表示 |
| Signature verification | 使用公钥验证消息和签名的不可伪造认证关系 | 只有实现并证明对应安全游戏后才能使用该名称 |

A0 的 LWE 解密实验只研究数值正确性和门控语义。它不得在标题、API 或论文结论中被称为“验签”。

## 4. Research hypotheses

- **H1, compiled correctness:** 在显式有限输入域内，固定神经验证网络能够与精确整数 oracle 保持逐输入一致。
- **H2a, parser-boundary soundness:** 任何未通过规范 parser 的运行时输入在进入神经 core 前被拒绝并映射为 `V_nn = 0`；请求方不能绕过 adapter 直接调用只接受规范 `b` 的 core。因此接受集合被约束在规范域 `D` 内。
- **H2b, in-domain soundness preservation:** 对规范域 `D` 内的全部输入，`V_nn(a) = 1` 蕴含 `V_ref(a) = 1`；数值近似不能在 `D` 内扩大接受集合。该性质由逐分量误差证书与全域穷举支持，不覆盖 `D` 之外的非规范表示，也不蕴含 credential 不可伪造性。
- **H3, fail-closed gating:** 验证失败、输入畸形或落入数值模糊区时，不调用受保护业务能力，也不释放 logits 或中间特征。
- **H4, request binding:** 在挑战响应或签名阶段，修改消息、模型、身份、scope、工具参数、时间或 nonce 的任一字段都会导致拒绝。
- **H5, mandatory authorization path:** 在阶段 B，Router、规划专家和业务专家都无法绕过协调器与工具网关自行获得或提交权限。
- **H6, tiered capability composition:** 在二元 protected-model 门控闭合后，协调器可以把拒绝受保护能力与授予独立 public capability 组合，而不泄露受保护路径的 logits、中间特征或副作用。

H1 可以通过证明、穷举 toy 域和差分测试共同支持。H2a-H6 涉及安全或能力隔离性质，不能仅用分类准确率或有限随机测试代替证明。

## 5. Prior work and novelty boundary

当前本地参考资料包括：

- `paper/How to Securely Implement Cryptography in Deep Neural Networks.pdf`：形式化密码功能的 ReLU DNN 实现，展示连续实数输入对自然密码实现的密钥恢复风险，并提出输入净化和输出掩码变换。
- `paper/Planting Undetectable Backdoors in Machine Learning Models.pdf`：使用数字签名验证器构造神经网络中的不可复制触发机制；附录 C 给出基于格签名模方程的感知机/正弦验证网络。

因此，下列内容本身不构成项目创新：

- 神经网络能够表达布尔密码电路；
- 在模型中加入签名验证器并据此选择输出；
- 使用格密码模方程构造神经验证网络；
- 把密码触发条件与原有分类器组合。

计划主张的差异化方向是：

1. 面向定点/量化执行的 LWE 或格验证专用编译，而非仅证明网络可表达；
2. 对全部可表示输入证明接受集合不扩张，而非只报告随机样本一致率；
3. 给出逐层数值误差预算、禁止区和 fail-closed 离散门控；
4. 把验证证据、唯一授权提交点和受保护模型调用组合为明确的访问控制语义；
5. 实验研究量化、剪枝和微调对密码正确性与 soundness 的影响；
6. 在阶段 B 研究神经验证证据与 capability 工具网关的不可绕过组合。

上述新颖性判断目前只基于仓库内两篇论文，尚未完成系统文献检索，不能用于最终论文的 related-work 完整性声明。

## 6. Stage A architecture

### Route mapping: V0, V1-prep, V1 and V2

长期路线与既有 A0--A4 编号并行，不重命名已经实现的模块：

| Route | Existing project boundary | Claim boundary |
| --- | --- | --- |
| V0 | A0 toy LWE relation, A1 exact/neural compiler and A2 fail-closed model gate | 数值解锁、逐输入等价和零受保护调用；不是认证或签名 |
| V1-prep | 已闭合的 A3 request binding/freshness shell 与 A4 canonical `(y,z)` public relation compiler | 隔离并验证格代数神经内核；不提供身份认证或不可伪造性 |
| V1 | 已选择 V1-P2：FSwA-S Module-SIS Sigma protocol；V1-P1 普通矩阵方案保留为 baseline | 协议安全、身份/消息/scope/replay 绑定，以及独立的 polynomial neural soundness |
| V2 | 后续 ML-DSA exact reference and selected neural modules | 标准向量和允许模块的等价性；不是首篇论文 MVP |

V1-P2 在 `R_q=Z_q[X]/(X^N+1)` 上固定 `Abar=[A|I]`、`t=Abar*s`，验证
`Abar*z=u+c*t` 和 response coefficient infinity-norm。其 negacyclic polynomial products 可展开为
固定 affine/convolution layers，但当前 A4-C1 固定 `q=257`、普通 `8x72` matrix 和 signed-int8，不能
直接加载 module profile。V1 已新增 coefficient-domain exact relation、A3-v2 协议壳和
`V1-C1-MSIS` dependency-free neural construction；NTT 只作为后续 backend。神经 verifier 仍只产生 evidence，
唯一 coordinator 才能提交权限。

V0、V1 与 V2 必须作为独立可复现实现共存。V1 不得通过重命名、改写或替换 A0/V0 模块实现，V2
也不得覆盖 V0/V1；每条路线使用独立协议标识、registry、parser、adapter、测试与默认关闭入口。
跨路线只允许复用无协议语义的通用 helper，且不得形成 `V0 OR V1` 或 `V1 OR V2` fallback。

阶段 A 的规范数据流为：

```text
untrusted request (x, credential bytes)
  -> split business input x from credential bytes
  -> canonical credential parser and type/length checks
  -> local trusted profile selection
  -> deterministic neural verifier V_nn
  -> structured evidence only
  -> single access coordinator
  -> committed deny or model capability
  -> invoke protected CNN/DNN only on allow
  -> fixed response envelope
```

业务网络记为 `f_theta`，固定验证网络记为 `V_phi`。概念上的组合模型为：

```text
F(x, a) = f_theta(x)  when the coordinator commits allow
F(x, a) = DENY        otherwise
```

`phi` 由可信本地 profile 编译并冻结；业务训练只能更新 `theta`。验证器产生证据，协调器提交最终 gate。请求方不得提交 gate，也不得指定 `q`、矩阵、算法、阈值或弱化 profile。

credential 分支与业务特征 `x` 必须具有独立 schema、解析和调用边界。初步方案中的 “Secret Trigger” 是否适合作为研究术语留待后续评估；在基础阶段，无论采用何种名称，认证材料都按显式 credential 处理，不能依赖业务输入中的隐藏模式获得权限。

硬门控不能只是先计算 `f_theta(x)` 再用浮点数相乘。原型至少要通过可观测的调用计数证明拒绝路径没有调用受保护业务网络；如果框架限制导致只能遮蔽输出，论文必须把保证收缩为“黑盒输出不释放”，不得声称计算未发生。

能力分级分两步研究。第一步只实现 `DENY`/protected model 的二元门控，闭合验证与零受保护调用基础；第二步才允许协调器根据本地策略选择独立 public capability 或 protected capability。public capability 可以由独立模型、head 或服务入口实现，是否以及如何映射到浅层/深层表示属于后续实验问题。

## 7. Stage A research increments

### A0: Exact relation and toy LWE numeric unlock

精确规格见 `docs/A0_PROTOCOL_SPEC.md`。A0-v1 固定 `n=32`、`m=8`、`q=257`，使用本地 slot registry 解析 `A_slot`，并只接受包含版本、profile、slot 和八个 `b` 分量的 23 字节凭据。参考接受半径为 12，未来神经阈值为 8，目标逐分量总误差上界为 4。

目标是通过精确整数 oracle 和测试向量研究 LWE 判决间隔与神经数值误差的组合，而不是建立密码安全级别。

约束如下：

- 该阶段是非生产数值实验，不是身份认证或数字签名；
- 不允许请求方任意提交 `A`、`q`、噪声分布或阈值；
- 必须明确 token issuer 和请求方各自掌握的信息；
- 必须分析 `A = 0, b = floor(q/2)` 等 chosen-input 直接解锁；
- 必须分析可自适应选择 `b` 时形成的判决/decryption oracle；
- toy secret 只作为临时测试材料生成，不进入仓库、日志或可分发模型 checkpoint；
- A0 的结果只能支持“数值等价”和“黑盒 toy gate”结论。

### A1: Quantized neural verifier

构造无关的共同契约见 `docs/A1_NUMERICAL_SPEC.md`，首个构造决定见
`docs/A1_CONSTRUCTION_DECISION.md`，首个 PyTorch 物理映射见
`docs/A1_BACKEND_DECISION.md`。A1 把 A0 的精确关系编译为固定神经网络；主路线使用纯整数
ReLU 语义，浮点实现只允许作为比较基线。

A1 首先固定与具体网络构造无关的共同验证流水线：

```text
canonical credential tensor
  -> trusted profile compiled and frozen locally
  -> affine/residual stage
  -> canonical modular-distance stage
  -> per-component threshold stage
  -> eight-way conjunction
  -> structured evidence without authority
```

该分解吸收线性映射、模关系、噪声 margin 和逻辑门控的研究方向。规范解析、可信 profile 选择和最终授权仍位于网络边界之外；A1-C1 已选择只使用固定整数 affine/ReLU 的主 core，普通 `%`、Floor、比较和 evidence 组装不计入神经网络。

共同 `A1-INT32-S1` profile 已固定 credential tensor 为 shape `(8,)`、`int32`、scale `1` 和逐元素范围 `[0,256]`。本地编译器使用 A0 精确 `int64` 语义生成规范相位锚点 `t=(A_slot*s_test) mod 257`；A1-C1 把 `t` 折叠到第一层 bias，并在紧残差域 `[-256,256]` 上用五个 ReLU/分量精确计算 modular distance。

A1-C1 已固定：

- 主图拓扑为 `8->40->16->1`，即三个 affine+ReLU blocks 和 57 个 ReLU；
- 五个 ReLU/分量实现 exact distance，两个 ReLU/分量实现整数阈值，一个 ReLU 实现八路 AND；
- 所有主图数值使用 `int32`、scale `1`、zero-point `0`，编译 `t` 使用 `int64`；
- 全域手工分段恒等式加 513 个残差穷举作为首个证明路线，逐分量总误差为 `0`；
- 普通整数 Floor/modulo/compare 只作为互斥对照基线，不能成为 runtime fallback；
- compiled 参数不可训练，转换后必须重新验证完整性质。

通用 ReLU sawtooth、Sigmoid 和显式 runtime `A*s` 未被选为主路线；MASK 只保留为 A2 输出遮蔽
对照。`src/can/verifier/` 已实现 dependency-free 和 A1-B1 PyTorch CPU exact-integer backends，
并以 unit/differential/security tests 穷尽残差、距离、AND 和逐分量 canonical coefficient；该
结论仍不能外推到 qint8、CUDA、export 或其他未验证 backend。

A1-B1 已选择 `CAN-TORCH-CPU-EXACT-v1` 作为首个部署目标：Linux x86_64、CPython 3.11、
官方 PyTorch `2.13.0+cpu` wheel、CPU eager mode、`int32` weight/bias/activation、`int64`
reduction、scale `1` 和 zero-point `0`。它不使用 qint8 quantized tensor、CUDA 或 export，且
任何不支持 exact integer affine/ReLU 的环境都禁用 backend，不得回退。该 backend 已从指定
wheel 在内存中构建 non-persistent buffers，并通过 513 residual、129 distance、9 AND sums、
逐分量全部 canonical coefficient、A0 向量族和 fail-closed artifact tests。该结论只适用于当前
Linux x86_64/CPython 3.11/PyTorch `2.13.0+cpu` toy 环境。

### A2: CNN/DNN integration

A2-E1 已在 `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md` 中唯一选择 Fashion-MNIST 和
`784 -> 256 -> 128 -> 10` float32 MLP，固定 CPU package tuple、数据资源、确定性 split/训练、
准确率/延迟指标、artifact 生命周期和外部响应 envelope。固定依赖、数据资源、严格输入校验、
MLP、两次同种子无门控 baseline、唯一协调器和二元前置硬门控已完成；baseline 和 gate 重训均
得到 `88.08%` test accuracy 和相同预测/模型摘要，10,000 个 gated top-1 标签全部匹配，拒绝
探针和安全测试保持零 protected-model calls。MNIST 与 LeNet 不作为 runtime 备选或回退。

A2 已完成单一 protected model 的二元硬门控，并由
`docs/A2_CAPABILITY_EXPERIMENT_SPEC.md` 固定 A2-E2 public/protected capability 分级实验。
主路线选择独立 `784 -> 64 -> 2` public MLP，把 Fashion-MNIST 输出限制为 footwear/non-footwear
两类；它与 protected MLP 不共享权重、head、feature 或 artifact。public entry 默认关闭，只能由
本地可信部署配置绑定；protected 验证失败仍固定 deny，不能降级到 public。独立 public baseline
现已完成：两次固定十 epoch 训练均得到 `99.85%` coarse test accuracy、相同 prediction/state
digest 和相同 determinism fingerprint。默认关闭的本地 public policy、version-2 envelopes 和单一
三态协调器现已实现，调用矩阵、并发、异常、不可升级和无 fallback 测试通过。D-024 授权的独立
trusted materializer 已按固定协议重建并保存两个 ignored `state_dict`，canonical state digest 精确
匹配已验收值；no-training evaluator 已完成 10,000-image 三态报告，两个预测摘要均与 baseline
一致，拒绝探针和默认关闭探针均为零模型调用。

必须记录：

- 合法请求的分类准确率和延迟；
- 非法请求的拒绝行为和延迟；
- 参数量、内存和计算开销；
- 拒绝路径的受保护业务网络调用计数；
- 输出 envelope 是否泄露 logits、中间特征或细粒度失败原因。

### A3: Challenge-response and request binding

精确协议见 `docs/A3_CHALLENGE_RESPONSE_SPEC.md`。A3-v1 已固定 133 字节 proof message：

```text
domain || version || model_id || identity || scope || issued_at || expires_at || nonce || H(input)
```

`H(input)` 对严格规范的单张 float32 Fashion-MNIST 图像使用固定 big-endian IEEE-754 编码和
SHA-256；challenge 使用可信时钟、60 秒 TTL 和 32 字节服务端 nonce。A4 verifier 只返回绑定精确
message digest 与 identity 的 evidence，唯一协调器随后原子执行 `PENDING -> CONSUMED`；只有该状态
转换的唯一成功者可以调用 protected model。并发 replay、过期、tamper、状态异常和错误 evidence
必须在提交前保持零 protected-model calls。

nonce 已用状态位于模型外部。该状态只保证协议新鲜性，不负责密码验证，因此不改变“验证计算位于
神经网络内部”的研究边界。A3 运行时壳已实现；A4 toy exact relation/adapter 已可在显式本地 profile
下组合，但 A0/A1 numeric evidence 仍不得接入该路径，没有 A4 profile 时入口默认关闭。完全无状态
的前馈网络不得声称能够独立阻止 replay。

### A4 / V1-prep: Public-key lattice relation compilation

安全承载版本应让模型只持有公开验证信息，签名私钥留在模型之外。优先选择已有、经过同行评审的格签名关系作为研究对象；在没有严格安全定义和归约前，不自行声称设计了安全的新签名方案。

首个关系已在 `docs/A4_GPV_RELATION_SPEC.md` 中选择 GPV STOC 2008 的 probabilistic
full-domain-hash 短原像公开验证谓词。非生产 `A4-GPV-PFDH-TOY-v1` 固定 `q=257`、八分量
syndrome、72 分量 signed-int8 向量、`||z||_inf <= 1`、32 字节 salt 和 105 字节唯一 proof；精确
reference 检查 SHAKE256 hash-to-syndrome 与 `A*z mod q` 等式。公开 profile 构造期校验满行秩并
保持不可变，reference/adapter 不包含私钥、trapdoor、signer 或授权能力。

当前参数和测试 gadget 矩阵只用于 conformance，公开可构造 valid proof，不继承 GPV 的
random-oracle/SIS 不可伪造性结论。`docs/A4_NEURAL_CONSTRUCTION_DECISION.md` 已冻结并实现
`CAN-RELU-A4-PFDH-TOY-v1`：`80 -> 3600 -> 1153 -> 1` 三层 exact integer affine/ReLU graph，
用 144 个 norm-violation units、3456 个 residual hinges、1152 个 point pulses 和最终硬 AND 在全部
canonical `(y,z)` 上满足 `V_nn == V_ref`。SHAKE256 仍属于可信 canonical preprocessing。

当前 A4 是 V1-prep 的代数编译基线，不是 V1 身份认证协议。普通矩阵 `V1-P1` 保留在
`docs/V1_PROTOCOL_SELECTION_DECISION.md` 作为历史 baseline；当前主路线 `V1-P2` 已在
`docs/V1_MODULE_SIS_PROTOCOL_DECISION.md` 选择 FSwA-S 的底层交互式 Module-SIS Sigma
protocol。它固定 module ring、commit-first transcript、sparse ternary polynomial challenge、
`z=y+c*s`、bounded-uniform rejection、`Abar*z=u+c*t` 和 coefficient infinity-norm。当前已新增
非生产 exact conformance relation、A3-v2 wrapper 与 V1-C1 neural route；该实现不把 A4 gadget
conformance 或引用论文写成当前认证安全结论。

### V1-P2: Interactive Module-SIS identification

V1-P2 的 prover 持有短 module vector `s=(s1,s2)`，本地 registry 只保存公开 `Abar=[A|I]` 与
`t=Abar*s=A*s1+s2`。prover 先发送 `u=Abar*y`，协调器绑定既有 133-byte A3 message 并从本地
challenge set 采样 fixed-weight ternary polynomial `c`；只有满足 `||z||_inf<=B` 的
`z=y+c*s` 才进入 exact verifier。每个 commitment 只允许一个终态响应尝试，abort、expiry、tamper、
replay 和并发重复提交均不得触发受保护模型。

协议 completeness/HVZK/special soundness、A3 request binding/replay 和未来 neural
`V_nn=1 -> V_ref=1` 是三类独立义务。Fiat--Shamir with aborts 的 ROM/QROM 转换和签名 API 继续
延期，不从交互式选择自动推出。非生产 `N=8,q=257,k_mod=2,ell_mod=2,eta=1,gamma=8,kappa=2,B=6`
profile、coefficient-domain exact reference 与 A3-v2 已实现并通过 canonical、差分、route-isolation、
replay/并发和零 protected-call 测试；非生产 generated-key、SHAKE256 samplers、single-attempt
emit/abort、fresh-transcript retry、exhaustion、公开 manifest 与 exact differential 也已实现。固定
`56 -> 11056 -> 17 -> 1` V1-C1 graph 已用 coefficient residual point pulses、norm violations 和 final
hard AND 对全部 canonical input 证明 `V_nn==V_ref`，并已接入独立 A3-v2 neural evidence route。生产
prover、密码安全参数、NTT、PyTorch/qint8/CUDA/export 和性能结论仍未实现。
M-LWE public-key pseudorandomness、M-SIS knowledge soundness、A3 replay binding 和 neural soundness
必须分别陈述。

`docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md` 冻结了独立的非生产 `V1-P2-PSR-E1` 实验契约：toy
`s`/`y` domain、domain-separated deterministic seed、`u=Abar*y`、`z=y+c*s`、`||z||_inf<=B`
emit/abort、fresh-transcript retry、secret lifecycle、理论 emit probability、测试向量和统计指标。
当前 `N=8,q=257,eta=1,gamma=8,kappa=2` profile 的理论单次 emit probability 约为
`0.00018699146739962278`，期望约 5348 次尝试；这只用于解释 toy abort rate，不是性能或安全参数。
`src/can/experiments/v1_psr.py` 已实现该规格的 generated public profile、三个 domain-separated
SHAKE256 rejection sampler、固定 112-challenge set、commit-first single attempt、emit/abort、A3-v2
fresh-transcript retry/exhaustion harness、分阶段 latency、无 secret manifest 和 emitted-response
exact differential。它不把 bounded-uniform 计数、abort 率、retry 成功率或 emitted-response
completeness 提升为 M-LWE/M-SIS、HVZK、Fiat--Shamir 或不可伪造性结论。

ML-DSA 属于 V2 后期标准比较目标，不作为首个神经验证器。其哈希、编码、NTT、范数和 hint 检查
需要分别处理，且不得通过宽松数值容差改变规范接受集合。V2 的量化或导出只对经过证明和差分测试的
固定转换负责，不承诺任意剪枝、微调或结构修改后的等价性。

## 8. Formal correctness and security obligations

设 `D` 为规范输入域，`E` 为部署数值格式可表示的全部输入，`V_ref` 为精确验证关系，`V_nn` 为神经实现。

### Typed canonicalization

只有通过规范解析的输入才进入 `D`。未知字段、重复字段、错误长度、错误 dtype、非有限值、越界整数和非规范编码直接拒绝。

### Completeness

对具有规定 margin 的合法输入：

```text
for all a in D_valid_margin: V_ref(a) = 1 -> V_nn(a) = 1
```

### Soundness preservation

最低安全目标是单向包含：

```text
for all a in E: V_nn(a) = 1 -> canonical(a) exists and V_ref(canonical(a)) = 1
```

如果无法证明该性质，只能声称实验一致性，不能声称密码 soundness。

### Error budget

对每层误差建立组合界：

```text
epsilon_total = sum(propagated epsilon_i)
```

合法 margin、拒绝 margin 和 `epsilon_total` 的关系必须足以同时支持 completeness 和 soundness。单独证明 `epsilon_total < Delta` 只覆盖远离边界的样本，不自动约束攻击者构造的边界输入。

### Zero protected side effects

解析失败、验证失败、replay、tamper、过期、scope 提升和 capability 不匹配必须产生零受保护副作用。测试必须通过 mock/counter 验证业务模型或工具没有被调用。

## 9. Stage B architecture

阶段 B 不把生成式专家当成信任根。强制数据流为：

```text
untrusted request
  -> canonical parser
  -> deterministic authentication verifier
  -> authentication evidence
  -> deterministic/local authorization policy
  -> single authorization coordinator
  -> short-lived capability
  -> router, planner and business expert
  -> tool gateway revalidates capability against actual arguments
  -> protected tool side effect
  -> structured audit result
```

认证专家、授权策略、规划专家和业务专家可以在系统组织上称为专家，但只有确定性验证器、协调器和工具网关处于权限提交链。Router 或自然语言输出不能构造 `allow`、capability 或安全上下文。

capability 至少绑定：主体、模型、工具、资源、实际参数或参数哈希、scope、到期时间、nonce、最大调用次数和请求摘要。

## 10. Experiment and test matrix

### Differential correctness

- 大规模确定性种子的合法、非法和边界向量；
- toy 小参数域的可行穷举；
- `V_ref` 与 `V_nn` 逐样本比较；
- 中心化模数边界、最大噪声、最小噪声和舍入边界；
- 每个 dtype、量化 scale 和目标设备分别验证。

### Adversarial validation

- 任意或零矩阵、chosen `b`、错误 `q` 和弱化 profile；
- 未知/重复字段、错误 shape、错误长度、bool/int 混淆；
- NaN、Inf、负零、溢出、非规范模表示和非法编码；
- 输入、身份、model ID、scope、工具参数和 nonce 篡改；
- replay、并发 nonce 复用、过期和跨模型复用；
- 软门控泄漏、直接调用业务网络和 Router 绕过；
- 量化、剪枝、微调和模型导出后的性质退化。

### Construction and capability comparisons

- 二元 `DENY`/protected gate 与 public/protected capability 分级；
- A2-E2 主路线的独立 public model；独立 head、共享 trunk 与浅层/深层只保留为后续比较；
- `CAN-RELU-EXACT-v1` 与普通整数 exact-ops 基线；
- 主 ReLU 构造与后续可选 Sigmoid、通用 sawtooth 或 MASK 非安全对照；
- 主锚点常量折叠与 compiler audit 中的显式矩阵算术；
- 不同构造的网络深度、宽度、误差界、假接受和受保护调用计数。

数值构造比较在 A1 主实现通过后启动，capability/MASK 比较只在二元 A2 门控通过后启动。候选路线不得通过回退或 OR 组合扩大主安全路径的接受集合。

### Performance

- 业务准确率相对基线变化；
- 合法和拒绝路径的延迟分布；
- 模型大小、内存、算子数和吞吐变化；
- 安全检查与业务计算的独立开销。

任何“未观察到假接受”的实验结论都必须附带样本规模、种子、参数域和置信限制，不能替代不可伪造性或 soundness 证明。

## 11. Planned implementation stack

MVP 采用 Python 3.11（当前验证解释器为 3.11.9），当前 A1-B1 使用 PyTorch `2.13.0+cpu`，
A2-E1 固定使用 torchvision `0.28.0+cpu`、Fashion-MNIST 和 float32 MLP；A2-E2 已实现并复现
独立 two-class public MLP baseline，并已集成默认关闭的本地 public entry policy 与三态协调器；
已验收权重的完整三态报告和本地 state/manifest 生命周期均已核验。`pyproject.toml` 精确锁定直接依赖，
`requirements-dev.lock` 记录开发依赖，`requirements-ml.lock` 记录 Python 3.11.9 下解析出的 ML
环境闭包。A1-B1 和 A2-E1 已安装并核验官方 torch/torchvision CPU wheels；Fashion-MNIST、
license、split、MLP 和两次 baseline 已核验。A3-v1 运行时壳已实现 canonical message/input digest、
可信 nonce 状态、原子 consume 和默认关闭 coordinator；A4 GPV toy exact reference、公开 profile、
proof parser、A3 adapter 和固定 neural verifier 已实现。V1-P2 公开 profile/parser、系数域 exact
reference、A3-v2 单次终态协调器、exact/neural evidence adapters 和 V1-C1 dependency-free graph 也已实现；
CUDA/qint8 路线继续延期。

开发质量工具固定为 pytest 9.1.1、Ruff 0.15.22 和 mypy 2.3.0。当前 `src/can/reference/` 已实现
A0-v1 的规范解析、不可变 registry、精确模运算和 evidence-only `V_ref`，以及 A4 无私钥公开
profile、105 字节 proof parser、hash-to-syndrome 和 exact relation，以及 V1-P2 公开
profile/parser/registry 和 coefficient-domain exact relation；
`src/can/verifier/` 已实现 `CAN-RELU-EXACT-v1` compiled profile、dependency-free evaluator、
A1-B1 PyTorch CPU exact backend、A4-C1/V1-C1 neural verifiers 和 evidence-only adapters，并具有全域
差分与防御性安全测试。
`src/can/model/`、`src/can/access/` 和 `src/can/experiments/` 已实现 A2-E1 MLP、无门控
baseline、单一协调器、二元硬门控和 gate 报告，以及 A2-E2 独立 public MLP/baseline、三态
协调器、trusted materializer、只评估报告入口、A3-v1 默认关闭协议壳、A4 evidence adapter，以及
V1-P2 A3-v2 状态机与 exact/neural adapters；qint8/CUDA/export backend 仍未完成。

实现结构的计划边界为：

```text
src/can/reference/       exact integer oracle and vector generation
src/can/verifier/        compiled fixed neural/quantized verifier
src/can/model/           business MLP or LeNet
src/can/access/          evidence types and single coordinator
src/can/experiments/     reproducible experiment entry points
tests/                   unit, integration, differential and security tests
docs/            protocol, proof obligations and experiment reports
```

这些包目录已经创建；reference、verifier、A2 MLP、baseline、gate experiment 与 A2-E1 单一
协调器已实现。所有后续跨模块授权流仍必须经过唯一协调器和网关。

## 12. Publication strategy

第一篇论文应以理论构造加最小系统验证为主：

- 一个定义明确的格验证关系；
- 一个定点/量化神经编译；
- completeness、soundness-preservation 和误差预算；
- 一个 LeNet/MLP 黑盒访问控制原型；
- 与两篇本地相关工作的直接比较；
- 攻击性边界测试和门控不可绕过实验。

阶段 B 更适合形成后续系统论文或独立扩展。它需要 capability 语义、工具网关、提示注入/Router 绕过实验和审计设计，不能用来掩盖阶段 A 缺少密码证明的问题。

## 13. Explicit non-goals

- 首版不实现完整 ML-DSA 神经验证。
- 不声称 toy LWE 解锁具备数字签名不可伪造性。
- 不以分类准确率代替密码正确性。
- 不声称抵抗白盒读权重、修改推理代码、删层或直接调用中间层。
- 首版不解决 TEE、安全启动、远程证明、完整侧信道或拒绝服务。
- 不实现攻击性后门或可用于未授权访问的利用工具；相关论文仅用于理解并防止绕过。
- 不把研究原型描述为生产安全产品。

## 14. Open decisions

- 独立模型主路线闭合后，是否值得另设 checkpoint 比较独立 head、共享 trunk 或浅层/深层表示；
- LWE/SIS 派生关系相对其他密码关系是否更适合神经编译；
- MASK/层内零化只作为输出遮蔽对照还是能支持更窄的实验主张；其结果不得替代协调器前置硬门控；
- V1-P2 的密码安全参数和 reviewed library adapter；fresh-transcript retry 与 toy 统计 harness 已闭合；
- 已冻结 V1-C1 range ledger 如何扩展到更大参数或 NTT/PyTorch/qint8/CUDA/export backend，同时保持
  独立的全域范围证明与差分；
- A1-B1 CPU exact baseline 闭合后，是否以及如何引入 qint8、CUDA 或 export 第二 backend；
- 是否在手工分段证明与完整有限域穷举之外增加 SMT/形式验证机器证书；
- 黑盒查询次数、输出可见性和时间侧信道在首篇论文中的具体范围；
- 量化、剪枝和微调哪些属于受支持转换；
- 完整 related-work 检索和论文目标 venue。
