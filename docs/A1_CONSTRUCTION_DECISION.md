# A1 Fixed ReLU Construction Decision

## 1. Status and scope

本文档固定 A1 的首个可执行神经构造，决定编号为 `A1-C1`，候选标识为
`CAN-RELU-EXACT-v1`。它实现 `docs/A1_NUMERICAL_SPEC.md` 定义的 A0-v1 toy modular-distance
relation，但不改变 A0 的 wire encoding、可信 registry、接受半径或安全声明边界。

本决定选择构造、证明方法和实现验收边界。`src/can/verifier/a1.py` 已实现 dependency-free
exact-integer conformance backend，unit/differential/security tests 已覆盖完整标量 toy 域、
reference 差分和 no-fallback 边界。`docs/A1_BACKEND_DECISION.md` 已选择 CPU-only PyTorch
exact-integer 首个部署路线，但当前仍未安装或实现该 backend，未实现业务模型或协调器，也不
产生可分发权重；现有结论不能外推到尚未验证的部署 backend。

`CAN-RELU-EXACT-v1` 是非生产、单 profile、固定小参数研究构造。它不提供身份认证、不可伪造性、
replay 防护、请求绑定或白盒安全，不得称为签名验证器。

## 2. Decision summary

A1-C1 作出以下决定：

1. 主候选使用固定整数 affine 与 ReLU，直接实现有界残差域上的精确 modular distance；
2. verifier core 不使用 `%`、Floor、`abs`、比较、Sigmoid、动态分支或业务输出 MASK；
3. 本地编译器把固定 `A_slot*s_test` 折叠为规范锚点 `t`，再折叠进第一层 bias；
4. 每个分量使用 5 个 ReLU 计算精确距离、2 个 ReLU 计算精确整数阈值；
5. 八个分量并行，并用 1 个 ReLU 实现精确八路 AND；
6. 网络拓扑固定为 `8 -> 40 -> 16 -> 1`，共 3 个 affine+ReLU 层和 57 个 ReLU；
7. weight、bias、activation 和 affine 结果的共同语义均为有符号 `int32`、scale `1`，无舍入；
   具体 backend 可以显式 widening 到更宽精确 accumulator，但必须证明结果不变并转回 `int32`；
8. 正确性使用分段恒等式手工证明，并在实现中穷尽全部 513 个残差值和 reference 差分；
9. 普通整数 Floor/modulo/compare 流水线只作为非神经对照基线，不得成为运行时 fallback；
10. 主候选的逐分量数值误差目标为 `0`，从而严格满足 A1 要求的 `<=4`。

该决定选择的是固定有限域关系的精确 CPWL 构造，不声称解决通用或任意模数的 neural modulo。

## 3. Allowed operator and claim boundary

令 `rho(z) = max(0,z)`。主候选 core 只允许：

- 固定 shape 的有符号整数 affine map；
- 固定逐元素 ReLU `rho`；
- 编译期确定的 reshape/layout，不改变数值或请求选择；
- 层间精确 `int32` 值传递。

下列操作属于可信适配器或编译器，不计入神经 core：

- A0 23 字节 parser 与精确类型、长度、范围检查；
- 本地 `profile_id`/`slot_id` registry 查询；
- 使用 `int64` 计算和校验规范锚点 `t`；
- compiled profile 完整性、版本和 shape 检查；
- 把最终精确标量 `0`/`1` 转换为无授权能力 evidence；
- 后续协调器的 allow/deny 提交和业务模型调用。

论文和实现可以把主候选称为“固定整数 ReLU verifier network”，因为 relation 的 residual、
modular distance、阈值和 AND 均由固定 affine/ReLU graph 计算。不得把 parser、registry、
evidence 或协调器描述成神经层，也不得暗示 raw credential parsing 位于网络内部。

主候选 core 禁止：

- `%`、Floor、除法、显式 modulo 或 `abs`；
- `<`、`<=`、`==` 等关系比较参与 relation；
- 依赖浮点、Sigmoid、softmax 或概率阈值；
- 输入相关控制流、动态 shape、广播或隐式类型转换；
- `V_ref` 调用或任何 OR/fallback 路线；
- 对业务 logits、feature 或 protected-model 输出做 MASK。

## 4. Trusted compilation choice

本地编译器继续按 A1 数值规格计算：

```text
t_i = mod_q(sum_j(A_slot[i,j] * s_test[j]))
```

编译使用精确 `int64`，输出八个 `int32`、scale `1`、范围 `[0,256]` 的规范锚点。主候选不把
`t` 作为请求输入，而是把它折叠到第一层的固定 bias。

选择常量折叠而不是在 runtime graph 中显式保留 `A*s`，理由是：

- A0 的 `A_slot` 和 `s_test` 在每个 compiled profile 中均固定，`A*s` 不依赖 credential；
- 保留常量子图不会增加输入相关 relation，只增加算子、artifact 和导出优化差异；
- 编译器仍必须以 A0 reference 语义重算并逐分量验证 `t`，因此折叠不会改变 relation；
- 后续公钥关系若包含消息或签名相关矩阵运算，不能套用本决定自动折叠；A1-C1 只适用于 A0 toy
  profile。

该选择收缩论文主张：A1-C1 证明固定 toy relation 可由小型 ReLU graph 精确执行，不声称已经
实现通用 LWE 解密电路或安全承载格签名验证。

`t` 足以帮助白盒持有者构造接受 credential，因此 compiled bias 与 `t` 一样按 toy
secret-bearing artifact 管理，不得提交、记录或分发。常量折叠不改善 A0 的密码安全性。

## 5. Primary network construction

### 5.1 Input and residual

对每个分量，令：

```text
u_i = b_i - t_i
```

由 A1 规范，`b_i,t_i in [0,256]`，所以 `u_i in [-256,256]`。`u_i` 是推导中的语义值；
主网络不需要单独物理 materialize 它。

### 5.2 Layer 1: exact bounded modular distance hinges

每个分量使用五个 ReLU：

```text
h_i,0 = rho(-u_i)
h_i,1 = rho(u_i + 129)
h_i,2 = rho(u_i + 1)
h_i,3 = rho(u_i)
h_i,4 = rho(u_i - 128)
```

精确距离的 affine trace 为：

```text
d_i = -129
      + h_i,0
      + 2*h_i,1
      - h_i,2
      - 2*h_i,3
      + 2*h_i,4
```

部署时第一层直接以 `b_i` 为输入，其五个 pre-activation 为：

```text
-b_i + t_i
 b_i + (129 - t_i)
 b_i + (1 - t_i)
 b_i - t_i
 b_i + (-128 - t_i)
```

因此 `t_i` 只进入可信固定 bias。八个分量并行，Layer 1 为 `8 -> 40` block-sparse affine
加 ReLU，不存在跨分量连接。

### 5.3 Layer 2: exact integer threshold

对整数 `d_i in [0,128]`，定义：

```text
y_i,9 = rho(9 - d_i)
y_i,8 = rho(8 - d_i)
g_i   = y_i,9 - y_i,8
```

若 `d_i <= 8`，两项之差精确为 `1`；若 `d_i >= 9`，两项都为 `0`。所以
`g_i in {0,1}` 且精确实现 inclusive threshold `d_i <= 8`。

实现不必 materialize `d_i`。把第 5.2 节的 affine trace 代入后，Layer 2 的两个
pre-activation 直接为：

```text
9 - d_i = 138 - h_i,0 - 2*h_i,1 + h_i,2 + 2*h_i,3 - 2*h_i,4
8 - d_i = 137 - h_i,0 - 2*h_i,1 + h_i,2 + 2*h_i,3 - 2*h_i,4
```

八个分量并行，Layer 2 为 `40 -> 16` block-sparse affine 加 ReLU。`g_i` 是下一层 affine
中的成对差，不需要额外激活层。

### 5.4 Layer 3: exact eight-way conjunction

因为每个 `g_i` 精确位于 `{0,1}`，八路 AND 可写为：

```text
v = rho(sum_i(g_i) - 7)
```

只有八个分量全部通过时，和为 `8` 且 `v=1`；其余情况下和至多为 `7` 且 `v=0`。把
`g_i=y_i,9-y_i,8` 代入，Layer 3 是 `16 -> 1` affine 加 ReLU，bias 为 `-7`。

最终 `v` 是精确 `int32` 标量 `0` 或 `1`。可信适配器只把该标量转换为 evidence，不执行
额外关系判断或较弱验证。

## 6. Formal correctness argument

### 6.1 Modular-distance identity

主候选距离函数为：

```text
D_relu(u) = -129
            + rho(-u)
            + 2*rho(u+129)
            - rho(u+1)
            - 2*rho(u)
            + 2*rho(u-128)
```

在规范域 `u in [-256,256]` 上分段化简为：

| Interval | `D_relu(u)` | Exact A0 modular distance |
| --- | --- | --- |
| `[-256,-129]` | `-u-129` | `abs((u+257)-128)` |
| `[-129,-1]` | `u+129` | `abs((u+257)-128)` |
| `[-1,0]` | `128` | 两端整数输入均为 `128` |
| `[0,128]` | `128-u` | `abs(u-128)` |
| `[128,256]` | `u-128` | `abs(u-128)` |

各重叠端点给出相同值。对负整数 `u`，规范 residue 是 `u+257`；对非负整数 `u`，规范
residue 是 `u`。因此对全部 513 个可达整数：

```text
D_relu(u) = abs(mod_q(u) - 128) = d
```

区间 `[-1,0]` 的实数插值只是构造所需的连续连接；A1 输入只包含其两个整数端点，不据此
扩张输入域或安全主张。

### 6.2 Threshold identity

对全部整数 `d in [0,128]`：

```text
rho(9-d) - rho(8-d) = 1  iff d <= 8
rho(9-d) - rho(8-d) = 0  iff d >= 9
```

该恒等式依赖整数、scale `1` 和精确 ReLU。对任意实数输入不作相同二值声明，外部适配器也
不得接受浮点 credential tensor。

### 6.3 AND identity

对 `g_i in {0,1}`，`sum_i(g_i)` 只可能位于 `{0,...,8}`，所以：

```text
rho(sum_i(g_i)-7) = 1  iff every g_i=1
rho(sum_i(g_i)-7) = 0  otherwise
```

### 6.4 Error and security consequences

在精确整数 affine/ReLU 语义下，每层 local error 和 total error 均为 `0`：

```text
for every canonical b:
    d_hat_i = d_i
    epsilon_total = 0 <= 4
```

因此：

- 所有 `ISSUER_CORE` 输入完整接受；
- `REFERENCE_GUARD` 中 `d_i<=8` 的输入接受，`d_i>=9` 的输入保守拒绝；
- 所有 `d_i>=13` 的 reference reject 输入拒绝；
- 实际上主候选接受集合是 reference 接受集合的严格子集或相等子集，不会扩大；
- `V_nn=1 -> V_ref=1` 直接成立。

该结论以输入 guard、compiled profile 完整性和 exact-int backend 语义为前提。浮点重写、非单位
scale、饱和、overflow、近似 ReLU 或改变 bias 都使本证明失效并必须拒绝加载。

## 7. Numeric profile and range ledger

所有主候选 weight、bias、activation 和 affine 结果的共同语义使用有符号 `int32`、scale `1`、
zero-point `0`，且可达 accumulator 也全部落入 `int32`。编译器计算 `t` 时继续使用 `int64`。
主图无重标定、无舍入、无 upper clamp、无饱和；`rho(z)=max(0,z)` 自身的 lower clamp 就是
规范 ReLU，不计作额外数值饱和。

`docs/A1_BACKEND_DECISION.md` 选择 weight/bias/activation 的物理 storage 为 `int32`，product
为已证明安全的 `int32`，row reduction 与 pre-activation 为显式 `int64`，ReLU 后再精确转回
`int32`。这是不改变共同语义的 accumulator widening，不产生误差或新的接受值。

### 7.1 Layer 1 ranges

| Unit per component | Pre-activation range | ReLU output range | Bias range over `t` |
| --- | --- | --- | --- |
| `rho(-u)` | `[-256,256]` | `[0,256]` | `t in [0,256]` |
| `rho(u+129)` | `[-127,385]` | `[0,385]` | `129-t in [-127,129]` |
| `rho(u+1)` | `[-255,257]` | `[0,257]` | `1-t in [-255,1]` |
| `rho(u)` | `[-256,256]` | `[0,256]` | `-t in [-256,0]` |
| `rho(u-128)` | `[-384,128]` | `[0,128]` | `-128-t in [-384,-128]` |

每个 Layer 1 neuron 只有一个非零 input weight，取值为 `-1` 或 `1`。

### 7.2 Layer 2 ranges

| Semantic value | Exact range |
| --- | --- |
| distance trace `d` | `[0,128]` |
| `9-d` pre-activation | `[-119,9]` |
| `8-d` pre-activation | `[-120,8]` |
| `y_9` | `[0,9]` |
| `y_8` | `[0,8]` |
| paired difference `g` | `{0,1}` |

若 affine backend 以任意固定次序累加 Layer 2 的独立项，不利用项间相关性的保守 accumulator
包络也只在 `[-1145,907]` 内。

### 7.3 Layer 3 ranges

Layer 3 的精确 semantic pre-activation 为 `[-7,1]`，ReLU 输出为 `{0,1}`。忽略
`y_9/y_8` 成对相关性时，任意固定累加次序的保守 accumulator 包络为 `[-71,65]`。

所有范围远离 `int32` 边界。实现仍必须对生成的每个 compiled profile 重新验证矩阵、bias、
shape 和范围，不能仅依赖本文档常量。

## 8. Depth, width and operation accounting

主候选采用以下统一计数：

| Metric | Value | Counting rule |
| --- | --- | --- |
| input width | `8` | one canonical coefficient per component |
| affine depth | `3` | `8->40`, `40->16`, `16->1` |
| ReLU depth | `3` | each affine output immediately applies ReLU |
| hidden affine+ReLU blocks | `2` | final scalar ReLU is output block, not hidden block |
| ReLU units | `57` | `40+16+1` |
| dense weights | `976` | `8*40 + 40*16 + 16*1` |
| bias entries | `57` | one per ReLU unit |
| dense scalar parameters | `1033` | dense weights plus biases |
| nonzero weights | `136` | `40 + 16*5 + 16` |
| sparse scalar parameters | `193` | nonzero weights plus all biases |

“3 层”在本项目中必须写成“3 个 affine+ReLU blocks”；如果某框架把输出 activation 排除在
layer count 外，必须同时报告 affine depth、ReLU depth 和完整拓扑，不能只写模糊的“2--3 层”。

参数全部固定且不可训练。dense 和 block-sparse 实现必须生成相同输出；性能报告需要分别说明
逻辑非零算子数和 backend 实际执行的 dense 算子数。

## 9. Comparison baseline and no-fallback rule

对照基线标识为 `A1-EXACT-OPS-v1`，按普通整数程序执行：

```text
u = b - t
k = floor(u/257)
p = u - 257*k
d = abs(p-128)
g = (d <= 8)
v = all(g)
```

它用于：

- 生成逐层预期值和测试 trace；
- 区分“普通算子包装”与 affine/ReLU graph 的算子、延迟和可审计性；
- 与 `V_ref` 进行三方差分，定位 compiler、network 或 adapter 错误。

它不属于神经 verifier，不进入 protected-model 请求路径，不产生可提交 capability，也不能在主
候选异常或拒绝时接管请求。实验代码必须把 baseline 与主候选作为互斥入口；禁止
`CAN-RELU-EXACT-v1 OR A1-EXACT-OPS-v1` 和 `V_nn OR V_ref`。

`V_ref` 仍是 A0 relation 的唯一精确事实源；对照基线只是 A1 语义测试工具，不创建第三个安全
判定主体。

## 10. Candidate disposition

| Candidate or hypothesis | Decision | Reason and boundary |
| --- | --- | --- |
| bounded CPWL ReLU modular distance | primary | 五个 ReLU 在全部 `u in [-256,256]` 上精确，error `0` |
| generic ReLU sawtooth modulo | not selected | 当前单周期紧域无需更深通用 sawtooth；以后扩模数时重新评估 |
| Floor/`%`/`abs`/compare | comparison baseline only | 精确但属于普通整数程序，不支持主神经算子声明 |
| Sigmoid threshold | rejected from main path | 浮点近似、非二值输出并需要额外比较，当前没有全域必要性 |
| MASK or internal zeroing | deferred A2 comparison only | 可能仍计算 protected model，不能替代协调器前置硬门控 |
| explicit runtime `A*s` | compiler audit comparison only | A0 中 `A,s` 固定，runtime 常量子图不增加输入相关 relation |
| folded anchor `t` in bias | primary | 最小图且与 A0 exact compiler interface 逐分量等价 |
| interval/SMT proof | optional secondary evidence | 首构造已有短分段证明和完整有限域穷举；后续可增加机器证书 |
| “Secret Trigger” terminology | not used in main route | 主输入是显式 credential，避免与隐藏业务模式或后门混淆 |
| shallow/deep capability mapping | deferred to A2 | 不属于 A1 数值 relation 或 verifier graph |
| LWE/SIS general compatibility | remains open | A1-C1 只证明固定 A0 toy relation，不足以比较安全承载关系 |

Sigmoid、MASK 和 exact-ops baseline 即使后续用于性能对照，也不能与主候选做 OR 组合、扩大
接受集合或改变固定外部 deny envelope。

## 11. Implemented conformance backend contract

初始实现位于 `src/can/verifier/a1.py`。该 dependency-free exact-integer conformance backend
执行显式固定 affine/ReLU graph，不调用普通 modulo/compare baseline，并因所有可达值位于
`int32` 范围内而与 A1-C1 整数语义等价。它不承担硬件性能或已部署量化结论；A1-B1 已固定
PyTorch CPU exact-integer 路线和官方 CPU wheel，但该目标 backend 仍未安装或实现。

当前实现保持以下边界：

- 不可变 `A1CompiledProfile`：候选 ID、A0 profile/slot、固定三层 weights/biases、数值 profile；
- 本地 compiler：从已校验 A0 slot 与 toy secret 构建 `t` 和固定 graph，并执行范围检查；
- 私有 core evaluator：只接收规范八分量输入与不可变 compiled profile；
- 公共 adapter：复用 A0 parser，执行无回退 profile lookup，只返回不可变 A1 evidence；
- 测试专用 trace：可观察 Layer 1、distance、threshold 和 AND，但不进入外部 evidence；
- exact-ops baseline：只能位于测试/实验边界，主 adapter 不导入或调用它。

所有新公开 API 具有类型标注和简洁中文 docstring。实现没有引入保存 profile、secret、evidence
或授权状态的全局可变对象，也不写出 compiled artifact。

初始实现不加入业务输入、协调器、LeNet/MLP、public capability、nonce 或工具网关。这些边界在
A1 verifier 正确性闭合后分别进入 A2/A3/阶段 B。

## 12. Required proof and test artifacts

conformance backend 当前提供以下证据，任何目标 backend 仍必须重新提供。

### Compiler and structure

- 从 A0 `int64` 语义重算 `t`，并验证所有第一层 bias；
- compiled profile、weight 和 bias 不可变，错误候选 ID/shape/dtype/range fail closed；
- graph 拓扑严格为 `8->40->16->1`，每层零/非零连接符合本决定；
- dense 与 block-sparse evaluator 若同时存在，逐输入结果一致；
- optimizer/training API 不接收 verifier 参数。

### Exhaustive arithmetic

- 穷尽 `u=-256..256`，逐值证明 ReLU distance 与 exact modular distance 相等；
- 穷尽 `d=0..128`，逐值证明 ReLU threshold 输出精确为 `0/1`；
- 穷尽通过分量数 `0..8`，证明最终 AND；
- 穷尽每个启用 slot、分量和 `b_i=0..256`；
- 单独覆盖 `u=-256,-129,-1,0,128,256` 和 `d=4,5,8,9,12,13`。

### Differential and security boundary

- 复用全部 A0 正向、guard、reject、wrap、bit-zero、mixed-component 和畸形向量；
- 对 canonical 输入三方比较 `V_ref`、exact-ops baseline 和 ReLU graph 的距离/判定；
- false accept 必须为 `0`，issuer-core false reject 必须为 `0`；
- 对 `REFERENCE_GUARD` 明确记录阈值 9--12 的预期保守拒绝，不误报为回归；
- instrumentation 证明主 adapter 不调用 `V_ref` 或 exact-ops baseline；
- 请求方不能提交 `t`、weights、bias、threshold、scale、candidate ID、evidence 或 gate；
- parse/profile/config/core 异常都只产生无授权能力的稳定拒绝 evidence。

对应测试位于 `tests/unit/test_a1_verifier.py`、`tests/differential/test_a1_differential.py` 和
`tests/security/test_a1_verifier_security.py`。每个目标 backend 后续都必须重新运行全部性质
测试。浮点、量化、导出或设备转换若不能保持精确整数 affine/ReLU 语义，不得复用 A1-C1 的
error `0` 证明。

## 13. Security implications and residual risks

A1-C1 提供的新增结论仅是：在固定 A0 toy profile、规范输入和精确整数 affine/ReLU 语义下，
一个三层固定网络可以精确实现更窄阈值 relation，因此不会因数值近似产生假接受。

它不提供：

- A0 credential 的不可伪造性、身份认证、消息绑定或 replay 防护；
- 对 adaptive chosen-`b` 查询或公开/泄露 `t` 的抵抗；
- 白盒权重保密、删层防护、模型完整性或运行时不可绕过；
- public-key lattice signature、ML-DSA 或生产 LWE 安全级别；
- protected-model 零调用保证；协调器和业务模型尚未实现；
- 尚未实现的 PyTorch CPU backend、CUDA、qint8 或模型导出的一致性结论；
- 新颖性结论；通用 ReLU 密码编译已有先例，系统 related-work 检索仍未完成。

常量折叠还带来明确论文风险：如果主张写得过宽，审稿人可以正确指出它只是固定有限关系的
小型 CPWL 编译。因此论文必须同时披露固定 profile、黑盒假设、toy 参数和安全不保证，并把创新
评估聚焦于全域数值 soundness 与 fail-closed 访问控制组合，而不是“ReLU 能表达 modulo”。

## 14. Acceptance criteria for this decision

A1 构造决定在以下条件全部成立时完成：

1. 主候选、对照基线和候选标识唯一固定；
2. 神经 core 与 parser、registry、evidence、协调器边界明确；
3. 主 core 只使用固定整数 affine/ReLU，不使用普通 modulo、比较或 fallback；
4. 三层拓扑、weights/bias 公式、range ledger 和参数计数可直接指导实现；
5. modular distance、整数阈值和八路 AND 具有覆盖完整有限域的恒等式证明；
6. 主候选逐分量 error 为 `0`，满足 A1 `<=4` 契约和单向 soundness；
7. 常量折叠 `t` 的研究主张收缩、artifact 风险和 compiler 等价义务已经记录；
8. ReLU/sawtooth、Floor/modulo/compare、Sigmoid、MASK 和显式 `A*s` 均有明确处置；
9. 下一实现 checkpoint 的 API、测试、无回退和非目标边界明确；
10. 研究、安全、工作日志和治理检查与本决定一致。

本文档对应的 dependency-free conformance backend 与完整 toy 域测试已经实现；A1-B1 也已按
相同 graph、range 和 no-fallback 性质实现 PyTorch CPU exact-integer 映射并重新执行完整性质
测试。这仍不等于业务门控、认证或生产安全已经完成。
