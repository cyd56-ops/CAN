# A1 Numerical and Operator Contract

## 1. Status and claim boundary

本文档固定阶段 A1 的构造无关数值与算子契约，版本号为 `A1-v1`。它承接
`docs/A0_PROTOCOL_SPEC.md` 的精确 relation，规定未来固定神经/量化实现必须保持的输入边界、
中间语义、误差上界、判定规则和证据边界。

本文档自身不是神经验证器实现，也不选择具体激活函数、网络深度、量化后端或形式验证工具。
后续 `docs/A1_CONSTRUCTION_DECISION.md` 已选择 `CAN-RELU-EXACT-v1`，`src/can/verifier/`
已实现 dependency-free 和 A1-B1 PyTorch CPU exact-integer backends 并通过完整 toy 域测试。
当前事实只支持指定 Linux x86_64/CPython 3.11/PyTorch `2.13.0+cpu` 的整数 graph 数值结论，
不支持其他 backend、密码 soundness 或安全访问控制主张。

A1 继续使用 A0 的非生产 toy relation。它不提供身份认证、不可伪造性、请求绑定、replay
防护或白盒安全，也不得被称为数字签名验证。

## 2. Normative terms and trust boundary

本文档中的“必须”“拒绝”和“不得”是 A1 实现的规范要求。“候选”表示尚未进入当前正确性或
安全主张的后续构造。

A1 的完整边界为：

```text
untrusted business input x       untrusted credential bytes
          |                                  |
          |                         exact A0-v1 parser
          |                                  |
          |                         local registry lookup
          |                                  |
          |                         canonical b tensor
          |                                  |
          |                    fixed numerical verifier core
          |                                  |
          |                         evidence without authority
          |                                  |
          +------------------ single coordinator ------------------+
                                                                    |
                                                    committed deny or capability
```

以下操作位于数值 verifier core 之外：

- 23 字节 credential 的规范解析、类型和长度检查；
- `profile_id`/`slot_id` 的本地 registry 查询和已编译 profile 选择；
- 业务输入 `x` 的解析、预处理和业务模型调用；
- evidence 到 allow/deny 或 capability 的提交；
- 审计、响应 envelope、nonce 和 replay 状态。

以下语义必须由候选 verifier core 实现并逐算子披露：

- 可信固定参数参与的仿射/残差计算；
- 规范模距离；
- 八个逐分量阈值判定；
- 八路合取。

候选实现必须说明每个语义算子由哪些部署原语承载，以及这些原语为何属于论文声明的“神经
验证器”。框架外 `%`、Floor、比较或条件分支不能在未披露时被包装成神经网络主张。

## 3. Canonical input and separation contract

只有 A0 parser 与本地 registry 均成功后，才允许构造 A1 credential tensor。规范输入为：

```text
b = (b_0, ..., b_7) in {0, ..., 256}^8
```

初始 `A1-INT32-S1` 共同 profile 固定以下 API 边界：

- tensor shape 必须精确为 `(8,)`，不接受隐式 batch、广播或 reshape；
- storage dtype 必须为有符号 `int32`，semantic scale 必须为 `1`；
- 每个元素必须是规范整数 `0 <= b_i < 257`；
- `bool`、浮点、字符串、对象 tensor、NaN、Inf 和隐式可转换值均不接受；
- shape、dtype、scale 或范围不匹配时，在数值 core 运行前 fail closed；
- 请求方不能直接调用只接受已规范 tensor 的内部 core。

业务输入 `x` 和 `b` 必须使用不同 schema、不同 tensor 和不同调用参数。不得把 credential
拼接到图像、embedding、prompt 或其他业务特征中，也不得从业务输入中的隐藏模式推导
credential、profile 或 gate。业务训练数据和梯度不得进入固定 verifier 参数。

如果以后支持 batch，必须使用独立规格定义每个元素的 profile 绑定、拒绝原子性和输出顺序；
`A1-v1` 不接受 batch。

## 4. Trusted profile compilation

每个启用的 `(profile_id, slot_id)` 只能由本地可信编译器生成一个不可变 compiled profile。
编译器以经过 A0 registry 校验的 `A_slot` 和进程内 toy `s_test` 为输入，使用精确 `int64`
中间量计算规范相位锚点：

```text
t_i = mod_q(sum_{j=0..31}(A_slot[i,j] * s_test[j]))
t in {0, ..., 256}^8
```

compiled profile 至少固定：

| Field | A1-v1 value or constraint |
| --- | --- |
| specification version | `A1-v1` |
| source protocol | `A0-v1` |
| `profile_id` | `1` |
| `slot_id` | one exact enabled local `uint32` slot |
| component count | `8` |
| modulus `q` | `257` |
| target center `h` | `128` |
| issuer radius | `4` |
| neural threshold | `8` |
| reference radius | `12` |
| target distance error | at most `4` per component |
| canonical anchor `t` | shape `(8,)`, `int32`, scale `1`, range `[0,256]` |
| input profile | shape `(8,)`, `int32`, scale `1`, range `[0,256]` |

编译必须满足以下条件：

- 所有常量只能从本地 A0 profile 和 registry 得到，不能被请求字段覆盖；
- 编译时矩阵乘加使用精确 `int64`，不得依赖溢出、饱和或浮点舍入；
- 生成后参数不可训练、不可由业务 optimizer 更新，也不能在请求之间变化；
- 加载、量化、导出或设备迁移后必须重新验证 profile 版本、shape、dtype、scale、范围和内容
  完整性；不匹配直接拒绝启用该 profile；
- 未知、禁用或证明材料缺失的 profile 不得回退到其他 slot、较宽阈值或 reference oracle。

规范锚点 `t` 是所有候选构造必须匹配的证明接口，不强制部署实现一定把 `t` 直接存为权重。
候选可以显式计算 `A_slot * s_test`，也可以常量折叠为 `t`，但必须证明其输出与上述锚点逐分量
一致。该选择留待后续构造决定。

`s_test`、包含它的权重以及足以直接构造接受 credential 的 `t` 都按 toy secret-bearing
artifact 管理：只能存在于测试内存或测试临时目录，不得提交、记录或分发。该策略不把 A0
提升为安全认证方案；能够读取 verifier 参数的白盒持有者仍可恢复或绕过 relation。

## 5. Semantic tensors and exact ranges

下表固定共同语义，不要求候选采用同名物理 tensor。候选若融合算子或改变物理布局，必须给出
到这些语义值的逐输入等价映射。

| Symbol | Meaning | Shape | Semantic dtype | Scale | Exact reachable range |
| --- | --- | --- | --- | --- | --- |
| `b` | canonical credential coefficients | `(8,)` | `int32` | `1` | `[0,256]` |
| `t` | trusted canonical phase anchors | `(8,)` | `int32` | `1` | `[0,256]` |
| `u` | affine residual `b-t` | `(8,)` | `int32` | `1` | `[-256,256]` |
| `k` | Euclidean quotient `floor(u/257)` | `(8,)` | `int32` | `1` | `{-1,0}` |
| `p` | canonical phase `u-257k` | `(8,)` | `int32` | `1` | `[0,256]` |
| `c` | centered target residual `p-128` | `(8,)` | `int32` | `1` | `[-128,128]` |
| `d` | exact distance `abs(c)` | `(8,)` | `int32` | `1` | `[0,128]` |
| `d_hat` | candidate distance estimate | `(8,)` | signed fixed point | declared | finite, candidate-certified |
| `g` | component pass bits | `(8,)` | `int32` | `1` | `{0,1}` |
| `v` | eight-way conjunction | scalar | `int32` | `1` | `{0,1}` |

`int32` 是共同 API 与精确工作语义；可信 profile 编译的矩阵乘加使用 `int64`。候选内部采用
更窄 storage dtype、非单位 scale、融合 tensor 或其他量化表示时，必须额外给出每个物理
tensor 的 shape、storage dtype、零点、scale、实值解释、可达整数范围和累加器范围。未声明
或不能证明无溢出的内部格式不符合 A1-v1。

## 6. Exact operator semantics

对每个分量 `i`，共同流水线定义如下：

```text
u_i = int32(b_i) - int32(t_i)
k_i = floor(u_i / 257)
p_i = u_i - 257 * k_i
c_i = p_i - 128
d_i = abs(c_i)
g_i = 1 iff d_hat_i <= 8, else 0
v = 1 iff every g_i = 1, else 0
```

这里的 `floor` 是向负无穷取整的欧几里得商，不是向零截断。由于 `u_i` 的精确范围是
`[-256,256]`，`k_i` 只能是 `-1` 或 `0`，并且 `p_i` 唯一落在 `[0,256]`。

该语义与 A0 oracle 一致，因为：

```text
t_i = mod_q(<A_slot[i], s_test>)
mod_q(b_i - <A_slot[i], s_test>) = mod_q(b_i - t_i) = p_i
d_i = abs(center_q(p_i - 128)) = abs(p_i - 128)
```

最后一个等号成立是因为 `p_i - 128` 已位于 `[-128,128]`。实现仍必须按规范证明该范围，
不能以删除中心化步骤为由改变其他 profile 的语义。

`g` 和 `v` 是精确离散值。Sigmoid 概率、非整数软 gate、把业务 logits 乘以浮点 gate、
或者“多数分量通过”都不等价于上述阈值和八路 AND。任何分量失败都必须令 `v=0`。

## 7. Rounding, saturation and overflow

共同 `A1-INT32-S1` 语义不执行重标定，因此 `b`、`t`、`u`、`k`、`p`、`c`、`d`、`g`
和 `v` 都没有舍入误差。除法只用于定义精确欧几里得商。

候选若引入定点重标定，必须把 scale 写入本地可信 profile，并使用确定、跨后端可复现的
round-to-nearest, ties-to-even。对实数 `y`，它选择距离 `y` 最近的整数；恰好位于两个整数
中点时选择偶数。候选必须用整数或有理数公式定义该操作，不能依赖未固定的语言或设备默认值。

以下行为不允许进入接受路径：

- 有符号整数 wraparound；
- 未声明的饱和、clamp 或截断；
- 向零除法替代负数 floor；
- 浮点 NaN/Inf 传播；
- backend-dependent rounding；
- 用更宽的数值容差补偿转换误差。

候选必须在启用 compiled profile 前证明每个可达累加器都位于 storage dtype 范围内。量化、
导出或设备转换若改变该证明前提，profile 必须拒绝加载。运行时检测到算术异常时只产生内部
配置拒绝 evidence，不执行 reference fallback，也不调用受保护模型。

## 8. Error model and propagation budget

令共同精确阶段为 `y_j = f_j(y_{j-1})`，候选阶段为
`y_hat_j = f_hat_j(y_hat_{j-1})`。对连续或分段内 Lipschitz 的阶段，可使用：

```text
E_0 = input representation error
E_j <= epsilon_j + L_j * E_{j-1}
```

其中 `epsilon_j` 是候选算子相对精确算子的局部最坏误差，`L_j` 是精确算子在已证明可达
区域内的敏感度上界。每个 `epsilon_j`、`L_j` 和可达区域都必须有推导或机器可检查证书，
不能只由随机样本估计。

规范模约减在跨越 residue 分支时不具有可直接用于全域证明的普通实数 Lipschitz 上界。
候选不得把一个未经分支证明的 `L=1` 穿过 modulo 边界。它必须选择以下至少一种方法：

1. 证明商/分支 `k` 对全部可达输入均与精确语义一致，再在每个分支内传播误差；
2. 对有限的全部 `u_i in {-256, ..., 256}` 直接证明组合 modular-distance 输出；
3. 使用区间分析、SMT 或其他形式方法直接证明组合输出误差，而不跨不连续点套用局部界。

A1-v1 最终需要逐分量组合证书：

```text
for every canonical b and enabled compiled profile:
    abs(d_hat_i - d_i) <= 4  for every i in {0, ..., 7}
```

如果候选把 affine、modulo、中心化、绝对值进一步拆层，必须记录各层误差及其传播；如果候选
融合这些算子，则必须记录融合层相对精确 `d_i` 的直接最坏误差。两种形式都必须覆盖完整可达
域，且最终上界不得超过 `4`。

阈值和 AND 必须对其离散输入精确执行，预算为零。不得把阈值或 AND 的额外近似误差隐藏在
`epsilon_target=4` 中。

## 9. Acceptance regions and proof obligations

候选的唯一数值接受规则为：

```text
V_nn(b, compiled_profile) = 1 iff every d_hat_i <= 8
```

任何 `d_hat_i > 8` 都拒绝。内部诊断可以区分 `9..16` 的证明 margin 带和更远的拒绝值，
但该区别不能返回给请求方，也不能触发较弱验证路线。

### 9.1 Total input soundness

令 `E_api` 为 verifier 入口能够收到的全部运行时对象。定义 `canonical(a)` 仅在对象满足第 3 节
的精确 shape、dtype、scale、范围并解析到唯一启用 compiled profile 时存在。完整适配器必须
满足：

```text
for every a in E_api:
    V_nn_total(a) = 1
    -> canonical(a) exists
       and V_ref(canonical(a)) = 1
```

非规范对象在 core 前返回 `0`。外部请求方不能绕过适配器直接调用只接受规范 `b` 的 core。

### 9.2 Issuer-core completeness

对 A0 诚实 issuer 生成的 credential，每个 `d_i <= 4`。若逐分量误差证书成立，则：

```text
d_hat_i <= d_i + 4 <= 8
```

因此八个分量全部通过。该 completeness 只覆盖 A0 `ISSUER_CORE`，不要求
`REFERENCE_GUARD` 中的每个输入都被神经实现接受。

### 9.3 One-sided soundness preservation

若神经实现接受，则每个 `d_hat_i <= 8`。由误差证书：

```text
d_i <= d_hat_i + 4 <= 12
```

所以 A0 reference 的八个分量都位于接受半径内：

```text
V_nn(b, compiled_profile) = 1 -> V_ref(canonical(b)) = 1
```

这只证明神经实现没有扩大 A0 relation 的接受集合，不证明 A0 credential 不可伪造。特别是，
它不解决 adaptive chosen-`b`、replay、输入替换或白盒读取。

### 9.4 Boundary consequences

- `d_i <= 4` 必须接受；
- `5 <= d_i <= 12` 可以接受或保守拒绝；
- `d_i >= 13` 必须拒绝，因为误差下界给出 `d_hat_i >= 9`；
- 任何证明未覆盖、profile 未启用或转换后证书失效的输入必须拒绝。

## 10. Evidence and authorization boundary

verifier 只产生不可变、无授权能力的内部 evidence。A1 实现可以定义稳定的代码，例如
`INPUT_REJECT`、`PROFILE_REJECT`、`NUMERIC_REJECT` 和 `NUMERIC_ACCEPT`，但必须满足：

- acceptance 只能由规范输入、启用 compiled profile 和第 9 节数值规则推导；
- evidence 不包含 gate、allow/deny 提交、authorization context 或 capability；
- 请求方不能提交或反序列化一个 evidence 作为 verifier 输入；
- 精确距离、`d_hat`、分支、profile 常量和误差 trace 只允许出现在测试或不含秘密的内部实验
  trace 中；
- 所有外部拒绝使用同一固定 envelope；
- 只有后续唯一协调器可以把接受 evidence 提交为一次模型能力；
- `V_nn` 失败、异常或不支持时不得调用 `V_ref`，更不得使用 `V_nn OR V_ref`。

当前 checkpoint 没有实现协调器，因此本文档只固定接口义务，不能声称已经证明拒绝路径对业务
模型具有零调用。

## 11. Required conformance and differential tests

未来 A1 实现至少必须包含以下确定性测试。

### Input and profile boundary

- 只接受 shape `(8,)`、`int32`、scale `1` 和逐元素 `[0,256]`；
- 拒绝隐式 batch、广播、错误 dtype、bool、浮点、非有限值、越界值和直接内部 core 调用；
- 未知/禁用 profile、slot 错配、阈值/scale/`q` 被篡改和证书缺失均无回退；
- 编译锚点与 A0 `int64` 矩阵乘加逐分量一致，compiled profile 冻结且不进入业务 optimizer；
- credential tensor 与业务 tensor 不能互换、拼接或共享解析入口。

### Exhaustive semantic boundary

- 对每个 `u in {-256, ..., 256}` 验证 quotient、canonical phase 和 exact distance；
- 覆盖每个 exact distance `0..128`、阈值 `4/5/8/9/12/13` 和两侧模 wrap；
- 对每个启用 slot、每个分量和每个 canonical `b_i in {0, ..., 256}` 比较候选与精确语义；
- 证明逐分量映射与 exact AND 的可分离组合覆盖全部 `257^8` 规范向量，而不是把有限随机向量
  当作全域覆盖；
- 若采用非单位 scale，穷尽或形式验证全部舍入中点和饱和边界。

### Differential and security behavior

- 复用 A0 的 issuer-core、reference-guard、first-reject、bit-zero、wrap、mixed-component、
  malformed 和 chosen-parameter 向量族；
- 分别报告 false reject 与 false accept；任一 false accept 都违反目标 soundness theorem；
- 对所有 issuer-core 向量要求零 false reject；
- 用 instrumentation 证明运行时不调用 `V_ref` 作为 fallback；
- 在后续协调器存在后，用 mock/counter 证明所有拒绝产生零 protected-model 调用；
- 量化、导出、设备迁移、剪枝或微调后重新执行完整性质测试，证书不再成立时拒绝加载。

确定性实验必须记录非秘密 seed、compiled profile 标识、候选构造版本、storage dtype、scale、
rounding、目标设备和准确命令。随机测试只能补充，不能替代上述有限域覆盖或证明。

## 12. Freeze, training and artifact rules

- verifier 参数和 compiled profile 在业务训练前生成并冻结；
- optimizer 参数组必须显式排除 verifier；加载训练 checkpoint 后重新检查内容完整性；
- 不支持的微调、剪枝、量化或导出默认使证明证书失效；
- 不把 `s_test`、`t`、含 secret-bearing 常量的权重、credential 集合、模型 checkpoint 或原始
  实验 dump 提交到仓库；
- 可复现测试在测试框架临时目录或内存中从显式非秘密 seed 重建 toy fixture，并在测试后清理；
- 日志不得包含可重放 credential、完整 profile 常量或逐请求数值 trace。

## 13. Deferred construction hypotheses

下列问题不由 A1-v1 共同规格本身裁决；后续 `docs/A1_CONSTRUCTION_DECISION.md` 已固定首个
构造的处置。表中状态是当前动态结论，不改变本规格的构造无关语义：

| Hypothesis or choice | Current status | Required decision evidence |
| --- | --- | --- |
| “Secret Trigger” 术语 | not used in A1-C1 | 主路线只使用显式 credential，避免隐藏业务模式 |
| 浅层/深层能力映射 | deferred to A2 | 独立 capability、输出隔离和零 protected-path 调用实验 |
| LWE/SIS 的神经兼容性 | deferred | 与其他关系的算子、证明和复杂度比较 |
| 2--3 层规模 | resolved for A1-C1 | 三个 affine+ReLU blocks，完整报告深度/宽度计数 |
| 少量 ReLU modulo/sawtooth | bounded exact ReLU selected | 五个 ReLU/分量覆盖全部 `u`；通用 sawtooth 仍延期 |
| Floor/取模/比较算子边界 | comparison baseline only | 不计入主神经 core，也不能作为 runtime fallback |
| Sigmoid 容错路线 | rejected from A1-C1 main path | 可作后续浮点对照，不能用平均准确率替代全域界 |
| MASK/层内零化 | deferred to A2 comparison | 只能作为输出遮蔽对照，不替代协调器前置硬门控 |
| 显式 `A*s` 与常量折叠 | folded `t` selected | 显式矩阵算术只作 compiler audit，不进入 runtime core |
| 区间、SMT 或手工证明 | hand proof plus exhaustive domain | SMT/形式证书可作为后续附加证据 |
| 首个 PyTorch backend | CPU exact-integer implemented in A1-B1 | `int32` storage、`int64` reduction、scale `1`，185 项完整测试通过 |
| qint8/CUDA/export | deferred after A1-B1 | 新 backend ID、完整物理语义、误差证书和全域差分 |

候选之间不得通过 OR 或弱回退组合扩大接受集合。任何候选只有在满足本规格的输入、误差、
evidence 和全域证明义务后，才能进入主实验路线。

## 14. Acceptance criteria for this specification

A1 数值/算子规格在以下条件全部成立时完成：

1. credential 与业务输入具有独立 schema、tensor 和调用边界；
2. 所有安全参数和 compiled profile 只由本地可信配置生成并冻结；
3. 共同 tensor 的 shape、dtype、scale、范围、舍入和溢出语义完整固定；
4. affine、规范模距离、逐分量阈值、八路 AND 和 evidence 语义可直接指导实现；
5. modulo 不连续边界具有单独的分支证明或全域组合证明要求；
6. 逐分量总距离误差目标固定为 `<=4`，阈值和 AND 不引入额外近似误差；
7. issuer-core completeness 和 `V_nn=1 -> V_ref=1` 从 A0 常量直接推出；
8. 全部非规范 API 输入、证明缺口和转换失效都 fail closed；
9. 差分测试矩阵能区分实验一致性、有限域覆盖和形式证明；
10. 延期假设没有进入当前正确性、安全性或复杂度主张。

`CAN-RELU-EXACT-v1` conformance backend 已实现本规格的输入、compiled profile、误差、
evidence 和差分测试边界；A1-B1 已实现并验证首个 PyTorch CPU exact-integer 目标，但业务模型
零调用和性能实验仍属于后续 checkpoint。任何后端转换都必须重新证明或穷尽验证本规格，不能
继承未经检查的结论。
