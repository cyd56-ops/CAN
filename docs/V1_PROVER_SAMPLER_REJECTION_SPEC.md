# V1-P2 Non-Production Prover, Sampler and Rejection Experiment Specification

## 1. Status and claim boundary

本文档冻结 V1-P2 的独立非生产 prover/sampler/rejection 实验契约，checkpoint 标识为
`V1-P2-PSR-E1`。它位于 V1-prep 的 A3/A4 代数内核之后，只为交互式 Module-SIS conformance
实验定义临时 secret、mask、challenge、response、abort 和 retry 的可复现语义。

本 checkpoint 不实现生产 key generation、密码库 adapter、签名 API、Fiat--Shamir、ML-DSA 或
神经 verifier。它不选择安全参数，不证明 M-LWE/M-SIS concrete security、HVZK、special soundness、
主动冒充安全或不可伪造性。实验结果只能支持 toy distribution、completeness、abort-rate 和
implementation conformance 结论。

V0/A0、V1-P1、A3-v1、A4-C1 和已经闭合的 V1-P2 exact/A3-v2 接受路径保持独立且不改写。

当前实现状态：`src/can/experiments/v1_psr.py` 已闭合第 17 节的步骤 1--3，包括临时 generated-key
fixture、三个 SHAKE256 sampler、112-challenge set、commit-first single attempt、emit/abort、A3-v2
fresh-transcript retry/exhaustion、分阶段 latency、公开 manifest 和 exact differential。

## 2. Normative terms and trust boundary

以下术语具有规范含义：

- `MUST` 表示实验实现必须满足的契约；`MUST NOT` 表示禁止行为。
- `s` 是只存在于临时 prover 实验上下文的 toy short secret；verifier、registry、日志和模型不得持有它。
- `y` 是每次 transcript 新生成的 bounded-uniform mask；它不得跨 transcript、challenge 或 retry 复用。
- `c` 由本地可信协调器从固定 challenge set 选择，prover 不选择或覆盖它。
- `z = y + c*s` 使用未约减的整数/negacyclic convolution；只有 exact verifier 负责模 `q` relation。
- `emit` 表示 response 通过 rejection 条件并可提交；`abort` 表示该 transcript 不产生 response。

随机源、profile、challenge set、retry budget 和时钟均由可信实验 harness 或本地配置提供。请求方
不能选择算法、参数、seed、profile、secret、mask、challenge 或 rejection policy。

## 3. Fixed toy profile and generated-key separation

本 checkpoint 使用既有非生产参数：

```text
protocol_id = CAN-V1-FSWA-MSIS-ID-v1
N = 8
q = 257
k_mod = 2
ell_mod = 2
eta = 1
gamma = 8
kappa = 2
B = gamma - kappa*eta = 6
```

每个 module polynomial 有 8 个系数，`s` 和 `y` 各有 `ell_mod + k_mod = 4` 个 polynomial，
因此每个 mask/secret 有 32 个系数。`s` 的 toy domain 是逐系数 `[-eta, eta] = {-1,0,1}`；
`y` 的 toy domain 是逐系数 `[-gamma, gamma] = [-8,8]`。

`src/can/reference/v1.py` 中的固定 `V1_CONFORMANCE_TARGET` 是公开 arithmetic fixture，不是
由 secret 生成的 public key。prover 实验 MUST 使用临时 generated-key fixture：给定公开 matrix
和临时 `s`，在内存中计算 `t = Abar*s mod (q, X^N+1)`，再构造只含公开 `t` 的 verifier profile。
不得把 secret、seed 或 generated-key target 写回仓库、日志、checkpoint 或分发 artifact。

## 4. Key relation and temporary secret lifecycle

对 `s=(s_1,s_2)`，实验使用：

```text
t = A*s_1 + s_2 mod (q, X^N+1)
```

或等价地 `t = Abar*s`。`Abar`、`t` 可以复制到公开 verifier fixture；`s` 只允许存在于 prover
调用栈和测试 harness 的短生命周期对象中。

每次实验 run MUST：

1. 从显式 toy seed 生成一次临时 `s`；
2. 计算并校验 `t`，随后只把 public profile 交给 verifier；
3. 在每次 transcript 结束时丢弃 `y`、`z` 和 transcript-local intermediate；
4. 在 run 结束时释放 secret reference，并确认日志、序列化对象和异常文本不包含 `s`。

测试可以使用 Python 临时目录保存非敏感 vector manifest，但不得保存 secret、mask、response collection
或含 secret 的模型 artifact。该生命周期是 toy hygiene，不是生产内存清零保证。

## 5. Canonical deterministic seed contract

实验 seed 必须是恰好 32 字节，由 harness 显式提供。seed 只用于 toy reproducibility，不能解释为
生产密钥随机性。所有派生流都使用 domain-separated labels：

```text
CAN-V1-PSR-SECRET-v1\0
CAN-V1-PSR-MASK-v1\0
CAN-V1-PSR-CHALLENGE-v1\0
```

每个 role 的 byte stream 固定为以下 block 串接，整数使用 unsigned big-endian：

```text
block_j = SHAKE256(
  role_domain || seed:32 || trial_index:u64 || retry_index:u64 || j:u32
).digest(64)
stream = block_0 || block_1 || ...
```

secret 固定使用 `trial_index=0,retry_index=0`；mask 和 server challenge 使用各自 role domain 以及
当前 `(trial_index,retry_index)`。`trial_index` 标识一次独立 end-to-end trial，`retry_index` 从 0
开始标识该 trial 内的 fresh transcript。服务端 challenge 派生流只存在于可信 harness/coordinator，
不得通过 prover API 暴露。

同一 seed、同一 public matrix、同一 profile 和同一 retry policy MUST 产生相同的 secret、challenge
序列、mask retry 序列、emit/abort 序列和摘要。不同 role 或 counter tuple MUST 使用不同的 stream；
不得把一个随机流的剩余字节同时解释为 secret 和 mask。本 SHAKE256 stream 只定义 toy reproducibility，
不是生产 DRBG，也不允许把实验 seed 传给 verifier 或客户端。

## 6. Exact sampler semantics

### Secret sampler

`sample_secret(seed)` 对 32 个位置独立、均匀采样 `{-1,0,1}`，并按 polynomial-major、power-ascending
顺序组装为 4 个 polynomial。逐 byte 消费 secret stream：只接受 `b<255`，再映射为
`(b mod 3)-1`。`b=255` 丢弃并继续；不得用浮点、truthy 转换或直接对所有 byte 取余。

### Mask sampler

`sample_mask(seed, trial_index, retry_index)` 对 32 个位置独立、均匀采样 `[-8,8]`。逐 byte 消费
mask stream：只接受 `b<255`，映射为 `(b mod 17)-8`；`b=255` 丢弃。mask 每次 retry 都必须重新
采样，不能通过修改上一次 `y` 或复用同一 `y` 变更 challenge。

### Challenge sampler

挑战集合固定为：8 个系数取 `{-1,0,1}`，非零系数恰好为 `kappa=2`。集合大小为
`C(8,2)*2^2 = 112`；服务端按本地配置均匀选择，不允许 prover 选择。canonical challenge list 按
`(position_0,position_1,sign_0,sign_1)` 字典序排列，其中 `0<=position_0<position_1<8`，sign 顺序为
`-1,+1`。逐 byte 消费 challenge stream：只接受 `b<224`，选择 list index `b mod 112`；`b>=224`
丢弃。challenge seed 不进入客户端 response。

### No reduction in the prover

`sample_secret`、`sample_mask` 和 `z = y + c*s` 阶段不得对系数做模 `q` reduction、wraparound 或
canonical residue conversion。response 的 signed-i32 wire encoding 只在通过 emit 条件后执行；超出
规范 response range 的对象必须 abort/reject，不得静默截断。

## 7. Commitment and response computation

每个 fresh transcript 的顺序固定为：

```text
s <- sample_secret(run_seed)                         # run-local, not sent
y <- sample_mask(run_seed, trial_index, retry_index)
u = Abar*y mod (q, X^N+1)                  # commitment, public residue
coordinator binds A3-v2 input and samples c
z = y + c*s                                # exact integer negacyclic convolution
```

`u` 必须使用既有 coefficient-domain exact convolution 和 canonical public residue encoding。challenge
只能在 `u` 提交后产生；prover 不得看到未来 challenge，也不得因 challenge 选择而回滚或改写 `u`。

response 的 coefficient order、transcript binding 和 wire encoding 必须复用既有 V1-P2 exact parser；
本 checkpoint 不新增第二种 response 表示。

## 8. Emit and abort rule

定义 `||z||_inf` 为 32 个 response coefficients 的最大绝对值。仅当：

```text
all(abs(z_i_j) <= B=6)
```

时 emit `z`。否则该 transcript MUST 发送显式 abort 或由 prover harness 标记 abort；不得提交一个
截断、取模、饱和或重新编码的 response。

在当前 toy profile 中，`|(c*s)_i| <= kappa*eta = 2`，因此每个 mask coordinate 在最坏 shift 下有
13 个可 emit 值，单个 coefficient 的条件 emit probability 为 `13/17`，32 个独立系数的理论
单次 emit probability 为：

```text
p_emit = (13/17)^32 ~= 0.00018699146739962278
E[attempts] = 1/p_emit ~= 5347.837598722525
```

该高 abort rate 是 toy 参数的预期现象，不是 sampler 自动失败；它也说明该 profile 不适合性能、
DoS 或生产安全结论。实验报告 MUST 同时给出理论值和观测值。

更精确地说，对任意固定 toy `s,c`，每个 shift coefficient `d=(c*s)_i` 都在 `[-2,2]`，映射
`y_i -> z_i=y_i+d` 在 emit 条件下把恰好 13 个 mask 值双射到 `[-6,6]`。因此条件 emitted `z`
在 `[-6,6]^32` 上均匀，且上述 `p_emit` 与固定 `s,c` 无关。这是当前有限 cube 的直接计数引理；
它不自动证明完整交互协议的 HVZK、lossiness、special soundness 或任何 Fiat--Shamir 转换。

## 9. Fresh-transcript retry policy

一次 retry 只能创建全新的 transcript：

```text
new A3 nonce
new transcript_id
new y
new u
new server challenge c
```

旧 transcript 的 `u`、`c`、`y`、`z`、nonce 和 abort 状态不得复用。A3-v2 的单次终态语义保持不变：
abort、expiry 或一个 parsed response attempt 都终结当前 transcript。

`max_attempts` 必须由本地可信配置固定且为正整数。retry exhaustion 返回 deny，并在进入受保护模型
前产生零 protected calls。客户端不能通过 payload 请求更大的 budget、关闭 rejection 或保留旧状态。

## 10. Completeness and distribution obligations

对每个 generated-key fixture 和每个 honest emitted transcript，测试必须验证：

```text
Abar*z = u + c*t mod (q, X^N+1)
||z||_inf <= B
```

因此 exact reference 对 emitted honest response 不得产生 false reject。该结论是代数 completeness，
不等同于 secrecy、knowledge soundness 或 unforgeability。

测试还必须分别报告：

- mask sampler 的 domain/range 与 marginal frequency；
- challenge set 的 uniformity 与 exact weight；
- abort rate 与理论 `p_emit` 的偏差；
- emitted honest response 的 exact verifier false-reject count；
- retry exhaustion count、成功 attempt 分布和端到端 latency。

此外必须在小域或完整当前 domain 上验证上述 fixed-`s,c` 的 13-to-13 translation/truncation 计数，
并把该组合计数与随机频率检验分开记录。

有限样本只能支持实验统计；不得把观察到的 abort/accept 频率写成安全归约。

## 11. Test-vector plan

测试向量使用显式 seed 和临时目录 manifest，不提交 secret 内容。至少包括：

| Family | Required cases | Expected property |
| --- | --- | --- |
| `secret-domain` | all-zero、all-`-1`、all-`1`、mixed boundary | shape、domain、deterministic digest；不出现在 verifier/log |
| `mask-domain` | all `-8`、all `8`、zero、mixed endpoints | exact inclusive `[-8,8]` and unbiased sampler contract |
| `challenge-domain` | all 112 support/sign choices、wrong weight、non-ternary | uniform local set and parser rejection |
| `convolution` | `X^8=-1` wraparound、positive/negative products | `u`/`z` exact coefficient semantics |
| `emit-boundary` | `||z||_inf=5,6,7` and mixed boundary coefficients | 6 emits; 7 aborts; no clipping |
| `freshness` | same mask reused, same `u`, same challenge, new transcript | reuse rejected or absent from accepted path |
| `retry` | forced abort prefix, first-success, exhaustion | fresh transcript per retry; zero protected calls on exhaustion |
| `differential` | emitted responses against `verify_v1_ref` | zero honest false rejects; exact relation preserved |
| `lifecycle` | exception, abort, success and retry cleanup | no secret/mask in evidence, logs or serialized public profile |

每个 vector 还要记录 protocol/profile digest、seed digest、attempt index、challenge digest、emit/abort
结果和 public `u`/`t` digest；不得记录原始 `s`、`y` 或 `z` collection。

## 12. Measurement contract

实验至少运行三个独立 toy seeds；每个 seed 的 retry budget、attempt count 和 challenge policy 必须
固定并记录。报告应给出：

```text
seed_digest, profile_digest, sample_count, max_attempts
emit_count, abort_count, observed_abort_rate
theoretical_emit_probability, observed_emit_probability
mean/median/p95 attempts-to-emit
exact_false_rejects, retry_exhaustions
```

latency 必须拆分 mask sampling、convolution、reject decision、exact verification 和 A3/coordinator
部分；不得把受保护业务模型 latency 混入 sampler 结论。CPU-only 单机结果不能外推到 accelerator、
分布式部署或生产吞吐。

## 13. Security and authorization boundary

prover/sampler 只产生 commitment、response 或 abort；它不产生 evidence、allow、capability 或模型
调用。verifier 只产生既有 immutable evidence；唯一 coordinator 负责 claim transcript、提交授权并
在 exact accept 后调用受保护模型。

解析失败、challenge mismatch、norm failure、equation failure、abort、expiry、replay、retry exhaustion
和内部异常必须在 pre-commit 阶段保持零受保护副作用。consume 后的 protected callback 异常不得回滚
transcript，也不得重用同一 nonce 重试。

## 14. Theorem-condition separation

本 checkpoint 只记录下列尚未满足或尚未证明的条件：

- `q,N,k_mod,ell_mod,eta,gamma,kappa` 不是安全参数，尚未通过 estimator 选择；
- `t=Abar*s` 的 public-key distribution 与 M-LWE 假设尚未做 concrete reduction/parameter check；
- 多 challenge transcript 的 special soundness/M-SIS knowledge extraction 尚未证明；
- bounded-uniform rejection 的隐私、lossiness、HVZK 和 abort-loop 分析尚未在 CAN 参数上复核；
- A3-v2 的 replay/request binding 是协议状态性质，不由 sampler 或 neural arithmetic 提供；
- V1-C1 已对固定 toy profile 的 canonical input 证明 `V_nn==V_ref`；该算术义务仍与 sampler、
  M-LWE/M-SIS 和协议安全主张分离；
- 交互式协议不自动给出 Fiat--Shamir ROM/QROM 或签名不可伪造性。

引用 FSwA-S、Dilithium 或相关分析只能说明协议来源，不表示当前 toy fixture 继承对应安全级别。

## 15. Artifact and logging policy

禁止提交或持久化：toy secret、private key、mask、response collection、random stream state、完整
transcript、模型 checkpoint、数据库和生产凭据。允许保存的 manifest 只能包含公开 profile digest、
seed digest、计数、统计量和测试结果摘要。

异常和审计输出不得包含 secret、mask、response 原文、完整业务输入或可重放 token。任何发现 secret
进入日志或 artifact 的测试必须失败并清理临时目录。

## 16. Deferred and excluded scope

以下内容不属于本 checkpoint：

- 生产 keygen、真实密钥生命周期、审查密码库和安全参数选择；
- Gaussian sampler、标准 DRBG、签名 API、Fiat--Shamir、ML-DSA；
- NTT、qint8/CUDA/export 和性能优化；
- 分布式 durable transcript、TLS、速率限制、DoS 防护和侧信道；
- CIFAR-100/ResNet-18、模型训练和业务 gate；
- 任何把 toy abort rate、成功率或测试伪造结果描述为不可伪造性。

## 17. Implementation checkpoint sequence

本 checkpoint 的后续实现顺序固定为：

1. 已根据本契约生成临时 generated-key fixtures 和 deterministic vector manifest；
2. 已独立实现 toy sampler/single-attempt prover，并与既有 exact reference 做 emitted-response differential；
3. 已接入 A3-v2 fresh-transcript retry，验证 abort、expiry、replay、concurrency 和零 protected calls；
4. 已实现 `V1-C1-MSIS` 的 coefficient-domain neural construction，并覆盖 independent differential、
   no-fallback 和 A3-v2 neural route zero-call 边界；
5. 下一步冻结 V1-M1 GPU/software tuple，随后才运行 CIFAR-100/ResNet-18 baseline；
6. 安全参数、生产 library adapter 和 Fiat--Shamir 另设独立 checkpoint。

## 18. Acceptance criteria

本规格 checkpoint 只有在以下条件全部满足时才算冻结：

1. secret/mask/challenge domain、seed、counter、polynomial order 和 exact arithmetic 已明确；
2. `u=Abar*y`、`z=y+c*s`、emit/abort 和 fresh retry 语义与 V1-P2/A3-v2 一致；
3. 理论 emit probability、toy 高 abort rate 和 retry exhaustion 解释已记录；
4. test-vector families 覆盖边界、负向、差分、生命周期和零副作用要求；
5. M-LWE/M-SIS、rejection、A3、neural 和 Fiat--Shamir 主张保持分离；
6. 没有新增运行时 keygen、生产密码实现或 secret artifact；V1 neural code 仅使用公开不可变 profile；
7. README、SECURITY、RESEARCH_DESIGN、V1-P2 决策和 PROJECT_WORKLOG 对该 checkpoint 一致。
