# V1 Module-SIS Sigma Protocol Selection Decision

## 1. Status and claim boundary

本文档冻结 CAN 当前 V1 主协议决定，决定编号为 `V1-P2`，协议族标识为
`CAN-V1-FSWA-MSIS-ID-v1`。项目采用 Boudgoust--Takahashi 2023/ESORICS 2023 明确列出的
`FSwA-S` module-lattice Fiat--Shamir-with-aborts signature 的底层交互式
commit--challenge--response protocol。该方案是 Lyubashevsky/GLP 路线的 Module 版本，也可视为
不含 public-key compression、HighBits、hint 和标准 hash/encoding 的 vanilla Dilithium baseline。

V1-P2 首先只实现交互式 identification。它不是 ML-DSA、不是生产认证系统，也不自动继承签名的
ROM/QROM 安全结论。后续 checkpoint 已实现非生产 conformance parameters、exact verifier、
A3-v2 协议壳，以及 `V1-P2-PSR-E1` 的临时 generated-key、确定性 sampler、single-attempt
emit/abort、fresh-transcript retry/exhaustion harness 和公开 manifest；生产 key generation/prover、
密码安全参数、NTT、PyTorch/qint8/CUDA/export 和性能结论仍未实现。`V1-C1-MSIS` 已实现
`CAN-RELU-V1-MSIS-COEFF-v1` dependency-free exact integer graph，并经独立 valid/tamper differential
与 A3-v2 neural evidence route 验收；该 graph 只证明固定 toy profile 的 compiled arithmetic，不提供
身份认证、M-LWE/M-SIS、HVZK、主动冒充、Fiat--Shamir 或不可伪造性结论。

`docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md` 现冻结独立非生产 `V1-P2-PSR-E1` 实验契约。该文档只
规定 toy secret/mask domain、deterministic seed、bounded-uniform emit/abort、fresh-transcript retry、
临时 secret 生命周期和测试向量；当前已实现其中 generated-key/sampler/single-attempt/retry 部分，但没有
改变 exact verifier 或 A3-v2 接受路径，也不使当前 profile 继承 FSwA-S、Dilithium 或
Fiat--Shamir 的安全参数与定理。

V1-P2 位于已闭合的 V1-prep 之后；V1-prep 只提供 A3 请求绑定壳和 A4 toy 公开关系的神经编译
经验，不是身份认证，也不向 V1-P2 提供密码学安全结论或可直接复用的 verifier。

原 `V1-P1` 普通矩阵 SIS 决策保留为历史与实验对照，不再是当前 V1 实现目标。V0、V1-P1、V1-P2
和 V2 均不得通过重命名、改写或覆盖前序路线实现；V1-P2 必须新增独立 module、registry、parser、
adapter、evidence 类型和测试。

## 2. Selected protocol and primary sources

V1-P2 的规范来源是：

1. Katharina Boudgoust and Akira Takahashi, *Sequential Half-Aggregation of Lattice-Based
   Signatures*, ESORICS 2023. Algorithm 1 明确给出 `FSwA-S`：`Abar=[A|I]`、`t=Abar*s`、
   `u=Abar*y`、`z=c*s+y`、rejection 和 `Abar*z=c*t+u`；作者称其为 GLP12 的 module variant 或
   vanilla Dilithium，并说明安全基础为 M-LWE 与 M-SIS：<https://eprint.iacr.org/2023/159.pdf>。
2. CRYSTALS-Dilithium specification v3.1. 该文档定义 `R_q=Z_q[X]/(X^n+1)`、module matrix、
   sparse challenge、uniform masking、rejection sampling、Module-LWE public key 与 Module-SIS
   soundness，并明确完整 Dilithium 另含 HighBits、compression 和 hints：
   <https://pq-crystals.org/dilithium/data/dilithium-specification-round3-20210208.pdf>。
3. Eike Kiltz, Vadim Lyubashevsky and Christian Schaffner, *A Concrete Treatment of Fiat-Shamir
   Signatures in the Quantum Random-Oracle Model*, EUROCRYPT 2018. 该工作把 Dilithium 类方案明确
   建模为 canonical identification scheme，并区分 MLWE、MSIS、lossiness、HVZK 和 QROM 转换：
   <https://www.iacr.org/archive/eurocrypt2018/10822196/10822196.pdf>。
4. Julien Devevey et al., *A Detailed Analysis of Fiat-Shamir with Aborts*, CRYPTO 2023/2024.
   该工作修正 abort-loop 分析中的技术问题，因此 CAN 不从交互式 V1-P2 自动推出非交互签名结论：
   <https://eprint.iacr.org/2023/245.pdf>。

这些来源冻结协议语义和证明边界，不表示当前 toy conformance 参数或未来另选的 CAN 参数继承其
安全级别。

## 3. Candidate disposition

| Candidate | Public verification core | Disposition |
| --- | --- | --- |
| V1-P1 ordinary matrix SIS | `A*r=T*c+a mod q`, squared `L2` bound | retained as historical/simple baseline |
| FSwA-S module variant of GLP12 | `Abar*z=u+c*t` over `R_q`, coefficient infinity norm | selected as V1-P2 |
| Full Dilithium/ML-DSA verifier | HighBits, decomposition, hints, SHAKE and standard encoding | retained as V2 |
| General short-vector ZK proof systems | commitments plus product/range proofs | useful later, but too large for first neural identity relation |

V1-P2 is selected because it preserves one exact public module equation and a bounded coefficient check while
introducing the polynomial/module structure needed to study convolutional neural compilation. It does not require
the V1 verifier to implement ML-DSA hints or approximate equality.

## 4. Roles and trust assumptions

- **Prover:** holds the short module vector `s`, samples mask `y`, computes the response and performs rejection.
- **Verifier service:** holds only public `Abar,t`, the trusted profile and A3-v2 transcript state.
- **Trusted registry:** maps one identity to one enabled V1-P2 profile/public key; the request cannot select the ring,
  ranks, modulus, challenge set, bounds, key or backend.
- **A3/V1 coordinator:** snapshots business input, binds the commitment, samples the challenge after commitment,
  atomically claims one response attempt and is the sole authorization commit point.
- **Neural verifier:** future deterministic evidence producer only; it never owns `s`, transcript state or model
  authority.

## 5. Parameterized module profile

A concrete non-production conformance profile must freeze:

```text
protocol_id = CAN-V1-FSWA-MSIS-ID-v1
profile_id, profile_digest, identity_id
N, q
k_mod, ell_mod
eta, gamma, kappa
B = gamma - kappa*eta
R_q = Z_q[X] / (X^N + 1)
A in R_q^(k_mod x ell_mod)
Abar = [A | I_k] in R_q^(k_mod x (ell_mod + k_mod))
S_eta = coefficient-bounded secret domain
D = U(S_gamma), the exact bounded uniform mask distribution
C = {c in R: coefficients in {-1,0,1}, weight(c)=kappa}
retry and expiry policy
```

`N` must be a power of two. An NTT backend may additionally require prime `q` with `q=1 mod 2N`, but the
canonical exact relation is polynomial multiplication in `R_q`, not an NTT representation. The first reference
implementation must use coefficient-domain negacyclic convolution as the semantic oracle.

`B`, coefficient widths, product bounds, convolution accumulators and modular-reduction ranges are audited exact
integers. A toy conformance profile and any future security profile are separate and cannot share claims by name.

当前非生产 conformance profile 固定为：

```text
N=8, q=257
k_mod=2, ell_mod=2
eta=1, gamma=8, kappa=2
B=gamma-kappa*eta=6
public/commitment coefficient: canonical u32 residue in [0,256]
challenge coefficient: canonical i8 in {-1,0,1}, exact weight 2
response coefficient: canonical signed i32, relation bound [-6,6]
reference accumulator: unbounded exact Python integer before modulo
```

固定公开 matrix/target 与 public-key digest 进入 `src/can/reference/v1.py`。在 relation bound 内，一个
`A*z` polynomial 的单系数绝对值至多 `N*(q-1)*B=12288`，两个 matrix columns 加 identity response
后 LHS 至多 `24582`；`c*t` 单系数至多 `kappa*(q-1)=512`，加 commitment residue 后 RHS 的保守
绝对值至多 `768`。这些界只支持当前 exact conformance oracle；未来固定宽度 neural/backend 必须
重新给出逐层 accumulator ledger。

## 6. Key relation

Write the short secret as one module vector:

```text
s in S_eta^(ell_mod + k_mod)
t = Abar*s mod (q, X^N+1)
pk = (profile_digest, identity_id, t)
sk = s
```

Equivalently, for `s=(s1,s2)`, `t=A*s1+s2`. This makes `(A,t)` an M-LWE-style public key distribution while
the accepting-transcript difference yields an M-SIS-style short relation. The verifier stores no secret polynomial,
mask seed, rejection state or key-generation randomness.

## 7. A3-v2 commit-first transcript

The interactive protocol is:

```text
1. Prover samples y <- U(S_gamma)^(ell_mod+k_mod).
2. Prover computes u = Abar*y in R_q^k_mod and sends canonical commitment bytes with identity and input x.
3. Coordinator validates u, snapshots x and creates the existing canonical 133-byte A3 message m_a3.
4. Coordinator samples c uniformly from the fixed local challenge set C after receiving u.
5. Coordinator stores (profile, m_a3, input snapshot, u, c, transcript_id) as PENDING.
6. Prover computes z = y + c*s using negacyclic polynomial multiplication.
7. Prover emits z only when ||z||_inf <= B; otherwise it aborts, destroys y and starts with fresh state.
8. Coordinator atomically claims one response attempt, invokes the deterministic verifier, commits authorization
   only for exact accept, and then calls the protected model with the stored input snapshot.
```

Commitment `u`, mask `y`, challenge `c`, response `z`, A3 nonce and transcript identifier are single-use.

## 8. Binding and transcript identifiers

The server computes:

```text
commitment_digest = SHA256(commitment_bytes)
binding_digest = SHA256(
  "CAN-V1-MSIS-BIND-v1\0" ||
  profile_digest ||
  m_a3 ||
  commitment_digest
)
transcript_id = SHA256(
  "CAN-V1-MSIS-TRANSCRIPT-v1\0" ||
  binding_digest ||
  challenge_bytes
)
```

The response does not carry an independently trusted challenge, key, ring or parameter selector. All such values
must match the pending local state. The evidence binds identity, profile, A3 message, commitment, challenge and
transcript digests without carrying authority.

## 9. Canonical wire encodings

The concrete profile will use fixed-width big-endian coefficients. For the first conformance profile, require
`q<2^32`, canonical public residues in `[0,q)`, signed response coefficients in exact int32 and challenge
coefficients encoded only as `0xff`, `0x00` or `0x01`.

```text
commitment_bytes =
  b"CAN-V1-MSIS-COM-v1\0" ||
  profile_id:u16 ||
  u[0][0]:u32 || ... || u[k_mod-1][N-1]:u32

challenge_bytes =
  b"CAN-V1-MSIS-CHAL-v1\0" ||
  profile_id:u16 ||
  c[0]:i8 || ... || c[N-1]:i8

response_bytes =
  b"CAN-V1-MSIS-RESP-v1\0" ||
  transcript_id:32-bytes ||
  z[0][0]:i32 || ... || z[ell_mod+k_mod-1][N-1]:i32

abort_bytes =
  b"CAN-V1-MSIS-ABORT-v1\0" ||
  transcript_id:32-bytes
```

Coefficient order is polynomial-vector-major, then ascending power `X^0,...,X^(N-1)`. Unknown/trailing bytes,
wrong exact type/length/domain, non-canonical residues, incorrect challenge weight, bool/int confusion and response
range-certificate failure reject before arithmetic.

## 10. Rejection and abort semantics

With uniform bounded masking, the selected FSwA-S profile uses:

```text
z = y + c*s
emit z iff ||z||_inf <= B
B = gamma - kappa*eta
```

An abort is a normal prover outcome, not verifier false rejection. Abort, expiry and any parsed response attempt are
terminal for that transcript. A new attempt requires a fresh A3 nonce, `y`, commitment and transcript identifier.
The server never changes `c` for an existing commitment and never offers multiple verification attempts for one
transcript.

## 11. Exact verification relation

After canonical parsing and pending-state matching, the exact verifier accepts iff:

```text
all centered coefficients of z satisfy abs(z_i_j) <= B
and
Abar*z mod (q, X^N+1) == u + c*t mod (q, X^N+1)
```

Correctness follows from:

```text
Abar*z = Abar*(y+c*s) = Abar*y + c*(Abar*s) = u + c*t.
```

All polynomial products use exact negacyclic convolution. The reference relation never accepts approximate
equality, client-selected tolerance, NTT-domain aliases or non-canonical coefficient representations.

## 12. Neural construction boundary

V1-P2 已实现独立 exact relation 和 `V1-C1-MSIS` graph；它不能加载 V0/A1 或 A4-C1 profile。

The first neural construction should use coefficient-domain expansion:

```text
fixed public Abar and t
runtime canonical (u,c,z)
-> exact affine negacyclic convolutions Abar*z and c*t
-> coefficient residuals Abar*z-u-c*t
-> modular-zero point-pulse checks
-> coefficient infinity-norm violations for z
-> final hard AND
```

NTT is an optional later backend, not part of the first semantic oracle. Direct convolution exposes every sign,
wraparound and accumulator bound for review and maps naturally to fixed affine or convolutional neural layers.
The frozen toy profile yields the dependency-free exact integer topology `56 -> 11056 -> 17 -> 1`: layer one contains
three ReLU hinges for every residual multiple in the complete bound ledger plus two norm-violation units per response
coefficient; layer two produces residual point pulses and one norm accumulator; layer three is the hard conjunction.
The implementation uses exact `int64` reduction and `int32` activations, proves `V_nn==V_ref` over all canonical
inputs, and has no reference fallback. Any NTT, PyTorch, qint8, CUDA or export mapping needs its own range proof and
differential testing.

## 13. Security games and proof separation

The V1-P2 analysis keeps separate:

1. **Completeness:** every honest emitted response satisfies the exact module equation and bound.
2. **HVZK/witness privacy:** accepted/aborted transcript distributions satisfy the cited rejection conditions.
3. **Special soundness/knowledge:** accepting transcripts with related commitment and distinct challenges yield a
   short M-SIS relation under the concrete theorem conditions.
4. **Public-key pseudorandomness:** the distribution `t=A*s1+s2` is an M-LWE question, separate from verifier
   arithmetic.
5. **Request binding/replay:** supplied by A3-v2 state and coordinator, not by M-SIS or M-LWE.
6. **Neural soundness:** the fixed V1-C1 graph proves `V_nn==V_ref` for canonical toy inputs; this concerns only
   compiled arithmetic.
7. **Fiat--Shamir/signature security:** ROM/QROM and SelfTargetMSIS claims are deferred to V2 or a separate
   signature checkpoint.

Finite forgery tests cannot establish any of these cryptographic reductions.

## 14. Completeness and retry policy

Measurements must separate prover abort rate, verifier false rejection among emitted honest responses, end-to-end
success within retry budget and latency including discarded masks. Canonical emitted honest responses require zero
verifier false rejections. Retry exhaustion is deny and produces zero protected-model calls.

## 15. Evidence and authorization boundary

当前 V1-P2 exact verifier 返回只含稳定结果码的 immutable evidence；A3-v2 adapter 只增加公开 binding
digests。verifier 不能读取/消费 A3 state、创建授权 context、调用模型或披露 response-dependent
internal arithmetic。

The coordinator claims the transcript before verification and commits access only for exact accept evidence bound
to the stored profile/message/commitment/challenge. Reject, abort, expiry, replay, tamper, route mismatch and internal
error all produce zero protected-model calls.

## 16. Required tests for the implementation checkpoints

已闭合的 exact-reference/A3-v2 checkpoint 覆盖：

- ring/profile construction, `X^N=-1`, coefficient ordering and independent negacyclic-convolution vectors;
- every wire domain, exact length, residue boundary, signed coefficient boundary and challenge weight;
- `Abar=[A|I]`, `t=Abar*s` conformance for temporary toy fixtures and no secret-key storage;
- valid response, `||z||_inf=B`, equation mutation, sign/wraparound and accumulator boundaries;
- identity/message/input/scope/nonce/commitment/challenge/transcript/response tamper;
- abort, expiry, same-commitment reuse, replay and concurrent duplicate response;
- one terminal response attempt and zero protected calls for every non-accept path;
- V0, V1-P1, A3-v1 and A4 credentials rejected by the V1-P2 parser with no fallback;
- exact-reference differential vectors independent of the later neural implementation.

The completed neural checkpoint additionally covers the fixed graph/range ledger, independently generated accepting
vectors, equation and norm tampering, foreign route/type confusion, no reference/access/model fallback, and A3-v2
neural-route acceptance at most once with reject/foreign-route zero protected calls.

Toy secrets and masks stay in process memory or pytest temporary directories and are never committed.

## 17. Deferred and excluded scope

Deferred items are:

- concrete security parameters, estimator output and production library selection;
- production key generation/prover, reviewed uniform sampler, rejection implementation and secret lifecycle;
- production `V1-P2-PSR-E1` prover/keygen, security parameters and reviewed library adapter;
- NTT, PyTorch/qint8/CUDA/export and performance;
- Fiat--Shamir signing, SelfTargetMSIS, ROM/QROM and strong-unforgeability claims;
- ML-DSA compression, decomposition, HighBits/LowBits, hints and standard encodings;
- distributed/durable state, TLS/channel binding, rate limiting, DoS and white-box guarantees.

No V0/V1-P1/A3-v1/A4 code is migrated or replaced by V1-P2.

## 18. Acceptance criteria for this decision

V1-P2 is accepted as the current protocol-selection checkpoint when:

1. the reviewed FSwA-S module protocol and its primary sources are unambiguous;
2. ring, module key relation, transcript, challenge, response, rejection and exact equation are frozen;
3. canonical coefficient ordering/encodings and A3-v2 binding are explicit;
4. M-LWE, M-SIS, A3 replay, neural soundness and Fiat--Shamir claims remain separate;
5. direct coefficient-domain convolution is the first exact semantic backend and NTT is deferred;
6. V1-P1 remains a baseline while V1-P2 becomes the only current V1 implementation target;
7. worklog, research, security and governance documents agree on the next exact-reference checkpoint.
