# V1 Lattice Identification Protocol Selection Decision

## 1. Status and claim boundary

本文档保留 CAN 首个 V1 格身份协议设计 baseline，决定编号为 `V1-P1`，协议族标识为
`CAN-V1-LYU12-SIS-ID-v1`。选择的是 Lyubashevsky 2012 矩阵格签名所对应的交互式
Sigma protocol，并以 Liu--Zhandry 2019 明确抽取和分析的 commit--challenge--response 形式作为
协议语义来源。

项目负责人现已决定当前 V1 主路线直接采用 Module-SIS Sigma protocol。V1-P1 因此不再是实现
目标，只作为普通矩阵 SIS 的历史设计与比较 baseline 保留；当前规范见
`docs/V1_MODULE_SIS_PROTOCOL_DECISION.md` 的 `V1-P2`。这一状态变化不删除本文档，也不允许未来
用 V1-P2 覆盖 V0、V1-P1 或 A4/V1-prep 代码。

V1-P1 是协议选择和组合决定，不是已实现的认证系统。本 checkpoint 不实现 key generation、
prover、Gaussian sampler、rejection sampler、exact verifier 或 neural verifier，也不冻结生产安全
参数。任何后续 toy conformance profile 必须标记为非生产；任何安全参数主张必须另有 estimator、
实现审查和具体安全分析。

V1-P1 首先采用交互式 identification。Fiat--Shamir 非交互变换、签名 API 和 ML-DSA 均不属于本
决定的已实现或已证明范围。

V0、V1 与 V2 是并行保留的可复现路线。V1 必须新增独立 package/module、协议标识、trusted
registry、wire parser、A3-v2 adapter 和测试，不得把 A0/V0 文件重命名或改写成 V1。V2 后续同样
通过独立 ML-DSA adapter 接入，不覆盖 V0 或 V1。只有不携带协议语义且接受域一致的通用整数 helper
才可显式复用；任何 route fallback 或交叉接受均禁止。

## 2. Selected protocol and primary sources

V1-P1 的规范来源是：

1. Vadim Lyubashevsky, *Lattice Signatures without Trapdoors*, EUROCRYPT 2012. 原始一般矩阵方案
   使用短秘密矩阵 `S`、公开 `A,T=AS`、Gaussian mask、稀疏 ternary challenge、response
   `r=y+Sc` 和 rejection sampling：<https://eprint.iacr.org/2011/537.pdf>。
2. Qipeng Liu and Mark Zhandry, *Revisiting Post-Quantum Fiat-Shamir*, 2019. Section 5.2 从上述签名
   明确抽取交互式 Sigma protocol，并给出 commitment、challenge、response、二范数验证、HVZK、
   weak completeness 和 2-soundness 分析：<https://eprint.iacr.org/2019/262.pdf>。
3. Julien Devevey, Pouria Fallahpour, Alain Passelegue, Damien Stehle and Keita Xagawa,
   *A Detailed Analysis of Fiat-Shamir with Aborts*, CRYPTO 2023/updated 2024. 该工作说明 abort loop 的
   ROM/QROM 分析存在不能忽略的技术条件和既有证明错误，因此本决定不把交互式 V1-P1 自动升级为
   非交互签名结论：<https://eprint.iacr.org/2023/245.pdf>。

引用这些来源只冻结协议语义和证明边界，不表示当前 CAN 参数或实现继承其安全定理。

## 3. Candidate comparison and disposition

| Candidate | Public verification core | First-V1 disposition |
| --- | --- | --- |
| Lyubashevsky 2008/2009 ideal-lattice ID and Fiat--Shamir with aborts | homomorphic ideal-lattice hash, box/ring response domain | reviewed baseline, but ring/domain machinery is less aligned with current standard-matrix compiler |
| Lyubashevsky 2012 plus Liu--Zhandry 2019 extracted Sigma protocol | `Ar = Tc + a mod q` and `||r||_2 <= B` | selected as V1-P1 |
| Silva--Campello--Dahab LWE identification | noisy LWE public key plus Stern-style commitments, permutations and Hamming-weight checks | deferred; substantially enlarges the first neural relation |
| ML-DSA | Module-LWE/Module-SIS standard signature verification with decomposition and hints | retained as V2, not a V1 shortcut |

The selected protocol keeps the verifier relation exact and public. It does not call toy LWE decryption and does
not introduce a client-selected tolerance. Its principal new neural difficulty is the exact squared Euclidean norm,
not noisy decryption.

## 4. Roles and trust assumptions

- **Prover:** holds the secret matrix `S`, samples `y`, performs rejection sampling and never exposes `S`.
- **Verifier service:** holds only the local public profile, A3 state and public matrices `A,T`.
- **Trusted registry:** maps an identity to exactly one enabled V1 profile and public key; the request cannot choose
  `A`, `T`, `q`, dimensions, challenge set, norm bound, sampler profile or algorithm.
- **A3/V1 coordinator:** snapshots the business input, creates the A3 message, stores the commitment/challenge
  state, atomically claims one response attempt and is the only authorization commit point.
- **Neural verifier:** future deterministic evidence producer only; it never owns the secret, nonce state or model.

The black-box, trusted-host and unsupported-white-box assumptions remain those in `SECURITY.md`.

## 5. Parameterized public profile

A concrete V1 profile must freeze the following trusted values before key loading:

```text
protocol_id = CAN-V1-LYU12-SIS-ID-v1
profile_id, profile_digest, identity_id
n, m, k
q with 3 <= q < 2^32
d, kappa
sigma sampler identifier and exact parameters
rejection constant M and retry policy
B2, the exact integer squared-response bound
A in Z_q^(n x m)
T in Z_q^(n x k)
challenge set C = {c in {-1,0,1}^k : ||c||_1 <= kappa}
```

`B2` is stored as an audited exact integer. A verifier never evaluates floating `eta*sigma*sqrt(m)` at runtime.
The profile must prove that every encoded coefficient, product, reduction and squared-norm accumulator fits the
selected exact integer backend. A future conformance profile and a future security profile are separate objects and
must not share claims merely because they use the same protocol identifier.

## 6. Key relation

Key generation samples a small secret matrix

```text
S in {-d,...,d}^(m x k)
```

and publishes:

```text
A <- Z_q^(n x m)
T = A*S mod q
pk = (A,T,profile_digest,identity_id)
sk = S
```

The verifier and model store only `pk`. `S`, Gaussian sampler state and prover randomness remain outside the
repository, verifier, model, logs and client-controlled request fields. V1-P1 uses the exact SIS-style public relation
`T=AS`; it does not use a noisy public key `T=AS+E`.

## 7. A3-bound commit-first transcript

V1-P1 requires a new A3-v2 wrapper because the prover commitment must exist before the verifier challenge.
The existing A3-v1 133-byte business message remains unchanged and is embedded in the new binding.

```text
1. Prover samples y <- D_sigma^m and computes a = A*y mod q.
2. Prover sends canonical commitment bytes together with identity and business input x.
3. Coordinator validates a, snapshots x and creates the canonical A3 133-byte message m_a3.
4. Coordinator samples one c uniformly from the fixed local set C after receiving a.
5. Coordinator stores (profile, m_a3, input snapshot, a, c, transcript_id) as PENDING.
6. Prover computes r = y + S*c and performs the specified rejection sampling.
7. On sampler acceptance, prover sends one canonical response; otherwise it aborts and starts again with fresh y/a.
8. Coordinator atomically claims the one response attempt, invokes the deterministic verifier, commits authorization
   only for exact accept, and then calls the protected model with the stored input snapshot.
```

A commitment, mask `y`, challenge or response must never be reused in another transcript.

## 8. Binding and transcript identifiers

Let `commitment_bytes` be the exact encoding from Section 9. The server computes:

```text
commitment_digest = SHA256(commitment_bytes)
binding_digest = SHA256(
  "CAN-V1-BIND-v1\0" ||
  profile_digest ||
  m_a3 ||
  commitment_digest
)
transcript_id = SHA256(
  "CAN-V1-TRANSCRIPT-v1\0" ||
  binding_digest ||
  challenge_bytes
)
```

The evidence binds `profile_digest`, `identity_id`, the A3 message digest, `commitment_digest` and
`transcript_id`. The challenge is not accepted from a response payload alone; it must exactly match the trusted
pending state. This external state binding supplies model/input/scope/freshness semantics that are not part of the
bare Sigma relation.

## 9. Canonical wire encodings

All integers use fixed-width big-endian encoding. Unknown bytes, trailing bytes, wrong domains, non-exact `bytes`,
wrong profile binding, non-canonical residues and type confusion reject.

```text
commitment_bytes =
  b"CAN-V1-COM-v1\0" ||
  profile_id:u16 ||
  a[0]:u32 || ... || a[n-1]:u32

challenge_bytes =
  b"CAN-V1-CHAL-v1\0" ||
  profile_id:u16 ||
  c[0]:i8 || ... || c[k-1]:i8

response_bytes =
  b"CAN-V1-RESP-v1\0" ||
  transcript_id:32-bytes ||
  r[0]:i32 || ... || r[m-1]:i32

abort_bytes =
  b"CAN-V1-ABORT-v1\0" ||
  transcript_id:32-bytes
```

Every `a[i]` must satisfy `0 <= a[i] < q`. Each challenge byte must be exactly `0xff`, `0x00` or `0x01`,
representing `-1`, `0` or `1`, and the decoded vector must lie in `C`. Response coefficients are canonical signed
32-bit integers and must additionally satisfy the profile range certificate before any matrix operation. Public keys
are loaded from the trusted registry and are never accepted in these wire objects.

## 10. Rejection sampling and abort semantics

For `r=y+Sc`, the prover outputs `r` according to the Lyubashevsky rejection rule:

```text
min(D_sigma^m(r) / (M * D_(Sc,sigma)^m(r)), 1)
```

Otherwise the prover emits `abort_bytes` or lets the pending transcript expire, destroys `y`, and retries only with
a fresh commitment. Abort is a normal protocol outcome, not verifier false rejection and not evidence of an attack.

The server never changes `c` for an existing commitment. Explicit abort is terminal. A parsed response attempt is
atomically claimed exactly once; valid, invalid and malformed-after-envelope responses all make that transcript
terminal. This prevents concurrent duplicate verification and any attempt to obtain multiple verifier challenges for
the same commitment. A new attempt requires a new A3 nonce, commitment and transcript identifier.

## 11. Exact verification relation

After canonical parsing and exact pending-state matching, define:

```text
u = (a + T*c) mod q
norm2 = sum_j r[j]^2
```

The exact reference verifier accepts iff:

```text
norm2 <= B2
and
A*r mod q == u
```

Equivalently, `A*r = T*c+a mod q`. The verifier uses public information only, is deterministic and side-effect free,
and returns structured evidence without authority. Parser/state/configuration failures and arithmetic range failures
are distinct stable reject classes but produce the same external deny envelope.

## 12. A4-C1 compatibility decision

The equation half can be normalized to the same abstract form as A4-C1:

```text
y_core = (a + T*c) mod q
z_core = r
A*z_core mod q == y_core
```

The current A4-C1 implementation cannot be reused directly because it fixes `q=257`, dimensions `8 x 72`, signed
int8 response coefficients and `||z||_inf<=1`. V1-P1 instead has profile-sized matrices, signed int32 response
encoding and an exact squared Euclidean norm bound.

Therefore V1 must define a new exact relation and a new `V1-C1` neural construction. The A4 point-pulse modular
equality technique is a reusable proof pattern only. The squared-norm check may require a bounded lookup/pulse
square construction or another explicitly proved integer network; it cannot be silently replaced with an infinity
norm, margin or approximate floating comparison.

## 13. Security games and proof separation

The V1 analysis must keep the following claims separate:

1. **Protocol completeness:** an honest non-aborting response satisfies the exact verifier; aggregate success also
   depends on the rejection probability and retry policy.
2. **Transcript privacy/HVZK:** rejection sampling makes accepted response distributions statistically independent
   of the secret within the cited parameter conditions.
3. **Special soundness/knowledge:** two accepting responses for the same commitment and different challenges yield
   a short SIS witness under the cited protocol analysis.
4. **Active impersonation:** must be stated for the concrete interactive protocol and parameter profile, not inferred
   from finite tests.
5. **Request binding/replay:** supplied by the A3 message, server challenge state, transcript identifier, terminal
   single-attempt lifecycle and coordinator; it is not implied by SIS.
6. **Neural soundness:** future `V_nn=1 -> V_ref=1` proof concerns only the compiled verifier and does not prove the
   protocol security games above.

Classical interactive, quantum interactive and ROM/QROM non-interactive claims must each name their exact theorem,
assumptions and parameter conditions.

## 14. Completeness and retry policy

V1-P1 intentionally has weak per-attempt completeness because the prover may abort. A correct implementation must
measure separately:

- prover-side rejection/abort rate;
- verifier false rejection among responses the honest prover actually emitted;
- end-to-end success within the locally configured retry budget;
- latency distribution including rejected prover samples but excluding any protected model call before accept.

The required verifier false rejection count for canonical emitted honest responses is zero. A sampler abort is never
relabelled as verifier rejection. Retry exhaustion returns deny and produces zero protected model calls.

## 15. Evidence and authorization boundary

The future V1 verifier returns immutable evidence containing only stable codes and public binding digests. It cannot
consume nonce state, create an authorization context, call a model or return secret-dependent diagnostics.

The coordinator atomically claims the transcript before deterministic verification to enforce one response attempt.
Only exact accept evidence for the same stored profile/message/commitment/challenge may commit protected access.
All reject, abort, expiry, replay, tamper, profile mismatch and internal-error paths produce zero protected model
calls. The request cannot submit evidence, decision, policy, key or an alternate verifier route.

## 16. Required tests for the next implementation checkpoint

The exact-reference checkpoint must add deterministic unit, integration and security tests for:

- every wire domain, exact length, coefficient boundary, non-canonical residue and ternary challenge encoding;
- trusted profile/key dimensions, immutability, `T=AS` conformance for toy fixtures and no private-key storage;
- valid non-aborted transcript, norm boundary, equation tamper and arithmetic overflow boundaries;
- commitment, A3 message, identity, scope, input, nonce, challenge, transcript and response mutation;
- explicit abort, expiry, same-commitment reuse, same-response replay and concurrent duplicate response;
- one terminal response attempt and zero protected calls for every non-accept path;
- exact-reference differential vectors independent of any future neural implementation;
- explicit demonstration that A4-C1 rejects or cannot load the incompatible V1 profile rather than falling back.

Toy secrets and sampler randomness remain in process memory or pytest temporary directories and are never committed.

## 17. Deferred and excluded scope

The following are deferred:

- concrete conformance and security parameter tuples, estimator output and production cryptographic library choice;
- key generation, prover, Gaussian/rejection sampler and secret-key lifecycle implementation;
- V1 exact reference, `V1-C1` neural construction, PyTorch/qint8/CUDA/export and performance measurement;
- Fiat--Shamir signing, bounded/unbounded abort-loop ROM/QROM proofs and strong unforgeability;
- noisy `T=AS+E`, LWE rounding, decomposition, hints, Module-LWE/Module-SIS and ML-DSA;
- distributed/durable A3 state, TLS/channel binding, rate limiting, DoS and white-box guarantees.

V0/A0、A1/A2、A3-v1 和 A4/V1-prep 的既有代码与测试不属于迁移或替换对象；后续 V1/V2 只允许
通过新增隔离模块演进，并必须加入 route-confusion 和 no-fallback 回归测试。

No deferred item may be described as implemented or inherited from A4-C1.

## 18. Acceptance criteria for this decision

V1-P1 is accepted as the protocol-selection checkpoint when:

1. the selected reviewed protocol, exact public relation and primary sources are unambiguous;
2. key, commitment, challenge, response, rejection, abort and retry semantics are frozen;
3. A3 binding, canonical encodings, single-attempt state and evidence/coordinator boundaries are explicit;
4. SIS protocol security, A3 replay binding and neural soundness claims remain separate;
5. direct A4-C1 incompatibilities and the required new V1 exact/neural relation are recorded;
6. non-interactive Fiat--Shamir and all implementation work remain explicitly deferred;
7. research, security, worklog and governance documents agree on the next exact-reference checkpoint.
