# A3 Challenge-Response and Request-Binding Protocol

## 1. Status and claim boundary

本文档固定阶段 A3 的协议规格，版本号为 `A3-v1`。A3 只定义受保护模型请求的规范消息、
challenge/nonce 生命周期、请求绑定、原子单次消费、evidence-only verifier 边界和唯一协调器提交顺序。

A3 属于 `V1-prep` 的请求绑定与 freshness 基础设施，不是完整的 V1 格身份认证协议。当前 V1-P2
已选择 reviewed FSwA-S 交互式 Module-SIS Sigma protocol，并在 A3 message、nonce 和 coordinator
边界之上单独定义协议安全游戏。由于该协议要求 commitment 先于 challenge，现有 A3-v1 不能原样作为 V1-P2 wire
protocol；后续必须新增独立 A3-v2 commit-first wrapper，同时保留 A3-v1 代码和测试。

A3 本身不定义签名、MAC 或其他认证证明关系，不提供不可伪造性，也不把 A0 的 23 字节 toy LWE
credential 升格为认证材料。A4 已选择 GPV PFDH 关系并实现 toy exact adapter；A4-C1 也已实现固定
integer affine/ReLU verifier 与 neural adapter。但当前小参数和公开 gadget 测试 profile 不满足生产
安全定义；完整组合仍不能称为生产公钥认证。A3 规格完成只表示协议语义已闭合，不表示不可伪造性
已实现。

本规格面向单进程、可信黑盒科研原型，不是生产网络协议。HTTP、TLS、账户注册、分布式状态、
持久化、高可用、速率限制和拒绝服务防护均不在 A3-v1 范围内。

## 2. Decision summary

A3-v1 固定以下主路线：

- challenge 由本地可信 issuer 生成，不接受客户端提供的时间或 nonce；
- 签名/证明消息是 133 字节固定二进制编码，绑定协议版本、模型、身份、scope、签发/到期时间、
  32 字节 nonce 和规范业务输入的 SHA-256 摘要；
- 当前受保护模型固定为 `CAN-A2-FMNIST-MLP-v1`，scope 固定为单图 top-1 prediction；
- proof 是 A3 层的 opaque bytes；当前 A4 GPV profile 由本地 registry 固定为 105 字节唯一编码，
  其他 profile 仍必须各自定义更窄的 exact encoding；
- verifier 只返回与精确消息摘要绑定的不可变 evidence，不返回 decision、context 或 capability；
- 唯一协调器在 verifier 接受后执行原子 `PENDING -> CONSUMED`，只有该状态转换的唯一成功者可以
  提交受保护模型决定；
- parser、binding、expiry、replay、verifier、policy 或状态错误全部 fail closed，并在提交前保持零
  protected-model calls；
- A0/A1 numeric evidence、A2 public response、客户端布尔值或预计算 decision 都不能进入 A3 接受路径。

无状态前馈网络不负责 replay 防护。可信 nonce store 是协议必要组件，但它不执行密码验证，也不改变
A4 验证计算进入固定神经 verifier 的研究边界。

## 3. Roles and trust assumptions

### Requester

- 可以申请任意已公开 identity 的 challenge，并提交任意 message、proof 和业务输入；
- 可以篡改、延迟、复制、顺序或并发 replay 历史响应；
- 不能提交 public key、算法、profile、模型实现、scope policy、evidence、decision 或 nonce 状态；
- 可以观察固定 challenge/deny/protected 响应及可见延迟。

### Trusted entry and challenge issuer

- 在启动时固定 A3 版本、受保护 entry、`model_id`、`scope_id`、TTL、时钟、随机源和状态存储；
- 严格验证 challenge 请求并为规范业务输入创建快照和摘要；
- 从可信时钟和密码学安全随机源生成时间与 nonce；
- challenge 签发不调用 verifier、public model 或 protected model。

### Local identity and verification registry

- 将 32 字节 `identity_id` 映射到唯一启用的本地公开验证 profile；
- 固定 proof 算法、公开验证信息、proof 编码和 verifier backend；
- 未知、重复、禁用、过期或不完整 entry fail closed；
- 私钥、签名 secret 和客户端提供的 public key 不进入 registry、verifier 或模型。

### Evidence-only verifier

- 接收本地 profile、精确 133 字节 message 和 opaque proof bytes；
- 只返回不可变、无授权能力且绑定 `SHA-256(message)` 的结构化 evidence；
- 不读取或修改 nonce 状态，不提交权限，不调用业务模型；
- 当前 A4 GPV exact 或 A4-C1 neural adapter 可实例化该接口，但没有本地 profile 时默认产生配置
  拒绝；toy profile 不提供不可伪造性。

### Trusted nonce store and coordinator

- nonce store 提供线程安全、线性化的创建、读取和条件消费；
- coordinator 是唯一 decision commit point，并拥有唯一受保护模型调用入口；
- store 返回的状态结果不是 authorization capability；只有 coordinator 可以据此提交决定；
- 任何状态不确定性、异常或原子性缺失都必须拒绝。

## 4. Fixed A3-v1 profile and typed request fields

| Name | Value | Meaning |
| --- | --- | --- |
| protocol name | `CAN-A3-BOUND-CHALLENGE-v1` | protocol/profile identifier |
| `version` | unsigned integer `1` | canonical message version |
| external response version | exact integer `3` | separates A3 envelopes from A2 |
| `model_id` | unsigned integer `1` | local mapping to `CAN-A2-FMNIST-MLP-v1` |
| `identity_id` | exactly 32 bytes | opaque local identity-registry key |
| `scope_id` | unsigned integer `1` | local mapping to one-image top-1 prediction |
| challenge TTL | exactly `60_000` ms | fixed non-production experiment lifetime |
| nonce | exactly 32 bytes | trusted CSPRNG output, never client-selected |
| input digest | exactly 32 bytes | SHA-256 of the canonical A3 input encoding |
| message | exactly 133 bytes | canonical proof message defined below |
| proof | exact `bytes`, length `1..65_535` | A3 outer bound; current A4 GPV profile narrows it to exactly 105 bytes |

All security-critical integers must be exact built-in integers; `bool` is rejected. All byte fields must be exact
immutable `bytes`; `bytearray`, `memoryview`, text and implicit conversions are rejected. Unknown, duplicate,
missing or extra fields are rejected by a closed transport/API schema before protocol processing.

The logical challenge request contains only:

```text
version, model_id, identity_id, scope_id, image
```

The logical proof-response request contains only:

```text
message, proof, image
```

There is no algorithm, key, verifier, profile, policy, entry, model backend, evidence, decision, timestamp or
nonce override field. A network transport is deferred; any future JSON/CBOR/HTTP adapter must preserve these
exact types and duplicate-field rejection rather than invent alternate encodings.

## 5. Canonical business-input digest

A3-v1 binds exactly one protected Fashion-MNIST image. The entry first applies the existing A2 protected input
contract and additionally rejects negative zero. The accepted snapshot is exactly:

- exact `torch.Tensor`;
- CPU, strided, contiguous, `float32`;
- shape `(1,1,28,28)`;
- finite values in the closed interval `[0,1]`;
- no element whose IEEE-754 binary32 bit pattern is negative zero (`0x80000000`).

The entry detaches and clones the tensor before hashing. The cloned snapshot, not the caller-owned tensor, is used
for all subsequent digest checks and the eventual protected-model call. This prevents caller mutation between
verification and invocation.

Remove the batch dimension and encode the `(1,28,28)` image as:

```text
input_bytes =
    b"CAN-A3-INPUT-v1\x00"
    || u8(3)
    || u16_be(1)
    || u16_be(28)
    || u16_be(28)
    || 784 row-major IEEE-754 binary32 values in big-endian byte order

H(input) = SHA-256(input_bytes)
```

No numeric conversion, clipping, rounding, resize, normalization, alternate stride walk or host-endian dump is
allowed. A semantically similar but bitwise different positive float32 image has a different digest. NaN, infinity,
negative zero, float64, non-contiguous views and values outside `[0,1]` reject instead of being normalized.

## 6. Canonical 133-byte message encoding

The exact message is:

```text
b"CAN-A3-MSG-v1\x00"
|| u8(version)
|| u32_be(model_id)
|| identity_id[32]
|| u16_be(scope_id)
|| u64_be(issued_at_ms)
|| u64_be(expires_at_ms)
|| nonce[32]
|| H(input)[32]
```

| Offset | Size | Field | Validation |
| ---: | ---: | --- | --- |
| `0` | 14 | domain separator | exact bytes `CAN-A3-MSG-v1\x00` |
| `14` | 1 | `version` | exactly `1` |
| `15` | 4 | `model_id` | unsigned big-endian, exactly `1` for this profile |
| `19` | 32 | `identity_id` | exact local registry key |
| `51` | 2 | `scope_id` | unsigned big-endian, exactly `1` |
| `53` | 8 | `issued_at_ms` | trusted server UTC Unix time in milliseconds |
| `61` | 8 | `expires_at_ms` | exactly `issued_at_ms + 60_000` |
| `69` | 32 | `nonce` | trusted, non-reused 256-bit challenge |
| `101` | 32 | `H(input)` | digest from section 5 |

The total length is exactly 133 bytes. Prefixes, suffixes, alternate domain tags, non-canonical integer widths,
text encodings and field reordering are rejected. The verifier receives these exact bytes; it does not reconstruct
an ad hoc string, dictionary or framework tensor.

## 7. Challenge issuance protocol

Challenge issuance follows this order:

1. Validate the closed challenge-request schema and exact field types.
2. Require the trusted entry's fixed `version`, `model_id` and `scope_id`.
3. Resolve `identity_id` to one enabled local verification profile/version without accepting client key material.
4. Validate, detach and clone the image; compute the canonical input digest from the snapshot.
5. Read trusted wall and monotonic clocks. Set `expires_at_ms = issued_at_ms + 60_000` and a monotonic deadline
   exactly 60 seconds after issuance.
6. Generate 32 bytes from the trusted CSPRNG. A nonce collision within the current process lifetime is a hard
   issuer error; generate no challenge and do not overwrite an existing record.
7. Build the exact 133-byte message and atomically insert a `PENDING` record.
8. Return a fixed challenge envelope containing the exact message bytes.

The monotonic deadline is authoritative for local expiry so wall-clock rollback cannot extend a challenge. The
signed wall timestamps remain exact binding and audit fields. Clock failure, overflow, backward monotonic movement,
CSPRNG failure, registry failure, duplicate nonce or store failure returns one fixed challenge-deny envelope and
creates no usable challenge.

Issuance never calls the proof verifier or either A2 model. It stores the message digest and binding metadata, not
the raw image. Multiple outstanding challenges for one identity are allowed in A3-v1; rate limiting is deferred.

## 8. Proof-response parsing and binding checks

The response entry performs these checks before proof verification:

1. Validate exact request schema, exact `bytes` message/proof types and the A3 outer proof-length bound.
2. Parse the exact 133-byte message and reject every non-canonical field.
3. Require local trusted `version`, `model_id` and `scope_id`; no route is selected from request data.
4. Resolve the message `identity_id` through the local registry and require the profile/version to equal the
   immutable value captured at challenge issuance.
5. Validate, detach and clone the response image, recompute `H(input)` and compare it with the message field.
6. Read the nonce record and require exact stored message digest, identity, model, scope and input digest equality.
7. Require state `PENDING` and trusted monotonic time strictly before the stored deadline.

Expiry uses the half-open interval:

```text
issued_at <= now < expires_at
```

At the exact deadline the challenge is expired. A client wall clock is never consulted. Missing, consumed, expired,
wrongly bound or unreadable state maps to the same external deny and does not call the proof verifier when the
failure is already known.

## 9. Evidence-only verifier contract

After all pre-verification checks pass, the coordinator calls exactly one locally fixed verifier adapter with:

```text
trusted_verification_profile, exact_message_bytes, exact_proof_bytes
```

The A4 adapter returns one exact immutable evidence type with at least:

```text
evidence_version
result_code
identity_id
message_sha256
local_profile_id
```

Only an exact A4-defined accept result whose `identity_id` and `message_sha256` equal the current request may
continue. A boolean, mapping, subclass, caller-created object, A0 `ReferenceEvidence`, A1 `A1Evidence`, A2 response,
wrong-message evidence, unknown result code or verifier exception rejects.

Evidence contains no nonce-store handle, coordinator decision, model reference, context or capability. Detailed
proof failure information is internal only. A3 provides no fallback verifier, `A4 OR A1` route or ordinary-library
verification alternative. Without one explicitly configured local A4 public-verification profile, the protected A3
entry is disabled and returns configuration deny.

## 10. Trusted nonce lifecycle and state contract

Each record has immutable binding data plus one lifecycle state:

```text
ABSENT -> PENDING -> CONSUMED
                 -> logically EXPIRED at deadline
```

The record stores at least:

- exact nonce and `SHA-256(message)`;
- identity, model, scope and input digest;
- local verification profile/version selected at issuance;
- signed issuance/expiry timestamps and monotonic deadline;
- exact state and, after success, non-sensitive consumption time/result code.

Rules:

- creation is insert-only; duplicate keys never replace records;
- `CONSUMED` is terminal and cannot be reset, renewed or cloned;
- expiry is terminal for authorization even if physical cleanup has not run;
- cleanup may remove expired/consumed records, but removed records remain reject-only and a nonce is never reused
  during the process lifetime;
- restart invalidates every outstanding in-memory challenge; an old message has no record and rejects;
- the store is injected trusted state owned by one coordinator, not module-level global mutable authorization state;
- A3-v1 claims only single-process/thread-safe linearizability, not multi-process or distributed consistency.

Invalid proof, malformed input and pre-verification binding failure do not consume a pending challenge. A valid
proof reaches one atomic conditional consume. This permits retry after malformed/invalid proof while preserving
at-most-one successful protected invocation. Rate limiting and proof-query abuse remain separate concerns.

## 11. Atomic consume and coordinator commit order

After exact accept evidence, the coordinator invokes one linearizable operation equivalent to:

```text
consume_if_pending(
    nonce,
    expected_message_sha256,
    expected_identity,
    expected_local_profile,
    expected_model,
    expected_scope,
    expected_input_digest,
)
```

The operation succeeds for at most one caller and atomically changes `PENDING` to `CONSUMED`. It rechecks all
binding fields and reads the trusted monotonic clock to recheck expiry inside the same critical section/transaction;
a caller-provided time or prior non-atomic read is never sufficient.

Normative protected flow:

```text
untrusted message + proof + image
-> strict parsing and canonical image snapshot
-> exact challenge-state and request-binding checks
-> one locally fixed evidence-only verifier
-> exact accept evidence for this message
-> atomic PENDING -> CONSUMED
-> local policy check and one coordinator commit
-> exactly one protected-model call on the same snapshot
-> fixed protected response
```

If atomic consume loses a concurrent race, expires, detects drift or raises, the request denies with zero
protected-model calls. Once consume succeeds it is never rolled back, including on policy denial, process error,
model exception, invalid model output or response-construction failure. This gives at-most-once invocation, not
exactly-once delivery. A model failure after a committed decision may have one recorded model entry and must not be
misreported as a zero-call rejection or retried under the same nonce.

## 12. External responses, audit and information release

A3-v1 uses exact version-3 logical envelopes:

```text
CHALLENGE = {"version": 3, "status": "challenge", "message": <133 exact bytes>}
DENY      = {"version": 3, "status": "deny"}
PROTECTED = {"version": 3, "status": "protected", "class_id": 0..9}
```

Transport serialization of the bytes-valued challenge is deferred. Every failure after challenge issuance uses the
same deny envelope. Responses never expose proof, input digest separately, identity registry details, verifier
evidence, nonce state, policy, backend, logits, confidence, features or detailed error reason.

Internal audit uses stable bounded result codes and may record event version, local profile identifier, model/scope,
timestamps, counters and truncated/dedicated digests for correlation. It must not record raw proof, complete message,
nonce, raw input, public key material beyond approved identifiers, logits, features or reusable authorization data.
Audit failure before commit fails closed if it prevents proving state or call-count invariants.

## 13. Security games and formal properties

The A3 implementation must state its results conditionally on the future proof verifier. Let `AcceptProof(pk,m,p)`
denote the exact A4 reference relation and let `Invoke(nonce)` mean entering the protected model for that challenge.

### Canonical binding

For every protected invocation, the invoked image snapshot, trusted model, identity and scope must equal the fields
of the exact message accepted by the verifier and atomically consumed:

```text
Invoke(n) -> AcceptProof(pk_identity, m, p)
             and H(invoked_input) = m.input_digest
             and local_model = m.model_id
             and local_scope = m.scope_id
```

This is a protocol-composition claim conditional on A4 verifier correctness/unforgeability. A3 alone does not prove
that an attacker cannot construct `p`.

### At-most-once protected invocation

For one issued nonce under one linearizable store:

```text
sum(Invoke(nonce)) <= 1
```

This property must hold for sequential replay, concurrent replay, verifier delay, model exceptions and response
failure. It is an at-most-once side-effect boundary, not exactly-once execution or response delivery.

### Expiry

No request whose atomic consume linearization point is at or after the monotonic deadline may invoke the protected
model, even if proof verification started before expiry.

### Tamper resistance

Changing version, model, identity, scope, either timestamp, nonce, input digest, image or proof must reject unless
the attacker supplies a proof accepted for the newly constructed exact message and that message has its own pending
unexpired challenge record. Reusing a proof for a different field tuple is not accepted.

### No downgrade or evidence injection

No A0/A1 evidence, public response, boolean, alternate algorithm, client key/profile or verifier exception can
produce an A3 consume or protected invocation. A verifier/store/configuration failure cannot fall back to A2-E1's
replayable toy gate or A2-E2 public capability.

## 14. Required acceptance matrix

| Case | Verifier calls | Successful consumes | Protected calls | External result |
| --- | ---: | ---: | ---: | --- |
| valid issued message + future valid proof + same image | 1 | 1 | 1 | `PROTECTED` |
| malformed challenge request | 0 | 0 | 0 | challenge deny |
| RNG/clock/store/registry issuance failure | 0 | 0 | 0 | challenge deny |
| malformed message/proof/image | 0 | 0 | 0 | `DENY` |
| unknown/disabled identity or local profile | 0 | 0 | 0 | `DENY` |
| model/scope/version/timestamp/nonce tamper | 0 | 0 | 0 | `DENY` |
| different image or input-digest tamper | 0 | 0 | 0 | `DENY` |
| invalid proof or exact reject evidence | 1 | 0 | 0 | `DENY` |
| A0/A1/public/client-created evidence injection | 0 | 0 | 0 | `DENY` |
| expired at precheck or atomic consume | 0 or 1 | 0 | 0 | `DENY` |
| first sequential valid submission | 1 | 1 | 1 | `PROTECTED` |
| later sequential replay | 0 | 0 | 0 | `DENY` |
| N concurrent valid submissions | up to N | exactly 1 | exactly 1 | one protected, N-1 deny |
| verifier exception/wrong evidence message | 1 | 0 | 0 | `DENY` |
| store read/consume exception or uncertain state | 0 or 1 | 0 or unknown | 0 | `DENY` |
| local policy denial after valid consume | 1 | 1 | 0 | `DENY` |
| model/output failure after commit | 1 | 1 | 1 entered | `DENY`, no retry/fallback |
| process restart with old outstanding message | 0 | 0 | 0 | `DENY` |

Tests must additionally cover every exact field boundary, bool/int confusion, bytes-like confusion, proof lengths
`0/1/65_535/65_536`, timestamp overflow, expiry immediately before/at/after deadline, nonce collision, record drift,
input mutation after snapshot, negative zero, all float32 image boundaries, state cleanup and deterministic
barrier-controlled concurrency. Call counts must instrument the real protected model boundary.

Security-game reports must separate properties proven by state-machine/locking structure from properties tested
with a deterministic local proof stub or publicly forgeable A4 gadget fixture. Neither can be reported as production
authentication or unforgeability.

## 15. State, artifact and logging policy

- nonce records are ephemeral trusted security state, not model parameters or verifier evidence;
- the A3-v1 implementation checkpoint uses only an injected in-memory single-process store and test temporary
  directories; it does not create a local database;
- raw image, proof, complete message, nonce and reusable response are never written to reports or checkpoints;
- deterministic test nonces are allowed only in tests through an explicitly injected fake random source;
- production secrets, private keys and real credentials are never generated or persisted by A3;
- crash recovery, durable consume, replication and state migration are unsupported and must not be inferred from
  thread-safe in-memory tests.

## 16. Deferred and excluded scope

A3-v1 deliberately excludes:

- implementing the selected V1-P2 polynomial exact relation, A3-v2 commit-first state machine, prover or sampler;
- replacing, renaming or rewriting A3-v1/A0/V0 code as V1; later routes use separate modules and tests;
- ML-DSA, NTT, hint, norm or complete signature encoding logic;
- using A0 toy LWE unlock, shared secrets in model weights or public encryption as authentication;
- key enrollment, identity proofing, key rotation, revocation and account recovery;
- distributed/multi-process nonce consistency, durable crash recovery and high availability;
- TLS/channel binding, secure response delivery, network endpoint authorization and rate limiting;
- Stage B bearer capability, Router, MoE, agent and tool gateway;
- qint8, CUDA, export, MASK, shared trunk/head and alternate model architectures;
- white-box integrity, TEE, secure boot, remote attestation, denial of service and complete side-channel protection.

A captured valid proof/message/image tuple can race the legitimate submitter; A3 guarantees at most one bound model
invocation, not which network peer receives the response. Preventing theft or proving peer/session identity requires
an authenticated transport/session design outside this checkpoint.

## 17. Implementation checkpoint sequence

1. **completed:** freeze the canonical input/message encoding, challenge lifecycle, evidence-only
   adapter contract, atomic consume order, security games and acceptance matrix without runtime authentication code.
2. **completed:** implement A3 canonicalization/parser, trusted in-memory nonce store, fixed envelopes and single
   coordinator protocol shell, disabled by default without an A4 profile.
3. **completed reference checkpoint:** select reviewed GPV PFDH, freeze the 105-byte proof/public profile and exact
   relation, and implement the no-private-key reference adapter with explicit toy/gadget limitations.
4. **completed neural checkpoint:** compile and validate A4-C1 for canonical `(y,z)` without making an
   authentication or unforgeability claim.
5. **completed baseline checkpoint:** freeze V1-P1 ordinary matrix SIS as a retained comparison design.
6. **selected current protocol checkpoint:** freeze V1-P2 FSwA-S Module-SIS, its ring, commit-first transcript,
   polynomial encoding, bounded-uniform rejection, security games and A4-C1 incompatibilities.
7. **completed V1-P2 checkpoints:** implement the separate coefficient-domain exact reference, A3-v2 protocol shell,
   dependency-free V1-C1 graph and neural evidence route; preserve A3-v1 and V0 without fallback.
8. **next checkpoint:** freeze the V1-M1 GPU/software tuple before any CIFAR-100/ResNet-18 baseline or training.

No checkpoint may call A0/A1 as an authentication fallback, describe proof-stub/gadget tests as production
authentication, or relax A3 canonical encoding and nonce semantics to accommodate a weaker verifier route.

## 18. Acceptance criteria for this specification

This specification is complete when:

1. The exact input digest and 133-byte message encoding are unambiguous.
2. Time, nonce generation, TTL, state transitions, cleanup and restart behavior are fixed.
3. Requesters cannot choose algorithms, keys, profiles, policy, evidence, decision or nonce state.
4. The verifier is evidence-only and evidence is bound to the exact message digest and identity.
5. Atomic consume rechecks binding and expiry, and only its unique winner may reach the model.
6. Sequential/concurrent replay, tamper, expiry and every pre-commit failure have explicit zero-call expectations.
7. Post-commit model failure is distinguished from zero-call protocol rejection and never rolls state back.
8. A0/A1 evidence and public responses are structurally excluded from the A3 path.
9. Security games distinguish A3-v1 guarantees, A4 conformance, V1-P2 M-LWE/M-SIS assumptions and neural soundness.
10. Toy, single-process, black-box, non-production and deferred-scope limitations are explicit.
