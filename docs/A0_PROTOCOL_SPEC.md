# A0 Toy LWE Numerical Unlock Protocol

## 1. Status and claim boundary

本文档固定阶段 A0 的可执行协议规格，版本号为 `A0-v1`。A0 只研究三个问题：精确整数 LWE-shaped relation、后续神经实现的数值 margin，以及验证结果对业务模型的 fail-closed 门控。

A0 是非生产、非安全参数的数值实验。它不是数字签名，不提供身份认证、不可伪造性、请求绑定或 replay 防护。任何 API、实验报告和论文段落都必须使用 “toy LWE numerical unlock” 或 “toy capability gate”，不能使用“验签成功”描述 A0。

本文档只规定精确 relation `V_ref` 和未来 `V_nn` 必须满足的接口/误差契约，不实现密码代码或神经网络。

## 2. Roles and trust assumptions

### Trusted experiment harness

- 根据显式实验 seed 在进程内生成 toy 参数；
- 创建本地只读 profile/slot registry；
- 充当 token issuer，生成合法和负向测试凭据；
- 不把 toy secret 或含 secret 的模型 artifact 写入仓库。

### Local profile registry

- 由验证服务本地加载，不受请求方控制；
- 固定 `q`、维度、阈值、矩阵和数值 profile；
- 将 `(profile_id, slot_id)` 映射为一个 `A_slot` 矩阵；
- 未知、重复、禁用或不完整 slot 一律拒绝。

### Requester

- 只能提交固定格式的 credential bytes；
- 可以任意修改 `profile_id`、`slot_id` 和 `b` 字节并重复查询；
- 不能直接提交 `A`、`q`、阈值、噪声 profile、量化 scale 或 gate；
- 在 A0 威胁模型中可以观察最终 allow/deny 对应的黑盒响应。

### Verifier and coordinator

- 规范解析器把 credential bytes 转换为精确有界整数；
- registry 解析出可信 `A_slot`；
- `V_ref` 或未来 `V_nn` 只产生 evidence；
- 唯一协调器提交 allow/deny，deny 时不得调用受保护业务模型。

## 3. Fixed toy profile

`A0-v1` 只定义一个 profile。参数没有密码安全含义，不能映射为 LWE 安全级别。

| Name | Value | Meaning |
| --- | --- | --- |
| `version` | `1` | wire-format version |
| `profile_id` | `1` | local profile selector |
| `n` | `32` | toy secret dimension |
| `m` | `8` | independent LWE components |
| `q` | `257` | modulus |
| `h` | `128` | encoding center for unlock bit 1 |
| `S` | `{0,1}^32` | toy secret domain |
| `A` domain | `Z_q^(8×32)` | local slot matrix domain |
| noise | centered binomial, `eta=4` | each error lies in `[-4,4]` |
| `B_issue` | `4` | maximum issuer noise magnitude |
| `T_ref` | `12` | exact reference acceptance radius |
| `T_nn` | `8` | future neural acceptance threshold |
| `epsilon_target` | `4` | future per-component total error bound |
| arithmetic dtype | signed integer, at least 32 bits | reference implementation should use `int64` |
| byte order | big-endian | all multi-byte wire fields |

Centered-binomial noise is defined component-wise as:

```text
e_i = sum_{k=1..4} (u_{i,k} - v_{i,k})
u_{i,k}, v_{i,k} in {0,1}
```

Therefore `-4 <= e_i <= 4` deterministically. This bounded distribution is selected for a finite correctness experiment, not as a production LWE recommendation.

For one matrix row, the unreduced dot product is bounded by:

```text
0 <= <A_i, s_test> <= 32 * 256 = 8192
```

All reference intermediates use signed `int64`; implementations must not rely on silent integer wraparound.

## 4. Runtime parameter generation

For each experiment run:

1. The harness obtains an explicit test seed from the experiment configuration.
2. It samples `s_test` uniformly from `{0,1}^32`.
3. For every enabled `slot_id`, it samples `A_slot` uniformly from `Z_q^(8×32)`.
4. It rejects and resamples any all-zero row of `A_slot`.
5. It stores `(profile_id, slot_id) -> A_slot` in an immutable, process-local registry.
6. It passes `s_test` only to the toy issuer and reference/neural verifier construction path.

The seed and `s_test` have no production secrecy value, but artifacts containing them are still treated as test keys: they may exist only in test memory or a test temporary directory and are not committed, logged or embedded in a distributed checkpoint.

`A_slot` need not be secret. Its integrity and non-client-controlled origin are required. Loading a registry entry with the wrong shape, an out-of-range coefficient or an all-zero row fails closed before verification.

## 5. Token issuance relation

For a positive unlock token, the trusted toy issuer samples `e in [-4,4]^8` from the fixed centered-binomial distribution and computes:

```text
b = A_slot * s_test + e + h * 1_m  (mod q)
```

For negative bit-zero test vectors it computes:

```text
b_zero = A_slot * s_test + e  (mod q)
```

All elements of `b` are emitted in the canonical range `[0, q-1]`.

The requester receives only `(version, profile_id, slot_id, b)`. The wire credential never contains `A`, `q`, thresholds, noise parameters, `s_test`, a claimed gate or a claimed authorization decision.

An honestly issued positive token satisfies the issuer-core bound in every component. This is only a correctness statement; possession, copying or replay of the byte string is sufficient to reuse it in A0.

## 6. Canonical wire encoding

Every A0 credential is exactly 23 bytes:

| Offset | Size | Field | Validation |
| --- | --- | --- | --- |
| `0` | 1 byte | `version` | must equal `1` exactly |
| `1` | 2 bytes | `profile_id` | unsigned big-endian; must equal `1` |
| `3` | 4 bytes | `slot_id` | unsigned big-endian; must resolve to one enabled local slot |
| `7` | 16 bytes | `b[0..7]` | eight unsigned big-endian 16-bit integers, each `<257` |

Parsing rules:

- input must be a byte string of exactly 23 bytes;
- no alternate JSON, text, float tensor or variable-length encoding is accepted;
- trailing bytes, prefixes, duplicate encodings and implicit coercions are rejected;
- `b_i >= q` is rejected rather than reduced modulo `q`;
- unknown version/profile/slot is rejected without fallback;
- a request containing an `A` field is structurally impossible in this encoding; any appended representation changes the length and is rejected;
- parser and registry failures do not call either verifier or business model.

The public response uses one fixed deny envelope for all parse, registry and verification failures. Detailed reason codes are test/internal audit data only.

## 7. Exact modular semantics

Canonical reduction returns a value in `[0,256]`:

```text
mod_q(z) = z mod 257
```

Centered reduction is defined without implementation-dependent negative-remainder behavior:

```text
center_q(z):
    r = mod_q(z)
    if r <= 128:
        return r
    return r - 257
```

Thus `center_q(z)` is always in `[-128,128]`.

For a parsed credential and locally resolved matrix:

```text
phase_i = mod_q(b_i - <A_slot[i], s_test>)
d_i = abs(center_q(phase_i - h))
D = max_i(d_i)
```

The exact reference regions are:

| Region | Exact condition | Meaning |
| --- | --- | --- |
| `ISSUER_CORE` | `D <= 4` | all honestly issued positive tokens must lie here |
| `REFERENCE_GUARD` | `5 <= D <= 12` | accepted by `V_ref`, but not required for future neural completeness |
| `REJECT` | `D >= 13` | rejected by `V_ref` |

The binary oracle is:

```text
V_ref(credential) = 1 iff parsing succeeds, registry lookup succeeds and D <= 12
```

This narrow relation is stricter than ordinary nearest-center bit decoding. It is chosen to reserve an explicit numerical proof margin, not to define a standard encryption scheme.

## 8. Future neural error contract

A1 must produce per-component estimates `d_hat_i` with a proved bound over the entire accepted input representation:

```text
abs(d_hat_i - d_i) <= epsilon_target = 4
```

The future neural decision is:

```text
V_nn = 1 iff every d_hat_i <= T_nn = 8
```

Any component with `d_hat_i > 8` rejects. Values in the diagnostic band `9..16` are explicitly ambiguous and fail closed; values above 16 also reject. The diagnostic distinction is never returned to the requester.

The selected constants establish the intended proof obligations:

### Issuer-core completeness

For an honestly issued token, `d_i <= 4`. If the A1 error contract holds, then `d_hat_i <= 8`, so every component passes.

### One-sided soundness preservation

If `V_nn = 1`, then every `d_hat_i <= 8`. Under the error contract, every exact `d_i <= 12`, so `V_ref = 1`.

Therefore the target theorem is:

```text
V_nn(credential) = 1 -> V_ref(credential) = 1
```

This theorem preserves the A0 exact relation only. It does not prove that an attacker cannot find an A0 credential accepted by `V_ref`.

## 9. Reference oracle pseudocode

```text
verify_ref(raw_credential, registry, s_test):
    parsed = parse_exact_23_bytes(raw_credential)
    if parsed is invalid:
        return evidence(PARSE_REJECT, accepted=false)

    profile = registry.lookup_profile(parsed.profile_id)
    slot = profile.lookup_enabled_slot(parsed.slot_id)
    if slot is invalid:
        return evidence(REGISTRY_REJECT, accepted=false)

    D = 0
    for i in 0..7:
        dot = 0
        for j in 0..31:
            dot = dot + int64(slot.A[i,j]) * int64(s_test[j])
        phase = mod_q(int64(parsed.b[i]) - dot)
        distance = abs(center_q(phase - 128))
        D = max(D, distance)

    if D <= 4:
        return evidence(ISSUER_CORE, accepted=true)
    if D <= 12:
        return evidence(REFERENCE_GUARD, accepted=true)
    return evidence(REJECT, accepted=false)
```

`V_ref` is an oracle for vector generation, differential testing and proof comparison. It is not part of the final research claim that verification is executed by a neural network. A deployed A1 experiment must not silently fall back to `V_ref` when `V_nn` fails.

The evidence object contains no capability. Exact distances and region detail are available only to tests/internal experiment traces; an external caller observes only the fixed allow/deny response produced by the coordinator.

## 10. Required test-vector families

For every generated slot, let `t_i = <A_slot[i],s_test> mod q`. The vector generator must cover:

| Family | Construction | Expected `V_ref` |
| --- | --- | --- |
| Core positive | `b_i=t_i+128+e_i`, every `e_i in [-4,4]` | accept, `ISSUER_CORE` |
| Guard boundary | one or more distances exactly `5` and `12` | accept, `REFERENCE_GUARD` |
| First reject | one distance exactly `13` | reject |
| Bit-zero | `b_i=t_i+e_i`, every `e_i in [-4,4]` | reject |
| Modular wrap | positive vectors whose canonical `b_i` crosses `0/256` | same result as mathematical relation |
| Mixed component | seven core components and one reject component | reject, proving AND aggregation |
| Non-canonical `b` | any encoded `b_i=257` | parse reject before verification |
| Unknown profile | `profile_id != 1` | registry reject |
| Unknown slot | absent or disabled `slot_id` | registry reject |
| Wrong length | every length from `0..22` and representative lengths `>23` | parse reject |
| Client-supplied `A` attempt | append or prepend matrix bytes | length/format reject |
| Invalid registry | wrong shape/range or an all-zero row | registry load failure |

Additional property tests must cover every exact distance `0..128`, every canonical `b_i` value `0..256`, each component position, repeated credentials and deterministic regeneration from a recorded non-secret experiment seed.

Future A1 differential tests compare `V_nn` and `V_ref` across all generated families. Tests must separately report false reject and false accept counts; a nonzero false accept violates the target soundness theorem.

## 11. Attack analysis and non-guarantees

### Client-chosen matrix and parameter downgrade

The wire format contains no `A`, `q`, thresholds or profile parameters. `A=0, b=h` cannot directly select a zero matrix because `A` comes from the local registry, and registry generation/load rejects all-zero rows. Unknown profiles do not trigger a weaker fallback.

This is a structural input-integrity property, not a cryptographic proof.

### Random forgery

For fixed `A_slot` and `s_test`, a uniformly random canonical `b_i` lands within the exact radius 12 with probability `25/257` per component. With eight independently selected components, the random-vector acceptance probability is `(25/257)^8`, which is below `10^-8`.

This calculation is only a sanity metric. It is not a security parameter or an unforgeability bound.

### Adaptive chosen-b oracle

The requester can submit arbitrary `b` and observe black-box allow/deny. Because `q`, `m` and thresholds are tiny, exhaustive or adaptive search may find an accepted vector. Rate limiting cannot turn A0 into a cryptographically secure authentication scheme. A0 explicitly provides no chosen-ciphertext or chosen-query security.

### Replay and input substitution

The credential contains no nonce, time, identity, model ID, scope or input hash. A copied valid byte string can be replayed and used with a different business input. These properties are intentionally deferred to A3.

### Secret and white-box exposure

The experiment verifier uses `s_test` in its computation. Anyone who reads the model/parameters or controls the runtime may recover or bypass it. A0 assumes a trusted black-box runtime and provides no white-box guarantee.

### Public-key authentication

A0 has no public encryption or public signing API. The trusted toy issuer uses `s_test` to create vectors. This avoids claiming that public encryption of an unlock bit authenticates the requester, but it still does not create a secure MAC or signature scheme.

## 12. Coordinator behavior

- Parse/registry reject and `V_ref/V_nn` reject all map to the same external deny envelope.
- Only the coordinator may convert accepted evidence into a one-request model capability.
- Deny must produce zero calls to the protected LeNet/MLP and release no logits or intermediate features.
- The requester cannot pass a precomputed evidence object or gate.
- A0 replay is allowed by the protocol but each individual request still receives a fresh coordinator decision.
- No implementation may use `V_nn OR V_ref` as a fallback route.

## 13. Artifact and logging policy

The following must not be committed or logged:

- `s_test` or any seed treated as secret;
- serialized verifier weights containing `s_test`;
- issued credential collections intended to remain private;
- model checkpoints, local databases or raw experiment dumps.

Reproducible public tests should regenerate non-secret toy fixtures inside the test temporary directory. Reports may contain profile constants, aggregate results, code commit identifiers and hashes of generated artifacts, but not secret-bearing artifacts themselves.

## 14. Acceptance criteria

A0 protocol specification is complete when all of the following are true:

1. The only accepted wire format is the 23-byte `A0-v1` encoding.
2. `q`, dimensions, thresholds, error distribution and arithmetic semantics are fixed above.
3. The client cannot submit or replace `A`, `q`, thresholds or the decision.
4. Issuance and verification relations are unambiguous and use `int64` intermediates.
5. Core, guard, reject, wrap, malformed and chosen-parameter test families are defined.
6. The future A1 completeness and one-sided soundness inequalities follow from the selected constants.
7. chosen-`b`, replay, input substitution and white-box exposure remain explicitly unsupported.
8. A0 is consistently labeled numerical unlock rather than authentication or signature verification.

## 15. Deferred decisions

The following are deliberately outside A0 and remain for later milestones:

- the exact neural construction for modulo, absolute distance, threshold and eight-way AND;
- the permitted neural operator set and whether periodic activations are used;
- the formal method used to establish `epsilon_target <= 4`;
- production-sized LWE parameters or any claimed LWE security level;
- challenge-response, message binding and replay state;
- a reviewed public-key lattice signature relation;
- PyTorch version, quantization backend and target device.
