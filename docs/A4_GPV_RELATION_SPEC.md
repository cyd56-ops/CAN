# A4 GPV Public-Verification Relation Specification

## 1. Status and claim boundary

本文档固定阶段 A4 的首个公钥格关系规格，profile 名称为
`A4-GPV-PFDH-TOY-v1`。它采用 Gentry、Peikert、Vaikuntanathan 在 STOC 2008 给出的
probabilistic full-domain-hash (PFDH) 短原像签名范式：公开验证检查签名向量属于固定短向量域，
且公开函数输出等于消息与 salt 的 full-domain hash。

A4 是长期 `V1-prep` 的代数编译基线，不是完整 V1 身份认证协议。它先隔离 canonical `(y,z)`
关系的精确语义和 neural soundness；当前 V1-P2 现已选择 reviewed FSwA-S 交互式 Module-SIS Sigma
protocol，并须独立证明 M-LWE/M-SIS、transcript/challenge/rejection 安全。V1 将新增独立实现，不改写或替换本 A4/V1-prep
reference、neural graph 和测试。

本 profile 使用小维度、非安全参数和项目自有的确定性 SHAKE256 hash-to-syndrome 编码，只用于：

- 冻结 A3 的 exact message/proof 公钥验证关系；
- 实现不持有私钥的精确 reference adapter；
- 为后续固定神经 verifier 定义 canonical tensor、整数范围和证明义务。

它不是 GPV 论文的具体安全参数实例，不继承论文的 SUF-CMA 归约，不是 ML-DSA/FIPS 204 实现，
也不得描述为生产认证。只有完整安全参数、key generation/signing/hash 实例化、密码分析和组合证明
另行闭合后，才能提出相应不可伪造性主张。

## 2. Relation selection and reviewed sources

GPV PFDH 被选择为首个 A4 关系，原因是其公开验证谓词可以明确分解为：

```text
canonical message and salt
-> full-domain hash y in Z_q^n
-> canonical short vector z
-> public matrix product A*z mod q
-> exact equality and norm conjunction
-> evidence only
```

该分解只包含确定性 hash、公开整数线性算术、规范模约减、短向量检查和逻辑 AND，适合作为神经
编译研究的第一关系。完整 ML-DSA 还需要分别处理标准编码、多个 hash、NTT、多项式运算、范数和
hint 检查，因此按项目既有范围只保留为后期标准比较目标。

规范来源：

- Craig Gentry, Chris Peikert, Vinod Vaikuntanathan, “Trapdoors for Hard Lattices and New
  Cryptographic Constructions,” STOC 2008, DOI `10.1145/1374376.1374407`；作者版本：
  `https://www.mit.edu/~vinodv/papers/trapcvp.pdf`。第 5.2 节定义 PFDH：签名为 salt 与短原像，
  验证检查 domain membership 和公开函数等式。
- NIST FIPS 204, “Module-Lattice-Based Digital Signature Standard,”
  `https://doi.org/10.6028/NIST.FIPS.204`。本项目仅用它固定 ML-DSA 的标准比较边界，不实现或声称
  兼容该标准。

## 3. Roles and trust assumptions

### Requester

- 只能通过 A3 response 提交精确 133 字节 message 与精确 A4 proof bytes；
- 可以任意构造、篡改、复制和 replay proof；
- 不能提交或替换公开矩阵、profile、模数、维度、norm bound、hash 算法或 verifier；
- 不能提交 evidence、allow/deny 或 capability。

### Local identity registry

- 将 A3 的 32 字节 `identity_id` 绑定到一个唯一、启用且不可变的 A4 public profile；
- 在构造时一次性校验公开矩阵 shape、范围和模 `q` 满行秩；
- profile 构造后不在每次请求中重复扫描可信矩阵；
- 不保存私钥、trapdoor、签名随机性或 signer 接口。

### Reference verifier and A3 adapter

- reference verifier 只使用公开 profile、canonical message 和 proof；
- reference evidence 不具有权限；A3 adapter 只把 relation 结果映射为与 exact message digest 和
  identity 绑定的 `A3Evidence`；
- verifier 不读取或修改 nonce 状态，不调用模型，不提交权限；
- 唯一 A3 coordinator 仍负责 freshness、原子 consume 和 protected-model decision。

## 4. Fixed toy conformance profile

`A4-GPV-PFDH-TOY-v1` 固定以下参数。它们没有密码安全级别含义。

| Name | Value | Meaning |
| --- | ---: | --- |
| profile version | `1` | local relation and proof version |
| `profile_id` | `1` | local trusted profile identifier |
| `q` | `257` | prime modulus |
| `n` | `8` | syndrome and public-matrix row count |
| `m` | `72` | short-vector and public-matrix column count |
| coefficient encoding | signed 8-bit | proof vector representation |
| `beta_inf` | `1` | exact accepted infinity-norm bound |
| salt length | `32` bytes | PFDH salt |
| message length | `133` bytes | exact A3-v1 message |
| proof length | `105` bytes | version, salt and 72 signed coefficients |
| arithmetic | signed exact integer | reference intermediates fit signed 32-bit |

For every encoded coefficient, `-128 <= z_j <= 127`. Before modular equality is considered, the relation requires
`max_j(abs(z_j)) <= 1`. The largest representable unreduced row sum is bounded by
`72 * 256 * 128 = 2,359,296`; accepted vectors have the tighter bound `72 * 256 = 18,432`.

The choice `m = 8 * 9` permits simple, explicitly weak gadget-matrix conformance fixtures that represent each
syndrome coefficient with nine public binary columns. Such fixtures require no private key and are intentionally
forgeable; they test the verifier relation only and must never be used as evidence of signature security.

## 5. Canonical public profile

One local profile contains exactly:

```text
profile_id = 1
identity_id = 32 exact bytes
A in Z_257^(8 x 72)
```

Every `A[i][j]` is an exact built-in integer in `[0,256]`; `bool`, floats, alternate residue encodings and implicit
coercions are rejected. The matrix must have exactly 8 rows and 72 columns and rank 8 over `Z_257`. Full row rank
is a configuration sanity condition, not a concrete security test.

The implementation copies all rows into immutable tuples at construction. Client bytes contain no matrix,
`profile_id`, `q`, dimension, norm bound or key selector. A3 chooses the profile only from the locally trusted
`identity_id` registry.

For reproducible audit, the public-key digest is:

```text
SHA-256(
    b"CAN-A4-GPV-PK-v1\x00"
    || u16_be(profile_id)
    || u16_be(q)
    || u16_be(n)
    || u16_be(m)
    || u8(beta_inf)
    || row-major u16_be(A[i][j])
)
```

The digest is public metadata and has no authorization meaning.

## 6. Canonical message and hash-to-syndrome

The A4 message is exactly the 133 bytes defined by `docs/A3_CHALLENGE_RESPONSE_SPEC.md`. The standalone A4
boundary requires the exact A3 domain, version, model, identity, scope and 60,000 ms TTL encoding; nonce and input
digest remain opaque signed bytes. The message `identity_id` must equal the local profile identity.

The PFDH target `y in Z_257^8` is derived from the exact message and proof salt:

```text
xof_input =
    b"CAN-A4-GPV-PFDH-H2S-v1\x00"
    || u16_be(profile_id)
    || message[133]
    || salt[32]

stream = SHAKE256(xof_input)
```

Read the XOF stream as consecutive unsigned big-endian 16-bit candidates. Reject only candidate `65535`; for every
other candidate append `candidate mod 257`. Stop after eight coefficients. Since `65535 = 255 * 257`, each accepted
candidate maps uniformly to `[0,256]` under the ideal-XOF model. There is no modulo-bias fallback, finite candidate
limit or client-selected hash.

This concrete SHAKE256 mapping is a deterministic research instantiation, not the random oracle assumed by the
GPV proof. Hash-to-syndrome is trusted canonical preprocessing in A4-v1. Until a later decision compiles it into the
network, claims about the neural core must say that the core verifies the algebraic GPV predicate over canonical
`(y,z)`, not that SHAKE256 itself is executed by neural layers.

## 7. Exact 105-byte proof encoding

The only A4 proof encoding is:

```text
u8(version = 1)
|| salt[32]
|| z[0..71] as signed two's-complement int8
```

| Offset | Size | Field | Validation |
| ---: | ---: | --- | --- |
| `0` | 1 | version | exactly `1` |
| `1` | 32 | salt | opaque exact bytes |
| `33` | 72 | `z` | each byte decoded as one canonical signed int8 |

The total length is exactly 105 bytes. Prefixes, suffixes, shorter forms, text, JSON, variable-width integers,
unsigned reinterpretation, alternate negative encodings and implicit conversion reject. The proof contains no
profile, matrix, public key, algorithm selector, claimed syndrome, norm, decision or evidence.

All signed int8 encodings parse canonically. A vector coefficient outside `[-1,1]` is therefore a well-formed proof
that fails the relation's norm check, rather than an alternate parser representation.

## 8. Exact public-verification relation

For parsed proof `(salt,z)`, trusted profile `A`, and canonical message `M`:

```text
y = hash_to_syndrome(profile_id, M, salt)
norm_ok = max_j(abs(z_j)) <= 1
syndrome_i = mod_257(sum_j A[i][j] * z[j])
equation_ok = for every i in 0..7: syndrome_i == y_i

V_ref(profile, M, proof) = 1 iff norm_ok and equation_ok
```

The implementation evaluates `norm_ok` before the matrix relation and uses exact Python integers with canonical
non-negative modulo output. It never reduces proof coefficients modulo `q`, clips a long vector, accepts an
equivalent variable-length encoding or falls back to A0/A1/another signature verifier.

Stable internal evidence codes distinguish message, proof-parse, norm, equation and configuration rejection plus
relation acceptance. A3 maps all non-accept codes to its fixed deny path; external callers never receive the
detailed A4 code.

## 9. Evidence-only A3 adapter contract

The local adapter binds exactly one immutable public profile and is callable only as:

```text
adapter(exact_message_bytes, exact_proof_bytes) -> A3Evidence
```

On exact relation acceptance it produces A3's exact accept evidence containing the local `identity_id`,
`profile_id` and `SHA-256(message)`. Every other result produces proof/config rejection evidence with the same
binding fields. The evidence contains no matrix, vector, salt, syndrome, proof, nonce-store handle, decision,
model, gate or capability.

Constructing the adapter requires no private key and exposes no signer, key generation or proof-generation API.
The A3 coordinator independently checks evidence type, code, identity, profile and message digest before atomic
nonce consume.

## 10. Neural verifier reference contract

`docs/A4_NEURAL_CONSTRUCTION_DECISION.md` now freezes decision `A4-C1`, candidate
`CAN-RELU-A4-PFDH-TOY-v1`. Its canonical core inputs are:

```text
z: shape (72,), signed int32, scale 1, each value in [-128,127]
y: shape (8,), signed int32, scale 1, each value in [0,256]
compiled A: shape (8,72), signed int32, scale 1, each value in [0,256]
```

`A` is trusted, frozen and folded into weights where the selected construction permits. Parser, A3 message checks,
SHAKE256 hash-to-syndrome, evidence construction, nonce state and authorization remain outside the neural core.

The fixed graph is `80 -> 3600 -> 1153 -> 1` affine+ReLU blocks. It uses 144 norm-violation hinges and
3456 residual hinges for the fixed multiple set `K={-72,...,71}`, then 1152 exact point pulses, one norm
accumulator and a final `rho(sum(p)-v-7)` conjunction. The construction decision gives the complete range
ledger and proof that this output equals the exact predicate over all canonical integer inputs.

The implemented dependency-free backend preserves exact semantics for:

- 72-way `abs(z_j) <= 1` conjunction;
- eight public affine row sums and canonical modulo-257 equality with `y`;
- the final conjunction of norm and all eight equations;
- every accumulator, rounding, saturation and representation range.

Let `D` be all exact 105-byte proofs paired with a canonical message and trusted profile, and let `E` be every
representable canonical `(y,z)` tensor after preprocessing. Required obligations are:

```text
Completeness: for all a in D, V_ref(a) = 1 -> V_nn(a) = 1
Soundness:    for all e in E, V_nn(e) = 1 -> V_ref(canonical(e)) = 1
```

Finite random tests cannot establish the soundness inclusion. The construction must provide a complete finite-domain
argument, machine-checkable proof, or a rigorously bounded decomposition whose inequalities cover every input.
Diagnostic ambiguity always rejects.

## 11. Security analysis and non-guarantees

### Client-selected key or profile

The proof has no key/profile field. Appending a matrix or selector changes the fixed length and rejects. Identity is
inside the signed A3 message and resolves only to local trusted configuration.

### Message, salt and vector tamper

Changing any canonical message or salt byte changes `y` except with a hash collision; changing `z` changes the
public equation or norm. The exact verifier recomputes all values and never trusts a claimed syndrome.

### Replay and substitution

The signature relation alone is stateless. A3 binds model, identity, scope, time, nonce and input digest and performs
atomic consume. A valid A4 proof without the matching pending A3 record cannot authorize a protected call.

### Toy forgery and concrete security

The fixed dimensions and norm bound are not selected by a lattice security estimator. Test gadget matrices make
valid vectors publicly computable by design. Full row rank does not make such a key secure. Therefore successful
verification proves only membership in this exact public relation, not possession of a production signing key.

The project does not implement GPV trapdoor generation, Gaussian preimage sampling, signing, key rotation,
revocation or real-key lifecycle. It does not claim the concrete SHAKE256 mapping satisfies the paper's random-oracle
proof assumptions. ML-DSA compatibility, white-box integrity, side channels, distributed replay state and production
authentication remain unsupported.

## 12. Required test families

- exact proof round trip and fixed hash-to-syndrome vectors;
- public profile immutability, matrix shape/range/type and mod-257 full-rank checks;
- relation accept using an explicitly weak, no-private-key gadget fixture;
- first norm reject at `abs(z_j)=2`, signed-int8 endpoints and mixed vectors;
- equation reject for every message/salt/vector component tamper family;
- wrong message/proof type, length, version, identity, domain, model, scope and TTL;
- appended/prepended key/profile/decision bytes and type confusion;
- evidence fields contain no authority, proof, salt, syndrome or private material;
- A3 integration: invalid proof does not consume, exact accept consumes once, replay and tamper produce zero extra
  protected-model calls;
- instrumentation/AST checks that A4 has no signer, private key, trapdoor, A0/A1 fallback or protected-model call.

All deterministic fixtures use explicit non-secret constants and may exist only in test memory or pytest temporary
directories. Random forgery sampling is optional experimental evidence and cannot replace an unforgeability proof.

## 13. Artifact and logging policy

Public matrices and their digest may be stored as local trusted configuration, but this checkpoint keeps fixtures in
source tests and does not serialize a runtime registry. No private key, trapdoor, Gaussian sampler state, issued proof
collection, model checkpoint or raw audit dump may be committed or logged.

Internal audit may record bounded result codes, profile ID and dedicated/truncated public digests. It must not record
raw proof, salt, full message, nonce, input, signature vector or replayable evidence.

## 14. Implementation checkpoint sequence

1. This checkpoint freezes the reviewed relation mapping, exact profile/encoding and neural proof obligations.
2. Implement the dependency-free public profile, parser, hash-to-syndrome, exact `V_ref` and A3 evidence adapter.
3. Add focused unit/integration/security tests, then run the complete quality suite.
4. `A4-C1` freezes the fixed neural construction and proof method for the canonical `(y,z)` relation.
5. The dependency-free A4-C1 evaluator, neural evidence adapter and A3 composition tests close V1-prep.
6. V1-P2 has selected a concrete reviewed interactive Module-SIS protocol; its security remains a separate
   obligation, and its polynomial module relation requires a new coefficient-domain exact/neural construction.
7. The next checkpoint adds separate V1-P2 exact-reference and A3-v2 modules without modifying A4-C1 or V0 code.
8. V2 may later evaluate larger parameters or a standardized ML-DSA comparison, but it is not a prerequisite for
   the current A4/V1-prep checkpoint.

Signer/key generation, production parameters, Stage B capability/tool gateway, qint8/CUDA/export, MASK and shared
head/trunk are excluded from this sequence.

## 15. Acceptance criteria

This specification/reference checkpoint is complete only when:

1. GPV PFDH is identified as the reviewed source relation and its claim boundary is explicit.
2. The only proof format is the exact 105-byte encoding.
3. Message hashing, signed coefficient decoding, norm and modular-equation semantics are unambiguous.
4. Public configuration is immutable, full-rank checked once and never client-selected.
5. The reference verifier and A3 adapter contain no private key or signing API and return evidence only.
6. Positive, negative, tamper, type-confusion and no-authority tests pass.
7. Neural completeness/soundness obligations cover all representable canonical inputs.
8. Toy parameters, weak conformance fixtures, random-oracle gap and non-production limitations remain explicit.
9. No A0/A1 fallback, Stage B expansion or complete ML-DSA claim is introduced.
