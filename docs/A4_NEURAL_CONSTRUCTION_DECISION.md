# A4 Fixed ReLU Construction Decision

## 1. Status and claim boundary

本文档冻结 A4/V1-prep canonical `(y,z)` 代数谓词的首个固定神经构造，决定编号为 `A4-C1`，
候选标识为 `CAN-RELU-A4-PFDH-TOY-v1`。这是非生产、单 profile、固定小参数的 exact-integer
conformance 构造；它不提供 GPV/ISIS 不可伪造性、身份认证、白盒完整性或生产安全。

本决定及当前 dependency-free implementation 回答如何把已经冻结的 A4 relation 编译为固定
affine/ReLU graph。A3 message/parser、
SHAKE256 hash-to-syndrome、proof decoding、evidence、nonce store 和 coordinator 都在 neural core
之外。当前 V1-P2 已选择 FSwA-S Module-SIS relation，但其 negacyclic polynomial convolution、
module dimensions、larger modulus 与 runtime challenge polynomial 仍必须另行定义 relation 和 neural construction，
不能把本决定直接当作 V1 verifier。

## 2. Fixed canonical domain

Neural core 输入是一个固定顺序的 80 分量整数向量：

```text
input = y[0..7] || z[0..71]
```

其中：

| Field | Shape | Exact domain | Meaning |
| --- | ---: | --- | --- |
| `y` | 8 | `[0,256]` | trusted hash-to-syndrome output |
| `z` | 72 | `[-128,127]` | parsed signed-int8 proof coefficients |
| `A` | 8 x 72 | `[0,256]` | copied, frozen local public profile |
| `q` | 1 | `257` | trusted fixed modulus |

The parser supplies exact signed `int32` values with scale `1` and zero-point `0`. Floating-point, bool,
implicit casts, non-canonical modulo representations and inputs outside these ranges never enter the core.

## 3. Exact reference predicate

For each row `i`, define the unreduced integer residual:

```text
d_i = sum_j A[i][j] * z[j] - y[i]
```

The A4 reference relation is:

```text
V_ref(y,z) = 1 iff
  every z[j] is in {-1,0,1}
  and every d_i is divisible by 257
```

The first condition is exactly `||z||_inf <= 1`; the second is equivalent to
`A*z mod 257 == y`. The neural construction proves equality to this predicate over the complete canonical
integer domain, not only over valid proofs or random samples.

## 4. Construction decision

Let `rho(a) = max(0,a)`. The selected graph has three affine+ReLU blocks:

```text
80 -> 3600 -> 1153 -> 1
```

All affine maps are fixed at profile compilation. All reductions use exact signed `int64`; weights, biases and
stored activations use signed `int32` whenever the range certificate permits the narrowing. There is no runtime
`%`, Floor, `abs`, comparison, data-dependent branch, float conversion, Sigmoid, MASK or reference fallback.

The point-pulse construction is intentionally width-heavy. It makes the all-input proof local and explicit; a
smaller sawtooth/modulo construction is deferred until a separate decision proves its range and boundary behavior.

## 5. Layer 1: norm violations and residual hinges

For each coefficient `z[j]`, create two norm-violation units:

```text
u_j_plus  = rho(z[j] - 1)
u_j_minus = rho(-z[j] - 1)
```

Their sum is zero exactly when `z[j]` is in `{-1,0,1}`. Across all 72 coefficients this creates 144 units.

For each row residual `d_i` and every integer

```text
K = {-72, -71, ..., 70, 71}
```

create three hinge units:

```text
h_i,k_plus  = rho(d_i - k*q + 1)
h_i,k_zero  = rho(d_i - k*q)
h_i,k_minus = rho(d_i - k*q - 1)
```

There are 144 possible multiples per row and 8 rows, hence `8 * 144 * 3 = 3456` residual hinge units. The
first layer width is `144 + 3456 = 3600`.

## 6. Layer 2: exact point pulses and norm accumulator

For every residual hinge triple compute:

```text
p_i,k = rho(h_i,k_plus - 2*h_i,k_zero + h_i,k_minus)
```

For an exact integer `d_i`, `p_i,k` is `1` exactly when `d_i = k*q`, and `0` otherwise. This is the discrete
second difference of a ReLU hinge and is a non-negative triangular pulse on integer inputs.

In parallel compute one norm accumulator:

```text
v = rho(sum_j (u_j_plus + u_j_minus))
```

Layer 2 therefore has `8 * 144 = 1152` pulse units plus one norm unit, for width `1153`.

## 7. Layer 3: final fail-closed conjunction

The output bit is:

```text
g = rho(sum_i sum_k p_i,k - v - 7)
```

For canonical integer inputs, each row contributes at most one pulse, so the pulse sum is in `[0,8]`.
The output is therefore always exactly `0` or `1`:

- all eight equations true and `v=0`: `g = rho(8 - 0 - 7) = 1`;
- any equation false: pulse sum at most `7`, so `g=0`;
- any norm violation: `v >= 1`, so `g=0` even if all equation pulses are present.

The core output is evidence input for the coordinator; it is not itself an authorization decision.

## 8. Multiple coverage and range ledger

For any canonical `z` before the norm check, the residual bounds are:

```text
-2,359,552 <= d_i <= 2,340,864
```

For norm-valid `z`, the tighter bounds are:

```text
-18,688 <= d_i <= 18,432
```

The multiples covered by `K` are `-18,504` through `18,247`; these are exactly all multiples of 257 in the
norm-valid interval. The next multiples are outside that interval. Thus a valid modular equality always has a
corresponding pulse, while invalid norm inputs are rejected by `v` regardless of pulse coverage.

The largest first-layer residual hinge activation is bounded by `2,359,369`. Norm violations sum to at most
`72 * 127 = 9,144`. Layer 2 pulse outputs are in `{0,1}`, and the final preactivation is in `[-9,151,1]`.
All values fit signed `int32` storage after exact `int64` reductions; no saturation or wraparound is used.

## 9. Topology and parameter accounting

The semantic dense tensor shapes are:

| Layer | Weight shape | Bias shape | ReLU output |
| --- | --- | --- | ---: |
| 1 | `(3600,80)` | `(3600,)` | 3600 |
| 2 | `(1153,3600)` | `(1153,)` | 1153 |
| 3 | `(1,1153)` | `(1,)` | 1 |

This is `4,439,953` weight slots and `4,754` bias slots, or `4,444,707` scalar slots before storage
compression. The compiled graph is structurally sparse: at most `257,185` weight entries are non-zero under
the public profile. A sparse evaluator may exploit that structure only if it preserves the same affine/ReLU
semantics and receives a separate backend/conformance check.

## 10. Completeness proof

For every canonical integer input accepted by `V_ref`, each `z[j]` is in `{-1,0,1}`, hence `v=0`. Each
`d_i` is a multiple of 257. The range lemma places that multiple in `K`, so exactly one `p_i,k` is `1` for
each row. The final preactivation is `1`, and `g=1`. This proves completeness for the complete canonical
accepted set.

## 11. Soundness-preservation proof

Assume `g=1`. Because all pulse terms are non-negative integer bits and each row has at most one pulse, the
pulse sum is at most 8. The final ReLU can be positive only when the pulse sum is 8 and `v=0`. Therefore every
row has a pulse, so each `d_i` is a covered multiple of 257, and every coefficient has no norm violation. Thus
`V_ref(y,z)=1`.

Consequently the construction establishes the stronger in-domain equality:

```text
for every canonical (y,z): V_nn(y,z) == V_ref(y,z)
```

The proof depends on exact integer inputs and fixed profile compilation. It does not cover arbitrary floating
inputs, modified weights, model code changes, white-box tampering, or a later noisy LWE/Module-LWE protocol.

## 12. Allowed preprocessing and evidence boundary

Trusted preprocessing may parse the A4 proof, validate the A3 message, compute SHAKE256 hash-to-syndrome and
produce canonical `(y,z)`. The neural core receives no salt, raw message bytes, profile selector or public matrix
input. It returns only an exact bit/trace that the local adapter converts into evidence. The coordinator remains
the only permission submission point and atomically consumes the external nonce.

## 13. Required construction tests

The current implementation checkpoint includes:

- exact signed-int8 domain and `(y,z)` layout rejection;
- scalar pulse lemma over every representable residual integer in the proved range;
- scalar norm-violation lemma over all signed-int8 values;
- all `K` endpoints and the first outside multiples;
- relation differential tests against `verify_a4_ref` for valid, norm-invalid and equation-invalid vectors;
- graph topology, int32 storage, int64 reduction and output-bit assertions;
- profile immutability, no `%`/Floor/abs/compare/fallback and no-authority AST checks;
- A3 invalid-proof retry, atomic consume and replay zero-extra-call tests;
- full project unit, integration, security, lint, format, mypy, pip and governance checks.

Finite scalar/exhaustive tests support the proof but do not replace the algebraic all-input argument above.

## 14. Deferred alternatives

The following remain outside this decision:

- sawtooth/folded modulo constructions with a smaller topology;
- exact PyTorch CPU backend or any qint8/CUDA/export implementation;
- signer/keygen, production parameters and security estimation;
- a concrete V1 challenge-response protocol with noisy LWE, Module-LWE, hints or rejection sampling;
- ML-DSA standard vectors and complete neural hash/encoding implementation.

## 15. Acceptance criteria

This construction decision is accepted when:

1. The graph, canonical input order, fixed `K`, ranges and integer semantics are frozen.
2. Completeness and `V_nn=1 -> V_ref=1` are established for all canonical integer inputs by the lemmas above.
3. The implementation reproduces the graph without relation fallback or client-selected parameters.
4. Focused positive/negative, boundary, type, A3 composition and no-authority tests pass.
5. Toy/non-production/V1-prep claim limits remain explicit in the worklog, research and security documents.
