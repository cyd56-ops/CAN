# A2-E2 Public/Protected Capability Experiment Specification

## 1. Status and claim boundary

This document freezes the design of the A2-E2 capability-tier experiment. The independent ungated public
model baseline and locally bound three-state coordinator have now been implemented. The complete evaluation
runner accepts only already materialized models whose state digests equal the accepted baselines and never
trains either model. The project owner has now explicitly permitted a separate deterministic materializer to
recreate the accepted states under the frozen baseline protocol; that materialization and the complete
10,000-image three-state report have now passed, while the evaluator remains no-training. A2-E2 does not change the completed
A2-E1 binary gate, issue a bearer capability, or add security-bearing cryptography.

A2-E2 studies whether one trusted coordinator can expose an independently defined public model function
without invoking or leaking the protected A2-E1 model path. The word `capability` in this experiment means
an internal, per-request coordinator decision. It is not an identity credential, authorization token or Stage B
tool capability and cannot be stored, transferred or replayed as authority.

The supported claim is limited to the configured black-box entries, selected CPU runtime and explicit call and
output instrumentation. It does not establish authentication, unforgeability, request binding, white-box
non-bypass, endpoint authorization, model confidentiality or production access control.

## 2. Decision summary

A2-E2 selects an **independent public model** as the primary tier implementation. It rejects a shared head,
shared trunk, shallow/deep early exit and MASK as the primary experiment because each would share protected
computation or features and weaken the auditable zero-protected-call boundary.

The experiment has exactly three committed decisions:

| Decision | Trusted entry binding | Required computation | External result |
| --- | --- | --- | --- |
| `DENY` | any | no model call | fixed deny envelope |
| `PUBLIC` | enabled public entry | independent public model only | coarse two-class result |
| `PROTECTED` | protected entry plus exact A1 accept evidence | protected A2-E1 MLP only | protected ten-class result |

The protected entry preserves A2-E1 exactly: every parse, profile, configuration, verifier, numeric or internal
failure commits `DENY`. It can never turn a failed verification into `PUBLIC`. The public entry is a separate
locally bound service entry and is absent by default; it does not accept or evaluate a credential.

## 3. Trusted configuration and entry binding

Local trusted configuration fixes all of the following before accepting requests:

- experiment version and response schema;
- whether the public entry exists, with the default set to disabled;
- the public entry binding and independent public model identity;
- the protected entry binding, A1-B1 verifier, A2-E1 protected model and registry;
- the only coordinator implementation and its model invocation adapters;
- CPU backend, dtype, data preprocessing, output labels and audit schema.

The untrusted request schema contains no policy, entry kind, capability, model/head ID, backend, device,
profile override, evidence, decision, route or fallback field. A trusted listener or test harness binds a request
to one entry before parsing its payload. A client may invoke an exposed public service, but no field in a public
or protected payload can relabel that entry or select its implementation.

Enabling the public entry requires an explicit local configuration change and a startup audit event. Missing,
unknown, duplicate, dynamically changed or internally inconsistent configuration disables the public entry and
fails protected requests closed. There is no automatic public mode based on verifier result, exception,
latency, load or model availability.

## 4. Public functional and output scope

The public function is a coarse Fashion-MNIST classifier with exactly two semantic classes:

| Public class | Source Fashion-MNIST labels |
| --- | --- |
| `0` (`NON_FOOTWEAR`) | `0, 1, 2, 3, 4, 6, 8` |
| `1` (`FOOTWEAR`) | `5, 7, 9` |

This mapping is a fixed experiment label transformation, not an authorization policy. The public response may
release only the coarse class ID. It must not release the original ten-class label, logits, probabilities,
confidence, embeddings, intermediate activations, verifier evidence, detailed rejection reason or model
identifier.

The public model is `CAN-A2-FMNIST-PUBLIC-MLP-v1`, a float32 CPU MLP with topology
`784 -> 64 -> 2` and 50,370 trainable parameters. It consumes the same canonical image tensor contract as
A2-E1 so input effects can be compared, but it has an independent constructor, initialization, optimizer
state, trained parameters, state digest and report. No public parameter may be derived from or copied out of
the protected model.

The model predicts a meaningful but deliberately coarser task. Coarseness alone is not the isolation claim;
the required isolation comes from independent computation and the output contract.

## 5. Protected functional and output scope

The protected function remains `CAN-A2-FMNIST-MLP-v1` from
`docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`: `784 -> 256 -> 128 -> 10`, float32 CPU, returning only a
top-1 class ID in `[0,9]` after the coordinator commits `PROTECTED`.

A2-E2 does not retrain, tune, wrap with a shared head, change the input contract or alter accepted A2-E1
predictions. A protected run over the canonical 10,000-image test set must remain label-for-label identical to
the accepted A2-E1 gate result and retain its prediction and model-state digests.

Protected logits, probabilities, confidence and features remain protected assets. They must never be computed
on the public path and must not appear in any public, protected or deny envelope.

## 6. Single-coordinator three-state semantics

Exactly one coordinator is the commit point for all three decisions. Parsers, verifiers and models cannot
construct a decision or external response.

The normative protected flow is:

```text
trusted PROTECTED entry binding + untrusted image + raw credential
-> strict image and credential parsing
-> fixed local A1-B1 verifier
-> evidence without authority
-> one coordinator commit
   NUMERIC_ACCEPT -> PROTECTED -> exactly one protected-model call
   anything else  -> DENY      -> zero public/protected-model calls
```

The normative public flow is:

```text
trusted enabled PUBLIC entry binding + untrusted image
-> strict image parsing
-> one coordinator commit
   valid local policy -> PUBLIC -> exactly one independent public-model call
   anything else      -> DENY   -> zero public/protected-model calls
```

The coordinator must commit once at most. `PUBLIC` and `PROTECTED` are mutually exclusive. A model exception,
invalid model output or response-construction error terminates in the fixed deny response; it never invokes the
other model. Instrumentation may record that the selected model was entered, but must not describe such an
error as a zero-selected-model-call path.

## 7. Non-upgrade and non-reuse rules

A public response is data only. It is not accepted by the protected credential parser, coordinator, verifier or
model adapter as evidence, a decision, a context or a capability. Its bytes, fields and coarse class ID cannot
be replayed, relabelled or embedded to reach the protected branch.

The following transitions are forbidden:

- `PUBLIC -> PROTECTED` without a new protected-entry request and exact A1 acceptance;
- `DENY -> PUBLIC` because verification or protected computation failed;
- `PUBLIC -> PUBLIC` by replaying a response as authority rather than making a new public request;
- changing entry binding, policy, model, head, backend or output schema through request data;
- treating a public model result as verifier evidence or a coordinator decision.

The experiment creates no cross-request authorization state and no reusable token. A0 credentials remain
replayable on the protected entry as an explicitly unsupported A0 property; A2-E2 neither fixes nor worsens
that protocol fact.

## 8. Response envelopes

A2-E2 uses a new fixed response version so it does not silently change A2-E1 envelopes:

```text
DENY      = {"version": 2, "status": "deny"}
PUBLIC    = {"version": 2, "status": "public", "coarse_class_id": 0|1}
PROTECTED = {"version": 2, "status": "protected", "class_id": 0..9}
```

Responses use exact field sets and exact built-in types. Unknown, missing or additional fields are not emitted.
No response includes credential bytes, entry binding, policy, evidence, verifier code, profile, model name,
backend, logits, probability, confidence, feature, timing breakdown or internal exception.

The deny envelope is identical for public input/configuration failure and protected parsing/verification/
configuration failure. The public and protected success envelopes are intentionally distinguishable because
their output meanings differ; neither can be parsed as authorization input.

## 9. Model and artifact separation

The public and protected models must have separate:

- Python modules/classes and constructors;
- parameter tensors and storage identities;
- initialization and training runs;
- optimizer state and canonical state digests;
- invocation counters and timing collectors;
- ignored experiment reports and any temporary serialization paths.

The public implementation must not import or call a protected forward helper, access protected parameters,
attach a hook to protected modules, reuse protected logits/features, or select a protected head. A common pure
image validator may be reused only if it contains no model parameters, feature computation, policy or decision.

Generated data caches, JSON reports and local-only materialized state stay under existing ignored `data/a2/`
and `artifacts/a2/` roots. The trusted materializer may retain only CPU float32 `state_dict` files under
`artifacts/a2/local-states/`, together with a manifest containing the canonical state digest and file digest.
Optimizer state, full-model pickle, credentials, images, logits and features must not be retained or committed.
The manifest and loader must fail closed on topology, dtype, device, file digest or canonical state drift. The
state files remain local research artifacts and must not enter reports, responses, publication bundles or
version control.

## 10. Deterministic public baseline protocol

Implementation must close an ungated public baseline before extending the coordinator. It reuses only the
already verified Fashion-MNIST resources, canonical `[0,1]` float32 image preprocessing, deterministic
55,000/5,000 train/validation split and pinned CPU package tuple from A2-E1.

The public training constants are fixed independently of the protected baseline:

| Parameter | Value |
| --- | --- |
| `PYTHONHASHSEED` / Python / NumPy / torch seed | `20260730` |
| train/validation split and seed | reuse the fixed A2-E1 indices / `20260724` |
| public train-loader seed | `20260731` |
| loss | `torch.nn.CrossEntropyLoss()` |
| optimizer | `torch.optim.Adam` |
| learning rate / betas / epsilon | `0.001` / `(0.9, 0.999)` / `1e-8` |
| weight decay / AMSGrad | `0` / disabled |
| epochs | exactly 10 |
| train/evaluation batch | `128` / `256` |
| selected checkpoint | final epoch only |
| test evaluation | once after epoch ten |
| smoke accuracy floor | `>= 90.0%` on the two-class 10,000-image test set |

There is no scheduler, early stopping, mixed precision, gradient clipping, hyperparameter search or protected
model initialization. These values are constants, not CLI or request parameters. The test-set support is fixed
at 7,000 `NON_FOOTWEAR` and 3,000 `FOOTWEAR` examples. At least two clean processes with the fixed
`PYTHONHASHSEED` must produce identical ordered predictions, metrics and canonical model-state digest.

Acceptance uses a predeclared coarse-task accuracy floor only as a broken-experiment detector. The observed
accuracy, loss, confusion matrix, class support, parameter count, state digest, ordered-prediction digest and
latency must be reported; none is a security metric. The implementation checkpoint must not tune based on the
test set or alter the protected A2-E1 baseline.

两次独立进程现已按固定条件完成十 epoch 训练，均得到 test loss `0.007989783663357957`、accuracy
`99.85%`（`9985/10000`）、confusion matrix `[[6989,11],[4,2996]]`、ordered prediction SHA-256
`f54b2351606f21ff31fc7c23ed394c4dbe13ccb9b150a7fe10b6b27076926f0a`、model-state SHA-256
`b71980ebd3fb6e1a729b77109c98d3b4580e9e9cf8d3a28296cf6c18d1c122be` 和 determinism
fingerprint `e4fbf9c09afc3aaada32dd60f7368346a64138497178618c78ac0b1baeb4c14f`。模型为 50,370
parameters/201,480 parameter bytes；两次 batch-1 median 为 `70.2/65.9 us`，batch-256 median 为
`1464.0/1340.001 us`。这些分类与本机延迟结果不是安全指标或跨平台保证。

## 11. Required acceptance matrix

The implementation must provide unit, integration and defensive security tests for at least this matrix:

| Case | Committed decision | Verifier calls | Public calls | Protected calls |
| --- | --- | ---: | ---: | ---: |
| public entry disabled | `DENY`/entry absent | 0 | 0 | 0 |
| valid enabled public request | `PUBLIC` | 0 | 1 | 0 |
| malformed public image | `DENY` | 0 | 0 | 0 |
| valid protected request plus exact accept | `PROTECTED` | 1 | 0 | 1 |
| protected parse/profile/config/numeric reject | `DENY` | 1 at most | 0 | 0 |
| verifier exception or inactive backend | `DENY` | 1 at most | 0 | 0 |
| invalid public model output/exception | `PUBLIC`, external deny | 0 | 1 | 0 |
| invalid protected model output/exception | `PROTECTED`, external deny | 1 | 0 | 1 |

Tests must additionally cover:

- exact type, dtype, device, shape, layout, finiteness and range validation on both entries;
- unknown/duplicate fields and injection of entry, policy, capability, model/head, backend, evidence or decision;
- stable exact envelopes and absence of protected logits/features and detailed verifier evidence;
- public response replay, field relabelling and embedding as protected credential/evidence/context;
- no verifier, public or protected fallback after parser, verifier, model or response errors;
- distinct public/protected module and parameter storage identities;
- independent counters under repeated and concurrent public, protected and rejected requests;
- public requests cannot change protected counters, labels, state digest or output sequence;
- all 10,000 protected test labels remain equal to A2-E1 and all public outputs remain in `{0,1}`;
- reports and logs contain no credential, image, secret, logits, features or reusable authority.

Call-count assertions must wrap the real selected module boundary rather than infer non-invocation from the
response. Concurrency tests need deterministic barriers or bounded synchronization and must not store security
state in a global mutable variable.

## 12. Experiment comparison and metrics

The primary comparison is A2-E1 binary gate versus A2-E2 independent public/protected models. Report:

- protected label equality, accuracy and model state relative to A2-E1;
- public coarse-task loss, accuracy, confusion matrix and prediction/state digests;
- public/protected parameter counts and parameter-storage separation;
- model-only and end-to-end median/p95 latency after fixed warm-up for public, protected and deny paths;
- verifier-only and coordinator-only latency using the A2-E1 methodology;
- verifier, commit, public-model and protected-model invocation counts;
- response size and exact field sets;
- runtime/package/hardware tuple and sample counts.

Timing is an observability and overhead result, not a constant-time or side-channel claim. Measurements from the
current WSL2 CPU cannot be generalized to other hosts, devices or framework versions.

Independent head, shared trunk/shallow-deep exit and independent non-model service remain documented
alternatives, not co-equal runtime routes. A later comparison may implement one only through a separate
checkpoint with its own leakage and call-boundary definition. It cannot become fallback for this primary route.

## 13. Audit and observability

Internal experiment events use stable bounded result codes for entry, committed tier, completion/failure and
call counters. They may record response version, local policy version and non-sensitive model version, but not
raw input, credential, evidence detail, logits, features, parameters or reusable response bodies.

External clients cannot write counters, durations or audit fields. Returned snapshots must not expose mutable
internal state. Instrumentation failure must fail the affected request closed if it prevents verifying the
zero-protected-call invariant; it cannot disable counting and continue silently.

## 14. Deferred and excluded scope

A2-E2 excludes:

- shared trunk/head, shallow/deep early exit, MASK and internal activation zeroing;
- MNIST, LeNet and any new dataset;
- qint8, CUDA, ROCm, accelerator, compile, export, TorchScript or ONNX;
- challenge/response, nonce state, replay prevention and business-input binding;
- identity, account, network endpoint authorization or rate limiting;
- Stage B bearer capability, Router, MoE, agent and tool gateway;
- security-bearing lattice signatures or authentication;
- white-box integrity, TEE, secure boot, remote attestation and side-channel protection.

These exclusions cannot be introduced as conveniences, recovery modes or OR-composed alternatives.

## 15. Implementation checkpoint sequence

A2-E2 must proceed in two reviewable checkpoints:

1. **completed:** implement and reproduce only the independent ungated public coarse classifier, strict
   input/label contract, deterministic report and artifact tests;
2. **completed:** after accepting that baseline, extend one
   coordinator with the locally bound three-state experiment, fixed version-2 envelopes, call instrumentation
   and the complete acceptance matrix.

Checkpoint 1 did not edit the A2-E1 coordinator or expose a public entry. Checkpoint 2 must not retrain or alter
either accepted model baseline. D-024 now records the separate trusted materializer exception: it may
deterministically recreate the exact accepted states, persist only the local `state_dict` artifacts above, and
then hand the verified in-memory models to the unchanged evaluator.

The checkpoint-2 coordinator, defensive tests and evaluation runner are implemented. The runner has no CLI
route, policy, credential or training overrides; a trusted local caller must supply both in-memory model
instances, and their canonical state digests must match the accepted values before dataset evaluation. D-024
adds a separate trusted materializer and local state loader; neither is an untrusted request route, and the
evaluator still contains no training helper call.

The accepted-state run reproduced protected/public canonical state digests
`88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613` and
`b71980ebd3fb6e1a729b77109c98d3b4580e9e9cf8d3a28296cf6c18d1c122be`. The integrated 10,000-image
prediction digests equal both accepted baselines; the full label pass entered each selected model exactly
10,000 times, while the rejection probe entered neither model. These are reproducibility and black-box
isolation results, not authentication or production-security claims.

## 16. Security and research interpretation

Success would show that, in the measured black-box prototype, a locally enabled independent public function can
be executed without entering the protected model/head and without using verifier failure as downgrade logic.
It would provide call-isolation and output-scope evidence for capability-tier composition.

It would not show that a remote caller is authenticated, that an A0 credential is unforgeable or fresh, that a
public response cannot aid generic black-box model extraction, or that a process owner cannot call either model
directly. Classification accuracy, zero observed leakage and finite adversarial tests do not prove a
cryptographic security property.

## 17. Acceptance criteria for this specification

This specification is complete when:

- one primary public implementation and exact coarse function/output are fixed;
- public enablement is local, explicit, audited and disabled by default;
- one coordinator's mutually exclusive `DENY`/`PUBLIC`/`PROTECTED` semantics are unambiguous;
- protected verification failure remains A2-E1 deny and cannot select public;
- request schemas cannot select or inject policy, tier, model/head, backend, evidence or decision;
- public/protected model and artifact independence is testable;
- zero protected-model/head calls and zero protected logits/features on public/deny paths are explicit;
- public output cannot be upgraded, relabelled or reused as protected authority;
- deterministic baseline, acceptance matrix, metrics, artifact policy and checkpoint order are fixed;
- excluded security, runtime, model and Stage B scope is explicit.
