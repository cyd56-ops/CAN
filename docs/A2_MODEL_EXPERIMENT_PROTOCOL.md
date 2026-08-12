# A2 Minimum Business Model Experiment Protocol

## 1. Status and claim boundary

本文档固定阶段 A2 的首个最小业务模型实验，决定编号为 `A2-E1`，实验标识为
`CAN-A2-FMNIST-MLP-v1`。它承接 A1-B1 已实现的 evidence-only verifier。2026-07-23 的后续
baseline checkpoint 已安装并核验固定 CPU 依赖和 Fashion-MNIST 资源，实现严格输入校验、MLP 与
确定性无门控训练/评估入口，并完成两次同种子十 epoch 复现。2026-07-29 gate checkpoint 已实现
单一协调器、固定响应和二元前置硬门控，并完成 10,000 标签等价、拒绝零调用与延迟实验。

该实验是单机 CPU、toy credential 和黑盒服务假设下的科研原型。它只能评估分类 baseline、
门控开销和拒绝路径的受保护模型零调用，不能证明身份认证、不可伪造、replay 防护、白盒安全或
生产访问控制安全。

## 2. Decision summary

| Item | Decision |
| --- | --- |
| experiment ID | `CAN-A2-FMNIST-MLP-v1` |
| dataset | torchvision `FashionMNIST`, official train/test split |
| business model | float32 MLP, `784 -> 256 -> 128 -> 10` |
| framework tuple | Linux x86_64, CPython 3.11.*, torch `2.13.0+cpu`, torchvision `0.28.0+cpu` |
| device | CPU only |
| data root | `data/a2/` |
| training/validation split | 55,000 / 5,000 from the official 60,000 training examples |
| test set | official 10,000 examples, evaluated only after the fixed ten epochs |
| preprocessing | `ToTensor()` only; float32 in `[0,1]`; no augmentation or normalization |
| optimizer | Adam, learning rate `1e-3`, ten epochs, batch size 128 |
| gate | verifier evidence -> one local coordinator -> protected model |
| response | fixed deny envelope or one top-1 class index; no logits/features/evidence |
| excluded | MNIST, LeNet, public capability, MASK, qint8, CUDA, export and Stage B |

Fashion-MNIST is selected over MNIST because its ten-class task is less saturated while preserving the same small
`1x28x28` input and fixed train/test sizes. The MLP is selected over LeNet to minimize operator and latency variables
in the first gate experiment. LeNet remains a later comparison and is not an alternate runtime route.

## 3. Framework and installation contract

The only A2-E1 package tuple is:

```text
(Linux, x86_64, CPython 3.11.*, torch 2.13.0+cpu, torchvision 0.28.0+cpu)
```

The official CPU index query on 2026-07-23 listed `torchvision 0.28.0+cpu`. Read-only inspection of the cached
CPython 3.11 torchvision 0.28.0 wheel metadata reported `Requires-Dist: torch (==2.13.0)`, Python
`>=3.10,!=3.14.1`, and direct runtime dependencies on NumPy and Pillow. This matches the installed
`torch 2.13.0+cpu` local build under Python 3.11.

实现 checkpoint 实际只从官方 CPU wheel index 执行：

```bash
.venv/bin/python -m pip install \
  --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  'torchvision==0.28.0+cpu'
```

解析文件为 `torchvision-0.28.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`，缓存 wheel body 的
SHA-256 为 `1dad604dfc0177ecebe0891bd9701fe2c62ec3f7819a247be541b3fb6effee99`。实测
metadata/runtime 版本为 torch `2.13.0+cpu`、torchvision `0.28.0+cpu`、NumPy `2.4.4` 和
Pillow `12.2.0`；CUDA/HIP 均为 `None`、CUDA unavailable、torchvision CPU operators 可用，
`pip check` 通过。`requirements-ml.lock` 记录 CPython 3.11.9 下的完整解析闭包。PyPI 默认
wheels、nightly、Conda、源代码构建、CUDA wheels 和版本范围均不受支持。

## 4. Dataset source, identity and license

A2-E1 uses `torchvision.datasets.FashionMNIST` from torchvision `0.28.0+cpu`. The loader fixes this mirror and
resource identity:

```text
mirror: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
train-images-idx3-ubyte.gz  md5=8d4fb7e6c68d591d4c3dfef9ec88bf0d
train-labels-idx1-ubyte.gz  md5=25c81989df183df01b3e8a0aad5dffbe
t10k-images-idx3-ubyte.gz   md5=bef4ecab320f06d8554ea6380940ec79
t10k-labels-idx1-ubyte.gz   md5=bb300cfdad3c16e7a12a480ee83cd310
```

The canonical local root passed to torchvision is `data/a2/`, yielding `data/a2/FashionMNIST/raw/`. 实现
checkpoint 已验证全部 loader MD5，并记录压缩文件 SHA-256：

```text
train-images-idx3-ubyte.gz  sha256=3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84
train-labels-idx1-ubyte.gz  sha256=a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845
t10k-images-idx3-ubyte.gz   sha256=346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073
t10k-labels-idx1-ubyte.gz   sha256=67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5
```

训练前的解码检查满足：

- train images `(60000, 28, 28)`, train labels `(60000,)`;
- test images `(10000, 28, 28)`, test labels `(10000,)`;
- image dtype `uint8`; labels are exact integers in `[0,9]`;
- no missing, additional or silently replaced resource is accepted.

The Fashion-MNIST upstream repository publishes an MIT license. Torchvision itself is BSD-licensed and explicitly
states that its dataset utility does not grant dataset rights. 上游 GitHub contents API 返回 license blob
`6bc221fc3a933bbcba0161c926d69e4897382eda`；本地 ignored 副本
`artifacts/a2/licenses/Fashion-MNIST-LICENSE` 的 SHA-256 为
`13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1`，内容为 Zalando SE 2017
MIT notice。Raw/processed data 和 license artifact 不再分发，也不进入提交候选。

The loader's only configured mirror is HTTP and its embedded integrity value is MD5. These are reproducibility
identifiers, not a modern authenticity guarantee. The initial acquisition and license verification remain an
explicit residual supply-chain risk; model accuracy is not a security claim.

## 5. Canonical preprocessing and split

The fixed transform is only `torchvision.transforms.ToTensor()`. It converts each grayscale image to a contiguous
float32 tensor of shape `(1,28,28)` with values divided by 255 into `[0,1]`. No normalization, resize, crop, flip,
noise, augmentation or label transform is allowed.

The official test split remains untouched. The official 60,000-example training split is divided with a dedicated
CPU generator:

```text
GLOBAL_SEED       = 20260723
SPLIT_SEED        = 20260724
TRAIN_LOADER_SEED = 20260725
permutation       = torch.randperm(60000, generator=CPU generator seeded with SPLIT_SEED)
train_indices     = permutation[:55000]
validation_indices= permutation[55000:]
```

实现记录的 ordered little-endian int64 index SHA-256 为：train
`04812202f0f2671f8289fa0f9d3993fbbf3b16cc41321a04e0cb9d7975d20241`，validation
`3120ff0db03161b410bd7f8e0809b248e181755817b26df2377fede9490429c7`。Training uses
`shuffle=True` with a distinct CPU generator seeded once with `TRAIN_LOADER_SEED`;
validation and test use `shuffle=False`. `num_workers=0`, `drop_last=False`, train batch size 128 and evaluation
batch size 256 are fixed. No test example participates in model selection or hyperparameter tuning.

## 6. Business model contract

The protected model is one float32 CPU module with this exact topology:

```text
Flatten(start_dim=1)
Linear(784, 256, bias=True)
ReLU()
Linear(256, 128, bias=True)
ReLU()
Linear(128, 10, bias=True)
```

It has exactly 235,146 trainable scalar parameters. It contains no dropout, batch normalization, convolution,
residual path, pretrained parameter or verifier parameter. The verifier is never placed in the business optimizer,
and the business loss cannot update any compiled verifier buffer.

The model accepts only exact CPU `torch.float32` batches of shape `(N,1,28,28)`, with `N >= 1`, finite values in
`[0,1]`, and contiguous layout. Labels used by training/evaluation are exact CPU `torch.int64` vectors of shape
`(N,)` in `[0,9]`. Invalid type, shape, device, finiteness, range or label values fail before protected inference.

## 7. Deterministic training protocol

Each training process must start with `PYTHONHASHSEED=20260723` and set Python, NumPy and torch seeds to
`GLOBAL_SEED`. It must call `torch.use_deterministic_algorithms(True)`, use CPU only, set intra-op and inter-op
thread counts to one before work begins, and record process affinity, CPU model, WSL2/native Linux status and all
package versions.

Training is fixed as follows:

| Parameter | Value |
| --- | --- |
| loss | `torch.nn.CrossEntropyLoss()` |
| optimizer | `torch.optim.Adam` |
| learning rate | `0.001` |
| betas | `(0.9, 0.999)` |
| epsilon | `1e-8` |
| weight decay | `0` |
| AMSGrad | disabled |
| epochs | exactly 10 |
| train batch | 128 |
| gradient dtype | float32 |

There is no scheduler, early stopping, gradient clipping, mixed precision, compilation, quantization, pruning,
fine-tuning or hyperparameter search. Validation loss and top-1 accuracy are reported after every epoch, but the
final epoch is always the selected baseline; the official test set is evaluated once after epoch ten. Two complete
same-seed runs on the supported tuple must produce identical ordered test predictions, metrics and canonical
state-tensor SHA-256 before the deterministic claim is recorded.

## 8. Baseline metrics and acceptance

The ungated baseline report must include:

- epoch training loss, validation loss and validation top-1 accuracy;
- final test mean cross-entropy and top-1 accuracy over exactly 10,000 examples;
- per-class correct/count/accuracy and a `10x10` integer confusion matrix;
- exact parameter count, parameter bytes and serialized state size measured only in a temporary directory;
- model-only batch-1 and batch-256 latency under the method in section 12;
- package/runtime configuration, all seeds, split hashes, dataset hashes and state-tensor hash.

The smoke acceptance floor is final test top-1 accuracy `>= 85.0%`. This threshold detects a broken baseline; it is
not a promised result or a security property. Any paper result must report the observed value and repeated-run
conditions instead of replacing them with this floor.

两次独立进程执行均得到 test loss `0.33665058851242063`、top-1 accuracy `88.08%`
（`8808/10000`）、相同的 ordered prediction SHA-256
`e5b48d60c19304e54c412416abd0201e9c747afd00830b93af9122a738a2e4a7` 和 canonical model-state
SHA-256 `88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613`。十个类别的准确率依次为
`91.6, 97.0, 84.6, 90.3, 72.1, 96.5, 63.5, 92.5, 95.9, 96.8` percent；完整 confusion matrix
保留在 ignored JSON reports 中。两次运行的确定性 fingerprint 均为
`a59a9a9ac2797261eb824af564d6fa64a3c3e19fa43886b2349aa48bccaf7d53`。该结果超过 smoke floor，
只支持指定单机 CPU tuple 上的无门控分类 baseline，不支持任何访问控制或安全主张。

模型共有 235,146 个 float32 参数、940,584 parameter bytes；临时序列化测得 943,357 bytes，文件
SHA-256 为 `3d52c291d2ae3fd57ec1fbcfb388b7cc4377b855f2b28682c207be3291acd339`，随后已删除。

The implemented gate yields the same top-1 label as the ungated baseline for all 10,000 accepted test examples.
Both paths have ordered prediction SHA-256
`e5b48d60c19304e54c412416abd0201e9c747afd00830b93af9122a738a2e4a7`. The gate changes latency but not
business labels or accuracy. Logits remain available only to internal metric code and never enter an external
response.

## 9. Evidence, coordinator and model boundary

The A2-E1 trusted call path is fixed as:

```text
raw business input + raw 23-byte credential
-> canonical business-input validation
-> locally fixed A1-B1 verifier/backend
-> exact A1Evidence
-> one locally configured coordinator
-> committed internal deny or protected-model invocation
-> fixed external response envelope
```

The public experiment entry accepts raw business input and raw credential only. It does not accept `A1Evidence`,
`accepted`, `allow`, a capability/context, model ID, verifier candidate, backend, device, threshold, profile,
weights or policy. The coordinator invokes the locally configured verifier itself. Only exact
`A1EvidenceCode.NUMERIC_ACCEPT` may reach its internal allow branch; every other code, wrong evidence type,
exception or missing policy commits deny.

Evidence remains evidence only. It is not returned to the caller and cannot be reused as a capability. The
coordinator creates no Stage B token and holds no global mutable authorization state. A0 credentials remain
replayable by design; repeated numeric acceptance is not authentication or freshness.

## 10. Fail-closed response and zero-call contract

The external envelopes are structurally fixed:

```text
deny:  {version: 1, status: "deny"}
allow: {version: 1, status: "ok", class_id: integer in [0,9]}
```

No response contains logits, probabilities, intermediate features, evidence, profile, slot, detailed reject reason,
timing trace or model reference. Internal audit may record a stable coarse result code and request correlation ID,
but never the raw credential or image.

The protected model must be called only after the coordinator commits the allow branch. A trusted invocation counter
around the protected module is part of tests and benchmark instrumentation, not request input. The count must remain
zero for malformed business inputs, malformed credentials, every non-accept evidence code, verifier/config/operator
exception, caller-supplied evidence attempts, invalid model input and concurrent/repeated rejected requests.

Multiplying already-computed logits by a gate, running both branches, output masking, MASK layers and exception-time
fallback do not satisfy this contract. Direct business-model access by a process owner remains outside the black-box
threat model and must not be described as prevented.

## 11. Required tests

The implementation checkpoint must add unit, integration and security coverage for:

- exact topology, parameter count, input/label type, shape, device, finiteness and range validation;
- fixed dataset resource identities, sample counts, transform, split sizes/order and seed-derived hashes;
- same-seed deterministic initialization, loader order, predictions and metrics;
- legal accepted credential: one coordinator commit and exactly one protected forward per request batch;
- parse/profile/config/numeric rejection: stable deny envelope and zero protected forwards;
- evidence/type/boolean confusion, caller-supplied decision/evidence, unknown fields and backend selection attempts;
- verifier exception, inactive/tampered backend and policy/config error with no fallback and zero protected forwards;
- credential and business-input tamper, replay of rejected inputs and concurrent rejected submissions;
- repeated accepted A0 credential behavior documented as replayable, without creating reusable authorization state;
- deny responses contain no logits, features, evidence or fine-grained reason;
- allow labels match the ungated baseline over all 10,000 test examples;
- generated data, checkpoints and reports stay in ignored roots; test artifacts use pytest temporary directories.

Random invalid samples supplement rather than replace explicit boundary cases. A2 tests do not weaken or skip the
existing A0/A1 full-domain differential and no-fallback tests.

## 12. Latency and overhead method

Latency measurement uses `time.perf_counter_ns()` on the supported CPU tuple with model/verifier in evaluation and
inference mode, intra-op/inter-op threads fixed to one, `num_workers=0`, and process affinity recorded. Each path gets
100 untimed warm-up iterations followed by 1,000 individually recorded iterations over a fixed cyclic order of the
first 1,000 official test examples.

Report median, p95 and p99 microseconds for:

1. canonicalization plus protected model only, batch 1;
2. accepted verifier -> coordinator -> protected model end to end, batch 1;
3. rejected verifier -> coordinator -> deny, batch 1, with protected call count zero;
4. protected model only at batch 256 for throughput context.

Also report accepted-path absolute and percentage overhead relative to path 1, verifier-only latency, coordinator-only
latency where separable, peak resident memory observation and invocation counts. No cross-machine performance claim
is allowed, and rejected latency is not claimed constant-time.

两次无门控 baseline 的 batch-1 median/p95/p99 分别为
`110.8/230.5/308.0 us` 与 `104.9/205.0/273.3 us`；batch-256 分别为
`2987.4/3748.9/4157.2 us` 与 `2790.8/3338.2/3759.8 us`。环境为 WSL2、Intel i7-1260P、
affinity CPUs 0--15、torch intra/inter-op threads 均为 1；peak RSS 分别为 404,312 和 405,084 KiB。

2026-07-29 gate 运行在同一环境与方法下得到：model-only batch-1 median/p95/p99
`99.0/201.4/270.6 us`，accepted end-to-end `1849.2/2601.6/3127.8 us`，rejected end-to-end
`1570.5/2225.8/2744.4 us`，verifier-only `1245.5/1853.7/2198.2 us`。accepted 内部
coordinator median/p95/p99 为 `85.4/140.6/176.7 us`，accepted median overhead 为 `1750.2 us`
或 `1767.88%`；batch-256 model-only median 为 `3327.5 us`，peak RSS 为 404,980 KiB。accepted
latency run 的 1,100 次请求均各提交一次 allow 并调用一次模型；rejected latency run 的 1,100 次
请求均提交 deny 且零模型调用。这些本机观测不外推到其他平台，也不声称拒绝路径 constant-time。

## 13. Artifact, cache and cleanup policy

All dataset files live under ignored `data/a2/`. All reports, temporary state files, license snapshots and timing
traces live under ignored `artifacts/a2/`; optional training checkpoints live under ignored `checkpoints/a2/`.
Tests use pytest temporary directories.

No dataset, model state, optimizer state, pickle, NumPy array, checkpoint, generated report or large binary is
committed. A baseline may serialize state once in a temporary directory solely to measure size and SHA-256; that file
must be deleted before the checkpoint closes. The local data cache may be retained for reproducibility but must be
listed as ignored and excluded from the commit-ready file list. Cleanup verification must search the project outside
the ignored roots for `.pt`, `.pth`, `.ckpt`, `.onnx`, `.safetensors`, `.pkl`, `.pickle`, `.npy` and `.npz` files.

本 checkpoint 保留 ignored `data/a2/` cache、license snapshot 和
`artifacts/a2/baseline-repeat-{1,2}.json` 与 `artifacts/a2/gate.json` reports。临时 state file 已
清理；ignored roots 之外未生成上述模型、pickle 或 NumPy artifacts。gate report 不含 secret、
credential、图像、logits、features 或 evidence。

## 14. Deferred and excluded scope

The following remain outside A2-E1:

- MNIST, LeNet and any second dataset/model comparison;
- public/protected capability tiers, public heads/models and shallow/deep capability mapping;
- MASK, output multiplication, soft gates and business-network-internal credential triggers;
- qint8, CUDA, ROCm, accelerator, `torch.compile`, TorchScript, ONNX and other export;
- challenge-response, nonce consumption, identity/request binding and any security-bearing cryptography;
- Stage B capability issuance, Router, MoE, agent or tool gateway;
- white-box integrity, TEE, secure boot, remote attestation and side-channel guarantees.

None may be added as an implementation convenience or fallback. A later candidate needs its own decision and tests.

## 15. Next implementation checkpoint boundary

固定依赖、数据核验、MLP、两次 baseline 以及 section 9--12 的单一 coordinator 和二元硬门控均已
实现并核验，A2-E1 checkpoint 已闭合。公共入口只接收单张规范业务输入与原始 credential，固定
调用本地 A1-B1 backend，仅 exact `NUMERIC_ACCEPT` 可提交 allow；实测全部 10,000 个 test labels
与无门控 baseline 相同，所测拒绝路径均为零 protected-model calls。

后续工作不修改 A2-E1。A2-E2 public/protected capability 的独立模型主路线和分步实现边界已由
`docs/A2_CAPABILITY_EXPERIMENT_SPEC.md` 固定。独立 public coarse-model 无门控 baseline 已按该
规格实现并通过两次确定性复现；下一 checkpoint 才能在不重训或改变两个 baseline 的前提下集成
本地绑定的三态协调器。在该 checkpoint 前，不实现 MASK、CUDA/qint8/export、challenge/replay
state、Stage B、安全承载密码或其他数据集/模型路线。

## 16. Official references

- PyTorch CPU wheel index: `https://download.pytorch.org/whl/cpu`
- torchvision project and compatibility information: `https://github.com/pytorch/vision`
- Fashion-MNIST upstream repository and license: `https://github.com/zalandoresearch/fashion-mnist`
- Fashion-MNIST paper: `https://arxiv.org/abs/1708.07747`

## 17. Acceptance criteria for this decision

协议选择、无门控 baseline 与 coordinator/gate 条目均已满足：

- exactly one dataset, model, package tuple and CPU route are selected;
- data identity, source, license handling, cache and cleanup rules are explicit;
- preprocessing, split, seeds, model, optimizer, budget and baseline metrics are deterministic and testable;
- evidence cannot be supplied by the requester or treated as authority;
- one coordinator is the only gate commit point and all rejects require zero protected-model calls;
- response envelopes prevent protected logits/features and detailed verifier evidence from escaping;
- required positive, negative, replay, tamper, concurrency and artifact tests are enumerated;
- public capability, MASK, qint8/CUDA/export, Stage B and security-bearing cryptography remain excluded;
- 固定 package/data/model/training 已按本文实现、复现并核算，generated artifacts 仅保留在 ignored roots；
- coordinator、固定响应、10,000 标签等价和拒绝零 protected-model calls 已实现并核验，A2-E1
  在 toy、黑盒、单机 CPU 限制下闭合。
