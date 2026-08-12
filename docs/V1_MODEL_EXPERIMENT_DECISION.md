# V1 CIFAR-100 ResNet-18 Model Experiment Decision

## 1. Status and claim boundary

本文档冻结 CAN 的 V1 主业务模型实验路线，决定编号为 `V1-M1`，可信 profile 标识为
`CAN-V1-CIFAR100-RESNET18-v1`。V1-M1 选择 CIFAR-100 与 CIFAR-style ResNet-18，用于评估
Module-SIS 神经认证器控制更现实 CNN 的准确率、调用隔离和端到端开销。

本决定只选择数据集、模型族、输入边界、实验顺序和验收义务，不下载数据、不训练模型、不生成
权重或报告，也不表示当前 CPU-only 环境已适合完成 ResNet-18 训练。

## 2. Decision summary

V1 的实验层次固定为：

```text
V0/V1-prep regression baseline:
  existing Fashion-MNIST + MLP artifacts and tests

V1 protocol conformance:
  Module-SIS exact/neural verifier + deterministic toy fixtures

V1 headline protected-model experiment:
  CIFAR-100 + CIFAR-style ResNet-18
```

Fashion-MNIST/MLP 不被替换、删除、重训或改写。它继续提供快速回归、历史开销对照和 route-isolation
证据，但不能成为 V1 认证失败时的 fallback。

## 3. Route and implementation isolation

V1-M1 必须新增独立 model、dataset、experiment、adapter、profile、artifact namespace 和测试。禁止：

- 把现有 `CAN-A2-FMNIST-MLP-v1` 重命名为 V1；
- 修改 A2/A3-v1 parser 使其同时接受 CIFAR 输入；
- 让请求方选择 Fashion-MNIST 或 CIFAR 作为更弱认证路线；
- 在 V1 模型异常时调用 A2 protected/public model；
- 复用包含 A2 模型形状、class count 或 policy 语义的 helper。

只允许复用无模型、数据集或协议语义的确定性通用 helper。

## 4. Dataset identity and supply-chain boundary

数据集固定为 CIFAR-100：50,000 个训练样本、10,000 个测试样本、100 个 fine classes、每个样本为
`32x32` RGB 图像。后续 baseline checkpoint 必须在下载前冻结：

- authoritative source URL、archive name、size 和 cryptographic digest；
- dataset/version identity、fine-label ordering 和 license/redistribution status；
- train/validation/test policy，禁止使用 test labels 选择训练配置；
- decoded sample count、shape、dtype、label range 和 canonical dataset digest。

数据、解压目录和派生缓存保持 ignored，不提交或再分发。

## 5. Canonical V1 business input

V1-M1 外部业务输入固定为单张 canonical RGB image：

```text
exact type: uint8 tensor or an equivalently strict byte parser selected by the local profile
shape: (1, 3, 32, 32)
range: 0..255
channel order: RGB
coefficient order: channel-major, then row-major, then column-major
```

规范摘要输入为：

```text
b"CAN-V1-CIFAR100-INPUT-v1\0" ||
profile_digest ||
u16_be(3) || u16_be(32) || u16_be(32) ||
3072 exact pixel bytes
```

入口必须先验证、detach 和 clone，再计算摘要并保存同一快照。A3-v2 绑定该摘要；请求方不能提交摘要
替代原始图像，也不能提交 shape、normalization、channel order 或 model selector。

## 6. CIFAR-style ResNet-18 architecture

V1-M1 不是未经修改的 ImageNet stem。模型固定采用 CIFAR-style ResNet-18：

```text
input: 3x32x32
stem: Conv2d(3,64,kernel=3,stride=1,padding=1,bias=False) + BatchNorm2d + ReLU
max-pool: absent
residual stages: BasicBlock counts [2,2,2,2]
stage strides: [1,2,2,2]
global average pool: 1x1
classifier: Linear(512,100)
dropout: absent
output: float32 logits with shape (N,100)
```

卷积、BatchNorm、ReLU、residual addition、pooling 和 classifier 的具体 PyTorch module graph 必须在
实现 checkpoint 冻结并通过结构测试。请求路径只使用 `eval()` inference；训练态 BatchNorm 不进入
认证后模型调用。

## 7. Trusted preprocessing

规范摘要绑定 section 5 的原始 uint8 快照。可信本地 model profile 随后执行固定转换：

```text
uint8 -> float32 / 255
-> fixed channel-wise normalization
-> ResNet-18
```

mean/std、舍入、layout、contiguity 和 batch 规则必须在训练前冻结。数据增强只用于训练 pipeline，
不得进入请求处理路径或改变已绑定的推理快照。

## 8. Training environment decision boundary

当前已核验环境是 PyTorch CPU tuple，只支持既有 A1/A2 结论。V1-M1 训练前必须另做只读环境探测并
选择一个明确 tuple：Python、PyTorch、torchvision、device、CPU/GPU 型号、driver/runtime、线程数和
deterministic-algorithm policy。

没有完成该决定前，不安装 accelerator package、不下载预训练权重，也不声称 ResNet-18 可复现或
达到任何准确率。ImageNet pretrained weights 不是默认依赖；若未来采用，必须单独冻结来源、摘要和
迁移学习主张。

计算资源通知边界固定如下：

- generated-key/sampler/vector、toy prover/rejection、A3-v2 retry 和 dependency-free `V1-C1-MSIS`
  checkpoint 已在 `LOCAL_OK` CPU-only 本机闭合；
- 当前 `PROJECT_WORKLOG.md` 的唯一下一步已进入 V1-M1 GPU tuple 冻结，标记为 `SERVER_REQUIRED` 且已
  通知项目负责人；
- 通知和环境冻结完成前，不得在任意临时 GPU 上生成论文正式 baseline、选择超参数或回填阈值；
- 服务器只作为 V1-M1 完整训练、两次复现、多 seed/消融和 GPU 性能测量资源；本机继续承担开发、
  smoke test、CPU inference、artifact 校验和安全回归。

当前最低实用资源估计是单 NVIDIA GPU、至少 8 GiB VRAM 和 16 GiB system RAM；更稳妥目标是
12--16 GiB VRAM 和 32 GiB system RAM。该估计不是冻结 tuple，实际 GPU、driver/CUDA 和 package
版本仍由后续 `SERVER_REQUIRED` checkpoint 决定。

## 9. Reproducibility protocol

后续 baseline 规格必须在首次训练前冻结：

- train/validation split 和所有 seeds；
- augmentation、normalization、batch size 和 data-loader order；
- optimizer、scheduler、learning rate、weight decay、epochs 和 checkpoint-selection rule；
- deterministic flags、thread/device settings 和影响复现的环境变量；
- top-1/top-5 accuracy、loss、ordered prediction digest、state digest 和 environment fingerprint。

同一获准环境至少运行两次。接受阈值必须预注册，不能在观察 test accuracy 后回填。

## 10. Baseline acceptance

无门控 baseline 必须先于认证门控实验独立验收。至少要求：

- 两次运行满足预注册的 validation/test accuracy 规则；
- ordered test predictions、模型结构摘要和关键环境 fingerprint 可比较；
- 非规范 dtype/shape/range、NaN/Inf 进入模型前即被 parser/adapter 拒绝；
- 模型参数量、float32 parameter bytes、batch-1/batch-N latency 和 throughput 被记录；
- accepted weight artifact 只有摘要和 manifest 可进入版本控制，权重本身保持 ignored。

分类准确率是业务模型指标，不是密码安全指标。

## 11. A3-v2 binding contract

A3-v2 协议核心只接收由本地可信 adapter 产生的 `input_digest`、`model_id` 和 `profile_digest`，不解析
CIFAR tensor。V1-M1 adapter 负责 section 5 的 canonicalization，并把同一 immutable snapshot 交给
协调器保存。

现有 A3-v1 的 `(1,1,28,28)` float32 encoding 和 133-byte message 保持不变。A3-v2 可以继续使用
固定长度 message envelope，因为业务输入只以 32-byte digest 进入消息，但必须使用新 version/domain
并拒绝 A3-v1 message、Fashion-MNIST input 和跨 profile digest。

## 12. Coordinator and model boundary

V1 coordinator 的数据流固定为：

```text
untrusted CIFAR input + V1 commitment
-> trusted V1-M1 input adapter and snapshot
-> A3-v2 pending transcript
-> Module-SIS exact/neural evidence
-> sole coordinator authorization commit
-> trusted preprocessing of the stored snapshot
-> exactly one ResNet-18 invocation
```

Verifier 不导入模型，不调用模型，也不持有私钥。ResNet-18、Router 和请求方都不是权限提交点。

## 13. Gate experiment matrix

V1 主实验必须报告：

| Route | Expected verifier calls | Expected ResNet calls | Required result |
| --- | ---: | ---: | --- |
| valid emitted response | 1 | 1 | gated prediction equals ungated baseline |
| malformed/tampered response | at most 1 | 0 | fixed deny |
| abort or retry exhaustion | 0 | 0 | fixed deny |
| expired/replayed transcript | 0 | 0 | fixed deny |
| input/model/profile mismatch | 0 | 0 | fixed deny |
| verifier/model internal error | at most 1 | 0 or one entered failing call | no fallback |

必须分别测量 exact verifier 和未来 neural verifier，不得把二者实现为请求方可选择的 `A or B` 路线。

## 14. Performance reporting

报告至少分离：

- input canonicalization/hash；
- commitment/challenge state operations；
- exact/neural verification；
- coordinator commit；
- trusted preprocessing；
- ResNet-18 inference；
- accepted/rejected end-to-end latency 和 throughput。

V1-M1 的主要价值是观察认证成本相对现实 CNN inference 的比例；不得只报告总延迟而隐藏 verifier
或模型成本，也不得把本机结果外推为跨设备性能保证。

## 15. Artifact and secret policy

CIFAR data、trained weights、optimizer state、checkpoint、transcript collection 和 profiler dump 均位于
ignored roots。manifest 可以记录公开 dataset/model/environment digest、metrics 和 prediction digest，
但不得包含 V1 secret polynomial、mask、rejection state、原始认证 response 或可恢复私钥的信息。

测试 key、mask 和临时 model state 只进入 pytest temporary directory 或进程内存，并在测试后清理。

## 16. Required tests

实现 V1-M1 时至少增加：

- architecture、parameter shape、class count、BatchNorm eval 和 deterministic forward unit tests；
- uint8 exact type/shape/range/channel/order、trailing/duplicate field 和 profile mismatch parser tests；
- canonical digest、snapshot mutation、normalization 和 direct/gated prediction equivalence tests；
- valid、tamper、replay、expiry、abort、concurrent duplicate response 和 zero-call security tests；
- A2/A3-v1/Fashion-MNIST input 被 V1 route 拒绝且无 fallback 的 route-isolation tests；
- accepted artifact/manifest/dataset digest 和 no-secret/no-large-binary checks；
- 明确标记的 dataset/training/performance integration tests，不使默认 unit suite 隐式下载或训练。

## 17. Deferred and excluded scope

当前延期：训练设备和 package tuple、dataset archive digest、validation split、augmentation、optimizer、
scheduler、epochs、accuracy threshold、accepted weights、模型训练与门控报告。

不属于 V1-M1：ImageNet、ViT/WideResNet 对比、adversarial robustness、模型水印、白盒权重保护、
分布式训练、生产 serving、完整侧信道、V2 ML-DSA 和 Stage B 工具网关。

## 18. Acceptance criteria for this decision

本路线决定在以下条件满足时闭合：

1. CIFAR-100/CIFAR-style ResNet-18 是唯一 V1 headline protected-model experiment；
2. Fashion-MNIST/MLP 作为独立已验收回归基线保留，且不存在 fallback；
3. canonical uint8 input、trusted preprocessing、model graph 和 A3-v2 adapter 边界明确；
4. 训练环境和超参数在后续下载/训练前另行冻结；
5. gate correctness、zero-call、安全、性能和 artifact 测试义务明确；
6. 工作日志、研究设计、安全文档、V1/A3 规格、README 和治理检查一致。
