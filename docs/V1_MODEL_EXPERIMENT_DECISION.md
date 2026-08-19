# V1 CIFAR-100 ResNet-18 Model Experiment Decision

## 1. Status and claim boundary

本文档冻结 CAN 的 V1 主业务模型实验路线，决定编号为 `V1-M1`，可信 profile 标识为
`CAN-V1-CIFAR100-RESNET18-v1`。V1-M1 选择 CIFAR-100 与 CIFAR-style ResNet-18，用于评估
Module-SIS 神经认证器控制更现实 CNN 的准确率、调用隔离和端到端开销。

本决定及其本地 implementation checkpoint 固定数据集、模型族、输入边界、实验顺序和验收义务；本机只
实现严格 archive/parser、ResNet-18、adapter、训练选择和测试，不下载数据、不训练模型、不生成权重或正式
报告，也不表示当前 CPU-only 环境已适合完成 ResNet-18 训练。

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

本地 implementation 位于 `src/can/model/v1_cifar100_resnet.py`、
`src/can/access/v1_m1_adapter.py` 和 `src/can/experiments/v1_m1_baseline.py`。它们不导入 A2
model、dataset、parser 或 authorization route。`src/can/access/v1_m1_adapter.py` 已以现有 V1-C1 neural
evidence 和 A3-v2 状态机实现 `AuthenticatedR2`：该组合模型的公开入口处理 raw CIFAR tensor 与显式
credential，但不向请求方暴露单独的受保护 R2 forward。

## 4. Dataset identity and supply-chain boundary

数据集固定为 CIFAR-100 Python archive：50,000 个训练样本、10,000 个测试样本、100 个 fine classes、
每个样本为 `32x32` RGB 图像。下载入口、文件身份和验证顺序在首次下载前固定如下：

| Field | Frozen value |
| --- | --- |
| First-party dataset page | [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Archive URL | `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` |
| Archive filename | `cifar-100-python.tar.gz` |
| Exact archive size | `169001437` bytes (`161 MB` as stated by the first-party page) |
| SHA-256 | `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` |
| First-party MD5 cross-check | `eb9058c3a382ffc7106e4002c42a8d85` |
| Archive layout | `cifar-100-python/train`, `test`, and `meta` |
| Labels used by V1 | `fine_labels` only; `coarse_labels` are parsed only to verify the source layout and never enter training, validation, test, model input, or authorization |

首方页面给出 archive、161 MB 和 MD5；SHA-256 与精确字节数由同一 URL 的
[Hugging Face Datasets CIFAR-100 source record](https://huggingface.co/datasets/uoft-cs/cifar100/commit/8b6fcfb2e5cfcfb8387e8e3d932e103d2a3f0758)
交叉记录。下载时必须先匹配 byte size、SHA-256 和 MD5，任一不匹配即删除本次下载并拒绝解压；MD5
只用于与首方发布记录交叉比对，SHA-256 才是接受条件。

首方页面要求引用 Krizhevsky (2009)，但截至本决定冻结时未声明许可证或再分发条款。因此当前许可状态为
`UNLICENSED_OR_UNSPECIFIED`：只可从上述首方 URL 获取并用于本项目的本地研究实验；archive、解压样本、
派生缓存和 weights 均保持 ignored，禁止镜像、提交、发布、再分发或把数据包含进 artifact/release。任何外部
发布或再使用先要求独立的许可审查，不能从本决定推断获得授权。

`meta[b"fine_label_names"]` 的原始顺序是 V1 的唯一 fine-label index mapping，固定为：

```text
apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle,
bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle,
chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup,
dinosaur, dolphin, elephant, flatfish, forest, fox, girl, hamster, house,
kangaroo, keyboard, lamp, lawn_mower, leopard, lion, lizard, lobster, man,
maple_tree, motorcycle, mountain, mouse, mushroom, oak_tree, orange, orchid,
otter, palm_tree, pear, pickup_truck, pine_tree, plain, plate, poppy,
porcupine, possum, rabbit, raccoon, ray, road, rocket, rose, sea, seal,
shark, shrew, skunk, skyscraper, snail, snake, spider, squirrel, streetcar,
sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor,
train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm
```

下载后的 data loader 必须验证 `train` 为 50,000 条、`test` 为 10,000 条、每条 `data` 为 3,072 个
`uint8`、fine label 为整数 `0..99`，并且 `meta` 的有序列表与上文逐字一致。implementation 还要求
`train`/`test` 的五个规范字段与 `meta` 的两个规范字段精确匹配，并将解析前的三个解压文件逐字节同已验证
archive 内的对应成员比较。decoded dataset digest 的
固定算法为：对 `train` 后 `test` 的每个 source-order record 依次更新 SHA-256，初值为
`b"CAN-V1-CIFAR100-DECODED-v1\0"`，每条追加 `u8(fine_label) || 3072 raw channel-major pixel bytes`；
最终 digest 和每个 split 的 count 必须写入 ignored manifest。该 manifest 只能在 archive 三项校验全通过后
生成，且不含图像、response 或 secret。

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
parameter count: 11,220,132
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

mean/std、舍入、layout、contiguity 和 batch 规则固定如下：

| Stage | Frozen transform |
| --- | --- |
| train only | raw RGB `uint8` image -> zero-pad 4 pixels on every side -> uniform random `32x32` crop -> horizontal flip with probability `0.5` |
| validation/test/request | no crop, flip, resize, color transform, random transform, or PIL round-trip |
| all model paths | channel-major `uint8` -> contiguous `float32` -> exact division by `255.0` -> `(x - mean) / std` with `mean=(0.5071, 0.4867, 0.4408)` and `std=(0.2675, 0.2565, 0.2761)` |

训练增强只能使用训练 split，并由该次 run 的已冻结 RNG 状态驱动。验证、测试和受保护请求使用完全相同的
无随机 trusted preprocessing；不得进入请求处理路径或改变已绑定的原始 uint8 快照。

## 8. Training environment decision boundary

既有 A1/A2 结论仍只绑定 CPU tuple；V1-M1 的 GPU 环境子步骤已于 2026-08-15 在已授权的 AutoDL
单 GPU 容器完成。它是 V1 的独立实验环境，不替代或修改 `requirements-ml.lock` 中的 A2 CPU
环境。冻结事实如下：

| Field | Frozen observation |
| --- | --- |
| Base image | Miniconda/conda3, Ubuntu 22.04, image label CUDA 11.8 |
| OS | Ubuntu 22.04.1 LTS (`jammy`) |
| Conda environment | `can-v1`, CPython 3.11.9 |
| GPU | NVIDIA RTX A4000, 16,376 MiB VRAM |
| Host driver | 580.82.07; `nvidia-smi` reports driver CUDA compatibility 13.0 |
| PyTorch runtime | `torch==2.13.0+cu126`, `torch.version.cuda == "12.6"` |
| torchvision | `torchvision==0.28.0+cu126` |
| Compute/memory probe | 6 vCPU; 251 GiB RAM total, 227 GiB available; root overlay 30 GiB with 18 GiB free at probe time |
| Torch wheel SHA-256 | `0f4e49e334e24b552f694f6315e0676fb3f816fb0f727871b9c6d1f73784cc25` |
| Torchvision wheel SHA-256 | `92f53415dd68e56b6f912441997ab0e78fcd6245b1706ee6e88ce2df917248fa` |

`nvidia-smi` 的 CUDA 13.0 是 driver compatibility 读数，不是本实验 wheel 所用 runtime；正式记录的
PyTorch runtime 是 wheel 自带的 CUDA 12.6。环境创建时旧版 Conda 只支持 `classic` solver；可复现的
激活和 wheel 安装命令为：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate can-v1
python -m pip install --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.13.0+cu126' 'torchvision==0.28.0+cu126'
python -m pip check
```

Conda environment 的创建必须使用 Python 3.11.9；若镜像内 Conda 不支持 `libmamba`，使用
`--solver classic`。镜像自带 Python 3.10 不是 V1-M1 环境解释器。

同一实例重启验证和更换服务器重建步骤见 `docs/V1_AUTODL_ENVIRONMENT_SETUP.md`。该手册只记录
environment provenance，不能替代本决定中尚未冻结的数据与训练协议。

已完成的远程 smoke 只证明 CUDA runtime 可用：随机 CIFAR-shape ResNet-18 一次 CUDA 前向/反向的
loss 有限，峰值 allocated memory 为 106.13 MiB；同进程、同配置两次得到
`loss_a == loss_b == 5.407203674316406`。固定 policy 为 `PYTHONHASHSEED=1729`、
`CUBLAS_WORKSPACE_CONFIG=:4096:8`、Python/NumPy/PyTorch CPU/CUDA seeds `1729`、
`torch.use_deterministic_algorithms(True)`、`cudnn.benchmark=False`、
`cudnn.deterministic=True`，以及 CUDA matmul/cuDNN TF32 均关闭。它不是两次完整训练复现，也不构成
准确率、吞吐量或显存容量结论。

数据与训练协议及其 isolated implementation 现已冻结，但尚未下载或执行。ImageNet pretrained weights
不是依赖；两次 baseline 都从随机初始化的 CIFAR-style ResNet-18 训练。实现和本机测试不构成任何
准确率、吞吐量、显存或可复现训练结果，首次正式下载/训练前仍必须按工作日志进入 `SERVER_REQUIRED`。

计算资源通知边界固定如下：

- generated-key/sampler/vector、toy prover/rejection、A3-v2 retry 和 dependency-free `V1-C1-MSIS`
  checkpoint 已在 `LOCAL_OK` CPU-only 本机闭合；
- V1-M1 GPU tuple、数据/训练协议和 isolated code/unit/security tests 均已冻结；下一步的首次下载或
  正式训练为 `SERVER_REQUIRED`；
- 不得在任意 GPU 上以观察到的结果选择超参数、修改阈值或执行未记录的重试；
- 服务器只作为 V1-M1 完整训练、两次复现、多 seed/消融和 GPU 性能测量资源；本机继续承担开发、
  smoke test、CPU inference、artifact 校验和安全回归。

当前最低实用资源估计是单 NVIDIA GPU、至少 8 GiB VRAM 和 16 GiB system RAM；更稳妥目标是
12--16 GiB VRAM 和 32 GiB system RAM。上述 A4000 tuple 满足该资源下限；下文的 batch/epoch 是预注册
训练协议，不能因资源探测或训练中观察到的指标而调整。

## 9. Reproducibility protocol

V1-M1 的 two-run baseline 规格在首次训练前固定如下：

| Field | Frozen value |
| --- | --- |
| Runs | `R1` seed `1729` and `R2` seed `1730`; no retry and no third run before a new documented decision |
| Train/validation split | For each fine label, retain the first 50 occurrences in archive `train` source order for validation and use the remaining 450 for training: 5,000 validation and 45,000 training records |
| Test split | The unmodified 10,000-record archive `test`; never used for split, epoch selection, hyperparameter choice, or retry decisions |
| Batch/data loading | train batch `128`, validation/test batch `256`, `num_workers=4`, `pin_memory=True`, `persistent_workers=False`, `prefetch_factor=2`, train `shuffle=True`, validation/test `shuffle=False`, `drop_last=False` |
| Data-loader randomness | One run-local `torch.Generator` seeded with that run's seed drives train shuffle and worker base seeds; worker Python/NumPy seeds derive from PyTorch's assigned worker seed |
| Loss | `CrossEntropyLoss` with no label smoothing, class weighting, mixup, cutmix, or auxiliary loss |
| Optimizer | SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=0.0005`, `nesterov=True` |
| Scheduler | `CosineAnnealingLR(T_max=200, eta_min=0.0)`, stepped once after each completed epoch; exactly 200 epochs, no warmup and no early stopping |
| Determinism | Per-run `PYTHONHASHSEED`, Python/NumPy/PyTorch CPU/CUDA seeds all equal the run seed; `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, `cudnn.deterministic=True`, and CUDA matmul/cuDNN TF32 disabled |
| Progress observability | After data loading and deterministic setup, emit a `training started` stdout line. Refresh one fixed-width stdout progress bar after every completed train, validation, and test batch using only run/seed, stage, epoch and batch counts; it starts at `0.00%` and reaches `100.00%` after the final test batch. Do not emit a second progress line at epoch boundaries. After the selected state, manifest and report are atomically persisted, re-render the completed bar and emit `training completed`; aggregate epoch metrics remain in the report for selection and reproducibility. This is not an input, checkpoint criterion, artifact field, latency measurement, or source of per-sample/model-state data. |
| Checkpoint selection | Evaluate validation after every epoch; replace the in-memory best state only when validation top-1 is strictly higher, retaining the earlier epoch on a tie. After epoch 200, load this best-validation state and evaluate test exactly once; then atomically persist the selected CPU state, manifest and report under the ignored V1-M1 artifact root. |

每次 run 记录每 epoch train/validation loss 与 top-1、final test loss/top-1/top-5、selected epoch、ordered
test prediction digest、canonical state digest、model structure/parameter summary、dataset manifest digest 和
environment fingerprint。测试集结果不参与 checkpoint 或 run 的选择。二次运行结束后仅以 validation top-1
选择 gate 使用的 accepted baseline：较高者胜出，平局选择较小 seed；这项选择在读取两次 test metrics 后也
不得改变。

## 10. Baseline acceptance

无门控 baseline 必须先于认证门控实验独立验收。预注册接受条件为：

- `R1` 和 `R2` 均达到 validation top-1 `>= 70.00%`、test top-1 `>= 70.00%`；
- 两次 validation top-1 的绝对差和 test top-1 的绝对差均不得超过 `2.00` percentage points；
- 选定 checkpoint 必须源自上文 validation-only rule，且两次 run 均产生完整 dataset/environment/prediction/state
  manifest；
- ordered test predictions、模型结构摘要和关键环境 fingerprint 可比较；
- 非规范 dtype/shape/range、NaN/Inf 进入模型前即被 parser/adapter 拒绝；
- 模型参数量、float32 parameter bytes、batch-1/batch-N latency 和 throughput 被记录；
- accepted weight artifact 只有摘要和 manifest 可进入版本控制，权重本身保持 ignored。

任一条件失败即报告未验收，不得根据 test 结果调整增强、优化器、epoch、阈值或重新运行；新的探索性实验须以
新的决策和独立 artifact namespace 开始。分类准确率是业务模型指标，不是密码安全指标。

## 11. A3-v2 binding contract

A3-v2 协议核心只接收由本地可信 adapter 产生的 `input_digest`、`model_id` 和 `profile_digest`，不解析
CIFAR tensor。V1-M1 adapter 负责 section 5 的 canonicalization，并把同一 immutable snapshot 交给
协调器保存。

现有 A3-v1 的 `(1,1,28,28)` float32 encoding 和 133-byte message 保持不变。A3-v2 可以继续使用
固定长度 message envelope，因为业务输入只以 32-byte digest 进入消息，但必须使用新 version/domain
并拒绝 A3-v1 message、Fashion-MNIST input 和跨 profile digest。

## 12. Coordinator and model boundary

V1-M1 的主路线固定为物理组合对象 `AuthenticatedR2=(V_phi,C,f_theta)`，其中 `V_phi` 是已证明
`V_nn==V_ref` 的冻结 V1-C1 神经 Gate Layer，`C` 是 A3-v2 transcript/authorization coordinator，`f_theta`
是由 validation-only rule 选定、并由 `PROJECT_WORKLOG.md` 记录的冻结 R2 ResNet-18。`V_phi` 不读取业务图像 tensor，`f_theta` 不读取 credential
原始字段；二者只由 `C` 持有的 canonical input digest、profile 和一次性 transcript 绑定。当前阶段不训练
`V_phi`，不重训/微调 R2，不实施图像 trigger、soft gate、`gate * logits` 或输出遮蔽路线。

`V_phi` 的固定性具体指：由 coefficient-domain `V_ref`、完整 canonical range ledger 和可信公开
Module-SIS profile 确定性生成 topology/weights/bias，不经过 credential 数据训练，并在组合对象构造后
保持不可训练。NNAES 只作为“密码关系到固定神经图”的构造类比；不同于其 key-specific AES 网络，
`AuthenticatedR2` 的 Gate Layer 不嵌入签名私钥、prover secret 或静态隐藏 trigger。对非规范连续输入
不做神经插值，必须在进入 Gate Layer 前由 parser fail closed。

`AuthenticatedR2` 的数据流固定为：

```text
untrusted CIFAR input + V1 commitment
-> trusted V1-M1 input adapter and snapshot
-> A3-v2 pending transcript
-> V1-C1 neural Gate Layer evidence for canonical response
-> internal sole coordinator authorization commit
-> trusted preprocessing of the stored snapshot only on allow
-> exactly one internal R2 ResNet-18 invocation, or fixed deny
```

Verifier 不导入或调用 R2，也不持有私钥；它只输出 evidence。`C` 是唯一权限提交点，ResNet-18、Router 和
请求方都不是权限提交点。拒绝时 `AuthenticatedR2` 不执行 R2 forward、不释放 logits 或中间特征；“模型内”
描述的是固定 Gate Layer 作为组合模型的神经子模块，不是对白盒读取、修改权重或绕过组合 API 的防御主张。

当前 `AuthenticatedR2` 是 `V1-M1-C1` 的内部最小 reference：只验收 `DENY/protected`、冻结 R2 等价和
拒绝零调用。C1 的 accepted-R2 服务器报告通过后，`V1-M1-C2` 才增加显式 public entry 与二专家硬路由：
`PUBLIC` 只执行公共专家 `E0`，`PROTECTED` 只执行冻结受保护专家 `E1`，`DENY` 不执行二者；protected
失败不能降级 public。多个 protected experts、`allowed_mask` 和受约束 task router 不属于本决定的
M1-C2 实现，延后至独立 V1-M2 决策。

## 13. Gate experiment matrix

V1 主实验必须报告：

| Route | Expected verifier calls | Expected ResNet calls | Required result |
| --- | ---: | ---: | --- |
| valid emitted response | 1 neural Gate Layer call | 1 | gated prediction equals the frozen R2 baseline |
| malformed/tampered response | at most 1 | 0 | fixed deny |
| abort or retry exhaustion | 0 | 0 | fixed deny |
| expired/replayed transcript | 0 | 0 | fixed deny |
| input/model/profile mismatch | 0 | 0 | fixed deny |
| verifier/model internal error | at most 1 | 0 or one entered failing call | no fallback |

exact verifier 只作为 V1-C1 differential/oracle 对照；Gate Layer 是 `AuthenticatedR2` 的唯一配置神经验证路线。请求方不得选择 exact/neural 的 `A or B` 路线，也不得通过单独 R2 entry 绕过组合模型。

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

`can.experiments.v1_m1_c1` 是固定的无训练 accepted-R2 evaluator。它只接受 `run-2` 的 manifest、
baseline report 与 state，先核验 R2 canonical state digest、baseline prediction digest、官方 archive、
decoded dataset digest 与冻结 `cuda:0` 环境，之后才加载模型。它把 archive `test` 的同一 10,000 张原始
图像分别送入直接 R2 与 `AuthenticatedR2` allow 路径，并要求逐元素 logits 与 ordered prediction digest
相等。每次 allow 都由 index 确定性生成新的公开 V1-P2 conformance commitment/response，以符合
A3-v2 对同一 commitment 禁止重用的状态契约；这不生成、读取或保存 secret，也不是身份认证或不可伪造性
实验。外部 forward hook 只在进程内暂存一次 gated logits 以作比较，绝不写入 report。

性能报告固定使用 100 次 warmup、1,000 次串行 observation，并在每个 observation 的前后同步 CUDA。
它分别记录 input canonicalization/hash、commitment/challenge state、response encoding、neural Gate Layer、
无 R2 的 accepted Gate+coordinator、trusted preprocess/H2D、R2 inference，以及 accepted/rejected
端到端 latency/throughput。coordinator 的单独值是从无 R2 accepted path 与 neural Gate Layer median
相减得到的非负 residual estimate，不能误称为直接逐指令剖析。

## 15. Artifact and secret policy

CIFAR data、trained weights、optimizer state、checkpoint、transcript collection 和 profiler dump 均位于
ignored roots。`run_v1_m1_baseline` 将 selected CPU `state_dict`、`manifest.json` 和 `report.json` 写入
`artifacts/v1-m1/run-{1,2}/`；文件以原子创建方式写入、拒绝覆盖和 symlink，且 report/manifest 只记录公开
dataset/model/environment digest、metrics 和 prediction digest，不包含 state tensor 内容。不得记录 V1 secret
polynomial、mask、rejection state、原始认证 response 或可恢复私钥。

C1 evaluator 仅向 `artifacts/v1-m1/c1/accepted-r2-report.json` 原子写入一次 ignored report；该 report
包含 accepted R2 的公开 digest、环境、调用计数、结果 digest、拒绝隔离计数和性能统计，不包含图像、state、
raw credential、transcript 或 logits。已存在该路径时 fail closed，未经新的书面实验决定不得通过删除或覆盖
来重跑。

测试 key、mask 和临时 model state 只进入 pytest temporary directory 或进程内存，并在测试后清理。

## 16. Required tests

实现 V1-M1 时至少增加：

- architecture、parameter shape、class count、BatchNorm eval 和 deterministic forward unit tests；
- uint8 exact type/shape/range/channel/order、trailing/duplicate field 和 profile mismatch parser tests；
- canonical digest、snapshot mutation、normalization 和 direct/gated prediction equivalence tests；
- valid、tamper、replay、expiry、abort、concurrent duplicate response 和 zero-call security tests；
- `AuthenticatedR2` 的组成、冻结参数、credential/image 路径隔离、allow 与直接冻结 R2 prediction digest
  等价、reject 零 R2 forward/logits、以及 R2 direct-entry 不可由公开组合 API 选择的测试；
- Gate Layer compiler identity、公开 profile digest、固定 topology/parameter digest、不含私钥/静态
  trigger，以及实值/NaN/Inf/非规范 credential 在进入 neural core 前拒绝的测试；
- A2/A3-v1/Fashion-MNIST input 被 V1 route 拒绝且无 fallback 的 route-isolation tests；
- selected state、manifest/report、duplicate output 与 symlink artifact-root 的正负向测试，以及 no-secret/
  no-large-binary checks；
- C1 evaluator 的 run-2-only state/manifest/report 绑定、fresh public conformance credential sequence、
  tamper/replay/expiry/abort/route-confusion 零 R2 calls 与 C1 report overwrite/symlink 拒绝测试；
- training-start、single fixed-width batch progress `0.00%`/`100.00%`、无 epoch 重复 stdout 与 artifact 成功后的
  training-completed stdout tests；
- 明确标记的 dataset/training/performance integration tests，不使默认 unit suite 隐式下载或训练。

## 17. Deferred and excluded scope

R1/R2 baseline 是否通过、选定 state、artifact digest 和服务器 provenance 属于动态事实，唯一记录于
`PROJECT_WORKLOG.md`；本决定不复制 artifact 或权重。固定 `AuthenticatedR2` Gate Layer 本地组合及其
conformance allow/reject 等价与零调用测试已经通过。no-training C1 evaluator 已在本地实现并通过其
确定性 unit/focused tests；当前待执行的是已授权服务器上的 accepted-R2 10,000-image 等价、隔离和端到端
性能报告。C2 的 cut、public-head、训练选择规则和任何阈值失败后的探索性后续决策仍需另行冻结。

不属于 V1-M1：ImageNet、ViT/WideResNet 对比、adversarial robustness、模型水印、白盒权重保护、
分布式训练、生产 serving、完整侧信道、联合训练的 learned authentication gate、图像/提示词 Secret
Trigger、多受保护专家 MoE mask、V2 ML-DSA 和 Stage B 工具网关。

## 18. Acceptance criteria for this decision

本路线决定在以下条件满足时闭合：

1. CIFAR-100/CIFAR-style ResNet-18 是唯一 V1 headline protected-model experiment；
2. Fashion-MNIST/MLP 作为独立已验收回归基线保留，且不存在 fallback；
3. canonical uint8 input、trusted preprocessing、model graph 和 A3-v2 adapter 边界明确；
4. 训练环境、数据供应链、切分、预处理、训练协议和 acceptance threshold 在下载/训练前已冻结；
5. gate correctness、zero-call、安全、性能和 artifact 测试义务明确；
6. 工作日志、研究设计、安全文档、V1/A3 规格、README 和治理检查一致。
