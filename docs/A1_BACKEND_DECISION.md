# A1 PyTorch Exact-Integer Backend Decision

## 1. Status and claim boundary

本文档固定 `CAN-RELU-EXACT-v1` 的首个 PyTorch 部署 backend，决定编号为 `A1-B1`，
backend 标识为 `CAN-TORCH-CPU-EXACT-v1`。它承接 `docs/A1_NUMERICAL_SPEC.md` 的共同契约和
`docs/A1_CONSTRUCTION_DECISION.md` 的固定三层 graph，不改变 A0 wire encoding、接受阈值、
evidence 或授权边界。

2026-07-23 的后续实现 checkpoint 已按本文完成 A1-B1：指定 CPU wheel 已安装和核验，
`src/can/verifier/a1_torch.py` 已实现 in-memory exact-integer backend、startup activation gate 和
raw-bytes evidence adapter，独立 unit/differential/security tests 已通过。后续 A2-E1 checkpoint 已
单独安装 torchvision、核验 Fashion-MNIST 并实现 MLP baseline；这些依赖和结果不进入 A1-B1
runtime contract。协调器和 capability 仍未实现。

本决定仍只服务于 A0 非生产 toy 数值解锁。即使未来 backend 全部测试通过，也不能据此声称
身份认证、不可伪造性、replay 防护、白盒安全、业务模型零调用或生产部署安全。

## 2. Decision summary

A1-B1 固定以下首个目标：

| Item | Decision |
| --- | --- |
| backend ID | `CAN-TORCH-CPU-EXACT-v1` |
| graph | `8 -> 40 -> 16 -> 1`，三个 affine+ReLU blocks |
| operating system | Linux `x86_64` |
| Python | CPython `3.11.*` |
| framework | PyTorch `2.13.0+cpu` |
| install channel | `https://download.pytorch.org/whl/cpu` |
| device | `torch.device("cpu")` only |
| weight/bias/activation storage | `torch.int32`, scale `1`, zero-point `0` |
| product storage | `torch.int32`，由逐层范围证明无溢出 |
| affine reduction/pre-activation | `torch.int64` |
| ReLU | `torch.clamp(z, min=0)`，不设置 upper clamp |
| execution | eager mode under `torch.inference_mode()` and `module.eval()` |
| parameter form | non-trainable, non-persistent registered buffers |
| quantized tensor engine | not selected；不使用 `qint8`、`quint8` 或 `torch.ao.nn.quantized.Linear` |
| export | disabled；不使用 `torch.compile`、TorchScript、ONNX 或其他序列化/转换 |

该目标是“PyTorch 上的精确定点整数 conformance backend”，不是常见的 8-bit post-training
quantization。A1-C1 的逻辑数值仍是 `int32`/scale `1`；`int64` reduction 是不改变结果的物理
累加器 widening，不增加容差，也不改变 candidate ID。

## 3. Read-only environment probe

2026-07-23 在 `/home/kali/CAN` 执行了只读探测，结果如下：

| Probe | Observed result | Decision consequence |
| --- | --- | --- |
| kernel/platform | Linux WSL2 `6.18.33.2-microsoft-standard-WSL2`, `x86_64`, glibc `2.38` | 选择 Linux x86_64 CPU wheel |
| CPU | Intel Core i7-1260P, 16 logical CPUs, AVX2 visible | 只支持本机 CPU 基线；不外推其他 CPU 性能 |
| Python | `.venv` CPython `3.11.9` | 与项目 `==3.11.*` 约束一致 |
| memory | about 7.6 GiB total, 6.8 GiB available at probe time | 足以执行 1,033 个 dense scalar parameters 的 toy graph |
| disk | about 941 GiB available at probe time | 不构成 wheel 或数据集下载授权 |
| PyTorch | `torch` and `torchvision` not installed | 本 checkpoint 不运行 backend 行为测试 |
| NVIDIA | `nvidia-smi` unavailable | 没有可验证的 CUDA 路线 |
| AMD | `rocminfo` unavailable | 没有可验证的 ROCm 路线 |

“命令不可用”不等价于证明宿主机没有 GPU；它只说明当前工作环境没有可复现、可验证的 GPU
工具链。因此 CUDA、ROCm、MPS、XPU 和其他 accelerator 都不在 A1-B1 支持集合内。

## 4. Version and installation channel

PyTorch 官方 Get Started 页面在探测时把 `2.13.0` 标为 stable，Linux 安装说明支持 Python
3.10--3.14，并要求无 CUDA/ROCm 需求时选择 CPU compute platform。官方 CPU wheel index 同时
列出了本环境对应的构建：

```text
torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl
sha256=6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b
```

实现 checkpoint 实际执行的初始安装命令为：

```bash
.venv/bin/python -m pip install \
  --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.13.0+cpu'
```

实际安装文件为
`torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`；pip 缓存中的对应 Zip body 实测
SHA-256 为 `6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b`。
`importlib.metadata.version("torch")` 与 `torch.__version__` 均为 `2.13.0+cpu`，
`torch.version.cuda`/`torch.version.hip` 均为 `None`，`torch.cuda.is_available()` 为 `False`。
该 A1 实现 checkpoint 当时 `torchvision` 的 module spec 为 `None`。PyTorch CPU index 曾把
setuptools 降至 78.1.0，随后已按 `pyproject.toml` 恢复为 83.0.0，`pip check` 通过。

`torchvision` 不参与 A1 verifier，不得成为 A1-B1 runtime 依赖。后续 A2-E1 已在独立 checkpoint
安装 `torchvision==0.28.0+cpu` 并核验其 wheel/hash 与 Fashion-MNIST 资源；该环境变化不改变
A1-B1 的 candidate ID、operator mapping 或已执行性质测试。

不得使用 PyPI 默认 channel、nightly、Conda、源代码构建、CUDA wheel 或“兼容版本范围”替代
上述构建。升级 PyTorch 的 patch/minor 版本也属于新的 backend 转换，不能自动继承 A1-B1
证据。

## 5. Runtime and device contract

A1-B1 的支持 tuple 固定为：

```text
(Linux, x86_64, CPython 3.11.*, torch 2.13.0+cpu, device=cpu, eager mode)
```

backend 初始化和每次调用前的可信适配器必须验证：

- 输入来自现有 A0 23 字节 parser，而不是请求方提供的 tensor；
- canonical credential tensor 的 shape 精确为 `(8,)`、dtype 为 `torch.int32`、device 为 CPU；
- graph buffers 全部位于 CPU，shape、dtype、layout、stride 和内容与本地 compiled profile 一致；
- 没有请求字段、环境变量或调用参数可以选择 device、dtype、candidate、scale 或算子路线；
- module 处于 evaluation mode，执行位于 `torch.inference_mode()` 中；
- 输出精确为 shape `(1,)`、dtype `torch.int32`，且值属于 `{0,1}`。

不允许对 backend 调用 `.cuda()`、`.to("cuda")`、`.half()`、`.float()` 或其他迁移/转换后继续
提供验证服务。发现设备或 dtype 漂移时，适配器返回现有 `CONFIG_REJECT` evidence，backend
保持禁用，不调用 dependency-free evaluator、exact-ops 或 `V_ref`。

线程数和 CPU kernel 调度不参与 relation。所有乘加在证明范围内使用精确整数，不因归约次序
变化而改变结果。性能实验仍必须记录 `torch.get_num_threads()`、进程 affinity、CPU 型号、
WSL2/原生 Linux 状态和 warm-up 方法，不能把本机数字外推到其他环境。

## 6. Tensor and module layout

固定 dense row-major layout 如下：

| Tensor | Shape | Required stride | Storage dtype | Scale / zero-point | Role |
| --- | --- | --- | --- | --- | --- |
| `input` | `(8,)` | `(1,)` | `torch.int32` | `1 / 0` | canonical `b` |
| `weight_1` | `(40,8)` | `(8,1)` | `torch.int32` | `1 / 0` | Layer 1 weights |
| `bias_1` | `(40,)` | `(1,)` | `torch.int32` | `1 / 0` | folded-anchor bias |
| `activation_1` | `(40,)` | `(1,)` | `torch.int32` | `1 / 0` | Layer 1 ReLU output |
| `weight_2` | `(16,40)` | `(40,1)` | `torch.int32` | `1 / 0` | Layer 2 weights |
| `bias_2` | `(16,)` | `(1,)` | `torch.int32` | `1 / 0` | Layer 2 bias |
| `activation_2` | `(16,)` | `(1,)` | `torch.int32` | `1 / 0` | Layer 2 ReLU output |
| `weight_3` | `(1,16)` | `(16,1)` | `torch.int32` | `1 / 0` | Layer 3 weights |
| `bias_3` | `(1,)` | `(1,)` | `torch.int32` | `1 / 0` | Layer 3 bias |
| `activation_3` | `(1,)` | `(1,)` | `torch.int32` | `1 / 0` | exact output bit |

每层临时 `products` 使用与 weight 相同的 shape 和 `torch.int32`；row reduction、bias add、
pre-activation 和 ReLU 临时值使用 `torch.int64`。ReLU 输出只有在该层 range certificate 已在
backend 启用前验证后，才精确转换回 `torch.int32`。

weights 和 biases 必须由现有 `A1CompiledProfile` 在内存中构造，并使用
`register_buffer(..., persistent=False)` 注册。module 必须满足：

- `tuple(module.parameters()) == ()`；
- 所有 buffers 的 `requires_grad` 为 `False`；
- `state_dict()` 不含 verifier weights/biases；
- 不把 verifier 放入业务 optimizer；
- 不暴露接受外部 tensor、weights、bias、threshold 或 device 的公共 core API。

PyTorch tensor 本身在白盒 Python 进程内可被原地修改，因此“不可变”仍依赖可信宿主代码。
backend 启用前必须逐元素比较 compiled profile 与 buffers；本项目不声称抵抗持有进程控制权的
攻击者。

## 7. Affine and ReLU operator mapping

对每个固定 layer，A1-B1 只允许以下等价映射：

```text
expanded_input = activation.unsqueeze(0).expand_as(weight)
products_i32 = torch.mul(weight, expanded_input)
accumulator_i64 = torch.sum(products_i32, dim=1, dtype=torch.int64)
preactivation_i64 = torch.add(accumulator_i64, bias.to(torch.int64))
relu_i64 = torch.clamp(preactivation_i64, min=0)
next_activation_i32 = relu_i64.to(torch.int32)
```

`unsqueeze`/`expand_as` 的目标 shape 只来自可信固定 weight；`torch.mul` 的两个参数在调用时 shape
已经相同，不依赖隐式 broadcast。`torch.sum(..., dtype=torch.int64)` 显式要求在归约前转为
`int64`，避免依赖默认 dtype promotion。`torch.clamp(..., min=0)` 在此仅实现
`rho(z)=max(0,z)`；不设置 `max`，也不是用于掩盖 overflow 的额外饱和。

三层必须逐层执行上述序列，不得融合、重排或替换为数值语义未验证的 kernel。首版明确不使用：

- `torch.nn.Linear`、`torch.mv`、`torch.matmul` 或 BLAS/GEMV 快捷路径；
- `torch.ao.nn.quantized.Linear`、QNNPACK、FBGEMM、oneDNN quantized lowering；
- float cast、fake quantization、observer、calibration 或训练；
- `torch.compile`、TorchScript、ONNX、ExecuTorch 或第三方 runtime；
- `%`、Floor、`abs`、关系比较、Sigmoid 或输入相关分支实现 relation；
- exception 时的 dependency-free、exact-ops 或 `V_ref` fallback。

以后若以 `torch.mv` 或 fused linear 优化 affine，它必须获得新的 backend ID，并从安装探测、
逐层 trace、全域差分和 artifact 检查重新开始，不能被当作 A1-B1 的内部无风险优化。

## 8. Overflow, rounding and saturation

A1-B1 不执行重标定，因此 scale 固定为 `1`、zero-point 固定为 `0`，没有量化 rounding。固定
range ledger 给出：

| Stage | Product range | Conservative affine range | ReLU/storage range |
| --- | --- | --- | --- |
| Layer 1 | `[-256,256]` | `[-384,385]` | `[0,385]` |
| Layer 2 | `[-770,770]` | `[-1145,907]` | `[0,9]` for semantic outputs |
| Layer 3 | `[-9,9]` | `[-71,65]` | `{0,1}` |

所有 product 都安全位于 `int32`，所有 reduction 和 bias add 位于 `int64`，所有 ReLU 输出又安全
位于 `int32`。从 `relu_i64` 转为 `int32` 是由证明保证的恒等转换，不允许通过 upper clamp、
wraparound 或 silent saturation 使越界值“可用”。

backend loader 必须从实际 compiled profile 重算 weight/bias 范围和上述保守包络，不能只信任
本文常量。证明失败、shape 变化、未知 stride、算子返回错误 dtype、cast 前范围不成立或输出
不是 exact bit 时统一 `CONFIG_REJECT` 并禁用 backend。

因为 relation path 没有除法和 rescale，ties-to-even 规则在 A1-B1 中不被触发。未来任何非单位
scale、qint8 或导出路线必须重新定义 rounding、saturation 和逐层误差，不得继承 error `0`。

## 9. Quantization disposition

首个 backend 不选择 PyTorch quantized tensor engine。官方 `torch.ao.nn.quantized.Linear` 使用
quantized tensor 输入/输出，默认 qint8 weight，并具有输出 scale 与 zero-point；这会引入与
A1-C1 当前 exact-int graph 不同的表示、重标定和 backend lowering。

因此 A1-B1 的结论是：

- 保留 A1 的“固定点/量化研究”总方向，但先建立 scale `1` 的精确 PyTorch 整数基线；
- qint8/quint8、per-tensor/per-channel quantization、torchao 和硬件 quantized kernels 均延期；
- 后续 8-bit 候选必须使用新的 candidate/backend ID，逐 tensor 声明 scale/zero-point、
  accumulator、rounding、saturation 和误差证书；
- 任何量化候选只要产生一例 false accept、无法证明全域包含或需要 exact 路线兜底，就不得启用。

这项延期是 correctness 选择，不是性能结论。本文不声称 qint8 一定无法实现 A1 relation，只说明
它不能无证明地继承 `CAN-RELU-EXACT-v1` 的零误差主张。

## 10. Activation gate and fail-closed behavior

实现完成后，backend 只有在一次显式 startup activation gate 全部通过时才可接收请求。gate 至少
包含：

1. 验证 OS、architecture、Python、torch public/local version 和 CPU-only build；
2. 验证三层 buffer 的 shape、stride、dtype、device、内容和 non-persistent 状态；
3. 验证 module 没有 parameters、grad、training-only layer 或 optimizer membership；
4. 对 `mul -> sum(dtype=int64) -> add -> clamp(min=0) -> cast(int32)` 执行边界 micro-probe；
5. 对实际 compiled profile 执行逐层 range ledger 和完整 backend differential matrix；
6. 验证 raw credential adapter 只能返回现有稳定 evidence，且没有 fallback import/call；
7. 验证没有 compiled model/state artifact 写入仓库或持久目录。

activation 是单进程本地可信配置行为，不由请求触发。任何一步失败都必须使该 backend 整体保持
disabled；请求只能得到 `CONFIG_REJECT`。禁止在同一服务中写成：

```text
torch_backend OR dependency_free_backend OR exact_ops OR V_ref
```

dependency-free evaluator 继续作为测试事实源，但不能在 A1-B1 的运行时异常、设备不支持或配置
失败时接管请求。要恢复服务只能修复可信环境、重新生成 backend 并完整重跑 activation gate。

## 11. Required conformance and differential tests

A1-B1 实现验收必须具有独立的 unit/differential/security tests，并至少覆盖以下矩阵；当前实现
已按下列矩阵执行。

### Environment and module structure

- 精确接受 `2.13.0+cpu`/CPU，拒绝错误版本、CUDA/HIP build、非 CPU device 和 dtype 漂移；
- 验证所有固定 shapes、strides、buffers、non-persistent state 和零 Parameters；
- 验证外部调用不能提交 tensor、candidate、device、weight、bias、threshold、scale 或 evidence；
- 验证 backend module 未安装时 dependency-free A0/A1 包仍可导入，且不存在隐式 fallback。

### Exact arithmetic and trace

- 对每层执行 operator micro-probe，断言 product、accumulator、pre-activation、ReLU 和 cast dtype；
- 穷尽 `u=-256..256`，逐值比较五-ReLU distance 与 dependency-free trace；
- 穷尽 `d=0..128` 和 AND sum `0..8`；
- 对每个启用 slot、每个分量、全部 `b_i=0..256` 比较逐层 tensor；
- 复用 issuer-core、reference-guard、first-reject、wrap、bit-zero、mixed-component 和 malformed
  向量族；
- false accept 必须为 `0`，issuer-core false reject 必须为 `0`；
- 输出必须始终为 exact `torch.int32` bit，不接受浮点或 truthy/falsy 转换。

### Fail-closed and artifact behavior

- buffer tamper、错误 shape/stride、错误 dtype/device、range certificate 失败和 operator exception
  都只返回 `CONFIG_REJECT`；
- instrumentation 证明不调用 dependency-free core、exact-ops 或 `V_ref` fallback；
- `state_dict()` 不包含 verifier buffers，测试临时目录之外没有 `.pt`/`.pth`/`.ckpt` 等产物；
- 后续协调器存在后，所有 backend/config/numeric reject 的 protected-model 调用计数必须为零。

每次 framework 升级、device 迁移、dtype/scale 变化、量化、fuse、export、prune 或 fine-tune 后，
上述矩阵与当时完整项目测试必须全部重跑。A1-B1 实现的首个 acceptance run 还必须先重跑本决定
形成时已有的 134 项测试，确保引入可选 torch 边界没有破坏 dependency-free 路线。

目标质量命令为：

```bash
.venv/bin/python -m pytest tests/unit
.venv/bin/python -m pytest tests/differential
.venv/bin/python -m pytest tests/integration
.venv/bin/python -m pytest tests/security
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src tests
.venv/bin/python -m pip check
bash -n scripts/check_governance_docs.sh
./scripts/check_governance_docs.sh
```

backend acceptance 报告必须写明实际 test count；不能用 skip 或“应当通过”替代未运行的 torch
tests。随机向量只能补充，不能替代有限域分解与逐分量穷尽。

## 12. Compiled artifact policy

`t` 已折叠进 Layer 1 bias，weights/biases 因而属于 toy secret-bearing artifacts。A1-B1 固定：

- compiled profile 和 tensor 只在进程内或 pytest 临时目录中从 toy fixture 重建；
- module buffers 使用 `persistent=False`，不得写入业务 model checkpoint；
- 禁止 `torch.save`、pickle full module、TorchScript save、ONNX export、safetensors 或 checkpoint；
- 不提交 `.pt`、`.pth`、`.ckpt`、compiled weights、credential 集合、trace dump 或 benchmark dump；
- 日志不得包含 `t`、完整 bias/weight、credential、逐层 trace 或可复用模型 artifact；
- 测试结束后由 pytest 临时目录生命周期清理临时产物。

当前 `.gitignore` 已覆盖 `artifacts/`、`checkpoints/`、`*.pt`、`*.pth` 和 `*.ckpt`。由于本路线
禁止其他 export 格式，当前不应生成 `.onnx`、`.safetensors` 或 pickle；未来一旦允许任何新格式，
必须在同一 checkpoint 先增加 ignore 和安全测试。

non-persistent buffer 只减少意外进入 `state_dict` 的风险，不提供秘密保护。白盒持有者仍可读取
内存中的 tensor，A0 credential 也仍可被构造或 replay。

## 13. Performance and portability claims

A1-B1 的性能定位是正确性优先的 eager dense baseline：

- 物理执行 dense elementwise multiply 和 row reduction，即使逻辑 nonzero weights 只有 136；
- `int64` reduction 和逐层 tensor 转换可能比优化的 GEMV/qint8 kernel 更慢；
- 当前只有 WSL2、i7-1260P、CPython 3.11.9、PyTorch `2.13.0+cpu` 这一实测环境，不存在 CUDA
  或跨平台数据；
- backend 已安装并完成正确性验收，但尚未进行正式 latency、throughput、memory 或 speedup
  benchmark；startup gate 的单次观察不能作为性能实验。

后续实验必须分别报告逻辑 sparse operation count、实际 dense tensor operation、warm/cold latency、
线程数和 resident memory。不得把 CPU exact baseline 描述为量化加速，也不得把单机结果推广到
原生 Linux、其他 CPU、GPU、mobile 或 production service。

## 14. Disable, rollback and migration conditions

下列任一条件立即禁用 A1-B1：

- wheel 来源、hash、version、Python、OS、architecture 或 device 不匹配；
- operator 不支持所需整数 dtype，返回错误 dtype，或 trace 与 dependency-free backend 不一致；
- compiled buffer 的 shape、stride、content、scale、zero-point 或 range certificate 不匹配；
- 任一 false accept、非 exact output bit、overflow、wrap、rounding 或 upper saturation 被观察到；
- module 出现 trainable Parameter、optimizer membership、persistent state 或未授权 artifact；
- 运行路径导入/调用 float、dependency-free、exact-ops 或 `V_ref` fallback；
- framework upgrade、quantization、fusion、export、device migration、prune 或 fine-tune 未完成复测。

这里的“rollback”只表示禁用未通过 gate 的 backend 并恢复到没有 A1-B1 服务的状态，不表示自动
切换到更弱 verifier。dependency-free backend 仍可离线用于诊断和差分测试，但不承接部署请求。

## 15. Next implementation checkpoint boundary

A1-B1 实现 checkpoint 已完成以下范围：

1. 按第 4 节安装并验证 CPU wheel；
2. 在 `src/can/verifier/a1_torch.py` 建立显式 optional-dependency 边界；
3. 从现有 compiled profile 在内存中构建 non-persistent buffers；
4. 实现第 7 节唯一 operator sequence 和 raw-bytes evidence adapter；
5. 新增第 11 节测试并运行完整质量矩阵；
6. 更新工作日志中的实际版本、命令、结果、文件清单和残余风险。

实际新增 `src/can/verifier/a1_torch.py` 及三个独立测试文件；旧 134 项基线和包含新测试的 185 项
完整套件均通过。没有生成 `.pt`/`.pth`/`.ckpt`/ONNX/safetensors/pickle artifact，也没有下载
MNIST/Fashion-MNIST、实现 LeNet/MLP、协调器、capability、CUDA、qint8 或 export。A1-B1 的 CPU
exact 性质现已闭合到本文限定的单机 toy 范围。后续 A2-E1 已独立选择 Fashion-MNIST/MLP；该
选择不扩大本文 backend 的性质或依赖范围。

## 16. Official references

以下页面均在 2026-07-23 只读核对：

- [PyTorch Get Started - Locally](https://pytorch.org/get-started/locally/)
- [PyTorch official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)
- [PyTorch 2.13 `torch.mul`](https://docs.pytorch.org/docs/2.13/generated/torch.mul.html)
- [PyTorch 2.13 `torch.sum`](https://docs.pytorch.org/docs/2.13/generated/torch.sum.html)
- [PyTorch 2.13 `torch.add`](https://docs.pytorch.org/docs/2.13/generated/torch.add.html)
- [PyTorch 2.13 `torch.clamp`](https://docs.pytorch.org/docs/2.13/generated/torch.clamp.html)
- [PyTorch 2.13 `Module.register_buffer`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
- [PyTorch 2.13 `inference_mode`](https://docs.pytorch.org/docs/2.13/generated/torch.autograd.grad_mode.inference_mode.html)
- [PyTorch 2.13 quantized `Linear`](https://docs.pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.Linear.html)

官方 API 文档说明了算子和 module 行为，但不能替代具体 wheel、dtype、device 和 graph 的实测。
因此 A1-B1 把完整 activation gate 作为启用前置条件。

## 17. Acceptance criteria for this decision

A1-B1 决策仅在以下条件全部成立时完成：

1. 首个 OS/Python/framework/device tuple 和精确 wheel channel/version/hash 唯一固定；
2. 三层 tensor shape、layout、storage、accumulator、scale 和 zero-point 可直接实施；
3. affine/ReLU 具有唯一 PyTorch operator mapping，rounding/overflow/saturation 语义明确；
4. qint8、CUDA、export、fusion 和其他候选有明确处置且不构成弱 fallback；
5. startup activation、禁用、rollback 和迁移条件全部 fail closed；
6. 现有 134 项测试、目标 backend 全域差分和安全测试复跑边界完整固定；
7. compiled artifact 只在内存/临时目录生成且禁止提交、记录、序列化或分发；
8. 性能主张严格限制于未来实际测试的本机 CPU 环境；
9. 下一 checkpoint 范围不包含业务模型、协调器、数据集或阶段 B；
10. 研究设计、安全文档、README、工作日志和治理检查与本决定一致。

本文最初只完成部署路线选择；后续实现 checkpoint 已满足对应 wheel、runtime、全域差分、
fail-closed 和 artifact 验收。该结果仍不表示认证、不可伪造、业务模型门控、跨平台部署或生产
安全已经完成。
