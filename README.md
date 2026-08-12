# CAN: Lattice-Based Neural Model Access Control

CAN 是一个非生产科研原型，研究如何将格密码验证关系编译为固定的定点或量化神经网络，并用其结果对 CNN/DNN 模型能力实施 fail-closed 访问控制。

当前仓库已完成 A0-v1 toy 精确整数 oracle、A1 构造无关数值/算子规格、
`CAN-RELU-EXACT-v1` dependency-free exact-integer conformance backend，以及首个 CPU-only
PyTorch exact-integer backend `CAN-TORCH-CPU-EXACT-v1`，并已通过当前 toy 域全域差分与防御性
安全测试。A2-E1 已安装并核验 CPU-only torchvision/Fashion-MNIST，实现严格输入校验的
`784 -> 256 -> 128 -> 10` MLP 和确定性无门控训练/评估入口；两次十 epoch 运行均得到 `88.08%`
test accuracy 和相同预测/模型摘要。单一协调器与二元前置硬门控也已实现：只有本地 A1-B1 的
精确 `NUMERIC_ACCEPT` 才会在协调器提交 allow 后调用一次 protected MLP。真实 gate 实验的
10,000 个标签与 baseline 全部一致，拒绝探针为零 protected-model calls。A0 只用于研究数值
解锁关系，不是数字签名、身份认证或生产安全方案。A2-E2 已实现独立 `784 -> 64 -> 2` public
coarse-model baseline，以及默认关闭、本地绑定的 `DENY`/`PUBLIC`/`PROTECTED` 三态协调器；79 项
focused 测试覆盖调用隔离、固定 version-2 响应、并发、异常、不可升级和无 fallback。两次 public
baseline 训练均得到 `99.85%` coarse test accuracy 和相同预测/模型摘要。独立 trusted
materializer 已按固定协议重新物化两个 accepted state，严格校验 manifest、文件摘要和 canonical
state digest 后，由 no-training evaluator 完成真实 10,000-image 三态报告；protected/public 预测
摘要均与 baseline 一致，拒绝探针为零模型调用。A3-v1 现已实现默认关闭的 133 字节请求绑定协议壳，
覆盖规范输入摘要、60 秒 challenge、可信 nonce 状态、原子单次消费、evidence-only verifier 边界和
并发 replay 验收语义。A4 现已选择 GPV PFDH 短原像公开验证关系，冻结非生产
`A4-GPV-PFDH-TOY-v1` 的 105 字节 proof、SHAKE256 hash-to-syndrome、公开矩阵关系和神经
soundness/completeness 义务，并实现无私钥 exact reference 与 A3 evidence adapter。该 toy profile
及公开 gadget 测试夹具不提供不可伪造性。A4-C1 已冻结并实现
`CAN-RELU-A4-PFDH-TOY-v1` dependency-free exact graph：canonical 输入固定为 `(y,z)`，拓扑
`80 -> 3600 -> 1153 -> 1`，使用整数 affine/ReLU point pulses、norm violation 和最终硬 AND，
并通过 A3 neural evidence adapter 保持原子 consume/单次 protected call 边界。
V1-P2 已选择 Boudgoust--Takahashi 给出的 `FSwA-S` module-lattice protocol 的底层交互式 Sigma
形式。它在 `R_q=Z_q[X]/(X^N+1)` 上使用 `Abar=[A|I]`、`t=Abar*s`、commitment `u=Abar*y`、
sparse ternary polynomial challenge、response `z=y+c*s` 和 bounded-uniform rejection，公开验证核心
为 `Abar*z=u+c*t` 与 coefficient infinity-norm bound。V1-P1 普通矩阵 SIS 方案保留为历史 baseline；
当前已冻结非生产 `N=8,q=257,k_mod=2,ell_mod=2,eta=1,gamma=8,kappa=2,B=6` conformance
profile，并实现 canonical polynomial parser、公开 registry、coefficient-domain exact reference、
A3-v2 commit-first 单次终态协调器和 evidence adapter。`V1-P2-PSR-E1` 已实现非生产 generated-key
fixture、domain-separated SHAKE256 secret/mask/challenge sampler、single-attempt prover、A3-v2
fresh-transcript retry harness、公开摘要 manifest 和 exact differential。`CAN-RELU-V1-MSIS-COEFF-v1`
现已将固定公开 profile 编译为 dependency-free exact integer affine/ReLU graph
`56 -> 11056 -> 17 -> 1`，以 coefficient residual point pulses、norm violations 和最终硬 AND
实现 V1-P2 relation；neural adapter 只产生 A3-v2 evidence，accept 最多一次 protected call，relation
reject 与 foreign-route bytes 均为零 protected calls。生产 keygen/prover、密码安全参数、NTT、
PyTorch/qint8/CUDA/export 和性能结论仍未实现。

长期路线固定为 `V0 -> V1-prep -> V1 -> V2`：V0 是 A0/A1/A2 toy LWE 数值解锁与硬门控；V1-prep
是 A3 请求绑定/新鲜性和 A4 canonical `(y,z)` 神经代数内核；V1-P2 已完成 reviewed Module-SIS
challenge-response/身份协议选择和非生产 exact/A3-v2 conformance 协议壳，并已实现独立
generated-key/sampler/single-attempt/retry 实验边界及 `V1-C1-MSIS` coefficient-domain neural
construction。下一步先在服务器冻结 V1-M1 GPU/software tuple，之后才开始 CIFAR-100/ResNet-18
baseline。V2 才研究 ML-DSA 标准 verifier。V0、V1 和 V2 必须作为独立
可复现代码路线共存，后续路线不得
重命名、改写或覆盖前序路线。V1-prep 不提供身份认证，V2 不属于首篇论文 MVP。

## Repository map

- `docs/RESEARCH_DESIGN.md`：研究问题、阶段范围、形式化义务和论文定位；
- `docs/A0_PROTOCOL_SPEC.md`：A0-v1 toy LWE 数值解锁的精确关系与编码；
- `docs/A1_NUMERICAL_SPEC.md`：A1 构造无关的 tensor、算子、误差和证明契约；
- `docs/A1_CONSTRUCTION_DECISION.md`：A1 固定 ReLU 主构造、对照基线和证明路线；
- `docs/A1_BACKEND_DECISION.md`：A1 首个 PyTorch CPU exact-integer 物理映射和复测边界；
- `docs/A2_MODEL_EXPERIMENT_PROTOCOL.md`：Fashion-MNIST/MLP baseline、指标和二元硬门控实验协议；
- `docs/A2_CAPABILITY_EXPERIMENT_SPEC.md`：A2-E2 独立 public model、三态提交和隔离验收规格；
- `docs/A3_CHALLENGE_RESPONSE_SPEC.md`：A3-v1 请求绑定、challenge/nonce 状态和原子 consume 规格；
- `docs/A4_GPV_RELATION_SPEC.md`：A4 GPV-PFDH toy 公钥关系、105 字节 proof 和神经证明契约；
- `docs/A4_NEURAL_CONSTRUCTION_DECISION.md`：A4-C1 固定 ReLU graph、范围账本和全输入证明；
- `docs/V1_PROTOCOL_SELECTION_DECISION.md`：保留的 V1-P1 普通矩阵 SIS 历史设计 baseline；
- `docs/V1_MODULE_SIS_PROTOCOL_DECISION.md`：当前 V1-P2 FSwA-S Module-SIS Sigma 协议、商环、
  commit-first transcript、canonical polynomial encoding、安全游戏和 neural 边界；
- `docs/V1_PROVER_SAMPLER_REJECTION_SPEC.md`：V1-P2 非生产 prover/sampler/rejection 的 toy domain、
  deterministic seed、bounded-uniform emit/abort、fresh retry、测试向量和安全主张边界；
- `docs/V1_MODEL_EXPERIMENT_DECISION.md`：V1 CIFAR-100/CIFAR-style ResNet-18 输入、模型和实验边界；
- `SECURITY.md`：信任模型、安全边界和明确不保证的性质；
- `PROJECT_WORKLOG.md`：当前状态、任务和唯一下一步；
- `src/can/reference/`：A0-v1 oracle、A4 public relation，以及 V1-P2 公开 profile/parser/系数域 exact relation；
- `src/can/verifier/`：A1-C1 compiled profile、dependency-free graph、A1-B1 PyTorch CPU backend，
  以及 A4-C1/V1-C1 dependency-free sparse exact graph 和 evidence-only adapters；
- `src/can/model/a2_mlp.py`：A2-E1 float32 CPU MLP 与严格业务 tensor 校验；
- `src/can/model/a2_public_mlp.py`：A2-E2 独立 two-class public MLP、严格输入与 coarse-label 映射；
- `src/can/experiments/a2_baseline.py`：固定数据核验、确定性训练、评估和报告入口；
- `src/can/experiments/a2_public_baseline.py`：独立 public 模型的确定性训练、评估和报告入口；
- `src/can/access/a2_gate.py`：A2-E1 单一协调器、固定响应、前置硬门控和内部计数/计时；
- `src/can/experiments/a2_gate.py`：10,000 标签等价、拒绝零调用和门控延迟实验入口；
- `src/can/access/a2_capability.py`：A2-E2 默认关闭的本地策略、三态协调器、固定响应和调用计数；
- `src/can/access/a3_protocol.py`：A3-v1 默认关闭的 canonical message/parser、nonce store、请求绑定和 freshness 协议壳；
- `src/can/access/a4_adapter.py`：把 A4 exact relation 映射为 A3 message/identity-bound evidence；
- `src/can/access/a3_v2.py`、`src/can/access/v1_adapter.py`：V1-P2 commit-first 单次终态状态机与
  exact/neural evidence adapters；
- `src/can/experiments/a2_capability.py`：只接受已验收模型 state 的三态标签/延迟报告入口；
- `src/can/experiments/a2_materialize.py`：受信确定性重建、本地 state/manifest 校验和报告编排入口；
- `src/can/experiments/v1_psr.py`：V1-P2 非生产 generated-key、SHAKE256 sampler、single-attempt
  prover、A3-v2 retry harness 和无 secret 公开 vector manifest；
- `tests/`：单元、集成、差分和安全测试；
- `paper/`：本地参考论文，版权与再分发许可确认前不纳入版本控制。

## Development

项目固定使用 Python 3.11。创建环境并安装轻量开发依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

在缺少系统 `ensurepip` 的 Debian/Ubuntu 环境中，`venv` 仍会创建隔离解释器，但不会创建 pip。此时使用已验证的 fallback：

```bash
python3 -m pip --python .venv install 'pip==24.0'
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps -e .
```

运行完整质量检查：

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src tests
./scripts/check_governance_docs.sh
```

PyTorch 和 torchvision 位于可选的 `ml` 依赖组，轻量 bootstrap 不安装该大型依赖组。复现
A1-B1 时只安装官方 CPU wheel，不安装 torchvision：

```bash
.venv/bin/python -m pip install \
  --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.13.0+cpu'
```

当前已核验 CPython 3.11 Linux x86_64 的 torch 和 torchvision CPU wheels。A2 环境的完整解析
闭包位于 `requirements-ml.lock`；复现时仍只能使用官方 CPU index，不能由默认 PyPI、CUDA wheel
或未锁定版本替代：

```bash
.venv/bin/python -m pip install \
  --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu \
  -r requirements-ml.lock
PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_baseline --repeat 1
PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_baseline --repeat 2
.venv/bin/python -m can.experiments.a2_baseline --compare
PYTHONHASHSEED=20260723 .venv/bin/python -m can.experiments.a2_gate --run
PYTHONHASHSEED=20260730 .venv/bin/python -m can.experiments.a2_public_baseline --repeat 1
PYTHONHASHSEED=20260730 .venv/bin/python -m can.experiments.a2_public_baseline --repeat 2
.venv/bin/python -m can.experiments.a2_public_baseline --compare
.venv/bin/python -m can.experiments.a2_materialize --run
```

Fashion-MNIST cache 和 protected/public baseline/gate JSON reports 分别位于 ignored `data/a2/` 与
`artifacts/a2/`，不属于提交候选。materializer 只在 `artifacts/a2/local-states/` 保存 CPU float32
`state_dict` 与 canonical manifest；这些本地 state、manifest 和 `capability.json` 均保持 ignored，
不得提交、上传或作为发布 artifact。数据获取、hash、license、硬件和实验限制见 A2 协议。
