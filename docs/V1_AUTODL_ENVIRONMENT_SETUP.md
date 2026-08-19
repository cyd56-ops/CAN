# V1 AutoDL Environment Setup and Migration Guide

## 1. Purpose and boundary

本手册记录 V1-M1 已验证的 AutoDL 单 GPU 环境如何在同一实例重启后确认，或在更换服务器时重建。
它只处理 Conda、PyTorch/CUDA 和环境 provenance；不授权下载 CIFAR-100、安装预训练权重、选择训练
超参数或启动正式训练。那些动作仍以 `PROJECT_WORKLOG.md` 的唯一下一步为准。

本手册不替代 A2 的 CPU-only `requirements-ml.lock`。不得以 V1 GPU wheel 覆盖 A1/A2 已验收环境。

## 2. Frozen V1-M1 tuple

| Field | Frozen value |
| --- | --- |
| Platform/image | AutoDL single-GPU container; Miniconda/conda3, Ubuntu 22.04 image, CUDA 11.8 image label |
| OS | Ubuntu 22.04.1 LTS (`jammy`) |
| Conda environment | `can-v1` |
| Python | CPython 3.11.9 |
| GPU | NVIDIA RTX A4000, 16,376 MiB VRAM |
| Driver | 580.82.07 |
| PyTorch | `torch==2.13.0+cu126`, `torch.version.cuda == "12.6"` |
| torchvision | `torchvision==0.28.0+cu126` |
| torch wheel SHA-256 | `0f4e49e334e24b552f694f6315e0676fb3f816fb0f727871b9c6d1f73784cc25` |
| torchvision wheel SHA-256 | `92f53415dd68e56b6f912441997ab0e78fcd6245b1706ee6e88ce2df917248fa` |

`nvidia-smi` 显示的 CUDA 13.0 是 driver compatibility 值；正式记录的 PyTorch runtime 是 wheel
报告的 CUDA 12.6。两者不能互相替代。

## 3. Same-instance restart procedure

仅在 AutoDL 控制台对同一算力市场实例执行“关机/停止”，不要选择释放、重置系统、删除或弹性部署容器
停止。平台说明同一实例关机后通常保留数据和环境；连续关机 15 天或实例被释放时数据会被清空。重要数据
仍应同步到版本控制或持久化/外部存储。

关机前，保存小型恢复记录到数据盘：

```bash
mkdir -p /root/autodl-tmp/can-v1-backup
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate can-v1
conda env export --from-history > /root/autodl-tmp/can-v1-backup/conda-history.yml
python -m pip freeze > /root/autodl-tmp/can-v1-backup/pip-freeze.txt
python --version > /root/autodl-tmp/can-v1-backup/runtime.txt
python -c 'import torch; print(torch.__version__)' >> /root/autodl-tmp/can-v1-backup/runtime.txt
python -c 'import torchvision; print(torchvision.__version__)' >> /root/autodl-tmp/can-v1-backup/runtime.txt
python -c 'import torch; print(torch.version.cuda)' >> /root/autodl-tmp/can-v1-backup/runtime.txt
```

重新开机后逐行验证：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate can-v1
ls -lh /root/autodl-tmp/can-v1-backup
python --version
python -c 'import torch; print(torch.__version__)'
python -c 'import torchvision; print(torchvision.__version__)'
python -c 'import torch; print(torch.version.cuda)'
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

只有出现 Python 3.11.9、torch `2.13.0+cu126`、torchvision `0.28.0+cu126`、CUDA `12.6`、
`True` 和 NVIDIA RTX A4000，才可认定同一实例环境仍可使用。若 `conda activate can-v1` 失败，先运行
`conda env list`；不得在未确认实例类型或数据保留状态时直接重装、重置或释放。

## 4. Replacement-server procedure

更换服务器、释放后重建、系统重置或切换为弹性部署时，默认按全新环境处理。`/root` 中的 Conda
environment 不能被假定会迁移到另一台机器。新机器需要重新执行以下步骤：

1. 选择 Ubuntu 22.04 + Miniconda/conda3 的单 GPU 实例，优先相同 RTX A4000 16 GiB 配置；记录 GPU、
   VRAM、driver、OS、CPU/RAM 和磁盘可用空间。
2. 创建 Python 3.11.9 environment。旧 Conda 不支持 `libmamba` 时必须使用 `classic` solver：

   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda create -y --solver classic -n can-v1 python=3.11.9 pip -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
   conda activate can-v1
   python --version
   ```

3. 安装 pinned GPU wheels，随后执行依赖一致性检查：

   ```bash
   python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu126 'torch==2.13.0+cu126' 'torchvision==0.28.0+cu126'
   python -m pip check
   ```

4. 重新获得 primary wheel SHA-256，并与 section 2 比较。pip cache 是实现细节，优先在干净的审计目录
   下载 wheel 再计算摘要：

   ```bash
   mkdir -p /root/autodl-tmp/can-v1-wheel-audit
   python -m pip download --no-deps --only-binary=:all: --dest /root/autodl-tmp/can-v1-wheel-audit --index-url https://download.pytorch.org/whl/cu126 'torch==2.13.0+cu126' 'torchvision==0.28.0+cu126'
   sha256sum /root/autodl-tmp/can-v1-wheel-audit/*
   ```

5. 重跑 section 3 的版本/GPU 验证。若 GPU、driver、PyTorch、torchvision、wheel SHA-256 或
   `torch.version.cuda` 任一不同，不能声称新机器复现了 frozen tuple；必须将新 tuple 和 smoke
   结果记录到 `PROJECT_WORKLOG.md` 与 V1 决策后，才可用于正式训练。

## 5. Formal V1-M1 baseline procedure

本节仅在 `PROJECT_WORKLOG.md` 的唯一下一步为 `SERVER_REQUIRED` 的 V1-M1 baseline 时适用。每条命令均在
已冻结的 AutoDL A4000 实例中、仓库根目录执行。不得更换数据来源、模型、切分、预处理、超参数、阈值或
训练次数；本流程不进入 gate、性能测量、V2、Fiat--Shamir、ML-DSA 或 Stage B。

### 5.1 GitHub SSH access and checkout

先根据服务器状态选择唯一对应的操作。**首次访问当前实例**是指还没有 CAN checkout，或尚未配置
`can-github` SSH alias；**再次登录同一实例**是指 `/root/.ssh/id_ed25519_can_github` 与已有 CAN
checkout 均仍存在。后者只更新已有 checkout，绝不再次 clone、下载数据或重建 R2 artifact。实例被释放、
系统重置或私钥/checkout 丢失时，按 section 4 视为新服务器，再执行首次访问流程。

#### 5.1.1 First access to this instance

服务器必须拥有一个已添加到具备 `cyd56-ops/CAN` 访问权限 GitHub 账户的专用 Ed25519 public key。
先确认私钥文件存在且权限正确；若文件不存在才生成新 key，并仅把 `.pub` 内容添加到 GitHub。私钥绝不
复制、提交、上传或写入训练 artifact。

```bash
install -d -m 700 /root/.ssh
test -f /root/.ssh/id_ed25519_can_github
test -f /root/.ssh/id_ed25519_can_github.pub
chmod 600 /root/.ssh/id_ed25519_can_github
chmod 644 /root/.ssh/id_ed25519_can_github.pub
```

仅当以上 `test` 命令失败时，生成替换服务器专用 key；随后只复制输出的 public key 到 GitHub，绝不显示
或复制私钥：

```bash
ssh-keygen -t ed25519 -C "can-autodl-server" -f /root/.ssh/id_ed25519_can_github
cat /root/.ssh/id_ed25519_can_github.pub
```

为 CAN 专门配置 host alias，使 Git 每次登录都直接选择该 private key，而不是依赖短暂的
`ssh-agent` 状态。以下命令只在 `/root/.ssh/config` 缺少 `Host can-github` 时追加 stanza，不覆盖已有
SSH 配置：

```bash
touch /root/.ssh/config
if ! grep -Fqx 'Host can-github' /root/.ssh/config; then
  cat >> /root/.ssh/config <<'EOF'
Host can-github
    HostName github.com
    User git
    IdentityFile /root/.ssh/id_ed25519_can_github
    IdentitiesOnly yes
EOF
fi
chmod 600 /root/.ssh/config
ssh -T git@can-github
```

测试输出必须包含预期 GitHub 用户名。首次 checkout 才执行：

```bash
git clone git@can-github:cyd56-ops/CAN.git CAN
cd CAN
git switch main
git status --short
git rev-parse HEAD
```

#### 5.1.2 Repeat login to the same instance

再次登录、重开 shell 或重新启动同一实例时，SSH alias 会从 `/root/.ssh/config` 重新读取 private key，
因此不需要重新生成 key、重新添加 GitHub public key、再次 clone，或默认启动 `ssh-agent`。在已有
checkout 中执行：

```bash
cd <existing-CAN-checkout>
ssh -T git@can-github
git remote set-url origin git@can-github:cyd56-ops/CAN.git
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

`git status --short` 输出非空时停止并保留服务器改动；不得用 `reset --hard`、`checkout --` 或强制 pull
清理服务器目录。`ssh -T` 出现 `Permission denied (publickey)` 时，先确认 alias、private key 和文件权限：

```bash
ssh -G can-github | grep -E '^(hostname|user|identityfile|identitiesonly) '
ls -l /root/.ssh/id_ed25519_can_github /root/.ssh/id_ed25519_can_github.pub
ssh -T -o IdentitiesOnly=yes -i /root/.ssh/id_ed25519_can_github git@github.com
```

`Could not open a connection to your authentication agent` 仅表示当前 shell 没有运行 `ssh-agent`，不表示
GitHub public key 或服务器 private key 已丢失。上述 alias 对未加 passphrase 的 key 不依赖 agent；若
private key 设置了 passphrase，才在当前 shell 启动 agent 并加载 key：

```bash
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/id_ed25519_can_github
ssh -T git@can-github
```

新的 shell、重启或 agent 退出后需要重复这三行，但无需重新生成或上传 public key。

### 5.2 Environment and local preflight

激活已冻结环境，并确认 Python、GPU 与 two primary wheels：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate can-v1
python --version
python -c 'import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -m pip install -e '.[dev]'
python -m pip check
```

预期为 Python `3.11.9`、NVIDIA RTX A4000、`torch==2.13.0+cu126`、
`torchvision==0.28.0+cu126`、`torch.version.cuda == "12.6"` 及 CUDA available。任一不符时停止；
同一实例按 section 3 复核，换服或重置按 section 4 重建并更新工作日志，不能把不同 tuple 称为本次冻结
环境。

在下载前运行不产生数据、权重或正式结果的 focused test suite：

```bash
python -m pytest \
  tests/unit/test_v1_cifar100_resnet.py \
  tests/unit/test_v1_m1_adapter.py \
  tests/unit/test_v1_m1_baseline.py \
  tests/security/test_v1_m1_route_security.py \
  tests/security/test_v1_m1_artifact_security.py
```

### 5.3 Official archive download and validation

唯一允许的下载入口是以下首方 URL。数据、解压内容和 artifact 均在 ignored roots 中，禁止提交、上传、
镜像或再分发：

```bash
mkdir -p data/v1-m1
curl --fail --location --retry 3 --retry-delay 5 \
  --output data/v1-m1/cifar-100-python.tar.gz \
  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

完成后必须分别检查 size、SHA-256 和 MD5：

```bash
stat --printf='%s\n' data/v1-m1/cifar-100-python.tar.gz
sha256sum data/v1-m1/cifar-100-python.tar.gz
md5sum data/v1-m1/cifar-100-python.tar.gz
```

唯一接受值依次为：

```text
169001437
85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7
eb9058c3a382ffc7106e4002c42a8d85
```

任一值不匹配即拒绝 archive，不得解压或训练。记录失败输出后，只能删除准确目标
`data/v1-m1/cifar-100-python.tar.gz` 并从同一首方 URL 重试；不得改用镜像。不得让两个下载器同时写入
同一个 archive。若服务器已经安装 `aria2c` 且单连接下载不可接受，可先停止 `curl`，再仅对同一 URL 使用
`aria2c --continue=true --max-connection-per-server=8 --split=8`；最终仍以三项完整性校验为唯一接受条件。

三项均通过后显式解压：

```bash
tar -xzf data/v1-m1/cifar-100-python.tar.gz -C data/v1-m1
```

runner 会在训练前重新验证 archive、解压成员与 canonical decoded dataset digest；因此不允许手动修改
`cifar-100-python/train`、`test` 或 `meta`。

### 5.4 Two pre-registered baseline runs

只有 section 5.2--5.3 均成功后，依序执行这两个独立 run。不得并发运行、重试、增加第三个 seed 或在
观察到结果后修改配置。训练器没有 CLI；以下 Python API 是唯一正式调用。完成数据加载和 deterministic
setup 后，run 立即输出 `V1-M1 training started`，并在每个 train、validation 与 test batch 完成时以
`\r` 和 `flush=True` 刷新一条固定宽度的 `V1-M1 progress` 进度条。进度条只含 run/seed、stage、epoch 和
batch 计数，初始为 `0.00%`，最终 test batch 后为 `100.00%`；train loss、validation loss/top-1 与当前
best-validation checkpoint 的指标只写入 report，不在 stdout 重复打印。selected
state、manifest 和 report 均原子写入 ignored `artifacts/v1-m1/run-{1,2}/` 成功后，run 重绘完成状态并输出
`V1-M1 training completed`。
该可观测性不读取或写入样本、预测、权重或 secret，也不改变训练、随机性、选模或 artifact。

```bash
PYTHONHASHSEED=1729 CUBLAS_WORKSPACE_CONFIG=:4096:8 python - <<'PY'
from pathlib import Path

import torch

from can.experiments.v1_m1_baseline import run_v1_m1_baseline

result = run_v1_m1_baseline(Path("data/v1-m1"), 1, torch.device("cuda:0"))
print(result.selected_epoch, result.test.top1_percent, result.artifacts.root)
PY
```

```bash
PYTHONHASHSEED=1730 CUBLAS_WORKSPACE_CONFIG=:4096:8 python - <<'PY'
from pathlib import Path

import torch

from can.experiments.v1_m1_baseline import run_v1_m1_baseline

result = run_v1_m1_baseline(Path("data/v1-m1"), 2, torch.device("cuda:0"))
print(result.selected_epoch, result.test.top1_percent, result.artifacts.root)
PY
```

若 run 已创建对应 `run-1` 或 `run-2` 目录，runner 会拒绝覆盖。应保留失败输出和已有 artifact，停止并在
工作日志记录原因；不得删除结果后隐式重跑。

已启动的 Python 进程不会加载后续源码修改；R1 与 R2 均须在其开始时记录 `git rev-parse HEAD`。若为
R2 部署包含 batch stdout progress 的新 checkpoint，必须在工作日志中保留 R1/R2 的两个 source HEAD 和这项
observability-only 差异，不能把它误记为相同源码运行。

### 5.5 Completion handoff

两次 run 都结束后，保留每个 run 的 terminal output、`manifest.json` 与 `report.json`。验收前必须确认：

- 两次 validation 与 test top-1 均至少 `70.00%`；
- 两次 validation top-1 的绝对差和 test top-1 的绝对差均不超过 `2.00` percentage points；
- 两个 manifest 均包含相同已验证 archive identity、完整 environment/dataset/state/prediction 摘要；
- 只按 validation top-1 选择后续 gate 的 accepted state，平局取较小 seed；test 结果不参与选择。

满足条件前，不将任一 weight 标记为 accepted，不开始 gate 或性能报告。权重、数据和 generated artifact
持续保持 ignored；仅公开摘要、命令、环境和验收结果可进入工作日志。

## 6. References

- AutoDL: [实例中的数据保留](https://www.autodl.com/docs/instance_data/)
- AutoDL: [本地数据盘](https://www.autodl.com/docs/local_disk/)
- `docs/V1_MODEL_EXPERIMENT_DECISION.md` section 8
- `PROJECT_WORKLOG.md` D-038
