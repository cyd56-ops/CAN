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

## 5. Before a formal training run

在服务器上执行任何数据下载前，先确认工作日志的唯一下一步不再是 `LOCAL_OK` 的协议冻结。随后才允许：

1. 检出已记录的 Git commit，并保存 `git rev-parse HEAD` 与 `git status --short`；
2. 依据已冻结的数据集来源、SHA-256 和许可证下载 CIFAR-100；
3. 校验 archive digest、文件数量、shape、dtype、label range 与 fine-label ordering；
4. 以预注册的 seeds、batch、optimizer、scheduler、epochs 和 checkpoint rule 完成两次独立 baseline；
5. 将数据、weights、optimizer state、checkpoint 和 profiler output 保持在 ignored roots，且不提交。

环境可用不等于训练协议已冻结，也不等于正式实验已开始。

## 6. References

- AutoDL: [实例中的数据保留](https://www.autodl.com/docs/instance_data/)
- AutoDL: [本地数据盘](https://www.autodl.com/docs/local_disk/)
- `docs/V1_MODEL_EXPERIMENT_DECISION.md` section 8
- `PROJECT_WORKLOG.md` D-038
