"""A2-E1 Fashion-MNIST MLP 的确定性无门控 baseline。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import struct
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
import torchvision  # type: ignore[import-untyped]
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import FashionMNIST  # type: ignore[import-untyped]
from torchvision.transforms import ToTensor  # type: ignore[import-untyped]

from can.model.a2_mlp import (
    A2_CLASS_COUNT,
    A2_EXPERIMENT_ID,
    A2_PARAMETER_COUNT,
    A2FashionMNISTMLP,
    validate_a2_images,
    validate_a2_labels,
)

GLOBAL_SEED: Final = 20_260_723
SPLIT_SEED: Final = 20_260_724
TRAIN_LOADER_SEED: Final = 20_260_725
TRAIN_SIZE: Final = 55_000
VALIDATION_SIZE: Final = 5_000
TEST_SIZE: Final = 10_000
TRAIN_BATCH_SIZE: Final = 128
EVALUATION_BATCH_SIZE: Final = 256
EPOCH_COUNT: Final = 10
SMOKE_ACCURACY_PERCENT: Final = 85.0
A2_DATA_ROOT: Final = Path("data/a2")
A2_REPORT_ROOT: Final = Path("artifacts/a2")
TORCH_VERSION: Final = "2.13.0+cpu"
TORCHVISION_VERSION: Final = "0.28.0+cpu"
NUMPY_VERSION: Final = "2.4.4"
PILLOW_VERSION: Final = "12.2.0"


@dataclass(frozen=True, slots=True)
class A2DataResource:
    """固定一个 Fashion-MNIST 压缩资源及其完整性摘要。"""

    filename: str
    md5: str
    sha256: str


A2_DATA_RESOURCES: Final = (
    A2DataResource(
        "train-images-idx3-ubyte.gz",
        "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    ),
    A2DataResource(
        "train-labels-idx1-ubyte.gz",
        "25c81989df183df01b3e8a0aad5dffbe",
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    ),
    A2DataResource(
        "t10k-images-idx3-ubyte.gz",
        "bef4ecab320f06d8554ea6380940ec79",
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    ),
    A2DataResource(
        "t10k-labels-idx1-ubyte.gz",
        "bb300cfdad3c16e7a12a480ee83cd310",
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    ),
)


class A2BaselineError(RuntimeError):
    """表示 A2-E1 环境、数据或实验结果不满足固定协议。"""


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """记录一个固定训练 epoch 的聚合指标。"""

    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy_percent: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """记录不暴露单样本 logits 的分类评估结果。"""

    loss: float
    accuracy_percent: float
    correct: int
    total: int
    per_class_correct: tuple[int, ...]
    per_class_count: tuple[int, ...]
    per_class_accuracy_percent: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    predictions_sha256: str


@dataclass(frozen=True, slots=True)
class A2DataBundle:
    """保存固定 split 的 loader、test dataset 与索引摘要。"""

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test_dataset: Dataset[tuple[torch.Tensor, int]]
    train_indices_sha256: str
    validation_indices_sha256: str


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_int64_values(values: torch.Tensor | Sequence[int]) -> str:
    digest = hashlib.sha256()
    iterable = values.tolist() if type(values) is torch.Tensor else values
    for value in iterable:
        if type(value) is not int:
            raise A2BaselineError("int64 hash input contains a non-exact integer")
        digest.update(struct.pack("<q", value))
    return digest.hexdigest()


def _validate_environment() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise A2BaselineError("A2-E1 requires Linux x86_64")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise A2BaselineError("A2-E1 requires CPython 3.11")
    expected_versions = {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "numpy": NUMPY_VERSION,
        "pillow": PILLOW_VERSION,
    }
    for distribution, expected in expected_versions.items():
        if importlib.metadata.version(distribution) != expected:
            raise A2BaselineError(f"unsupported {distribution} version")
    if (
        str(torch.__version__) != TORCH_VERSION
        or str(torchvision.__version__) != TORCHVISION_VERSION
    ):
        raise A2BaselineError("runtime and distribution versions diverged")
    if torch.version.cuda is not None or torch.version.hip is not None or torch.cuda.is_available():
        raise A2BaselineError("A2-E1 does not support accelerator builds")
    if not torchvision.extension._has_ops():
        raise A2BaselineError("torchvision CPU operators are unavailable")


def _configure_determinism() -> None:
    if os.environ.get("PYTHONHASHSEED") != str(GLOBAL_SEED):
        raise A2BaselineError(f"PYTHONHASHSEED must equal {GLOBAL_SEED}")
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")


def _validate_data_resources(data_root: Path) -> dict[str, dict[str, str]]:
    raw_root = data_root / "FashionMNIST" / "raw"
    expected_names = {resource.filename for resource in A2_DATA_RESOURCES}
    observed_names = {path.name for path in raw_root.glob("*.gz") if path.is_file()}
    if observed_names != expected_names:
        raise A2BaselineError("Fashion-MNIST compressed resource set changed")

    result: dict[str, dict[str, str]] = {}
    for resource_item in A2_DATA_RESOURCES:
        path = raw_root / resource_item.filename
        observed_md5 = _file_digest(path, "md5")
        observed_sha256 = _file_digest(path, "sha256")
        if observed_md5 != resource_item.md5 or observed_sha256 != resource_item.sha256:
            raise A2BaselineError(
                f"Fashion-MNIST resource digest changed: {resource_item.filename}"
            )
        result[resource_item.filename] = {"md5": observed_md5, "sha256": observed_sha256}
    return result


def _build_split_indices() -> tuple[torch.Tensor, torch.Tensor, str, str]:
    generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    permutation = torch.randperm(TRAIN_SIZE + VALIDATION_SIZE, generator=generator)
    train_indices = permutation[:TRAIN_SIZE].contiguous()
    validation_indices = permutation[TRAIN_SIZE:].contiguous()
    return (
        train_indices,
        validation_indices,
        _hash_int64_values(train_indices),
        _hash_int64_values(validation_indices),
    )


def _validate_dataset(dataset: FashionMNIST, *, train: bool) -> None:
    expected_size = TRAIN_SIZE + VALIDATION_SIZE if train else TEST_SIZE
    if len(dataset) != expected_size:
        raise A2BaselineError("Fashion-MNIST split size changed")
    if (
        type(dataset.data) is not torch.Tensor
        or dataset.data.dtype is not torch.uint8
        or tuple(dataset.data.shape) != (expected_size, 28, 28)
        or type(dataset.targets) is not torch.Tensor
        or dataset.targets.dtype is not torch.int64
        or tuple(dataset.targets.shape) != (expected_size,)
        or int(dataset.targets.min().item()) != 0
        or int(dataset.targets.max().item()) != A2_CLASS_COUNT - 1
    ):
        raise A2BaselineError("Fashion-MNIST decoded tensor contract changed")
    image, label = dataset[0]
    validate_a2_images(cast(torch.Tensor, image).unsqueeze(0).contiguous())
    if type(label) is not int or not 0 <= label < A2_CLASS_COUNT:
        raise A2BaselineError("Fashion-MNIST sample label is non-canonical")


def _load_data(data_root: Path) -> A2DataBundle:
    transform = ToTensor()
    full_train = FashionMNIST(root=data_root, train=True, transform=transform, download=False)
    test = FashionMNIST(root=data_root, train=False, transform=transform, download=False)
    _validate_dataset(full_train, train=True)
    _validate_dataset(test, train=False)

    train_indices, validation_indices, train_hash, validation_hash = _build_split_indices()
    train_subset = Subset(full_train, train_indices.tolist())
    validation_subset = Subset(full_train, validation_indices.tolist())
    loader_generator = torch.Generator(device="cpu").manual_seed(TRAIN_LOADER_SEED)
    train_loader = DataLoader(
        train_subset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return A2DataBundle(
        train_loader,
        validation_loader,
        test_loader,
        cast(Dataset[tuple[torch.Tensor, int]], test),
        train_hash,
        validation_hash,
    )


def _train_epoch(
    model: A2FashionMNISTMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Adam,
    criterion: nn.CrossEntropyLoss,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        validate_a2_images(images)
        validate_a2_labels(labels, images.shape[0])
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
    if sample_count < 1:
        raise A2BaselineError("training loader produced no samples")
    return loss_sum / sample_count


def _train_model(
    data: A2DataBundle, *, emit_progress: bool
) -> tuple[A2FashionMNISTMLP, tuple[EpochMetrics, ...]]:
    model = A2FashionMNISTMLP().to(device="cpu", dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    epoch_metrics: list[EpochMetrics] = []
    for epoch in range(1, EPOCH_COUNT + 1):
        training_loss = _train_epoch(model, data.train_loader, optimizer, criterion)
        validation = _evaluate(model, data.validation_loader)
        epoch_metrics.append(
            EpochMetrics(epoch, training_loss, validation.loss, validation.accuracy_percent)
        )
        if emit_progress:
            print(
                f"epoch={epoch} train_loss={training_loss:.8f} "
                f"validation_loss={validation.loss:.8f} "
                f"validation_accuracy={validation.accuracy_percent:.4f}%",
                flush=True,
            )
    return model, tuple(epoch_metrics)


def _evaluate(
    model: A2FashionMNISTMLP,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> EvaluationMetrics:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    confusion = torch.zeros((A2_CLASS_COUNT, A2_CLASS_COUNT), dtype=torch.int64)
    prediction_values: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            validate_a2_images(images)
            validate_a2_labels(labels, images.shape[0])
            logits = model(images)
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
            predictions = logits.argmax(dim=1)
            prediction_values.extend(int(value) for value in predictions.tolist())
            total += labels.shape[0]
            correct += int((predictions == labels).sum().item())
            bins = torch.bincount(
                labels * A2_CLASS_COUNT + predictions,
                minlength=A2_CLASS_COUNT * A2_CLASS_COUNT,
            ).reshape(A2_CLASS_COUNT, A2_CLASS_COUNT)
            confusion += bins
    if total < 1:
        raise A2BaselineError("evaluation loader produced no samples")
    per_class_count = confusion.sum(dim=1)
    per_class_correct = confusion.diag()
    per_class_accuracy = tuple(
        float(per_class_correct[index].item()) * 100.0 / int(per_class_count[index].item())
        for index in range(A2_CLASS_COUNT)
    )
    return EvaluationMetrics(
        loss=loss_sum / total,
        accuracy_percent=correct * 100.0 / total,
        correct=correct,
        total=total,
        per_class_correct=tuple(int(value) for value in per_class_correct.tolist()),
        per_class_count=tuple(int(value) for value in per_class_count.tolist()),
        per_class_accuracy_percent=per_class_accuracy,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion.tolist()),
        predictions_sha256=_hash_int64_values(prediction_values),
    )


def _hash_model_state(model: A2FashionMNISTMLP) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("<Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _measure_serialized_state(model: A2FashionMNISTMLP) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="can-a2-state-") as directory:
        path = Path(directory) / "state.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size, _file_digest(path, "sha256")


def _nearest_rank(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank] / 1_000.0


def _time_model(model: A2FashionMNISTMLP, batches: Sequence[torch.Tensor]) -> dict[str, float]:
    if not batches:
        raise A2BaselineError("latency batches must not be empty")
    model.eval()
    with torch.inference_mode():
        for index in range(100):
            model(batches[index % len(batches)])
        elapsed_ns: list[int] = []
        for index in range(1_000):
            batch = batches[index % len(batches)]
            start = time.perf_counter_ns()
            model(batch)
            elapsed_ns.append(time.perf_counter_ns() - start)
    return {
        "median_us": _nearest_rank(elapsed_ns, 0.50),
        "p95_us": _nearest_rank(elapsed_ns, 0.95),
        "p99_us": _nearest_rank(elapsed_ns, 0.99),
    }


def _measure_latency(
    model: A2FashionMNISTMLP, test_dataset: Dataset[tuple[torch.Tensor, int]]
) -> dict[str, object]:
    first_images: list[torch.Tensor] = []
    for index in range(1_000):
        image, _ = test_dataset[index]
        if type(image) is not torch.Tensor:
            raise A2BaselineError("test transform returned a non-tensor image")
        first_images.append(image)
    stacked = torch.stack(first_images).contiguous()
    validate_a2_images(stacked)
    batch_one = tuple(stacked[index : index + 1] for index in range(stacked.shape[0]))
    batch_256 = (stacked[:EVALUATION_BATCH_SIZE].contiguous(),)
    return {
        "method": "100 warm-up plus 1000 perf_counter_ns observations, one CPU thread",
        "batch_1": _time_model(model, batch_one),
        "batch_256": _time_model(model, batch_256),
    }


def _cpu_model_name() -> str:
    completed = subprocess.run(
        ["lscpu", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if type(payload) is not dict or type(payload.get("lscpu")) is not list:
        raise A2BaselineError("lscpu JSON has the wrong schema")
    for entry in payload["lscpu"]:
        if type(entry) is dict and entry.get("field") == "Model name:":
            value = entry.get("data")
            if type(value) is str and value:
                return value
    raise A2BaselineError("lscpu JSON does not contain a CPU model")


def _environment_report() -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model_name(),
        "cpu_affinity": affinity,
        "wsl2": "microsoft" in platform.release().lower(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "pillow": importlib.metadata.version("pillow"),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def _determinism_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_report(report: dict[str, object], repeat: int) -> Path:
    A2_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = A2_REPORT_ROOT / f"baseline-repeat-{repeat}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def run_a2_baseline(repeat: int) -> Path:
    """执行一次固定 A2-E1 baseline 并写入 ignored JSON 报告。"""
    if type(repeat) is not int or repeat not in (1, 2):
        raise A2BaselineError("repeat must be exactly 1 or 2")
    _validate_environment()
    _configure_determinism()
    resource_hashes = _validate_data_resources(A2_DATA_ROOT)
    data = _load_data(A2_DATA_ROOT)

    model, epoch_metrics = _train_model(data, emit_progress=True)

    test_metrics = _evaluate(model, data.test_loader)
    if test_metrics.total != TEST_SIZE:
        raise A2BaselineError("official test sample count changed")
    if test_metrics.accuracy_percent < SMOKE_ACCURACY_PERCENT:
        raise A2BaselineError("A2-E1 test accuracy is below the smoke floor")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    if parameter_count != A2_PARAMETER_COUNT:
        raise A2BaselineError("A2-E1 parameter count changed")
    state_sha256 = _hash_model_state(model)
    serialized_size, serialized_sha256 = _measure_serialized_state(model)
    latency = _measure_latency(model, data.test_dataset)

    deterministic_payload: dict[str, object] = {
        "experiment_id": A2_EXPERIMENT_ID,
        "resource_hashes": resource_hashes,
        "train_indices_sha256": data.train_indices_sha256,
        "validation_indices_sha256": data.validation_indices_sha256,
        "epochs": [
            {
                "epoch": item.epoch,
                "training_loss_hex": item.training_loss.hex(),
                "validation_loss_hex": item.validation_loss.hex(),
                "validation_accuracy_percent_hex": item.validation_accuracy_percent.hex(),
            }
            for item in epoch_metrics
        ],
        "test_loss_hex": test_metrics.loss.hex(),
        "test_accuracy_percent_hex": test_metrics.accuracy_percent.hex(),
        "test_confusion_matrix": test_metrics.confusion_matrix,
        "test_predictions_sha256": test_metrics.predictions_sha256,
        "model_state_sha256": state_sha256,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": A2_EXPERIMENT_ID,
        "repeat": repeat,
        "environment": _environment_report(),
        "data": {
            "root": str(A2_DATA_ROOT),
            "resources": resource_hashes,
            "train_size": TRAIN_SIZE,
            "validation_size": VALIDATION_SIZE,
            "test_size": TEST_SIZE,
            "train_indices_sha256": data.train_indices_sha256,
            "validation_indices_sha256": data.validation_indices_sha256,
        },
        "training": {
            "global_seed": GLOBAL_SEED,
            "split_seed": SPLIT_SEED,
            "loader_seed": TRAIN_LOADER_SEED,
            "epochs": [asdict(item) for item in epoch_metrics],
        },
        "test": asdict(test_metrics),
        "model": {
            "topology": "784->256->128->10",
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "state_sha256": state_sha256,
            "temporary_serialized_bytes": serialized_size,
            "temporary_serialized_sha256": serialized_sha256,
        },
        "latency": latency,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "determinism_fingerprint": _determinism_fingerprint(deterministic_payload),
    }
    return _write_report(report, repeat)


def compare_a2_repeats() -> str:
    """比较两份固定报告并返回一致的确定性指纹。"""
    fingerprints: list[str] = []
    for repeat in (1, 2):
        path = A2_REPORT_ROOT / f"baseline-repeat-{repeat}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise A2BaselineError("A2-E1 report is missing or malformed") from error
        if type(payload) is not dict or set(payload) != {
            "schema_version",
            "experiment_id",
            "repeat",
            "environment",
            "data",
            "training",
            "test",
            "model",
            "latency",
            "peak_rss_kib",
            "determinism_fingerprint",
        }:
            raise A2BaselineError("A2-E1 report schema changed")
        fingerprint = payload.get("determinism_fingerprint")
        if (
            payload.get("schema_version") != 1
            or payload.get("experiment_id") != A2_EXPERIMENT_ID
            or payload.get("repeat") != repeat
            or type(fingerprint) is not str
            or len(fingerprint) != 64
        ):
            raise A2BaselineError("A2-E1 report identity changed")
        fingerprints.append(fingerprint)
    if fingerprints[0] != fingerprints[1]:
        raise A2BaselineError("A2-E1 repeated runs are not deterministic")
    return fingerprints[0]


def main(argv: Sequence[str] | None = None) -> int:
    """解析固定 repeat/compare 命令并运行 A2-E1 baseline。"""
    parser = argparse.ArgumentParser(description="Run the fixed CAN A2-E1 CPU baseline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--repeat", type=int, choices=(1, 2))
    mode.add_argument("--compare", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.compare:
        fingerprint = compare_a2_repeats()
        print(json.dumps({"determinism_fingerprint": fingerprint}, sort_keys=True))
        return 0
    output_path = run_a2_baseline(cast(int, arguments.repeat))
    print(json.dumps({"report": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
