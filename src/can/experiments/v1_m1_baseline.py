"""V1-M1 的本地 archive 完整性与冻结 baseline 运行计划。"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import random
import struct
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from can.access.v1_m1_adapter import normalize_v1_m1_uint8_batch
from can.model.v1_cifar100_resnet import (
    V1_M1_CLASS_COUNT,
    V1_M1_MODEL_PROFILE_ID,
    V1_M1_PARAMETER_COUNT,
    V1Cifar100ResNet18,
)

V1_M1_EXPERIMENT_ID: Final = "CAN-V1-CIFAR100-RESNET18-v1"
V1_M1_DATA_ROOT: Final = Path("data/v1-m1")
V1_M1_ARTIFACT_ROOT: Final = Path("artifacts/v1-m1")
V1_M1_ARCHIVE_FILENAME: Final = "cifar-100-python.tar.gz"
V1_M1_ARCHIVE_SIZE: Final = 169_001_437
V1_M1_ARCHIVE_SHA256: Final = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
V1_M1_ARCHIVE_MD5: Final = "eb9058c3a382ffc7106e4002c42a8d85"
V1_M1_RUN_SEEDS: Final = (1729, 1730)
V1_M1_TRAIN_BATCH_SIZE: Final = 128
V1_M1_EVALUATION_BATCH_SIZE: Final = 256
V1_M1_WORKER_COUNT: Final = 4
V1_M1_TRAIN_SIZE: Final = 45_000
V1_M1_VALIDATION_SIZE: Final = 5_000
V1_M1_TEST_SIZE: Final = 10_000
V1_M1_EPOCH_COUNT: Final = 200
V1_M1_LEARNING_RATE: Final = 0.1
V1_M1_MOMENTUM: Final = 0.9
V1_M1_WEIGHT_DECAY: Final = 0.0005
V1_M1_ACCEPTANCE_TOP1_PERCENT: Final = 70.0
V1_M1_ACCEPTANCE_REPEAT_DELTA_PERCENT: Final = 2.0
V1_M1_TRAIN_PER_CLASS: Final = 450
V1_M1_VALIDATION_PER_CLASS: Final = 50
V1_M1_MAX_STATE_BYTES: Final = 64 * 1024 * 1024
V1_M1_STATE_FILENAME: Final = "best-validation-state.pt"
V1_M1_MANIFEST_FILENAME: Final = "manifest.json"
V1_M1_REPORT_FILENAME: Final = "report.json"
V1_M1_FINE_LABEL_NAMES: Final = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)


class V1M1BaselineError(RuntimeError):
    """表示 V1-M1 的固定数据资源或 baseline 计划不满足协议。"""


@dataclass(frozen=True, slots=True)
class V1M1ArchiveManifest:
    """记录已验证 archive 的公开文件身份, 不含数据内容。"""

    filename: str
    byte_size: int
    sha256: str
    md5: str


@dataclass(frozen=True, slots=True)
class V1M1TrainingConfig:
    """保存一个固定 V1-M1 baseline run 的不可变训练配置。"""

    run_index: int
    seed: int
    train_batch_size: int = V1_M1_TRAIN_BATCH_SIZE
    evaluation_batch_size: int = V1_M1_EVALUATION_BATCH_SIZE
    worker_count: int = V1_M1_WORKER_COUNT
    epoch_count: int = V1_M1_EPOCH_COUNT
    learning_rate: float = V1_M1_LEARNING_RATE
    momentum: float = V1_M1_MOMENTUM
    weight_decay: float = V1_M1_WEIGHT_DECAY
    nesterov: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        expected_seed = _seed_for_run(self.run_index)
        if self.seed != expected_seed:
            raise V1M1BaselineError("V1-M1 run seed is not pre-registered")
        if (
            self.train_batch_size != V1_M1_TRAIN_BATCH_SIZE
            or self.evaluation_batch_size != V1_M1_EVALUATION_BATCH_SIZE
            or self.worker_count != V1_M1_WORKER_COUNT
            or self.epoch_count != V1_M1_EPOCH_COUNT
            or self.learning_rate != V1_M1_LEARNING_RATE
            or self.momentum != V1_M1_MOMENTUM
            or self.weight_decay != V1_M1_WEIGHT_DECAY
            or self.nesterov is not True
            or self.pin_memory is not True
            or self.persistent_workers is not False
            or self.prefetch_factor != 2
        ):
            raise V1M1BaselineError("V1-M1 training configuration changed")


@dataclass(frozen=True, slots=True)
class V1M1BaselinePlan:
    """绑定已验证 archive 与一个预注册 run 的无副作用计划。"""

    archive: V1M1ArchiveManifest
    training: V1M1TrainingConfig


@dataclass(frozen=True, slots=True)
class V1M1ArtifactPaths:
    """保存一个固定 V1-M1 run 的 ignored artifact 路径。"""

    root: Path
    state: Path
    manifest: Path
    report: Path


@dataclass(frozen=True, slots=True)
class V1M1DataBundle:
    """保存已验证 CIFAR-100 split、loaders 与 decoded dataset digest。"""

    manifest: V1M1ArchiveManifest
    decoded_sha256: str
    train_loader: DataLoader[tuple[Tensor, Tensor]]
    validation_loader: DataLoader[tuple[Tensor, Tensor]]
    test_loader: DataLoader[tuple[Tensor, Tensor]]


@dataclass(frozen=True, slots=True)
class V1M1EvaluationMetrics:
    """记录一个 V1-M1 evaluation split 的公开聚合指标。"""

    loss: float
    top1_percent: float
    top5_percent: float
    correct_top1: int
    correct_top5: int
    total: int
    predictions_sha256: str


@dataclass(frozen=True, slots=True)
class V1M1EpochMetrics:
    """记录一个 V1-M1 training epoch 的训练和验证聚合指标。"""

    epoch: int
    training_loss: float
    validation: V1M1EvaluationMetrics


@dataclass(frozen=True, slots=True)
class V1M1BaselineResult:
    """保存一次正式 V1-M1 baseline run 的指标与本地 artifact 路径。"""

    plan: V1M1BaselinePlan
    dataset_sha256: str
    selected_epoch: int
    epochs: tuple[V1M1EpochMetrics, ...]
    test: V1M1EvaluationMetrics
    state_sha256: str
    artifacts: V1M1ArtifactPaths


class _V1M1ProgressReporter:
    """在不影响训练状态的前提下渲染 V1-M1 的 batch 进度。"""

    __slots__ = ("_completed_batches", "_config", "_total_batches")

    def __init__(self, config: V1M1TrainingConfig, total_batches: int) -> None:
        self._config = config
        self._total_batches = total_batches
        self._completed_batches = 0

    def start(self, first_train_batch_count: int) -> None:
        print(
            f"V1-M1 training started run={self._config.run_index} seed={self._config.seed} "
            f"epochs={self._config.epoch_count} total_batches={self._total_batches}",
            flush=True,
        )
        self._render("train", 0, 0, first_train_batch_count)

    def complete_batch(
        self,
        stage: str,
        epoch: int,
        stage_batch: int,
        stage_batch_count: int,
    ) -> None:
        self._completed_batches += 1
        self._render(stage, epoch, stage_batch, stage_batch_count)

    def finish(self, final_test_batch_count: int) -> None:
        self._render(
            "complete",
            self._config.epoch_count,
            final_test_batch_count,
            final_test_batch_count,
        )
        print(flush=True)
        print(
            f"V1-M1 training completed run={self._config.run_index} seed={self._config.seed} "
            f"completed_batches={self._completed_batches}/{self._total_batches}",
            flush=True,
        )

    def _render(self, stage: str, epoch: int, stage_batch: int, stage_batch_count: int) -> None:
        progress = _format_v1_m1_batch_progress(
            self._config,
            self._completed_batches,
            self._total_batches,
            stage,
            epoch,
            stage_batch,
            stage_batch_count,
        )
        print(f"\r{progress}", end="", flush=True)


class _V1M1Dataset(Dataset[tuple[Tensor, Tensor]]):
    """将已验证 CIFAR raw tensors 提供给固定 train 或 evaluation transform。"""

    __slots__ = ("_indices", "_labels", "_pixels", "_training")

    def __init__(
        self,
        pixels: Tensor,
        labels: Tensor,
        indices: Tensor,
        *,
        training: bool,
    ) -> None:
        self._pixels = pixels
        self._labels = labels
        self._indices = indices
        self._training = training

    def __len__(self) -> int:
        return int(self._indices.numel())

    def __getitem__(self, offset: int) -> tuple[Tensor, Tensor]:
        source_index = int(self._indices[offset].item())
        image = self._pixels[source_index]
        if self._training:
            image = _augment_v1_m1_training_image(image)
        normalized = normalize_v1_m1_uint8_batch(image.unsqueeze(0)).squeeze(0)
        return normalized, self._labels[source_index]


def _seed_for_run(run_index: object) -> int:
    if type(run_index) is not int or not 1 <= run_index <= len(V1_M1_RUN_SEEDS):
        raise V1M1BaselineError("V1-M1 run index must select one pre-registered run")
    return V1_M1_RUN_SEEDS[run_index - 1]


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as error:
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 resource is invalid") from error
    if type(value) is not dict:
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 resource is not an exact dict")
    if not all(type(key) is str for key in value):
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 keys are not canonical strings")
    return cast(dict[str, object], value)


def _decode_split(path: Path, *, expected_size: int) -> tuple[Tensor, Tensor]:
    value = _load_pickle(path)
    if set(value) != {"batch_label", "coarse_labels", "data", "filenames", "fine_labels"}:
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 split fields changed")
    data = value.get("data")
    fine_labels = value.get("fine_labels")
    coarse_labels = value.get("coarse_labels")
    filenames = value.get("filenames")
    batch_label = value.get("batch_label")
    if (
        type(data) is not np.ndarray
        or data.dtype != np.uint8
        or tuple(data.shape) != (expected_size, 3 * 32 * 32)
        or not data.flags.c_contiguous
        or type(fine_labels) is not list
        or type(coarse_labels) is not list
        or len(fine_labels) != expected_size
        or len(coarse_labels) != expected_size
        or type(filenames) is not list
        or len(filenames) != expected_size
        or not all(type(filename) is str for filename in filenames)
        or type(batch_label) is not str
    ):
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 split shape changed")
    if any(type(label) is not int or not 0 <= label < V1_M1_CLASS_COUNT for label in fine_labels):
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 fine labels are non-canonical")
    if any(type(label) is not int or not 0 <= label < 20 for label in coarse_labels):
        raise V1M1BaselineError("V1-M1 decoded CIFAR-100 coarse labels are non-canonical")
    pixels = (
        torch.from_numpy(data)
        .reshape(expected_size, 3, 32, 32)
        .clone(memory_format=torch.contiguous_format)
    )
    labels = torch.tensor(fine_labels, dtype=torch.int64)
    return pixels, labels


def _decode_meta(path: Path) -> None:
    value = _load_pickle(path)
    if set(value) != {"coarse_label_names", "fine_label_names"}:
        raise V1M1BaselineError("V1-M1 CIFAR-100 metadata fields changed")
    fine_names = value.get("fine_label_names")
    coarse_names = value.get("coarse_label_names")
    if type(fine_names) is not list or tuple(fine_names) != V1_M1_FINE_LABEL_NAMES:
        raise V1M1BaselineError("V1-M1 CIFAR-100 fine label order changed")
    if (
        type(coarse_names) is not list
        or len(coarse_names) != 20
        or not all(type(name) is str for name in coarse_names)
    ):
        raise V1M1BaselineError("V1-M1 CIFAR-100 coarse label metadata changed")


def _decoded_digest(
    train_pixels: Tensor,
    train_labels: Tensor,
    test_pixels: Tensor,
    test_labels: Tensor,
) -> str:
    digest = hashlib.sha256(b"CAN-V1-CIFAR100-DECODED-v1\x00")
    for pixels, labels in ((train_pixels, train_labels), (test_pixels, test_labels)):
        for index in range(pixels.shape[0]):
            digest.update(bytes((int(labels[index].item()),)))
            digest.update(pixels[index].reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def _build_v1_m1_split_indices(labels: Tensor) -> tuple[Tensor, Tensor]:
    if tuple(labels.shape) != (V1_M1_TRAIN_SIZE + V1_M1_VALIDATION_SIZE,):
        raise V1M1BaselineError("V1-M1 train labels have the wrong size")
    validation: list[Tensor] = []
    training: list[Tensor] = []
    for label in range(V1_M1_CLASS_COUNT):
        positions = torch.nonzero(labels == label, as_tuple=False).reshape(-1)
        if positions.numel() != V1_M1_TRAIN_PER_CLASS + V1_M1_VALIDATION_PER_CLASS:
            raise V1M1BaselineError("V1-M1 train split is not class-balanced")
        validation.append(positions[:V1_M1_VALIDATION_PER_CLASS])
        training.append(positions[V1_M1_VALIDATION_PER_CLASS:])
    validation_indices = torch.cat(validation).contiguous()
    training_indices = torch.cat(training).contiguous()
    if (
        validation_indices.numel() != V1_M1_VALIDATION_SIZE
        or training_indices.numel() != V1_M1_TRAIN_SIZE
    ):
        raise V1M1BaselineError("V1-M1 split cardinality changed")
    return training_indices, validation_indices


def _verify_extracted_member_bytes(data_root: Path, extracted_root: Path) -> None:
    """确认将要解析的三个文件与已验证 archive 的对应成员逐字节相同。"""
    archive_path = data_root / V1_M1_ARCHIVE_FILENAME
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for filename in ("train", "test", "meta"):
                member = archive.getmember(f"cifar-100-python/{filename}")
                extracted_path = extracted_root / filename
                if (
                    not member.isfile()
                    or not extracted_path.is_file()
                    or extracted_path.is_symlink()
                ):
                    raise V1M1BaselineError("V1-M1 extracted CIFAR-100 member is unavailable")
                source = archive.extractfile(member)
                if source is None:
                    raise V1M1BaselineError("V1-M1 archive member cannot be read")
                source_digest = hashlib.sha256()
                with source, extracted_path.open("rb") as extracted:
                    while source_chunk := source.read(1024 * 1024):
                        source_digest.update(source_chunk)
                    extracted_digest = hashlib.sha256()
                    for extracted_chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                        extracted_digest.update(extracted_chunk)
                if source_digest.digest() != extracted_digest.digest():
                    raise V1M1BaselineError("V1-M1 extracted CIFAR-100 member differs from archive")
    except (KeyError, OSError, tarfile.TarError) as error:
        raise V1M1BaselineError("V1-M1 CIFAR-100 archive cannot validate extracted data") from error


def _augment_v1_m1_training_image(image: Tensor) -> Tensor:
    padded = functional.pad(image, (4, 4, 4, 4), mode="constant", value=0)
    top = int(torch.randint(0, 9, ()).item())
    left = int(torch.randint(0, 9, ()).item())
    cropped = padded[:, top : top + 32, left : left + 32]
    if bool(torch.rand(()) < 0.5):
        cropped = torch.flip(cropped, dims=(2,))
    return cropped.contiguous()


def _load_v1_m1_data(data_root: Path, config: V1M1TrainingConfig) -> V1M1DataBundle:
    manifest = verify_v1_m1_archive(data_root)
    extracted_root = data_root / "cifar-100-python"
    if not extracted_root.is_dir():
        raise V1M1BaselineError("V1-M1 verified archive has not been explicitly extracted")
    _verify_extracted_member_bytes(data_root, extracted_root)
    train_pixels, train_labels = _decode_split(
        extracted_root / "train", expected_size=V1_M1_TRAIN_SIZE + V1_M1_VALIDATION_SIZE
    )
    test_pixels, test_labels = _decode_split(extracted_root / "test", expected_size=V1_M1_TEST_SIZE)
    _decode_meta(extracted_root / "meta")
    training_indices, validation_indices = _build_v1_m1_split_indices(train_labels)
    train_dataset = _V1M1Dataset(train_pixels, train_labels, training_indices, training=True)
    validation_dataset = _V1M1Dataset(
        train_pixels,
        train_labels,
        validation_indices,
        training=False,
    )
    test_indices = torch.arange(V1_M1_TEST_SIZE, dtype=torch.int64)
    test_dataset = _V1M1Dataset(test_pixels, test_labels, test_indices, training=False)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.worker_count,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=config.worker_count,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=config.worker_count,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=False,
    )
    return V1M1DataBundle(
        manifest=manifest,
        decoded_sha256=_decoded_digest(train_pixels, train_labels, test_pixels, test_labels),
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


def _configure_v1_m1_determinism(seed: int) -> None:
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise V1M1BaselineError("V1-M1 PYTHONHASHSEED does not match the pre-registered run")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise V1M1BaselineError("V1-M1 CUBLAS_WORKSPACE_CONFIG is not frozen")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _hash_predictions(predictions: list[int]) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        digest.update(struct.pack(">q", prediction))
    return digest.hexdigest()


def _evaluate_v1_m1(
    model: V1Cifar100ResNet18,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    progress: _V1M1ProgressReporter | None = None,
    stage: str = "validation",
    epoch: int = 0,
) -> V1M1EvaluationMetrics:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top5 = 0
    predictions: list[int] = []
    with torch.inference_mode():
        batch_count = len(loader)
        for batch_index, (images, labels) in enumerate(loader, start=1):
            inputs = images.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            logits = model(inputs)
            total_loss += float(criterion(logits, targets).item()) * targets.shape[0]
            top5 = torch.topk(logits, k=5, dim=1).indices
            predicted = top5[:, 0]
            correct_top1 += int((predicted == targets).sum().item())
            correct_top5 += int((top5 == targets.unsqueeze(1)).any(dim=1).sum().item())
            total += targets.shape[0]
            predictions.extend(int(value) for value in predicted.cpu().tolist())
            if progress is not None:
                progress.complete_batch(stage, epoch, batch_index, batch_count)
    if total < 1:
        raise V1M1BaselineError("V1-M1 evaluation loader is empty")
    return V1M1EvaluationMetrics(
        loss=total_loss / total,
        top1_percent=correct_top1 * 100.0 / total,
        top5_percent=correct_top5 * 100.0 / total,
        correct_top1=correct_top1,
        correct_top5=correct_top5,
        total=total,
        predictions_sha256=_hash_predictions(predictions),
    )


def _format_v1_m1_batch_progress(
    config: V1M1TrainingConfig,
    completed_batches: int,
    total_batches: int,
    stage: str,
    epoch: int,
    stage_batch: int,
    stage_batch_count: int,
) -> str:
    """构造固定宽度、只含公开计数的 batch 进度条。"""
    percent = completed_batches * 100.0 / total_batches
    bar_width = 30
    filled = int(completed_batches * bar_width / total_batches)
    bar = "#" * filled + "-" * (bar_width - filled)
    return (
        f"V1-M1 progress [{bar}] {percent:6.2f}% "
        f"run={config.run_index} seed={config.seed} stage={stage} "
        f"epoch={epoch}/{config.epoch_count} batch={stage_batch}/{stage_batch_count}"
    )


def _train_v1_m1_epoch(
    model: V1Cifar100ResNet18,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: SGD,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
    *,
    progress: _V1M1ProgressReporter,
    epoch: int,
) -> float:
    """执行一个固定 V1-M1 train epoch 并返回样本加权 loss。"""
    model.train()
    total_loss = 0.0
    total = 0
    batch_count = len(loader)
    for batch_index, (images, labels) in enumerate(loader, start=1):
        inputs = images.to(device, non_blocking=True)
        targets = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * targets.shape[0]
        total += targets.shape[0]
        progress.complete_batch("train", epoch, batch_index, batch_count)
    if total < 1:
        raise V1M1BaselineError("V1-M1 training loader is empty")
    return total_loss / total


def _clone_model_state(model: V1Cifar100ResNet18) -> dict[str, Tensor]:
    """将当前 state_dict 固定为 CPU 上独立的临时 best-validation 快照。"""
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _hash_model_state(model: V1Cifar100ResNet18) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(struct.pack(">I", contiguous.ndim))
        for dimension in contiguous.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _artifact_paths(artifact_root: Path, run_index: int) -> V1M1ArtifactPaths:
    if not isinstance(artifact_root, Path):
        raise V1M1BaselineError("V1-M1 artifact root must be pathlib.Path")
    _seed_for_run(run_index)
    if artifact_root.is_symlink() or (artifact_root.exists() and not artifact_root.is_dir()):
        raise V1M1BaselineError("V1-M1 artifact root is not a canonical directory")
    root = artifact_root / f"run-{run_index}"
    if root.exists() or root.is_symlink():
        raise V1M1BaselineError("V1-M1 artifact directory already exists")
    return V1M1ArtifactPaths(
        root=root,
        state=root / V1_M1_STATE_FILENAME,
        manifest=root / V1_M1_MANIFEST_FILENAME,
        report=root / V1_M1_REPORT_FILENAME,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise V1M1BaselineError(f"refusing to overwrite V1-M1 artifact: {path.name}")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o600)
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise V1M1BaselineError(f"refusing to overwrite V1-M1 artifact: {path.name}") from error
    except V1M1BaselineError:
        raise
    except OSError as error:
        raise V1M1BaselineError("V1-M1 artifact write failed") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_state(path: Path, state: Mapping[str, Tensor]) -> tuple[int, str]:
    for name, value in state.items():
        if type(name) is not str or type(value) is not Tensor:
            raise V1M1BaselineError("V1-M1 selected state has a non-canonical entry")
        if value.device.type != "cpu" or value.device.index is not None:
            raise V1M1BaselineError("V1-M1 selected state must remain on the CPU")
        if value.layout is not torch.strided or not value.is_contiguous():
            raise V1M1BaselineError("V1-M1 selected state layout changed")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise V1M1BaselineError("V1-M1 selected state contains non-finite values")
    if path.exists() or path.is_symlink():
        raise V1M1BaselineError(f"refusing to overwrite V1-M1 state: {path.name}")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as stream:
            torch.save(dict(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        byte_size = temporary_path.stat().st_size
        if byte_size < 1 or byte_size > V1_M1_MAX_STATE_BYTES:
            raise V1M1BaselineError("V1-M1 serialized state size is outside the fixed bound")
        temporary_path.chmod(0o600)
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise V1M1BaselineError(f"refusing to overwrite V1-M1 state: {path.name}") from error
    except V1M1BaselineError:
        raise
    except (OSError, RuntimeError) as error:
        raise V1M1BaselineError("V1-M1 state write failed") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return byte_size, _file_digest(path, "sha256")


def _evaluation_report(metrics: V1M1EvaluationMetrics) -> dict[str, object]:
    return {
        "loss_hex": metrics.loss.hex(),
        "top1_percent_hex": metrics.top1_percent.hex(),
        "top5_percent_hex": metrics.top5_percent.hex(),
        "correct_top1": metrics.correct_top1,
        "correct_top5": metrics.correct_top5,
        "total": metrics.total,
        "predictions_sha256": metrics.predictions_sha256,
    }


def _model_report() -> dict[str, object]:
    return {
        "profile_id": V1_M1_MODEL_PROFILE_ID,
        "topology": "CIFAR-style ResNet-18 [2,2,2,2]",
        "parameter_count": V1_M1_PARAMETER_COUNT,
        "float32_parameter_bytes": V1_M1_PARAMETER_COUNT * 4,
    }


def _environment_report(device: torch.device) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _write_v1_m1_artifacts(
    result: V1M1BaselineResult,
    state: Mapping[str, Tensor],
    device: torch.device,
) -> None:
    paths = result.artifacts
    try:
        paths.root.parent.mkdir(parents=True, exist_ok=True)
        if paths.root.parent.is_symlink() or not paths.root.parent.is_dir():
            raise V1M1BaselineError("V1-M1 artifact parent is not a canonical directory")
        paths.root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise V1M1BaselineError("V1-M1 artifact directory already exists") from error
    except OSError as error:
        raise V1M1BaselineError("V1-M1 artifact directory cannot be created") from error

    state_bytes, state_file_sha256 = _atomic_save_state(paths.state, state)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": V1_M1_EXPERIMENT_ID,
        "run_index": result.plan.training.run_index,
        "data": {
            "archive": asdict(result.plan.archive),
            "decoded_sha256": result.dataset_sha256,
            "train_size": V1_M1_TRAIN_SIZE,
            "validation_size": V1_M1_VALIDATION_SIZE,
            "test_size": V1_M1_TEST_SIZE,
        },
        "training": asdict(result.plan.training),
        "model": _model_report(),
        "selection": {
            "selected_epoch": result.selected_epoch,
            "rule": "strictly higher validation top-1; retain earlier epoch on a tie",
        },
        "state": {
            "filename": paths.state.name,
            "state_dict_only": True,
            "optimizer_state_saved": False,
            "byte_size": state_bytes,
            "file_sha256": state_file_sha256,
            "canonical_state_sha256": result.state_sha256,
        },
    }
    _atomic_write_bytes(
        paths.manifest,
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": V1_M1_EXPERIMENT_ID,
        "run_index": result.plan.training.run_index,
        "environment": _environment_report(device),
        "manifest_filename": paths.manifest.name,
        "manifest_sha256": _file_digest(paths.manifest, "sha256"),
        "epochs": [
            {
                "epoch": item.epoch,
                "training_loss_hex": item.training_loss.hex(),
                "validation": _evaluation_report(item.validation),
            }
            for item in result.epochs
        ],
        "test": _evaluation_report(result.test),
        "selected_epoch": result.selected_epoch,
        "state_sha256": result.state_sha256,
    }
    _atomic_write_bytes(
        paths.report,
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def verify_v1_m1_archive(data_root: Path) -> V1M1ArchiveManifest:
    """验证本地已有 CIFAR-100 archive, 绝不下载、解压或生成数据。"""
    if not isinstance(data_root, Path):
        raise V1M1BaselineError("V1-M1 data root must be pathlib.Path")
    archive_path = data_root / V1_M1_ARCHIVE_FILENAME
    if not archive_path.is_file():
        raise V1M1BaselineError("V1-M1 CIFAR-100 archive is unavailable")
    if archive_path.stat().st_size != V1_M1_ARCHIVE_SIZE:
        raise V1M1BaselineError("V1-M1 CIFAR-100 archive byte size changed")
    sha256 = _file_digest(archive_path, "sha256")
    md5 = _file_digest(archive_path, "md5")
    if sha256 != V1_M1_ARCHIVE_SHA256 or md5 != V1_M1_ARCHIVE_MD5:
        raise V1M1BaselineError("V1-M1 CIFAR-100 archive digest changed")
    return V1M1ArchiveManifest(
        filename=V1_M1_ARCHIVE_FILENAME,
        byte_size=V1_M1_ARCHIVE_SIZE,
        sha256=sha256,
        md5=md5,
    )


def build_v1_m1_baseline_plan(data_root: Path, run_index: int) -> V1M1BaselinePlan:
    """构造一个仅可用于预注册 run 的无训练 baseline plan。"""
    config = V1M1TrainingConfig(run_index=run_index, seed=_seed_for_run(run_index))
    return V1M1BaselinePlan(archive=verify_v1_m1_archive(data_root), training=config)


def run_v1_m1_baseline(
    data_root: Path,
    run_index: int,
    device: torch.device,
    artifact_root: Path = V1_M1_ARTIFACT_ROOT,
) -> V1M1BaselineResult:
    """在已冻结 CUDA 环境运行一个 V1-M1 baseline 并写入 ignored artifact。"""
    if type(device) is not torch.device or device.type != "cuda":
        raise V1M1BaselineError("V1-M1 baseline requires an explicit CUDA device")
    config = V1M1TrainingConfig(run_index=run_index, seed=_seed_for_run(run_index))
    artifacts = _artifact_paths(artifact_root, run_index)
    data = _load_v1_m1_data(data_root, config)
    _configure_v1_m1_determinism(config.seed)

    model = V1Cifar100ResNet18().to(device=device, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epoch_count, eta_min=0.0)
    train_batch_count = len(data.train_loader)
    validation_batch_count = len(data.validation_loader)
    test_batch_count = len(data.test_loader)
    progress = _V1M1ProgressReporter(
        config,
        config.epoch_count * (train_batch_count + validation_batch_count) + test_batch_count,
    )
    progress.start(train_batch_count)
    best_epoch = 0
    best_validation_top1 = -1.0
    best_state: dict[str, Tensor] | None = None
    epochs: list[V1M1EpochMetrics] = []
    for epoch in range(1, config.epoch_count + 1):
        training_loss = _train_v1_m1_epoch(
            model,
            data.train_loader,
            optimizer,
            criterion,
            device,
            progress=progress,
            epoch=epoch,
        )
        validation = _evaluate_v1_m1(
            model,
            data.validation_loader,
            device,
            progress=progress,
            stage="validation",
            epoch=epoch,
        )
        metrics = V1M1EpochMetrics(
            epoch=epoch,
            training_loss=training_loss,
            validation=validation,
        )
        epochs.append(metrics)
        if validation.top1_percent > best_validation_top1:
            best_epoch = epoch
            best_validation_top1 = validation.top1_percent
            best_state = _clone_model_state(model)
        scheduler.step()
    if best_state is None:
        raise V1M1BaselineError("V1-M1 baseline did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    test = _evaluate_v1_m1(
        model,
        data.test_loader,
        device,
        progress=progress,
        stage="test",
        epoch=config.epoch_count,
    )
    result = V1M1BaselineResult(
        plan=V1M1BaselinePlan(archive=data.manifest, training=config),
        dataset_sha256=data.decoded_sha256,
        selected_epoch=best_epoch,
        epochs=tuple(epochs),
        test=test,
        state_sha256=_hash_model_state(model),
        artifacts=artifacts,
    )
    _write_v1_m1_artifacts(result, best_state, device)
    progress.finish(test_batch_count)
    return result


__all__ = [
    "V1_M1_ACCEPTANCE_REPEAT_DELTA_PERCENT",
    "V1_M1_ACCEPTANCE_TOP1_PERCENT",
    "V1_M1_ARCHIVE_FILENAME",
    "V1_M1_ARCHIVE_MD5",
    "V1_M1_ARCHIVE_SHA256",
    "V1_M1_ARCHIVE_SIZE",
    "V1_M1_ARTIFACT_ROOT",
    "V1_M1_DATA_ROOT",
    "V1_M1_EPOCH_COUNT",
    "V1_M1_EXPERIMENT_ID",
    "V1_M1_MANIFEST_FILENAME",
    "V1_M1_REPORT_FILENAME",
    "V1_M1_RUN_SEEDS",
    "V1_M1_STATE_FILENAME",
    "V1M1ArchiveManifest",
    "V1M1ArtifactPaths",
    "V1M1BaselineError",
    "V1M1BaselinePlan",
    "V1M1BaselineResult",
    "V1M1EpochMetrics",
    "V1M1EvaluationMetrics",
    "V1M1TrainingConfig",
    "build_v1_m1_baseline_plan",
    "run_v1_m1_baseline",
    "verify_v1_m1_archive",
]
