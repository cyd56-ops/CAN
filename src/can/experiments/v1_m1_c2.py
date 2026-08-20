"""V1-M1-C2 public head 的预注册训练、选择与 artifact 边界。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from can.access.v1_m1_adapter import V1_M1_INPUT_PROFILE_SHA256, normalize_v1_m1_uint8_batch
from can.access.v1_m1_c2 import V1M1C2Cut, V1M1C2PublicHead
from can.experiments import v1_m1_baseline as baseline
from can.experiments.v1_m1_c1 import load_v1_m1_c1_accepted_r2_details
from can.model import V1Cifar100ResNet18

V1_M1_C2_EXPERIMENT_ID: Final = "CAN-V1-M1-C2-PUBLIC-HEAD-v1"
V1_M1_C2_ARTIFACT_ROOT: Final = Path("artifacts/v1-m1/c2")
V1_M1_C2_STATE_FILENAME: Final = "accepted-public-head.pt"
V1_M1_C2_MANIFEST_FILENAME: Final = "manifest.json"
V1_M1_C2_REPORT_FILENAME: Final = "report.json"
V1_M1_C2_RUN_NAMES: Final = ("H1", "H2")
V1_M1_C2_RUN_SEEDS: Final = {"H1": 1729, "H2": 1730}
V1_M1_C2_HEAD_EPOCH_COUNT: Final = 50
V1_M1_C2_TRAIN_BATCH_SIZE: Final = 128
V1_M1_C2_EVALUATION_BATCH_SIZE: Final = 256
V1_M1_C2_WORKER_COUNT: Final = 4
V1_M1_C2_LEARNING_RATE: Final = 0.1
V1_M1_C2_MOMENTUM: Final = 0.9
V1_M1_C2_WEIGHT_DECAY: Final = 0.0005
V1_M1_C2_ACCEPTANCE_TOP1_PERCENT: Final = 75.0
V1_M1_C2_STABILITY_DELTA_PERCENT: Final = 2.0
V1_M1_C2_COARSE_DIGEST_DOMAIN: Final = b"CAN-V1-M1-C2-COARSE-LABELS-v1\x00"
V1_M1_C2_MAX_STATE_BYTES: Final = 4 * 1024 * 1024


class V1M1C2ExperimentError(RuntimeError):
    """表示 C2 public-head runner 或 artifact 不满足冻结契约。"""


V1M1C2RunName: TypeAlias = Literal["H1", "H2"]


@dataclass(frozen=True, slots=True)
class V1M1C2HeadTrainingConfig:
    """保存一个预注册 H1/H2、cut 和 head-only 训练配置。"""

    run_name: V1M1C2RunName
    cut: V1M1C2Cut
    seed: int
    epoch_count: int = V1_M1_C2_HEAD_EPOCH_COUNT
    train_batch_size: int = V1_M1_C2_TRAIN_BATCH_SIZE
    evaluation_batch_size: int = V1_M1_C2_EVALUATION_BATCH_SIZE
    worker_count: int = V1_M1_C2_WORKER_COUNT
    learning_rate: float = V1_M1_C2_LEARNING_RATE
    momentum: float = V1_M1_C2_MOMENTUM
    weight_decay: float = V1_M1_C2_WEIGHT_DECAY
    nesterov: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.run_name not in V1_M1_C2_RUN_NAMES:
            raise V1M1C2ExperimentError("C2 run name is not pre-registered")
        if type(self.cut) is not V1M1C2Cut:
            raise V1M1C2ExperimentError("C2 training cut is not canonical")
        if self.seed != V1_M1_C2_RUN_SEEDS[self.run_name]:
            raise V1M1C2ExperimentError("C2 training seed is not pre-registered")
        if (
            self.epoch_count != V1_M1_C2_HEAD_EPOCH_COUNT
            or self.train_batch_size != V1_M1_C2_TRAIN_BATCH_SIZE
            or self.evaluation_batch_size != V1_M1_C2_EVALUATION_BATCH_SIZE
            or self.worker_count != V1_M1_C2_WORKER_COUNT
            or self.learning_rate != V1_M1_C2_LEARNING_RATE
            or self.momentum != V1_M1_C2_MOMENTUM
            or self.weight_decay != V1_M1_C2_WEIGHT_DECAY
            or self.nesterov is not True
            or self.pin_memory is not True
            or self.persistent_workers is not False
            or self.prefetch_factor != 2
        ):
            raise V1M1C2ExperimentError("C2 head training configuration changed")


@dataclass(frozen=True, slots=True)
class V1M1C2HeadMetrics:
    """记录 coarse validation/test 的公开聚合指标。"""

    loss: float
    top1_percent: float
    correct_top1: int
    total: int
    predictions_sha256: str


@dataclass(frozen=True, slots=True)
class V1M1C2EpochMetrics:
    """记录一个 head epoch 的 validation 指标。"""

    epoch: int
    validation: V1M1C2HeadMetrics


@dataclass(frozen=True, slots=True)
class V1M1C2HeadRunResult:
    """保存一个 H1/H2 candidate 的 validation-only 结果和 CPU head state。"""

    config: V1M1C2HeadTrainingConfig
    selected_epoch: int
    validation: V1M1C2HeadMetrics
    epochs: tuple[V1M1C2EpochMetrics, ...]
    state_sha256: str
    state: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class V1M1C2ArtifactPaths:
    """保存最终 accepted public head 的 ignored artifact 路径。"""

    root: Path
    state: Path
    manifest: Path
    report: Path


@dataclass(frozen=True, slots=True)
class V1M1C2ExperimentResult:
    """保存 H1/H2 选择结果和最终 public test 指标。"""

    accepted_cut: V1M1C2Cut
    accepted: V1M1C2HeadRunResult
    h1_candidates: tuple[V1M1C2HeadRunResult, ...]
    h2: V1M1C2HeadRunResult
    test: V1M1C2HeadMetrics
    coarse_labels_sha256: str
    accepted_r2_state_sha256: str
    artifacts: V1M1C2ArtifactPaths


class _V1M1C2Dataset(Dataset[tuple[Tensor, Tensor]]):
    """为 public head 提供 canonical image 与 official coarse label。"""

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
            image = baseline._augment_v1_m1_training_image(image)
        normalized = normalize_v1_m1_uint8_batch(image.unsqueeze(0)).squeeze(0)
        return normalized, self._labels[source_index]


@dataclass(frozen=True, slots=True)
class _V1M1C2DataBundle:
    """保存 C2 train/validation/test loader 与两个公开数据摘要。"""

    archive: baseline.V1M1ArchiveManifest
    decoded_sha256: str
    coarse_labels_sha256: str
    train_loader: DataLoader[tuple[Tensor, Tensor]]
    validation_loader: DataLoader[tuple[Tensor, Tensor]]
    test_loader: DataLoader[tuple[Tensor, Tensor]]


def _coarse_labels_digest(train_labels: Tensor, test_labels: Tensor) -> str:
    """按官方 archive 顺序固定 coarse-label digest。"""
    if train_labels.dtype is not torch.int64 or test_labels.dtype is not torch.int64:
        raise V1M1C2ExperimentError("C2 coarse labels must use int64")
    digest = hashlib.sha256(V1_M1_C2_COARSE_DIGEST_DOMAIN)
    for split_name, labels in ((b"train\x00", train_labels), (b"test\x00", test_labels)):
        digest.update(split_name)
        if labels.ndim != 1 or not bool(((labels >= 0) & (labels < 20)).all().item()):
            raise V1M1C2ExperimentError("C2 coarse labels are outside the official range")
        for label in labels.tolist():
            digest.update(struct.pack(">B", int(label)))
    return digest.hexdigest()


def _load_c2_data(data_root: Path, config: V1M1C2HeadTrainingConfig) -> _V1M1C2DataBundle:
    """读取已验证 CIFAR archive, 绝不下载或修改数据。"""
    try:
        archive = baseline.verify_v1_m1_archive(data_root)
    except baseline.V1M1BaselineError as error:
        raise V1M1C2ExperimentError(str(error)) from error
    extracted_root = data_root / "cifar-100-python"
    if not extracted_root.is_dir():
        raise V1M1C2ExperimentError("verified C2 archive has not been explicitly extracted")
    baseline._verify_extracted_member_bytes(data_root, extracted_root)
    train_pixels, train_fine, train_coarse = baseline._decode_split_with_coarse(
        extracted_root / "train",
        expected_size=baseline.V1_M1_TRAIN_SIZE + baseline.V1_M1_VALIDATION_SIZE,
    )
    test_pixels, test_fine, test_coarse = baseline._decode_split_with_coarse(
        extracted_root / "test",
        expected_size=baseline.V1_M1_TEST_SIZE,
    )
    baseline._decode_meta(extracted_root / "meta")
    training_indices, validation_indices = baseline._build_v1_m1_split_indices(train_fine)
    train_dataset = _V1M1C2Dataset(train_pixels, train_coarse, training_indices, training=True)
    validation_dataset = _V1M1C2Dataset(
        train_pixels,
        train_coarse,
        validation_indices,
        training=False,
    )
    test_indices = torch.arange(baseline.V1_M1_TEST_SIZE, dtype=torch.int64)
    test_dataset = _V1M1C2Dataset(test_pixels, test_coarse, test_indices, training=False)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=config.worker_count,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.worker_count,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.worker_count,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    return _V1M1C2DataBundle(
        archive=archive,
        decoded_sha256=baseline._decoded_digest(
            train_pixels,
            train_fine,
            test_pixels,
            test_fine,
        ),
        coarse_labels_sha256=_coarse_labels_digest(train_coarse, test_coarse),
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


def _prefix(model: V1Cifar100ResNet18, cut: V1M1C2Cut, images: Tensor) -> Tensor:
    """执行 frozen R2 从 stem 到完整 candidate stage 的 prefix。"""
    output = model.layer1(model.stem(images))
    if cut is V1M1C2Cut.LAYER2:
        return cast(Tensor, model.layer2(output))
    output = model.layer2(output)
    if cut is V1M1C2Cut.LAYER3:
        return cast(Tensor, model.layer3(output))
    output = model.layer3(output)
    return cast(Tensor, model.layer4(output))


def _configure_c2_determinism(seed: int) -> None:
    """为同一进程内的 H1/H2 固定 CUDA policy 与独立 RNG seed。"""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise V1M1C2ExperimentError("C2 CUBLAS_WORKSPACE_CONFIG is not frozen")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _head_state_digest(state: Mapping[str, Tensor]) -> str:
    """按名称、dtype、shape 和 C-order bytes 计算 head state digest。"""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or type(value) is not Tensor:
            raise V1M1C2ExperimentError("public head state is not canonical")
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(struct.pack(">I", contiguous.ndim))
        for dimension in contiguous.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _clone_head_state(head: V1M1C2PublicHead) -> dict[str, Tensor]:
    """复制 public head state 到 CPU, 避免保存 R2 module。"""
    return {
        name: value.detach().cpu().contiguous().clone() for name, value in head.state_dict().items()
    }


def _hash_predictions(predictions: list[int]) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        digest.update(struct.pack(">q", prediction))
    return digest.hexdigest()


def _evaluate_head(
    model: V1Cifar100ResNet18,
    head: V1M1C2PublicHead,
    cut: V1M1C2Cut,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> V1M1C2HeadMetrics:
    """在 frozen prefix 上评估 coarse head, 并返回公开 digest。"""
    model.eval()
    head.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            inputs = images.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            logits = head(_prefix(model, cut, inputs))
            total_loss += float(criterion(logits, targets).item()) * targets.shape[0]
            predicted = logits.argmax(dim=1)
            correct += int((predicted == targets).sum().item())
            total += int(targets.shape[0])
            predictions.extend(int(value) for value in predicted.cpu().tolist())
    if total < 1:
        raise V1M1C2ExperimentError("C2 evaluation loader is empty")
    return V1M1C2HeadMetrics(
        loss=total_loss / total,
        top1_percent=correct * 100.0 / total,
        correct_top1=correct,
        total=total,
        predictions_sha256=_hash_predictions(predictions),
    )


def _train_head(
    model: V1Cifar100ResNet18,
    config: V1M1C2HeadTrainingConfig,
    data: _V1M1C2DataBundle,
    device: torch.device,
) -> V1M1C2HeadRunResult:
    """执行一次固定 head-only 训练, 按 validation 严格提升选择 state。"""
    _configure_c2_determinism(config.seed)
    generator = data.train_loader.generator
    if type(generator) is torch.Generator:
        generator.manual_seed(config.seed)
    head = V1M1C2PublicHead(config.cut.channels).to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = SGD(
        head.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epoch_count, eta_min=0.0)
    criterion = nn.CrossEntropyLoss()
    best_validation = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs: list[V1M1C2EpochMetrics] = []
    for epoch in range(1, config.epoch_count + 1):
        head.train()
        for images, labels in data.train_loader:
            inputs = images.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = _prefix(model, config.cut, inputs)
            loss = criterion(head(features.detach()), targets)
            loss.backward()
            optimizer.step()
        validation = _evaluate_head(model, head, config.cut, data.validation_loader, device)
        epochs.append(V1M1C2EpochMetrics(epoch=epoch, validation=validation))
        if validation.top1_percent > best_validation:
            best_epoch = epoch
            best_validation = validation.top1_percent
            best_state = _clone_head_state(head)
        scheduler.step()
    if best_state is None:
        raise V1M1C2ExperimentError("C2 head did not produce a validation checkpoint")
    head.load_state_dict(best_state, strict=True)
    selected_validation = _evaluate_head(model, head, config.cut, data.validation_loader, device)
    return V1M1C2HeadRunResult(
        config=config,
        selected_epoch=best_epoch,
        validation=selected_validation,
        epochs=tuple(epochs),
        state_sha256=_head_state_digest(best_state),
        state=best_state,
    )


def _select_h1_cut(candidates: tuple[V1M1C2HeadRunResult, ...]) -> V1M1C2Cut:
    """按浅到深和预注册阈值选择首个 H1 candidate。"""
    by_cut = {candidate.config.cut: candidate for candidate in candidates}
    if set(by_cut) != set(V1M1C2Cut):
        raise V1M1C2ExperimentError("H1 must contain exactly three candidate cuts")
    for cut in V1M1C2Cut:
        if by_cut[cut].validation.top1_percent >= V1_M1_C2_ACCEPTANCE_TOP1_PERCENT:
            return cut
    raise V1M1C2ExperimentError("C2 public utility did not meet the pre-registered threshold")


def _select_accepted_head(
    h1: V1M1C2HeadRunResult,
    h2: V1M1C2HeadRunResult,
) -> V1M1C2HeadRunResult:
    """仅按 validation 选择 H1/H2, 平局固定选择 H1。"""
    if h1.config.cut is not h2.config.cut:
        raise V1M1C2ExperimentError("H1/H2 accepted cuts do not match")
    if h1.validation.top1_percent >= h2.validation.top1_percent:
        return h1
    return h2


def _artifact_paths(artifact_root: Path) -> V1M1C2ArtifactPaths:
    """解析一次性 C2 accepted head artifact 路径, 不创建目录。"""
    if not isinstance(artifact_root, Path):
        raise V1M1C2ExperimentError("C2 artifact root must be pathlib.Path")
    if artifact_root.is_symlink() or (artifact_root.exists() and not artifact_root.is_dir()):
        raise V1M1C2ExperimentError("C2 artifact root is not a canonical directory")
    if artifact_root.exists() and any(
        path.exists() or path.is_symlink()
        for path in (
            artifact_root / V1_M1_C2_STATE_FILENAME,
            artifact_root / V1_M1_C2_MANIFEST_FILENAME,
            artifact_root / V1_M1_C2_REPORT_FILENAME,
        )
    ):
        raise V1M1C2ExperimentError("C2 accepted artifact already exists")
    return V1M1C2ArtifactPaths(
        root=artifact_root,
        state=artifact_root / V1_M1_C2_STATE_FILENAME,
        manifest=artifact_root / V1_M1_C2_MANIFEST_FILENAME,
        report=artifact_root / V1_M1_C2_REPORT_FILENAME,
    )


def _write_c2_artifacts(
    paths: V1M1C2ArtifactPaths,
    accepted: V1M1C2HeadRunResult,
    h1_candidates: tuple[V1M1C2HeadRunResult, ...],
    h2: V1M1C2HeadRunResult,
    test: V1M1C2HeadMetrics,
    data: _V1M1C2DataBundle,
    accepted_r2_state_sha256: str,
    device: torch.device,
) -> None:
    """原子保存只含 public head state 与公开 manifest/report 的 artifact。"""
    try:
        paths.root.parent.mkdir(parents=True, exist_ok=True)
        if paths.root.exists():
            if paths.root.is_symlink() or not paths.root.is_dir():
                raise V1M1C2ExperimentError("C2 artifact root is not a canonical directory")
        else:
            paths.root.mkdir(mode=0o700)
    except (OSError, V1M1C2ExperimentError) as error:
        raise V1M1C2ExperimentError("C2 artifact directory cannot be created") from error
    try:
        state_bytes, state_file_sha256 = baseline._atomic_save_state(paths.state, accepted.state)
        if state_bytes > V1_M1_C2_MAX_STATE_BYTES:
            raise V1M1C2ExperimentError("C2 public head state is too large")
        manifest = {
            "schema_version": 1,
            "experiment_id": V1_M1_C2_EXPERIMENT_ID,
            "accepted_cut": accepted.config.cut.value,
            "accepted_run": accepted.config.run_name,
            "accepted_r2_state_sha256": accepted_r2_state_sha256,
            "input_profile_sha256": V1_M1_INPUT_PROFILE_SHA256.hex(),
            "data": {
                "archive": asdict(data.archive),
                "decoded_sha256": data.decoded_sha256,
                "coarse_labels_sha256": data.coarse_labels_sha256,
                "train_size": baseline.V1_M1_TRAIN_SIZE,
                "validation_size": baseline.V1_M1_VALIDATION_SIZE,
                "test_size": baseline.V1_M1_TEST_SIZE,
            },
            "head": {
                "topology": "AdaptiveAvgPool2d(1)->Flatten->Linear(C_cut,20)",
                "channels": accepted.config.cut.channels,
                "class_count": 20,
                "state_dict_only": True,
            },
            "selection": {
                "threshold_percent": V1_M1_C2_ACCEPTANCE_TOP1_PERCENT,
                "stability_delta_percent": V1_M1_C2_STABILITY_DELTA_PERCENT,
                "rule": (
                    "first validation-passing shallow cut; validation-only H1/H2 selection; "
                    "H1 tie-break"
                ),
            },
            "state": {
                "filename": paths.state.name,
                "byte_size": state_bytes,
                "file_sha256": state_file_sha256,
                "canonical_state_sha256": accepted.state_sha256,
            },
        }
        baseline._atomic_write_bytes(
            paths.manifest,
            (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        report = {
            "schema_version": 1,
            "experiment_id": V1_M1_C2_EXPERIMENT_ID,
            "environment": baseline._environment_report(device),
            "manifest_filename": paths.manifest.name,
            "manifest_sha256": baseline._file_digest(paths.manifest, "sha256"),
            "h1_candidates": [
                {
                    "cut": item.config.cut.value,
                    "selected_epoch": item.selected_epoch,
                    "validation_top1_percent": item.validation.top1_percent,
                    "validation_predictions_sha256": item.validation.predictions_sha256,
                    "state_sha256": item.state_sha256,
                }
                for item in h1_candidates
            ],
            "h2": {
                "cut": h2.config.cut.value,
                "selected_epoch": h2.selected_epoch,
                "validation_top1_percent": h2.validation.top1_percent,
                "validation_predictions_sha256": h2.validation.predictions_sha256,
                "state_sha256": h2.state_sha256,
            },
            "accepted": {
                "run": accepted.config.run_name,
                "cut": accepted.config.cut.value,
                "validation_top1_percent": accepted.validation.top1_percent,
                "selected_epoch": accepted.selected_epoch,
                "state_sha256": accepted.state_sha256,
            },
            "test": asdict(test),
        }
        baseline._atomic_write_bytes(
            paths.report,
            (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        )
    except Exception as error:
        raise V1M1C2ExperimentError("C2 accepted artifact write failed") from error


def preflight_v1_m1_c2(artifact_root: Path | None = None) -> dict[str, object]:
    """执行不读数据、不训练、不写 artifact 的 C2 runner preflight。"""
    expected_parameters = {"layer2": 2_580, "layer3": 5_140, "layer4": 10_260}
    candidates: list[dict[str, object]] = []
    for cut in V1M1C2Cut:
        head = V1M1C2PublicHead(cut.channels)
        parameter_count = sum(parameter.numel() for parameter in head.parameters())
        if parameter_count != expected_parameters[cut.value]:
            raise V1M1C2ExperimentError("C2 public head parameter count changed")
        candidates.append(
            {"cut": cut.value, "channels": cut.channels, "parameter_count": parameter_count}
        )
    if artifact_root is not None:
        _artifact_paths(artifact_root)
    return {
        "experiment_id": V1_M1_C2_EXPERIMENT_ID,
        "run_names": list(V1_M1_C2_RUN_NAMES),
        "run_seeds": dict(V1_M1_C2_RUN_SEEDS),
        "candidate_heads": candidates,
        "threshold_percent": V1_M1_C2_ACCEPTANCE_TOP1_PERCENT,
        "stability_delta_percent": V1_M1_C2_STABILITY_DELTA_PERCENT,
        "writes_artifact": False,
        "downloads_data": False,
        "trains": False,
    }


def run_v1_m1_c2(
    data_root: Path,
    accepted_artifact_root: Path,
    device: torch.device,
    artifact_root: Path = V1_M1_C2_ARTIFACT_ROOT,
) -> V1M1C2ExperimentResult:
    """在 accepted R2 上按 H1/H2 预注册规则训练并保存 public head。"""
    if type(device) is not torch.device or device.type != "cuda":
        raise V1M1C2ExperimentError("C2 public-head training requires an explicit CUDA device")
    paths = _artifact_paths(artifact_root)
    h1_config = V1M1C2HeadTrainingConfig("H1", V1M1C2Cut.LAYER2, 1729)
    data = _load_c2_data(data_root, h1_config)
    accepted_r2 = load_v1_m1_c1_accepted_r2_details(accepted_artifact_root, device)
    model = accepted_r2.model
    h1_candidates = tuple(
        _train_head(
            model,
            V1M1C2HeadTrainingConfig("H1", cut, V1_M1_C2_RUN_SEEDS["H1"]),
            data,
            device,
        )
        for cut in V1M1C2Cut
    )
    accepted_cut = _select_h1_cut(h1_candidates)
    h2 = _train_head(
        model,
        V1M1C2HeadTrainingConfig("H2", accepted_cut, V1_M1_C2_RUN_SEEDS["H2"]),
        data,
        device,
    )
    h1_selected = next(item for item in h1_candidates if item.config.cut is accepted_cut)
    if (
        h2.validation.top1_percent < V1_M1_C2_ACCEPTANCE_TOP1_PERCENT
        or abs(h1_selected.validation.top1_percent - h2.validation.top1_percent)
        > V1_M1_C2_STABILITY_DELTA_PERCENT
    ):
        raise V1M1C2ExperimentError("C2 H1/H2 stability acceptance failed")
    accepted = _select_accepted_head(h1_selected, h2)
    accepted_head = V1M1C2PublicHead(accepted_cut.channels).to(device=device, dtype=torch.float32)
    accepted_head.load_state_dict(accepted.state, strict=True)
    test = _evaluate_head(model, accepted_head, accepted_cut, data.test_loader, device)
    if test.top1_percent < V1_M1_C2_ACCEPTANCE_TOP1_PERCENT:
        raise V1M1C2ExperimentError("C2 accepted public head test threshold failed")
    _write_c2_artifacts(
        paths,
        accepted,
        h1_candidates,
        h2,
        test,
        data,
        accepted_r2.canonical_state_sha256,
        device,
    )
    return V1M1C2ExperimentResult(
        accepted_cut=accepted_cut,
        accepted=accepted,
        h1_candidates=h1_candidates,
        h2=h2,
        test=test,
        coarse_labels_sha256=data.coarse_labels_sha256,
        accepted_r2_state_sha256=accepted_r2.canonical_state_sha256,
        artifacts=paths,
    )


__all__ = [
    "V1_M1_C2_ACCEPTANCE_TOP1_PERCENT",
    "V1_M1_C2_ARTIFACT_ROOT",
    "V1_M1_C2_EXPERIMENT_ID",
    "V1_M1_C2_HEAD_EPOCH_COUNT",
    "V1_M1_C2_RUN_NAMES",
    "V1_M1_C2_RUN_SEEDS",
    "V1M1C2ArtifactPaths",
    "V1M1C2EpochMetrics",
    "V1M1C2ExperimentError",
    "V1M1C2ExperimentResult",
    "V1M1C2HeadMetrics",
    "V1M1C2HeadRunResult",
    "V1M1C2HeadTrainingConfig",
    "V1M1C2RunName",
    "preflight_v1_m1_c2",
    "run_v1_m1_c2",
]
