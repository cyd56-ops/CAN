"""V1-M1-C2 accepted state 的无训练验收、隔离与性能报告。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import struct
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, cast
from unittest.mock import patch

import torch
import torchvision  # type: ignore[import-untyped]
from torch import Tensor, nn

from can.access import (
    A3_V2_CHALLENGE_TTL_MS,
    A3V2Clock,
    A3V2TranscriptStore,
    V1M1C2Coordinator,
    V1M1C2Cut,
    V1M1C2Policy,
    V1M1C2PublicHead,
)
from can.access.v1_m1_adapter import V1_M1_INPUT_PROFILE_SHA256
from can.experiments import v1_m1_baseline as baseline
from can.experiments import v1_m1_c1 as c1
from can.experiments import v1_m1_c2 as training
from can.model import V1_M1_CLASS_COUNT, V1Cifar100ResNet18
from can.reference import V1_PROFILE_ID, V1Abort, V1Challenge, V1Response
from can.verifier import V1NeuralProfile, verify_v1_neural

V1_M1_C2_EVALUATOR_EXPERIMENT_ID: Final = "CAN-V1-M1-C2-ACCEPTED-STATE-v1"
V1_M1_C2_CORRECTION_EXPERIMENT_ID: Final = "CAN-V1-M1-C2-METADATA-CORRECTION-v1"
V1_M1_C2_CORRECTION_FILENAME: Final = "metadata-correction.json"
V1_M1_C2_ACCEPTED_REPORT_FILENAME: Final = "accepted-state-report.json"
V1_M1_C2_LATENCY_WARMUPS: Final = 100
V1_M1_C2_LATENCY_OBSERVATIONS: Final = 1_000
V1_M1_C2_ABORT_ATTEMPTS: Final = 3
_V1_M1_C2_MAX_JSON_BYTES: Final = 2_000_000
_V1_M1_C2_DEVICE: Final = torch.device("cuda:0")


class V1M1C2EvaluatorError(RuntimeError):
    """表示 C2 accepted-state artifact 或 evaluator 违反冻结契约。"""


@dataclass(frozen=True, slots=True)
class V1M1C2AcceptedPublicHead:
    """保存已核验的 C2 public head、选择信息和 artifact 摘要。"""

    cut: V1M1C2Cut
    accepted_run: str
    head: V1M1C2PublicHead
    decoded_data_sha256: str
    coarse_labels_sha256: str
    state_sha256: str
    state_file_sha256: str
    manifest_sha256: str
    report_sha256: str
    training_report: Mapping[str, object]
    metadata_correction_sha256: str | None


@dataclass(frozen=True, slots=True)
class _C2TestData:
    """保存 evaluator 使用的 canonical test pixels 与 fine/coarse labels。"""

    pixels: Tensor
    fine_labels: Tensor
    coarse_labels: Tensor


class _CounterNonce:
    __slots__ = ("_counter",)

    def __init__(self) -> None:
        self._counter = 0

    def __call__(self, size: int) -> bytes:
        self._counter += 1
        return self._counter.to_bytes(size, byteorder="big", signed=False)


class _MutableClock:
    __slots__ = ("monotonic_ns", "wall_ms")

    def __init__(self) -> None:
        self.wall_ms = 1_700_000_000_000
        self.monotonic_ns = 5_000_000_000


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect_exact_keys(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys or not all(type(key) is str for key in value):
        raise V1M1C2EvaluatorError(f"{name} fields are not canonical")
    return cast(dict[str, object], value)


def _reject_duplicate_json_keys(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("JSON object contains duplicate or non-string fields")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise V1M1C2EvaluatorError(f"{name} is missing or symlinked")
    try:
        size = path.stat().st_size
        if size < 1 or size > _V1_M1_C2_MAX_JSON_BYTES:
            raise V1M1C2EvaluatorError(f"{name} has an invalid byte size")
        value: object = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except V1M1C2EvaluatorError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V1M1C2EvaluatorError(f"{name} is not canonical JSON") from error
    if type(value) is not dict:
        raise V1M1C2EvaluatorError(f"{name} root must be an exact object")
    return cast(dict[str, object], value)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise V1M1C2EvaluatorError("C2 artifact cannot be read") from error
    return digest.hexdigest()


def _finite_percentage(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise V1M1C2EvaluatorError("C2 metric percentage is not canonical")
    return value


def _validate_metric(value: object, name: str, *, expected_total: int | None = None) -> float:
    metric = _expect_exact_keys(
        value,
        {"loss", "top1_percent", "correct_top1", "total", "predictions_sha256"},
        name,
    )
    if (
        type(metric["loss"]) is not float
        or not math.isfinite(metric["loss"])
        or type(metric["correct_top1"]) is not int
        or type(metric["total"]) is not int
        or metric["total"] < 1
        or not 0 <= metric["correct_top1"] <= metric["total"]
        or not _is_sha256(metric["predictions_sha256"])
    ):
        raise V1M1C2EvaluatorError(f"{name} values are not canonical")
    if expected_total is not None and metric["total"] != expected_total:
        raise V1M1C2EvaluatorError(f"{name} total changed")
    top1 = _finite_percentage(metric["top1_percent"])
    expected_top1 = metric["correct_top1"] * 100.0 / metric["total"]
    if top1 != expected_top1:
        raise V1M1C2EvaluatorError(f"{name} percentage is inconsistent")
    return top1


def _parse_cut(value: object) -> V1M1C2Cut:
    if type(value) is not str:
        raise V1M1C2EvaluatorError("C2 cut is not canonical")
    try:
        return V1M1C2Cut(value)
    except ValueError as error:
        raise V1M1C2EvaluatorError("C2 cut is not registered") from error


def _validate_c2_manifest(
    manifest: dict[str, object],
    state_path: Path,
) -> tuple[V1M1C2Cut, str, str, str, str, str, bool]:
    parsed = _expect_exact_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "accepted_cut",
            "accepted_run",
            "accepted_r2_state_sha256",
            "input_profile_sha256",
            "data",
            "head",
            "selection",
            "state",
        },
        "C2 manifest",
    )
    if parsed["schema_version"] != 1 or parsed["experiment_id"] != training.V1_M1_C2_EXPERIMENT_ID:
        raise V1M1C2EvaluatorError("C2 manifest identity changed")
    cut = _parse_cut(parsed["accepted_cut"])
    accepted_run = parsed["accepted_run"]
    if accepted_run not in training.V1_M1_C2_RUN_NAMES:
        raise V1M1C2EvaluatorError("C2 accepted run is not registered")
    if parsed["input_profile_sha256"] != V1_M1_INPUT_PROFILE_SHA256.hex():
        raise V1M1C2EvaluatorError("C2 input profile binding changed")

    data = _expect_exact_keys(
        parsed["data"],
        {
            "archive",
            "decoded_sha256",
            "coarse_labels_sha256",
            "train_size",
            "validation_size",
            "test_size",
        },
        "C2 manifest data",
    )
    archive = _expect_exact_keys(
        data["archive"], {"filename", "byte_size", "sha256", "md5"}, "C2 archive"
    )
    if archive != {
        "filename": baseline.V1_M1_ARCHIVE_FILENAME,
        "byte_size": baseline.V1_M1_ARCHIVE_SIZE,
        "sha256": baseline.V1_M1_ARCHIVE_SHA256,
        "md5": baseline.V1_M1_ARCHIVE_MD5,
    }:
        raise V1M1C2EvaluatorError("C2 archive identity changed")
    if (
        not _is_sha256(data["decoded_sha256"])
        or not _is_sha256(data["coarse_labels_sha256"])
        or data["train_size"] != baseline.V1_M1_TRAIN_SIZE
        or data["validation_size"] != baseline.V1_M1_VALIDATION_SIZE
        or data["test_size"] != baseline.V1_M1_TEST_SIZE
    ):
        raise V1M1C2EvaluatorError("C2 manifest data contract changed")

    head = _expect_exact_keys(
        parsed["head"], {"topology", "channels", "class_count", "state_dict_only"}, "C2 head"
    )
    if head != {
        "topology": "AdaptiveAvgPool2d(1)->Flatten->Linear(C_cut,20)",
        "channels": cut.channels,
        "class_count": 20,
        "state_dict_only": True,
    }:
        raise V1M1C2EvaluatorError("C2 head contract changed")
    selection = _expect_exact_keys(
        parsed["selection"], {"threshold_percent", "stability_delta_percent", "rule"}, "selection"
    )
    if selection != {
        "threshold_percent": training.V1_M1_C2_ACCEPTANCE_TOP1_PERCENT,
        "stability_delta_percent": training.V1_M1_C2_STABILITY_DELTA_PERCENT,
        "rule": (
            "first validation-passing shallow cut; validation-only H1/H2 selection; H1 tie-break"
        ),
    }:
        raise V1M1C2EvaluatorError("C2 selection contract changed")

    state = _expect_exact_keys(
        parsed["state"],
        {"filename", "byte_size", "file_sha256", "canonical_state_sha256"},
        "C2 state",
    )
    if (
        state_path.is_symlink()
        or not state_path.is_file()
        or state["filename"] != training.V1_M1_C2_STATE_FILENAME
        or type(state["byte_size"]) is not int
        or not 1 <= state["byte_size"] <= training.V1_M1_C2_MAX_STATE_BYTES
        or not _is_sha256(state["file_sha256"])
        or not _is_sha256(state["canonical_state_sha256"])
        or state_path.stat().st_size != state["byte_size"]
        or _file_digest(state_path) != state["file_sha256"]
    ):
        raise V1M1C2EvaluatorError("C2 public-head state binding changed")
    decoded_sha256 = cast(str, data["decoded_sha256"])
    r2_binding = parsed["accepted_r2_state_sha256"]
    if not _is_sha256(r2_binding):
        raise V1M1C2EvaluatorError("C2 accepted R2 binding is not canonical")
    legacy_bug = r2_binding == decoded_sha256
    if not legacy_bug and r2_binding != c1.V1_M1_C1_ACCEPTED_STATE_SHA256:
        raise V1M1C2EvaluatorError("C2 accepted R2 state binding changed")
    return (
        cut,
        accepted_run,
        decoded_sha256,
        cast(str, data["coarse_labels_sha256"]),
        cast(str, state["canonical_state_sha256"]),
        state["file_sha256"],
        legacy_bug,
    )


def _validate_training_report(
    report: dict[str, object],
    manifest_path: Path,
    cut: V1M1C2Cut,
    accepted_run: str,
    state_sha256: str,
) -> None:
    parsed = _expect_exact_keys(
        report,
        {
            "schema_version",
            "experiment_id",
            "environment",
            "manifest_filename",
            "manifest_sha256",
            "h1_candidates",
            "h2",
            "accepted",
            "test",
        },
        "C2 training report",
    )
    if (
        parsed["schema_version"] != 1
        or parsed["experiment_id"] != training.V1_M1_C2_EXPERIMENT_ID
        or parsed["manifest_filename"] != training.V1_M1_C2_MANIFEST_FILENAME
        or parsed["manifest_sha256"] != _file_digest(manifest_path)
    ):
        raise V1M1C2EvaluatorError("C2 training report binding changed")
    environment = _expect_exact_keys(
        parsed["environment"],
        {
            "platform",
            "python",
            "torch",
            "cuda_runtime",
            "device",
            "device_name",
            "python_hash_seed",
            "cublas_workspace_config",
            "deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        },
        "C2 training environment",
    )
    if (
        type(environment["platform"]) is not str
        or environment["python"] != "3.11.9"
        or environment["torch"] != "2.13.0+cu126"
        or environment["cuda_runtime"] != "12.6"
        or environment["device"] != "cuda:0"
        or environment["device_name"] != "NVIDIA RTX A4000"
        or environment["python_hash_seed"] != "1730"
        or environment["cublas_workspace_config"] != ":4096:8"
        or environment["deterministic_algorithms"] is not True
        or environment["cudnn_benchmark"] is not False
        or environment["cudnn_deterministic"] is not True
        or environment["cuda_matmul_allow_tf32"] is not False
        or environment["cudnn_allow_tf32"] is not False
    ):
        raise V1M1C2EvaluatorError("C2 training environment binding changed")
    candidates = parsed["h1_candidates"]
    if type(candidates) is not list or len(candidates) != 3:
        raise V1M1C2EvaluatorError("C2 H1 candidates changed")
    h1_scores: dict[V1M1C2Cut, float] = {}
    h1_states: dict[V1M1C2Cut, str] = {}
    for expected_cut, value in zip(V1M1C2Cut, candidates, strict=True):
        candidate = _expect_exact_keys(
            value,
            {
                "cut",
                "selected_epoch",
                "validation_top1_percent",
                "validation_predictions_sha256",
                "state_sha256",
            },
            "C2 H1 candidate",
        )
        if (
            candidate["cut"] != expected_cut.value
            or type(candidate["selected_epoch"]) is not int
            or not 1 <= candidate["selected_epoch"] <= training.V1_M1_C2_HEAD_EPOCH_COUNT
            or not _is_sha256(candidate["validation_predictions_sha256"])
            or not _is_sha256(candidate["state_sha256"])
        ):
            raise V1M1C2EvaluatorError("C2 H1 candidate values changed")
        h1_scores[expected_cut] = _finite_percentage(candidate["validation_top1_percent"])
        h1_states[expected_cut] = cast(str, candidate["state_sha256"])
    selected_cut = next(
        (candidate for candidate in V1M1C2Cut if h1_scores[candidate] >= 75.0),
        None,
    )
    if selected_cut is not cut:
        raise V1M1C2EvaluatorError("C2 H1 shallow-cut selection changed")

    h2 = _expect_exact_keys(
        parsed["h2"],
        {
            "cut",
            "selected_epoch",
            "validation_top1_percent",
            "validation_predictions_sha256",
            "state_sha256",
        },
        "C2 H2",
    )
    h2_score = _finite_percentage(h2["validation_top1_percent"])
    if (
        h2["cut"] != cut.value
        or type(h2["selected_epoch"]) is not int
        or not 1 <= h2["selected_epoch"] <= training.V1_M1_C2_HEAD_EPOCH_COUNT
        or not _is_sha256(h2["validation_predictions_sha256"])
        or not _is_sha256(h2["state_sha256"])
        or h2_score < training.V1_M1_C2_ACCEPTANCE_TOP1_PERCENT
        or abs(h1_scores[cut] - h2_score) > training.V1_M1_C2_STABILITY_DELTA_PERCENT
    ):
        raise V1M1C2EvaluatorError("C2 H2 stability contract changed")
    expected_run = "H1" if h1_scores[cut] >= h2_score else "H2"
    expected_state = h1_states[cut] if expected_run == "H1" else cast(str, h2["state_sha256"])
    expected_epoch = (
        cast(dict[str, object], candidates[tuple(V1M1C2Cut).index(cut)])["selected_epoch"]
        if expected_run == "H1"
        else h2["selected_epoch"]
    )
    accepted = _expect_exact_keys(
        parsed["accepted"],
        {"run", "cut", "validation_top1_percent", "selected_epoch", "state_sha256"},
        "C2 accepted head",
    )
    if (
        accepted["run"] != accepted_run
        or accepted_run != expected_run
        or accepted["cut"] != cut.value
        or accepted["state_sha256"] != state_sha256
        or state_sha256 != expected_state
        or accepted["selected_epoch"] != expected_epoch
        or accepted["validation_top1_percent"]
        != (h1_scores[cut] if accepted_run == "H1" else h2_score)
    ):
        raise V1M1C2EvaluatorError("C2 accepted head selection changed")
    if _validate_metric(parsed["test"], "C2 test", expected_total=baseline.V1_M1_TEST_SIZE) < 75.0:
        raise V1M1C2EvaluatorError("C2 test utility threshold failed")


def _load_head_state(
    state_path: Path,
    cut: V1M1C2Cut,
    expected_state_sha256: str,
    device: torch.device,
) -> V1M1C2PublicHead:
    if state_path.is_symlink() or not state_path.is_file():
        raise V1M1C2EvaluatorError("C2 public-head state is missing or symlinked")
    try:
        value: object = torch.load(state_path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise V1M1C2EvaluatorError("C2 public-head state cannot be loaded") from error
    if type(value) is not dict or set(value) != {"classifier.weight", "classifier.bias"}:
        raise V1M1C2EvaluatorError("C2 public-head state contains unexpected entries")
    state = cast(dict[str, object], value)
    expected_shapes = {
        "classifier.weight": (20, cut.channels),
        "classifier.bias": (20,),
    }
    tensors: dict[str, Tensor] = {}
    for name, shape in expected_shapes.items():
        tensor = state[name]
        if (
            type(tensor) is not Tensor
            or tensor.dtype is not torch.float32
            or tensor.device.type != "cpu"
            or tensor.device.index is not None
            or tensor.layout is not torch.strided
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise V1M1C2EvaluatorError("C2 public-head tensor contract changed")
        tensors[name] = tensor
    if training._head_state_digest(tensors) != expected_state_sha256:
        raise V1M1C2EvaluatorError("C2 canonical public-head state digest changed")
    head = V1M1C2PublicHead(cut.channels)
    head.load_state_dict(tensors, strict=True)
    head.to(device=device, dtype=torch.float32).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def _validate_correction(
    correction: dict[str, object],
    manifest_path: Path,
    report_path: Path,
    state_path: Path,
    decoded_sha256: str,
) -> None:
    parsed = _expect_exact_keys(
        correction,
        {
            "schema_version",
            "experiment_id",
            "reason",
            "source_manifest_sha256",
            "source_report_sha256",
            "source_state_file_sha256",
            "legacy_mislabeled_value",
            "accepted_r2_state_sha256",
        },
        "C2 metadata correction",
    )
    if parsed != {
        "schema_version": 1,
        "experiment_id": V1_M1_C2_CORRECTION_EXPERIMENT_ID,
        "reason": "legacy runner wrote decoded_data_sha256 under accepted_r2_state_sha256",
        "source_manifest_sha256": _file_digest(manifest_path),
        "source_report_sha256": _file_digest(report_path),
        "source_state_file_sha256": _file_digest(state_path),
        "legacy_mislabeled_value": decoded_sha256,
        "accepted_r2_state_sha256": c1.V1_M1_C1_ACCEPTED_STATE_SHA256,
    }:
        raise V1M1C2EvaluatorError("C2 metadata correction binding changed")


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise V1M1C2EvaluatorError("C2 artifact directory is not canonical")
    if path.exists() or path.is_symlink():
        raise V1M1C2EvaluatorError("refusing to overwrite an existing C2 evaluator artifact")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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
        os.link(temporary_path, path, follow_symlinks=False)
    except FileExistsError as error:
        raise V1M1C2EvaluatorError(
            "refusing to overwrite an existing C2 evaluator artifact"
        ) from error
    except OSError as error:
        raise V1M1C2EvaluatorError("C2 evaluator artifact write failed") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def materialize_v1_m1_c2_metadata_correction(
    artifact_root: Path = training.V1_M1_C2_ARTIFACT_ROOT,
) -> Path:
    """为已知旧 runner 的错误字段追加不可覆盖的摘要修正记录。"""
    if (
        not isinstance(artifact_root, Path)
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
    ):
        raise V1M1C2EvaluatorError("C2 artifact root is missing or symlinked")
    manifest_path = artifact_root / training.V1_M1_C2_MANIFEST_FILENAME
    report_path = artifact_root / training.V1_M1_C2_REPORT_FILENAME
    state_path = artifact_root / training.V1_M1_C2_STATE_FILENAME
    manifest = _read_json_object(manifest_path, "C2 manifest")
    cut, accepted_run, decoded, _coarse, state_digest, _file_sha, legacy_bug = (
        _validate_c2_manifest(manifest, state_path)
    )
    _validate_training_report(
        _read_json_object(report_path, "C2 training report"),
        manifest_path,
        cut,
        accepted_run,
        state_digest,
    )
    if not legacy_bug:
        raise V1M1C2EvaluatorError("C2 artifact does not require metadata correction")
    correction_path = artifact_root / V1_M1_C2_CORRECTION_FILENAME
    correction = {
        "schema_version": 1,
        "experiment_id": V1_M1_C2_CORRECTION_EXPERIMENT_ID,
        "reason": "legacy runner wrote decoded_data_sha256 under accepted_r2_state_sha256",
        "source_manifest_sha256": _file_digest(manifest_path),
        "source_report_sha256": _file_digest(report_path),
        "source_state_file_sha256": _file_digest(state_path),
        "legacy_mislabeled_value": decoded,
        "accepted_r2_state_sha256": c1.V1_M1_C1_ACCEPTED_STATE_SHA256,
    }
    _atomic_write_json(correction_path, correction)
    return correction_path


def load_v1_m1_c2_accepted_public_head(
    artifact_root: Path,
    device: torch.device,
) -> V1M1C2AcceptedPublicHead:
    """核验 C2 training artifact, 加载冻结 public head。"""
    if type(device) is not torch.device:
        raise V1M1C2EvaluatorError("C2 evaluator device must use torch.device")
    if (
        not isinstance(artifact_root, Path)
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
    ):
        raise V1M1C2EvaluatorError("C2 artifact root is missing or symlinked")
    manifest_path = artifact_root / training.V1_M1_C2_MANIFEST_FILENAME
    report_path = artifact_root / training.V1_M1_C2_REPORT_FILENAME
    state_path = artifact_root / training.V1_M1_C2_STATE_FILENAME
    manifest = _read_json_object(manifest_path, "C2 manifest")
    cut, accepted_run, decoded, coarse, state_digest, state_file_digest, legacy_bug = (
        _validate_c2_manifest(manifest, state_path)
    )
    report = _read_json_object(report_path, "C2 training report")
    _validate_training_report(report, manifest_path, cut, accepted_run, state_digest)
    correction_sha256: str | None = None
    if legacy_bug:
        correction_path = artifact_root / V1_M1_C2_CORRECTION_FILENAME
        _validate_correction(
            _read_json_object(correction_path, "C2 metadata correction"),
            manifest_path,
            report_path,
            state_path,
            decoded,
        )
        correction_sha256 = _file_digest(correction_path)
    return V1M1C2AcceptedPublicHead(
        cut=cut,
        accepted_run=accepted_run,
        head=_load_head_state(state_path, cut, state_digest, device),
        decoded_data_sha256=decoded,
        coarse_labels_sha256=coarse,
        state_sha256=state_digest,
        state_file_sha256=state_file_digest,
        manifest_sha256=_file_digest(manifest_path),
        report_sha256=_file_digest(report_path),
        training_report=report,
        metadata_correction_sha256=correction_sha256,
    )


def _load_c2_test_data(
    data_root: Path,
    expected_decoded_sha256: str,
    expected_coarse_sha256: str,
) -> _C2TestData:
    if not isinstance(data_root, Path):
        raise V1M1C2EvaluatorError("C2 data root must be pathlib.Path")
    baseline.verify_v1_m1_archive(data_root)
    extracted_root = data_root / "cifar-100-python"
    if extracted_root.is_symlink() or not extracted_root.is_dir():
        raise V1M1C2EvaluatorError("verified CIFAR-100 archive has not been explicitly extracted")
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
    decoded_sha256 = baseline._decoded_digest(
        train_pixels,
        train_fine,
        test_pixels,
        test_fine,
    )
    coarse_sha256 = training._coarse_labels_digest(train_coarse, test_coarse)
    del train_pixels, train_fine, train_coarse
    if decoded_sha256 != expected_decoded_sha256:
        raise V1M1C2EvaluatorError("C2 decoded data digest does not match the artifact")
    if coarse_sha256 != expected_coarse_sha256:
        raise V1M1C2EvaluatorError("C2 coarse-label digest does not match the artifact")
    return _C2TestData(test_pixels, test_fine, test_coarse)


def _prefix_terminal_module(model: V1Cifar100ResNet18, cut: V1M1C2Cut) -> nn.Module:
    return {
        V1M1C2Cut.LAYER2: model.layer2,
        V1M1C2Cut.LAYER3: model.layer3,
        V1M1C2Cut.LAYER4: model.layer4,
    }[cut]


class _RouteInstrumentation:
    """通过实验 hook 记录真实 route calls, 捕获不落盘的当前 logits。"""

    __slots__ = (
        "counts",
        "protected_logits",
        "public_logits",
        "route",
    )

    def __init__(self) -> None:
        self.route: str | None = None
        self.counts = {
            "public": {"prefix": 0, "public_head": 0, "protected_suffix": 0},
            "protected": {"prefix": 0, "public_head": 0, "protected_suffix": 0},
        }
        self.public_logits: Tensor | None = None
        self.protected_logits: Tensor | None = None

    def prefix_start_hook(self, _module: nn.Module, _inputs: object) -> None:
        if self.route in self.counts:
            self.counts[self.route]["prefix"] += 1

    def public_head_start_hook(self, _module: nn.Module, _inputs: object) -> None:
        if self.route == "public":
            self.counts["public"]["public_head"] += 1

    def public_head_output_hook(self, _module: nn.Module, _inputs: object, output: object) -> None:
        if self.route != "public":
            return
        if type(output) is not Tensor or self.public_logits is not None:
            raise V1M1C2EvaluatorError("public head hook did not observe exactly one tensor")
        self.public_logits = output.detach().cpu().contiguous().clone()

    def protected_suffix_start_hook(self, _module: nn.Module, _inputs: object) -> None:
        if self.route == "protected":
            self.counts["protected"]["protected_suffix"] += 1

    def protected_suffix_output_hook(
        self, _module: nn.Module, _inputs: object, output: object
    ) -> None:
        if self.route != "protected":
            return
        if type(output) is not Tensor or self.protected_logits is not None:
            raise V1M1C2EvaluatorError("protected suffix hook did not observe exactly one tensor")
        self.protected_logits = output.detach().cpu().contiguous().clone()

    def take_public_logits(self) -> Tensor:
        if self.public_logits is None:
            raise V1M1C2EvaluatorError("public route did not execute its head")
        logits = self.public_logits
        self.public_logits = None
        return logits

    def take_protected_logits(self) -> Tensor:
        if self.protected_logits is None:
            raise V1M1C2EvaluatorError("protected route did not execute its suffix")
        logits = self.protected_logits
        self.protected_logits = None
        return logits


def _build_c2_coordinator(
    model: V1Cifar100ResNet18,
    accepted: V1M1C2AcceptedPublicHead,
    neural_profile: V1NeuralProfile,
    *,
    store: A3V2TranscriptStore | None = None,
    events: list[str] | Callable[[str], None] | None = None,
) -> V1M1C2Coordinator:
    if type(neural_profile) is not V1NeuralProfile:
        raise V1M1C2EvaluatorError("C2 neural profile is not canonical")
    return V1M1C2Coordinator(
        neural_profile,
        model,
        cut=accepted.cut,
        public_head=accepted.head,
        policy=V1M1C2Policy(public_entry_enabled=True),
        store=store,
        challenge_sampler=c1._fixed_challenge_sampler,
        event_sink=(
            None if events is None else events.append if isinstance(events, list) else events
        ),
    )


def _update_public_digests(
    logits_digest: hashlib._Hash,
    predictions_digest: hashlib._Hash,
    logits: Tensor,
) -> int:
    if (
        logits.dtype is not torch.float32
        or logits.device.type != "cpu"
        or logits.device.index is not None
        or not logits.is_contiguous()
        or tuple(logits.shape) != (1, 20)
    ):
        raise V1M1C2EvaluatorError("C2 public logits contract changed")
    logits_digest.update(logits.numpy().tobytes())
    prediction = int(logits.argmax(dim=1).item())
    predictions_digest.update(struct.pack(">q", prediction))
    return prediction


def _evaluate_routes(
    model: V1Cifar100ResNet18,
    accepted: V1M1C2AcceptedPublicHead,
    device: torch.device,
    data: _C2TestData,
    neural_profile: V1NeuralProfile,
) -> dict[str, object]:
    baseline_logits, baseline_predictions = c1._evaluate_direct_r2_baseline_reference(
        model,
        device,
        data.pixels,
    )
    if baseline_predictions != c1.V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256:
        raise V1M1C2EvaluatorError("direct R2 predictions do not match accepted R2")
    print(
        f"C2 direct R2 reference evaluated={baseline.V1_M1_TEST_SIZE}/{baseline.V1_M1_TEST_SIZE}",
        flush=True,
    )

    events: list[str] = []
    coordinator = _build_c2_coordinator(model, accepted, neural_profile, events=events)
    instrumentation = _RouteInstrumentation()
    handles = (
        _prefix_terminal_module(model, accepted.cut).register_forward_pre_hook(
            instrumentation.prefix_start_hook
        ),
        accepted.head.register_forward_pre_hook(instrumentation.public_head_start_hook),
        accepted.head.register_forward_hook(instrumentation.public_head_output_hook),
        model.classifier.register_forward_pre_hook(instrumentation.protected_suffix_start_hook),
        model.classifier.register_forward_hook(instrumentation.protected_suffix_output_hook),
    )
    public_logits_digest = hashlib.sha256()
    public_predictions_digest = hashlib.sha256()
    direct_logits_digest = hashlib.sha256()
    direct_predictions_digest = hashlib.sha256()
    split_logits_digest = hashlib.sha256()
    split_predictions_digest = hashlib.sha256()
    public_correct = 0
    direct_correct = 0
    split_correct = 0
    max_absolute_error = 0.0
    max_relative_error = 0.0
    public_events = (
        "preprocess_start",
        "prefix_start",
        "public_head_start",
        "response_release",
    )
    protected_events = (
        "verifier_accept",
        "coordinator_commit(PROTECTED)",
        "preprocess_start",
        "prefix_start",
        "suffix_start",
        "internal_result_commit",
        "response_release",
    )
    try:
        for index in range(baseline.V1_M1_TEST_SIZE):
            image = data.pixels[index : index + 1].contiguous()
            fine_label = int(data.fine_labels[index].item())
            coarse_label = int(data.coarse_labels[index].item())

            events.clear()
            instrumentation.route = "public"
            public_result = coordinator.handle_public(image)
            instrumentation.route = None
            public_logits = instrumentation.take_public_logits()
            if tuple(events) != public_events or public_result.get("status") != "public":
                raise V1M1C2EvaluatorError("C2 public event order or response changed")
            public_prediction = _update_public_digests(
                public_logits_digest,
                public_predictions_digest,
                public_logits,
            )
            if public_result.get("coarse_class_id") != public_prediction:
                raise V1M1C2EvaluatorError("C2 public response differs from captured logits")
            public_correct += int(public_prediction == coarse_label)

            direct_logits = c1._direct_logits(model, image, device)
            direct_prediction = c1._update_logit_digest(direct_logits_digest, direct_logits)
            c1._update_prediction_digest(direct_predictions_digest, direct_prediction)
            direct_correct += int(direct_prediction == fine_label)

            response_polynomials = c1._conformance_response(index)
            commitment = c1._build_conformance_commitment(
                neural_profile.public_profile,
                response_polynomials,
            )
            events.clear()
            issued = coordinator.begin_protected(image, commitment)
            response = c1._response_for_issue(cast(dict[str, object], issued), response_polynomials)
            instrumentation.route = "protected"
            protected_result = coordinator.respond_protected(response)
            instrumentation.route = None
            split_logits = instrumentation.take_protected_logits()
            if tuple(events) != protected_events or protected_result.get("status") != "protected":
                raise V1M1C2EvaluatorError("C2 protected event order or response changed")
            split_prediction = c1._update_logit_digest(split_logits_digest, split_logits)
            c1._update_prediction_digest(split_predictions_digest, split_prediction)
            if protected_result.get("class_id") != split_prediction:
                raise V1M1C2EvaluatorError("C2 protected response differs from split logits")
            split_correct += int(split_prediction == fine_label)
            difference = (direct_logits - split_logits).abs()
            max_absolute_error = max(max_absolute_error, float(difference.max().item()))
            denominator = torch.maximum(
                torch.maximum(direct_logits.abs(), split_logits.abs()),
                torch.tensor(1e-12, dtype=torch.float32),
            )
            max_relative_error = max(
                max_relative_error,
                float((difference / denominator).max().item()),
            )
            if not torch.equal(direct_logits, split_logits):
                raise V1M1C2EvaluatorError("C2 split protected logits differ from direct R2")
            if (
                index + 1
            ) % baseline.V1_M1_EVALUATION_BATCH_SIZE == 0 or index + 1 == baseline.V1_M1_TEST_SIZE:
                print(
                    f"C2 public/protected evaluated={index + 1}/{baseline.V1_M1_TEST_SIZE}",
                    flush=True,
                )
    finally:
        instrumentation.route = None
        for handle in handles:
            handle.remove()
    snapshot = coordinator.snapshot()
    expected_calls = baseline.V1_M1_TEST_SIZE
    if (
        instrumentation.public_logits is not None
        or instrumentation.protected_logits is not None
        or instrumentation.counts["public"]
        != {"prefix": expected_calls, "public_head": expected_calls, "protected_suffix": 0}
        or instrumentation.counts["protected"]
        != {"prefix": expected_calls, "public_head": 0, "protected_suffix": expected_calls}
        or snapshot.verifier_calls != expected_calls
        or snapshot.allow_commits != expected_calls
        or snapshot.protected_calls != expected_calls
    ):
        raise V1M1C2EvaluatorError("C2 accepted route call accounting changed")
    if (
        public_correct * 100.0 / expected_calls < training.V1_M1_C2_ACCEPTANCE_TOP1_PERCENT
        or direct_correct != split_correct
        or direct_logits_digest.hexdigest() != split_logits_digest.hexdigest()
        or direct_predictions_digest.hexdigest() != split_predictions_digest.hexdigest()
    ):
        raise V1M1C2EvaluatorError("C2 accepted route metrics or digests changed")
    return {
        "test_size": expected_calls,
        "public": {
            "class_count": 20,
            "correct_top1": public_correct,
            "top1_percent": public_correct * 100.0 / expected_calls,
            "logits_sha256": public_logits_digest.hexdigest(),
            "predictions_sha256": public_predictions_digest.hexdigest(),
        },
        "protected": {
            "class_count": V1_M1_CLASS_COUNT,
            "correct_top1": split_correct,
            "top1_percent": split_correct * 100.0 / expected_calls,
            "baseline_batch_size": baseline.V1_M1_EVALUATION_BATCH_SIZE,
            "baseline_direct_logits_sha256": baseline_logits,
            "baseline_direct_predictions_sha256": baseline_predictions,
            "comparison_batch_size": 1,
            "bitwise_logits_equal": True,
            "top1_predictions_equal": True,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "direct_logits_sha256": direct_logits_digest.hexdigest(),
            "split_logits_sha256": split_logits_digest.hexdigest(),
            "direct_predictions_sha256": direct_predictions_digest.hexdigest(),
            "split_predictions_sha256": split_predictions_digest.hexdigest(),
            "direct_correct_top1": direct_correct,
            "direct_top1_percent": direct_correct * 100.0 / expected_calls,
        },
        "call_matrix": {
            "public_success": {
                "verifier": 0,
                **instrumentation.counts["public"],
                "external_status": "public",
            },
            "protected_success": {
                "verifier": snapshot.verifier_calls,
                **instrumentation.counts["protected"],
                "external_status": "protected",
            },
        },
        "event_order": {
            "public_success": list(public_events),
            "protected_success": list(protected_events),
        },
    }


def _new_store(clock: _MutableClock | None = None) -> A3V2TranscriptStore:
    selected = _MutableClock() if clock is None else clock
    return A3V2TranscriptStore(
        clock=A3V2Clock(lambda: selected.wall_ms, lambda: selected.monotonic_ns),
        random_bytes=_CounterNonce(),
    )


def _issue_response(
    coordinator: V1M1C2Coordinator,
    image: Tensor,
    neural_profile: V1NeuralProfile,
    index: int,
) -> tuple[bytes, bytes]:
    response_polynomials = c1._conformance_response(index)
    commitment = c1._build_conformance_commitment(
        neural_profile.public_profile,
        response_polynomials,
    )
    issued = coordinator.begin_protected(image, commitment)
    return commitment, c1._response_for_issue(
        cast(dict[str, object], issued),
        response_polynomials,
    )


def _probe_fail_closed(
    model: V1Cifar100ResNet18,
    accepted: V1M1C2AcceptedPublicHead,
    image: Tensor,
    neural_profile: V1NeuralProfile,
) -> dict[str, object]:
    tamper_events: list[str] = []
    tamper = _build_c2_coordinator(model, accepted, neural_profile, events=tamper_events)
    _commitment, response = _issue_response(tamper, image, neural_profile, 0)
    if tamper.respond_protected(c1._tampered_response(response)) != {
        "version": 5,
        "status": "deny",
    }:
        raise V1M1C2EvaluatorError("C2 canonical tamper did not deny")
    tamper_snapshot = tamper.snapshot()
    if (
        tamper_snapshot.verifier_calls != 1
        or tamper_snapshot.protected_calls != 0
        or any("prefix_start" in event for event in tamper_events)
    ):
        raise V1M1C2EvaluatorError("C2 canonical tamper isolation changed")

    malformed = _build_c2_coordinator(model, accepted, neural_profile)
    if malformed.respond_protected(b"bad") != {"version": 5, "status": "deny"}:
        raise V1M1C2EvaluatorError("C2 malformed response did not deny")
    malformed_snapshot = malformed.snapshot()
    if malformed_snapshot.verifier_calls != 0 or malformed_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 malformed response invoked protected work")

    replay = _build_c2_coordinator(model, accepted, neural_profile)
    _commitment, response = _issue_response(replay, image, neural_profile, 1)
    if replay.respond_protected(response).get("status") != "protected":
        raise V1M1C2EvaluatorError("C2 replay probe initial request did not protect")
    before_replay = replay.snapshot()
    if replay.respond_protected(response) != {"version": 5, "status": "deny"}:
        raise V1M1C2EvaluatorError("C2 replay did not deny")
    after_replay = replay.snapshot()
    if after_replay.protected_calls != before_replay.protected_calls:
        raise V1M1C2EvaluatorError("C2 replay invoked protected work")

    expiry_clock = _MutableClock()
    expiry = _build_c2_coordinator(
        model,
        accepted,
        neural_profile,
        store=_new_store(expiry_clock),
    )
    _commitment, response = _issue_response(expiry, image, neural_profile, 2)
    expiry_clock.monotonic_ns += A3_V2_CHALLENGE_TTL_MS * 1_000_000
    if expiry.respond_protected(response) != {"version": 5, "status": "deny"}:
        raise V1M1C2EvaluatorError("C2 expiry did not deny")
    expiry_snapshot = expiry.snapshot()
    if expiry_snapshot.expiries != 1 or expiry_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 expiry isolation changed")

    abort = _build_c2_coordinator(model, accepted, neural_profile)
    for index in range(V1_M1_C2_ABORT_ATTEMPTS):
        response_polynomials = c1._conformance_response(index + 3)
        commitment = c1._build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        issued = abort.begin_protected(image, commitment)
        transcript_id = issued.get("transcript_id")
        if issued.get("status") != "challenge" or type(transcript_id) is not bytes:
            raise V1M1C2EvaluatorError("C2 abort probe did not receive a challenge")
        if abort.abort_protected(V1Abort(transcript_id).encode()) != {
            "version": 5,
            "status": "deny",
        }:
            raise V1M1C2EvaluatorError("C2 abort did not deny")
    abort_snapshot = abort.snapshot()
    if abort_snapshot.aborts != V1_M1_C2_ABORT_ATTEMPTS or abort_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 abort isolation changed")

    source = _build_c2_coordinator(model, accepted, neural_profile)
    target = _build_c2_coordinator(model, accepted, neural_profile)
    _commitment, response = _issue_response(source, image, neural_profile, 7)
    if target.respond_protected(response) != {"version": 5, "status": "deny"}:
        raise V1M1C2EvaluatorError("C2 route confusion did not deny")
    target_snapshot = target.snapshot()
    if target_snapshot.verifier_calls != 0 or target_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 route confusion invoked protected work")

    cross_input_source = _build_c2_coordinator(model, accepted, neural_profile)
    cross_input_target = _build_c2_coordinator(model, accepted, neural_profile)
    _commitment, source_response = _issue_response(
        cross_input_source,
        image,
        neural_profile,
        8,
    )
    other_image = image.clone()
    other_image[0, 0, 0, 0] ^= 1
    _issue_response(cross_input_target, other_image, neural_profile, 9)
    if cross_input_target.respond_protected(source_response) != {
        "version": 5,
        "status": "deny",
    }:
        raise V1M1C2EvaluatorError("C2 cross-input transcript confusion did not deny")
    cross_input_snapshot = cross_input_target.snapshot()
    if cross_input_snapshot.verifier_calls != 0 or cross_input_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 cross-input confusion invoked protected work")

    injection = _build_c2_coordinator(model, accepted, neural_profile)
    if injection.handle_public(image, cut="layer2") != {"version": 5, "status": "deny"}:
        raise V1M1C2EvaluatorError("C2 public route injection did not deny")
    response_polynomials = c1._conformance_response(8)
    commitment = c1._build_conformance_commitment(
        neural_profile.public_profile,
        response_polynomials,
    )
    if injection.begin_protected(image, commitment, entry="public") != {
        "version": 5,
        "status": "deny",
    }:
        raise V1M1C2EvaluatorError("C2 protected route injection did not deny")
    injection_snapshot = injection.snapshot()
    if injection_snapshot.verifier_calls != 0 or injection_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 route injection invoked protected work")

    invalid_input = _build_c2_coordinator(model, accepted, neural_profile)
    wrong_shape = torch.zeros((1, 3, 31, 32), dtype=torch.uint8)
    if invalid_input.handle_public(wrong_shape) != {"version": 5, "status": "deny"}:
        raise V1M1C2EvaluatorError("C2 invalid public input did not deny")
    invalid_commitment = c1._build_conformance_commitment(
        neural_profile.public_profile,
        c1._conformance_response(10),
    )
    if invalid_input.begin_protected(wrong_shape, invalid_commitment) != {
        "version": 5,
        "status": "deny",
    }:
        raise V1M1C2EvaluatorError("C2 invalid protected input did not deny")
    invalid_input_snapshot = invalid_input.snapshot()
    if invalid_input_snapshot.verifier_calls != 0 or invalid_input_snapshot.protected_calls != 0:
        raise V1M1C2EvaluatorError("C2 invalid input invoked protected work")

    concurrent = _build_c2_coordinator(model, accepted, neural_profile)
    _commitment, response = _issue_response(concurrent, image, neural_profile, 11)
    with ThreadPoolExecutor(max_workers=8) as executor:
        concurrent_results = tuple(executor.map(concurrent.respond_protected, (response,) * 32))
    concurrent_snapshot = concurrent.snapshot()
    if (
        sum(result.get("status") == "protected" for result in concurrent_results) != 1
        or concurrent_snapshot.verifier_calls != 1
        or concurrent_snapshot.protected_calls != 1
    ):
        raise V1M1C2EvaluatorError("C2 concurrent duplicate response isolation changed")

    return {
        "canonical_relation_tamper": {
            "verifier": tamper_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "malformed_response": {
            "verifier": malformed_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "replay": {
            "verifier": after_replay.verifier_calls - before_replay.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "initial_protected_calls": before_replay.protected_calls,
            "replay_additional_protected_calls": (
                after_replay.protected_calls - before_replay.protected_calls
            ),
            "external_status": "deny",
        },
        "expiry": {
            "verifier": expiry_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "abort_retry_exhaustion": {
            "abort_count": abort_snapshot.aborts,
            "verifier": abort_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "route_confusion": {
            "verifier": target_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "cross_input_transcript_confusion": {
            "verifier": cross_input_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "route_field_injection": {
            "verifier": injection_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "invalid_input": {
            "verifier": invalid_input_snapshot.verifier_calls,
            "prefix": 0,
            "public_head": 0,
            "protected_suffix": 0,
            "external_status": "deny",
        },
        "concurrent_duplicate_response": {
            "attempts": len(concurrent_results),
            "protected_responses": 1,
            "deny_responses": 31,
            "verifier": concurrent_snapshot.verifier_calls,
            "prefix": 1,
            "public_head": 0,
            "protected_suffix": concurrent_snapshot.protected_calls,
        },
    }


def _failure_probe(
    model: V1Cifar100ResNet18,
    accepted: V1M1C2AcceptedPublicHead,
    image: Tensor,
    neural_profile: V1NeuralProfile,
    stage: str,
) -> dict[str, object]:
    events: list[str] = []
    coordinator = _build_c2_coordinator(model, accepted, neural_profile, events=events)
    counts = {"prefix": 0, "public_head": 0, "protected_suffix": 0}

    def count_prefix(_module: nn.Module, _inputs: object) -> None:
        counts["prefix"] += 1

    def count_public_head(_module: nn.Module, _inputs: object) -> None:
        counts["public_head"] += 1

    def count_protected_suffix(_module: nn.Module, _inputs: object) -> None:
        counts["protected_suffix"] += 1

    count_handles = (
        _prefix_terminal_module(model, accepted.cut).register_forward_pre_hook(count_prefix),
        accepted.head.register_forward_pre_hook(count_public_head),
        model.classifier.register_forward_pre_hook(count_protected_suffix),
    )
    try:
        if stage == "public_head":

            def fail_public(_module: nn.Module, _inputs: object) -> None:
                raise RuntimeError("synthetic public-head failure")

            failure_handle = accepted.head.register_forward_pre_hook(fail_public)
            try:
                result = coordinator.handle_public(image)
            finally:
                failure_handle.remove()
            expected_event = "public_execution_error"
        else:
            response_polynomials = c1._conformance_response(
                {"preprocessing": 10, "prefix": 11, "suffix": 12, "extraction": 13}[stage]
            )
            commitment = c1._build_conformance_commitment(
                neural_profile.public_profile,
                response_polynomials,
            )
            issued = coordinator.begin_protected(image, commitment)
            response = c1._response_for_issue(cast(dict[str, object], issued), response_polynomials)
            if stage == "preprocessing":
                with patch(
                    "can.access.v1_m1_c2._preprocess_v1_m1_snapshot",
                    side_effect=RuntimeError("synthetic preprocessing failure"),
                ):
                    result = coordinator.respond_protected(response)
                expected_event = "protected_execution_error:preprocess"
            elif stage == "prefix":

                def fail_prefix(_module: nn.Module, _inputs: object) -> None:
                    raise RuntimeError("synthetic prefix failure")

                failure_handle = _prefix_terminal_module(
                    model, accepted.cut
                ).register_forward_pre_hook(fail_prefix)
                try:
                    result = coordinator.respond_protected(response)
                finally:
                    failure_handle.remove()
                expected_event = "protected_execution_error:prefix"
            elif stage == "suffix":

                def fail_suffix(_module: nn.Module, _inputs: object) -> None:
                    raise RuntimeError("synthetic suffix failure")

                failure_handle = model.classifier.register_forward_pre_hook(fail_suffix)
                try:
                    result = coordinator.respond_protected(response)
                finally:
                    failure_handle.remove()
                expected_event = "protected_execution_error:suffix"
            else:

                def invalidate_logits(
                    _module: nn.Module,
                    _inputs: object,
                    output: object,
                ) -> Tensor:
                    if type(output) is not Tensor:
                        raise RuntimeError("synthetic extraction probe saw non-tensor")
                    return output[:, :99].contiguous()

                failure_handle = model.classifier.register_forward_hook(invalidate_logits)
                try:
                    result = coordinator.respond_protected(response)
                finally:
                    failure_handle.remove()
                expected_event = "protected_result_extraction_error"
    finally:
        for count_handle in count_handles:
            count_handle.remove()
    if result != {"version": 5, "status": "deny"} or expected_event not in events:
        raise V1M1C2EvaluatorError(f"C2 {stage} failure probe changed")
    snapshot = coordinator.snapshot()
    expected_counts = {
        "public_head": {"prefix": 1, "public_head": 1, "protected_suffix": 0},
        "preprocessing": {"prefix": 0, "public_head": 0, "protected_suffix": 0},
        "prefix": {"prefix": 1, "public_head": 0, "protected_suffix": 0},
        "suffix": {"prefix": 1, "public_head": 0, "protected_suffix": 1},
        "extraction": {"prefix": 1, "public_head": 0, "protected_suffix": 1},
    }[stage]
    expected_verifier = 0 if stage == "public_head" else 1
    expected_protected = 0 if stage == "public_head" else 1
    if (
        counts != expected_counts
        or snapshot.verifier_calls != expected_verifier
        or snapshot.protected_calls != expected_protected
    ):
        raise V1M1C2EvaluatorError(f"C2 {stage} failure call accounting changed")
    return {
        "external_status": "deny",
        "verifier": snapshot.verifier_calls,
        "protected_operation_started": snapshot.protected_calls,
        **counts,
        "events": events,
    }


def _probe_execution_failures(
    model: V1Cifar100ResNet18,
    accepted: V1M1C2AcceptedPublicHead,
    image: Tensor,
    neural_profile: V1NeuralProfile,
) -> dict[str, object]:
    return {
        stage: _failure_probe(model, accepted, image, neural_profile, stage)
        for stage in ("public_head", "preprocessing", "prefix", "suffix", "extraction")
    }


class _TimingEvents:
    __slots__ = ("_device", "values")

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self.values: list[tuple[str, int]] = []

    def reset(self) -> None:
        self.values.clear()

    def __call__(self, event: str) -> None:
        torch.cuda.synchronize(self._device)
        self.values.append((event, time.perf_counter_ns()))

    def timestamp(self, event: str) -> int:
        matches = [timestamp for name, timestamp in self.values if name == event]
        if len(matches) != 1:
            raise V1M1C2EvaluatorError(f"C2 timing event is not unique: {event}")
        return matches[0]


def _measure_operation(
    device: torch.device, operation: Callable[[], object]
) -> tuple[int, int, object]:
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize(device)
    finished = time.perf_counter_ns()
    return started, finished, result


def _top1_only(logits: Tensor) -> int:
    return int(logits.argmax(dim=1).item())


def _public_envelope_only() -> dict[str, object]:
    return {"version": 5, "status": "public", "coarse_class_id": 0}


def _protected_envelope_only() -> dict[str, object]:
    return {"version": 5, "status": "protected", "class_id": 0}


def _measure_latency(
    model: V1Cifar100ResNet18,
    accepted: V1M1C2AcceptedPublicHead,
    device: torch.device,
    test_pixels: Tensor,
    neural_profile: V1NeuralProfile,
) -> dict[str, object]:
    timing_events = _TimingEvents(device)
    accepted_coordinator = _build_c2_coordinator(
        model,
        accepted,
        neural_profile,
        events=timing_events,
    )
    rejected_coordinator = _build_c2_coordinator(model, accepted, neural_profile)

    def public_request(index: int) -> None:
        result = accepted_coordinator.handle_public(
            test_pixels[
                index % baseline.V1_M1_TEST_SIZE : index % baseline.V1_M1_TEST_SIZE + 1
            ].contiguous()
        )
        if result.get("status") != "public":
            raise V1M1C2EvaluatorError("C2 latency public request did not succeed")

    def protected_material(index: int) -> tuple[Tensor, bytes, bytes]:
        normalized_index = index % baseline.V1_M1_TEST_SIZE
        image = test_pixels[normalized_index : normalized_index + 1].contiguous()
        response_polynomials = c1._conformance_response(index)
        commitment = c1._build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        issued = accepted_coordinator.begin_protected(image, commitment)
        response = c1._response_for_issue(cast(dict[str, object], issued), response_polynomials)
        return image, commitment, response

    def rejected_material(index: int) -> bytes:
        normalized_index = index % baseline.V1_M1_TEST_SIZE
        image = test_pixels[normalized_index : normalized_index + 1].contiguous()
        response_polynomials = c1._conformance_response(index)
        commitment = c1._build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        issued = rejected_coordinator.begin_protected(image, commitment)
        response = c1._response_for_issue(cast(dict[str, object], issued), response_polynomials)
        return c1._tampered_response(response)

    for index in range(V1_M1_C2_LATENCY_WARMUPS):
        public_request(index)
        _image, _commitment, response = protected_material(index + 20_000)
        if accepted_coordinator.respond_protected(response).get("status") != "protected":
            raise V1M1C2EvaluatorError("C2 latency protected warmup did not succeed")
        if rejected_coordinator.respond_protected(rejected_material(index + 30_000)) != {
            "version": 5,
            "status": "deny",
        }:
            raise V1M1C2EvaluatorError("C2 latency reject warmup did not deny")

    samples: dict[str, list[int]] = {
        "credential_generation": [],
        "commitment_challenge_state": [],
        "response_encoding": [],
        "neural_verification": [],
        "public_preprocessing": [],
        "public_prefix": [],
        "public_head_and_top1": [],
        "public_top1_extraction_isolated": [],
        "public_response_construction_isolated": [],
        "public_response_release": [],
        "protected_verification_and_precommit": [],
        "protected_coordinator_commit": [],
        "protected_preprocessing": [],
        "protected_prefix": [],
        "protected_suffix": [],
        "protected_top1_and_response_release": [],
        "protected_top1_extraction_isolated": [],
        "protected_response_construction_isolated": [],
        "public_end_to_end": [],
        "protected_end_to_end": [],
        "rejected_end_to_end": [],
    }
    fixed_commitment = c1._build_conformance_commitment(
        neural_profile.public_profile,
        c1.V1_M1_C1_RESPONSE,
    )
    fixed_challenge = V1Challenge(V1_PROFILE_ID, c1.V1_M1_C1_CHALLENGE).encode()
    fixed_response = V1Response(bytes(32), c1.V1_M1_C1_RESPONSE).encode()
    fixed_public_logits = torch.zeros((1, 20), dtype=torch.float32, device=device)
    fixed_protected_logits = torch.zeros((1, 100), dtype=torch.float32, device=device)
    for index in range(V1_M1_C2_LATENCY_OBSERVATIONS):
        image = test_pixels[index : index + 1].contiguous()
        response_polynomials = c1._conformance_response(index + 40_000)
        samples["credential_generation"].append(
            c1._measure_ns(
                device,
                partial(
                    c1._build_conformance_commitment,
                    neural_profile.public_profile,
                    response_polynomials,
                ),
            )
        )
        commitment = c1._build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        samples["commitment_challenge_state"].append(
            c1._measure_ns(
                device,
                partial(accepted_coordinator.begin_protected, image, commitment),
            )
        )
        issued = accepted_coordinator.begin_protected(
            image,
            c1._build_conformance_commitment(
                neural_profile.public_profile,
                c1._conformance_response(index + 50_000),
            ),
        )
        transcript_id = issued.get("transcript_id")
        if type(transcript_id) is not bytes:
            raise V1M1C2EvaluatorError("C2 latency response encoding lacks transcript")
        samples["response_encoding"].append(
            c1._measure_ns(
                device,
                V1Response(transcript_id, response_polynomials).encode,
            )
        )
        samples["neural_verification"].append(
            c1._measure_ns(
                device,
                lambda: verify_v1_neural(
                    fixed_commitment,
                    fixed_challenge,
                    fixed_response,
                    bytes(32),
                    neural_profile,
                ),
            )
        )
        samples["public_top1_extraction_isolated"].append(
            c1._measure_ns(device, partial(_top1_only, fixed_public_logits))
        )
        samples["public_response_construction_isolated"].append(
            c1._measure_ns(device, _public_envelope_only)
        )
        samples["protected_top1_extraction_isolated"].append(
            c1._measure_ns(device, partial(_top1_only, fixed_protected_logits))
        )
        samples["protected_response_construction_isolated"].append(
            c1._measure_ns(device, _protected_envelope_only)
        )

        timing_events.reset()
        public_started, public_finished, public_result = _measure_operation(
            device,
            partial(accepted_coordinator.handle_public, image),
        )
        if cast(dict[str, object], public_result).get("status") != "public":
            raise V1M1C2EvaluatorError("C2 timed public request did not succeed")
        preprocess_start = timing_events.timestamp("preprocess_start")
        prefix_start = timing_events.timestamp("prefix_start")
        head_start = timing_events.timestamp("public_head_start")
        release = timing_events.timestamp("response_release")
        samples["public_preprocessing"].append(prefix_start - preprocess_start)
        samples["public_prefix"].append(head_start - prefix_start)
        samples["public_head_and_top1"].append(release - head_start)
        samples["public_response_release"].append(public_finished - release)
        samples["public_end_to_end"].append(public_finished - public_started)

        protected_response_polynomials = c1._conformance_response(index + 60_000)
        protected_commitment = c1._build_conformance_commitment(
            neural_profile.public_profile,
            protected_response_polynomials,
        )
        protected_issued = accepted_coordinator.begin_protected(image, protected_commitment)
        protected_response = c1._response_for_issue(
            cast(dict[str, object], protected_issued),
            protected_response_polynomials,
        )
        timing_events.reset()
        protected_started, protected_finished, protected_result = _measure_operation(
            device,
            partial(accepted_coordinator.respond_protected, protected_response),
        )
        if cast(dict[str, object], protected_result).get("status") != "protected":
            raise V1M1C2EvaluatorError("C2 timed protected request did not succeed")
        verifier_accept = timing_events.timestamp("verifier_accept")
        coordinator_commit = timing_events.timestamp("coordinator_commit(PROTECTED)")
        preprocess_start = timing_events.timestamp("preprocess_start")
        prefix_start = timing_events.timestamp("prefix_start")
        suffix_start = timing_events.timestamp("suffix_start")
        internal_commit = timing_events.timestamp("internal_result_commit")
        release = timing_events.timestamp("response_release")
        samples["protected_verification_and_precommit"].append(verifier_accept - protected_started)
        samples["protected_coordinator_commit"].append(coordinator_commit - verifier_accept)
        samples["protected_preprocessing"].append(prefix_start - preprocess_start)
        samples["protected_prefix"].append(suffix_start - prefix_start)
        samples["protected_suffix"].append(internal_commit - suffix_start)
        samples["protected_top1_and_response_release"].append(protected_finished - internal_commit)
        samples["protected_end_to_end"].append(protected_finished - protected_started)

        rejected_response = rejected_material(index + 70_000)
        rejected_started, rejected_finished, rejected_result = _measure_operation(
            device,
            partial(rejected_coordinator.respond_protected, rejected_response),
        )
        if rejected_result != {"version": 5, "status": "deny"}:
            raise V1M1C2EvaluatorError("C2 timed rejected request did not deny")
        samples["rejected_end_to_end"].append(rejected_finished - rejected_started)

    summaries = {name: c1._timing_summary(values) for name, values in samples.items()}
    verification_median = cast(int, summaries["neural_verification"]["median_ns"])
    combined_median = cast(
        int,
        summaries["protected_verification_and_precommit"]["median_ns"],
    )
    return {
        "method": "100 warmups, 1000 serialized observations, CUDA synchronize at segment events",
        "segments": summaries,
        "verification_precommit_residual_estimate_median_ns": max(
            0,
            combined_median - verification_median,
        ),
        "throughput_per_second": {
            name: V1_M1_C2_LATENCY_OBSERVATIONS * 1_000_000_000 / sum(samples[name])
            for name in ("public_end_to_end", "protected_end_to_end", "rejected_end_to_end")
        },
    }


def prepare_v1_m1_c2_accepted_artifact(
    artifact_root: Path = training.V1_M1_C2_ARTIFACT_ROOT,
) -> Path | None:
    """验证 artifact; 仅对已知旧 runner 追加一次 metadata correction。"""
    manifest_path = artifact_root / training.V1_M1_C2_MANIFEST_FILENAME
    state_path = artifact_root / training.V1_M1_C2_STATE_FILENAME
    manifest = _read_json_object(manifest_path, "C2 manifest")
    _cut, _run, _decoded, _coarse, _state, _file, legacy_bug = _validate_c2_manifest(
        manifest,
        state_path,
    )
    if not legacy_bug:
        return None
    correction_path = artifact_root / V1_M1_C2_CORRECTION_FILENAME
    if correction_path.exists() and not correction_path.is_symlink():
        return correction_path
    return materialize_v1_m1_c2_metadata_correction(artifact_root)


def _accepted_report_path(artifact_root: Path) -> Path:
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise V1M1C2EvaluatorError("C2 artifact root is missing or symlinked")
    report_path = artifact_root / V1_M1_C2_ACCEPTED_REPORT_FILENAME
    if report_path.exists() or report_path.is_symlink():
        raise V1M1C2EvaluatorError("refusing to overwrite an existing C2 accepted-state report")
    return report_path


def run_v1_m1_c2_evaluator(
    data_root: Path = baseline.V1_M1_DATA_ROOT,
    accepted_r2_root: Path = baseline.V1_M1_ARTIFACT_ROOT,
    c2_artifact_root: Path = training.V1_M1_C2_ARTIFACT_ROOT,
    device: torch.device = _V1_M1_C2_DEVICE,
) -> Path:
    """执行一次无训练 C2 accepted-state evaluator, 写入 ignored report。"""
    try:
        c1._validate_frozen_server_environment(device)
        report_path = _accepted_report_path(c2_artifact_root)
        accepted_r2 = c1.load_v1_m1_c1_accepted_r2_details(accepted_r2_root, device)
    except c1.V1M1C1EvaluatorError as error:
        raise V1M1C2EvaluatorError(str(error)) from error
    accepted = load_v1_m1_c2_accepted_public_head(c2_artifact_root, device)
    if accepted.decoded_data_sha256 != accepted_r2.decoded_data_sha256:
        raise V1M1C2EvaluatorError("C2 public head and accepted R2 bind different data")
    data = _load_c2_test_data(
        data_root,
        accepted.decoded_data_sha256,
        accepted.coarse_labels_sha256,
    )
    _public_profile, neural_profile, _commitment = c1._build_public_conformance_material()
    routes = _evaluate_routes(accepted_r2.model, accepted, device, data, neural_profile)
    fail_closed = _probe_fail_closed(
        accepted_r2.model,
        accepted,
        data.pixels[0:1].contiguous(),
        neural_profile,
    )
    execution_failures = _probe_execution_failures(
        accepted_r2.model,
        accepted,
        data.pixels[1:2].contiguous(),
        neural_profile,
    )
    latency = _measure_latency(
        accepted_r2.model,
        accepted,
        device,
        data.pixels,
        neural_profile,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": V1_M1_C2_EVALUATOR_EXPERIMENT_ID,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
        "accepted_r2": {
            "run_index": c1.V1_M1_C1_RUN_INDEX,
            "selected_epoch": c1.V1_M1_C1_ACCEPTED_SELECTED_EPOCH,
            "canonical_state_sha256": accepted_r2.canonical_state_sha256,
            "baseline_predictions_sha256": c1.V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256,
        },
        "public_head_artifact": {
            "cut": accepted.cut.value,
            "accepted_run": accepted.accepted_run,
            "canonical_state_sha256": accepted.state_sha256,
            "state_file_sha256": accepted.state_file_sha256,
            "manifest_sha256": accepted.manifest_sha256,
            "training_report_sha256": accepted.report_sha256,
            "metadata_correction_sha256": accepted.metadata_correction_sha256,
            "contains_only_public_head_state": True,
        },
        "artifact_validation": {
            "accepted_r2_exact_model_and_state": True,
            "input_profile_binding": V1_M1_INPUT_PROFILE_SHA256.hex(),
            "decoded_data_binding": accepted.decoded_data_sha256,
            "coarse_labels_binding": accepted.coarse_labels_sha256,
            "public_head_exact_state_entries": [
                "classifier.bias",
                "classifier.weight",
            ],
            "request_selectable_cut_or_policy": False,
            "unmet_conditions": [],
        },
        "training_selection": {
            "h1_candidates": accepted.training_report["h1_candidates"],
            "h2": accepted.training_report["h2"],
            "accepted": accepted.training_report["accepted"],
            "test": accepted.training_report["test"],
        },
        "routes": routes,
        "fail_closed": fail_closed,
        "execution_failures": execution_failures,
        "latency": latency,
        "schemas": {
            "public": ["version", "status", "coarse_class_id"],
            "protected": ["version", "status", "class_id"],
            "deny": ["version", "status"],
        },
        "credential": {
            "kind": "public V1-P2 conformance fixture; no secret is generated, loaded, or reported",
            "challenge_weight": 2,
        },
        "scope": (
            "No R2/public-head training, fine-tuning, state mutation, data download, "
            "or artifact publication. Black-box trusted-entry evaluation only."
        ),
    }
    _atomic_write_json(report_path, report)
    return report_path


def main() -> None:
    """准备旧 artifact 元数据并运行固定 C2 accepted-state evaluator。"""
    prepare_v1_m1_c2_accepted_artifact()
    report_path = run_v1_m1_c2_evaluator()
    print(f"C2 accepted-state report written: {report_path}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "V1_M1_C2_ACCEPTED_REPORT_FILENAME",
    "V1_M1_C2_CORRECTION_EXPERIMENT_ID",
    "V1_M1_C2_CORRECTION_FILENAME",
    "V1_M1_C2_EVALUATOR_EXPERIMENT_ID",
    "V1M1C2AcceptedPublicHead",
    "V1M1C2EvaluatorError",
    "load_v1_m1_c2_accepted_public_head",
    "materialize_v1_m1_c2_metadata_correction",
    "prepare_v1_m1_c2_accepted_artifact",
    "run_v1_m1_c2_evaluator",
]
