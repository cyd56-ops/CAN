"""V1-M1-C1 已接受 R2 的无训练 Gate Layer 验收与性能报告。"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import struct
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, cast

import torch
import torchvision  # type: ignore[import-untyped]
from torch import Tensor

from can.access import (
    A3_V2_CHALLENGE_TTL_MS,
    A3V2Clock,
    A3V2ProtocolCoordinator,
    A3V2TranscriptStore,
    AuthenticatedR2,
    V1M1AccessCoordinator,
    V1M1InputAdapter,
    build_v1_a3_v2_neural_profile,
    normalize_v1_m1_uint8_batch,
)
from can.experiments import v1_m1_baseline as baseline
from can.model import (
    V1_M1_CLASS_COUNT,
    V1_M1_MODEL_PROFILE_ID,
    V1_M1_PARAMETER_COUNT,
    V1Cifar100ResNet18,
)
from can.reference import (
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RING_DEGREE,
    V1Abort,
    V1Challenge,
    V1Commitment,
    V1PublicProfile,
    V1Response,
    build_v1_conformance_profile,
    parse_v1_response,
    v1_negacyclic_convolution,
)
from can.verifier import V1NeuralProfile, compile_v1_neural_profile, verify_v1_neural

V1_M1_C1_EXPERIMENT_ID: Final = "CAN-V1-M1-C1-ACCEPTED-R2-v1"
V1_M1_C1_RUN_INDEX: Final = 2
V1_M1_C1_REPORT_DIRECTORY: Final = "c1"
V1_M1_C1_REPORT_FILENAME: Final = "accepted-r2-report.json"
V1_M1_C1_ACCEPTED_STATE_SHA256: Final = (
    "c0733e293c398f58edd3ae6c6cb5c9c217572274b095cb9c4ace282f5c101343"
)
V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256: Final = (
    "08ab99b57698fe5ccc45d85b0916b03c211ca5ccd1d080251ddd08d38d097131"
)
V1_M1_C1_ACCEPTED_SELECTED_EPOCH: Final = 175
V1_M1_C1_LATENCY_WARMUPS: Final = 100
V1_M1_C1_LATENCY_OBSERVATIONS: Final = 1_000
V1_M1_C1_ABORT_ATTEMPTS: Final = 3
V1_M1_C1_IDENTITY: Final = bytes(range(32))
V1_M1_C1_CHALLENGE: Final = (1, 0, 0, 0, 0, 0, 0, -1)
V1_M1_C1_RESPONSE: Final = (
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0),
    (-1, 0, 1, 0, 0, 0, 0, 0),
    (0, -1, 0, 1, 0, 0, 0, 0),
)
_V1_M1_C1_MAX_JSON_BYTES: Final = 1_000_000
_V1_M1_C1_DEVICE: Final = torch.device("cuda:0")


class V1M1C1EvaluatorError(RuntimeError):
    """表示 C1 accepted-state evaluator 不满足固定实验契约。"""


@dataclass(frozen=True, slots=True)
class V1M1C1AcceptedR2:
    """保存已核验 R2 及彼此独立的数据和模型摘要。"""

    model: V1Cifar100ResNet18
    decoded_data_sha256: str
    canonical_state_sha256: str


class _CounterNonce:
    """为单进程 evaluator 生成不可复用的固定宽度 nonce。"""

    __slots__ = ("_counter",)

    def __init__(self) -> None:
        self._counter = 0

    def __call__(self, size: int) -> bytes:
        self._counter += 1
        return self._counter.to_bytes(size, byteorder="big", signed=False)


class _MutableClock:
    """只供 expiry probe 使用的可信时钟替身。"""

    __slots__ = ("monotonic_ns", "wall_ms")

    def __init__(self) -> None:
        self.wall_ms = 1_700_000_000_000
        self.monotonic_ns = 5_000_000_000


class _LogitCapture:
    """在实验外部捕获一次冻结 R2 forward 的 logits。"""

    __slots__ = ("count", "enabled", "value")

    def __init__(self) -> None:
        self.count = 0
        self.enabled = False
        self.value: Tensor | None = None

    def __call__(self, _module: object, _inputs: object, output: object) -> None:
        if not self.enabled:
            return
        if type(output) is not Tensor or self.value is not None:
            raise V1M1C1EvaluatorError("R2 forward hook did not observe exactly one tensor")
        self.count += 1
        self.value = output.detach().cpu().contiguous().clone()

    def take(self) -> Tensor:
        """取走当前一次 forward 的 CPU logits 快照。"""
        if self.value is None:
            raise V1M1C1EvaluatorError("Gate Layer did not invoke R2 for an accepted request")
        value = self.value
        self.value = None
        return value


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise V1M1C1EvaluatorError("accepted artifact cannot be read") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect_exact_keys(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys or not all(type(key) is str for key in value):
        raise V1M1C1EvaluatorError(f"{name} fields are not canonical")
    return cast(dict[str, object], value)


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise V1M1C1EvaluatorError(f"{name} is missing or symlinked")
    try:
        if path.stat().st_size < 1 or path.stat().st_size > _V1_M1_C1_MAX_JSON_BYTES:
            raise V1M1C1EvaluatorError(f"{name} has an invalid byte size")
        payload = path.read_bytes()
        value: object = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except V1M1C1EvaluatorError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V1M1C1EvaluatorError(f"{name} is not canonical JSON") from error
    if type(value) is not dict:
        raise V1M1C1EvaluatorError(f"{name} root must be an exact object")
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


def _validate_evaluation_report(value: object, name: str) -> None:
    report = _expect_exact_keys(
        value,
        {
            "loss_hex",
            "top1_percent_hex",
            "top5_percent_hex",
            "correct_top1",
            "correct_top5",
            "total",
            "predictions_sha256",
        },
        name,
    )
    if (
        not all(
            type(report[field]) is str
            for field in ("loss_hex", "top1_percent_hex", "top5_percent_hex")
        )
        or type(report["correct_top1"]) is not int
        or type(report["correct_top5"]) is not int
        or type(report["total"]) is not int
        or not _is_sha256(report["predictions_sha256"])
    ):
        raise V1M1C1EvaluatorError(f"{name} values are not canonical")


def _validate_baseline_manifest(
    manifest: dict[str, object],
    state_path: Path,
) -> tuple[str, str]:
    parsed = _expect_exact_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "run_index",
            "data",
            "training",
            "model",
            "selection",
            "state",
        },
        "accepted manifest",
    )
    if (
        parsed["schema_version"] != 1
        or parsed["experiment_id"] != baseline.V1_M1_EXPERIMENT_ID
        or parsed["run_index"] != V1_M1_C1_RUN_INDEX
    ):
        raise V1M1C1EvaluatorError("accepted manifest identity changed")

    data = _expect_exact_keys(
        parsed["data"],
        {"archive", "decoded_sha256", "train_size", "validation_size", "test_size"},
        "accepted manifest data",
    )
    archive = _expect_exact_keys(
        data["archive"], {"filename", "byte_size", "sha256", "md5"}, "archive"
    )
    if (
        archive["filename"] != baseline.V1_M1_ARCHIVE_FILENAME
        or archive["byte_size"] != baseline.V1_M1_ARCHIVE_SIZE
        or archive["sha256"] != baseline.V1_M1_ARCHIVE_SHA256
        or archive["md5"] != baseline.V1_M1_ARCHIVE_MD5
        or not _is_sha256(data["decoded_sha256"])
        or data["train_size"] != baseline.V1_M1_TRAIN_SIZE
        or data["validation_size"] != baseline.V1_M1_VALIDATION_SIZE
        or data["test_size"] != baseline.V1_M1_TEST_SIZE
    ):
        raise V1M1C1EvaluatorError("accepted manifest data contract changed")

    expected_training = baseline.V1M1TrainingConfig(
        run_index=V1_M1_C1_RUN_INDEX,
        seed=baseline.V1_M1_RUN_SEEDS[V1_M1_C1_RUN_INDEX - 1],
    )
    if parsed["training"] != {
        "run_index": expected_training.run_index,
        "seed": expected_training.seed,
        "train_batch_size": expected_training.train_batch_size,
        "evaluation_batch_size": expected_training.evaluation_batch_size,
        "worker_count": expected_training.worker_count,
        "epoch_count": expected_training.epoch_count,
        "learning_rate": expected_training.learning_rate,
        "momentum": expected_training.momentum,
        "weight_decay": expected_training.weight_decay,
        "nesterov": expected_training.nesterov,
        "pin_memory": expected_training.pin_memory,
        "persistent_workers": expected_training.persistent_workers,
        "prefetch_factor": expected_training.prefetch_factor,
    }:
        raise V1M1C1EvaluatorError("accepted manifest training contract changed")
    model = _expect_exact_keys(
        parsed["model"],
        {"profile_id", "topology", "parameter_count", "float32_parameter_bytes"},
        "accepted manifest model",
    )
    if model != {
        "profile_id": V1_M1_MODEL_PROFILE_ID,
        "topology": "CIFAR-style ResNet-18 [2,2,2,2]",
        "parameter_count": V1_M1_PARAMETER_COUNT,
        "float32_parameter_bytes": V1_M1_PARAMETER_COUNT * 4,
    }:
        raise V1M1C1EvaluatorError("accepted manifest model contract changed")
    selection = _expect_exact_keys(parsed["selection"], {"selected_epoch", "rule"}, "selection")
    if selection != {
        "selected_epoch": V1_M1_C1_ACCEPTED_SELECTED_EPOCH,
        "rule": "strictly higher validation top-1; retain earlier epoch on a tie",
    }:
        raise V1M1C1EvaluatorError("accepted manifest selection changed")

    state = _expect_exact_keys(
        parsed["state"],
        {
            "filename",
            "state_dict_only",
            "optimizer_state_saved",
            "byte_size",
            "file_sha256",
            "canonical_state_sha256",
        },
        "accepted manifest state",
    )
    if (
        state["filename"] != baseline.V1_M1_STATE_FILENAME
        or state["state_dict_only"] is not True
        or state["optimizer_state_saved"] is not False
        or type(state["byte_size"]) is not int
        or not 1 <= state["byte_size"] <= baseline.V1_M1_MAX_STATE_BYTES
        or not _is_sha256(state["file_sha256"])
        or state["canonical_state_sha256"] != V1_M1_C1_ACCEPTED_STATE_SHA256
    ):
        raise V1M1C1EvaluatorError("accepted manifest state contract changed")
    if state_path.stat().st_size != state["byte_size"]:
        raise V1M1C1EvaluatorError("accepted state byte size changed")
    return cast(str, data["decoded_sha256"]), cast(str, state["file_sha256"])


def _validate_baseline_report(
    report: dict[str, object],
    manifest_path: Path,
) -> None:
    parsed = _expect_exact_keys(
        report,
        {
            "schema_version",
            "experiment_id",
            "run_index",
            "environment",
            "manifest_filename",
            "manifest_sha256",
            "epochs",
            "test",
            "selected_epoch",
            "state_sha256",
        },
        "accepted baseline report",
    )
    if (
        parsed["schema_version"] != 1
        or parsed["experiment_id"] != baseline.V1_M1_EXPERIMENT_ID
        or parsed["run_index"] != V1_M1_C1_RUN_INDEX
        or parsed["manifest_filename"] != baseline.V1_M1_MANIFEST_FILENAME
        or parsed["manifest_sha256"] != _file_digest(manifest_path)
        or parsed["selected_epoch"] != V1_M1_C1_ACCEPTED_SELECTED_EPOCH
        or parsed["state_sha256"] != V1_M1_C1_ACCEPTED_STATE_SHA256
    ):
        raise V1M1C1EvaluatorError("accepted baseline report binding changed")
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
        "accepted baseline environment",
    )
    if (
        not all(
            type(environment[field]) is str
            for field in ("platform", "python", "torch", "device", "device_name")
        )
        or (
            environment["cuda_runtime"] is not None and type(environment["cuda_runtime"]) is not str
        )
        or (
            environment["python_hash_seed"] is not None
            and type(environment["python_hash_seed"]) is not str
        )
        or (
            environment["cublas_workspace_config"] is not None
            and type(environment["cublas_workspace_config"]) is not str
        )
        or not all(
            type(environment[field]) is bool
            for field in (
                "deterministic_algorithms",
                "cudnn_benchmark",
                "cudnn_deterministic",
                "cuda_matmul_allow_tf32",
                "cudnn_allow_tf32",
            )
        )
    ):
        raise V1M1C1EvaluatorError("accepted baseline environment is not canonical")
    epochs = parsed["epochs"]
    if type(epochs) is not list or len(epochs) != baseline.V1_M1_EPOCH_COUNT:
        raise V1M1C1EvaluatorError("accepted baseline report epoch count changed")
    for expected_epoch, value in enumerate(epochs, start=1):
        epoch = _expect_exact_keys(value, {"epoch", "training_loss_hex", "validation"}, "epoch")
        if epoch["epoch"] != expected_epoch or type(epoch["training_loss_hex"]) is not str:
            raise V1M1C1EvaluatorError("accepted baseline report epoch changed")
        _validate_evaluation_report(epoch["validation"], "epoch validation")
    _validate_evaluation_report(parsed["test"], "accepted test")
    test = cast(dict[str, object], parsed["test"])
    if (
        test["total"] != baseline.V1_M1_TEST_SIZE
        or test["predictions_sha256"] != V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256
    ):
        raise V1M1C1EvaluatorError("accepted baseline test reference changed")


def _validate_loaded_state(model: V1Cifar100ResNet18, value: object) -> None:
    if type(value) is not dict:
        raise V1M1C1EvaluatorError("accepted state must contain an exact dict")
    loaded = cast(dict[str, object], value)
    expected = model.state_dict()
    if tuple(loaded) != tuple(expected):
        raise V1M1C1EvaluatorError("accepted state keys changed")
    state: dict[str, Tensor] = {}
    for name, expected_tensor in expected.items():
        candidate = loaded[name]
        if type(candidate) is not Tensor:
            raise V1M1C1EvaluatorError("accepted state has a non-tensor value")
        if (
            candidate.dtype is not expected_tensor.dtype
            or candidate.device.type != "cpu"
            or candidate.device.index is not None
            or candidate.layout is not torch.strided
            or not candidate.is_contiguous()
            or tuple(candidate.shape) != tuple(expected_tensor.shape)
        ):
            raise V1M1C1EvaluatorError("accepted state tensor contract changed")
        if candidate.is_floating_point() and not bool(torch.isfinite(candidate).all().item()):
            raise V1M1C1EvaluatorError("accepted state contains non-finite values")
        state[name] = candidate
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise V1M1C1EvaluatorError("accepted state does not match R2 topology") from error
    if baseline._hash_model_state(model) != V1_M1_C1_ACCEPTED_STATE_SHA256:
        raise V1M1C1EvaluatorError("accepted R2 canonical state digest changed")
    model.eval()


def load_v1_m1_c1_accepted_r2(
    artifact_root: Path,
    device: torch.device,
) -> tuple[V1Cifar100ResNet18, str]:
    """一次性核验 accepted R2 artifact, 并将冻结模型加载到指定 CUDA 设备。"""
    accepted = load_v1_m1_c1_accepted_r2_details(artifact_root, device)
    return accepted.model, accepted.decoded_data_sha256


def load_v1_m1_c1_accepted_r2_details(
    artifact_root: Path,
    device: torch.device,
) -> V1M1C1AcceptedR2:
    """核验 accepted R2, 分别返回数据摘要与模型状态摘要。"""
    if type(device) is not torch.device or device.type != "cuda" or device.index != 0:
        raise V1M1C1EvaluatorError("C1 evaluator requires explicit cuda:0")
    if not isinstance(artifact_root, Path):
        raise V1M1C1EvaluatorError("artifact root must be pathlib.Path")
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise V1M1C1EvaluatorError("artifact root is missing or symlinked")
    run_root = artifact_root / f"run-{V1_M1_C1_RUN_INDEX}"
    manifest_path = run_root / baseline.V1_M1_MANIFEST_FILENAME
    state_path = run_root / baseline.V1_M1_STATE_FILENAME
    report_path = run_root / baseline.V1_M1_REPORT_FILENAME
    if (
        run_root.is_symlink()
        or not run_root.is_dir()
        or state_path.is_symlink()
        or not state_path.is_file()
    ):
        raise V1M1C1EvaluatorError("accepted R2 artifact is missing or symlinked")
    manifest = _read_json_object(manifest_path, "accepted manifest")
    decoded_sha256, expected_state_file_sha256 = _validate_baseline_manifest(manifest, state_path)
    if _file_digest(state_path) != expected_state_file_sha256:
        raise V1M1C1EvaluatorError("accepted state file digest changed")
    _validate_baseline_report(
        _read_json_object(report_path, "accepted baseline report"), manifest_path
    )
    model = V1Cifar100ResNet18().to(device="cpu", dtype=torch.float32)
    try:
        value = torch.load(state_path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise V1M1C1EvaluatorError("accepted R2 state cannot be loaded") from error
    _validate_loaded_state(model, value)
    return V1M1C1AcceptedR2(
        model=model.to(device).eval(),
        decoded_data_sha256=decoded_sha256,
        canonical_state_sha256=V1_M1_C1_ACCEPTED_STATE_SHA256,
    )


def _load_raw_test_split(data_root: Path, expected_decoded_sha256: str) -> tuple[Tensor, Tensor]:
    if not isinstance(data_root, Path):
        raise V1M1C1EvaluatorError("data root must be pathlib.Path")
    baseline.verify_v1_m1_archive(data_root)
    extracted_root = data_root / "cifar-100-python"
    if extracted_root.is_symlink() or not extracted_root.is_dir():
        raise V1M1C1EvaluatorError("verified CIFAR-100 archive has not been explicitly extracted")
    baseline._verify_extracted_member_bytes(data_root, extracted_root)
    train_pixels, train_labels = baseline._decode_split(
        extracted_root / "train",
        expected_size=baseline.V1_M1_TRAIN_SIZE + baseline.V1_M1_VALIDATION_SIZE,
    )
    test_pixels, test_labels = baseline._decode_split(
        extracted_root / "test",
        expected_size=baseline.V1_M1_TEST_SIZE,
    )
    baseline._decode_meta(extracted_root / "meta")
    decoded_sha256 = baseline._decoded_digest(train_pixels, train_labels, test_pixels, test_labels)
    del train_pixels, train_labels
    if decoded_sha256 != expected_decoded_sha256:
        raise V1M1C1EvaluatorError("decoded CIFAR-100 digest does not match accepted R2 manifest")
    return test_pixels, test_labels


def _conformance_response(index: int) -> tuple[tuple[int, ...], ...]:
    """从样本序号确定性构造一个有界公开 response。"""
    if type(index) is not int or index < 0:
        raise V1M1C1EvaluatorError("conformance response index is not canonical")
    if index == 0:
        return V1_M1_C1_RESPONSE
    value = index
    coefficients: list[int] = []
    for _ in range(4 * V1_RING_DEGREE):
        coefficients.append(value % 13 - 6)
        value //= 13
    return tuple(
        tuple(coefficients[offset : offset + V1_RING_DEGREE])
        for offset in range(0, 4 * V1_RING_DEGREE, V1_RING_DEGREE)
    )


def _build_conformance_commitment(
    public_profile: V1PublicProfile,
    response: tuple[tuple[int, ...], ...],
) -> bytes:
    challenge = V1Challenge(V1_PROFILE_ID, V1_M1_C1_CHALLENGE)
    commitment_rows: list[tuple[int, ...]] = []
    for row_index in range(2):
        lhs = [0] * V1_RING_DEGREE
        for column_index in range(2):
            product = v1_negacyclic_convolution(
                public_profile.matrix[row_index][column_index],
                response[column_index],
            )
            lhs = [left + right for left, right in zip(lhs, product, strict=True)]
        lhs = [
            value + identity_value
            for value, identity_value in zip(lhs, response[2 + row_index], strict=True)
        ]
        target = v1_negacyclic_convolution(challenge.coefficients, public_profile.target[row_index])
        commitment_rows.append(
            tuple(
                (value - target_value) % V1_MODULUS
                for value, target_value in zip(lhs, target, strict=True)
            )
        )
    return V1Commitment(V1_PROFILE_ID, commitment_rows).encode()


def _build_public_conformance_material() -> tuple[V1PublicProfile, V1NeuralProfile, bytes]:
    """构造公开 V1-P2 conformance 资料; 它不生成、读取或保存任何 secret。"""
    public_profile = build_v1_conformance_profile(V1_M1_C1_IDENTITY)
    commitment = _build_conformance_commitment(public_profile, _conformance_response(0))
    return public_profile, compile_v1_neural_profile(public_profile), commitment


def _fixed_challenge_sampler(_degree: int, _weight: int) -> tuple[int, ...]:
    return V1_M1_C1_CHALLENGE


def _response_for_issue(
    issued: dict[str, object],
    response: tuple[tuple[int, ...], ...],
) -> bytes:
    if issued.get("status") != "challenge" or type(issued.get("transcript_id")) is not bytes:
        raise V1M1C1EvaluatorError("C1 evaluator did not receive a challenge")
    return V1Response(cast(bytes, issued["transcript_id"]), response).encode()


def _direct_logits(model: V1Cifar100ResNet18, image: Tensor, device: torch.device) -> Tensor:
    inputs = _preprocess_and_h2d(image, device)
    return _r2_inference(model, inputs).detach().cpu().contiguous()


def _preprocess_and_h2d(image: Tensor, device: torch.device) -> Tensor:
    """测量可信 preprocessing 与 H2D copy 的组合阶段。"""
    return normalize_v1_m1_uint8_batch(image).to(device)


def _r2_inference(model: V1Cifar100ResNet18, inputs: Tensor) -> Tensor:
    """仅执行已可信预处理完成的冻结 R2 forward。"""
    with torch.inference_mode():
        return cast(Tensor, model(inputs))


def _update_logit_digest(digest: hashlib._Hash, logits: Tensor) -> int:
    if (
        logits.dtype is not torch.float32
        or logits.device.type != "cpu"
        or logits.device.index is not None
        or not logits.is_contiguous()
        or tuple(logits.shape) != (1, V1_M1_CLASS_COUNT)
    ):
        raise V1M1C1EvaluatorError("R2 logits contract changed")
    digest.update(logits.numpy().tobytes())
    return int(logits.argmax(dim=1).item())


def _update_prediction_digest(digest: hashlib._Hash, prediction: int) -> None:
    digest.update(struct.pack(">q", prediction))


def _evaluate_direct_r2_baseline_reference(
    model: V1Cifar100ResNet18,
    device: torch.device,
    test_pixels: Tensor,
) -> tuple[str, str]:
    """以原 baseline 的 batch-256 形态重新计算 direct R2 摘要。"""
    logits_digest = hashlib.sha256()
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, test_pixels.shape[0], baseline.V1_M1_EVALUATION_BATCH_SIZE):
            image_batch = test_pixels[
                start : start + baseline.V1_M1_EVALUATION_BATCH_SIZE
            ].contiguous()
            logits = _r2_inference(model, _preprocess_and_h2d(image_batch, device))
            predictions.extend(
                int(value)
                for value in torch.topk(logits, k=5, dim=1).indices[:, 0].detach().cpu().tolist()
            )
            cpu_logits = logits.detach().cpu().contiguous()
            logits_digest.update(cpu_logits.numpy().tobytes())
    return logits_digest.hexdigest(), baseline._hash_predictions(predictions)


def _evaluate_equivalence(
    model: V1Cifar100ResNet18,
    device: torch.device,
    test_pixels: Tensor,
    neural_profile: V1NeuralProfile,
) -> dict[str, object]:
    baseline_logits, baseline_predictions = _evaluate_direct_r2_baseline_reference(
        model,
        device,
        test_pixels,
    )
    if baseline_predictions != V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256:
        raise V1M1C1EvaluatorError("direct R2 predictions do not match the accepted R2 reference")
    print(
        f"C1 direct R2 reference evaluated={baseline.V1_M1_TEST_SIZE}/{baseline.V1_M1_TEST_SIZE}",
        flush=True,
    )
    direct_logits_digest = hashlib.sha256()
    direct_predictions_digest = hashlib.sha256()
    authenticated = AuthenticatedR2(
        neural_profile,
        model,
        challenge_sampler=_fixed_challenge_sampler,
    )
    captured = _LogitCapture()
    handle = model.register_forward_hook(captured)
    gated_logits_digest = hashlib.sha256()
    gated_predictions_digest = hashlib.sha256()
    try:
        for index in range(baseline.V1_M1_TEST_SIZE):
            image = test_pixels[index : index + 1].contiguous()
            response_polynomials = _conformance_response(index)
            sample_commitment = _build_conformance_commitment(
                neural_profile.public_profile,
                response_polynomials,
            )
            direct_logits = _direct_logits(model, image, device)
            _update_prediction_digest(
                direct_predictions_digest,
                _update_logit_digest(direct_logits_digest, direct_logits),
            )
            issued = authenticated.begin(image, sample_commitment)
            response = _response_for_issue(
                cast(dict[str, object], issued),
                response_polynomials,
            )
            captured.enabled = True
            try:
                if authenticated.respond(response) != {"version": 4, "status": "protected"}:
                    raise V1M1C1EvaluatorError("valid public conformance input was not protected")
            finally:
                captured.enabled = False
            gated_logits = captured.take()
            if not torch.equal(gated_logits, direct_logits):
                raise V1M1C1EvaluatorError("gated R2 logits differ from direct R2 logits")
            _update_prediction_digest(
                gated_predictions_digest,
                _update_logit_digest(gated_logits_digest, gated_logits),
            )
            if (
                index + 1
            ) % baseline.V1_M1_EVALUATION_BATCH_SIZE == 0 or index + 1 == baseline.V1_M1_TEST_SIZE:
                print(
                    f"C1 AuthenticatedR2 evaluated={index + 1}/{baseline.V1_M1_TEST_SIZE}",
                    flush=True,
                )
    finally:
        handle.remove()
    snapshot = authenticated.snapshot()
    if (
        captured.value is not None
        or captured.count != baseline.V1_M1_TEST_SIZE
        or snapshot.verifier_calls != baseline.V1_M1_TEST_SIZE
        or snapshot.allow_commits != baseline.V1_M1_TEST_SIZE
        or snapshot.protected_calls != baseline.V1_M1_TEST_SIZE
        or snapshot.protected_responses != baseline.V1_M1_TEST_SIZE
    ):
        raise V1M1C1EvaluatorError("accepted Gate Layer call accounting changed")
    direct_predictions = direct_predictions_digest.hexdigest()
    if direct_logits_digest.hexdigest() != gated_logits_digest.hexdigest():
        raise V1M1C1EvaluatorError("direct and gated logits digests differ")
    if direct_predictions != gated_predictions_digest.hexdigest():
        raise V1M1C1EvaluatorError("direct and gated prediction digests differ")
    return {
        "test_size": baseline.V1_M1_TEST_SIZE,
        "baseline_batch_size": baseline.V1_M1_EVALUATION_BATCH_SIZE,
        "baseline_direct_logits_sha256": baseline_logits,
        "baseline_direct_predictions_sha256": baseline_predictions,
        "gate_comparison_batch_size": 1,
        "direct_logits_sha256": direct_logits_digest.hexdigest(),
        "gated_logits_sha256": gated_logits_digest.hexdigest(),
        "direct_predictions_sha256": direct_predictions,
        "gated_predictions_sha256": gated_predictions_digest.hexdigest(),
        "accepted_r2_calls": snapshot.protected_calls,
        "accepted_gate_calls": snapshot.verifier_calls,
    }


def _tampered_response(response: bytes) -> bytes:
    original = parse_v1_response(response)
    parsed = V1Response(
        original.transcript_id,
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            V1_M1_C1_RESPONSE[1],
            V1_M1_C1_RESPONSE[2],
            V1_M1_C1_RESPONSE[3],
        ),
    )
    return parsed.encode()


def _probe_reject_isolation(
    model: V1Cifar100ResNet18,
    image: Tensor,
    neural_profile: V1NeuralProfile,
    commitment: bytes,
) -> dict[str, object]:
    tamper = AuthenticatedR2(neural_profile, model, challenge_sampler=_fixed_challenge_sampler)
    issued = tamper.begin(image, commitment)
    response = _response_for_issue(cast(dict[str, object], issued), _conformance_response(0))
    tamper_response = _tampered_response(response)
    if tamper.respond(tamper_response) != {"version": 4, "status": "deny"}:
        raise V1M1C1EvaluatorError("tampered C1 response did not deny")
    tamper_snapshot = tamper.snapshot()
    if tamper_snapshot.protected_calls != 0 or tamper_snapshot.verifier_calls != 1:
        raise V1M1C1EvaluatorError("tamper isolation call accounting changed")

    replay = AuthenticatedR2(neural_profile, model, challenge_sampler=_fixed_challenge_sampler)
    issued = replay.begin(image, commitment)
    response = _response_for_issue(cast(dict[str, object], issued), _conformance_response(0))
    if replay.respond(response) != {"version": 4, "status": "protected"}:
        raise V1M1C1EvaluatorError("replay probe initial request was not protected")
    before_replay = replay.snapshot()
    if replay.respond(response) != {"version": 4, "status": "deny"}:
        raise V1M1C1EvaluatorError("replayed C1 response did not deny")
    after_replay = replay.snapshot()
    if after_replay.protected_calls != before_replay.protected_calls:
        raise V1M1C1EvaluatorError("replayed response invoked R2")

    clock = _MutableClock()
    expiry = AuthenticatedR2(
        neural_profile,
        model,
        store=A3V2TranscriptStore(
            clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.monotonic_ns),
            random_bytes=_CounterNonce(),
        ),
        challenge_sampler=_fixed_challenge_sampler,
    )
    issued = expiry.begin(image, commitment)
    response = _response_for_issue(cast(dict[str, object], issued), _conformance_response(0))
    clock.monotonic_ns += A3_V2_CHALLENGE_TTL_MS * 1_000_000
    if expiry.respond(response) != {"version": 4, "status": "deny"}:
        raise V1M1C1EvaluatorError("expired C1 response did not deny")
    expiry_snapshot = expiry.snapshot()
    if expiry_snapshot.protected_calls != 0 or expiry_snapshot.expiries != 1:
        raise V1M1C1EvaluatorError("expiry isolation call accounting changed")

    abort = AuthenticatedR2(neural_profile, model, challenge_sampler=_fixed_challenge_sampler)
    for attempt in range(V1_M1_C1_ABORT_ATTEMPTS):
        response_polynomials = _conformance_response(attempt + 1)
        fresh_commitment = _build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        issued = abort.begin(image, fresh_commitment)
        transcript_id = issued.get("transcript_id")
        if issued.get("status") != "challenge" or type(transcript_id) is not bytes:
            raise V1M1C1EvaluatorError("abort probe did not receive a challenge")
        if abort.abort(V1Abort(transcript_id).encode()) != {"version": 4, "status": "deny"}:
            raise V1M1C1EvaluatorError("C1 abort did not deny")
    abort_snapshot = abort.snapshot()
    if abort_snapshot.protected_calls != 0 or abort_snapshot.aborts != V1_M1_C1_ABORT_ATTEMPTS:
        raise V1M1C1EvaluatorError("abort isolation call accounting changed")

    source = AuthenticatedR2(neural_profile, model, challenge_sampler=_fixed_challenge_sampler)
    target = AuthenticatedR2(neural_profile, model, challenge_sampler=_fixed_challenge_sampler)
    issued = source.begin(image, commitment)
    response = _response_for_issue(cast(dict[str, object], issued), _conformance_response(0))
    if target.respond(response) != {"version": 4, "status": "deny"}:
        raise V1M1C1EvaluatorError("route-confused C1 response did not deny")
    target_snapshot = target.snapshot()
    if target_snapshot.protected_calls != 0 or target_snapshot.verifier_calls != 0:
        raise V1M1C1EvaluatorError("route confusion invoked the target route")

    return {
        "tamper": {"r2_calls": 0, "gate_calls": tamper_snapshot.verifier_calls},
        "replay": {
            "initial_r2_calls": before_replay.protected_calls,
            "replay_additional_r2_calls": after_replay.protected_calls
            - before_replay.protected_calls,
        },
        "expiry": {"r2_calls": 0, "gate_calls": expiry_snapshot.verifier_calls},
        "abort_retry_exhaustion": {"r2_calls": 0, "abort_count": abort_snapshot.aborts},
        "route_confusion": {"r2_calls": 0, "gate_calls": target_snapshot.verifier_calls},
    }


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _measure_ns(device: torch.device, operation: Callable[[], object]) -> int:
    _synchronize(device)
    started = time.perf_counter_ns()
    operation()
    _synchronize(device)
    return time.perf_counter_ns() - started


def _timing_summary(samples: list[int]) -> dict[str, object]:
    if not samples:
        raise V1M1C1EvaluatorError("timing samples are empty")
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "median_ns": ordered[(len(ordered) - 1) // 2],
        "p95_ns": ordered[(len(ordered) * 95 + 99) // 100 - 1],
    }


def _measure_latency(
    model: V1Cifar100ResNet18,
    device: torch.device,
    test_pixels: Tensor,
    neural_profile: V1NeuralProfile,
    commitment: bytes,
) -> dict[str, object]:
    adapter = V1M1InputAdapter(neural_profile.identity_id)
    fixed_challenge = V1Challenge(V1_PROFILE_ID, V1_M1_C1_CHALLENGE).encode()
    neural_response = V1Response(bytes(32), V1_M1_C1_RESPONSE).encode()
    noop_route = build_v1_a3_v2_neural_profile(
        neural_profile,
        model_id=0x0001_0001,
        scope_id=1,
        input_profile_sha256=adapter.adapt(test_pixels[0:1].contiguous()).input_profile_sha256,
        protected_operation=lambda _snapshot: None,
    )
    noop_access = V1M1AccessCoordinator(
        adapter,
        A3V2ProtocolCoordinator((noop_route,), challenge_sampler=_fixed_challenge_sampler),
    )
    issue_coordinator = A3V2ProtocolCoordinator(
        (noop_route,),
        challenge_sampler=_fixed_challenge_sampler,
    )

    accepted_authenticated = AuthenticatedR2(
        neural_profile,
        model,
        challenge_sampler=_fixed_challenge_sampler,
    )
    accepted_index = [0]

    def accepted_request() -> None:
        index = accepted_index[0]
        image = test_pixels[index : index + 1].contiguous()
        accepted_index[0] = (index + 1) % baseline.V1_M1_TEST_SIZE
        response_polynomials = _conformance_response(index)
        sample_commitment = _build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        authenticated = accepted_authenticated
        response = _response_for_issue(
            cast(dict[str, object], authenticated.begin(image, sample_commitment)),
            response_polynomials,
        )
        if authenticated.respond(response) != {"version": 4, "status": "protected"}:
            raise V1M1C1EvaluatorError("latency accepted request did not protect")

    rejected_authenticated = AuthenticatedR2(
        neural_profile,
        model,
        challenge_sampler=_fixed_challenge_sampler,
    )
    rejected_index = [0]

    def rejected_request() -> None:
        index = rejected_index[0]
        image = test_pixels[index : index + 1].contiguous()
        rejected_index[0] = (index + 1) % baseline.V1_M1_TEST_SIZE
        response_polynomials = _conformance_response(index)
        sample_commitment = _build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        authenticated = rejected_authenticated
        response = _response_for_issue(
            cast(dict[str, object], authenticated.begin(image, sample_commitment)),
            response_polynomials,
        )
        if authenticated.respond(_tampered_response(response)) != {"version": 4, "status": "deny"}:
            raise V1M1C1EvaluatorError("latency rejected request did not deny")

    for _ in range(V1_M1_C1_LATENCY_WARMUPS):
        accepted_request()
        rejected_request()

    timings: dict[str, list[int]] = {
        "input_canonicalization_hash": [],
        "commitment_challenge_state": [],
        "credential_response_encoding": [],
        "neural_gate_layer": [],
        "accepted_gate_and_coordinator_without_r2": [],
        "trusted_preprocess_and_h2d": [],
        "r2_inference": [],
        "accepted_end_to_end": [],
        "rejected_end_to_end": [],
    }
    for index in range(V1_M1_C1_LATENCY_OBSERVATIONS):
        image = test_pixels[index : index + 1].contiguous()
        response_polynomials = _conformance_response(index)
        sample_commitment = _build_conformance_commitment(
            neural_profile.public_profile,
            response_polynomials,
        )
        timings["input_canonicalization_hash"].append(
            _measure_ns(device, partial(adapter.adapt, image))
        )
        trusted_input = adapter.adapt(image)
        timings["commitment_challenge_state"].append(
            _measure_ns(device, partial(issue_coordinator.begin, trusted_input, sample_commitment))
        )
        issued = noop_access.begin(image, sample_commitment)
        response = _response_for_issue(cast(dict[str, object], issued), response_polynomials)
        timings["credential_response_encoding"].append(
            _measure_ns(device, lambda: V1Response(bytes(32), V1_M1_C1_RESPONSE).encode())
        )
        timings["neural_gate_layer"].append(
            _measure_ns(
                device,
                lambda: verify_v1_neural(
                    commitment,
                    fixed_challenge,
                    neural_response,
                    bytes(32),
                    neural_profile,
                ),
            )
        )
        timings["accepted_gate_and_coordinator_without_r2"].append(
            _measure_ns(device, partial(noop_access.respond, response))
        )
        timings["trusted_preprocess_and_h2d"].append(
            _measure_ns(
                device,
                partial(_preprocess_and_h2d, image, device),
            )
        )
        timings["r2_inference"].append(
            _measure_ns(device, partial(_r2_inference, model, _preprocess_and_h2d(image, device)))
        )
        timings["accepted_end_to_end"].append(_measure_ns(device, accepted_request))
        timings["rejected_end_to_end"].append(_measure_ns(device, rejected_request))

    _synchronize(device)
    accepted_started = time.perf_counter_ns()
    for _ in range(V1_M1_C1_LATENCY_OBSERVATIONS):
        accepted_request()
    _synchronize(device)
    accepted_elapsed = time.perf_counter_ns() - accepted_started
    _synchronize(device)
    rejected_started = time.perf_counter_ns()
    for _ in range(V1_M1_C1_LATENCY_OBSERVATIONS):
        rejected_request()
    _synchronize(device)
    rejected_elapsed = time.perf_counter_ns() - rejected_started
    summaries = {name: _timing_summary(samples) for name, samples in timings.items()}
    coordinator_residual = max(
        0,
        cast(int, summaries["accepted_gate_and_coordinator_without_r2"]["median_ns"])
        - cast(int, summaries["neural_gate_layer"]["median_ns"]),
    )
    return {
        "method": "100 warmups, 1000 serialized observations, CUDA synchronize around each sample",
        "segments": summaries,
        "coordinator_commit_residual_estimate_median_ns": coordinator_residual,
        "throughput_per_second": {
            "accepted_end_to_end": V1_M1_C1_LATENCY_OBSERVATIONS * 1_000_000_000 / accepted_elapsed,
            "rejected_end_to_end": V1_M1_C1_LATENCY_OBSERVATIONS * 1_000_000_000 / rejected_elapsed,
        },
    }


def _validate_frozen_server_environment(device: torch.device) -> None:
    if type(device) is not torch.device or device != torch.device("cuda:0"):
        raise V1M1C1EvaluatorError("C1 evaluator requires cuda:0")
    if (
        not torch.cuda.is_available()
        or torch.version.cuda != "12.6"
        or str(torch.__version__) != "2.13.0+cu126"
        or str(torchvision.__version__) != "0.28.0+cu126"
        or torch.cuda.get_device_name(device) != "NVIDIA RTX A4000"
    ):
        raise V1M1C1EvaluatorError("runtime does not match the frozen V1 AutoDL environment")
    expected_seed = baseline.V1_M1_RUN_SEEDS[V1_M1_C1_RUN_INDEX - 1]
    if (
        os.environ.get("PYTHONHASHSEED") != str(expected_seed)
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise V1M1C1EvaluatorError(
            "C1 evaluator requires the accepted R2 deterministic environment variables"
        )
    baseline._configure_v1_m1_determinism(expected_seed)


def _report_path(artifact_root: Path) -> Path:
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise V1M1C1EvaluatorError("artifact root is missing or symlinked")
    report_root = artifact_root / V1_M1_C1_REPORT_DIRECTORY
    if report_root.exists() and (report_root.is_symlink() or not report_root.is_dir()):
        raise V1M1C1EvaluatorError("C1 report directory is not canonical")
    return report_root / V1_M1_C1_REPORT_FILENAME


def _atomic_write_report(path: Path, report: dict[str, object]) -> None:
    try:
        path.parent.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise V1M1C1EvaluatorError("C1 report directory cannot be created") from error
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise V1M1C1EvaluatorError("C1 report directory is not canonical")
    if path.exists() or path.is_symlink():
        raise V1M1C1EvaluatorError("refusing to overwrite an existing C1 report")
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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
        raise V1M1C1EvaluatorError("refusing to overwrite an existing C1 report") from error
    except OSError as error:
        raise V1M1C1EvaluatorError("C1 report write failed") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_v1_m1_c1_evaluator(
    data_root: Path = baseline.V1_M1_DATA_ROOT,
    artifact_root: Path = baseline.V1_M1_ARTIFACT_ROOT,
    device: torch.device = _V1_M1_C1_DEVICE,
) -> Path:
    """执行一次无训练 C1 accepted-R2 evaluator, 并写入 ignored report。"""
    _validate_frozen_server_environment(device)
    report_path = _report_path(artifact_root)
    model, decoded_sha256 = load_v1_m1_c1_accepted_r2(artifact_root, device)
    test_pixels, _test_labels = _load_raw_test_split(data_root, decoded_sha256)
    _public_profile, neural_profile, commitment = _build_public_conformance_material()
    equivalence = _evaluate_equivalence(model, device, test_pixels, neural_profile)
    reject_isolation = _probe_reject_isolation(
        model,
        test_pixels[0:1].contiguous(),
        neural_profile,
        commitment,
    )
    latency = _measure_latency(model, device, test_pixels, neural_profile, commitment)
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": V1_M1_C1_EXPERIMENT_ID,
        "accepted_r2": {
            "run_index": V1_M1_C1_RUN_INDEX,
            "selected_epoch": V1_M1_C1_ACCEPTED_SELECTED_EPOCH,
            "canonical_state_sha256": V1_M1_C1_ACCEPTED_STATE_SHA256,
            "baseline_predictions_sha256": V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
        "credential": {
            "kind": "public V1-P2 conformance fixture; no secret is generated, loaded, or reported",
            "challenge_weight": 2,
        },
        "equivalence": equivalence,
        "reject_isolation": reject_isolation,
        "latency": latency,
        "scope": (
            "No R2 training, fine-tuning, state mutation, data download, "
            "or state/report publication."
        ),
    }
    _atomic_write_report(report_path, report)
    return report_path


def main() -> None:
    """运行固定 AutoDL C1 accepted-R2 evaluator 命令行入口。"""
    report_path = run_v1_m1_c1_evaluator()
    print(f"C1 accepted-R2 report written: {report_path}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256",
    "V1_M1_C1_ACCEPTED_STATE_SHA256",
    "V1_M1_C1_EXPERIMENT_ID",
    "V1_M1_C1_REPORT_DIRECTORY",
    "V1_M1_C1_REPORT_FILENAME",
    "V1M1C1AcceptedR2",
    "V1M1C1EvaluatorError",
    "load_v1_m1_c1_accepted_r2",
    "load_v1_m1_c1_accepted_r2_details",
    "run_v1_m1_c1_evaluator",
]
