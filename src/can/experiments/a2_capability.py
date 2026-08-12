"""A2-E2 三态协调器的只评估实验与固定报告入口。"""

from __future__ import annotations

import json
import math
import resource
import time
from collections.abc import Callable, Sequence, Sized
from dataclasses import asdict
from pathlib import Path
from typing import Final, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from can.access import (
    A2_CAPABILITY_POLICY_VERSION,
    A2_CAPABILITY_RESPONSE_VERSION,
    A2CapabilityCoordinator,
    A2CapabilityPolicy,
    A2CapabilitySnapshot,
    A2CapabilityTimingSample,
)
from can.experiments import a2_baseline as protected_baseline
from can.experiments import a2_gate as gate_experiment
from can.experiments import a2_public_baseline as public_baseline
from can.model.a2_mlp import A2_EXPERIMENT_ID, A2_PARAMETER_COUNT, A2FashionMNISTMLP
from can.model.a2_public_mlp import (
    A2_PUBLIC_EXPERIMENT_ID,
    A2_PUBLIC_PARAMETER_COUNT,
    A2FashionMNISTPublicMLP,
)
from can.verifier.a1_torch import A1TorchBackend, verify_a1_torch

A2_CAPABILITY_EXPERIMENT_ID: Final = "CAN-A2-FMNIST-CAPABILITY-v1"
A2_CAPABILITY_REPORT_PATH: Final = protected_baseline.A2_REPORT_ROOT / "capability.json"
A2_EXPECTED_PROTECTED_PREDICTIONS_SHA256: Final = (
    "e5b48d60c19304e54c412416abd0201e9c747afd00830b93af9122a738a2e4a7"
)
A2_EXPECTED_PROTECTED_STATE_SHA256: Final = (
    "88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613"
)
A2_EXPECTED_PUBLIC_PREDICTIONS_SHA256: Final = (
    "f54b2351606f21ff31fc7c23ed394c4dbe13ccb9b150a7fe10b6b27076926f0a"
)
A2_EXPECTED_PUBLIC_STATE_SHA256: Final = (
    "b71980ebd3fb6e1a729b77109c98d3b4580e9e9cf8d3a28296cf6c18d1c122be"
)
A2_PROTECTED_BASELINE_REPORTS: Final = (
    protected_baseline.A2_REPORT_ROOT / "baseline-repeat-1.json",
    protected_baseline.A2_REPORT_ROOT / "baseline-repeat-2.json",
)
A2_PUBLIC_BASELINE_REPORTS: Final = (
    protected_baseline.A2_REPORT_ROOT / "public-baseline-repeat-1.json",
    protected_baseline.A2_REPORT_ROOT / "public-baseline-repeat-2.json",
)

_LATENCY_WARMUPS: Final = 100
_LATENCY_OBSERVATIONS: Final = 1_000
_BASELINE_REPORT_KEYS: Final = {
    "data",
    "determinism_fingerprint",
    "environment",
    "experiment_id",
    "latency",
    "model",
    "peak_rss_kib",
    "repeat",
    "schema_version",
    "test",
    "training",
}
_TEST_REPORT_KEYS: Final = {
    "accuracy_percent",
    "confusion_matrix",
    "correct",
    "loss",
    "per_class_accuracy_percent",
    "per_class_correct",
    "per_class_count",
    "predictions_sha256",
    "total",
}
_MODEL_REPORT_KEYS: Final = {
    "parameter_bytes",
    "parameter_count",
    "state_sha256",
    "temporary_serialized_bytes",
    "temporary_serialized_sha256",
    "topology",
}


class A2CapabilityExperimentError(RuntimeError):
    """表示 A2-E2 评估偏离已验收模型或三态实验契约。"""


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise A2CapabilityExperimentError("baseline report has duplicate or invalid keys")
        result[key] = value
    return result


def _load_json_report(path: Path) -> dict[str, object]:
    try:
        if not path.is_file() or path.stat().st_size > 100_000:
            raise A2CapabilityExperimentError("baseline report is missing or oversized")
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except A2CapabilityExperimentError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A2CapabilityExperimentError("baseline report is not canonical JSON") from error
    if type(value) is not dict:
        raise A2CapabilityExperimentError("baseline report root must be an object")
    return cast(dict[str, object], value)


def _exact_dict(value: object, keys: set[str], *, field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise A2CapabilityExperimentError(f"{field} fields changed")
    return cast(dict[str, object], value)


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise A2CapabilityExperimentError(f"{field} must be a canonical integer")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if type(value) not in (int, float):
        raise A2CapabilityExperimentError(f"{field} must be a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise A2CapabilityExperimentError(f"{field} must be a finite number")
    return number


def _exact_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise A2CapabilityExperimentError(f"{field} must be lowercase SHA-256")
    return value


def _exact_int_list(value: object, size: int, *, field: str) -> list[int]:
    if type(value) is not list or len(value) != size:
        raise A2CapabilityExperimentError(f"{field} shape changed")
    return [_exact_int(item, field=f"{field} item") for item in cast(list[object], value)]


def _finite_number_list(value: object, size: int, *, field: str) -> list[float]:
    if type(value) is not list or len(value) != size:
        raise A2CapabilityExperimentError(f"{field} shape changed")
    return [_finite_number(item, field=f"{field} item") for item in cast(list[object], value)]


def _validate_confusion_matrix(value: object, class_count: int) -> list[list[int]]:
    if type(value) is not list or len(value) != class_count:
        raise A2CapabilityExperimentError("baseline confusion matrix shape changed")
    result: list[list[int]] = []
    for row in value:
        if type(row) is not list or len(row) != class_count:
            raise A2CapabilityExperimentError("baseline confusion matrix shape changed")
        result.append(
            [_exact_int(item, field="confusion matrix count") for item in cast(list[object], row)]
        )
    return result


def _validate_baseline_report(
    report: dict[str, object],
    *,
    repeat: int,
    experiment_id: str,
    class_count: int,
    parameter_count: int,
    topology: str,
    predictions_sha256: str,
    state_sha256: str,
) -> dict[str, object]:
    if set(report) != _BASELINE_REPORT_KEYS:
        raise A2CapabilityExperimentError("baseline report top-level fields changed")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or type(report["repeat"]) is not int
        or report["repeat"] != repeat
        or type(report["experiment_id"]) is not str
        or report["experiment_id"] != experiment_id
    ):
        raise A2CapabilityExperimentError("baseline report identity changed")

    test = _exact_dict(report["test"], _TEST_REPORT_KEYS, field="baseline test")
    model = _exact_dict(report["model"], _MODEL_REPORT_KEYS, field="baseline model")
    if _exact_sha256(test["predictions_sha256"], field="prediction digest") != predictions_sha256:
        raise A2CapabilityExperimentError("baseline prediction digest changed")
    if _exact_sha256(model["state_sha256"], field="model-state digest") != state_sha256:
        raise A2CapabilityExperimentError("baseline model-state digest changed")
    if type(model["topology"]) is not str or model["topology"] != topology:
        raise A2CapabilityExperimentError("baseline topology changed")
    if _exact_int(model["parameter_count"], field="parameter_count") != parameter_count:
        raise A2CapabilityExperimentError("baseline parameter count changed")
    if _exact_int(model["parameter_bytes"], field="parameter_bytes") != parameter_count * 4:
        raise A2CapabilityExperimentError("baseline parameter bytes changed")
    _exact_int(
        model["temporary_serialized_bytes"],
        field="temporary serialized bytes",
        minimum=1,
    )
    _exact_sha256(model["temporary_serialized_sha256"], field="temporary serialization digest")
    if _exact_int(test["total"], field="test total", minimum=1) != protected_baseline.TEST_SIZE:
        raise A2CapabilityExperimentError("baseline test size changed")
    correct = _exact_int(test["correct"], field="test correct")
    if correct > protected_baseline.TEST_SIZE:
        raise A2CapabilityExperimentError("baseline correct count changed")
    loss = _finite_number(test["loss"], field="test loss")
    accuracy = _finite_number(test["accuracy_percent"], field="test accuracy")
    if loss < 0.0 or not 0.0 <= accuracy <= 100.0:
        raise A2CapabilityExperimentError("baseline metric range changed")
    confusion = _validate_confusion_matrix(test["confusion_matrix"], class_count)
    class_correct = _exact_int_list(
        test["per_class_correct"], class_count, field="per-class correct"
    )
    class_count_values = _exact_int_list(
        test["per_class_count"], class_count, field="per-class count"
    )
    class_accuracy = _finite_number_list(
        test["per_class_accuracy_percent"], class_count, field="per-class accuracy"
    )
    if (
        sum(sum(row) for row in confusion) != protected_baseline.TEST_SIZE
        or sum(class_count_values) != protected_baseline.TEST_SIZE
        or sum(class_correct) != correct
        or [sum(row) for row in confusion] != class_count_values
        or [confusion[index][index] for index in range(class_count)] != class_correct
        or not math.isclose(
            accuracy,
            correct * 100.0 / protected_baseline.TEST_SIZE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise A2CapabilityExperimentError("baseline aggregate metrics are inconsistent")
    for count, per_class_correct, per_class_accuracy in zip(
        class_count_values, class_correct, class_accuracy, strict=True
    ):
        if (
            count < 1
            or per_class_correct > count
            or not math.isclose(
                per_class_accuracy,
                per_class_correct * 100.0 / count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise A2CapabilityExperimentError("baseline per-class metrics are inconsistent")
    fingerprint = _exact_sha256(report["determinism_fingerprint"], field="determinism fingerprint")
    return {
        "test": test,
        "model": model,
        "determinism_fingerprint": fingerprint,
    }


def _load_accepted_baseline_references() -> dict[str, object]:
    protected_references = tuple(
        _validate_baseline_report(
            _load_json_report(path),
            repeat=index,
            experiment_id=A2_EXPERIMENT_ID,
            class_count=10,
            parameter_count=A2_PARAMETER_COUNT,
            topology="784->256->128->10",
            predictions_sha256=A2_EXPECTED_PROTECTED_PREDICTIONS_SHA256,
            state_sha256=A2_EXPECTED_PROTECTED_STATE_SHA256,
        )
        for index, path in enumerate(A2_PROTECTED_BASELINE_REPORTS, start=1)
    )
    public_references = tuple(
        _validate_baseline_report(
            _load_json_report(path),
            repeat=index,
            experiment_id=A2_PUBLIC_EXPERIMENT_ID,
            class_count=2,
            parameter_count=A2_PUBLIC_PARAMETER_COUNT,
            topology="784->64->2",
            predictions_sha256=A2_EXPECTED_PUBLIC_PREDICTIONS_SHA256,
            state_sha256=A2_EXPECTED_PUBLIC_STATE_SHA256,
        )
        for index, path in enumerate(A2_PUBLIC_BASELINE_REPORTS, start=1)
    )
    if protected_references[0] != protected_references[1]:
        raise A2CapabilityExperimentError("protected baseline repeats diverged")
    if public_references[0] != public_references[1]:
        raise A2CapabilityExperimentError("public baseline repeats diverged")
    return {
        "protected": protected_references[0],
        "public": public_references[0],
    }


def _single_image(dataset: Dataset[tuple[Tensor, int]], index: int) -> Tensor:
    return gate_experiment._single_image(dataset, index)


def _validate_materialized_models(
    protected_model: A2FashionMNISTMLP,
    public_model: A2FashionMNISTPublicMLP,
) -> tuple[str, str]:
    protected_state = protected_baseline._hash_model_state(protected_model)
    public_state = public_baseline._hash_model_state(public_model)
    if protected_state != A2_EXPECTED_PROTECTED_STATE_SHA256:
        raise A2CapabilityExperimentError("protected model is not the accepted baseline state")
    if public_state != A2_EXPECTED_PUBLIC_STATE_SHA256:
        raise A2CapabilityExperimentError("public model is not the accepted baseline state")
    return protected_state, public_state


def _validate_all_labels(
    protected_model: A2FashionMNISTMLP,
    public_model: A2FashionMNISTPublicMLP,
    dataset: Dataset[tuple[Tensor, int]],
    backend: A1TorchBackend,
    accepted_credential: bytes,
    rejected_credential: bytes,
) -> dict[str, object]:
    if len(cast(Sized, dataset)) != protected_baseline.TEST_SIZE:
        raise A2CapabilityExperimentError("official test dataset size changed")
    protected_state_before, public_state_before = _validate_materialized_models(
        protected_model, public_model
    )
    coordinator = A2CapabilityCoordinator(
        backend,
        protected_model,
        public_model=public_model,
        policy=A2CapabilityPolicy(public_entry_enabled=True),
    )
    protected_predictions: list[int] = []
    public_predictions: list[int] = []
    for index in range(protected_baseline.TEST_SIZE):
        image = _single_image(dataset, index)
        with torch.inference_mode():
            expected_public = int(public_model(image).argmax(dim=1).item())
            expected_protected = int(protected_model(image).argmax(dim=1).item())
        public_response = coordinator.handle_public(image)
        protected_response = coordinator.handle_protected(image, accepted_credential)
        if public_response != {
            "version": 2,
            "status": "public",
            "coarse_class_id": expected_public,
        } or protected_response != {
            "version": 2,
            "status": "protected",
            "class_id": expected_protected,
        }:
            raise A2CapabilityExperimentError("coordinator label diverged from direct model")
        public_predictions.append(expected_public)
        protected_predictions.append(expected_protected)

    rejection_response = coordinator.handle_protected(
        _single_image(dataset, 0), rejected_credential
    )
    if rejection_response != {"version": 2, "status": "deny"}:
        raise A2CapabilityExperimentError("protected rejection probe changed")
    expected_snapshot = A2CapabilitySnapshot(
        verifier_calls=protected_baseline.TEST_SIZE + 1,
        coordinator_commits=2 * protected_baseline.TEST_SIZE + 1,
        deny_commits=1,
        public_commits=protected_baseline.TEST_SIZE,
        protected_commits=protected_baseline.TEST_SIZE,
        public_model_calls=protected_baseline.TEST_SIZE,
        protected_model_calls=protected_baseline.TEST_SIZE,
        deny_responses=1,
        public_responses=protected_baseline.TEST_SIZE,
        protected_responses=protected_baseline.TEST_SIZE,
    )
    if coordinator.snapshot() != expected_snapshot:
        raise A2CapabilityExperimentError("three-state invocation counts changed")

    public_hash = public_baseline._hash_int64_values(public_predictions)
    protected_hash = protected_baseline._hash_int64_values(protected_predictions)
    if public_hash != A2_EXPECTED_PUBLIC_PREDICTIONS_SHA256:
        raise A2CapabilityExperimentError("integrated public predictions changed")
    if protected_hash != A2_EXPECTED_PROTECTED_PREDICTIONS_SHA256:
        raise A2CapabilityExperimentError("integrated protected predictions changed")
    protected_state_after, public_state_after = _validate_materialized_models(
        protected_model, public_model
    )
    if protected_state_after != protected_state_before or public_state_after != public_state_before:
        raise A2CapabilityExperimentError("coordinator changed a model state")
    return {
        "protected_predictions_sha256": protected_hash,
        "public_predictions_sha256": public_hash,
        "all_protected_labels_match_accepted_a2_e1": True,
        "all_public_outputs_canonical": True,
        "counts": asdict(expected_snapshot),
    }


def _nearest_rank_us(values: Sequence[int], percentile: float) -> float:
    if not values or not 0.0 < percentile <= 1.0:
        raise A2CapabilityExperimentError("latency statistics require canonical observations")
    if any(type(value) is not int or value < 0 for value in values):
        raise A2CapabilityExperimentError("latency observations changed type or range")
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank] / 1_000.0


def _latency_statistics(values: Sequence[int]) -> dict[str, float]:
    return {
        "median_us": _nearest_rank_us(values, 0.50),
        "p95_us": _nearest_rank_us(values, 0.95),
        "p99_us": _nearest_rank_us(values, 0.99),
    }


def _time_batches(
    batches: Sequence[Tensor], operation: Callable[[Tensor], object]
) -> dict[str, float]:
    if len(batches) != _LATENCY_OBSERVATIONS:
        raise A2CapabilityExperimentError("latency requires the fixed first 1,000 samples")
    for index in range(_LATENCY_WARMUPS):
        operation(batches[index % len(batches)])
    observations: list[int] = []
    for image in batches:
        start = time.perf_counter_ns()
        operation(image)
        observations.append(time.perf_counter_ns() - start)
    return _latency_statistics(observations)


def _stage_statistics(
    samples: Sequence[A2CapabilityTimingSample], attribute: str
) -> dict[str, float]:
    if len(samples) < _LATENCY_OBSERVATIONS:
        raise A2CapabilityExperimentError("coordinator retained too few timing samples")
    values = [getattr(sample, attribute) for sample in samples[-_LATENCY_OBSERVATIONS:]]
    return _latency_statistics(cast(list[int], values))


def _measure_latency(
    protected_model: A2FashionMNISTMLP,
    public_model: A2FashionMNISTPublicMLP,
    dataset: Dataset[tuple[Tensor, int]],
    backend: A1TorchBackend,
    accepted_credential: bytes,
    rejected_credential: bytes,
) -> dict[str, object]:
    batches = tuple(_single_image(dataset, index) for index in range(_LATENCY_OBSERVATIONS))
    policy = A2CapabilityPolicy(public_entry_enabled=True)
    public_coordinator = A2CapabilityCoordinator(
        backend, protected_model, public_model=public_model, policy=policy
    )
    protected_coordinator = A2CapabilityCoordinator(
        backend, protected_model, public_model=public_model, policy=policy
    )
    deny_coordinator = A2CapabilityCoordinator(
        backend, protected_model, public_model=public_model, policy=policy
    )

    with torch.inference_mode():
        public_model_only = _time_batches(batches, public_model)
        protected_model_only = _time_batches(batches, protected_model)
    verifier_only = _time_batches(
        batches, lambda _image: verify_a1_torch(accepted_credential, backend)
    )
    public_end_to_end = _time_batches(batches, public_coordinator.handle_public)
    protected_end_to_end = _time_batches(
        batches,
        lambda image: protected_coordinator.handle_protected(image, accepted_credential),
    )
    deny_end_to_end = _time_batches(
        batches,
        lambda image: deny_coordinator.handle_protected(image, rejected_credential),
    )
    deny_snapshot = deny_coordinator.snapshot()
    if deny_snapshot.public_model_calls != 0 or deny_snapshot.protected_model_calls != 0:
        raise A2CapabilityExperimentError("deny latency path invoked a model")

    public_samples = public_coordinator.timing_snapshot()
    protected_samples = protected_coordinator.timing_snapshot()
    deny_samples = deny_coordinator.timing_snapshot()
    return {
        "method": "100 warm-up plus 1000 perf_counter_ns observations, one CPU thread",
        "public_model_only": public_model_only,
        "protected_model_only": protected_model_only,
        "verifier_only": verifier_only,
        "public_end_to_end": public_end_to_end,
        "protected_end_to_end": protected_end_to_end,
        "deny_end_to_end": deny_end_to_end,
        "public_internal": {
            "validation": _stage_statistics(public_samples, "validation_ns"),
            "coordinator": _stage_statistics(public_samples, "coordinator_ns"),
            "public_model": _stage_statistics(public_samples, "public_model_ns"),
        },
        "protected_internal": {
            "validation": _stage_statistics(protected_samples, "validation_ns"),
            "verifier": _stage_statistics(protected_samples, "verifier_ns"),
            "coordinator": _stage_statistics(protected_samples, "coordinator_ns"),
            "protected_model": _stage_statistics(protected_samples, "protected_model_ns"),
        },
        "deny_internal": {
            "validation": _stage_statistics(deny_samples, "validation_ns"),
            "verifier": _stage_statistics(deny_samples, "verifier_ns"),
            "coordinator": _stage_statistics(deny_samples, "coordinator_ns"),
        },
        "public_counts": asdict(public_coordinator.snapshot()),
        "protected_counts": asdict(protected_coordinator.snapshot()),
        "deny_counts": asdict(deny_snapshot),
    }


def _compact_response_size(response: dict[str, object]) -> int:
    return len(
        json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )


def _write_capability_report(report: dict[str, object]) -> Path:
    A2_CAPABILITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = A2_CAPABILITY_REPORT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(A2_CAPABILITY_REPORT_PATH)
    return A2_CAPABILITY_REPORT_PATH


def run_a2_capability_experiment(
    protected_model: A2FashionMNISTMLP,
    public_model: A2FashionMNISTPublicMLP,
) -> Path:
    """只评估两个已验收模型, 验证三态隔离并写入固定 ignored 报告。"""
    protected_baseline._validate_environment()
    protected_baseline._configure_determinism()
    baseline_references = _load_accepted_baseline_references()
    _validate_materialized_models(protected_model, public_model)
    resource_hashes = protected_baseline._validate_data_resources(protected_baseline.A2_DATA_ROOT)
    data = protected_baseline._load_data(protected_baseline.A2_DATA_ROOT)
    backend, accepted_credential, rejected_credential = gate_experiment._build_toy_gate()
    label_validation = _validate_all_labels(
        protected_model,
        public_model,
        data.test_dataset,
        backend,
        accepted_credential,
        rejected_credential,
    )
    latency = _measure_latency(
        protected_model,
        public_model,
        data.test_dataset,
        backend,
        accepted_credential,
        rejected_credential,
    )
    disabled_probe = A2CapabilityCoordinator(backend, protected_model)
    disabled_response = disabled_probe.handle_public(_single_image(data.test_dataset, 0))
    if disabled_response != {"version": 2, "status": "deny"}:
        raise A2CapabilityExperimentError("default-disabled public entry changed")

    responses: dict[str, dict[str, object]] = {
        "deny": {"version": 2, "status": "deny"},
        "public": {"version": 2, "status": "public", "coarse_class_id": 0},
        "protected": {"version": 2, "status": "protected", "class_id": 0},
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": A2_CAPABILITY_EXPERIMENT_ID,
        "response_version": A2_CAPABILITY_RESPONSE_VERSION,
        "policy_version": A2_CAPABILITY_POLICY_VERSION,
        "toy_numerical_unlock_only": True,
        "no_training_performed": True,
        "environment": protected_baseline._environment_report(),
        "data": {
            "root": str(protected_baseline.A2_DATA_ROOT),
            "resources": resource_hashes,
            "test_size": protected_baseline.TEST_SIZE,
        },
        "accepted_baselines": baseline_references,
        "integration": label_validation,
        "default_disabled_probe": {
            "startup_event": asdict(disabled_probe.startup_audit_event()),
            "counts": asdict(disabled_probe.snapshot()),
        },
        "responses": {
            name: {
                "fields": sorted(response),
                "compact_json_bytes": _compact_response_size(response),
            }
            for name, response in responses.items()
        },
        "latency": latency,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return _write_capability_report(report)
