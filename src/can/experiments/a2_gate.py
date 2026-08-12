"""A2-E1 二元硬门控的确定性标签与延迟实验。"""

from __future__ import annotations

import argparse
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

from can.access import A2AccessCoordinator, A2AccessSnapshot, A2TimingSample
from can.experiments import a2_baseline as baseline
from can.model.a2_mlp import A2_EXPERIMENT_ID, A2FashionMNISTMLP, validate_a2_images
from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A0Slot,
    mod_q,
)
from can.verifier import A1CompiledRegistry, compile_a1_profile
from can.verifier.a1_torch import A1TorchBackend, compile_a1_torch_backend, verify_a1_torch

A2_GATE_EXPERIMENT_ID: Final = "CAN-A2-FMNIST-MLP-GATE-v1"
A2_GATE_REPORT_PATH: Final = baseline.A2_REPORT_ROOT / "gate.json"
A2_GATE_SLOT_ID: Final = 0xA2E1
A2_EXPECTED_PREDICTIONS_SHA256: Final = (
    "e5b48d60c19304e54c412416abd0201e9c747afd00830b93af9122a738a2e4a7"
)
A2_EXPECTED_MODEL_STATE_SHA256: Final = (
    "88062fee1b8d25672dcb7c3559369bfef49aa9907a6a3e9aabedb6b232318613"
)
_LATENCY_WARMUPS: Final = 100
_LATENCY_OBSERVATIONS: Final = 1_000


class A2GateExperimentError(RuntimeError):
    """表示门控实验偏离固定模型、关系或计量契约。"""


def _encode_credential(coefficients: Sequence[int]) -> bytes:
    if len(coefficients) != A0_COMPONENT_COUNT or any(
        type(value) is not int or not 0 <= value <= 256 for value in coefficients
    ):
        raise A2GateExperimentError("toy credential coefficients are non-canonical")
    return (
        bytes([A0_VERSION])
        + A0_PROFILE_ID.to_bytes(2, byteorder="big", signed=False)
        + A2_GATE_SLOT_ID.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in coefficients)
    )


def _build_toy_gate() -> tuple[A1TorchBackend, bytes, bytes]:
    slot = A0Slot(
        A2_GATE_SLOT_ID,
        [[row_index + 1] * A0_SECRET_SIZE for row_index in range(A0_COMPONENT_COUNT)],
    )
    profile = compile_a1_profile(slot, (1,) * A0_SECRET_SIZE)
    accepted_coefficients = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    rejected_coefficients = accepted_coefficients.copy()
    rejected_coefficients[0] = mod_q(rejected_coefficients[0] + 9)
    backend = compile_a1_torch_backend(A1CompiledRegistry([profile]))
    return (
        backend,
        _encode_credential(accepted_coefficients),
        _encode_credential(rejected_coefficients),
    )


def _single_image(dataset: Dataset[tuple[Tensor, int]], index: int) -> Tensor:
    image, label = dataset[index]
    if type(image) is not Tensor or type(label) is not int or not 0 <= label < 10:
        raise A2GateExperimentError("test dataset returned a non-canonical sample")
    batch = image.unsqueeze(0).contiguous()
    validate_a2_images(batch)
    return batch


def _validate_all_labels(
    model: A2FashionMNISTMLP,
    dataset: Dataset[tuple[Tensor, int]],
    backend: A1TorchBackend,
    accepted_credential: bytes,
    expected_predictions_sha256: str,
) -> tuple[str, A2AccessSnapshot]:
    if len(cast(Sized, dataset)) != baseline.TEST_SIZE:
        raise A2GateExperimentError("official test dataset size changed")
    coordinator = A2AccessCoordinator(backend, model)
    predictions: list[int] = []
    for index in range(baseline.TEST_SIZE):
        response = coordinator.handle(_single_image(dataset, index), accepted_credential)
        if set(response) != {"version", "status", "class_id"} or response["status"] != "ok":
            raise A2GateExperimentError("accepted gate path returned a non-canonical response")
        class_id = response["class_id"]
        if type(class_id) is not int or not 0 <= class_id < 10:
            raise A2GateExperimentError("accepted gate path returned a non-canonical class")
        predictions.append(class_id)

    prediction_hash = baseline._hash_int64_values(predictions)
    snapshot = coordinator.snapshot()
    if prediction_hash != expected_predictions_sha256:
        raise A2GateExperimentError("gated labels diverged from the ungated baseline")
    if snapshot != A2AccessSnapshot(
        verifier_calls=baseline.TEST_SIZE,
        coordinator_commits=baseline.TEST_SIZE,
        allow_commits=baseline.TEST_SIZE,
        deny_commits=0,
        protected_model_calls=baseline.TEST_SIZE,
        ok_responses=baseline.TEST_SIZE,
        deny_responses=0,
    ):
        raise A2GateExperimentError("accepted gate invocation counts changed")
    return prediction_hash, snapshot


def _nearest_rank_ns(values: Sequence[int], percentile: float) -> float:
    if not values or not 0.0 < percentile <= 1.0:
        raise A2GateExperimentError("latency statistics require canonical observations")
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank] / 1_000.0


def _latency_statistics(values: Sequence[int]) -> dict[str, float]:
    return {
        "median_us": _nearest_rank_ns(values, 0.50),
        "p95_us": _nearest_rank_ns(values, 0.95),
        "p99_us": _nearest_rank_ns(values, 0.99),
    }


def _time_batches(
    batches: Sequence[Tensor], operation: Callable[[Tensor], object]
) -> dict[str, float]:
    if len(batches) != _LATENCY_OBSERVATIONS:
        raise A2GateExperimentError("latency requires the fixed first 1,000 samples")
    for index in range(_LATENCY_WARMUPS):
        operation(batches[index % len(batches)])
    elapsed: list[int] = []
    for index in range(_LATENCY_OBSERVATIONS):
        start = time.perf_counter_ns()
        operation(batches[index])
        elapsed.append(time.perf_counter_ns() - start)
    return _latency_statistics(elapsed)


def _sample_stage_statistics(samples: Sequence[A2TimingSample], attribute: str) -> dict[str, float]:
    if len(samples) < _LATENCY_OBSERVATIONS:
        raise A2GateExperimentError("coordinator did not retain enough timing samples")
    measured = samples[-_LATENCY_OBSERVATIONS:]
    values = [getattr(sample, attribute) for sample in measured]
    if any(type(value) is not int or value < 0 for value in values):
        raise A2GateExperimentError("coordinator timing sample changed type or range")
    return _latency_statistics(cast(list[int], values))


def _measure_gate_latency(
    model: A2FashionMNISTMLP,
    dataset: Dataset[tuple[Tensor, int]],
    backend: A1TorchBackend,
    accepted_credential: bytes,
    rejected_credential: bytes,
) -> dict[str, object]:
    batches = tuple(_single_image(dataset, index) for index in range(_LATENCY_OBSERVATIONS))
    batch_256 = torch.cat(batches[: baseline.EVALUATION_BATCH_SIZE], dim=0).contiguous()
    validate_a2_images(batch_256)

    with torch.inference_mode():
        model_only = _time_batches(batches, model)
        batch_256_latency = baseline._time_model(model, (batch_256,))
    verifier_only = _time_batches(
        batches,
        lambda _batch: verify_a1_torch(accepted_credential, backend),
    )

    accepted_coordinator = A2AccessCoordinator(backend, model)
    accepted_end_to_end = _time_batches(
        batches,
        lambda batch: accepted_coordinator.handle(batch, accepted_credential),
    )
    rejected_coordinator = A2AccessCoordinator(backend, model)
    rejected_end_to_end = _time_batches(
        batches,
        lambda batch: rejected_coordinator.handle(batch, rejected_credential),
    )
    if rejected_coordinator.snapshot().protected_model_calls != 0:
        raise A2GateExperimentError("rejected latency path called the protected model")

    accepted_samples = accepted_coordinator.timing_snapshot()
    rejected_samples = rejected_coordinator.timing_snapshot()
    absolute_overhead = accepted_end_to_end["median_us"] - model_only["median_us"]
    percentage_overhead = absolute_overhead * 100.0 / model_only["median_us"]
    return {
        "method": "100 warm-up plus 1000 perf_counter_ns observations, one CPU thread",
        "model_only_batch_1": model_only,
        "accepted_end_to_end_batch_1": accepted_end_to_end,
        "rejected_end_to_end_batch_1": rejected_end_to_end,
        "verifier_only": verifier_only,
        "accepted_internal": {
            "validation": _sample_stage_statistics(accepted_samples, "validation_ns"),
            "verifier": _sample_stage_statistics(accepted_samples, "verifier_ns"),
            "coordinator": _sample_stage_statistics(accepted_samples, "coordinator_ns"),
            "protected_model": _sample_stage_statistics(accepted_samples, "protected_model_ns"),
        },
        "rejected_internal": {
            "validation": _sample_stage_statistics(rejected_samples, "validation_ns"),
            "verifier": _sample_stage_statistics(rejected_samples, "verifier_ns"),
            "coordinator": _sample_stage_statistics(rejected_samples, "coordinator_ns"),
        },
        "accepted_median_overhead_us": absolute_overhead,
        "accepted_median_overhead_percent": percentage_overhead,
        "model_only_batch_256": batch_256_latency,
        "accepted_counts": asdict(accepted_coordinator.snapshot()),
        "rejected_counts": asdict(rejected_coordinator.snapshot()),
    }


def _write_gate_report(report: dict[str, object]) -> Path:
    A2_GATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = A2_GATE_REPORT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(A2_GATE_REPORT_PATH)
    return A2_GATE_REPORT_PATH


def run_a2_gate_experiment() -> Path:
    """训练固定 MLP, 验证全部 gated 标签并写入 ignored JSON 报告。"""
    baseline._validate_environment()
    baseline._configure_determinism()
    resource_hashes = baseline._validate_data_resources(baseline.A2_DATA_ROOT)
    data = baseline._load_data(baseline.A2_DATA_ROOT)
    model, epoch_metrics = baseline._train_model(data, emit_progress=True)
    model.eval()
    direct_metrics = baseline._evaluate(model, data.test_loader)
    model_state_sha256 = baseline._hash_model_state(model)
    if (
        direct_metrics.predictions_sha256 != A2_EXPECTED_PREDICTIONS_SHA256
        or model_state_sha256 != A2_EXPECTED_MODEL_STATE_SHA256
    ):
        raise A2GateExperimentError("retrained baseline identity changed")

    backend, accepted_credential, rejected_credential = _build_toy_gate()
    gated_hash, accepted_counts = _validate_all_labels(
        model,
        data.test_dataset,
        backend,
        accepted_credential,
        direct_metrics.predictions_sha256,
    )
    rejection_probe = A2AccessCoordinator(backend, model)
    reject_response = rejection_probe.handle(
        _single_image(data.test_dataset, 0), rejected_credential
    )
    if reject_response != {"version": 1, "status": "deny"}:
        raise A2GateExperimentError("rejection probe response changed")
    if rejection_probe.snapshot().protected_model_calls != 0:
        raise A2GateExperimentError("rejection probe called the protected model")

    latency = _measure_gate_latency(
        model,
        data.test_dataset,
        backend,
        accepted_credential,
        rejected_credential,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": A2_GATE_EXPERIMENT_ID,
        "business_experiment_id": A2_EXPERIMENT_ID,
        "environment": baseline._environment_report(),
        "data": {
            "root": str(baseline.A2_DATA_ROOT),
            "resources": resource_hashes,
            "test_size": baseline.TEST_SIZE,
        },
        "training": {
            "global_seed": baseline.GLOBAL_SEED,
            "epochs": [
                {
                    "epoch": item.epoch,
                    "training_loss": item.training_loss,
                    "validation_loss": item.validation_loss,
                    "validation_accuracy_percent": item.validation_accuracy_percent,
                }
                for item in epoch_metrics
            ],
        },
        "baseline": {
            "test_accuracy_percent": direct_metrics.accuracy_percent,
            "predictions_sha256": direct_metrics.predictions_sha256,
            "model_state_sha256": model_state_sha256,
        },
        "gate": {
            "toy_numerical_unlock_only": True,
            "gated_predictions_sha256": gated_hash,
            "all_10000_labels_match": gated_hash == direct_metrics.predictions_sha256,
            "accepted_counts": asdict(accepted_counts),
            "rejected_probe_counts": asdict(rejection_probe.snapshot()),
        },
        "latency": latency,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return _write_gate_report(report)


def main(argv: Sequence[str] | None = None) -> int:
    """解析固定 gate 命令并执行 A2-E1 硬门控实验。"""
    parser = argparse.ArgumentParser(description="Run the fixed CAN A2-E1 gate experiment")
    parser.add_argument("--run", action="store_true", required=True)
    arguments = parser.parse_args(argv)
    if not arguments.run:
        raise A2GateExperimentError("gate experiment requires the fixed run mode")
    output_path = run_a2_gate_experiment()
    print(json.dumps({"report": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
