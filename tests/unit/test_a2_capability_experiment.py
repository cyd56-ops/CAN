"""A2-E2 三态只评估实验 helper 的单元测试。"""

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

import can.experiments.a2_baseline as protected_baseline
import can.experiments.a2_capability as capability_experiment
import can.experiments.a2_public_baseline as public_baseline
from can.model.a2_mlp import A2_EXPERIMENT_ID, A2_PARAMETER_COUNT, A2FashionMNISTMLP
from can.model.a2_public_mlp import (
    A2_PUBLIC_EXPERIMENT_ID,
    A2_PUBLIC_PARAMETER_COUNT,
    A2FashionMNISTPublicMLP,
)
from can.verifier import A1CompiledProfile
from can.verifier.a1_torch import A1TorchBackend


class _TinyImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self) -> None:
        self._images = tuple(
            torch.full((1, 28, 28), value, dtype=torch.float32) for value in (0.0, 0.5, 1.0)
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._images[index], index


def _models() -> tuple[A2FashionMNISTMLP, A2FashionMNISTPublicMLP]:
    torch.manual_seed(20_260_729)
    protected_model = A2FashionMNISTMLP().eval()
    torch.manual_seed(20_260_730)
    public_model = A2FashionMNISTPublicMLP().eval()
    return protected_model, public_model


def _baseline_report(
    *,
    repeat: int,
    experiment_id: str,
    class_count: int,
    parameter_count: int,
    topology: str,
    predictions_sha256: str,
    state_sha256: str,
) -> dict[str, object]:
    per_class_count = [10_000 // class_count] * class_count
    per_class_count[-1] += 10_000 - sum(per_class_count)
    confusion = [
        [count if row_index == column_index else 0 for column_index in range(class_count)]
        for row_index, count in enumerate(per_class_count)
    ]
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "repeat": repeat,
        "environment": {},
        "data": {},
        "training": {},
        "test": {
            "loss": 0.0,
            "accuracy_percent": 100.0,
            "correct": 10_000,
            "total": 10_000,
            "per_class_correct": per_class_count,
            "per_class_count": per_class_count,
            "per_class_accuracy_percent": [100.0] * class_count,
            "confusion_matrix": confusion,
            "predictions_sha256": predictions_sha256,
        },
        "model": {
            "topology": topology,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_count * 4,
            "state_sha256": state_sha256,
            "temporary_serialized_bytes": 1,
            "temporary_serialized_sha256": "c" * 64,
        },
        "latency": {},
        "peak_rss_kib": 1,
        "determinism_fingerprint": "d" * 64,
    }


def test_three_state_label_validation_matches_both_direct_models(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 coordinator 的两条成功路线必须逐标签等于各自直接模型。"""
    _, backend, accepted, rejected = a2_gate_fixture
    dataset = _TinyImageDataset()
    protected_model, public_model = _models()
    with torch.inference_mode():
        protected_predictions = [
            int(protected_model(dataset[index][0].unsqueeze(0)).argmax(dim=1).item())
            for index in range(len(dataset))
        ]
        public_predictions = [
            int(public_model(dataset[index][0].unsqueeze(0)).argmax(dim=1).item())
            for index in range(len(dataset))
        ]
    monkeypatch.setattr(protected_baseline, "TEST_SIZE", len(dataset))
    monkeypatch.setattr(
        capability_experiment,
        "A2_EXPECTED_PROTECTED_STATE_SHA256",
        protected_baseline._hash_model_state(protected_model),
    )
    monkeypatch.setattr(
        capability_experiment,
        "A2_EXPECTED_PUBLIC_STATE_SHA256",
        public_baseline._hash_model_state(public_model),
    )
    monkeypatch.setattr(
        capability_experiment,
        "A2_EXPECTED_PROTECTED_PREDICTIONS_SHA256",
        protected_baseline._hash_int64_values(protected_predictions),
    )
    monkeypatch.setattr(
        capability_experiment,
        "A2_EXPECTED_PUBLIC_PREDICTIONS_SHA256",
        public_baseline._hash_int64_values(public_predictions),
    )

    observed = capability_experiment._validate_all_labels(
        protected_model,
        public_model,
        dataset,
        backend,
        accepted,
        rejected,
    )

    counts = observed["counts"]
    assert type(counts) is dict
    assert counts == {
        "verifier_calls": 4,
        "coordinator_commits": 7,
        "deny_commits": 1,
        "public_commits": 3,
        "protected_commits": 3,
        "public_model_calls": 3,
        "protected_model_calls": 3,
        "deny_responses": 1,
        "public_responses": 3,
        "protected_responses": 3,
    }
    assert observed["all_protected_labels_match_accepted_a2_e1"] is True
    assert observed["all_public_outputs_canonical"] is True


def test_accepted_baseline_loader_requires_two_identical_canonical_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """聚合报告只能引用两个一致且摘要固定的 baseline repeats。"""
    protected_prediction = "1" * 64
    protected_state = "2" * 64
    public_prediction = "3" * 64
    public_state = "4" * 64
    protected_paths = (tmp_path / "protected-1.json", tmp_path / "protected-2.json")
    public_paths = (tmp_path / "public-1.json", tmp_path / "public-2.json")
    for repeat, path in enumerate(protected_paths, start=1):
        path.write_text(
            json.dumps(
                _baseline_report(
                    repeat=repeat,
                    experiment_id=A2_EXPERIMENT_ID,
                    class_count=10,
                    parameter_count=A2_PARAMETER_COUNT,
                    topology="784->256->128->10",
                    predictions_sha256=protected_prediction,
                    state_sha256=protected_state,
                )
            ),
            encoding="utf-8",
        )
    for repeat, path in enumerate(public_paths, start=1):
        path.write_text(
            json.dumps(
                _baseline_report(
                    repeat=repeat,
                    experiment_id=A2_PUBLIC_EXPERIMENT_ID,
                    class_count=2,
                    parameter_count=A2_PUBLIC_PARAMETER_COUNT,
                    topology="784->64->2",
                    predictions_sha256=public_prediction,
                    state_sha256=public_state,
                )
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(capability_experiment, "A2_PROTECTED_BASELINE_REPORTS", protected_paths)
    monkeypatch.setattr(capability_experiment, "A2_PUBLIC_BASELINE_REPORTS", public_paths)
    monkeypatch.setattr(
        capability_experiment,
        "A2_EXPECTED_PROTECTED_PREDICTIONS_SHA256",
        protected_prediction,
    )
    monkeypatch.setattr(
        capability_experiment, "A2_EXPECTED_PROTECTED_STATE_SHA256", protected_state
    )
    monkeypatch.setattr(
        capability_experiment,
        "A2_EXPECTED_PUBLIC_PREDICTIONS_SHA256",
        public_prediction,
    )
    monkeypatch.setattr(capability_experiment, "A2_EXPECTED_PUBLIC_STATE_SHA256", public_state)

    observed = capability_experiment._load_accepted_baseline_references()

    assert set(observed) == {"protected", "public"}
    protected_reference = observed["protected"]
    assert type(protected_reference) is dict
    assert set(protected_reference) == {
        "test",
        "model",
        "determinism_fingerprint",
    }


def test_json_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    """重复 JSON 字段不能被后值静默覆盖。"""
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(capability_experiment.A2CapabilityExperimentError):
        capability_experiment._load_json_report(path)


def test_latency_statistics_and_fixed_report_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """延迟使用 nearest-rank, 报告只写固定路径并原子替换。"""
    assert capability_experiment._latency_statistics(range(1_000, 101_000, 1_000)) == {
        "median_us": 50.0,
        "p95_us": 95.0,
        "p99_us": 99.0,
    }
    output_path = tmp_path / "a2" / "capability.json"
    monkeypatch.setattr(capability_experiment, "A2_CAPABILITY_REPORT_PATH", output_path)

    observed = capability_experiment._write_capability_report(
        {"schema_version": 1, "no_training_performed": True}
    )

    assert observed == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "no_training_performed": True,
    }
    assert not output_path.with_suffix(".json.tmp").exists()
