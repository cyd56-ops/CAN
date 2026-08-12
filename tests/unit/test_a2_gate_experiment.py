"""A2-E1 硬门控实验 helper 的单元测试。"""

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

import can.experiments.a2_baseline as baseline
import can.experiments.a2_gate as gate_experiment
from can.model.a2_mlp import A2FashionMNISTMLP
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


def test_all_label_validation_matches_direct_model_predictions(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逐样本 gated top-1 必须与同一业务模型的直接结果完全一致。"""
    _, backend, accepted, _ = a2_gate_fixture
    dataset = _TinyImageDataset()
    model = A2FashionMNISTMLP().eval()
    with torch.inference_mode():
        predictions = [
            int(model(dataset[index][0].unsqueeze(0)).argmax(dim=1).item())
            for index in range(len(dataset))
        ]
    expected_hash = baseline._hash_int64_values(predictions)
    monkeypatch.setattr(baseline, "TEST_SIZE", len(dataset))

    observed_hash, snapshot = gate_experiment._validate_all_labels(
        model,
        dataset,
        backend,
        accepted,
        expected_hash,
    )

    assert observed_hash == expected_hash
    assert snapshot.verifier_calls == len(dataset)
    assert snapshot.coordinator_commits == len(dataset)
    assert snapshot.protected_model_calls == len(dataset)


@pytest.mark.parametrize("coefficients", [[], [0] * 7, [0] * 7 + [257], [0] * 7 + [True]])
def test_toy_credential_encoder_rejects_noncanonical_values(
    coefficients: list[object],
) -> None:
    """实验 helper 不能把错误长度、范围或 bool 编码成 credential。"""
    with pytest.raises(gate_experiment.A2GateExperimentError):
        gate_experiment._encode_credential(coefficients)  # type: ignore[arg-type]


def test_latency_statistics_use_nearest_rank_microseconds() -> None:
    """延迟统计应以固定 nearest-rank 规则报告微秒。"""
    values = list(range(1_000, 101_000, 1_000))

    observed = gate_experiment._latency_statistics(values)

    assert observed == {"median_us": 50.0, "p95_us": 95.0, "p99_us": 99.0}


def test_gate_report_writer_uses_only_the_fixed_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门控报告必须通过临时文件原子写入固定 ignored 路径。"""
    output_path = tmp_path / "a2" / "gate.json"
    monkeypatch.setattr(gate_experiment, "A2_GATE_REPORT_PATH", output_path)

    observed = gate_experiment._write_gate_report({"schema_version": 1})

    assert observed == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert not output_path.with_suffix(".json.tmp").exists()
