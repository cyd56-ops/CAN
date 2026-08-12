"""A2-E1 确定性 baseline helper 的单元测试。"""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

import can.experiments.a2_baseline as baseline
from can.model.a2_mlp import A2_EXPERIMENT_ID, A2FashionMNISTMLP


def _tiny_loader(*, shuffle: bool) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    images = torch.linspace(0.0, 1.0, steps=20 * 28 * 28, dtype=torch.float32).reshape(
        20, 1, 28, 28
    )
    labels = torch.arange(20, dtype=torch.int64) % 10
    generator = torch.Generator(device="cpu").manual_seed(baseline.TRAIN_LOADER_SEED)
    dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(images, labels))
    return DataLoader(
        dataset,
        batch_size=5,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def test_a2_split_indices_have_fixed_sizes_coverage_and_hashes() -> None:
    """固定 seed 必须产生协议规定的唯一 split。"""
    train, validation, train_hash, validation_hash = baseline._build_split_indices()

    assert train.shape == (55_000,)
    assert validation.shape == (5_000,)
    assert torch.unique(torch.cat((train, validation))).numel() == 60_000
    assert train_hash == "04812202f0f2671f8289fa0f9d3993fbbf3b16cc41321a04e0cb9d7975d20241"
    assert validation_hash == "3120ff0db03161b410bd7f8e0809b248e181755817b26df2377fede9490429c7"


def test_data_resource_validation_rejects_missing_extra_and_tampered_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """数据资源集合或强摘要变化必须拒绝。"""
    raw_root = tmp_path / "FashionMNIST" / "raw"
    raw_root.mkdir(parents=True)
    payload = b"fixed-test-resource"
    resource = baseline.A2DataResource(
        "resource.gz",
        hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(baseline, "A2_DATA_RESOURCES", (resource,))

    with pytest.raises(baseline.A2BaselineError):
        baseline._validate_data_resources(tmp_path)

    path = raw_root / resource.filename
    path.write_bytes(payload)
    observed = baseline._validate_data_resources(tmp_path)
    assert observed[resource.filename]["sha256"] == resource.sha256

    (raw_root / "extra.gz").write_bytes(b"extra")
    with pytest.raises(baseline.A2BaselineError):
        baseline._validate_data_resources(tmp_path)
    (raw_root / "extra.gz").unlink()

    path.write_bytes(payload + b"tamper")
    with pytest.raises(baseline.A2BaselineError):
        baseline._validate_data_resources(tmp_path)


def test_tiny_training_and_evaluation_are_reproducible() -> None:
    """同种子、同 loader 顺序应产生相同状态和分类指标。"""
    results: list[tuple[str, baseline.EvaluationMetrics]] = []
    original_thread_count = torch.get_num_threads()
    original_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for _ in range(2):
            torch.manual_seed(baseline.GLOBAL_SEED)
            model = A2FashionMNISTMLP()
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            baseline._train_epoch(model, _tiny_loader(shuffle=True), optimizer, criterion)
            metrics = baseline._evaluate(model, _tiny_loader(shuffle=False))
            results.append((baseline._hash_model_state(model), metrics))
    finally:
        torch.use_deterministic_algorithms(original_determinism)
        torch.set_num_threads(original_thread_count)

    assert results[0] == results[1]
    assert results[0][1].total == 20
    assert sum(results[0][1].per_class_count) == 20
    assert sum(sum(row) for row in results[0][1].confusion_matrix) == 20


def test_determinism_fingerprint_uses_canonical_json() -> None:
    """字段顺序不能改变确定性 fingerprint。"""
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}

    assert baseline._determinism_fingerprint(first) == baseline._determinism_fingerprint(second)
    assert baseline._determinism_fingerprint(first) != baseline._determinism_fingerprint({"a": 2})


def test_compare_repeats_requires_exact_report_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺失、畸形或不同 fingerprint 的重复报告必须拒绝。"""
    monkeypatch.setattr(baseline, "A2_REPORT_ROOT", tmp_path)
    report = {
        "schema_version": 1,
        "experiment_id": A2_EXPERIMENT_ID,
        "repeat": 1,
        "environment": {},
        "data": {},
        "training": {},
        "test": {},
        "model": {},
        "latency": {},
        "peak_rss_kib": 1,
        "determinism_fingerprint": "a" * 64,
    }
    (tmp_path / "baseline-repeat-1.json").write_text(json.dumps(report), encoding="utf-8")
    report["repeat"] = 2
    (tmp_path / "baseline-repeat-2.json").write_text(json.dumps(report), encoding="utf-8")

    assert baseline.compare_a2_repeats() == "a" * 64

    report["determinism_fingerprint"] = "b" * 64
    (tmp_path / "baseline-repeat-2.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(baseline.A2BaselineError):
        baseline.compare_a2_repeats()


@pytest.mark.parametrize("repeat", [True, 0, 3, "1"])
def test_run_rejects_noncanonical_repeat_before_side_effects(repeat: object) -> None:
    """repeat 类型混淆或越界必须在环境和数据访问前拒绝。"""
    with pytest.raises(baseline.A2BaselineError):
        baseline.run_a2_baseline(repeat)  # type: ignore[arg-type]
