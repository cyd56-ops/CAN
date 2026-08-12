"""A2-E2 public baseline helper 的单元测试。"""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

import can.experiments.a2_baseline as protected_baseline
import can.experiments.a2_public_baseline as public_baseline
from can.model.a2_public_mlp import (
    A2_PUBLIC_EXPERIMENT_ID,
    A2FashionMNISTPublicMLP,
)


def _tiny_loader(*, shuffle: bool) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    images = torch.linspace(0.0, 1.0, steps=20 * 28 * 28, dtype=torch.float32).reshape(
        20, 1, 28, 28
    )
    source_labels = torch.arange(20, dtype=torch.int64) % 10
    generator = torch.Generator(device="cpu").manual_seed(public_baseline.PUBLIC_TRAIN_LOADER_SEED)
    dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(images, source_labels))
    return DataLoader(
        dataset,
        batch_size=5,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def test_public_split_and_data_resources_match_the_frozen_a2_dataset() -> None:
    """public baseline 必须复用 A2-E1 的资源 identity 和 split。"""
    train, validation, train_hash, validation_hash = public_baseline._build_split_indices()

    assert train.shape == (55_000,)
    assert validation.shape == (5_000,)
    assert torch.unique(torch.cat((train, validation))).numel() == 60_000
    assert train_hash == "04812202f0f2671f8289fa0f9d3993fbbf3b16cc41321a04e0cb9d7975d20241"
    assert validation_hash == "3120ff0db03161b410bd7f8e0809b248e181755817b26df2377fede9490429c7"
    public_resources = tuple(
        (item.filename, item.md5, item.sha256) for item in public_baseline.A2_PUBLIC_DATA_RESOURCES
    )
    protected_resources = tuple(
        (item.filename, item.md5, item.sha256) for item in protected_baseline.A2_DATA_RESOURCES
    )
    assert public_resources == protected_resources


def test_public_data_validation_rejects_missing_extra_and_tampered_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """public 数据资源集合或强摘要变化必须拒绝。"""
    raw_root = tmp_path / "FashionMNIST" / "raw"
    raw_root.mkdir(parents=True)
    payload = b"fixed-public-test-resource"
    data_resource = public_baseline.A2PublicDataResource(
        "resource.gz",
        hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(public_baseline, "A2_PUBLIC_DATA_RESOURCES", (data_resource,))

    with pytest.raises(public_baseline.A2PublicBaselineError):
        public_baseline._validate_data_resources(tmp_path)

    path = raw_root / data_resource.filename
    path.write_bytes(payload)
    observed = public_baseline._validate_data_resources(tmp_path)
    assert observed[data_resource.filename]["sha256"] == data_resource.sha256

    (raw_root / "extra.gz").write_bytes(b"extra")
    with pytest.raises(public_baseline.A2PublicBaselineError):
        public_baseline._validate_data_resources(tmp_path)
    (raw_root / "extra.gz").unlink()

    path.write_bytes(payload + b"tamper")
    with pytest.raises(public_baseline.A2PublicBaselineError):
        public_baseline._validate_data_resources(tmp_path)


def test_tiny_public_training_and_evaluation_are_reproducible() -> None:
    """同种子、同 public loader 顺序应产生相同状态和二分类指标。"""
    results: list[tuple[str, public_baseline.PublicEvaluationMetrics]] = []
    original_thread_count = torch.get_num_threads()
    original_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for _ in range(2):
            torch.manual_seed(public_baseline.PUBLIC_GLOBAL_SEED)
            model = A2FashionMNISTPublicMLP()
            optimizer = Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            public_baseline._train_epoch(model, _tiny_loader(shuffle=True), optimizer, criterion)
            metrics = public_baseline._evaluate(model, _tiny_loader(shuffle=False))
            results.append((public_baseline._hash_model_state(model), metrics))
    finally:
        torch.use_deterministic_algorithms(original_determinism)
        torch.set_num_threads(original_thread_count)

    assert results[0] == results[1]
    assert results[0][1].total == 20
    assert results[0][1].per_class_count == (14, 6)
    assert sum(sum(row) for row in results[0][1].confusion_matrix) == 20


def test_public_determinism_fingerprint_uses_canonical_json() -> None:
    """字段顺序不能改变 public determinism fingerprint。"""
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}

    assert public_baseline._determinism_fingerprint(
        first
    ) == public_baseline._determinism_fingerprint(second)
    assert public_baseline._determinism_fingerprint(
        first
    ) != public_baseline._determinism_fingerprint({"a": 2})


def test_compare_public_repeats_requires_exact_report_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺失、畸形或不同 fingerprint 的 public 报告必须拒绝。"""
    monkeypatch.setattr(public_baseline, "A2_REPORT_ROOT", tmp_path)
    report = {
        "schema_version": 1,
        "experiment_id": A2_PUBLIC_EXPERIMENT_ID,
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
    (tmp_path / "public-baseline-repeat-1.json").write_text(json.dumps(report), encoding="utf-8")
    report["repeat"] = 2
    (tmp_path / "public-baseline-repeat-2.json").write_text(json.dumps(report), encoding="utf-8")

    assert public_baseline.compare_a2_public_repeats() == "a" * 64

    report["determinism_fingerprint"] = "b" * 64
    (tmp_path / "public-baseline-repeat-2.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(public_baseline.A2PublicBaselineError):
        public_baseline.compare_a2_public_repeats()


@pytest.mark.parametrize("repeat", [True, 0, 3, "1"])
def test_public_run_rejects_noncanonical_repeat_before_side_effects(repeat: object) -> None:
    """public repeat 类型混淆或越界必须在环境和数据访问前拒绝。"""
    with pytest.raises(public_baseline.A2PublicBaselineError):
        public_baseline.run_a2_public_baseline(repeat)  # type: ignore[arg-type]


def test_public_latency_rejects_an_empty_batch_sequence() -> None:
    """空 latency 输入必须 fail closed。"""
    with pytest.raises(public_baseline.A2PublicBaselineError):
        public_baseline._time_model(A2FashionMNISTPublicMLP(), ())
