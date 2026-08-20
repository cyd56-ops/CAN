"""V1-M1-C2 accepted-state evaluator 的无训练报告集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from can.access import V1M1C2Cut, V1M1C2PublicHead
from can.experiments import v1_m1_c1 as c1
from can.experiments import v1_m1_c2 as training
from can.experiments import v1_m1_c2_evaluator as evaluator
from can.model import V1Cifar100ResNet18


def test_evaluator_composes_verified_artifacts_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """报告入口只组合 accepted states、路由证据和测量, 不调用训练。"""
    c2_root = tmp_path / "c2"
    c2_root.mkdir()
    model = V1Cifar100ResNet18().eval()
    head = V1M1C2PublicHead(512).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    accepted_r2 = c1.V1M1C1AcceptedR2(
        model=model,
        decoded_data_sha256="a" * 64,
        canonical_state_sha256=c1.V1_M1_C1_ACCEPTED_STATE_SHA256,
    )
    accepted_head = evaluator.V1M1C2AcceptedPublicHead(
        cut=V1M1C2Cut.LAYER4,
        accepted_run="H2",
        head=head,
        decoded_data_sha256="a" * 64,
        coarse_labels_sha256="b" * 64,
        state_sha256="c" * 64,
        state_file_sha256="d" * 64,
        manifest_sha256="e" * 64,
        report_sha256="f" * 64,
        training_report={
            "h1_candidates": [],
            "h2": {"cut": "layer4"},
            "accepted": {"run": "H2"},
            "test": {"top1_percent": 85.17},
        },
        metadata_correction_sha256="1" * 64,
    )
    data = evaluator._C2TestData(
        pixels=torch.zeros((1, 3, 32, 32), dtype=torch.uint8),
        fine_labels=torch.zeros(1, dtype=torch.int64),
        coarse_labels=torch.zeros(1, dtype=torch.int64),
    )
    _public_profile, neural_profile, _commitment = c1._build_public_conformance_material()

    monkeypatch.setattr(c1, "_validate_frozen_server_environment", lambda _device: None)
    monkeypatch.setattr(
        c1,
        "load_v1_m1_c1_accepted_r2_details",
        lambda _root, _device: accepted_r2,
    )
    monkeypatch.setattr(
        evaluator,
        "load_v1_m1_c2_accepted_public_head",
        lambda _root, _device: accepted_head,
    )
    monkeypatch.setattr(evaluator, "_load_c2_test_data", lambda *_args: data)
    monkeypatch.setattr(
        c1,
        "_build_public_conformance_material",
        lambda: (_public_profile, neural_profile, _commitment),
    )
    monkeypatch.setattr(evaluator, "_evaluate_routes", lambda *_args: {"status": "pass"})
    monkeypatch.setattr(evaluator, "_probe_fail_closed", lambda *_args: {"status": "pass"})
    monkeypatch.setattr(
        evaluator,
        "_probe_execution_failures",
        lambda *_args: {"status": "pass"},
    )
    monkeypatch.setattr(evaluator, "_measure_latency", lambda *_args: {"status": "pass"})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA RTX A4000")

    def forbidden_training(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("accepted-state evaluator must not train")

    monkeypatch.setattr(training, "_train_head", forbidden_training)

    report_path = evaluator.run_v1_m1_c2_evaluator(
        tmp_path / "data",
        tmp_path / "r2",
        c2_root,
        torch.device("cpu"),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["experiment_id"] == evaluator.V1_M1_C2_EVALUATOR_EXPERIMENT_ID
    assert report["public_head_artifact"]["cut"] == "layer4"
    assert report["public_head_artifact"]["contains_only_public_head_state"] is True
    assert report["routes"] == {"status": "pass"}
    assert report["scope"].startswith("No R2/public-head training")
