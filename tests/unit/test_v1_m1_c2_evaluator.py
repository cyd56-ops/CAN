"""V1-M1-C2 accepted-state evaluator 与 metadata correction 单元测试。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from can.access import V1M1C2Cut, V1M1C2PublicHead
from can.access.v1_m1_adapter import normalize_v1_m1_uint8_batch
from can.experiments import v1_m1_baseline as baseline
from can.experiments import v1_m1_c1 as c1
from can.experiments import v1_m1_c2 as training
from can.experiments import v1_m1_c2_evaluator as evaluator
from can.model import V1Cifar100ResNet18


def _metrics(top1_percent: float, *, total: int = 10_000) -> training.V1M1C2HeadMetrics:
    correct = int(top1_percent * total / 100.0)
    return training.V1M1C2HeadMetrics(
        loss=1.0,
        top1_percent=correct * 100.0 / total,
        correct_top1=correct,
        total=total,
        predictions_sha256="a" * 64,
    )


def _run(
    run_name: training.V1M1C2RunName,
    cut: V1M1C2Cut,
    top1_percent: float,
) -> training.V1M1C2HeadRunResult:
    config = training.V1M1C2HeadTrainingConfig(
        run_name,
        cut,
        training.V1_M1_C2_RUN_SEEDS[run_name],
    )
    state = {
        "classifier.weight": torch.zeros((20, cut.channels), dtype=torch.float32),
        "classifier.bias": torch.zeros(20, dtype=torch.float32),
    }
    validation = _metrics(top1_percent, total=100)
    return training.V1M1C2HeadRunResult(
        config=config,
        selected_epoch=1,
        validation=validation,
        epochs=(training.V1M1C2EpochMetrics(1, validation),),
        state_sha256=training._head_state_digest(state),
        state=state,
    )


def _write_artifact(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    r2_binding: str,
) -> None:
    h1_candidates = (
        _run("H1", V1M1C2Cut.LAYER2, 70.0),
        _run("H1", V1M1C2Cut.LAYER3, 80.0),
        _run("H1", V1M1C2Cut.LAYER4, 82.0),
    )
    h2 = _run("H2", V1M1C2Cut.LAYER3, 81.0)
    archive = baseline.V1M1ArchiveManifest(
        filename=baseline.V1_M1_ARCHIVE_FILENAME,
        byte_size=baseline.V1_M1_ARCHIVE_SIZE,
        sha256=baseline.V1_M1_ARCHIVE_SHA256,
        md5=baseline.V1_M1_ARCHIVE_MD5,
    )
    empty_dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], [])
    empty_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(empty_dataset)
    data = training._V1M1C2DataBundle(
        archive=archive,
        decoded_sha256="c" * 64,
        coarse_labels_sha256="d" * 64,
        train_loader=empty_loader,
        validation_loader=empty_loader,
        test_loader=empty_loader,
    )
    monkeypatch.setattr(
        baseline,
        "_environment_report",
        lambda _device: {
            "platform": "Linux-test",
            "python": "3.11.9",
            "torch": "2.13.0+cu126",
            "cuda_runtime": "12.6",
            "device": "cuda:0",
            "device_name": "NVIDIA RTX A4000",
            "python_hash_seed": "1730",
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    )
    training._write_c2_artifacts(
        training._artifact_paths(root),
        h2,
        h1_candidates,
        h2,
        _metrics(85.17),
        data,
        r2_binding,
        torch.device("cpu"),
    )


def _accepted_for_model(
    model: V1Cifar100ResNet18,
    cut: V1M1C2Cut = V1M1C2Cut.LAYER4,
) -> evaluator.V1M1C2AcceptedPublicHead:
    head = V1M1C2PublicHead(cut.channels).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return evaluator.V1M1C2AcceptedPublicHead(
        cut=cut,
        accepted_run="H2",
        head=head,
        decoded_data_sha256="a" * 64,
        coarse_labels_sha256="b" * 64,
        state_sha256="c" * 64,
        state_file_sha256="d" * 64,
        manifest_sha256="e" * 64,
        report_sha256="f" * 64,
        training_report={},
        metadata_correction_sha256=None,
    )


def test_c1_accepted_details_keep_data_and_state_digests_distinct() -> None:
    """显式 accepted-R2 结果不能再把 decoded data digest 当作 state digest。"""
    model = V1Cifar100ResNet18()
    accepted = c1.V1M1C1AcceptedR2(
        model=model,
        decoded_data_sha256="a" * 64,
        canonical_state_sha256="b" * 64,
    )

    assert accepted.decoded_data_sha256 == "a" * 64
    assert accepted.canonical_state_sha256 == "b" * 64
    assert accepted.decoded_data_sha256 != accepted.canonical_state_sha256


def test_legacy_c2_artifact_requires_bound_metadata_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 runner 的误标字段必须 fail closed, 追加 correction 后才可加载。"""
    root = tmp_path / "c2"
    _write_artifact(root, monkeypatch, r2_binding="c" * 64)

    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="correction"):
        evaluator.load_v1_m1_c2_accepted_public_head(root, torch.device("cpu"))

    correction_path = evaluator.materialize_v1_m1_c2_metadata_correction(root)
    accepted = evaluator.load_v1_m1_c2_accepted_public_head(root, torch.device("cpu"))

    assert correction_path.name == evaluator.V1_M1_C2_CORRECTION_FILENAME
    assert accepted.cut is V1M1C2Cut.LAYER3
    assert accepted.accepted_run == "H2"
    assert accepted.metadata_correction_sha256 is not None
    assert set(accepted.head.state_dict()) == {"classifier.weight", "classifier.bias"}


def test_corrected_runner_binding_needs_no_metadata_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """修复后的 runner 直接绑定 accepted R2 state, 不生成 correction。"""
    root = tmp_path / "c2"
    _write_artifact(root, monkeypatch, r2_binding=c1.V1_M1_C1_ACCEPTED_STATE_SHA256)

    assert evaluator.prepare_v1_m1_c2_accepted_artifact(root) is None
    accepted = evaluator.load_v1_m1_c2_accepted_public_head(root, torch.device("cpu"))
    assert accepted.metadata_correction_sha256 is None
    assert not (root / evaluator.V1_M1_C2_CORRECTION_FILENAME).exists()


def test_existing_accepted_report_fails_before_evaluation(tmp_path: Path) -> None:
    """已有正式 report 时入口在任何昂贵推理前拒绝覆盖。"""
    root = tmp_path / "c2"
    root.mkdir()
    report_path = root / evaluator.V1_M1_C2_ACCEPTED_REPORT_FILENAME
    report_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="overwrite"):
        evaluator._accepted_report_path(root)


def test_route_evaluator_records_public_utility_and_exact_protected_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """小型本地样本同时验收 public utility 与 direct/split bitwise equality。"""
    torch.manual_seed(7)
    model = V1Cifar100ResNet18().eval()
    accepted = _accepted_for_model(model)
    pixels = torch.stack(
        (
            torch.zeros((3, 32, 32), dtype=torch.uint8),
            torch.full((3, 32, 32), 255, dtype=torch.uint8),
        )
    )
    with torch.inference_mode():
        normalized = normalize_v1_m1_uint8_batch(pixels)
        fine = model(normalized).argmax(dim=1).to(torch.int64)
        features = model.layer4(model.layer3(model.layer2(model.layer1(model.stem(normalized)))))
        coarse = accepted.head(features).argmax(dim=1).to(torch.int64)
    data = evaluator._C2TestData(pixels, fine, coarse)
    _public_profile, neural_profile, _commitment = c1._build_public_conformance_material()
    monkeypatch.setattr(baseline, "V1_M1_TEST_SIZE", 2)
    monkeypatch.setattr(baseline, "V1_M1_EVALUATION_BATCH_SIZE", 1)
    monkeypatch.setattr(
        c1,
        "_evaluate_direct_r2_baseline_reference",
        lambda _model, _device, _pixels: (
            "0" * 64,
            c1.V1_M1_C1_ACCEPTED_PREDICTIONS_SHA256,
        ),
    )

    result = evaluator._evaluate_routes(
        model,
        accepted,
        torch.device("cpu"),
        data,
        neural_profile,
    )

    public = cast(dict[str, object], result["public"])
    protected = cast(dict[str, object], result["protected"])
    calls = cast(dict[str, object], result["call_matrix"])
    assert public["top1_percent"] == 100.0
    assert protected["top1_percent"] == 100.0
    assert protected["bitwise_logits_equal"] is True
    assert protected["max_absolute_error"] == 0.0
    assert cast(dict[str, object], calls["public_success"])["protected_suffix"] == 0
    assert cast(dict[str, object], calls["protected_success"])["public_head"] == 0


def test_fail_closed_and_execution_failure_probes_cover_runtime_matrix() -> None:
    """正式 probes 覆盖 tamper/replay/expiry/abort/concurrency 与 post-commit 错误。"""
    torch.manual_seed(11)
    model = V1Cifar100ResNet18().eval()
    accepted = _accepted_for_model(model, V1M1C2Cut.LAYER2)
    _public_profile, neural_profile, _commitment = c1._build_public_conformance_material()
    image = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)

    fail_closed = evaluator._probe_fail_closed(model, accepted, image, neural_profile)
    failures = evaluator._probe_execution_failures(model, accepted, image, neural_profile)

    assert cast(dict[str, object], fail_closed["canonical_relation_tamper"])["verifier"] == 1
    assert cast(dict[str, object], fail_closed["malformed_response"])["verifier"] == 0
    assert cast(dict[str, object], fail_closed["cross_input_transcript_confusion"])["verifier"] == 0
    concurrent = cast(dict[str, object], fail_closed["concurrent_duplicate_response"])
    assert concurrent["protected_responses"] == 1
    assert concurrent["deny_responses"] == 31
    expected_counts = {
        "public_head": (1, 1, 0),
        "preprocessing": (0, 0, 0),
        "prefix": (1, 0, 0),
        "suffix": (1, 0, 1),
        "extraction": (1, 0, 1),
    }
    for stage, (prefix, public_head, protected_suffix) in expected_counts.items():
        assert cast(dict[str, object], failures[stage])["external_status"] == "deny"
        assert cast(dict[str, object], failures[stage])["prefix"] == prefix
        assert cast(dict[str, object], failures[stage])["public_head"] == public_head
        assert cast(dict[str, object], failures[stage])["protected_suffix"] == protected_suffix
