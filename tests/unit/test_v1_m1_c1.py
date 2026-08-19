"""V1-M1-C1 accepted-state evaluator 的本地确定性单元测试。"""

from pathlib import Path
from typing import cast

import pytest
import torch

from can.access import AuthenticatedR2
from can.experiments import v1_m1_c1
from can.model import V1Cifar100ResNet18
from can.reference import V1_PROFILE_ID, V1Challenge, V1Response
from can.verifier import verify_v1_neural


def _model() -> V1Cifar100ResNet18:
    torch.manual_seed(0)
    return V1Cifar100ResNet18().eval()


def _image() -> torch.Tensor:
    return (
        torch.arange(3 * 32 * 32, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, 3, 32, 32)
    )


def test_public_conformance_material_is_neural_accepting() -> None:
    """C1 evaluator 的公开 fixture 必须通过固定 neural relation。"""
    _public_profile, neural_profile, commitment = v1_m1_c1._build_public_conformance_material()
    challenge = V1Challenge(
        V1_PROFILE_ID,
        v1_m1_c1.V1_M1_C1_CHALLENGE,
    ).encode()
    response = V1Response(bytes(32), v1_m1_c1.V1_M1_C1_RESPONSE).encode()

    evidence = verify_v1_neural(
        commitment,
        challenge,
        response,
        bytes(32),
        neural_profile,
    )

    assert evidence.code.value == "neural_accept"


def test_reject_probes_never_call_r2() -> None:
    """tamper/replay/expiry/abort/route confusion 均不得调用 protected R2。"""
    _public_profile, neural_profile, commitment = v1_m1_c1._build_public_conformance_material()
    result = v1_m1_c1._probe_reject_isolation(
        _model(),
        _image(),
        neural_profile,
        commitment,
    )

    assert result["tamper"] == {"r2_calls": 0, "gate_calls": 1}
    replay = cast(dict[str, object], result["replay"])
    expiry = cast(dict[str, object], result["expiry"])
    abort = cast(dict[str, object], result["abort_retry_exhaustion"])
    assert replay["replay_additional_r2_calls"] == 0
    assert expiry["r2_calls"] == 0
    assert abort["r2_calls"] == 0
    assert result["route_confusion"] == {"r2_calls": 0, "gate_calls": 0}


def test_fresh_public_conformance_credentials_allow_on_one_coordinator() -> None:
    """每个 fresh public credential 都能通过同一个 C1 coordinator。"""
    public_profile, neural_profile, _commitment = v1_m1_c1._build_public_conformance_material()
    authenticated = AuthenticatedR2(
        neural_profile,
        _model(),
        challenge_sampler=v1_m1_c1._fixed_challenge_sampler,
    )
    for index in range(3):
        response_polynomials = v1_m1_c1._conformance_response(index)
        commitment = v1_m1_c1._build_conformance_commitment(
            public_profile,
            response_polynomials,
        )
        issued = authenticated.begin(_image(), commitment)
        response = v1_m1_c1._response_for_issue(
            cast(dict[str, object], issued),
            response_polynomials,
        )

        assert authenticated.respond(response) == {"version": 4, "status": "protected"}

    snapshot = authenticated.snapshot()
    assert snapshot.allow_commits == 3
    assert snapshot.protected_calls == 3


def test_loader_requires_explicit_cuda_zero(tmp_path: Path) -> None:
    """loader 不接受 CPU 或隐式 CUDA device。"""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    with pytest.raises(v1_m1_c1.V1M1C1EvaluatorError, match="explicit cuda:0"):
        v1_m1_c1.load_v1_m1_c1_accepted_r2(artifact_root, torch.device("cpu"))


def test_report_writer_refuses_overwrite_and_symlink(tmp_path: Path) -> None:
    """C1 report 只允许创建一次且不能跟随 symlink。"""
    path = tmp_path / "report.json"
    v1_m1_c1._atomic_write_report(path, {"schema_version": 1})
    with pytest.raises(v1_m1_c1.V1M1C1EvaluatorError, match="overwrite"):
        v1_m1_c1._atomic_write_report(path, {"schema_version": 1})

    target = tmp_path / "target.json"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(v1_m1_c1.V1M1C1EvaluatorError, match="overwrite"):
        v1_m1_c1._atomic_write_report(link, {"schema_version": 1})


def test_report_path_does_not_create_output_before_evaluation(tmp_path: Path) -> None:
    """preflight 取 report path 时不得提前创建 ignored 输出目录。"""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    path = v1_m1_c1._report_path(artifact_root)

    assert path == artifact_root / "c1" / "accepted-r2-report.json"
    assert not path.parent.exists()


def test_timing_summary_has_fixed_percentile_contract() -> None:
    """latency summary 输出固定 samples/median/p95 字段。"""
    assert v1_m1_c1._timing_summary([30, 10, 20]) == {
        "samples": 3,
        "median_ns": 20,
        "p95_ns": 30,
    }
