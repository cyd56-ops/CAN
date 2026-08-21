"""V1-M1-C1 accepted-state evaluator 的本地确定性单元测试。"""

import hashlib
from pathlib import Path
from typing import cast

import pytest
import torch
import torchvision  # type: ignore[import-untyped]

from can.access import AuthenticatedR2
from can.experiments import v1_m1_baseline, v1_m1_c1
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


def test_direct_reference_reuses_baseline_batch_and_prediction_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """baseline reference 必须保持 batch-256、top-5 首项和 int64 摘要语义。"""
    monkeypatch.setattr(v1_m1_baseline, "V1_M1_EVALUATION_BATCH_SIZE", 2)
    pixels = _image().repeat(3, 1, 1, 1)
    model = _model()

    logits_sha256, predictions_sha256 = v1_m1_c1._evaluate_direct_r2_baseline_reference(
        model,
        torch.device("cpu"),
        pixels,
    )

    logits = torch.cat(
        [
            v1_m1_c1._direct_logits(model, pixels[0:2], torch.device("cpu")),
            v1_m1_c1._direct_logits(model, pixels[2:3], torch.device("cpu")),
        ]
    ).contiguous()
    predictions = [int(value) for value in torch.topk(logits, k=5, dim=1).indices[:, 0].tolist()]
    assert logits_sha256 == hashlib.sha256(logits.numpy().tobytes()).hexdigest()
    assert predictions_sha256 == v1_m1_baseline._hash_predictions(predictions)


def test_baseline_digest_mismatch_stops_before_gate_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """baseline 摘要失败必须在创建 Gate Layer 和逐张循环前终止。"""
    _public_profile, neural_profile, _commitment = v1_m1_c1._build_public_conformance_material()
    monkeypatch.setattr(
        v1_m1_c1,
        "_evaluate_direct_r2_baseline_reference",
        lambda _model, _device, _pixels: ("0" * 64, "0" * 64),
    )

    def forbidden_gate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Gate Layer must not be created after a baseline digest mismatch")

    monkeypatch.setattr(v1_m1_c1, "AuthenticatedR2", forbidden_gate)

    with pytest.raises(v1_m1_c1.V1M1C1EvaluatorError, match="accepted R2 reference"):
        v1_m1_c1._evaluate_equivalence(
            _model(),
            torch.device("cpu"),
            _image(),
            neural_profile,
        )


def test_server_environment_requires_and_applies_r2_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1 必须拒绝缺失的 R2 env, 并在接受后应用同一 deterministic policy。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA RTX A4000",
    )
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu126")
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torchvision, "__version__", "0.28.0+cu126")
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(v1_m1_c1.V1M1C1EvaluatorError, match="environment variables"):
        v1_m1_c1._validate_frozen_server_environment(torch.device("cuda:0"))

    observed: list[int] = []
    monkeypatch.setenv("PYTHONHASHSEED", "1730")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        v1_m1_baseline,
        "_configure_v1_m1_determinism",
        lambda seed: observed.append(seed),
    )

    v1_m1_c1._validate_frozen_server_environment(torch.device("cuda:0"))

    assert observed == [1730]


def test_loaded_r2_state_is_frozen_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """accepted state 通过摘要核验后必须变为不可训练模型。"""
    model = _model()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(
        v1_m1_baseline,
        "_hash_model_state",
        lambda _model: v1_m1_c1.V1_M1_C1_ACCEPTED_STATE_SHA256,
    )

    v1_m1_c1._validate_loaded_state(model, state)

    assert all(not parameter.requires_grad for parameter in model.parameters())


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
