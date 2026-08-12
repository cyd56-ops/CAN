"""A2-E1 协调器、响应与零 protected-call 边界的安全测试。"""

from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
import torch

import can.access.a2_gate as gate_module
import can.reference.a0 as reference_module
import can.verifier.a1 as a1_module
from can.access import A2AccessCoordinator
from can.model.a2_mlp import A2FashionMNISTMLP
from can.verifier import A1CompiledProfile, A1Evidence, A1EvidenceCode
from can.verifier.a1_torch import A1TorchBackend


def _images(value: float = 0.5) -> torch.Tensor:
    return torch.full((1, 1, 28, 28), value, dtype=torch.float32)


def _coordinator(backend: A1TorchBackend) -> A2AccessCoordinator:
    return A2AccessCoordinator(backend, A2FashionMNISTMLP().eval())


@pytest.mark.parametrize(
    "injected",
    [
        {"evidence": A1Evidence(A1EvidenceCode.NUMERIC_ACCEPT)},
        {"decision": "allow"},
        {"allow": True},
        {"backend": "dependency-free"},
        {"policy": {"threshold": 128}},
        {"model": "public"},
    ],
)
def test_requester_authority_and_route_injection_denies_before_verification(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    injected: dict[str, object],
) -> None:
    """请求方提交的 evidence、decision、策略或路线都必须无授权语义。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend)

    response = coordinator.handle(_images(), accepted, **injected)

    assert response == {"version": 1, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.deny_commits == 1
    assert snapshot.protected_model_calls == 0


@pytest.mark.parametrize(
    "credential",
    [None, True, bytearray(23), memoryview(bytes(23)), {"status": "numeric_accept"}],
)
def test_credential_type_confusion_returns_the_same_zero_call_envelope(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    credential: object,
) -> None:
    """非 exact bytes credential 必须经 parser 证据映射为统一 deny。"""
    _, backend, _, _ = a2_gate_fixture
    coordinator = _coordinator(backend)

    response = coordinator.handle(_images(), credential)

    assert response == {"version": 1, "status": "deny"}
    assert coordinator.snapshot().protected_model_calls == 0


@pytest.mark.parametrize("case", ["wrong_type", "exception"])
def test_verifier_contract_failure_has_no_fallback_or_model_call(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """伪造 evidence 类型或 verifier 异常必须 fail closed 且不走弱路线。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend)
    fallback_calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("fallback verifier was called")

    if case == "wrong_type":
        monkeypatch.setattr(gate_module, "verify_a1_torch", lambda *_args: object())
    else:
        monkeypatch.setattr(
            gate_module,
            "verify_a1_torch",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("forced verifier failure")),
        )
    monkeypatch.setattr(a1_module, "verify_a1", forbidden)
    monkeypatch.setattr(reference_module, "verify_ref", forbidden)

    response = coordinator.handle(_images(), accepted)

    assert response == {"version": 1, "status": "deny"}
    assert coordinator.snapshot().protected_model_calls == 0
    assert fallback_calls == 0


def test_backend_disabled_after_local_configuration_has_zero_model_calls(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """已配置 backend 后失活必须由固定 verifier 路径产生零调用 deny。"""
    _, shared_backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(shared_backend)
    shared_backend._disable()
    try:
        response = coordinator.handle(_images(), accepted)
    finally:
        shared_backend._active = True

    assert response == {"version": 1, "status": "deny"}
    assert coordinator.snapshot().protected_model_calls == 0


def test_allow_commit_precedes_the_only_protected_forward(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """protected forward 入口必须只能出现在已提交 allow 之后。"""
    _, backend, accepted, _ = a2_gate_fixture
    model = A2FashionMNISTMLP().eval()
    coordinator = A2AccessCoordinator(backend, model)
    original_forward = A2FashionMNISTMLP.forward

    def checked_forward(self: A2FashionMNISTMLP, images: torch.Tensor) -> torch.Tensor:
        snapshot = coordinator.snapshot()
        assert snapshot.allow_commits == 1
        assert snapshot.protected_model_calls == 1
        return original_forward(self, images)

    monkeypatch.setattr(A2FashionMNISTMLP, "forward", checked_forward)

    response = coordinator.handle(_images(), accepted)

    assert response["status"] == "ok"
    assert coordinator.snapshot().protected_model_calls == 1


def test_concurrent_rejected_replays_are_stable_and_have_zero_protected_calls(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """并发重复拒绝必须保持固定响应和线程安全的零调用计数。"""
    _, backend, _, rejected = a2_gate_fixture
    coordinator = _coordinator(backend)
    images = _images()

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = tuple(
            executor.map(lambda _index: coordinator.handle(images, rejected), range(32))
        )

    assert all(response == {"version": 1, "status": "deny"} for response in responses)
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 32
    assert snapshot.coordinator_commits == 32
    assert snapshot.deny_commits == 32
    assert snapshot.protected_model_calls == 0


def test_deny_envelope_never_exposes_internal_security_or_model_fields(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """拒绝响应只能包含固定 version/status 字段。"""
    _, backend, _, rejected = a2_gate_fixture
    response = _coordinator(backend).handle(_images(), rejected)

    assert set(response) == {"version", "status"}
    assert not (
        {
            "logits",
            "probabilities",
            "features",
            "evidence",
            "profile",
            "slot",
            "reason",
            "timing",
            "model",
            "capability",
        }
        & set(response)
    )


def test_missing_fields_and_claimed_evidence_as_credential_fail_closed(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """缺失字段或把 evidence 对象放入 credential 位置都不能获得权限。"""
    _, backend, _, _ = a2_gate_fixture
    coordinator = _coordinator(backend)

    responses = (
        coordinator.handle(),
        coordinator.handle(_images()),
        coordinator.handle(raw_credential=bytes(23)),
        coordinator.handle(_images(), cast(bytes, A1Evidence(A1EvidenceCode.NUMERIC_ACCEPT))),
    )

    assert all(response == {"version": 1, "status": "deny"} for response in responses)
    assert coordinator.snapshot().protected_model_calls == 0
