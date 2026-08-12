"""A2-E2 三态协调器的升级、fallback 与并发防御性测试。"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import can.access.a2_capability as capability_module
import can.reference.a0 as reference_module
import can.verifier.a1 as a1_module
from can.access import A2CapabilityCoordinator, A2CapabilityPolicy
from can.model.a2_mlp import A2FashionMNISTMLP
from can.model.a2_public_mlp import A2FashionMNISTPublicMLP
from can.verifier import A1CompiledProfile
from can.verifier.a1_torch import A1TorchBackend


def _images() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)


def _coordinator(backend: A1TorchBackend) -> A2CapabilityCoordinator:
    torch.manual_seed(20_260_729)
    protected_model = A2FashionMNISTMLP().eval()
    torch.manual_seed(20_260_730)
    public_model = A2FashionMNISTPublicMLP().eval()
    return A2CapabilityCoordinator(
        backend,
        protected_model,
        public_model=public_model,
        policy=A2CapabilityPolicy(public_entry_enabled=True),
    )


@pytest.mark.parametrize(
    "field",
    [
        "entry",
        "policy",
        "capability",
        "model",
        "head",
        "backend",
        "evidence",
        "decision",
        "allow",
        "status",
        "route",
        "fallback",
    ],
)
@pytest.mark.parametrize("entry", ["public", "protected"])
def test_request_fields_cannot_select_or_inject_a_capability_route(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    field: str,
    entry: str,
) -> None:
    """request payload 的路由、模型或 decision 字段必须统一 deny。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend)

    if entry == "public":
        response = coordinator.handle_public(_images(), **{field: "protected"})
    else:
        response = coordinator.handle_protected(_images(), accepted, **{field: "protected"})

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.public_model_calls == 0
    assert snapshot.protected_model_calls == 0


def test_public_response_cannot_be_reused_or_relabelled_as_protected_authority(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """public response 作为 credential/capability 重放时不能升级为 protected。"""
    _, backend, _, _ = a2_gate_fixture
    coordinator = _coordinator(backend)
    public_response = coordinator.handle_public(_images())

    credential_replay = coordinator.handle_protected(_images(), public_response)
    field_replay = coordinator.handle_protected(_images(), bytes(23), capability=public_response)

    assert public_response["status"] == "public"
    assert credential_replay == {"version": 2, "status": "deny"}
    assert field_replay == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.public_model_calls == 1
    assert snapshot.protected_model_calls == 0
    assert snapshot.protected_commits == 0


@pytest.mark.parametrize("case", ["wrong_type", "exception"])
def test_verifier_contract_failure_has_no_public_or_reference_fallback(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """伪造 evidence 或 verifier 异常不能转入 public/reference 路线。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend)
    fallback_calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("fallback was called")

    if case == "wrong_type":
        monkeypatch.setattr(capability_module, "verify_a1_torch", lambda *_args: object())
    else:
        monkeypatch.setattr(
            capability_module,
            "verify_a1_torch",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("forced verifier failure")),
        )
    monkeypatch.setattr(A2FashionMNISTPublicMLP, "forward", forbidden)
    monkeypatch.setattr(a1_module, "verify_a1", forbidden)
    monkeypatch.setattr(reference_module, "verify_ref", forbidden)

    response = coordinator.handle_protected(_images(), accepted)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.public_model_calls == 0
    assert snapshot.protected_model_calls == 0
    assert fallback_calls == 0


def test_public_model_failure_never_calls_protected_model(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """public model 异常必须 deny 且不能调用 protected fallback。"""
    _, backend, _, _ = a2_gate_fixture
    coordinator = _coordinator(backend)
    protected_calls = 0

    def public_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced public failure")

    def forbidden_protected(*args: object, **kwargs: object) -> object:
        nonlocal protected_calls
        protected_calls += 1
        raise AssertionError("protected fallback was called")

    monkeypatch.setattr(A2FashionMNISTPublicMLP, "forward", public_failure)
    monkeypatch.setattr(A2FashionMNISTMLP, "forward", forbidden_protected)

    response = coordinator.handle_public(_images())

    assert response == {"version": 2, "status": "deny"}
    assert coordinator.snapshot().public_model_calls == 1
    assert coordinator.snapshot().protected_model_calls == 0
    assert protected_calls == 0


def test_protected_model_failure_never_calls_public_model(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """protected model 异常必须 deny 且不能调用 public fallback。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend)
    public_calls = 0

    def protected_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced protected failure")

    def forbidden_public(*args: object, **kwargs: object) -> object:
        nonlocal public_calls
        public_calls += 1
        raise AssertionError("public fallback was called")

    monkeypatch.setattr(A2FashionMNISTMLP, "forward", protected_failure)
    monkeypatch.setattr(A2FashionMNISTPublicMLP, "forward", forbidden_public)

    response = coordinator.handle_protected(_images(), accepted)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.protected_model_calls == 1
    assert snapshot.public_model_calls == 0
    assert public_calls == 0


def test_concurrent_public_and_rejected_protected_requests_keep_exact_counts(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """并发 public/reject 请求必须保持互斥提交和线程安全计数。"""
    _, backend, _, rejected = a2_gate_fixture
    coordinator = _coordinator(backend)
    images = _images()

    def invoke(index: int) -> str:
        if index % 2 == 0:
            return coordinator.handle_public(images)["status"]
        return coordinator.handle_protected(images, rejected)["status"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = tuple(executor.map(invoke, range(32)))

    assert statuses.count("public") == 16
    assert statuses.count("deny") == 16
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 16
    assert snapshot.coordinator_commits == 32
    assert snapshot.public_commits == 16
    assert snapshot.deny_commits == 16
    assert snapshot.protected_commits == 0
    assert snapshot.public_model_calls == 16
    assert snapshot.protected_model_calls == 0


def test_concurrent_three_state_requests_keep_independent_exact_counts(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """并发 public、accepted protected 和 rejected protected 必须保持独立计数。"""
    _, backend, accepted, rejected = a2_gate_fixture
    coordinator = _coordinator(backend)
    images = _images()

    def invoke(index: int) -> str:
        route = index % 3
        if route == 0:
            return coordinator.handle_public(images)["status"]
        credential = accepted if route == 1 else rejected
        return coordinator.handle_protected(images, credential)["status"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        statuses = tuple(executor.map(invoke, range(36)))

    assert statuses.count("public") == 12
    assert statuses.count("protected") == 12
    assert statuses.count("deny") == 12
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 24
    assert snapshot.coordinator_commits == 36
    assert snapshot.public_commits == 12
    assert snapshot.protected_commits == 12
    assert snapshot.deny_commits == 12
    assert snapshot.public_model_calls == 12
    assert snapshot.protected_model_calls == 12


@pytest.mark.parametrize("entry", ["public", "protected", "deny"])
def test_external_envelopes_have_exact_fields_and_no_sensitive_details(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    entry: str,
) -> None:
    """三种外部结果只能包含固定字段, 不泄露模型或 verifier 细节。"""
    _, backend, accepted, rejected = a2_gate_fixture
    coordinator = _coordinator(backend)
    if entry == "public":
        response = coordinator.handle_public(_images())
        assert set(response) == {"version", "status", "coarse_class_id"}
    elif entry == "protected":
        response = coordinator.handle_protected(_images(), accepted)
        assert set(response) == {"version", "status", "class_id"}
    else:
        response = coordinator.handle_protected(_images(), rejected)
        assert set(response) == {"version", "status"}

    assert not (
        {
            "credential",
            "policy",
            "capability",
            "model",
            "backend",
            "evidence",
            "reason",
            "timing",
            "logits",
            "probabilities",
            "confidence",
            "features",
        }
        & set(response)
    )
