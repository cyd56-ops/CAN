"""A2-E2 三态协调器的单元测试。"""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from can.access import (
    A2_CAPABILITY_POLICY_VERSION,
    A2CapabilityConfigError,
    A2CapabilityCoordinator,
    A2CapabilityDecision,
    A2CapabilityPolicy,
    A2CapabilitySnapshot,
    A2CapabilityStartupEvent,
)
from can.model.a2_mlp import A2FashionMNISTMLP
from can.model.a2_public_mlp import A2FashionMNISTPublicMLP
from can.verifier import A1CompiledProfile
from can.verifier.a1_torch import A1TorchBackend


def _images() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)


def _models() -> tuple[A2FashionMNISTMLP, A2FashionMNISTPublicMLP]:
    torch.manual_seed(20_260_729)
    protected_model = A2FashionMNISTMLP().eval()
    torch.manual_seed(20_260_730)
    public_model = A2FashionMNISTPublicMLP().eval()
    return protected_model, public_model


def _coordinator(backend: A1TorchBackend, *, public_enabled: bool) -> A2CapabilityCoordinator:
    protected_model, public_model = _models()
    return A2CapabilityCoordinator(
        backend,
        protected_model,
        public_model=public_model if public_enabled else None,
        policy=A2CapabilityPolicy(public_entry_enabled=public_enabled),
    )


def test_default_policy_disables_public_entry_with_zero_model_calls(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """默认本地策略必须关闭 public entry 且不保留 public model。"""
    _, backend, _, _ = a2_gate_fixture
    protected_model, _ = _models()
    coordinator = A2CapabilityCoordinator(backend, protected_model)

    response = coordinator.handle_public(_images())

    assert response == {"version": 2, "status": "deny"}
    assert coordinator.startup_audit_event() == A2CapabilityStartupEvent(
        event_version=1,
        policy_version=A2_CAPABILITY_POLICY_VERSION,
        event_code="PUBLIC_ENTRY_DISABLED",
        public_entry_enabled=False,
    )
    assert coordinator.snapshot() == A2CapabilitySnapshot(
        verifier_calls=0,
        coordinator_commits=1,
        deny_commits=1,
        public_commits=0,
        protected_commits=0,
        public_model_calls=0,
        protected_model_calls=0,
        deny_responses=1,
        public_responses=0,
        protected_responses=0,
    )


def test_enabled_public_entry_commits_once_and_calls_only_public_model(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """合法 public entry 只能提交 PUBLIC 并调用一次独立模型。"""
    _, backend, _, _ = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    response = coordinator.handle_public(_images())

    assert response["version"] == 2
    assert response["status"] == "public"
    assert type(response["coarse_class_id"]) is int
    snapshot = coordinator.snapshot()
    assert snapshot.coordinator_commits == 1
    assert snapshot.public_commits == 1
    assert snapshot.public_model_calls == 1
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_model_calls == 0
    assert snapshot.public_responses == 1


def test_accepted_protected_entry_commits_once_and_calls_only_protected_model(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """exact accept evidence 才能提交 PROTECTED 并调用一次 protected model。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    response = coordinator.handle_protected(_images(), accepted)

    assert response["version"] == 2
    assert response["status"] == "protected"
    assert type(response["class_id"]) is int
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.coordinator_commits == 1
    assert snapshot.protected_commits == 1
    assert snapshot.protected_model_calls == 1
    assert snapshot.public_model_calls == 0
    assert snapshot.protected_responses == 1


def test_rejected_protected_entry_never_downgrades_to_public(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """启用 public 后 protected reject 仍必须 deny 且两个模型零调用。"""
    _, backend, _, rejected = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    response = coordinator.handle_protected(_images(), rejected)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.deny_commits == 1
    assert snapshot.public_commits == 0
    assert snapshot.protected_commits == 0
    assert snapshot.public_model_calls == 0
    assert snapshot.protected_model_calls == 0


@pytest.mark.parametrize(
    "raw_credential",
    [None, b"", bytes(23), {"status": "public"}, [0] * 23, True],
)
def test_malformed_protected_credential_denies_without_any_model_call(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    raw_credential: object,
) -> None:
    """credential 解析、类型或 profile 失败必须统一 deny 且不降级。"""
    _, backend, _, _ = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    response = coordinator.handle_protected(_images(), raw_credential)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.deny_commits == 1
    assert snapshot.public_model_calls == 0
    assert snapshot.protected_model_calls == 0


@pytest.mark.parametrize(
    "images",
    [
        None,
        torch.zeros((2, 1, 28, 28), dtype=torch.float32),
        torch.zeros((1, 1, 28, 28), dtype=torch.float64),
        torch.full((1, 1, 28, 28), float("nan"), dtype=torch.float32),
        torch.full((1, 1, 28, 28), -0.01, dtype=torch.float32),
        torch.full((1, 1, 28, 28), 1.01, dtype=torch.float32),
    ],
)
@pytest.mark.parametrize("entry", ["public", "protected"])
def test_malformed_entry_images_deny_before_verifier_or_model(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    images: object,
    entry: str,
) -> None:
    """两种 entry 的非规范图像都必须在 verifier/model 前 deny。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    if entry == "public":
        response = coordinator.handle_public(images)
    else:
        response = coordinator.handle_protected(images, accepted)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.public_model_calls == 0
    assert snapshot.protected_model_calls == 0


def test_local_configuration_requires_exact_policy_and_model_binding(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """public 启用状态、模型存在性和 policy 类型必须精确匹配。"""
    _, backend, _, _ = a2_gate_fixture
    protected_model, public_model = _models()

    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityPolicy(public_entry_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(
            backend,
            protected_model,
            public_model=public_model,
            policy=A2CapabilityPolicy(),
        )
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(
            backend,
            protected_model,
            policy=A2CapabilityPolicy(public_entry_enabled=True),
        )
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(
            backend,
            protected_model,
            policy=object(),  # type: ignore[arg-type]
        )

    protected_model, _ = _models()
    monkeypatch.setattr(backend, "_active", False)
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(backend, protected_model)


def test_constructor_rejects_training_hooks_and_shared_parameter_storage(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """模型状态、hook 或 public/protected 参数共享必须在启动时拒绝。"""
    _, backend, _, _ = a2_gate_fixture
    protected_model, public_model = _models()
    public_model.train()
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(
            backend,
            protected_model,
            public_model=public_model,
            policy=A2CapabilityPolicy(public_entry_enabled=True),
        )

    protected_model, public_model = _models()
    public_model.register_forward_hook(lambda *_args: None)
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(
            backend,
            protected_model,
            public_model=public_model,
            policy=A2CapabilityPolicy(public_entry_enabled=True),
        )

    protected_model, public_model = _models()
    public_layer = public_model._network[1]
    protected_layer = protected_model._network[1]
    assert isinstance(public_layer, nn.Linear)
    assert isinstance(protected_layer, nn.Linear)
    public_layer.weight = nn.Parameter(protected_layer.weight[64:128])
    with pytest.raises(A2CapabilityConfigError):
        A2CapabilityCoordinator(
            backend,
            protected_model,
            public_model=public_model,
            policy=A2CapabilityPolicy(public_entry_enabled=True),
        )


@pytest.mark.parametrize("entry", ["public", "protected"])
@pytest.mark.parametrize(
    "invalid_logits",
    [
        object(),
        torch.zeros((1, 2), dtype=torch.float64),
        torch.full((1, 2), float("nan"), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
    ],
)
def test_noncanonical_model_output_denies_after_one_selected_model_call(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    invalid_logits: object,
) -> None:
    """协调器必须独立拒绝被替换 forward 产生的非规范 logits。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    if entry == "public":
        monkeypatch.setattr(A2FashionMNISTPublicMLP, "forward", lambda *_args: invalid_logits)
        response = coordinator.handle_public(_images())
    else:
        monkeypatch.setattr(A2FashionMNISTMLP, "forward", lambda *_args: invalid_logits)
        response = coordinator.handle_protected(_images(), accepted)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.coordinator_commits == 1
    assert snapshot.deny_responses == 1
    if entry == "public":
        assert snapshot.public_commits == 1
        assert snapshot.public_model_calls == 1
        assert snapshot.protected_model_calls == 0
    else:
        assert snapshot.protected_commits == 1
        assert snapshot.protected_model_calls == 1
        assert snapshot.public_model_calls == 0


@pytest.mark.parametrize("entry", ["public", "protected"])
def test_model_exception_returns_deny_without_second_commit_or_other_model(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    """提交后的 selected model 异常只能 deny, 不能二次提交或调用另一模型。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)

    if entry == "public":
        monkeypatch.setattr(
            A2FashionMNISTPublicMLP,
            "forward",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("forced public failure")),
        )
        response = coordinator.handle_public(_images())
    else:
        monkeypatch.setattr(
            A2FashionMNISTMLP,
            "forward",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("forced protected failure")),
        )
        response = coordinator.handle_protected(_images(), accepted)

    assert response == {"version": 2, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.coordinator_commits == 1
    assert snapshot.deny_commits == 0
    assert snapshot.deny_responses == 1
    if entry == "public":
        assert snapshot.public_commits == 1
        assert snapshot.public_model_calls == 1
        assert snapshot.protected_model_calls == 0
    else:
        assert snapshot.protected_commits == 1
        assert snapshot.protected_model_calls == 1
        assert snapshot.public_model_calls == 0


def test_runtime_policy_drift_fails_before_public_model_call(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """启动后 runtime policy 漂移必须 fail closed。

    policy 是运行时可变的授权开关, 协调器每请求重新校验; 模型属于可信不可变配置,
    只在构造时校验一次, 构造后 owner 改动模型不再被每请求重校验 (参见 AGENTS.md
    Threat-model alignment, SECURITY.md 白盒 owner 篡改自有模型不在防御范围)。
    """
    _, backend, _, _ = a2_gate_fixture
    unique_policy = A2CapabilityPolicy(public_entry_enabled=True)
    protected_model, public_model = _models()
    coordinator = A2CapabilityCoordinator(
        backend,
        protected_model,
        public_model=public_model,
        policy=unique_policy,
    )
    object.__setattr__(unique_policy, "public_entry_enabled", False)

    assert coordinator.handle_public(_images()) == {"version": 2, "status": "deny"}
    assert coordinator.snapshot().public_model_calls == 0


def test_snapshots_and_startup_event_are_immutable_and_timing_is_bounded(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """审计/计时快照不得回写协调器或携带请求内容。"""
    _, backend, _, rejected = a2_gate_fixture
    coordinator = _coordinator(backend, public_enabled=True)
    coordinator.handle_public(_images())
    coordinator.handle_protected(_images(), rejected)

    snapshot = coordinator.snapshot()
    startup_event = coordinator.startup_audit_event()
    timings = coordinator.timing_snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.public_model_calls = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        startup_event.public_entry_enabled = False  # type: ignore[misc]
    assert len(timings) == 2
    assert timings[0].entry == "public"
    assert timings[0].committed_decision is A2CapabilityDecision.PUBLIC
    assert timings[1].entry == "protected"
    assert timings[1].committed_decision is A2CapabilityDecision.DENY
    assert all(sample.total_ns >= 0 for sample in timings)
    assert not hasattr(timings[0], "credential")
    assert not hasattr(timings[0], "images")
