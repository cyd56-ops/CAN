"""A2-E1 单一协调器与二元硬门控的单元测试。"""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest
import torch

from can.access import A2AccessCoordinator, A2CoordinatorConfigError
from can.model.a2_mlp import A2FashionMNISTMLP
from can.verifier import A1CompiledProfile
from can.verifier.a1_torch import A1TorchBackend


def _images(value: float = 0.25) -> torch.Tensor:
    return torch.full((1, 1, 28, 28), value, dtype=torch.float32)


def _coordinator(backend: A1TorchBackend) -> tuple[A2AccessCoordinator, A2FashionMNISTMLP]:
    torch.manual_seed(20_260_723)
    model = A2FashionMNISTMLP().eval()
    return A2AccessCoordinator(backend, model), model


def test_exact_numeric_accept_commits_once_then_calls_the_model_once(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """精确 accept evidence 才能提交 allow 并执行一次 protected forward。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator, model = _coordinator(backend)
    images = _images()
    with torch.inference_mode():
        expected = int(model(images).argmax(dim=1).item())

    response = coordinator.handle(images, accepted)

    assert response == {"version": 1, "status": "ok", "class_id": expected}
    assert coordinator.snapshot() == (
        coordinator.snapshot().__class__(
            verifier_calls=1,
            coordinator_commits=1,
            allow_commits=1,
            deny_commits=0,
            protected_model_calls=1,
            ok_responses=1,
            deny_responses=0,
        )
    )
    timing = coordinator.timing_snapshot()
    assert len(timing) == 1
    assert timing[0].response_ok
    assert timing[0].allow_committed
    assert timing[0].total_ns >= timing[0].protected_model_ns > 0


@pytest.mark.parametrize("credential_case", ["numeric", "parse", "profile", "type"])
def test_every_non_accept_evidence_has_one_fixed_zero_call_deny(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    credential_case: str,
) -> None:
    """所有非 accept evidence 必须映射为同一响应且不调用模型。"""
    _, backend, accepted, rejected = a2_gate_fixture
    credentials: dict[str, object] = {
        "numeric": rejected,
        "parse": accepted[:-1],
        "profile": accepted[:1] + b"\x00\x02" + accepted[3:],
        "type": bytearray(accepted),
    }
    coordinator, _ = _coordinator(backend)

    response = coordinator.handle(_images(), credentials[credential_case])

    assert response == {"version": 1, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.coordinator_commits == 1
    assert snapshot.deny_commits == 1
    assert snapshot.protected_model_calls == 0
    assert snapshot.deny_responses == 1


@pytest.mark.parametrize(
    "images",
    [
        None,
        True,
        torch.zeros((1, 1, 28, 28), dtype=torch.float64),
        torch.zeros((2, 1, 28, 28), dtype=torch.float32),
        torch.full((1, 1, 28, 28), float("nan"), dtype=torch.float32),
    ],
)
def test_invalid_public_images_deny_before_verification(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
    images: object,
) -> None:
    """非规范或多样本公共输入必须在 verifier 和模型之前拒绝。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator, _ = _coordinator(backend)

    response = coordinator.handle(images, accepted)

    assert response == {"version": 1, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.coordinator_commits == 1
    assert snapshot.protected_model_calls == 0


def test_replayed_accepted_credential_gets_fresh_decisions_without_capability_state(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """A0 合法重放应逐次重新验证, 不能创建可复用授权状态。"""
    _, backend, accepted, _ = a2_gate_fixture
    coordinator, _ = _coordinator(backend)

    responses = tuple(coordinator.handle(_images(), accepted) for _ in range(2))

    assert responses[0] == responses[1]
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 2
    assert snapshot.coordinator_commits == 2
    assert snapshot.allow_commits == 2
    assert snapshot.protected_model_calls == 2


def test_post_construction_model_drift_is_out_of_scope_and_not_rechecked(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """受保护模型只在构造时校验并冻结; 构造后由进程 owner 改动模型不再被每请求重校验。

    受保护模型属于可信不可变配置。按威胁模型 (AGENTS.md Threat-model alignment,
    SECURITY.md 白盒 owner 篡改自有模型不在防御范围), 协调器不在请求路径重复校验
    可信模型拓扑/状态。构造后主动改动模型属于 out-of-scope 的信任前提破坏, 协调器
    据此仍按 accept evidence 提交一次 protected forward, 而不是回退到每请求重校验兜底。
    构造时对非 eval 模型的 fail-closed 由
    test_coordinator_rejects_untrusted_or_noncanonical_local_configuration 覆盖。
    """
    _, backend, accepted, _ = a2_gate_fixture
    coordinator, model = _coordinator(backend)
    model.train()

    response = coordinator.handle(_images(), accepted)

    assert response["status"] == "ok"
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_model_calls == 1


def test_coordinator_rejects_untrusted_or_noncanonical_local_configuration(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """backend/model 必须由本地配置提供精确且已激活的固定类型。"""
    _, backend, _, _ = a2_gate_fixture
    with pytest.raises(A2CoordinatorConfigError):
        A2AccessCoordinator(cast(A1TorchBackend, object()), A2FashionMNISTMLP().eval())
    with pytest.raises(A2CoordinatorConfigError):
        A2AccessCoordinator(backend, A2FashionMNISTMLP())


def test_instrumentation_snapshot_is_immutable(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """调用计数快照不能反向修改协调器状态。"""
    _, backend, _, rejected = a2_gate_fixture
    coordinator, _ = _coordinator(backend)
    coordinator.handle(_images(), rejected)
    snapshot = coordinator.snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.deny_commits = 0  # type: ignore[misc]
    assert coordinator.snapshot().deny_commits == 1
