"""A2-E2 public/protected 三态协调器集成测试。"""

import torch

from can.access import (
    A2CapabilityCoordinator,
    A2CapabilityPolicy,
    A2CapabilitySnapshot,
)
from can.model.a2_mlp import A2FashionMNISTMLP
from can.model.a2_public_mlp import A2FashionMNISTPublicMLP
from can.verifier import A1CompiledProfile
from can.verifier.a1_torch import A1TorchBackend


def test_one_coordinator_commits_public_protected_and_deny_without_cross_calls(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """真实 backend 和两个模型必须形成互斥三态调用序列。"""
    _, backend, accepted, rejected = a2_gate_fixture
    torch.manual_seed(20_260_729)
    protected_model = A2FashionMNISTMLP().eval()
    torch.manual_seed(20_260_730)
    public_model = A2FashionMNISTPublicMLP().eval()
    coordinator = A2CapabilityCoordinator(
        backend,
        protected_model,
        public_model=public_model,
        policy=A2CapabilityPolicy(public_entry_enabled=True),
    )
    images = torch.linspace(0.0, 1.0, steps=28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)

    with torch.inference_mode():
        expected_public = int(public_model(images).argmax(dim=1).item())
        expected_protected = int(protected_model(images).argmax(dim=1).item())
    public_response = coordinator.handle_public(images)
    protected_response = coordinator.handle_protected(images, accepted)
    deny_response = coordinator.handle_protected(images, rejected)

    assert public_response == {
        "version": 2,
        "status": "public",
        "coarse_class_id": expected_public,
    }
    assert protected_response == {
        "version": 2,
        "status": "protected",
        "class_id": expected_protected,
    }
    assert deny_response == {"version": 2, "status": "deny"}
    assert coordinator.snapshot() == A2CapabilitySnapshot(
        verifier_calls=2,
        coordinator_commits=3,
        deny_commits=1,
        public_commits=1,
        protected_commits=1,
        public_model_calls=1,
        protected_model_calls=1,
        deny_responses=1,
        public_responses=1,
        protected_responses=1,
    )


def test_public_requests_do_not_change_protected_state_or_output(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """重复 public 调用不得改变 protected 参数、输出或调用计数。"""
    _, backend, accepted, _ = a2_gate_fixture
    torch.manual_seed(20_260_729)
    protected_model = A2FashionMNISTMLP().eval()
    torch.manual_seed(20_260_730)
    public_model = A2FashionMNISTPublicMLP().eval()
    coordinator = A2CapabilityCoordinator(
        backend,
        protected_model,
        public_model=public_model,
        policy=A2CapabilityPolicy(public_entry_enabled=True),
    )
    images = torch.linspace(0.0, 1.0, steps=28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    state_before = {
        name: parameter.detach().clone() for name, parameter in protected_model.state_dict().items()
    }
    with torch.inference_mode():
        output_before = protected_model(images).detach().clone()

    for _ in range(16):
        assert coordinator.handle_public(images)["status"] == "public"

    with torch.inference_mode():
        output_after = protected_model(images).detach().clone()
    assert torch.equal(output_after, output_before)
    assert all(
        torch.equal(parameter, state_before[name])
        for name, parameter in protected_model.state_dict().items()
    )
    assert coordinator.snapshot().protected_model_calls == 0
    assert coordinator.handle_protected(images, accepted)["status"] == "protected"
