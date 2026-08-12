"""A2-E1 原始 credential 到 protected MLP 的集成测试。"""

import torch

from can.access import A2AccessCoordinator
from can.model.a2_mlp import A2FashionMNISTMLP
from can.verifier import A1CompiledProfile
from can.verifier.a1_torch import A1TorchBackend


def test_a2_gate_preserves_business_label_and_blocks_rejected_relation(
    a2_gate_fixture: tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes],
) -> None:
    """完整入口应保持 allow 标签, 并在同一模型前硬拒绝无效 relation。"""
    _, backend, accepted, rejected = a2_gate_fixture
    torch.manual_seed(20_260_723)
    model = A2FashionMNISTMLP().eval()
    coordinator = A2AccessCoordinator(backend, model)
    images = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    with torch.inference_mode():
        expected = int(model(images).argmax(dim=1).item())

    allowed = coordinator.handle(images, accepted)
    denied = coordinator.handle(images, rejected)

    assert allowed == {"version": 1, "status": "ok", "class_id": expected}
    assert denied == {"version": 1, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.coordinator_commits == 2
    assert snapshot.protected_model_calls == 1
