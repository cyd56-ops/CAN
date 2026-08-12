"""A2-E1 业务 MLP 的单元测试。"""

import pytest
import torch
from torch import nn

from can.model.a2_mlp import (
    A2_CLASS_COUNT,
    A2_PARAMETER_COUNT,
    A2FashionMNISTMLP,
    A2ModelInputError,
    validate_a2_images,
    validate_a2_labels,
)


def test_a2_mlp_has_the_fixed_topology_and_parameter_count() -> None:
    """业务模型必须保持唯一 MLP 拓扑和参数量。"""
    torch.manual_seed(20_260_723)
    model = A2FashionMNISTMLP()

    assert tuple(type(module) for module in model.modules()) == (
        A2FashionMNISTMLP,
        nn.Sequential,
        nn.Flatten,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == A2_PARAMETER_COUNT
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_a2_mlp_returns_finite_float32_logits() -> None:
    """规范图像 batch 应产生十类内部 logits。"""
    model = A2FashionMNISTMLP().eval()
    images = torch.linspace(0.0, 1.0, steps=2 * 28 * 28, dtype=torch.float32).reshape(2, 1, 28, 28)

    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (2, A2_CLASS_COUNT)
    assert logits.dtype is torch.float32
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "images",
    [
        None,
        torch.zeros((1, 1, 28, 28), dtype=torch.float64),
        torch.zeros((1, 28, 28), dtype=torch.float32),
        torch.zeros((0, 1, 28, 28), dtype=torch.float32),
        torch.zeros((1, 1, 27, 28), dtype=torch.float32),
        torch.zeros((1, 1, 28, 28), dtype=torch.float32).transpose(2, 3),
        torch.full((1, 1, 28, 28), float("nan"), dtype=torch.float32),
        torch.full((1, 1, 28, 28), float("inf"), dtype=torch.float32),
        torch.full((1, 1, 28, 28), -0.01, dtype=torch.float32),
        torch.full((1, 1, 28, 28), 1.01, dtype=torch.float32),
    ],
)
def test_a2_images_fail_closed_on_noncanonical_values(images: object) -> None:
    """错误类型、shape、布局、有限性或范围必须拒绝。"""
    with pytest.raises(A2ModelInputError):
        validate_a2_images(images)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("labels", "batch_size"),
    [
        (None, 1),
        (torch.zeros(1, dtype=torch.int32), 1),
        (torch.zeros((1, 1), dtype=torch.int64), 1),
        (torch.zeros(2, dtype=torch.int64), 1),
        (torch.tensor([-1], dtype=torch.int64), 1),
        (torch.tensor([10], dtype=torch.int64), 1),
        (torch.tensor([0], dtype=torch.int64), True),
        (torch.tensor([0], dtype=torch.int64), 0),
    ],
)
def test_a2_labels_fail_closed_on_noncanonical_values(labels: object, batch_size: object) -> None:
    """标签类型、shape、范围和 bool/int 混淆必须拒绝。"""
    with pytest.raises(A2ModelInputError):
        validate_a2_labels(labels, batch_size)  # type: ignore[arg-type]


def test_a2_labels_accept_canonical_class_indices() -> None:
    """规范 int64 一维标签应通过验证。"""
    validate_a2_labels(torch.tensor([0, 9], dtype=torch.int64), 2)
