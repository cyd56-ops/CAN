"""A2-E2 独立 public coarse MLP 的单元测试。"""

import pytest
import torch
from torch import nn

from can.model.a2_mlp import A2FashionMNISTMLP
from can.model.a2_public_mlp import (
    A2_PUBLIC_CLASS_COUNT,
    A2_PUBLIC_PARAMETER_COUNT,
    A2FashionMNISTPublicMLP,
    A2PublicModelInputError,
    map_a2_public_labels,
    validate_a2_public_images,
    validate_a2_public_labels,
)


def test_public_mlp_has_fixed_independent_topology_and_parameter_count() -> None:
    """public 模型必须保持唯一拓扑且不共享 protected 参数存储。"""
    torch.manual_seed(20_260_730)
    public_model = A2FashionMNISTPublicMLP()
    protected_model = A2FashionMNISTMLP()

    assert tuple(type(module) for module in public_model.modules()) == (
        A2FashionMNISTPublicMLP,
        nn.Sequential,
        nn.Flatten,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    )
    assert (
        sum(parameter.numel() for parameter in public_model.parameters())
        == A2_PUBLIC_PARAMETER_COUNT
    )
    assert all(parameter.dtype is torch.float32 for parameter in public_model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in public_model.parameters())
    public_storage = {parameter.data_ptr() for parameter in public_model.parameters()}
    protected_storage = {parameter.data_ptr() for parameter in protected_model.parameters()}
    assert public_storage.isdisjoint(protected_storage)


def test_public_mlp_returns_finite_two_class_logits() -> None:
    """规范图像 batch 应产生两个 public coarse logits。"""
    model = A2FashionMNISTPublicMLP().eval()
    images = torch.linspace(0.0, 1.0, steps=2 * 28 * 28, dtype=torch.float32).reshape(2, 1, 28, 28)

    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (2, A2_PUBLIC_CLASS_COUNT)
    assert logits.dtype is torch.float32
    assert logits.device.type == "cpu"
    assert logits.is_contiguous()
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
def test_public_images_fail_closed_on_noncanonical_values(images: object) -> None:
    """错误类型、shape、布局、有限性或范围必须拒绝。"""
    with pytest.raises(A2PublicModelInputError):
        validate_a2_public_images(images)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("labels", "batch_size"),
    [
        (None, 1),
        (torch.zeros(1, dtype=torch.int32), 1),
        (torch.zeros((1, 1), dtype=torch.int64), 1),
        (torch.zeros(2, dtype=torch.int64), 1),
        (torch.tensor([-1], dtype=torch.int64), 1),
        (torch.tensor([2], dtype=torch.int64), 1),
        (torch.tensor([0], dtype=torch.int64), True),
        (torch.tensor([0], dtype=torch.int64), 0),
    ],
)
def test_public_labels_fail_closed_on_noncanonical_values(
    labels: object, batch_size: object
) -> None:
    """public 标签类型、shape、范围和 bool/int 混淆必须拒绝。"""
    with pytest.raises(A2PublicModelInputError):
        validate_a2_public_labels(labels, batch_size)  # type: ignore[arg-type]


def test_source_labels_map_to_the_fixed_public_classes() -> None:
    """Fashion-MNIST 十类必须精确映射为固定 footwear 二分类。"""
    source = torch.arange(10, dtype=torch.int64)

    mapped = map_a2_public_labels(source)

    assert mapped.tolist() == [0, 0, 0, 0, 0, 1, 0, 1, 0, 1]
    assert mapped.dtype is torch.int64
    assert mapped.is_contiguous()
    assert mapped.data_ptr() != source.data_ptr()
    validate_a2_public_labels(mapped, 10)


@pytest.mark.parametrize(
    "source_labels",
    [
        None,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.arange(6, dtype=torch.int64)[::2],
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([10], dtype=torch.int64),
    ],
)
def test_source_label_mapping_rejects_noncanonical_values(source_labels: object) -> None:
    """source label 的类型、shape、布局和范围变化必须拒绝。"""
    with pytest.raises(A2PublicModelInputError):
        map_a2_public_labels(source_labels)  # type: ignore[arg-type]
