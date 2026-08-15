"""V1-M1 CIFAR-style ResNet-18 单元测试。"""

import torch
from torch import nn

from can.model import (
    V1_M1_CLASS_COUNT,
    V1_M1_PARAMETER_COUNT,
    V1Cifar100ResNet18,
)


def test_v1_m1_resnet_has_the_frozen_cifar_topology() -> None:
    """模型必须保留 3x3 stem、四个双 block stage 和 100 类 classifier。"""
    model = V1Cifar100ResNet18()

    assert type(model.stem[0]) is nn.Conv2d
    assert model.stem[0].kernel_size == (3, 3)
    assert model.stem[0].stride == (1, 1)
    assert model.stem[0].bias is None
    layers = (model.layer1, model.layer2, model.layer3, model.layer4)
    assert tuple(len(layer) for layer in layers) == (2, 2, 2, 2)
    assert model.classifier.in_features == 512
    assert model.classifier.out_features == V1_M1_CLASS_COUNT
    assert sum(parameter.numel() for parameter in model.parameters()) == V1_M1_PARAMETER_COUNT


def test_v1_m1_resnet_returns_100_finite_float32_logits() -> None:
    """规范 float32 CIFAR batch 应生成有限的 100 类 logits。"""
    model = V1Cifar100ResNet18().eval()
    images = torch.zeros((2, 3, 32, 32), dtype=torch.float32)

    with torch.inference_mode():
        logits = model(images)

    assert tuple(logits.shape) == (2, V1_M1_CLASS_COUNT)
    assert logits.dtype is torch.float32
    assert logits.device.type == "cpu"
    assert bool(torch.isfinite(logits).all().item())
