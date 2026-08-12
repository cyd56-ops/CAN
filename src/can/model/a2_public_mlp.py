"""A2-E2 Fashion-MNIST 独立 public coarse MLP。"""

from typing import Final

import torch
from torch import nn

A2_PUBLIC_EXPERIMENT_ID: Final = "CAN-A2-FMNIST-PUBLIC-MLP-v1"
A2_PUBLIC_INPUT_SHAPE: Final = (1, 28, 28)
A2_PUBLIC_CLASS_COUNT: Final = 2
A2_PUBLIC_SOURCE_CLASS_COUNT: Final = 10
A2_PUBLIC_PARAMETER_COUNT: Final = 50_370
A2_PUBLIC_CLASS_NAMES: Final = ("NON_FOOTWEAR", "FOOTWEAR")
A2_PUBLIC_SOURCE_TO_COARSE: Final = (0, 0, 0, 0, 0, 1, 0, 1, 0, 1)
A2_PUBLIC_TEST_CLASS_SUPPORT: Final = (7_000, 3_000)


class A2PublicModelInputError(ValueError):
    """表示 A2-E2 public 模型输入不满足唯一规范表示。"""


def validate_a2_public_images(images: torch.Tensor) -> None:
    """严格验证 A2-E2 public float32 CPU 图像 batch。"""
    if type(images) is not torch.Tensor:
        raise A2PublicModelInputError("images must be exactly torch.Tensor")
    if images.dtype is not torch.float32:
        raise A2PublicModelInputError("images must use torch.float32")
    if images.device.type != "cpu" or images.device.index is not None:
        raise A2PublicModelInputError("images must remain on the CPU")
    if images.layout is not torch.strided or not images.is_contiguous():
        raise A2PublicModelInputError("images must use contiguous strided layout")
    if images.ndim != 4 or images.shape[0] < 1 or tuple(images.shape[1:]) != A2_PUBLIC_INPUT_SHAPE:
        raise A2PublicModelInputError("images have the wrong batch shape")
    if not bool(torch.isfinite(images).all().item()):
        raise A2PublicModelInputError("images must contain only finite values")
    if float(images.min().item()) < 0.0 or float(images.max().item()) > 1.0:
        raise A2PublicModelInputError("images must stay in the closed interval [0,1]")


def validate_a2_public_labels(labels: torch.Tensor, batch_size: int) -> None:
    """严格验证与图像 batch 对齐的二类 public 标签。"""
    if type(batch_size) is not int or batch_size < 1:
        raise A2PublicModelInputError("batch_size must be a positive exact int")
    if type(labels) is not torch.Tensor:
        raise A2PublicModelInputError("labels must be exactly torch.Tensor")
    if labels.dtype is not torch.int64:
        raise A2PublicModelInputError("labels must use torch.int64")
    if labels.device.type != "cpu" or labels.device.index is not None:
        raise A2PublicModelInputError("labels must remain on the CPU")
    if labels.layout is not torch.strided or not labels.is_contiguous():
        raise A2PublicModelInputError("labels must use contiguous strided layout")
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise A2PublicModelInputError("labels have the wrong batch shape")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= A2_PUBLIC_CLASS_COUNT:
        raise A2PublicModelInputError("labels must be canonical public class indices")


def map_a2_public_labels(source_labels: torch.Tensor) -> torch.Tensor:
    """把规范 Fashion-MNIST 十类标签映射为固定 public 二类标签。"""
    if type(source_labels) is not torch.Tensor:
        raise A2PublicModelInputError("source labels must be exactly torch.Tensor")
    if source_labels.dtype is not torch.int64:
        raise A2PublicModelInputError("source labels must use torch.int64")
    if source_labels.device.type != "cpu" or source_labels.device.index is not None:
        raise A2PublicModelInputError("source labels must remain on the CPU")
    if source_labels.layout is not torch.strided or not source_labels.is_contiguous():
        raise A2PublicModelInputError("source labels must use contiguous strided layout")
    if source_labels.ndim != 1 or source_labels.shape[0] < 1:
        raise A2PublicModelInputError("source labels have the wrong batch shape")
    if (
        int(source_labels.min().item()) < 0
        or int(source_labels.max().item()) >= A2_PUBLIC_SOURCE_CLASS_COUNT
    ):
        raise A2PublicModelInputError("source labels must be canonical Fashion-MNIST indices")

    footwear = (source_labels == 5) | (source_labels == 7) | (source_labels == 9)
    labels = footwear.to(dtype=torch.int64).contiguous()
    validate_a2_public_labels(labels, source_labels.shape[0])
    return labels


class A2FashionMNISTPublicMLP(nn.Module):
    """实现固定 `784->64->2` float32 CPU public 模型。"""

    def __init__(self) -> None:
        super().__init__()
        self._network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(784, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, A2_PUBLIC_CLASS_COUNT, bias=True),
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != A2_PUBLIC_PARAMETER_COUNT:
            raise RuntimeError("A2-E2 public model topology changed")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """验证规范图像 batch 并返回内部 public coarse logits。"""
        validate_a2_public_images(images)
        logits = self._network(images)
        if (
            type(logits) is not torch.Tensor
            or logits.dtype is not torch.float32
            or logits.device.type != "cpu"
            or logits.device.index is not None
            or logits.layout is not torch.strided
            or not logits.is_contiguous()
            or tuple(logits.shape) != (images.shape[0], A2_PUBLIC_CLASS_COUNT)
            or not bool(torch.isfinite(logits).all().item())
        ):
            raise RuntimeError("A2-E2 public model produced non-canonical logits")
        return logits
