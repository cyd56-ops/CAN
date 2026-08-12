"""A2-E1 Fashion-MNIST 受保护业务 MLP。"""

from typing import Final

import torch
from torch import nn

A2_EXPERIMENT_ID: Final = "CAN-A2-FMNIST-MLP-v1"
A2_INPUT_SHAPE: Final = (1, 28, 28)
A2_CLASS_COUNT: Final = 10
A2_PARAMETER_COUNT: Final = 235_146


class A2ModelInputError(ValueError):
    """表示 A2 业务模型输入不满足唯一规范表示。"""


def validate_a2_images(images: torch.Tensor) -> None:
    """严格验证 A2-E1 规范 float32 CPU 图像 batch。"""
    if type(images) is not torch.Tensor:
        raise A2ModelInputError("images must be exactly torch.Tensor")
    if images.dtype is not torch.float32:
        raise A2ModelInputError("images must use torch.float32")
    if images.device.type != "cpu" or images.device.index is not None:
        raise A2ModelInputError("images must remain on the CPU")
    if images.layout is not torch.strided or not images.is_contiguous():
        raise A2ModelInputError("images must use contiguous strided layout")
    if images.ndim != 4 or images.shape[0] < 1 or tuple(images.shape[1:]) != A2_INPUT_SHAPE:
        raise A2ModelInputError("images have the wrong batch shape")
    if not bool(torch.isfinite(images).all().item()):
        raise A2ModelInputError("images must contain only finite values")
    if float(images.min().item()) < 0.0 or float(images.max().item()) > 1.0:
        raise A2ModelInputError("images must stay in the closed interval [0,1]")


def validate_a2_labels(labels: torch.Tensor, batch_size: int) -> None:
    """严格验证与图像 batch 对齐的 A2-E1 标签。"""
    if type(batch_size) is not int or batch_size < 1:
        raise A2ModelInputError("batch_size must be a positive exact int")
    if type(labels) is not torch.Tensor:
        raise A2ModelInputError("labels must be exactly torch.Tensor")
    if labels.dtype is not torch.int64:
        raise A2ModelInputError("labels must use torch.int64")
    if labels.device.type != "cpu" or labels.device.index is not None:
        raise A2ModelInputError("labels must remain on the CPU")
    if labels.layout is not torch.strided or not labels.is_contiguous():
        raise A2ModelInputError("labels must use contiguous strided layout")
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise A2ModelInputError("labels have the wrong batch shape")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= A2_CLASS_COUNT:
        raise A2ModelInputError("labels must be canonical class indices")


class A2FashionMNISTMLP(nn.Module):
    """实现固定 `784->256->128->10` float32 CPU 业务模型。"""

    def __init__(self) -> None:
        super().__init__()
        self._network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(784, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, A2_CLASS_COUNT, bias=True),
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != A2_PARAMETER_COUNT:
            raise RuntimeError("A2-E1 model topology changed")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """验证规范图像 batch 并返回内部分类 logits。"""
        validate_a2_images(images)
        logits = self._network(images)
        if (
            type(logits) is not torch.Tensor
            or logits.dtype is not torch.float32
            or logits.device.type != "cpu"
            or tuple(logits.shape) != (images.shape[0], A2_CLASS_COUNT)
            or not bool(torch.isfinite(logits).all().item())
        ):
            raise RuntimeError("A2-E1 model produced non-canonical logits")
        return logits
