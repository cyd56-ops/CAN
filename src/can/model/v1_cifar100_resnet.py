"""V1-M1 CIFAR-100 的独立 CIFAR-style ResNet-18。"""

from __future__ import annotations

from typing import Final, cast

from torch import Tensor, nn

V1_M1_MODEL_PROFILE_ID: Final = "CAN-V1-CIFAR100-RESNET18-v1"
V1_M1_INPUT_SHAPE: Final = (3, 32, 32)
V1_M1_CLASS_COUNT: Final = 100
V1_M1_PARAMETER_COUNT: Final = 11_220_132


class _V1M1BasicBlock(nn.Module):
    """实现 CIFAR-style ResNet-18 的单个 BasicBlock。"""

    expansion: Final = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, images: Tensor) -> Tensor:
        """计算一个固定 residual block 的输出。"""
        residual = images
        output = self.relu(self.bn1(self.conv1(images)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            residual = self.downsample(images)
        return cast(Tensor, self.relu(output + residual))


class V1Cifar100ResNet18(nn.Module):
    """实现固定 CIFAR-style `ResNet-18` 的 100 类 logits 模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        self.layer1 = self._make_layer(64, 64, block_count=2, stride=1)
        self.layer2 = self._make_layer(64, 128, block_count=2, stride=2)
        self.layer3 = self._make_layer(128, 256, block_count=2, stride=2)
        self.layer4 = self._make_layer(256, 512, block_count=2, stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(512, V1_M1_CLASS_COUNT)
        if sum(parameter.numel() for parameter in self.parameters()) != V1_M1_PARAMETER_COUNT:
            raise RuntimeError("V1-M1 ResNet-18 parameter count changed")

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        *,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [_V1M1BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, block_count):
            blocks.append(_V1M1BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*blocks)

    def forward(self, images: Tensor) -> Tensor:
        """计算已由可信预处理提供的 float32 image batch 的 100 类 logits。"""
        output = self.stem(images)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return cast(Tensor, self.classifier(self.flatten(self.average_pool(output))))


__all__ = [
    "V1_M1_CLASS_COUNT",
    "V1_M1_INPUT_SHAPE",
    "V1_M1_MODEL_PROFILE_ID",
    "V1_M1_PARAMETER_COUNT",
    "V1Cifar100ResNet18",
]
