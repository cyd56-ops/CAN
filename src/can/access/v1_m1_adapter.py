"""V1-M1 CIFAR-100 输入快照、摘要和受保护模型调用边界。"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor

from can.access.a3_v2 import (
    A3_V2_RESPONSE_VERSION,
    A3V2DenyEnvelope,
    A3V2Envelope,
    A3V2ProtocolConfigError,
    A3V2ProtocolCoordinator,
    A3V2ProtocolSnapshot,
    A3V2TrustedInput,
)
from can.model.v1_cifar100_resnet import V1Cifar100ResNet18
from can.reference.v1 import V1_IDENTITY_SIZE

V1_M1_MODEL_ID: Final = 0x0001_0001
V1_M1_SCOPE_ID: Final = 1
V1_M1_CHANNEL_COUNT: Final = 3
V1_M1_IMAGE_HEIGHT: Final = 32
V1_M1_IMAGE_WIDTH: Final = 32
V1_M1_PIXEL_COUNT: Final = V1_M1_CHANNEL_COUNT * V1_M1_IMAGE_HEIGHT * V1_M1_IMAGE_WIDTH
V1_M1_INPUT_DOMAIN: Final = b"CAN-V1-CIFAR100-INPUT-v1\x00"
V1_M1_INPUT_PROFILE_DOMAIN: Final = b"CAN-V1-CIFAR100-PROFILE-v1\x00"
V1_M1_NORMALIZATION_MEAN: Final = (0.5071, 0.4867, 0.4408)
V1_M1_NORMALIZATION_STD: Final = (0.2675, 0.2565, 0.2761)
V1_M1_INPUT_PROFILE_SHA256: Final = hashlib.sha256(
    V1_M1_INPUT_PROFILE_DOMAIN
    + struct.pack(
        ">IHHH",
        V1_M1_MODEL_ID,
        V1_M1_CHANNEL_COUNT,
        V1_M1_IMAGE_HEIGHT,
        V1_M1_IMAGE_WIDTH,
    )
    + struct.pack(">6d", *V1_M1_NORMALIZATION_MEAN, *V1_M1_NORMALIZATION_STD)
).digest()


class V1M1InputError(ValueError):
    """表示 V1-M1 不可信业务图像不满足唯一输入契约。"""


@dataclass(frozen=True, slots=True)
class _V1M1ImageSnapshot:
    """保存仅由 V1-M1 adapter 生成的不可变引用快照。"""

    pixels: Tensor


class V1M1InputAdapter:
    """从单张不可信 CIFAR uint8 图像构造 A3-v2 trusted input。"""

    __slots__ = ("_identity_id",)

    def __init__(self, identity_id: bytes) -> None:
        if type(identity_id) is not bytes or len(identity_id) != V1_IDENTITY_SIZE:
            raise A3V2ProtocolConfigError("V1-M1 adapter identity is not canonical")
        self._identity_id = identity_id

    def adapt(self, image: object) -> A3V2TrustedInput:
        """验证、复制原始 uint8 图像并绑定 V1-M1 local profile。"""
        if type(image) is not torch.Tensor:
            raise V1M1InputError("V1-M1 image must be exactly torch.Tensor")
        if image.dtype is not torch.uint8:
            raise V1M1InputError("V1-M1 image must use torch.uint8")
        if image.device.type != "cpu" or image.device.index is not None:
            raise V1M1InputError("V1-M1 image must remain on the CPU")
        if image.layout is not torch.strided or not image.is_contiguous():
            raise V1M1InputError("V1-M1 image must use contiguous strided layout")
        if tuple(image.shape) != (
            1,
            V1_M1_CHANNEL_COUNT,
            V1_M1_IMAGE_HEIGHT,
            V1_M1_IMAGE_WIDTH,
        ):
            raise V1M1InputError("V1-M1 image has the wrong shape")
        pixels = image.detach().clone(memory_format=torch.contiguous_format)
        payload = b"".join(
            (
                V1_M1_INPUT_DOMAIN,
                V1_M1_INPUT_PROFILE_SHA256,
                struct.pack(
                    ">HHH",
                    V1_M1_CHANNEL_COUNT,
                    V1_M1_IMAGE_HEIGHT,
                    V1_M1_IMAGE_WIDTH,
                ),
                pixels.reshape(V1_M1_PIXEL_COUNT).numpy().tobytes(),
            )
        )
        return A3V2TrustedInput(
            model_id=V1_M1_MODEL_ID,
            identity_id=self._identity_id,
            scope_id=V1_M1_SCOPE_ID,
            input_profile_sha256=V1_M1_INPUT_PROFILE_SHA256,
            input_digest=hashlib.sha256(payload).digest(),
            snapshot=_V1M1ImageSnapshot(pixels),
        )


def _preprocess_v1_m1_snapshot(snapshot: _V1M1ImageSnapshot) -> Tensor:
    """将已由 adapter 冻结的 uint8 snapshot 转换为规范 float32 输入。"""
    return normalize_v1_m1_uint8_batch(snapshot.pixels)


def normalize_v1_m1_uint8_batch(images: Tensor) -> Tensor:
    """将可信 channel-major uint8 image batch 转换为固定 float32 normalization。"""
    normalized = images.to(dtype=torch.float32).div(255.0)
    mean = torch.tensor(V1_M1_NORMALIZATION_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
    standard_deviation = torch.tensor(V1_M1_NORMALIZATION_STD, dtype=torch.float32).reshape(
        1, 3, 1, 1
    )
    return ((normalized - mean) / standard_deviation).contiguous()


class V1M1ProtectedOperation:
    """把已提交的 V1-M1 snapshot 交给固定 eval-mode ResNet-18。"""

    __slots__ = ("_device", "_model")

    def __init__(self, model: V1Cifar100ResNet18) -> None:
        if type(model) is not V1Cifar100ResNet18:
            raise A3V2ProtocolConfigError("V1-M1 protected operation requires exact ResNet-18")
        if model.training:
            raise A3V2ProtocolConfigError("V1-M1 protected model must use eval mode")
        parameter = next(model.parameters(), None)
        if parameter is None or parameter.dtype is not torch.float32:
            raise A3V2ProtocolConfigError("V1-M1 protected model has no float32 parameters")
        self._model = model
        self._device = parameter.device

    def __call__(self, snapshot: object) -> Tensor:
        """在授权提交后仅执行一次预处理和 ResNet-18 inference。"""
        if type(snapshot) is not _V1M1ImageSnapshot:
            raise V1M1InputError("V1-M1 protected operation requires its trusted snapshot")
        images = _preprocess_v1_m1_snapshot(snapshot).to(self._device)
        with torch.inference_mode():
            logits = self._model(images)
        return cast(Tensor, logits)


class V1M1AccessCoordinator:
    """把原始 V1-M1 image 输入唯一地交给 adapter 和 A3-v2 协调器。"""

    __slots__ = ("_adapter", "_coordinator")

    def __init__(
        self,
        adapter: V1M1InputAdapter,
        coordinator: A3V2ProtocolCoordinator,
    ) -> None:
        if type(adapter) is not V1M1InputAdapter:
            raise A3V2ProtocolConfigError("V1-M1 coordinator requires its exact input adapter")
        if type(coordinator) is not A3V2ProtocolCoordinator:
            raise A3V2ProtocolConfigError("V1-M1 coordinator requires exact A3-v2 coordinator")
        self._adapter = adapter
        self._coordinator = coordinator

    def begin(self, image: object, commitment: object) -> A3V2Envelope:
        """从原始图像创建唯一 V1-M1 trusted input 后签发 challenge 或拒绝。"""
        try:
            trusted_input = self._adapter.adapt(image)
        except V1M1InputError:
            return _deny_envelope()
        return self._coordinator.begin(trusted_input, commitment)

    def respond(self, response: object) -> A3V2Envelope:
        """将 response 交给唯一 A3-v2 coordinator 提交最终决定。"""
        return self._coordinator.respond(response)

    def abort(self, abort: object) -> A3V2DenyEnvelope:
        """终结一个 V1-M1 pending transcript, 且不执行 protected operation。"""
        return self._coordinator.abort(abort)

    def snapshot(self) -> A3V2ProtocolSnapshot:
        """返回不含图像、transcript 或 evidence 的 A3-v2 计数快照。"""
        return self._coordinator.snapshot()


def _deny_envelope() -> A3V2DenyEnvelope:
    return {"version": A3_V2_RESPONSE_VERSION, "status": "deny"}


__all__ = [
    "V1_M1_CHANNEL_COUNT",
    "V1_M1_IMAGE_HEIGHT",
    "V1_M1_IMAGE_WIDTH",
    "V1_M1_INPUT_PROFILE_SHA256",
    "V1_M1_MODEL_ID",
    "V1_M1_NORMALIZATION_MEAN",
    "V1_M1_NORMALIZATION_STD",
    "V1_M1_SCOPE_ID",
    "V1M1AccessCoordinator",
    "V1M1InputAdapter",
    "V1M1InputError",
    "V1M1ProtectedOperation",
    "normalize_v1_m1_uint8_batch",
]
