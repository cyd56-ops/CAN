"""V1-M1 canonical uint8 adapter 与受保护 operation 单元测试。"""

import hashlib
import struct

import pytest
import torch

from can.access import (
    V1_M1_INPUT_PROFILE_SHA256,
    V1_M1_MODEL_ID,
    V1_M1_NORMALIZATION_MEAN,
    V1_M1_NORMALIZATION_STD,
    V1_M1_SCOPE_ID,
    A3V2ProtocolConfigError,
    V1M1InputAdapter,
    V1M1InputError,
    V1M1ProtectedOperation,
)
from can.access.v1_m1_adapter import _preprocess_v1_m1_snapshot, _V1M1ImageSnapshot
from can.model import V1Cifar100ResNet18

IDENTITY = bytes(range(32))


def _image() -> torch.Tensor:
    return (
        torch.arange(3 * 32 * 32, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, 3, 32, 32)
    )


def test_adapter_binds_cloned_pixels_to_the_fixed_v1_m1_profile() -> None:
    """adapter 应复制像素并以固定 profile 和 exact byte order 计算摘要。"""
    image = _image()
    trusted = V1M1InputAdapter(IDENTITY).adapt(image)
    payload = (
        b"CAN-V1-CIFAR100-INPUT-v1\x00"
        + V1_M1_INPUT_PROFILE_SHA256
        + struct.pack(">HHH", 3, 32, 32)
        + bytes(image.reshape(-1).tolist())
    )

    image.zero_()

    assert trusted.model_id == V1_M1_MODEL_ID
    assert trusted.identity_id == IDENTITY
    assert trusted.scope_id == V1_M1_SCOPE_ID
    assert trusted.input_profile_sha256 == V1_M1_INPUT_PROFILE_SHA256
    assert trusted.input_digest == hashlib.sha256(payload).digest()
    assert isinstance(trusted.snapshot, _V1M1ImageSnapshot)
    normalized = _preprocess_v1_m1_snapshot(trusted.snapshot)
    assert int(trusted.snapshot.pixels.reshape(-1)[1].item()) == 1
    expected = torch.tensor(-V1_M1_NORMALIZATION_MEAN[0] / V1_M1_NORMALIZATION_STD[0])
    assert torch.isclose(normalized[0, 0, 0, 0], expected)


@pytest.mark.parametrize(
    "image",
    [
        None,
        torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        torch.zeros((3, 32, 32), dtype=torch.uint8),
        torch.zeros((2, 3, 32, 32), dtype=torch.uint8),
        torch.zeros((1, 3, 31, 32), dtype=torch.uint8),
        torch.zeros((1, 3, 32, 32), dtype=torch.uint8).transpose(2, 3),
    ],
)
def test_adapter_rejects_noncanonical_untrusted_images(image: object) -> None:
    """类型、dtype、shape 与 layout 混淆必须在 adapter 边界被拒绝。"""
    with pytest.raises(V1M1InputError):
        V1M1InputAdapter(IDENTITY).adapt(image)


def test_adapter_rejects_noncanonical_trusted_identity() -> None:
    """本地 route identity 必须是精确的 V1 identity bytes。"""
    with pytest.raises(A3V2ProtocolConfigError):
        V1M1InputAdapter(b"short")


def test_preprocessing_uses_fixed_normalization_without_augmentation() -> None:
    """trusted snapshot 只能走确定性 float32 normalization。"""
    image = torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)
    trusted = V1M1InputAdapter(IDENTITY).adapt(image)

    assert isinstance(trusted.snapshot, _V1M1ImageSnapshot)
    normalized = _preprocess_v1_m1_snapshot(trusted.snapshot)

    expected = torch.tensor(
        [
            (1.0 - mean) / standard_deviation
            for mean, standard_deviation in zip(
                V1_M1_NORMALIZATION_MEAN,
                V1_M1_NORMALIZATION_STD,
                strict=True,
            )
        ],
        dtype=torch.float32,
    )
    assert normalized.dtype is torch.float32
    assert tuple(normalized.shape) == (1, 3, 32, 32)
    assert torch.allclose(normalized[0, :, 0, 0], expected)


def test_protected_operation_runs_only_on_a_trusted_snapshot() -> None:
    """固定 eval ResNet operation 必须拒绝伪造 snapshot。"""
    operation = V1M1ProtectedOperation(V1Cifar100ResNet18().eval())
    trusted = V1M1InputAdapter(IDENTITY).adapt(_image())

    logits = operation(trusted.snapshot)

    assert tuple(logits.shape) == (1, 100)
    with pytest.raises(V1M1InputError):
        operation(b"not a snapshot")


def test_protected_operation_rejects_training_mode_model_at_construction() -> None:
    """受保护 inference operation 只能持有已进入 eval 的可信模型。"""
    with pytest.raises(A3V2ProtocolConfigError):
        V1M1ProtectedOperation(V1Cifar100ResNet18())
