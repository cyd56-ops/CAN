"""A1-B1 PyTorch CPU exact-integer backend 的单元测试。"""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Sequence
from random import Random
from typing import cast

import pytest
import torch

from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_MODULUS,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A0Slot,
    mod_q,
)
from can.verifier import (
    A1CompiledProfile,
    A1CompiledRegistry,
    A1EvidenceCode,
    compile_a1_profile,
)
from can.verifier.a1 import _run_graph
from can.verifier.a1_torch import (
    A1_TORCH_BACKEND_ID,
    A1_TORCH_VERSION,
    A1TorchBackend,
    A1TorchBackendError,
    _evaluate_torch_with_trace,
    _execute_layer,
    compile_a1_torch_backend,
    verify_a1_torch,
)

TEST_SLOT_ID = 0xB101
TEST_SEED = 20260723


def _encode(
    coefficients: Sequence[int],
    *,
    profile_id: int = A0_PROFILE_ID,
    slot_id: int = TEST_SLOT_ID,
) -> bytes:
    return (
        bytes([A0_VERSION])
        + profile_id.to_bytes(2, byteorder="big", signed=False)
        + slot_id.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in coefficients)
    )


def _profile(seed: int = TEST_SEED, slot_id: int = TEST_SLOT_ID) -> A1CompiledProfile:
    random = Random(seed)
    secret = tuple(random.randrange(2) for _ in range(A0_SECRET_SIZE))
    rows: list[list[int]] = []
    for row_index in range(A0_COMPONENT_COUNT):
        row = [random.randrange(A0_MODULUS) for _ in range(A0_SECRET_SIZE)]
        if all(value == 0 for value in row):
            row[0] = row_index + 1
        rows.append(row)
    return compile_a1_profile(A0Slot(slot_id, rows), secret)


def _backend(profile: A1CompiledProfile) -> A1TorchBackend:
    return compile_a1_torch_backend(A1CompiledRegistry([profile]))


def _core_raw(profile: A1CompiledProfile) -> bytes:
    return _encode(
        [mod_q(anchor + A0_CENTER) for anchor in profile.anchors],
        slot_id=profile.slot_id,
    )


@pytest.fixture(scope="module")
def fixed_backend() -> tuple[A1CompiledProfile, A1TorchBackend]:
    """构建一次确定性随机 profile, 供只读单元测试复用。"""
    profile = _profile()
    return profile, _backend(profile)


@pytest.fixture(scope="module")
def residual_backend() -> tuple[dict[int, A1CompiledProfile], A1TorchBackend]:
    """构建覆盖正负 residual 两端的固定 profile。"""
    profiles = {
        0: A1CompiledProfile(TEST_SLOT_ID + 1, [0] * A0_COMPONENT_COUNT),
        256: A1CompiledProfile(TEST_SLOT_ID + 2, [256] * A0_COMPONENT_COUNT),
    }
    backend = compile_a1_torch_backend(A1CompiledRegistry(profiles.values()))
    return profiles, backend


def _torch_layers(
    coefficients: tuple[int, ...], backend: A1TorchBackend, slot_id: int
) -> tuple[tuple[int, ...], ...]:
    trace = _evaluate_torch_with_trace(coefficients, backend, A0_PROFILE_ID, slot_id)
    return tuple(tuple(layer.tolist()) for layer in trace.layers)


def test_backend_accepts_only_the_fixed_cpu_environment(
    fixed_backend: tuple[A1CompiledProfile, A1TorchBackend],
) -> None:
    """已激活 backend 应精确记录受支持的版本与 CPU-only 环境。"""
    _, backend = fixed_backend

    assert backend.backend_id == A1_TORCH_BACKEND_ID
    assert backend.active
    assert importlib.metadata.version("torch") == A1_TORCH_VERSION
    assert str(torch.__version__) == A1_TORCH_VERSION
    assert platform.system() == "Linux"
    assert platform.machine() == "x86_64"
    assert torch.version.cuda is None
    assert torch.version.hip is None
    assert not torch.cuda.is_available()


def test_backend_module_uses_only_nonpersistent_int32_cpu_buffers(
    fixed_backend: tuple[A1CompiledProfile, A1TorchBackend],
) -> None:
    """三层参数应只以内存态、无梯度、non-persistent buffers 存在。"""
    _, backend = fixed_backend
    module = next(iter(backend._entries.values())).module
    expected = {
        "weight_1": ((40, 8), (8, 1)),
        "bias_1": ((40,), (1,)),
        "weight_2": ((16, 40), (40, 1)),
        "bias_2": ((16,), (1,)),
        "weight_3": ((1, 16), (16, 1)),
        "bias_3": ((1,), (1,)),
    }

    assert tuple(module.parameters()) == ()
    assert tuple(module.children()) == ()
    assert not module.training
    assert module.state_dict() == {}
    assert module._non_persistent_buffers_set == set(expected)
    for name, tensor in module.named_buffers():
        shape, stride = expected[name]
        assert type(tensor) is torch.Tensor
        assert tensor.dtype is torch.int32
        assert tensor.device.type == "cpu"
        assert tuple(tensor.shape) == shape
        assert tensor.stride() == stride
        assert tensor.is_contiguous()
        assert not tensor.requires_grad


def test_fixed_operator_sequence_preserves_every_intermediate_dtype() -> None:
    """mul、sum、add、clamp 和 cast 应分别保持决策规定的整数 dtype。"""
    activation = torch.tensor((0, 256), dtype=torch.int32)
    weight = torch.tensor(((-1, 1), (1, -1)), dtype=torch.int32)
    bias = torch.tensor((3, -4), dtype=torch.int32)

    execution = _execute_layer(activation, weight, bias)

    assert execution.products.dtype is torch.int32
    assert execution.accumulator.dtype is torch.int64
    assert execution.preactivation.dtype is torch.int64
    assert execution.relu.dtype is torch.int64
    assert execution.output.dtype is torch.int32
    assert execution.products.tolist() == [[0, 256], [0, -256]]
    assert execution.accumulator.tolist() == [256, -256]
    assert execution.preactivation.tolist() == [259, -260]
    assert execution.relu.tolist() == [259, 0]
    assert execution.output.tolist() == [259, 0]


def test_torch_distance_matches_all_513_reachable_residuals(
    residual_backend: tuple[dict[int, A1CompiledProfile], A1TorchBackend],
) -> None:
    """PyTorch 五-ReLU 构造应穷尽匹配 residual -256..256。"""
    profiles, backend = residual_backend

    for residual in range(-256, 257):
        anchor = 256 if residual < 0 else 0
        profile = profiles[anchor]
        coefficient = residual + anchor
        coefficients = (coefficient,) * A0_COMPONENT_COUNT
        torch_layers = _torch_layers(coefficients, backend, profile.slot_id)

        assert torch_layers == _run_graph(coefficients, profile)
        layer1 = torch_layers[0]
        distances = tuple(
            -129
            + layer1[index]
            + 2 * layer1[index + 1]
            - layer1[index + 2]
            - 2 * layer1[index + 3]
            + 2 * layer1[index + 4]
            for index in range(0, 40, 5)
        )
        assert distances == (abs(mod_q(residual) - A0_CENTER),) * A0_COMPONENT_COUNT


def test_torch_threshold_matches_all_129_distances(
    residual_backend: tuple[dict[int, A1CompiledProfile], A1TorchBackend],
) -> None:
    """PyTorch 两-ReLU threshold 应穷尽实现距离 0..128 的 inclusive 8。"""
    profiles, backend = residual_backend
    profile = profiles[0]

    for distance in range(A0_CENTER + 1):
        coefficients = (A0_CENTER - distance,) * A0_COMPONENT_COUNT
        layer2 = _torch_layers(coefficients, backend, profile.slot_id)[1]
        gates = tuple(layer2[index] - layer2[index + 1] for index in range(0, 16, 2))
        assert gates == (int(distance <= 8),) * A0_COMPONENT_COUNT


def test_torch_final_relu_matches_all_nine_and_sums(
    residual_backend: tuple[dict[int, A1CompiledProfile], A1TorchBackend],
) -> None:
    """PyTorch 最终 ReLU 应对 pass count 0..8 精确实现八路 AND。"""
    profiles, backend = residual_backend
    profile = profiles[0]

    for pass_count in range(A0_COMPONENT_COUNT + 1):
        coefficients = (A0_CENTER,) * pass_count + (A0_CENTER - 9,) * (
            A0_COMPONENT_COUNT - pass_count
        )
        layers = _torch_layers(coefficients, backend, profile.slot_id)
        assert layers[-1] == (int(pass_count == A0_COMPONENT_COUNT),)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0, A1EvidenceCode.NUMERIC_ACCEPT),
        (8, A1EvidenceCode.NUMERIC_ACCEPT),
        (9, A1EvidenceCode.NUMERIC_REJECT),
        (128, A1EvidenceCode.NUMERIC_REJECT),
    ],
)
def test_raw_adapter_uses_the_fixed_threshold(
    fixed_backend: tuple[A1CompiledProfile, A1TorchBackend],
    offset: int,
    expected: A1EvidenceCode,
) -> None:
    """公共入口应只从 23 字节 credential 计算固定阈值 evidence。"""
    profile, backend = fixed_backend
    coefficients = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    coefficients[0] = mod_q(coefficients[0] + offset)

    evidence = verify_a1_torch(_encode(coefficients), backend)

    assert evidence.code is expected


def test_raw_adapter_returns_stable_preverification_codes(
    fixed_backend: tuple[A1CompiledProfile, A1TorchBackend],
) -> None:
    """解析、profile 和 backend 类型错误应稳定 fail closed。"""
    profile, backend = fixed_backend
    raw = _core_raw(profile)

    assert verify_a1_torch(raw + b"\x00", backend).code is A1EvidenceCode.PARSE_REJECT
    assert (
        verify_a1_torch(_encode([0] * A0_COMPONENT_COUNT, profile_id=2), backend).code
        is A1EvidenceCode.PROFILE_REJECT
    )
    assert (
        verify_a1_torch(_encode([0] * A0_COMPONENT_COUNT, slot_id=TEST_SLOT_ID + 100), backend).code
        is A1EvidenceCode.PROFILE_REJECT
    )
    assert verify_a1_torch(raw, cast(A1TorchBackend, object())).code is A1EvidenceCode.CONFIG_REJECT


def test_startup_gate_rejects_empty_or_wrong_registry_type() -> None:
    """无启用 profile 或 registry 类型混淆不能激活 backend。"""
    with pytest.raises(A1TorchBackendError):
        compile_a1_torch_backend(A1CompiledRegistry([]))
    with pytest.raises(A1TorchBackendError):
        compile_a1_torch_backend(cast(A1CompiledRegistry, object()))


@pytest.mark.parametrize(
    ("target", "replacement"),
    [
        ("package_version", lambda _name: "2.13.1+cpu"),
        ("machine", lambda: "aarch64"),
    ],
)
def test_startup_gate_rejects_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    replacement: object,
) -> None:
    """版本或 architecture 漂移不得创建可用 backend。"""
    import can.verifier.a1_torch as torch_module

    if target == "package_version":
        monkeypatch.setattr(torch_module, target, replacement)
    else:
        monkeypatch.setattr(platform, target, replacement)

    with pytest.raises(A1TorchBackendError):
        compile_a1_torch_backend(A1CompiledRegistry([_profile()]))
