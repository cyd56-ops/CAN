"""A1-C1 固定整数 ReLU conformance backend 的单元测试。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError, fields
from random import Random
from typing import cast

import pytest

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
    A1_CANDIDATE_ID,
    A1_INPUT_WIDTH,
    A1_LAYER_WIDTHS,
    A1_SCALE,
    A1AffineReluLayer,
    A1CompiledProfile,
    A1CompiledRegistry,
    A1EvaluationError,
    A1Evidence,
    A1EvidenceCode,
    A1ProfileLookupError,
    A1ProfileValidationError,
    compile_a1_profile,
    verify_a1,
)
from can.verifier.a1 import _evaluate_with_trace

TEST_SLOT_ID = 0x11223344
TEST_SEED = 20260723


def _encode(
    b: Sequence[int],
    *,
    profile_id: int = A0_PROFILE_ID,
    slot_id: int = TEST_SLOT_ID,
) -> bytes:
    return (
        bytes([A0_VERSION])
        + profile_id.to_bytes(2, byteorder="big", signed=False)
        + slot_id.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in b)
    )


def _fixture(seed: int = TEST_SEED) -> tuple[A0Slot, tuple[int, ...]]:
    random = Random(seed)
    secret = tuple(random.randrange(2) for _ in range(A0_SECRET_SIZE))
    rows: list[list[int]] = []
    for row_index in range(A0_COMPONENT_COUNT):
        row = [random.randrange(A0_MODULUS) for _ in range(A0_SECRET_SIZE)]
        if all(value == 0 for value in row):
            row[0] = row_index + 1
        rows.append(row)
    return A0Slot(TEST_SLOT_ID, rows), secret


def _core_credential(profile: A1CompiledProfile) -> bytes:
    return _encode([mod_q(anchor + A0_CENTER) for anchor in profile.anchors])


def _credential_for_offsets(
    profile: A1CompiledProfile,
    offsets: Sequence[int],
) -> bytes:
    return _encode(
        [
            mod_q(profile.anchors[index] + A0_CENTER + offsets[index])
            for index in range(A0_COMPONENT_COUNT)
        ]
    )


def test_compile_a1_profile_builds_the_fixed_immutable_graph() -> None:
    """compiler 应生成构造决定规定的拓扑、规模和不可变参数。"""
    slot, secret = _fixture()

    profile = compile_a1_profile(slot, secret)

    expected_anchors = tuple(
        mod_q(sum(value * secret[index] for index, value in enumerate(row))) for row in slot.matrix
    )
    assert profile.candidate_id == A1_CANDIDATE_ID
    assert profile.profile_id == A0_PROFILE_ID
    assert profile.slot_id == TEST_SLOT_ID
    assert profile.scale == A1_SCALE
    assert profile.anchors == expected_anchors
    assert tuple((layer.input_width, layer.output_width) for layer in profile.layers) == (
        (A1_INPUT_WIDTH, A1_LAYER_WIDTHS[0]),
        (A1_LAYER_WIDTHS[0], A1_LAYER_WIDTHS[1]),
        (A1_LAYER_WIDTHS[1], A1_LAYER_WIDTHS[2]),
    )

    dense_weights = sum(layer.input_width * layer.output_width for layer in profile.layers)
    nonzero_weights = sum(
        value != 0 for layer in profile.layers for row in layer.weights for value in row
    )
    bias_count = sum(layer.output_width for layer in profile.layers)
    assert dense_weights + bias_count == 1033
    assert nonzero_weights + bias_count == 193

    with pytest.raises(FrozenInstanceError):
        profile_attribute = "anchors"
        setattr(profile, profile_attribute, (0,) * A0_COMPONENT_COUNT)
    with pytest.raises(FrozenInstanceError):
        layer_attribute = "bias"
        setattr(profile.layers[0], layer_attribute, (0,) * A1_LAYER_WIDTHS[0])


def test_compiler_is_deterministic_for_the_same_local_inputs() -> None:
    """相同 slot 和 toy secret 应重建完全相同的 fixed graph。"""
    slot, secret = _fixture()

    first = compile_a1_profile(slot, secret)
    second = compile_a1_profile(slot, secret)

    assert first == second


@pytest.mark.parametrize(
    "anchors",
    [
        [0] * (A0_COMPONENT_COUNT - 1),
        [0] * (A0_COMPONENT_COUNT - 1) + [-1],
        [0] * (A0_COMPONENT_COUNT - 1) + [A0_MODULUS],
        [0] * (A0_COMPONENT_COUNT - 1) + [True],
    ],
)
def test_compiled_profile_rejects_wrong_anchor_shape_range_and_type(
    anchors: list[int],
) -> None:
    """compiled anchor 不能依赖隐式转换或非规范模表示。"""
    with pytest.raises(A1ProfileValidationError):
        A1CompiledProfile(TEST_SLOT_ID, anchors)


@pytest.mark.parametrize("invalid_secret", [[0] * 31, [0] * 31 + [2], [False] * 32])
def test_compiler_rejects_invalid_toy_secret(invalid_secret: list[int]) -> None:
    """compiler 只接受精确 32 位二进制 toy secret。"""
    slot, _ = _fixture()
    with pytest.raises(A1ProfileValidationError):
        compile_a1_profile(slot, invalid_secret)


def test_compiler_rejects_disabled_or_wrong_slot_type() -> None:
    """禁用或类型混淆的本地 slot 不能生成 compiled profile。"""
    slot, secret = _fixture()
    disabled = A0Slot(slot.slot_id, slot.matrix, enabled=False)

    with pytest.raises(A1ProfileValidationError):
        compile_a1_profile(disabled, secret)
    with pytest.raises(A1ProfileValidationError):
        compile_a1_profile(cast(A0Slot, object()), secret)


def test_affine_relu_layer_rejects_invalid_structure_and_bool_values() -> None:
    """通用层容器必须拒绝不一致 shape、空层和 bool/int 混淆。"""
    invalid_layers: tuple[tuple[Iterable[Iterable[int]], Iterable[int]], ...] = (
        ([], []),
        ([[1], [1, 2]], [0, 0]),
        ([[1]], []),
        ([[True]], [0]),
        ([[1]], [True]),
    )

    for weights, bias in invalid_layers:
        with pytest.raises(A1ProfileValidationError):
            A1AffineReluLayer(weights, bias)


def test_relu_distance_is_exact_for_every_reachable_residual() -> None:
    """五个 ReLU 应穷尽匹配全部 513 个 canonical residual。"""
    profiles = {
        0: A1CompiledProfile(TEST_SLOT_ID, [0] * A0_COMPONENT_COUNT),
        256: A1CompiledProfile(TEST_SLOT_ID, [256] * A0_COMPONENT_COUNT),
    }

    for residual in range(-256, 257):
        anchor = 256 if residual < 0 else 0
        coefficient = residual + anchor
        trace = _evaluate_with_trace(
            (coefficient,) * A0_COMPONENT_COUNT,
            profiles[anchor],
        )
        expected_distance = abs(mod_q(residual) - A0_CENTER)
        assert trace.distances == (expected_distance,) * A0_COMPONENT_COUNT


def test_relu_threshold_is_exact_for_every_distance() -> None:
    """两个 ReLU 应穷尽实现距离 0..128 的 inclusive threshold。"""
    profile = A1CompiledProfile(TEST_SLOT_ID, [0] * A0_COMPONENT_COUNT)

    for distance in range(A0_CENTER + 1):
        coefficient = A0_CENTER - distance
        trace = _evaluate_with_trace((coefficient,) * A0_COMPONENT_COUNT, profile)
        expected_gate = int(distance <= 8)
        assert trace.distances == (distance,) * A0_COMPONENT_COUNT
        assert trace.gates == (expected_gate,) * A0_COMPONENT_COUNT
        assert trace.output == expected_gate


def test_final_relu_is_exact_for_every_possible_pass_count() -> None:
    """最终 ReLU 应对通过分量数 0..8 精确实现八路 AND。"""
    profile = A1CompiledProfile(TEST_SLOT_ID, [0] * A0_COMPONENT_COUNT)

    for pass_count in range(A0_COMPONENT_COUNT + 1):
        coefficients = (A0_CENTER,) * pass_count + (A0_CENTER - 9,) * (
            A0_COMPONENT_COUNT - pass_count
        )
        trace = _evaluate_with_trace(coefficients, profile)
        assert sum(trace.gates) == pass_count
        assert trace.output == int(pass_count == A0_COMPONENT_COUNT)


@pytest.mark.parametrize(
    ("offset", "expected_code"),
    [
        (0, A1EvidenceCode.NUMERIC_ACCEPT),
        (4, A1EvidenceCode.NUMERIC_ACCEPT),
        (8, A1EvidenceCode.NUMERIC_ACCEPT),
        (9, A1EvidenceCode.NUMERIC_REJECT),
        (12, A1EvidenceCode.NUMERIC_REJECT),
        (13, A1EvidenceCode.NUMERIC_REJECT),
        (128, A1EvidenceCode.NUMERIC_REJECT),
    ],
)
def test_adapter_applies_the_exact_neural_threshold(
    offset: int,
    expected_code: A1EvidenceCode,
) -> None:
    """公共 adapter 应使用阈值 8 且不扩大 reference guard。"""
    slot, secret = _fixture()
    profile = compile_a1_profile(slot, secret)
    registry = A1CompiledRegistry([profile])
    raw = _credential_for_offsets(profile, [offset, *([0] * 7)])

    evidence = verify_a1(raw, registry)

    assert evidence.code is expected_code


def test_adapter_rejects_any_failed_component() -> None:
    """任一分量越过神经阈值都必须使最终 AND 拒绝。"""
    slot, secret = _fixture()
    profile = compile_a1_profile(slot, secret)
    registry = A1CompiledRegistry([profile])
    mixed = _credential_for_offsets(profile, [0, 0, 0, 9, 0, 0, 0, 0])

    evidence = verify_a1(mixed, registry)

    assert evidence.code is A1EvidenceCode.NUMERIC_REJECT
    assert not evidence.accepted


def test_adapter_returns_stable_preverification_reject_codes() -> None:
    """解析、profile 和可信配置失败应稳定 fail closed。"""
    slot, secret = _fixture()
    profile = compile_a1_profile(slot, secret)
    registry = A1CompiledRegistry([profile])
    core = _core_credential(profile)

    results = (
        verify_a1(core + b"\x00", registry),
        verify_a1(_encode([0] * A0_COMPONENT_COUNT, profile_id=2), registry),
        verify_a1(_encode([0] * A0_COMPONENT_COUNT, slot_id=TEST_SLOT_ID + 1), registry),
        verify_a1(core, cast(A1CompiledRegistry, object())),
    )

    assert [result.code for result in results] == [
        A1EvidenceCode.PARSE_REJECT,
        A1EvidenceCode.PROFILE_REJECT,
        A1EvidenceCode.PROFILE_REJECT,
        A1EvidenceCode.CONFIG_REJECT,
    ]
    assert all(not result.accepted for result in results)


def test_compiled_registry_is_immutable_and_has_no_lookup_fallback() -> None:
    """compiled registry 应拒绝重复项、错误类型和未知查询。"""
    profile = A1CompiledProfile(TEST_SLOT_ID, [0] * A0_COMPONENT_COUNT)
    registry = A1CompiledRegistry([profile])

    assert registry.profiles == (profile,)
    with pytest.raises(A1ProfileValidationError):
        A1CompiledRegistry([profile, profile])
    with pytest.raises(A1ProfileValidationError):
        A1CompiledRegistry(cast(Iterable[A1CompiledProfile], [object()]))
    with pytest.raises(A1ProfileLookupError):
        registry.lookup(2, TEST_SLOT_ID)
    with pytest.raises(A1ProfileLookupError):
        registry.lookup(A0_PROFILE_ID, TEST_SLOT_ID + 1)
    with pytest.raises(FrozenInstanceError):
        registry_attribute = "_profiles"
        setattr(registry, registry_attribute, {})


@pytest.mark.parametrize(
    "coefficients",
    [
        (0,) * (A0_COMPONENT_COUNT - 1),
        (0,) * (A0_COMPONENT_COUNT - 1) + (-1,),
        (0,) * (A0_COMPONENT_COUNT - 1) + (A0_MODULUS,),
        (0,) * (A0_COMPONENT_COUNT - 1) + (True,),
    ],
)
def test_private_core_trace_rejects_noncanonical_coefficients(
    coefficients: tuple[int, ...],
) -> None:
    """即使测试直接调用内部 trace 也不能绕过规范整数边界。"""
    profile = A1CompiledProfile(TEST_SLOT_ID, [0] * A0_COMPONENT_COUNT)
    with pytest.raises(A1EvaluationError):
        _evaluate_with_trace(coefficients, profile)


def test_evidence_has_no_authorization_or_numeric_trace_fields() -> None:
    """A1 evidence 不能携带授权原语或可利用的内部距离。"""
    evidence_fields = {item.name for item in fields(A1Evidence)}

    assert evidence_fields == {"code"}
    assert not (
        {
            "distance",
            "distances",
            "trace",
            "gate",
            "decision",
            "authorization",
            "capability",
        }
        & evidence_fields
    )
