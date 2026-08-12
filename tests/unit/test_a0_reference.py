"""A0-v1 精确整数参考实现的单元测试。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError
from random import Random
from typing import cast

import pytest

from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_CREDENTIAL_SIZE,
    A0_MODULUS,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A0Credential,
    A0EvidenceCode,
    A0Registry,
    A0Slot,
    CredentialParseError,
    ReferenceEvidence,
    RegistryLookupError,
    RegistryValidationError,
    center_q,
    mod_q,
    parse_credential,
    verify_ref,
)

TEST_SLOT_ID = 0x01020304
TEST_SEED = 20260723


def _encode(
    b: Sequence[int],
    *,
    version: int = A0_VERSION,
    profile_id: int = A0_PROFILE_ID,
    slot_id: int = TEST_SLOT_ID,
) -> bytes:
    """编码测试输入。该 helper 允许构造语义非法但字节合法的字段。"""
    return (
        bytes([version])
        + profile_id.to_bytes(2, byteorder="big", signed=False)
        + slot_id.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in b)
    )


def _fixture(seed: int = TEST_SEED) -> tuple[A0Registry, A0Slot, tuple[int, ...]]:
    """用显式非秘密 seed 构造确定性 toy fixture。"""
    random = Random(seed)
    secret = tuple(random.randrange(2) for _ in range(A0_SECRET_SIZE))
    matrix_rows: list[list[int]] = []
    for row_index in range(A0_COMPONENT_COUNT):
        row = [random.randrange(A0_MODULUS) for _ in range(A0_SECRET_SIZE)]
        if all(coefficient == 0 for coefficient in row):
            row[0] = row_index + 1
        matrix_rows.append(row)
    slot = A0Slot(TEST_SLOT_ID, matrix_rows)
    return A0Registry([slot]), slot, secret


def _dot_products(slot: A0Slot, secret: Sequence[int]) -> tuple[int, ...]:
    """计算测试向量所需的本地矩阵乘积。"""
    return tuple(
        sum(coefficient * secret[column] for column, coefficient in enumerate(row))
        for row in slot.matrix
    )


def _credential_for_offsets(
    slot: A0Slot,
    secret: Sequence[int],
    offsets: Sequence[int],
) -> bytes:
    """生成指定逐分量中心距离的规范 credential。"""
    products = _dot_products(slot, secret)
    b = [mod_q(products[index] + A0_CENTER + offsets[index]) for index in range(A0_COMPONENT_COUNT)]
    return _encode(b, slot_id=slot.slot_id)


def test_parse_credential_decodes_the_only_canonical_wire_format() -> None:
    """解析器应按大端序精确解码固定字段。"""
    b = tuple(range(A0_COMPONENT_COUNT))

    credential = parse_credential(_encode(b))

    assert credential == A0Credential(A0_VERSION, A0_PROFILE_ID, TEST_SLOT_ID, b)
    assert credential.b == b


@pytest.mark.parametrize("length", [*range(A0_CREDENTIAL_SIZE), 24, 25, 39, 279])
def test_parse_credential_rejects_every_short_and_representative_long_length(length: int) -> None:
    """错误长度、前缀和尾随数据都不应被解析。"""
    with pytest.raises(CredentialParseError):
        parse_credential(bytes(length))


@pytest.mark.parametrize(
    "raw_value",
    [bytearray(A0_CREDENTIAL_SIZE), memoryview(bytes(A0_CREDENTIAL_SIZE)), "x", True, None],
)
def test_parse_credential_rejects_type_confusion(raw_value: object) -> None:
    """安全关键输入仅接受精确 bytes 类型。"""
    with pytest.raises(CredentialParseError):
        parse_credential(cast(bytes, raw_value))


def test_parse_credential_rejects_wrong_version_and_noncanonical_b() -> None:
    """未知版本和未约减的模系数应在验证前拒绝。"""
    with pytest.raises(CredentialParseError):
        parse_credential(_encode([0] * A0_COMPONENT_COUNT, version=2))
    with pytest.raises(CredentialParseError):
        parse_credential(_encode([A0_MODULUS, *([0] * (A0_COMPONENT_COUNT - 1))]))


@pytest.mark.parametrize(
    ("value", "expected_mod", "expected_center"),
    [
        (-514, 0, 0),
        (-258, 256, -1),
        (-129, 128, 128),
        (-128, 129, -128),
        (-1, 256, -1),
        (0, 0, 0),
        (128, 128, 128),
        (129, 129, -128),
        (256, 256, -1),
        (257, 0, 0),
        (514, 0, 0),
    ],
)
def test_mod_q_and_center_q_have_exact_negative_and_boundary_semantics(
    value: int,
    expected_mod: int,
    expected_center: int,
) -> None:
    """模约减不应依赖实现相关的负余数规则。"""
    assert mod_q(value) == expected_mod
    assert center_q(value) == expected_center


@pytest.mark.parametrize("invalid_value", [True, 1.0, "1", None])
def test_modular_helpers_reject_type_confusion(invalid_value: object) -> None:
    """bool 和其他隐式可转换类型不能进入精确算术。"""
    with pytest.raises(TypeError):
        mod_q(cast(int, invalid_value))
    with pytest.raises(TypeError):
        center_q(cast(int, invalid_value))


def test_slot_and_registry_copy_inputs_and_are_frozen() -> None:
    """调用方后续修改容器不得改变已加载的可信 registry。"""
    matrix = [[row + column + 1 for column in range(A0_SECRET_SIZE)] for row in range(8)]
    slot = A0Slot(TEST_SLOT_ID, matrix)
    registry = A0Registry([slot])

    matrix[0][0] = 0

    assert registry.lookup(A0_PROFILE_ID, TEST_SLOT_ID).matrix[0][0] == 1
    assert isinstance(registry.slots, tuple)
    enabled_attribute = "enabled"
    slots_attribute = "_slots"
    with pytest.raises(FrozenInstanceError):
        setattr(slot, enabled_attribute, False)
    with pytest.raises(FrozenInstanceError):
        setattr(registry, slots_attribute, {})


@pytest.mark.parametrize(
    "matrix",
    [
        [[1] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT - 1)],
        [[1] * (A0_SECRET_SIZE - 1) for _ in range(A0_COMPONENT_COUNT)],
        [[1] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT - 1)] + [[0] * A0_SECRET_SIZE],
        [[1] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT - 1)] + [[-1] * A0_SECRET_SIZE],
        [[1] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT - 1)]
        + [[A0_MODULUS] * A0_SECRET_SIZE],
        [[1] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT - 1)] + [[True] * A0_SECRET_SIZE],
    ],
)
def test_slot_rejects_wrong_shape_range_zero_rows_and_bool_coefficients(
    matrix: list[list[int]],
) -> None:
    """不完整或可降级的本地矩阵应在加载时失败。"""
    with pytest.raises(RegistryValidationError):
        A0Slot(TEST_SLOT_ID, matrix)


def test_registry_rejects_duplicates_wrong_entry_types_and_disabled_lookup() -> None:
    """registry 不应接受重复 slot、类型混淆或禁用项回退。"""
    matrix = [[row + 1] * A0_SECRET_SIZE for row in range(A0_COMPONENT_COUNT)]
    slot = A0Slot(TEST_SLOT_ID, matrix)

    with pytest.raises(RegistryValidationError):
        A0Registry([slot, slot])
    with pytest.raises(RegistryValidationError):
        A0Registry(cast(Iterable[A0Slot], [object()]))

    disabled_registry = A0Registry([A0Slot(TEST_SLOT_ID, matrix, enabled=False)])
    with pytest.raises(RegistryLookupError):
        disabled_registry.lookup(A0_PROFILE_ID, TEST_SLOT_ID)


@pytest.mark.parametrize(
    ("offsets", "expected_code", "expected_acceptance", "expected_maximum"),
    [
        ([0] * 8, A0EvidenceCode.ISSUER_CORE, True, 0),
        ([-4, 4, 0, 0, 0, 0, 0, 0], A0EvidenceCode.ISSUER_CORE, True, 4),
        ([5, 0, 0, 0, 0, 0, 0, 0], A0EvidenceCode.REFERENCE_GUARD, True, 5),
        ([12, 0, 0, 0, 0, 0, 0, 0], A0EvidenceCode.REFERENCE_GUARD, True, 12),
        ([13, 0, 0, 0, 0, 0, 0, 0], A0EvidenceCode.REJECT, False, 13),
        ([0, 0, 0, 0, 0, 0, 0, 128], A0EvidenceCode.REJECT, False, 128),
    ],
)
def test_verify_ref_classifies_core_guard_and_reject_boundaries(
    offsets: list[int],
    expected_code: A0EvidenceCode,
    expected_acceptance: bool,
    expected_maximum: int,
) -> None:
    """最大逐分量距离应精确决定三个参考区域。"""
    registry, slot, secret = _fixture()
    raw = _credential_for_offsets(slot, secret, offsets)

    evidence = verify_ref(raw, registry, secret)

    assert evidence.code is expected_code
    assert evidence.accepted is expected_acceptance
    assert evidence.maximum_distance == expected_maximum


def test_verify_ref_rejects_bit_zero_and_mixed_component_vectors() -> None:
    """bit-zero 和任一分量越界都必须使八路 AND 拒绝。"""
    registry, slot, secret = _fixture()
    products = _dot_products(slot, secret)
    bit_zero = _encode([mod_q(product) for product in products])
    mixed = _credential_for_offsets(slot, secret, [0, 0, 0, 13, 0, 0, 0, 0])

    bit_zero_evidence = verify_ref(bit_zero, registry, secret)
    mixed_evidence = verify_ref(mixed, registry, secret)

    assert bit_zero_evidence.code is A0EvidenceCode.REJECT
    assert bit_zero_evidence.distances is not None
    assert min(bit_zero_evidence.distances) == A0_CENTER
    assert mixed_evidence.code is A0EvidenceCode.REJECT
    assert mixed_evidence.distances == (0, 0, 0, 13, 0, 0, 0, 0)


def test_verify_ref_preserves_modular_wraparound() -> None:
    """跨越 256/0 的正向向量应保持数学关系不变。"""
    matrix = [[5] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT)]
    secret = (1,) * A0_SECRET_SIZE
    slot = A0Slot(TEST_SLOT_ID, matrix)
    registry = A0Registry([slot])
    raw = _credential_for_offsets(slot, secret, [4, -4, 0, 1, -1, 2, -2, 3])

    credential = parse_credential(raw)
    evidence = verify_ref(raw, registry, secret)

    assert all(value < A0_CENTER for value in credential.b)
    assert evidence.code is A0EvidenceCode.ISSUER_CORE
    assert evidence.distances == (4, 4, 0, 1, 1, 2, 2, 3)


def test_verify_ref_covers_every_exact_distance() -> None:
    """每个可表示精确距离 0..128 都应按固定阈值分类。"""
    registry, slot, secret = _fixture()

    for distance in range(A0_CENTER + 1):
        raw = _credential_for_offsets(slot, secret, [distance, 0, 0, 0, 0, 0, 0, 0])
        evidence = verify_ref(raw, registry, secret)
        assert evidence.maximum_distance == distance
        assert evidence.accepted is (distance <= 12)


def test_verify_ref_covers_every_canonical_b_at_each_component() -> None:
    """八个位置的每个规范 ``b_i`` 值都应使用同一精确公式。"""
    registry, slot, secret = _fixture()
    products = _dot_products(slot, secret)
    core_b = [mod_q(product + A0_CENTER) for product in products]

    for component in range(A0_COMPONENT_COUNT):
        for canonical_value in range(A0_MODULUS):
            b = core_b.copy()
            b[component] = canonical_value
            evidence = verify_ref(_encode(b), registry, secret)
            expected_distance = abs(
                center_q(mod_q(canonical_value - products[component]) - A0_CENTER)
            )
            assert evidence.distances is not None
            assert evidence.distances[component] == expected_distance
            assert evidence.maximum_distance == expected_distance


def test_verify_ref_is_deterministic_for_seed_and_repeated_credential() -> None:
    """相同公开测试 seed 和输入应重建完全相同的内部证据。"""
    first_registry, first_slot, first_secret = _fixture(TEST_SEED)
    second_registry, second_slot, second_secret = _fixture(TEST_SEED)
    raw = _credential_for_offsets(first_slot, first_secret, [1, -2, 3, -4, 0, 1, -1, 2])

    first = verify_ref(raw, first_registry, first_secret)
    repeated = verify_ref(raw, first_registry, first_secret)
    regenerated = verify_ref(raw, second_registry, second_secret)

    assert first_slot == second_slot
    assert first_secret == second_secret
    assert first == repeated == regenerated


def test_verify_ref_returns_structured_fail_closed_preverification_codes() -> None:
    """解析、registry 和可信配置错误都应返回无距离的拒绝证据。"""
    registry, slot, secret = _fixture()
    raw = _credential_for_offsets(slot, secret, [0] * A0_COMPONENT_COUNT)

    cases = (
        verify_ref(raw + b"\x00", registry, secret),
        verify_ref(_encode([0] * 8, profile_id=2), registry, secret),
        verify_ref(_encode([0] * 8, slot_id=TEST_SLOT_ID + 1), registry, secret),
        verify_ref(raw, registry, cast(Sequence[int], [0] * 31)),
        verify_ref(raw, registry, cast(Sequence[int], [False] * 32)),
        verify_ref(raw, cast(A0Registry, object()), secret),
    )

    assert [evidence.code for evidence in cases] == [
        A0EvidenceCode.PARSE_REJECT,
        A0EvidenceCode.REGISTRY_REJECT,
        A0EvidenceCode.REGISTRY_REJECT,
        A0EvidenceCode.CONFIG_REJECT,
        A0EvidenceCode.CONFIG_REJECT,
        A0EvidenceCode.CONFIG_REJECT,
    ]
    assert all(not evidence.accepted for evidence in cases)
    assert all(evidence.distances is None for evidence in cases)


def test_reference_evidence_rejects_internally_inconsistent_construction() -> None:
    """结构化证据自身不应表示矛盾的区域或最大距离。"""
    with pytest.raises(ValueError):
        ReferenceEvidence(A0EvidenceCode.ISSUER_CORE, (0,) * 8, 1)
    with pytest.raises(ValueError):
        ReferenceEvidence(A0EvidenceCode.REFERENCE_GUARD, (4,) * 8, 4)
    with pytest.raises(ValueError):
        ReferenceEvidence(A0EvidenceCode.REJECT, (12,) * 8, 12)
