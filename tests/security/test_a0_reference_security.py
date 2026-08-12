"""A0-v1 参考边界的防御性安全测试。"""

from __future__ import annotations

from dataclasses import fields

import pytest

from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_MODULUS,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A0EvidenceCode,
    A0Registry,
    A0Slot,
    ReferenceEvidence,
    RegistryValidationError,
    mod_q,
    verify_ref,
)

TEST_SLOT_ID = 7


def _encode(b: list[int], *, profile_id: int = A0_PROFILE_ID) -> bytes:
    """编码安全边界测试使用的固定 credential。"""
    return (
        bytes([A0_VERSION])
        + profile_id.to_bytes(2, byteorder="big", signed=False)
        + TEST_SLOT_ID.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in b)
    )


def _fixed_oracle_fixture() -> tuple[A0Registry, A0Slot, tuple[int, ...]]:
    """构造可人工核对的固定 toy relation。"""
    slot = A0Slot(
        TEST_SLOT_ID,
        [[row + 1] * A0_SECRET_SIZE for row in range(A0_COMPONENT_COUNT)],
    )
    return A0Registry([slot]), slot, (1,) * A0_SECRET_SIZE


def _core_credential(slot: A0Slot, secret: tuple[int, ...]) -> bytes:
    """生成仅供本测试使用的 issuer-core credential。"""
    b = [
        mod_q(sum(value * secret[index] for index, value in enumerate(row)) + A0_CENTER)
        for row in slot.matrix
    ]
    return _encode(b)


def test_zero_matrix_rows_fail_at_the_trusted_registry_boundary() -> None:
    """本地 ``A=0`` 语义降级必须在 oracle 可调用前失败。"""
    with pytest.raises(RegistryValidationError):
        A0Slot(TEST_SLOT_ID, [[0] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT)])


def test_client_supplied_matrix_bytes_cannot_enter_the_wire_format() -> None:
    """在 credential 前后附加矩阵都只能得到固定解析拒绝。"""
    registry, slot, secret = _fixed_oracle_fixture()
    credential = _core_credential(slot, secret)
    supplied_matrix = bytes([1]) * (A0_COMPONENT_COUNT * A0_SECRET_SIZE)

    prefixed = verify_ref(supplied_matrix + credential, registry, secret)
    appended = verify_ref(credential + supplied_matrix, registry, secret)

    assert prefixed.code is A0EvidenceCode.PARSE_REJECT
    assert appended.code is A0EvidenceCode.PARSE_REJECT
    assert not prefixed.accepted and not appended.accepted


def test_chosen_center_b_does_not_override_the_local_matrix_phase() -> None:
    """直接选择 ``b=h`` 不得绕过可信矩阵对应的相位计算。"""
    registry, _, secret = _fixed_oracle_fixture()
    chosen_b = _encode([A0_CENTER] * A0_COMPONENT_COUNT)

    evidence = verify_ref(chosen_b, registry, secret)

    assert evidence.code is A0EvidenceCode.REJECT
    assert evidence.maximum_distance is not None
    assert evidence.maximum_distance > 12


def test_unknown_profile_has_no_weaker_fallback() -> None:
    """请求方选择未知 profile 时不得复用唯一可信 profile。"""
    registry, _, secret = _fixed_oracle_fixture()

    evidence = verify_ref(_encode([0] * A0_COMPONENT_COUNT, profile_id=2), registry, secret)

    assert evidence.code is A0EvidenceCode.REGISTRY_REJECT
    assert not evidence.accepted


def test_replay_remains_an_explicit_a0_limitation_without_mutating_registry() -> None:
    """A0 的重复查询会重放同一证据。验证过程不会产生隐藏授权状态。"""
    registry, slot, secret = _fixed_oracle_fixture()
    credential = _core_credential(slot, secret)
    slots_before = registry.slots

    first = verify_ref(credential, registry, secret)
    replayed = verify_ref(credential, registry, secret)

    assert first == replayed
    assert first.accepted
    assert registry.slots == slots_before


def test_reference_evidence_contains_no_authorization_primitive() -> None:
    """reference 接受不改变证据边界。证据结构不能携带授权原语。"""
    registry, slot, secret = _fixed_oracle_fixture()

    evidence = verify_ref(_core_credential(slot, secret), registry, secret)
    evidence_fields = {item.name for item in fields(ReferenceEvidence)}

    assert evidence.accepted
    assert evidence_fields == {"code", "distances", "maximum_distance"}
    assert not ({"gate", "decision", "capability", "authorization"} & evidence_fields)


def test_arbitrary_public_matrix_knowledge_is_not_an_issuer_api() -> None:
    """参考模块只判定固定字节。公开矩阵不能作为请求字段提交。"""
    registry, slot, secret = _fixed_oracle_fixture()
    canonical = _core_credential(slot, secret)
    encoded_public_row = bytes(value % A0_MODULUS for value in slot.matrix[0])

    evidence = verify_ref(canonical + encoded_public_row, registry, secret)

    assert evidence.code is A0EvidenceCode.PARSE_REJECT
    assert not evidence.accepted
