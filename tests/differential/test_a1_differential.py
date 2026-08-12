"""A1-C1 ReLU graph 与 A0 精确 oracle 的全域差分测试。"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

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
    mod_q,
    parse_credential,
    verify_ref,
)
from can.verifier import (
    A1CompiledProfile,
    A1CompiledRegistry,
    A1EvidenceCode,
    compile_a1_profile,
    verify_a1,
)
from can.verifier.a1 import _evaluate_with_trace

TEST_SLOT_ID = 0xA101
TEST_SEED = 20260724


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


def _fixture() -> tuple[A0Registry, A1CompiledRegistry, A1CompiledProfile, tuple[int, ...]]:
    random = Random(TEST_SEED)
    secret = tuple(random.randrange(2) for _ in range(A0_SECRET_SIZE))
    rows: list[list[int]] = []
    for row_index in range(A0_COMPONENT_COUNT):
        row = [random.randrange(A0_MODULUS) for _ in range(A0_SECRET_SIZE)]
        if all(value == 0 for value in row):
            row[0] = row_index + 1
        rows.append(row)
    slot = A0Slot(TEST_SLOT_ID, rows)
    profile = compile_a1_profile(slot, secret)
    return A0Registry([slot]), A1CompiledRegistry([profile]), profile, secret


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


def test_every_canonical_coefficient_at_each_component_matches_reference_distance() -> None:
    """八个位置的全部 `b_i=0..256` 应逐值匹配精确 oracle。"""
    reference_registry, compiled_registry, profile, secret = _fixture()
    core_b = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    false_accepts = 0
    issuer_false_rejects = 0

    for component in range(A0_COMPONENT_COUNT):
        for coefficient in range(A0_MODULUS):
            b = core_b.copy()
            b[component] = coefficient
            raw = _encode(b)
            reference = verify_ref(raw, reference_registry, secret)
            neural = verify_a1(raw, compiled_registry)
            trace = _evaluate_with_trace(parse_credential(raw).b, profile)

            assert reference.distances is not None
            assert trace.distances == reference.distances
            assert neural.accepted is (
                reference.maximum_distance is not None and reference.maximum_distance <= 8
            )
            false_accepts += int(neural.accepted and not reference.accepted)
            issuer_false_rejects += int(
                reference.code is A0EvidenceCode.ISSUER_CORE and not neural.accepted
            )

    assert false_accepts == 0
    assert issuer_false_rejects == 0


@pytest.mark.parametrize(
    ("offsets", "expected_neural_code", "expected_reference_code"),
    [
        ([0] * 8, A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.ISSUER_CORE),
        ([-4, 4, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.ISSUER_CORE),
        ([5, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.REFERENCE_GUARD),
        ([8, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.REFERENCE_GUARD),
        ([9, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REFERENCE_GUARD),
        ([12, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REFERENCE_GUARD),
        ([13, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REJECT),
        ([0, 0, 0, 13, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REJECT),
        ([128] * 8, A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REJECT),
    ],
)
def test_a0_vector_families_preserve_one_sided_soundness(
    offsets: list[int],
    expected_neural_code: A1EvidenceCode,
    expected_reference_code: A0EvidenceCode,
) -> None:
    """core、guard、reject、bit-zero 和 mixed 向量应保持单向包含。"""
    reference_registry, compiled_registry, profile, secret = _fixture()
    raw = _credential_for_offsets(profile, offsets)

    reference = verify_ref(raw, reference_registry, secret)
    neural = verify_a1(raw, compiled_registry)

    assert reference.code is expected_reference_code
    assert neural.code is expected_neural_code
    assert not neural.accepted or reference.accepted


def test_modular_wrap_vectors_match_reference() -> None:
    """跨越 256/0 的合法向量应保持逐分量距离和接受结果。"""
    matrix = [[5] * A0_SECRET_SIZE for _ in range(A0_COMPONENT_COUNT)]
    secret = (1,) * A0_SECRET_SIZE
    slot = A0Slot(TEST_SLOT_ID, matrix)
    reference_registry = A0Registry([slot])
    profile = compile_a1_profile(slot, secret)
    compiled_registry = A1CompiledRegistry([profile])
    raw = _credential_for_offsets(profile, [4, -4, 0, 1, -1, 2, -2, 3])

    reference = verify_ref(raw, reference_registry, secret)
    neural = verify_a1(raw, compiled_registry)
    trace = _evaluate_with_trace(parse_credential(raw).b, profile)

    assert all(value < A0_CENTER for value in parse_credential(raw).b)
    assert trace.distances == reference.distances
    assert neural.code is A1EvidenceCode.NUMERIC_ACCEPT
    assert reference.code is A0EvidenceCode.ISSUER_CORE


def test_malformed_and_unknown_profile_families_fail_before_numeric_core() -> None:
    """错误编码和未知 profile 应在 A1 数值 graph 前拒绝且无弱回退。"""
    _, compiled_registry, profile, _ = _fixture()
    core = _credential_for_offsets(profile, [0] * A0_COMPONENT_COUNT)

    malformed = verify_a1(core + b"\x00", compiled_registry)
    unknown = verify_a1(_encode([0] * A0_COMPONENT_COUNT, profile_id=2), compiled_registry)

    assert malformed.code is A1EvidenceCode.PARSE_REJECT
    assert unknown.code is A1EvidenceCode.PROFILE_REJECT
