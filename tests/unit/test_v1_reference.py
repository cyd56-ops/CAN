"""V1-P2 Module-SIS exact reference 单元测试。"""

from __future__ import annotations

import hashlib

import pytest

from can.reference import (
    V1_ABORT_SIZE,
    V1_CHALLENGE_SIZE,
    V1_COMMITMENT_DOMAIN,
    V1_COMMITMENT_SIZE,
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RESPONSE_BOUND,
    V1_RESPONSE_SIZE,
    V1_RING_DEGREE,
    V1Abort,
    V1Challenge,
    V1Commitment,
    V1EvidenceCode,
    V1ProfileValidationError,
    V1PublicProfile,
    V1PublicRegistry,
    V1RegistryLookupError,
    V1RegistryValidationError,
    V1Response,
    V1WireParseError,
    build_v1_conformance_profile,
    parse_v1_abort,
    parse_v1_challenge,
    parse_v1_commitment,
    parse_v1_response,
    v1_negacyclic_convolution,
    verify_v1_ref,
)

IDENTITY = bytes(range(32))
TRANSCRIPT_ID = hashlib.sha256(b"v1 transcript").digest()


def _challenge() -> V1Challenge:
    return V1Challenge(V1_PROFILE_ID, (1, 0, 0, 0, 0, 0, 0, -1))


def _response_polynomials(boundary: int = 1) -> tuple[tuple[int, ...], ...]:
    return (
        (boundary, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0),
        (-1, 0, 1, 0, 0, 0, 0, 0),
        (0, -1, 0, 1, 0, 0, 0, 0),
    )


def _accepting_commitment(
    profile: V1PublicProfile,
    challenge: V1Challenge,
    polynomials: tuple[tuple[int, ...], ...],
) -> V1Commitment:
    outputs: list[tuple[int, ...]] = []
    for row_index in range(2):
        lhs = [0] * V1_RING_DEGREE
        for column_index in range(2):
            product = v1_negacyclic_convolution(
                profile.matrix[row_index][column_index], polynomials[column_index]
            )
            lhs = [value + product_value for value, product_value in zip(lhs, product, strict=True)]
        lhs = [
            value + identity_value
            for value, identity_value in zip(lhs, polynomials[2 + row_index], strict=True)
        ]
        challenge_target = v1_negacyclic_convolution(
            challenge.coefficients, profile.target[row_index]
        )
        outputs.append(
            tuple(
                (value - target_value) % V1_MODULUS
                for value, target_value in zip(lhs, challenge_target, strict=True)
            )
        )
    return V1Commitment(V1_PROFILE_ID, outputs)


def test_fixed_profile_is_deterministic_public_only() -> None:
    """固定 conformance fixture 只含相同的公开 matrix、target 与摘要。"""
    first = build_v1_conformance_profile(IDENTITY)
    second = build_v1_conformance_profile(IDENTITY)

    assert first == second
    assert first.public_key_sha256 == second.public_key_sha256
    assert len(first.public_key_sha256) == 32
    assert build_v1_conformance_profile(bytes(reversed(IDENTITY))).public_key_sha256 != (
        first.public_key_sha256
    )


def test_registry_is_identity_bound_and_rejects_duplicates() -> None:
    """registry 只按规范 identity 返回唯一 profile。"""
    profile = build_v1_conformance_profile(IDENTITY)
    registry = V1PublicRegistry((profile,))

    assert registry.lookup(IDENTITY) is profile
    with pytest.raises(V1RegistryLookupError):
        registry.lookup(bytes([255]) * 32)
    with pytest.raises(V1RegistryValidationError):
        V1PublicRegistry((profile, profile))


def test_zero_module_row_is_rejected_at_profile_construction() -> None:
    """可信 profile 构造期拒绝退化全零 A row。"""
    profile = build_v1_conformance_profile(IDENTITY)
    matrix = [[list(polynomial) for polynomial in row] for row in profile.matrix]
    matrix[0] = [[0] * V1_RING_DEGREE for _ in range(2)]

    with pytest.raises(V1ProfileValidationError):
        V1PublicProfile(V1_PROFILE_ID, IDENTITY, matrix, profile.target)


def test_x_to_n_is_exactly_negative_one() -> None:
    """系数域 oracle 必须实现 X^N=-1 的负循环 wraparound。"""
    x_to_n_minus_one = (0, 0, 0, 0, 0, 0, 0, 1)
    x = (0, 1, 0, 0, 0, 0, 0, 0)

    assert v1_negacyclic_convolution(x_to_n_minus_one, x) == (-1, 0, 0, 0, 0, 0, 0, 0)


def test_all_wire_encodings_round_trip_at_fixed_sizes() -> None:
    """四类 wire object 必须固定长度并无损往返。"""
    profile = build_v1_conformance_profile(IDENTITY)
    challenge = _challenge()
    response = V1Response(TRANSCRIPT_ID, _response_polynomials())
    commitment = _accepting_commitment(profile, challenge, response.polynomials)
    abort = V1Abort(TRANSCRIPT_ID)

    assert len(commitment.encode()) == V1_COMMITMENT_SIZE
    assert len(challenge.encode()) == V1_CHALLENGE_SIZE
    assert len(response.encode()) == V1_RESPONSE_SIZE
    assert len(abort.encode()) == V1_ABORT_SIZE
    assert parse_v1_commitment(commitment.encode()) == commitment
    assert parse_v1_challenge(challenge.encode()) == challenge
    assert parse_v1_response(response.encode()) == response
    assert parse_v1_abort(abort.encode()) == abort


def test_commitment_coefficient_order_is_vector_then_ascending_power() -> None:
    """commitment wire 顺序固定为 vector-major 再按 X 次数递增。"""
    commitment = V1Commitment(
        V1_PROFILE_ID,
        (tuple(range(8)), tuple(range(8, 16))),
    ).encode()
    offset = len(V1_COMMITMENT_DOMAIN) + 2

    decoded = tuple(
        int.from_bytes(commitment[index : index + 4], "big")
        for index in range(offset, len(commitment), 4)
    )
    assert decoded == tuple(range(16))


def test_valid_exact_relation_accepts_norm_boundary() -> None:
    """满足公开方程且系数恰为 B 的 response 必须接受。"""
    profile = build_v1_conformance_profile(IDENTITY)
    challenge = _challenge()
    polynomials = _response_polynomials(V1_RESPONSE_BOUND)
    commitment = _accepting_commitment(profile, challenge, polynomials)
    response = V1Response(TRANSCRIPT_ID, polynomials)

    evidence = verify_v1_ref(
        commitment.encode(),
        challenge.encode(),
        response.encode(),
        TRANSCRIPT_ID,
        profile,
    )

    assert evidence.code is V1EvidenceCode.RELATION_ACCEPT
    assert evidence.accepted


def test_norm_is_checked_before_an_otherwise_valid_equation() -> None:
    """方程成立但 response 系数为 B+1 时必须固定 norm reject。"""
    profile = build_v1_conformance_profile(IDENTITY)
    challenge = _challenge()
    polynomials = _response_polynomials(V1_RESPONSE_BOUND + 1)
    commitment = _accepting_commitment(profile, challenge, polynomials)
    response = V1Response(TRANSCRIPT_ID, polynomials)

    evidence = verify_v1_ref(
        commitment.encode(),
        challenge.encode(),
        response.encode(),
        TRANSCRIPT_ID,
        profile,
    )

    assert evidence.code is V1EvidenceCode.NORM_REJECT
    assert not evidence.accepted


def test_equation_mutation_and_transcript_mutation_are_rejected() -> None:
    """commitment 或 transcript 变化不能保持 relation accept。"""
    profile = build_v1_conformance_profile(IDENTITY)
    challenge = _challenge()
    polynomials = _response_polynomials()
    commitment = _accepting_commitment(profile, challenge, polynomials)
    response = V1Response(TRANSCRIPT_ID, polynomials)
    changed = [list(polynomial) for polynomial in commitment.polynomials]
    changed[0][0] = (changed[0][0] + 1) % V1_MODULUS

    equation = verify_v1_ref(
        V1Commitment(V1_PROFILE_ID, changed).encode(),
        challenge.encode(),
        response.encode(),
        TRANSCRIPT_ID,
        profile,
    )
    transcript = verify_v1_ref(
        commitment.encode(),
        challenge.encode(),
        response.encode(),
        bytes(32),
        profile,
    )

    assert equation.code is V1EvidenceCode.EQUATION_REJECT
    assert transcript.code is V1EvidenceCode.TRANSCRIPT_REJECT


def test_parser_rejects_noncanonical_residue_and_challenge_weight() -> None:
    """q residue 与 challenge weight 边界在 arithmetic 前拒绝。"""
    profile = build_v1_conformance_profile(IDENTITY)
    commitment = bytearray(V1Commitment(V1_PROFILE_ID, ((0,) * 8, (0,) * 8)).encode())
    offset = len(V1_COMMITMENT_DOMAIN) + 2
    commitment[offset : offset + 4] = V1_MODULUS.to_bytes(4, "big")
    challenge = bytearray(_challenge().encode())
    challenge[-1] = 0

    with pytest.raises(V1WireParseError):
        parse_v1_commitment(bytes(commitment))
    with pytest.raises(V1WireParseError):
        parse_v1_challenge(bytes(challenge))
    evidence = verify_v1_ref(
        bytes(commitment),
        _challenge().encode(),
        V1Response(TRANSCRIPT_ID, _response_polynomials()).encode(),
        TRANSCRIPT_ID,
        profile,
    )
    assert evidence.code is V1EvidenceCode.COMMITMENT_PARSE_REJECT


def test_signed_int32_boundaries_parse_but_fail_relation_norm() -> None:
    """wire int32 端点是规范编码, 但不因可解析而绕过 B。"""
    values = [list(polynomial) for polynomial in _response_polynomials()]
    values[0][0] = -(1 << 31)
    values[1][0] = (1 << 31) - 1
    response = V1Response(TRANSCRIPT_ID, values)

    assert parse_v1_response(response.encode()) == response
    evidence = verify_v1_ref(
        V1Commitment(V1_PROFILE_ID, ((0,) * 8, (0,) * 8)).encode(),
        _challenge().encode(),
        response.encode(),
        TRANSCRIPT_ID,
        build_v1_conformance_profile(IDENTITY),
    )
    assert evidence.code is V1EvidenceCode.NORM_REJECT


@pytest.mark.parametrize("raw", [None, True, b"", bytearray(V1_COMMITMENT_SIZE)])
def test_wrong_commitment_type_or_length_is_rejected(raw: object) -> None:
    """类型混淆与错误长度不能进入 V1 arithmetic。"""
    with pytest.raises(V1WireParseError):
        parse_v1_commitment(raw)


def test_wrong_profile_type_fails_closed() -> None:
    """请求方提供的 dict 不能代替本地可信 public profile。"""
    evidence = verify_v1_ref(b"", b"", b"", TRANSCRIPT_ID, {"q": V1_MODULUS})

    assert evidence.code is V1EvidenceCode.CONFIG_REJECT
