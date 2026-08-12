"""A4 GPV-PFDH toy 公钥关系的精确单元测试。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, fields
from typing import cast

import pytest

from can.access import (
    A3EvidenceCode,
    A4ReferenceAdapter,
    build_a4_verification_profile,
)
from can.reference import (
    A4_MESSAGE_DOMAIN,
    A4_MODULUS,
    A4_PROFILE_ID,
    A4_PROOF_SIZE,
    A4_SYNDROME_SIZE,
    A4_VECTOR_SIZE,
    A4EvidenceCode,
    A4ProfileValidationError,
    A4Proof,
    A4ProofParseError,
    A4PublicProfile,
    A4ReferenceEvidence,
    hash_to_a4_syndrome,
    parse_a4_proof,
    verify_a4_ref,
)
from conftest import A4ProofFactory


def _gadget_matrix() -> list[list[int]]:
    matrix = [[0] * A4_VECTOR_SIZE for _ in range(A4_SYNDROME_SIZE)]
    for row_index in range(A4_SYNDROME_SIZE):
        for bit_index in range(9):
            matrix[row_index][row_index * 9 + bit_index] = 1 << bit_index
    return matrix


def test_proof_round_trip_preserves_signed_int8_endpoints() -> None:
    """proof parser 应唯一解码 signed-int8 两端和 salt。"""
    vector = [-128, 127, *([0] * (A4_VECTOR_SIZE - 2))]
    proof = A4Proof(1, bytes(range(32)), vector)

    parsed = parse_a4_proof(proof.encode())

    assert len(proof.encode()) == A4_PROOF_SIZE
    assert parsed == proof
    assert parsed.vector[:2] == (-128, 127)


@pytest.mark.parametrize(
    "raw_proof",
    [
        None,
        True,
        bytearray(A4_PROOF_SIZE),
        memoryview(bytes(A4_PROOF_SIZE)),
        bytes(A4_PROOF_SIZE - 1),
        bytes(A4_PROOF_SIZE + 1),
        bytes([2]) + bytes(A4_PROOF_SIZE - 1),
    ],
)
def test_proof_parser_rejects_type_length_and_version_confusion(raw_proof: object) -> None:
    """A4 proof 入口只接受 exact bytes、固定长度和固定版本。"""
    with pytest.raises(A4ProofParseError):
        parse_a4_proof(raw_proof)


def test_public_profile_copies_matrix_is_frozen_and_has_stable_digest() -> None:
    """可信公开矩阵只在构造期校验并复制为不可变状态。"""
    matrix = _gadget_matrix()
    profile = A4PublicProfile(A4_PROFILE_ID, bytes(range(32)), matrix)
    expected_digest = "edf496b8e09879da6ded3b9a15db5c50f6936ee21c6f70a69801311e4fa21b86"

    matrix[0][0] = 0

    assert profile.matrix[0][0] == 1
    assert profile.public_key_sha256.hex() == expected_digest
    profile_attribute = "profile_id"
    with pytest.raises(FrozenInstanceError):
        setattr(profile, profile_attribute, 2)


def test_public_profile_rejects_wrong_identity_shape_range_type_and_rank() -> None:
    """不完整、非规范或退化的本地公开配置必须在加载时失败。"""
    matrix = _gadget_matrix()
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(2, bytes(32), matrix)
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(A4_PROFILE_ID, bytes(31), matrix)
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(A4_PROFILE_ID, bytes(32), matrix[:-1])
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(A4_PROFILE_ID, bytes(32), [row[:-1] for row in matrix])

    out_of_range = _gadget_matrix()
    out_of_range[0][0] = A4_MODULUS
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(A4_PROFILE_ID, bytes(32), out_of_range)

    bool_coefficient = _gadget_matrix()
    bool_coefficient[0][0] = cast(int, True)
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(A4_PROFILE_ID, bytes(32), bool_coefficient)
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(
            A4_PROFILE_ID,
            bytes(32),
            cast(Iterable[Iterable[int]], [[0] * A4_VECTOR_SIZE] * A4_SYNDROME_SIZE),
        )


def test_hash_to_syndrome_has_a_fixed_vector_and_no_profile_override(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
) -> None:
    """SHAKE256 hash-to-syndrome 必须按固定 domain/profile/message/salt 再现。"""
    salt = bytes(range(32, 64))

    syndrome = hash_to_a4_syndrome(a4_message, salt, a4_profile)

    assert syndrome == (255, 236, 39, 203, 25, 108, 206, 192)
    assert all(0 <= value < A4_MODULUS for value in syndrome)
    with pytest.raises(A4ProofParseError):
        hash_to_a4_syndrome(a4_message, bytearray(salt), a4_profile)


def test_reference_accepts_the_public_gadget_conformance_vector(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """公开 gadget fixture 应在不使用私钥的情况下满足精确关系。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))

    evidence = verify_a4_ref(a4_message, proof, a4_profile)

    assert evidence == A4ReferenceEvidence(A4EvidenceCode.RELATION_ACCEPT)
    assert evidence.accepted
    assert {item.name for item in fields(A4ReferenceEvidence)} == {"code"}


def test_reference_distinguishes_norm_and_equation_rejection(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """首个超界系数与界内方程篡改应进入稳定且不同的内部拒绝码。"""
    proof = bytearray(a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64))))
    norm_tamper = proof.copy()
    norm_tamper[33] = 2
    equation_tamper = proof.copy()
    equation_tamper[-1] = 1

    norm_evidence = verify_a4_ref(a4_message, bytes(norm_tamper), a4_profile)
    equation_evidence = verify_a4_ref(a4_message, bytes(equation_tamper), a4_profile)

    assert norm_evidence.code is A4EvidenceCode.NORM_REJECT
    assert equation_evidence.code is A4EvidenceCode.EQUATION_REJECT
    assert not norm_evidence.accepted and not equation_evidence.accepted


@pytest.mark.parametrize(
    "offset",
    [0, 14, 15, 19, 51, 61],
)
def test_reference_rejects_noncanonical_or_wrongly_bound_message_fields(
    offset: int,
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """domain、版本、模型、identity、scope 与 TTL 篡改都应在 relation 前拒绝。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    tampered = bytearray(a4_message)
    tampered[offset] ^= 1

    evidence = verify_a4_ref(bytes(tampered), proof, a4_profile)

    assert evidence.code is A4EvidenceCode.MESSAGE_REJECT


def test_reference_rejects_message_and_profile_type_confusion(
    a4_profile: A4PublicProfile,
) -> None:
    """standalone reference 入口必须对非规范 message/profile fail closed。"""
    proof = bytes([1]) + bytes(A4_PROOF_SIZE - 1)

    assert verify_a4_ref(bytearray(133), proof, a4_profile).code is A4EvidenceCode.MESSAGE_REJECT
    assert verify_a4_ref(bytes(133), proof, object()).code is A4EvidenceCode.CONFIG_REJECT


def test_a3_adapter_binds_exact_accept_and_collapses_rejection(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """A4 adapter 只输出绑定 message/identity 的 A3 evidence。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    rejected = bytearray(proof)
    rejected[33] = 2
    adapter = A4ReferenceAdapter(a4_profile)

    accepted_evidence = adapter(a4_message, proof)
    rejected_evidence = adapter(a4_message, bytes(rejected))
    a3_profile = build_a4_verification_profile(a4_profile)

    assert accepted_evidence.code is A3EvidenceCode.PROOF_ACCEPT
    assert rejected_evidence.code is A3EvidenceCode.PROOF_REJECT
    assert accepted_evidence.identity_id == a4_profile.identity_id
    assert accepted_evidence.message_sha256 == hashlib.sha256(a4_message).digest()
    assert accepted_evidence.profile_id == A4_PROFILE_ID
    assert a3_profile.identity_id == a4_profile.identity_id
    assert a3_profile.profile_id == A4_PROFILE_ID
    assert type(a3_profile.verifier) is A4ReferenceAdapter


def test_message_size_and_domain_match_the_frozen_a3_contract(a4_message: bytes) -> None:
    """A4 reference 输入必须精确复用 A3-v1 的 133 字节消息边界。"""
    assert len(a4_message) == 133
    assert a4_message.startswith(A4_MESSAGE_DOMAIN)
