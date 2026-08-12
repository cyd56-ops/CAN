"""A4 GPV-PFDH toy 公钥关系的 dependency-free 精确参考实现。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

A4_PROFILE_VERSION: Final = 1
A4_PROFILE_ID: Final = 1
A4_MODULUS: Final = 257
A4_SYNDROME_SIZE: Final = 8
A4_VECTOR_SIZE: Final = 72
A4_NORM_BOUND: Final = 1
A4_SALT_SIZE: Final = 32
A4_IDENTITY_SIZE: Final = 32
A4_MESSAGE_SIZE: Final = 133
A4_PROOF_SIZE: Final = 105
A4_MESSAGE_DOMAIN: Final = b"CAN-A3-MSG-v1\x00"
A4_H2S_DOMAIN: Final = b"CAN-A4-GPV-PFDH-H2S-v1\x00"
A4_PUBLIC_KEY_DOMAIN: Final = b"CAN-A4-GPV-PK-v1\x00"

_A3_VERSION: Final = 1
_A3_MODEL_ID: Final = 1
_A3_SCOPE_ID: Final = 1
_A3_TTL_MS: Final = 60_000
_A3_IDENTITY_OFFSET: Final = 19
_A3_SCOPE_OFFSET: Final = 51
_A3_ISSUED_AT_OFFSET: Final = 53
_A3_EXPIRES_AT_OFFSET: Final = 61

A4Matrix = tuple[tuple[int, ...], ...]


class A4ProofParseError(ValueError):
    """表示 A4 proof 不是唯一的 105 字节编码。"""


class A4MessageValidationError(ValueError):
    """表示 A4 输入不是绑定本地 identity 的 canonical A3 message。"""


class A4ProfileValidationError(ValueError):
    """表示本地 A4 公开验证 profile 不满足固定配置约束。"""


class A4EvidenceCode(StrEnum):
    """定义 A4 reference 内部使用的稳定 evidence 码。"""

    MESSAGE_REJECT = "message_reject"
    PROOF_PARSE_REJECT = "proof_parse_reject"
    NORM_REJECT = "norm_reject"
    EQUATION_REJECT = "equation_reject"
    CONFIG_REJECT = "config_reject"
    RELATION_ACCEPT = "relation_accept"


def _rank_mod_q(matrix: A4Matrix) -> int:
    working = [list(row) for row in matrix]
    rank = 0
    for column in range(A4_VECTOR_SIZE):
        pivot = next(
            (
                row_index
                for row_index in range(rank, A4_SYNDROME_SIZE)
                if working[row_index][column] % A4_MODULUS != 0
            ),
            None,
        )
        if pivot is None:
            continue
        working[rank], working[pivot] = working[pivot], working[rank]
        inverse = pow(working[rank][column], -1, A4_MODULUS)
        working[rank] = [value * inverse % A4_MODULUS for value in working[rank]]
        for row_index in range(A4_SYNDROME_SIZE):
            if row_index == rank:
                continue
            factor = working[row_index][column] % A4_MODULUS
            if factor != 0:
                working[row_index] = [
                    (value - factor * pivot_value) % A4_MODULUS
                    for value, pivot_value in zip(working[row_index], working[rank], strict=True)
                ]
        rank += 1
        if rank == A4_SYNDROME_SIZE:
            return rank
    return rank


def _public_key_digest(matrix: A4Matrix) -> bytes:
    payload = bytearray(A4_PUBLIC_KEY_DOMAIN)
    payload.extend(A4_PROFILE_ID.to_bytes(2, byteorder="big", signed=False))
    payload.extend(A4_MODULUS.to_bytes(2, byteorder="big", signed=False))
    payload.extend(A4_SYNDROME_SIZE.to_bytes(2, byteorder="big", signed=False))
    payload.extend(A4_VECTOR_SIZE.to_bytes(2, byteorder="big", signed=False))
    payload.append(A4_NORM_BOUND)
    for row in matrix:
        for coefficient in row:
            payload.extend(coefficient.to_bytes(2, byteorder="big", signed=False))
    return hashlib.sha256(bytes(payload)).digest()


@dataclass(frozen=True, slots=True, init=False)
class A4PublicProfile:
    """保存经过构造期校验且不含私钥的 A4 公共 profile。"""

    profile_id: int
    identity_id: bytes
    matrix: A4Matrix = field(repr=False)
    public_key_sha256: bytes

    def __init__(
        self,
        profile_id: int,
        identity_id: bytes,
        matrix: Iterable[Iterable[int]],
    ) -> None:
        if type(profile_id) is not int or profile_id != A4_PROFILE_ID:
            raise A4ProfileValidationError("A4 profile_id must be exactly 1")
        if type(identity_id) is not bytes or len(identity_id) != A4_IDENTITY_SIZE:
            raise A4ProfileValidationError("A4 identity_id must be exactly 32 bytes")
        try:
            canonical_matrix = tuple(tuple(row) for row in matrix)
        except TypeError as error:
            raise A4ProfileValidationError("A4 matrix must contain iterable rows") from error
        if len(canonical_matrix) != A4_SYNDROME_SIZE:
            raise A4ProfileValidationError("A4 matrix has the wrong row count")
        for row in canonical_matrix:
            if len(row) != A4_VECTOR_SIZE:
                raise A4ProfileValidationError("A4 matrix has the wrong column count")
            if any(
                type(coefficient) is not int or not 0 <= coefficient < A4_MODULUS
                for coefficient in row
            ):
                raise A4ProfileValidationError("A4 matrix contains a non-canonical coefficient")
        if _rank_mod_q(canonical_matrix) != A4_SYNDROME_SIZE:
            raise A4ProfileValidationError("A4 matrix is not full row rank modulo 257")

        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "identity_id", identity_id)
        object.__setattr__(self, "matrix", canonical_matrix)
        object.__setattr__(self, "public_key_sha256", _public_key_digest(canonical_matrix))


@dataclass(frozen=True, slots=True, init=False)
class A4Proof:
    """保存规范解析后的 A4 PFDH salt 与 signed-int8 短向量。"""

    version: int
    salt: bytes
    vector: tuple[int, ...]

    def __init__(self, version: int, salt: bytes, vector: Iterable[int]) -> None:
        if type(version) is not int or version != A4_PROFILE_VERSION:
            raise A4ProofParseError("A4 proof version must be exactly 1")
        if type(salt) is not bytes or len(salt) != A4_SALT_SIZE:
            raise A4ProofParseError("A4 proof salt must be exactly 32 bytes")
        try:
            canonical_vector = tuple(vector)
        except TypeError as error:
            raise A4ProofParseError("A4 proof vector must be iterable") from error
        if len(canonical_vector) != A4_VECTOR_SIZE:
            raise A4ProofParseError("A4 proof vector has the wrong length")
        if any(
            type(coefficient) is not int or not -128 <= coefficient <= 127
            for coefficient in canonical_vector
        ):
            raise A4ProofParseError("A4 proof vector contains a non-canonical int8")

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "salt", salt)
        object.__setattr__(self, "vector", canonical_vector)

    def encode(self) -> bytes:
        """将 proof 编码为唯一的 105 字节形式。"""
        return bytes([self.version]) + self.salt + bytes(value & 0xFF for value in self.vector)


@dataclass(frozen=True, slots=True)
class A4ReferenceEvidence:
    """保存无授权能力的 A4 精确关系判定。"""

    code: A4EvidenceCode

    def __post_init__(self) -> None:
        if type(self.code) is not A4EvidenceCode:
            raise TypeError("A4 evidence code must use the exact enum type")

    @property
    def accepted(self) -> bool:
        """返回是否满足精确 A4 relation, 该值不是授权决定。"""
        return self.code is A4EvidenceCode.RELATION_ACCEPT


def parse_a4_proof(raw_proof: object) -> A4Proof:
    """严格解析唯一的 105 字节 A4 proof。"""
    if type(raw_proof) is not bytes or len(raw_proof) != A4_PROOF_SIZE:
        raise A4ProofParseError("A4 proof must be exactly 105 bytes")
    if raw_proof[0] != A4_PROFILE_VERSION:
        raise A4ProofParseError("A4 proof version is not canonical")
    vector = tuple(value - 256 if value >= 128 else value for value in raw_proof[33:])
    proof = A4Proof(raw_proof[0], raw_proof[1:33], vector)
    if proof.encode() != raw_proof:
        raise A4ProofParseError("A4 proof encoding is not canonical")
    return proof


def _validate_message(raw_message: object, profile: A4PublicProfile) -> bytes:
    if type(raw_message) is not bytes or len(raw_message) != A4_MESSAGE_SIZE:
        raise A4MessageValidationError("A4 message must be exactly 133 bytes")
    if raw_message[: len(A4_MESSAGE_DOMAIN)] != A4_MESSAGE_DOMAIN:
        raise A4MessageValidationError("A4 message domain is not canonical")
    if raw_message[14] != _A3_VERSION:
        raise A4MessageValidationError("A4 message version is not canonical")
    if int.from_bytes(raw_message[15:19], byteorder="big", signed=False) != _A3_MODEL_ID:
        raise A4MessageValidationError("A4 message model is not locally bound")
    if raw_message[_A3_IDENTITY_OFFSET:_A3_SCOPE_OFFSET] != profile.identity_id:
        raise A4MessageValidationError("A4 message identity does not match the local profile")
    if (
        int.from_bytes(
            raw_message[_A3_SCOPE_OFFSET:_A3_ISSUED_AT_OFFSET],
            byteorder="big",
            signed=False,
        )
        != _A3_SCOPE_ID
    ):
        raise A4MessageValidationError("A4 message scope is not locally bound")
    issued_at_ms = int.from_bytes(
        raw_message[_A3_ISSUED_AT_OFFSET:_A3_EXPIRES_AT_OFFSET],
        byteorder="big",
        signed=False,
    )
    expires_at_ms = int.from_bytes(
        raw_message[_A3_EXPIRES_AT_OFFSET : _A3_EXPIRES_AT_OFFSET + 8],
        byteorder="big",
        signed=False,
    )
    if issued_at_ms > (1 << 64) - 1 - _A3_TTL_MS or expires_at_ms != issued_at_ms + _A3_TTL_MS:
        raise A4MessageValidationError("A4 message TTL is not canonical")
    return raw_message


def _hash_to_syndrome_validated(message: bytes, salt: bytes, profile_id: int) -> tuple[int, ...]:
    xof = hashlib.shake_256(
        A4_H2S_DOMAIN + profile_id.to_bytes(2, byteorder="big", signed=False) + message + salt
    )
    output_length = A4_SYNDROME_SIZE * 2
    stream = xof.digest(output_length)
    offset = 0
    syndrome: list[int] = []
    while len(syndrome) < A4_SYNDROME_SIZE:
        if offset + 2 > len(stream):
            output_length += 32
            stream = xof.digest(output_length)
        candidate = int.from_bytes(stream[offset : offset + 2], byteorder="big", signed=False)
        offset += 2
        if candidate != 0xFFFF:
            syndrome.append(candidate % A4_MODULUS)
    return tuple(syndrome)


def hash_to_a4_syndrome(
    raw_message: object,
    salt: object,
    profile: object,
) -> tuple[int, ...]:
    """把 canonical A3 message 与 A4 salt 映射为八个规范 syndrome 系数。"""
    if type(profile) is not A4PublicProfile:
        raise A4ProfileValidationError("A4 hash requires the exact public profile type")
    message = _validate_message(raw_message, profile)
    if type(salt) is not bytes or len(salt) != A4_SALT_SIZE:
        raise A4ProofParseError("A4 hash salt must be exactly 32 bytes")
    return _hash_to_syndrome_validated(message, salt, profile.profile_id)


def verify_a4_ref(
    raw_message: object,
    raw_proof: object,
    profile: object,
) -> A4ReferenceEvidence:
    """执行无私钥 A4 精确公开验证关系并只返回结构化 evidence。"""
    if type(profile) is not A4PublicProfile:
        return A4ReferenceEvidence(A4EvidenceCode.CONFIG_REJECT)
    try:
        message = _validate_message(raw_message, profile)
    except A4MessageValidationError:
        return A4ReferenceEvidence(A4EvidenceCode.MESSAGE_REJECT)
    try:
        proof = parse_a4_proof(raw_proof)
    except A4ProofParseError:
        return A4ReferenceEvidence(A4EvidenceCode.PROOF_PARSE_REJECT)
    if max(abs(coefficient) for coefficient in proof.vector) > A4_NORM_BOUND:
        return A4ReferenceEvidence(A4EvidenceCode.NORM_REJECT)

    target = _hash_to_syndrome_validated(message, proof.salt, profile.profile_id)
    actual = tuple(
        sum(
            profile.matrix[row_index][column_index] * proof.vector[column_index]
            for column_index in range(A4_VECTOR_SIZE)
        )
        % A4_MODULUS
        for row_index in range(A4_SYNDROME_SIZE)
    )
    if actual != target:
        return A4ReferenceEvidence(A4EvidenceCode.EQUATION_REJECT)
    return A4ReferenceEvidence(A4EvidenceCode.RELATION_ACCEPT)


__all__ = [
    "A4_H2S_DOMAIN",
    "A4_IDENTITY_SIZE",
    "A4_MESSAGE_DOMAIN",
    "A4_MESSAGE_SIZE",
    "A4_MODULUS",
    "A4_NORM_BOUND",
    "A4_PROFILE_ID",
    "A4_PROFILE_VERSION",
    "A4_PROOF_SIZE",
    "A4_PUBLIC_KEY_DOMAIN",
    "A4_SALT_SIZE",
    "A4_SYNDROME_SIZE",
    "A4_VECTOR_SIZE",
    "A4EvidenceCode",
    "A4Matrix",
    "A4MessageValidationError",
    "A4ProfileValidationError",
    "A4Proof",
    "A4ProofParseError",
    "A4PublicProfile",
    "A4ReferenceEvidence",
    "hash_to_a4_syndrome",
    "parse_a4_proof",
    "verify_a4_ref",
]
