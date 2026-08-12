"""V1-P2 Module-SIS 非生产 profile 与系数域精确参考实现。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

V1_PROTOCOL_ID: Final = "CAN-V1-FSWA-MSIS-ID-v1"
V1_PROFILE_ID: Final = 1
V1_RING_DEGREE: Final = 8
V1_MODULUS: Final = 257
V1_MODULE_ROWS: Final = 2
V1_MODULE_COLUMNS: Final = 2
V1_SECRET_BOUND: Final = 1
V1_MASK_BOUND: Final = 8
V1_CHALLENGE_WEIGHT: Final = 2
V1_RESPONSE_BOUND: Final = 6
V1_IDENTITY_SIZE: Final = 32
V1_DIGEST_SIZE: Final = 32

V1_COMMITMENT_DOMAIN: Final = b"CAN-V1-MSIS-COM-v1\x00"
V1_CHALLENGE_DOMAIN: Final = b"CAN-V1-MSIS-CHAL-v1\x00"
V1_RESPONSE_DOMAIN: Final = b"CAN-V1-MSIS-RESP-v1\x00"
V1_ABORT_DOMAIN: Final = b"CAN-V1-MSIS-ABORT-v1\x00"
V1_PUBLIC_KEY_DOMAIN: Final = b"CAN-V1-MSIS-PK-v1\x00"

V1_RESPONSE_POLYNOMIALS: Final = V1_MODULE_COLUMNS + V1_MODULE_ROWS
V1_COMMITMENT_SIZE: Final = len(V1_COMMITMENT_DOMAIN) + 2 + V1_MODULE_ROWS * V1_RING_DEGREE * 4
V1_CHALLENGE_SIZE: Final = len(V1_CHALLENGE_DOMAIN) + 2 + V1_RING_DEGREE
V1_RESPONSE_SIZE: Final = (
    len(V1_RESPONSE_DOMAIN) + V1_DIGEST_SIZE + V1_RESPONSE_POLYNOMIALS * V1_RING_DEGREE * 4
)
V1_ABORT_SIZE: Final = len(V1_ABORT_DOMAIN) + V1_DIGEST_SIZE

V1Polynomial = tuple[int, ...]
V1ModuleVector = tuple[V1Polynomial, ...]
V1ModuleMatrix = tuple[tuple[V1Polynomial, ...], ...]


class V1ProfileValidationError(ValueError):
    """表示 V1-P2 本地公开 profile 不满足固定 conformance 契约。"""


class V1RegistryValidationError(ValueError):
    """表示 V1-P2 本地公开 registry 配置无效。"""


class V1RegistryLookupError(LookupError):
    """表示 identity 没有唯一启用的 V1-P2 公开 profile。"""


class V1WireParseError(ValueError):
    """表示 V1-P2 wire 输入不是唯一规范编码。"""


class V1EvidenceCode(StrEnum):
    """定义 V1-P2 exact reference 的稳定 evidence 码。"""

    COMMITMENT_PARSE_REJECT = "commitment_parse_reject"
    CHALLENGE_PARSE_REJECT = "challenge_parse_reject"
    RESPONSE_PARSE_REJECT = "response_parse_reject"
    TRANSCRIPT_REJECT = "transcript_reject"
    NORM_REJECT = "norm_reject"
    EQUATION_REJECT = "equation_reject"
    CONFIG_REJECT = "config_reject"
    RELATION_ACCEPT = "relation_accept"


def _canonical_nested(
    values: Iterable[Iterable[int]],
    *,
    outer_size: int,
    inner_size: int,
    minimum: int,
    maximum: int,
    error_type: type[ValueError],
    label: str,
) -> tuple[tuple[int, ...], ...]:
    try:
        canonical = tuple(tuple(row) for row in values)
    except TypeError as error:
        raise error_type(f"{label} must contain iterable polynomials") from error
    if len(canonical) != outer_size or any(len(row) != inner_size for row in canonical):
        raise error_type(f"{label} has the wrong shape")
    if any(
        type(coefficient) is not int or not minimum <= coefficient <= maximum
        for row in canonical
        for coefficient in row
    ):
        raise error_type(f"{label} contains a non-canonical coefficient")
    return canonical


def _canonical_matrix(values: Iterable[Iterable[Iterable[int]]]) -> V1ModuleMatrix:
    try:
        rows = tuple(tuple(tuple(polynomial) for polynomial in row) for row in values)
    except TypeError as error:
        raise V1ProfileValidationError("V1 matrix must contain iterable polynomial rows") from error
    if len(rows) != V1_MODULE_ROWS or any(len(row) != V1_MODULE_COLUMNS for row in rows):
        raise V1ProfileValidationError("V1 matrix has the wrong module shape")
    if any(len(polynomial) != V1_RING_DEGREE for row in rows for polynomial in row):
        raise V1ProfileValidationError("V1 matrix polynomial has the wrong degree")
    if any(
        type(coefficient) is not int or not 0 <= coefficient < V1_MODULUS
        for row in rows
        for polynomial in row
        for coefficient in polynomial
    ):
        raise V1ProfileValidationError("V1 matrix contains a non-canonical coefficient")
    if any(
        all(coefficient == 0 for polynomial in row for coefficient in polynomial) for row in rows
    ):
        raise V1ProfileValidationError("V1 matrix contains an all-zero module row")
    return rows


def _public_key_digest(
    identity_id: bytes,
    matrix: V1ModuleMatrix,
    target: V1ModuleVector,
) -> bytes:
    payload = bytearray(V1_PUBLIC_KEY_DOMAIN)
    payload.extend(V1_PROFILE_ID.to_bytes(2, byteorder="big", signed=False))
    for value in (
        V1_RING_DEGREE,
        V1_MODULUS,
        V1_MODULE_ROWS,
        V1_MODULE_COLUMNS,
        V1_SECRET_BOUND,
        V1_MASK_BOUND,
        V1_CHALLENGE_WEIGHT,
        V1_RESPONSE_BOUND,
    ):
        payload.extend(value.to_bytes(4, byteorder="big", signed=False))
    payload.extend(identity_id)
    for row in matrix:
        for polynomial in row:
            for coefficient in polynomial:
                payload.extend(coefficient.to_bytes(4, byteorder="big", signed=False))
    for polynomial in target:
        for coefficient in polynomial:
            payload.extend(coefficient.to_bytes(4, byteorder="big", signed=False))
    return hashlib.sha256(bytes(payload)).digest()


@dataclass(frozen=True, slots=True, init=False)
class V1PublicProfile:
    """保存构造期校验且不含 secret 的 V1-P2 公开 conformance profile。"""

    profile_id: int
    identity_id: bytes
    matrix: V1ModuleMatrix = field(repr=False)
    target: V1ModuleVector = field(repr=False)
    public_key_sha256: bytes

    def __init__(
        self,
        profile_id: int,
        identity_id: bytes,
        matrix: Iterable[Iterable[Iterable[int]]],
        target: Iterable[Iterable[int]],
    ) -> None:
        if type(profile_id) is not int or profile_id != V1_PROFILE_ID:
            raise V1ProfileValidationError("V1 profile_id must be exactly 1")
        if type(identity_id) is not bytes or len(identity_id) != V1_IDENTITY_SIZE:
            raise V1ProfileValidationError("V1 identity_id must be exactly 32 bytes")
        canonical_matrix = _canonical_matrix(matrix)
        canonical_target = _canonical_nested(
            target,
            outer_size=V1_MODULE_ROWS,
            inner_size=V1_RING_DEGREE,
            minimum=0,
            maximum=V1_MODULUS - 1,
            error_type=V1ProfileValidationError,
            label="V1 target",
        )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "identity_id", identity_id)
        object.__setattr__(self, "matrix", canonical_matrix)
        object.__setattr__(self, "target", canonical_target)
        object.__setattr__(
            self,
            "public_key_sha256",
            _public_key_digest(identity_id, canonical_matrix, canonical_target),
        )


class V1PublicRegistry:
    """以 identity 唯一索引不可变 V1-P2 公开 profile。"""

    __slots__ = ("_profiles",)

    def __init__(self, profiles: Sequence[V1PublicProfile]) -> None:
        if type(profiles) not in (tuple, list):
            raise V1RegistryValidationError("V1 profiles must be a trusted tuple or list")
        indexed: dict[bytes, V1PublicProfile] = {}
        for profile in profiles:
            if type(profile) is not V1PublicProfile:
                raise V1RegistryValidationError("V1 registry entry has the wrong type")
            if profile.identity_id in indexed:
                raise V1RegistryValidationError("V1 registry identity is duplicated")
            indexed[profile.identity_id] = profile
        self._profiles = indexed

    def lookup(self, identity_id: object) -> V1PublicProfile:
        """返回 identity 的唯一公开 profile, 未知或错误类型时拒绝。"""
        if type(identity_id) is not bytes or len(identity_id) != V1_IDENTITY_SIZE:
            raise V1RegistryLookupError("V1 identity lookup is not canonical")
        profile = self._profiles.get(identity_id)
        if profile is None:
            raise V1RegistryLookupError("unknown V1 identity")
        return profile


@dataclass(frozen=True, slots=True, init=False)
class V1Commitment:
    """保存规范解析后的公开 commitment module vector。"""

    profile_id: int
    polynomials: V1ModuleVector

    def __init__(self, profile_id: int, polynomials: Iterable[Iterable[int]]) -> None:
        if type(profile_id) is not int or profile_id != V1_PROFILE_ID:
            raise V1WireParseError("V1 commitment profile is not canonical")
        canonical = _canonical_nested(
            polynomials,
            outer_size=V1_MODULE_ROWS,
            inner_size=V1_RING_DEGREE,
            minimum=0,
            maximum=V1_MODULUS - 1,
            error_type=V1WireParseError,
            label="V1 commitment",
        )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "polynomials", canonical)

    def encode(self) -> bytes:
        """将 commitment 编码为固定宽度唯一 wire bytes。"""
        payload = bytearray(V1_COMMITMENT_DOMAIN)
        payload.extend(self.profile_id.to_bytes(2, byteorder="big", signed=False))
        for polynomial in self.polynomials:
            for coefficient in polynomial:
                payload.extend(coefficient.to_bytes(4, byteorder="big", signed=False))
        return bytes(payload)


@dataclass(frozen=True, slots=True, init=False)
class V1Challenge:
    """保存规范 fixed-weight ternary challenge polynomial。"""

    profile_id: int
    coefficients: V1Polynomial

    def __init__(self, profile_id: int, coefficients: Iterable[int]) -> None:
        if type(profile_id) is not int or profile_id != V1_PROFILE_ID:
            raise V1WireParseError("V1 challenge profile is not canonical")
        try:
            canonical = tuple(coefficients)
        except TypeError as error:
            raise V1WireParseError("V1 challenge coefficients must be iterable") from error
        if len(canonical) != V1_RING_DEGREE:
            raise V1WireParseError("V1 challenge has the wrong degree")
        if any(type(value) is not int or value not in (-1, 0, 1) for value in canonical):
            raise V1WireParseError("V1 challenge coefficient is not canonical ternary")
        if sum(value != 0 for value in canonical) != V1_CHALLENGE_WEIGHT:
            raise V1WireParseError("V1 challenge has the wrong Hamming weight")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "coefficients", canonical)

    def encode(self) -> bytes:
        """将 challenge 编码为 fixed-weight ternary 唯一 wire bytes。"""
        return (
            V1_CHALLENGE_DOMAIN
            + self.profile_id.to_bytes(2, byteorder="big", signed=False)
            + bytes(value & 0xFF for value in self.coefficients)
        )


@dataclass(frozen=True, slots=True, init=False)
class V1Response:
    """保存绑定 transcript 的 signed-int32 response module vector。"""

    transcript_id: bytes
    polynomials: V1ModuleVector

    def __init__(self, transcript_id: bytes, polynomials: Iterable[Iterable[int]]) -> None:
        if type(transcript_id) is not bytes or len(transcript_id) != V1_DIGEST_SIZE:
            raise V1WireParseError("V1 response transcript_id must be exactly 32 bytes")
        canonical = _canonical_nested(
            polynomials,
            outer_size=V1_RESPONSE_POLYNOMIALS,
            inner_size=V1_RING_DEGREE,
            minimum=-(1 << 31),
            maximum=(1 << 31) - 1,
            error_type=V1WireParseError,
            label="V1 response",
        )
        object.__setattr__(self, "transcript_id", transcript_id)
        object.__setattr__(self, "polynomials", canonical)

    def encode(self) -> bytes:
        """将 response 编码为 transcript 加 signed-int32 唯一 wire bytes。"""
        payload = bytearray(V1_RESPONSE_DOMAIN)
        payload.extend(self.transcript_id)
        for polynomial in self.polynomials:
            for coefficient in polynomial:
                payload.extend(coefficient.to_bytes(4, byteorder="big", signed=True))
        return bytes(payload)


@dataclass(frozen=True, slots=True)
class V1Abort:
    """保存一个显式终结 transcript 的规范 abort。"""

    transcript_id: bytes

    def __post_init__(self) -> None:
        if type(self.transcript_id) is not bytes or len(self.transcript_id) != V1_DIGEST_SIZE:
            raise V1WireParseError("V1 abort transcript_id must be exactly 32 bytes")

    def encode(self) -> bytes:
        """将 abort 编码为固定 domain 与 transcript identifier。"""
        return V1_ABORT_DOMAIN + self.transcript_id


@dataclass(frozen=True, slots=True)
class V1ReferenceEvidence:
    """保存不具有授权能力的 V1-P2 exact relation 判定。"""

    code: V1EvidenceCode

    def __post_init__(self) -> None:
        if type(self.code) is not V1EvidenceCode:
            raise TypeError("V1 evidence code must use the exact enum type")

    @property
    def accepted(self) -> bool:
        """返回是否满足 exact relation, 该值不是授权决定。"""
        return self.code is V1EvidenceCode.RELATION_ACCEPT


def parse_v1_commitment(raw_commitment: object) -> V1Commitment:
    """严格解析唯一的 V1-P2 commitment 编码。"""
    if type(raw_commitment) is not bytes or len(raw_commitment) != V1_COMMITMENT_SIZE:
        raise V1WireParseError("V1 commitment has the wrong type or length")
    if not raw_commitment.startswith(V1_COMMITMENT_DOMAIN):
        raise V1WireParseError("V1 commitment domain is not canonical")
    offset = len(V1_COMMITMENT_DOMAIN)
    profile_id = int.from_bytes(raw_commitment[offset : offset + 2], "big", signed=False)
    offset += 2
    coefficients = tuple(
        int.from_bytes(raw_commitment[index : index + 4], "big", signed=False)
        for index in range(offset, len(raw_commitment), 4)
    )
    commitment = V1Commitment(
        profile_id,
        tuple(
            coefficients[index : index + V1_RING_DEGREE]
            for index in range(0, len(coefficients), V1_RING_DEGREE)
        ),
    )
    if commitment.encode() != raw_commitment:
        raise V1WireParseError("V1 commitment encoding is not canonical")
    return commitment


def parse_v1_challenge(raw_challenge: object) -> V1Challenge:
    """严格解析唯一的 V1-P2 fixed-weight challenge 编码。"""
    if type(raw_challenge) is not bytes or len(raw_challenge) != V1_CHALLENGE_SIZE:
        raise V1WireParseError("V1 challenge has the wrong type or length")
    if not raw_challenge.startswith(V1_CHALLENGE_DOMAIN):
        raise V1WireParseError("V1 challenge domain is not canonical")
    offset = len(V1_CHALLENGE_DOMAIN)
    profile_id = int.from_bytes(raw_challenge[offset : offset + 2], "big", signed=False)
    coefficients = tuple(
        value - 256 if value >= 128 else value for value in raw_challenge[offset + 2 :]
    )
    challenge = V1Challenge(profile_id, coefficients)
    if challenge.encode() != raw_challenge:
        raise V1WireParseError("V1 challenge encoding is not canonical")
    return challenge


def parse_v1_response(raw_response: object) -> V1Response:
    """严格解析唯一的 V1-P2 signed-int32 response 编码。"""
    if type(raw_response) is not bytes or len(raw_response) != V1_RESPONSE_SIZE:
        raise V1WireParseError("V1 response has the wrong type or length")
    if not raw_response.startswith(V1_RESPONSE_DOMAIN):
        raise V1WireParseError("V1 response domain is not canonical")
    offset = len(V1_RESPONSE_DOMAIN)
    transcript_id = raw_response[offset : offset + V1_DIGEST_SIZE]
    offset += V1_DIGEST_SIZE
    coefficients = tuple(
        int.from_bytes(raw_response[index : index + 4], "big", signed=True)
        for index in range(offset, len(raw_response), 4)
    )
    response = V1Response(
        transcript_id,
        tuple(
            coefficients[index : index + V1_RING_DEGREE]
            for index in range(0, len(coefficients), V1_RING_DEGREE)
        ),
    )
    if response.encode() != raw_response:
        raise V1WireParseError("V1 response encoding is not canonical")
    return response


def parse_v1_abort(raw_abort: object) -> V1Abort:
    """严格解析唯一的 V1-P2 terminal abort 编码。"""
    if type(raw_abort) is not bytes or len(raw_abort) != V1_ABORT_SIZE:
        raise V1WireParseError("V1 abort has the wrong type or length")
    if not raw_abort.startswith(V1_ABORT_DOMAIN):
        raise V1WireParseError("V1 abort domain is not canonical")
    abort = V1Abort(raw_abort[len(V1_ABORT_DOMAIN) :])
    if abort.encode() != raw_abort:
        raise V1WireParseError("V1 abort encoding is not canonical")
    return abort


def v1_negacyclic_convolution(
    left: Sequence[int],
    right: Sequence[int],
) -> V1Polynomial:
    """按 ``X^N=-1`` 返回未约减的 V1-P2 exact negacyclic convolution。"""
    if type(left) not in (tuple, list) or type(right) not in (tuple, list):
        raise TypeError("V1 convolution inputs must be exact tuples or lists")
    if len(left) != V1_RING_DEGREE or len(right) != V1_RING_DEGREE:
        raise ValueError("V1 convolution inputs have the wrong degree")
    if any(type(value) is not int for value in (*left, *right)):
        raise TypeError("V1 convolution coefficients must be exact integers")
    output = [0] * V1_RING_DEGREE
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            degree = left_index + right_index
            if degree < V1_RING_DEGREE:
                output[degree] += left_value * right_value
            else:
                output[degree - V1_RING_DEGREE] -= left_value * right_value
    return tuple(output)


def _module_lhs(profile: V1PublicProfile, response: V1Response) -> V1ModuleVector:
    output: list[V1Polynomial] = []
    for row_index in range(V1_MODULE_ROWS):
        coefficients = [0] * V1_RING_DEGREE
        for column_index in range(V1_MODULE_COLUMNS):
            product = v1_negacyclic_convolution(
                profile.matrix[row_index][column_index],
                response.polynomials[column_index],
            )
            coefficients = [
                value + product_value
                for value, product_value in zip(coefficients, product, strict=True)
            ]
        coefficients = [
            value + identity_value
            for value, identity_value in zip(
                coefficients,
                response.polynomials[V1_MODULE_COLUMNS + row_index],
                strict=True,
            )
        ]
        output.append(tuple(value % V1_MODULUS for value in coefficients))
    return tuple(output)


def _module_rhs(
    profile: V1PublicProfile,
    commitment: V1Commitment,
    challenge: V1Challenge,
) -> V1ModuleVector:
    output: list[V1Polynomial] = []
    for row_index in range(V1_MODULE_ROWS):
        product = v1_negacyclic_convolution(challenge.coefficients, profile.target[row_index])
        output.append(
            tuple(
                (commitment_value + product_value) % V1_MODULUS
                for commitment_value, product_value in zip(
                    commitment.polynomials[row_index], product, strict=True
                )
            )
        )
    return tuple(output)


def verify_v1_ref(
    raw_commitment: object,
    raw_challenge: object,
    raw_response: object,
    expected_transcript_id: object,
    profile: object,
) -> V1ReferenceEvidence:
    """执行 V1-P2 coefficient-domain exact relation 并只返回 evidence。"""
    if type(profile) is not V1PublicProfile:
        return V1ReferenceEvidence(V1EvidenceCode.CONFIG_REJECT)
    try:
        commitment = parse_v1_commitment(raw_commitment)
    except V1WireParseError:
        return V1ReferenceEvidence(V1EvidenceCode.COMMITMENT_PARSE_REJECT)
    try:
        challenge = parse_v1_challenge(raw_challenge)
    except V1WireParseError:
        return V1ReferenceEvidence(V1EvidenceCode.CHALLENGE_PARSE_REJECT)
    try:
        response = parse_v1_response(raw_response)
    except V1WireParseError:
        return V1ReferenceEvidence(V1EvidenceCode.RESPONSE_PARSE_REJECT)
    if (
        type(expected_transcript_id) is not bytes
        or len(expected_transcript_id) != V1_DIGEST_SIZE
        or response.transcript_id != expected_transcript_id
        or commitment.profile_id != profile.profile_id
        or challenge.profile_id != profile.profile_id
    ):
        return V1ReferenceEvidence(V1EvidenceCode.TRANSCRIPT_REJECT)
    if any(
        abs(coefficient) > V1_RESPONSE_BOUND
        for polynomial in response.polynomials
        for coefficient in polynomial
    ):
        return V1ReferenceEvidence(V1EvidenceCode.NORM_REJECT)
    if _module_lhs(profile, response) != _module_rhs(profile, commitment, challenge):
        return V1ReferenceEvidence(V1EvidenceCode.EQUATION_REJECT)
    return V1ReferenceEvidence(V1EvidenceCode.RELATION_ACCEPT)


V1_CONFORMANCE_MATRIX: Final[V1ModuleMatrix] = (
    (
        (1, 2, 3, 4, 5, 6, 7, 8),
        (9, 10, 11, 12, 13, 14, 15, 16),
    ),
    (
        (17, 18, 19, 20, 21, 22, 23, 24),
        (25, 26, 27, 28, 29, 30, 31, 32),
    ),
)
V1_CONFORMANCE_TARGET: Final[V1ModuleVector] = (
    (33, 34, 35, 36, 37, 38, 39, 40),
    (41, 42, 43, 44, 45, 46, 47, 48),
)


def build_v1_conformance_profile(identity_id: bytes) -> V1PublicProfile:
    """用固定公开向量构造不可用于生产安全的 V1-P2 conformance profile。"""
    return V1PublicProfile(
        V1_PROFILE_ID,
        identity_id,
        V1_CONFORMANCE_MATRIX,
        V1_CONFORMANCE_TARGET,
    )


__all__ = [
    "V1_ABORT_DOMAIN",
    "V1_ABORT_SIZE",
    "V1_CHALLENGE_DOMAIN",
    "V1_CHALLENGE_SIZE",
    "V1_CHALLENGE_WEIGHT",
    "V1_COMMITMENT_DOMAIN",
    "V1_COMMITMENT_SIZE",
    "V1_CONFORMANCE_MATRIX",
    "V1_CONFORMANCE_TARGET",
    "V1_DIGEST_SIZE",
    "V1_IDENTITY_SIZE",
    "V1_MASK_BOUND",
    "V1_MODULE_COLUMNS",
    "V1_MODULE_ROWS",
    "V1_MODULUS",
    "V1_PROFILE_ID",
    "V1_PROTOCOL_ID",
    "V1_PUBLIC_KEY_DOMAIN",
    "V1_RESPONSE_BOUND",
    "V1_RESPONSE_DOMAIN",
    "V1_RESPONSE_POLYNOMIALS",
    "V1_RESPONSE_SIZE",
    "V1_RING_DEGREE",
    "V1_SECRET_BOUND",
    "V1Abort",
    "V1Challenge",
    "V1Commitment",
    "V1EvidenceCode",
    "V1ModuleMatrix",
    "V1ModuleVector",
    "V1Polynomial",
    "V1ProfileValidationError",
    "V1PublicProfile",
    "V1PublicRegistry",
    "V1ReferenceEvidence",
    "V1RegistryLookupError",
    "V1RegistryValidationError",
    "V1Response",
    "V1WireParseError",
    "build_v1_conformance_profile",
    "parse_v1_abort",
    "parse_v1_challenge",
    "parse_v1_commitment",
    "parse_v1_response",
    "v1_negacyclic_convolution",
    "verify_v1_ref",
]
