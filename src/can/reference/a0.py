"""A0-v1 toy LWE 数值解锁的精确整数参考实现。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

A0_VERSION: Final = 1
A0_PROFILE_ID: Final = 1
A0_SECRET_SIZE: Final = 32
A0_COMPONENT_COUNT: Final = 8
A0_MODULUS: Final = 257
A0_CENTER: Final = 128
A0_ISSUER_RADIUS: Final = 4
A0_REFERENCE_RADIUS: Final = 12
A0_CREDENTIAL_SIZE: Final = 23

Matrix = tuple[tuple[int, ...], ...]


class CredentialParseError(ValueError):
    """表示 A0 credential 不是唯一规范编码。"""


class RegistryValidationError(ValueError):
    """表示本地可信 registry 配置不满足 A0-v1 约束。"""


class RegistryLookupError(LookupError):
    """表示 credential 未解析到启用的本地 slot。"""


class A0EvidenceCode(StrEnum):
    """定义仅供内部实验和测试使用的稳定参考证据码。"""

    PARSE_REJECT = "parse_reject"
    REGISTRY_REJECT = "registry_reject"
    CONFIG_REJECT = "config_reject"
    ISSUER_CORE = "issuer_core"
    REFERENCE_GUARD = "reference_guard"
    REJECT = "reject"


@dataclass(frozen=True, slots=True, init=False)
class A0Credential:
    """保存通过结构和数值规范化检查的 A0-v1 credential。"""

    version: int
    profile_id: int
    slot_id: int
    b: tuple[int, ...]

    def __init__(
        self,
        version: int,
        profile_id: int,
        slot_id: int,
        b: Iterable[int],
    ) -> None:
        if type(version) is not int or version != A0_VERSION:
            raise CredentialParseError("unsupported A0 version")
        if type(profile_id) is not int or not 0 <= profile_id <= 0xFFFF:
            raise CredentialParseError("profile_id is not a canonical uint16")
        if type(slot_id) is not int or not 0 <= slot_id <= 0xFFFFFFFF:
            raise CredentialParseError("slot_id is not a canonical uint32")

        try:
            canonical_b = tuple(b)
        except TypeError as error:
            raise CredentialParseError("b must be an iterable of integers") from error
        if len(canonical_b) != A0_COMPONENT_COUNT:
            raise CredentialParseError("b has the wrong component count")
        if any(type(value) is not int or not 0 <= value < A0_MODULUS for value in canonical_b):
            raise CredentialParseError("b contains a non-canonical coefficient")

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "b", canonical_b)


@dataclass(frozen=True, slots=True, init=False)
class A0Slot:
    """保存一个经过完整校验且不可变的本地 A0 slot。"""

    slot_id: int
    matrix: Matrix
    enabled: bool

    def __init__(
        self,
        slot_id: int,
        matrix: Iterable[Iterable[int]],
        *,
        enabled: bool = True,
    ) -> None:
        if type(slot_id) is not int or not 0 <= slot_id <= 0xFFFFFFFF:
            raise RegistryValidationError("slot_id is not a canonical uint32")
        if type(enabled) is not bool:
            raise RegistryValidationError("enabled must be exactly bool")

        try:
            canonical_matrix = tuple(tuple(row) for row in matrix)
        except TypeError as error:
            raise RegistryValidationError("matrix must contain iterable rows") from error
        if len(canonical_matrix) != A0_COMPONENT_COUNT:
            raise RegistryValidationError("matrix has the wrong row count")

        for row in canonical_matrix:
            if len(row) != A0_SECRET_SIZE:
                raise RegistryValidationError("matrix has the wrong column count")
            if any(
                type(coefficient) is not int or not 0 <= coefficient < A0_MODULUS
                for coefficient in row
            ):
                raise RegistryValidationError("matrix contains a non-canonical coefficient")
            if all(coefficient == 0 for coefficient in row):
                raise RegistryValidationError("matrix contains an all-zero row")

        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "matrix", canonical_matrix)
        object.__setattr__(self, "enabled", enabled)


@dataclass(frozen=True, slots=True, init=False)
class A0Registry:
    """提供不可变、无回退的 A0-v1 本地 slot 查询。"""

    _slots: Mapping[int, A0Slot] = field(repr=False)

    def __init__(self, slots: Iterable[A0Slot]) -> None:
        try:
            slot_iterator = iter(slots)
        except TypeError as error:
            raise RegistryValidationError("slots must be iterable") from error

        indexed_slots: dict[int, A0Slot] = {}
        for slot in slot_iterator:
            if type(slot) is not A0Slot:
                raise RegistryValidationError("registry entries must be exactly A0Slot")
            if slot.slot_id in indexed_slots:
                raise RegistryValidationError("registry contains a duplicate slot_id")
            indexed_slots[slot.slot_id] = slot

        object.__setattr__(self, "_slots", MappingProxyType(indexed_slots))

    @property
    def slots(self) -> tuple[A0Slot, ...]:
        """返回不会暴露内部映射的不可变 slot 快照。"""
        return tuple(self._slots.values())

    def lookup(self, profile_id: int, slot_id: int) -> A0Slot:
        """按固定 profile 和 slot 查找启用项。未知项不执行回退。"""
        if type(profile_id) is not int or profile_id != A0_PROFILE_ID:
            raise RegistryLookupError("unknown A0 profile")
        if type(slot_id) is not int:
            raise RegistryLookupError("slot_id must be exactly int")

        slot = self._slots.get(slot_id)
        if slot is None or not slot.enabled:
            raise RegistryLookupError("unknown or disabled A0 slot")
        return slot


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    """记录无授权能力的 A0 参考判定证据。"""

    code: A0EvidenceCode
    distances: tuple[int, ...] | None = None
    maximum_distance: int | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not A0EvidenceCode:
            raise TypeError("code must be exactly A0EvidenceCode")

        non_verification_codes = {
            A0EvidenceCode.PARSE_REJECT,
            A0EvidenceCode.REGISTRY_REJECT,
            A0EvidenceCode.CONFIG_REJECT,
        }
        if self.code in non_verification_codes:
            if self.distances is not None or self.maximum_distance is not None:
                raise ValueError("pre-verification evidence cannot contain distances")
            return

        if self.distances is None or len(self.distances) != A0_COMPONENT_COUNT:
            raise ValueError("verification evidence requires eight distances")
        if any(type(value) is not int or not 0 <= value <= A0_CENTER for value in self.distances):
            raise ValueError("verification evidence contains an invalid distance")
        if self.maximum_distance != max(self.distances):
            raise ValueError("maximum_distance does not match distances")
        if self.code is A0EvidenceCode.ISSUER_CORE and self.maximum_distance > A0_ISSUER_RADIUS:
            raise ValueError("issuer-core evidence exceeds its radius")
        if self.code is A0EvidenceCode.REFERENCE_GUARD and not (
            A0_ISSUER_RADIUS < self.maximum_distance <= A0_REFERENCE_RADIUS
        ):
            raise ValueError("reference-guard evidence is outside its region")
        if self.code is A0EvidenceCode.REJECT and self.maximum_distance <= A0_REFERENCE_RADIUS:
            raise ValueError("reject evidence is inside the reference radius")

    @property
    def accepted(self) -> bool:
        """返回参考 relation 的判定。该证据不创建 gate 或 capability。"""
        return self.code in {
            A0EvidenceCode.ISSUER_CORE,
            A0EvidenceCode.REFERENCE_GUARD,
        }


def parse_credential(raw_credential: bytes) -> A0Credential:
    """严格解析唯一的 23 字节 A0-v1 credential 编码。"""
    if type(raw_credential) is not bytes:
        raise CredentialParseError("credential must be exactly bytes")
    if len(raw_credential) != A0_CREDENTIAL_SIZE:
        raise CredentialParseError("credential must contain exactly 23 bytes")

    version = raw_credential[0]
    if version != A0_VERSION:
        raise CredentialParseError("unsupported A0 version")

    profile_id = int.from_bytes(raw_credential[1:3], byteorder="big", signed=False)
    slot_id = int.from_bytes(raw_credential[3:7], byteorder="big", signed=False)
    b = tuple(
        int.from_bytes(raw_credential[offset : offset + 2], byteorder="big", signed=False)
        for offset in range(7, A0_CREDENTIAL_SIZE, 2)
    )
    return A0Credential(version, profile_id, slot_id, b)


def mod_q(value: int) -> int:
    """按 A0-v1 语义返回位于 ``[0, 256]`` 的规范模剩余。"""
    if type(value) is not int:
        raise TypeError("value must be exactly int")
    return value % A0_MODULUS


def center_q(value: int) -> int:
    """按 A0-v1 语义返回位于 ``[-128, 128]`` 的中心化剩余。"""
    remainder = mod_q(value)
    if remainder <= A0_CENTER:
        return remainder
    return remainder - A0_MODULUS


def verify_ref(
    raw_credential: bytes,
    registry: A0Registry,
    s_test: Sequence[int],
) -> ReferenceEvidence:
    """执行精确 A0-v1 relation。函数只返回无授权能力的结构化证据。"""
    try:
        credential = parse_credential(raw_credential)
    except CredentialParseError:
        return ReferenceEvidence(A0EvidenceCode.PARSE_REJECT)

    if type(registry) is not A0Registry:
        return ReferenceEvidence(A0EvidenceCode.CONFIG_REJECT)

    try:
        slot = registry.lookup(credential.profile_id, credential.slot_id)
    except RegistryLookupError:
        return ReferenceEvidence(A0EvidenceCode.REGISTRY_REJECT)

    try:
        secret = tuple(s_test)
    except TypeError:
        return ReferenceEvidence(A0EvidenceCode.CONFIG_REJECT)
    if len(secret) != A0_SECRET_SIZE or any(
        type(coefficient) is not int or coefficient not in (0, 1) for coefficient in secret
    ):
        return ReferenceEvidence(A0EvidenceCode.CONFIG_REJECT)

    distances = tuple(
        abs(
            center_q(
                mod_q(
                    credential.b[row_index]
                    - sum(
                        slot.matrix[row_index][column_index] * secret[column_index]
                        for column_index in range(A0_SECRET_SIZE)
                    )
                )
                - A0_CENTER
            )
        )
        for row_index in range(A0_COMPONENT_COUNT)
    )
    maximum_distance = max(distances)

    if maximum_distance <= A0_ISSUER_RADIUS:
        code = A0EvidenceCode.ISSUER_CORE
    elif maximum_distance <= A0_REFERENCE_RADIUS:
        code = A0EvidenceCode.REFERENCE_GUARD
    else:
        code = A0EvidenceCode.REJECT
    return ReferenceEvidence(code, distances, maximum_distance)
