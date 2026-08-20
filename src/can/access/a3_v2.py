"""A3-v2 model-independent commit-first transcript 与单次终态协调器。"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import Lock, RLock
from typing import Final, Literal, TypeAlias, TypedDict

from can.reference.v1 import (
    V1_CHALLENGE_WEIGHT,
    V1_DIGEST_SIZE,
    V1_IDENTITY_SIZE,
    V1_PROFILE_ID,
    V1_RING_DEGREE,
    V1Challenge,
    parse_v1_abort,
    parse_v1_challenge,
    parse_v1_commitment,
    parse_v1_response,
)

A3_V2_PROTOCOL_VERSION: Final = 2
A3_V2_RESPONSE_VERSION: Final = 4
A3_V2_CHALLENGE_TTL_MS: Final = 60_000
A3_V2_NONCE_SIZE: Final = 32
A3_V2_MESSAGE_DOMAIN: Final = b"CAN-A3-MSG-v2\x00"
A3_V2_BINDING_DOMAIN: Final = b"CAN-V1-MSIS-BIND-v1\x00"
A3_V2_TRANSCRIPT_DOMAIN: Final = b"CAN-V1-MSIS-TRANSCRIPT-v1\x00"
A3_V2_MESSAGE_SIZE: Final = 133

A3V2Verifier = Callable[[bytes, bytes, bytes, bytes], object]
A3V2ProtectedOperation = Callable[[object], object]
A3V2ChallengeSampler = Callable[[int, int], Sequence[int]]


class A3V2DenyEnvelope(TypedDict):
    """定义 A3-v2 固定拒绝响应。"""

    version: Literal[4]
    status: Literal["deny"]


class A3V2ChallengeEnvelope(TypedDict):
    """定义 A3-v2 commit-first challenge 响应。"""

    version: Literal[4]
    status: Literal["challenge"]
    message: bytes
    challenge: bytes
    transcript_id: bytes


class A3V2ProtectedEnvelope(TypedDict):
    """定义已提交且执行一次受保护操作后的固定响应。"""

    version: Literal[4]
    status: Literal["protected"]


A3V2Envelope: TypeAlias = A3V2DenyEnvelope | A3V2ChallengeEnvelope | A3V2ProtectedEnvelope


class A3V2EvidenceCode(Enum):
    """表示 V1 verifier adapter 产生的无授权能力证据码。"""

    RELATION_ACCEPT = "relation_accept"
    RELATION_REJECT = "relation_reject"
    CONFIG_REJECT = "config_reject"


class A3V2RouteDecision(Enum):
    """表示 A3-v2 协调器已经提交的内部 route decision。"""

    DENY = "deny"
    PUBLIC = "public"
    PROTECTED = "protected"


class A3V2ExecutionState(Enum):
    """表示已提交 route 的业务执行状态。"""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class A3V2InternalResultCode(Enum):
    """表示不进入公共响应的稳定内部结果码。"""

    INVALID_RESPONSE = "invalid_response"
    TRANSCRIPT_UNAVAILABLE = "transcript_unavailable"
    TRANSCRIPT_EXPIRED = "transcript_expired"
    ROUTE_UNAVAILABLE = "route_unavailable"
    VERIFICATION_REJECTED = "verification_rejected"
    INTERNAL_STATE_ERROR = "internal_state_error"
    PROTECTED_SUCCEEDED = "protected_succeeded"
    PROTECTED_EXECUTION_ERROR = "protected_execution_error"


class A3V2ProtocolConfigError(ValueError):
    """表示 A3-v2 本地可信 route 配置不满足固定契约。"""


class A3V2ProtocolInputError(ValueError):
    """表示 A3-v2 不可信 wire 输入或 route binding 无效。"""


class A3V2StateError(RuntimeError):
    """表示 A3-v2 可信时钟、随机源或 transcript state 失败。"""


class A3V2ProtectedExecutionError(RuntimeError):
    """表示受保护 callback 已提交后在固定阶段失败。"""

    __slots__ = ("stage",)

    def __init__(self, stage: str) -> None:
        if type(stage) is not str or not stage:
            raise A3V2StateError("A3-v2 execution error stage is not canonical")
        super().__init__(stage)
        self.stage = stage


@dataclass(slots=True)
class _A3V2OperationValue:
    value: object
    consumed: bool = False
    lock: Lock = field(default_factory=Lock)


@dataclass(frozen=True, slots=True)
class A3V2InternalExecutionResult:
    """保存协调器提交的 route、执行状态和一次性交付值。"""

    route_decision: A3V2RouteDecision
    execution_state: A3V2ExecutionState
    code: A3V2InternalResultCode
    failure_stage: str | None = None
    _operation_value: _A3V2OperationValue | None = field(default=None, repr=False)

    def consume_operation_value(self) -> object:
        """只允许可信 adapter 取得一次成功的 protected operation value。"""
        delivery = self._operation_value
        if (
            self.route_decision is not A3V2RouteDecision.PROTECTED
            or self.execution_state is not A3V2ExecutionState.SUCCEEDED
            or self.code is not A3V2InternalResultCode.PROTECTED_SUCCEEDED
            or delivery is None
        ):
            raise A3V2StateError("A3-v2 result has no successful operation value")
        with delivery.lock:
            if delivery.consumed:
                raise A3V2StateError("A3-v2 operation value was already consumed")
            value = delivery.value
            delivery.value = None
            delivery.consumed = True
            return value


@dataclass(frozen=True, slots=True)
class A3V2Clock:
    """封装 A3-v2 使用的可信 wall/monotonic clock。"""

    wall_time_ms: Callable[[], int]
    monotonic_ns: Callable[[], int]

    def __post_init__(self) -> None:
        if not callable(self.wall_time_ms) or not callable(self.monotonic_ns):
            raise A3V2ProtocolConfigError("A3-v2 clock callbacks must be callable")


@dataclass(frozen=True, slots=True)
class A3V2Message:
    """保存 A3-v2 model-independent 133 字节请求绑定消息。"""

    version: int
    model_id: int
    identity_id: bytes
    scope_id: int
    issued_at_ms: int
    expires_at_ms: int
    nonce: bytes
    input_digest: bytes

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != A3_V2_PROTOCOL_VERSION:
            raise A3V2ProtocolInputError("A3-v2 message version is not canonical")
        if type(self.model_id) is not int or not 0 <= self.model_id < 1 << 32:
            raise A3V2ProtocolInputError("A3-v2 message model_id is not canonical")
        if type(self.identity_id) is not bytes or len(self.identity_id) != V1_IDENTITY_SIZE:
            raise A3V2ProtocolInputError("A3-v2 message identity_id is not canonical")
        if type(self.scope_id) is not int or not 0 <= self.scope_id < 1 << 16:
            raise A3V2ProtocolInputError("A3-v2 message scope_id is not canonical")
        if type(self.issued_at_ms) is not int or not 0 <= self.issued_at_ms < 1 << 64:
            raise A3V2ProtocolInputError("A3-v2 issued_at_ms is not canonical")
        if type(self.expires_at_ms) is not int or not 0 <= self.expires_at_ms < 1 << 64:
            raise A3V2ProtocolInputError("A3-v2 expires_at_ms is not canonical")
        if (
            self.issued_at_ms > (1 << 64) - 1 - A3_V2_CHALLENGE_TTL_MS
            or self.expires_at_ms != self.issued_at_ms + A3_V2_CHALLENGE_TTL_MS
        ):
            raise A3V2ProtocolInputError("A3-v2 message TTL is not canonical")
        if type(self.nonce) is not bytes or len(self.nonce) != A3_V2_NONCE_SIZE:
            raise A3V2ProtocolInputError("A3-v2 nonce is not canonical")
        if type(self.input_digest) is not bytes or len(self.input_digest) != V1_DIGEST_SIZE:
            raise A3V2ProtocolInputError("A3-v2 input_digest is not canonical")

    def encode(self) -> bytes:
        """将 A3-v2 消息编码为固定 133 字节唯一形式。"""
        return (
            A3_V2_MESSAGE_DOMAIN
            + bytes([self.version])
            + self.model_id.to_bytes(4, byteorder="big", signed=False)
            + self.identity_id
            + self.scope_id.to_bytes(2, byteorder="big", signed=False)
            + self.issued_at_ms.to_bytes(8, byteorder="big", signed=False)
            + self.expires_at_ms.to_bytes(8, byteorder="big", signed=False)
            + self.nonce
            + self.input_digest
        )


def parse_a3_v2_message(raw_message: object) -> A3V2Message:
    """严格解析唯一的 133 字节 A3-v2 message。"""
    if type(raw_message) is not bytes or len(raw_message) != A3_V2_MESSAGE_SIZE:
        raise A3V2ProtocolInputError("A3-v2 message has the wrong type or length")
    if not raw_message.startswith(A3_V2_MESSAGE_DOMAIN):
        raise A3V2ProtocolInputError("A3-v2 message domain is not canonical")
    offset = len(A3_V2_MESSAGE_DOMAIN)
    message = A3V2Message(
        version=raw_message[offset],
        model_id=int.from_bytes(raw_message[offset + 1 : offset + 5], "big", signed=False),
        identity_id=raw_message[offset + 5 : offset + 37],
        scope_id=int.from_bytes(raw_message[offset + 37 : offset + 39], "big", signed=False),
        issued_at_ms=int.from_bytes(raw_message[offset + 39 : offset + 47], "big", signed=False),
        expires_at_ms=int.from_bytes(raw_message[offset + 47 : offset + 55], "big", signed=False),
        nonce=raw_message[offset + 55 : offset + 87],
        input_digest=raw_message[offset + 87 : offset + 119],
    )
    if message.encode() != raw_message:
        raise A3V2ProtocolInputError("A3-v2 message encoding is not canonical")
    return message


@dataclass(frozen=True, slots=True)
class A3V2TrustedInput:
    """保存仅由本地业务 adapter 构造的摘要与同一 opaque snapshot。"""

    model_id: int
    identity_id: bytes
    scope_id: int
    input_profile_sha256: bytes
    input_digest: bytes
    snapshot: object

    def __post_init__(self) -> None:
        if type(self.model_id) is not int or not 0 <= self.model_id < 1 << 32:
            raise A3V2ProtocolInputError("trusted input model_id is not canonical")
        if type(self.identity_id) is not bytes or len(self.identity_id) != V1_IDENTITY_SIZE:
            raise A3V2ProtocolInputError("trusted input identity_id is not canonical")
        if type(self.scope_id) is not int or not 0 <= self.scope_id < 1 << 16:
            raise A3V2ProtocolInputError("trusted input scope_id is not canonical")
        if (
            type(self.input_profile_sha256) is not bytes
            or len(self.input_profile_sha256) != V1_DIGEST_SIZE
        ):
            raise A3V2ProtocolInputError("trusted input profile digest is not canonical")
        if type(self.input_digest) is not bytes or len(self.input_digest) != V1_DIGEST_SIZE:
            raise A3V2ProtocolInputError("trusted input digest is not canonical")
        if self.snapshot is None:
            raise A3V2ProtocolInputError("trusted input snapshot cannot be None")


@dataclass(frozen=True, slots=True)
class A3V2Evidence:
    """保存 V1 relation adapter 产生的公开摘要绑定 evidence。"""

    code: A3V2EvidenceCode
    identity_id: bytes
    verification_profile_sha256: bytes
    commitment_sha256: bytes
    challenge_sha256: bytes
    response_sha256: bytes
    transcript_id: bytes

    def __post_init__(self) -> None:
        if type(self.code) is not A3V2EvidenceCode:
            raise TypeError("A3-v2 evidence code must use the exact enum type")
        if type(self.identity_id) is not bytes or len(self.identity_id) != V1_IDENTITY_SIZE:
            raise ValueError("A3-v2 evidence identity is not canonical")
        for digest in (
            self.verification_profile_sha256,
            self.commitment_sha256,
            self.challenge_sha256,
            self.response_sha256,
            self.transcript_id,
        ):
            if type(digest) is not bytes or len(digest) != V1_DIGEST_SIZE:
                raise ValueError("A3-v2 evidence digest is not canonical")


@dataclass(frozen=True, slots=True)
class A3V2VerificationProfile:
    """绑定一个 identity、业务 profile、V1 verifier 与受保护操作。"""

    identity_id: bytes
    verification_profile_sha256: bytes
    model_id: int
    scope_id: int
    input_profile_sha256: bytes
    verifier: A3V2Verifier
    protected_operation: A3V2ProtectedOperation

    def __post_init__(self) -> None:
        if type(self.identity_id) is not bytes or len(self.identity_id) != V1_IDENTITY_SIZE:
            raise A3V2ProtocolConfigError("A3-v2 profile identity is not canonical")
        for digest in (self.verification_profile_sha256, self.input_profile_sha256):
            if type(digest) is not bytes or len(digest) != V1_DIGEST_SIZE:
                raise A3V2ProtocolConfigError("A3-v2 profile digest is not canonical")
        if type(self.model_id) is not int or not 0 <= self.model_id < 1 << 32:
            raise A3V2ProtocolConfigError("A3-v2 profile model_id is not canonical")
        if type(self.scope_id) is not int or not 0 <= self.scope_id < 1 << 16:
            raise A3V2ProtocolConfigError("A3-v2 profile scope_id is not canonical")
        if not callable(self.verifier) or not callable(self.protected_operation):
            raise A3V2ProtocolConfigError("A3-v2 profile callbacks must be callable")


def compute_a3_v2_binding_digest(
    verification_profile_sha256: object,
    input_profile_sha256: object,
    message: object,
    commitment: object,
) -> bytes:
    """计算同时绑定验证 profile、业务 profile、message 与 commitment 的摘要。"""
    if (
        type(verification_profile_sha256) is not bytes
        or len(verification_profile_sha256) != V1_DIGEST_SIZE
        or type(input_profile_sha256) is not bytes
        or len(input_profile_sha256) != V1_DIGEST_SIZE
    ):
        raise A3V2ProtocolInputError("A3-v2 profile digest is not canonical")
    parsed_message = parse_a3_v2_message(message)
    canonical_message = parsed_message.encode()
    canonical_commitment = parse_v1_commitment(commitment).encode()
    return hashlib.sha256(
        A3_V2_BINDING_DOMAIN
        + verification_profile_sha256
        + input_profile_sha256
        + canonical_message
        + hashlib.sha256(canonical_commitment).digest()
    ).digest()


def compute_a3_v2_transcript_id(binding_digest: object, challenge: object) -> bytes:
    """计算绑定 server challenge 的唯一 A3-v2 transcript identifier。"""
    if type(binding_digest) is not bytes or len(binding_digest) != V1_DIGEST_SIZE:
        raise A3V2ProtocolInputError("A3-v2 binding digest is not canonical")
    canonical_challenge = parse_v1_challenge(challenge).encode()
    return hashlib.sha256(A3_V2_TRANSCRIPT_DOMAIN + binding_digest + canonical_challenge).digest()


def _default_wall_time_ms() -> int:
    return time.time_ns() // 1_000_000


def _default_monotonic_ns() -> int:
    return time.monotonic_ns()


def _default_challenge_sampler(degree: int, weight: int) -> Sequence[int]:
    support: set[int] = set()
    while len(support) < weight:
        support.add(secrets.randbelow(degree))
    coefficients = [0] * degree
    for index in support:
        coefficients[index] = 1 if secrets.randbits(1) else -1
    return tuple(coefficients)


@dataclass(frozen=True, slots=True)
class _A3V2TranscriptRecord:
    message: bytes
    commitment: bytes
    commitment_sha256: bytes
    challenge: bytes
    challenge_sha256: bytes
    transcript_id: bytes
    identity_id: bytes
    verification_profile_sha256: bytes
    input_profile_sha256: bytes
    snapshot: object | None
    deadline_ns: int
    state: Literal["PENDING", "CLAIMED", "ABORTED", "EXPIRED"]


class A3V2TranscriptStore:
    """提供单进程线程安全的 commitment/nonce/transcript 单次终态状态。"""

    __slots__ = (
        "_clock",
        "_clock_lock",
        "_last_monotonic_ns",
        "_lock",
        "_random_bytes",
        "_records",
        "_used_commitments",
        "_used_nonces",
    )

    def __init__(
        self,
        *,
        clock: A3V2Clock | None = None,
        random_bytes: Callable[[int], bytes] | None = None,
    ) -> None:
        self._clock = (
            A3V2Clock(_default_wall_time_ms, _default_monotonic_ns) if clock is None else clock
        )
        if type(self._clock) is not A3V2Clock:
            raise A3V2ProtocolConfigError("A3-v2 store requires the exact clock type")
        self._random_bytes = secrets.token_bytes if random_bytes is None else random_bytes
        if not callable(self._random_bytes):
            raise A3V2ProtocolConfigError("A3-v2 random source must be callable")
        self._clock_lock = RLock()
        self._last_monotonic_ns: int | None = None
        self._lock = RLock()
        self._records: dict[bytes, _A3V2TranscriptRecord] = {}
        self._used_commitments: set[bytes] = set()
        self._used_nonces: set[bytes] = set()

    def _wall_time_ms(self) -> int:
        value = self._clock.wall_time_ms()
        if type(value) is not int or value < 0 or value >= 1 << 64:
            raise A3V2StateError("trusted wall clock returned a non-canonical value")
        return value

    def _monotonic_ns(self) -> int:
        value = self._clock.monotonic_ns()
        if type(value) is not int or value < 0:
            raise A3V2StateError("trusted monotonic clock returned a non-canonical value")
        with self._clock_lock:
            if self._last_monotonic_ns is not None and value < self._last_monotonic_ns:
                raise A3V2StateError("trusted monotonic clock moved backwards")
            self._last_monotonic_ns = value
        return value

    def _allocate_nonce(self) -> bytes:
        nonce = self._random_bytes(A3_V2_NONCE_SIZE)
        if type(nonce) is not bytes or len(nonce) != A3_V2_NONCE_SIZE:
            raise A3V2StateError("trusted random source returned a non-canonical nonce")
        with self._lock:
            if nonce in self._used_nonces:
                raise A3V2StateError("nonce collision in the current A3-v2 state epoch")
            self._used_nonces.add(nonce)
        return nonce

    def _insert(self, record: _A3V2TranscriptRecord) -> None:
        if type(record) is not _A3V2TranscriptRecord or record.state != "PENDING":
            raise A3V2StateError("A3-v2 store accepts only pending exact records")
        with self._lock:
            if record.commitment_sha256 in self._used_commitments:
                raise A3V2StateError("commitment reuse is forbidden")
            if record.transcript_id in self._records:
                raise A3V2StateError("transcript collision is forbidden")
            self._used_commitments.add(record.commitment_sha256)
            self._records[record.transcript_id] = record

    def _claim(
        self,
        transcript_id: bytes,
        *,
        terminal_state: Literal["CLAIMED", "ABORTED"],
    ) -> tuple[_A3V2TranscriptRecord | None, Literal["claimed", "expired", "unavailable"]]:
        with self._lock:
            now_ns = self._monotonic_ns()
            record = self._records.get(transcript_id)
            if record is None or record.state != "PENDING":
                return None, "unavailable"
            if now_ns >= record.deadline_ns:
                self._records[transcript_id] = replace(
                    record,
                    snapshot=None,
                    state="EXPIRED",
                )
                return None, "expired"
            self._records[transcript_id] = replace(
                record,
                snapshot=None,
                state=terminal_state,
            )
            return record, "claimed"


@dataclass(frozen=True, slots=True)
class A3V2ProtocolSnapshot:
    """提供不含输入、wire transcript 或 evidence 的 A3-v2 计数。"""

    challenge_issues: int
    challenge_denies: int
    terminal_claims: int
    verifier_calls: int
    allow_commits: int
    deny_commits: int
    protected_calls: int
    protected_responses: int
    deny_responses: int
    aborts: int
    expiries: int


def _deny_envelope() -> A3V2DenyEnvelope:
    return {"version": A3_V2_RESPONSE_VERSION, "status": "deny"}


def _internal_deny_result(code: A3V2InternalResultCode) -> A3V2InternalExecutionResult:
    return A3V2InternalExecutionResult(
        A3V2RouteDecision.DENY,
        A3V2ExecutionState.NOT_STARTED,
        code,
    )


class A3V2ProtocolCoordinator:
    """以唯一协调器提交 A3-v2 终态决定并执行受保护操作。"""

    __slots__ = (
        "_abort_count",
        "_allow_commits",
        "_challenge_denies",
        "_challenge_issues",
        "_challenge_sampler",
        "_deny_commits",
        "_deny_responses",
        "_expiry_count",
        "_lock",
        "_profiles",
        "_protected_calls",
        "_protected_responses",
        "_store",
        "_terminal_claims",
        "_verifier_calls",
    )

    def __init__(
        self,
        profiles: Sequence[A3V2VerificationProfile] = (),
        *,
        store: A3V2TranscriptStore | None = None,
        challenge_sampler: A3V2ChallengeSampler | None = None,
    ) -> None:
        if type(profiles) not in (tuple, list):
            raise A3V2ProtocolConfigError("A3-v2 profiles must be a trusted tuple or list")
        indexed: dict[bytes, A3V2VerificationProfile] = {}
        for profile in profiles:
            if type(profile) is not A3V2VerificationProfile:
                raise A3V2ProtocolConfigError("A3-v2 route profile has the wrong type")
            if profile.identity_id in indexed:
                raise A3V2ProtocolConfigError("A3-v2 route identity is duplicated")
            indexed[profile.identity_id] = profile
        if store is not None and type(store) is not A3V2TranscriptStore:
            raise A3V2ProtocolConfigError("A3-v2 store has the wrong type")
        if challenge_sampler is not None and not callable(challenge_sampler):
            raise A3V2ProtocolConfigError("A3-v2 challenge sampler must be callable")
        self._profiles = indexed
        self._store = A3V2TranscriptStore() if store is None else store
        self._challenge_sampler = (
            _default_challenge_sampler if challenge_sampler is None else challenge_sampler
        )
        self._lock = Lock()
        self._challenge_issues = 0
        self._challenge_denies = 0
        self._terminal_claims = 0
        self._verifier_calls = 0
        self._allow_commits = 0
        self._deny_commits = 0
        self._protected_calls = 0
        self._protected_responses = 0
        self._deny_responses = 0
        self._abort_count = 0
        self._expiry_count = 0

    def snapshot(self) -> A3V2ProtocolSnapshot:
        """返回不含业务输入或 transcript 内容的内部计数。"""
        with self._lock:
            return A3V2ProtocolSnapshot(
                challenge_issues=self._challenge_issues,
                challenge_denies=self._challenge_denies,
                terminal_claims=self._terminal_claims,
                verifier_calls=self._verifier_calls,
                allow_commits=self._allow_commits,
                deny_commits=self._deny_commits,
                protected_calls=self._protected_calls,
                protected_responses=self._protected_responses,
                deny_responses=self._deny_responses,
                aborts=self._abort_count,
                expiries=self._expiry_count,
            )

    def _record_challenge(self, issued: bool) -> None:
        with self._lock:
            if issued:
                self._challenge_issues += 1
            else:
                self._challenge_denies += 1

    def _record_deny(
        self,
        *,
        terminal: bool = False,
        expired: bool = False,
        response: bool = True,
    ) -> None:
        with self._lock:
            if terminal:
                self._terminal_claims += 1
            if expired:
                self._expiry_count += 1
            self._deny_commits += 1
            if response:
                self._deny_responses += 1

    def begin(self, trusted_input: object, raw_commitment: object) -> A3V2Envelope:
        """验证 commitment, 并在可信业务摘要已冻结后签发 server challenge。"""
        try:
            if type(trusted_input) is not A3V2TrustedInput:
                raise A3V2ProtocolInputError("A3-v2 requires exact trusted adapter input")
            route = self._profiles.get(trusted_input.identity_id)
            if route is None:
                raise A3V2ProtocolConfigError("A3-v2 identity has no active route")
            if (
                trusted_input.model_id != route.model_id
                or trusted_input.scope_id != route.scope_id
                or trusted_input.input_profile_sha256 != route.input_profile_sha256
            ):
                raise A3V2ProtocolInputError("A3-v2 trusted input route does not match")
            commitment = parse_v1_commitment(raw_commitment)
            if commitment.profile_id != V1_PROFILE_ID:
                raise A3V2ProtocolInputError("A3-v2 commitment profile does not match")
            commitment_bytes = commitment.encode()
            issued_at_ms = self._store._wall_time_ms()
            monotonic_now = self._store._monotonic_ns()
            nonce = self._store._allocate_nonce()
            message = A3V2Message(
                A3_V2_PROTOCOL_VERSION,
                route.model_id,
                route.identity_id,
                route.scope_id,
                issued_at_ms,
                issued_at_ms + A3_V2_CHALLENGE_TTL_MS,
                nonce,
                trusted_input.input_digest,
            ).encode()
            sampled = self._challenge_sampler(V1_RING_DEGREE, V1_CHALLENGE_WEIGHT)
            challenge = V1Challenge(V1_PROFILE_ID, sampled).encode()
            commitment_sha256 = hashlib.sha256(commitment_bytes).digest()
            binding_digest = compute_a3_v2_binding_digest(
                route.verification_profile_sha256,
                route.input_profile_sha256,
                message,
                commitment_bytes,
            )
            transcript_id = compute_a3_v2_transcript_id(binding_digest, challenge)
            self._store._insert(
                _A3V2TranscriptRecord(
                    message=message,
                    commitment=commitment_bytes,
                    commitment_sha256=commitment_sha256,
                    challenge=challenge,
                    challenge_sha256=hashlib.sha256(challenge).digest(),
                    transcript_id=transcript_id,
                    identity_id=route.identity_id,
                    verification_profile_sha256=route.verification_profile_sha256,
                    input_profile_sha256=route.input_profile_sha256,
                    snapshot=trusted_input.snapshot,
                    deadline_ns=monotonic_now + A3_V2_CHALLENGE_TTL_MS * 1_000_000,
                    state="PENDING",
                )
            )
            self._record_challenge(True)
            return {
                "version": A3_V2_RESPONSE_VERSION,
                "status": "challenge",
                "message": message,
                "challenge": challenge,
                "transcript_id": transcript_id,
            }
        except Exception:
            self._record_challenge(False)
            return _deny_envelope()

    def commit_and_execute(self, raw_response: object) -> A3V2InternalExecutionResult:
        """原子 claim、提交 route, 并返回一次性交付的内部执行结果。"""
        terminal_claimed = False
        try:
            response = parse_v1_response(raw_response)
        except Exception:
            self._record_deny(response=False)
            return _internal_deny_result(A3V2InternalResultCode.INVALID_RESPONSE)
        try:
            record, outcome = self._store._claim(
                response.transcript_id,
                terminal_state="CLAIMED",
            )
            if outcome == "expired":
                self._record_deny(terminal=True, expired=True, response=False)
                return _internal_deny_result(A3V2InternalResultCode.TRANSCRIPT_EXPIRED)
            if outcome != "claimed" or record is None:
                self._record_deny(response=False)
                return _internal_deny_result(A3V2InternalResultCode.TRANSCRIPT_UNAVAILABLE)
            terminal_claimed = True
        except Exception:
            self._record_deny(response=False)
            return _internal_deny_result(A3V2InternalResultCode.INTERNAL_STATE_ERROR)
        try:
            route = self._profiles.get(record.identity_id)
            if (
                route is None
                or route.verification_profile_sha256 != record.verification_profile_sha256
                or route.input_profile_sha256 != record.input_profile_sha256
            ):
                self._record_deny(terminal=True, response=False)
                return _internal_deny_result(A3V2InternalResultCode.ROUTE_UNAVAILABLE)
            with self._lock:
                self._verifier_calls += 1
            response_bytes = response.encode()
            evidence = route.verifier(
                record.commitment,
                record.challenge,
                response_bytes,
                record.transcript_id,
            )
            if (
                type(evidence) is not A3V2Evidence
                or evidence.code is not A3V2EvidenceCode.RELATION_ACCEPT
                or evidence.identity_id != record.identity_id
                or evidence.verification_profile_sha256 != record.verification_profile_sha256
                or evidence.commitment_sha256 != record.commitment_sha256
                or evidence.challenge_sha256 != record.challenge_sha256
                or evidence.response_sha256 != hashlib.sha256(response_bytes).digest()
                or evidence.transcript_id != record.transcript_id
            ):
                self._record_deny(terminal=True, response=False)
                return _internal_deny_result(A3V2InternalResultCode.VERIFICATION_REJECTED)
            if record.snapshot is None:
                self._record_deny(terminal=True, response=False)
                return _internal_deny_result(A3V2InternalResultCode.INTERNAL_STATE_ERROR)
            with self._lock:
                self._terminal_claims += 1
                self._allow_commits += 1
                self._protected_calls += 1
            try:
                value = route.protected_operation(record.snapshot)
                return A3V2InternalExecutionResult(
                    A3V2RouteDecision.PROTECTED,
                    A3V2ExecutionState.SUCCEEDED,
                    A3V2InternalResultCode.PROTECTED_SUCCEEDED,
                    None,
                    _A3V2OperationValue(value),
                )
            except A3V2ProtectedExecutionError as error:
                return A3V2InternalExecutionResult(
                    A3V2RouteDecision.PROTECTED,
                    A3V2ExecutionState.FAILED,
                    A3V2InternalResultCode.PROTECTED_EXECUTION_ERROR,
                    error.stage,
                )
            except Exception:
                return A3V2InternalExecutionResult(
                    A3V2RouteDecision.PROTECTED,
                    A3V2ExecutionState.FAILED,
                    A3V2InternalResultCode.PROTECTED_EXECUTION_ERROR,
                    None,
                )
        except Exception:
            self._record_deny(terminal=terminal_claimed, response=False)
            return _internal_deny_result(A3V2InternalResultCode.INTERNAL_STATE_ERROR)

    def respond(self, raw_response: object) -> A3V2Envelope:
        """保持 C1 version-4 status-only 响应并丢弃内部 operation value。"""
        result = self.commit_and_execute(raw_response)
        if (
            result.route_decision is A3V2RouteDecision.PROTECTED
            and result.execution_state is A3V2ExecutionState.SUCCEEDED
        ):
            try:
                result.consume_operation_value()
            except A3V2StateError:
                with self._lock:
                    self._deny_responses += 1
                return _deny_envelope()
            with self._lock:
                self._protected_responses += 1
            protected: A3V2ProtectedEnvelope = {
                "version": A3_V2_RESPONSE_VERSION,
                "status": "protected",
            }
            return protected
        with self._lock:
            self._deny_responses += 1
        return _deny_envelope()

    def abort(self, raw_abort: object) -> A3V2DenyEnvelope:
        """原子终结一个规范 abort, 且不调用 verifier 或受保护操作。"""
        try:
            abort = parse_v1_abort(raw_abort)
            record, outcome = self._store._claim(
                abort.transcript_id,
                terminal_state="ABORTED",
            )
            if outcome == "expired":
                self._record_deny(terminal=True, expired=True)
                return _deny_envelope()
            if outcome != "claimed" or record is None:
                raise A3V2ProtocolInputError("A3-v2 abort transcript is unavailable")
            with self._lock:
                self._abort_count += 1
            self._record_deny(terminal=True)
            return _deny_envelope()
        except Exception:
            self._record_deny()
            return _deny_envelope()


__all__ = [
    "A3_V2_BINDING_DOMAIN",
    "A3_V2_CHALLENGE_TTL_MS",
    "A3_V2_MESSAGE_DOMAIN",
    "A3_V2_MESSAGE_SIZE",
    "A3_V2_NONCE_SIZE",
    "A3_V2_PROTOCOL_VERSION",
    "A3_V2_RESPONSE_VERSION",
    "A3_V2_TRANSCRIPT_DOMAIN",
    "A3V2ChallengeEnvelope",
    "A3V2Clock",
    "A3V2DenyEnvelope",
    "A3V2Envelope",
    "A3V2Evidence",
    "A3V2EvidenceCode",
    "A3V2ExecutionState",
    "A3V2InternalExecutionResult",
    "A3V2InternalResultCode",
    "A3V2Message",
    "A3V2ProtectedEnvelope",
    "A3V2ProtectedExecutionError",
    "A3V2ProtocolConfigError",
    "A3V2ProtocolCoordinator",
    "A3V2ProtocolInputError",
    "A3V2ProtocolSnapshot",
    "A3V2RouteDecision",
    "A3V2StateError",
    "A3V2TranscriptStore",
    "A3V2TrustedInput",
    "A3V2VerificationProfile",
    "compute_a3_v2_binding_digest",
    "compute_a3_v2_transcript_id",
    "parse_a3_v2_message",
]
