"""V1-P2-PSR-E1 非生产 prover、采样器与公开向量清单。"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, cast

from can.access import (
    A3V2ChallengeEnvelope,
    A3V2ProtocolCoordinator,
    A3V2ProtocolSnapshot,
    A3V2TrustedInput,
    parse_a3_v2_message,
)
from can.reference import (
    V1_CONFORMANCE_MATRIX,
    V1_DIGEST_SIZE,
    V1_MASK_BOUND,
    V1_MODULE_COLUMNS,
    V1_MODULE_ROWS,
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_PROTOCOL_ID,
    V1_RESPONSE_BOUND,
    V1_RESPONSE_POLYNOMIALS,
    V1_RING_DEGREE,
    V1_SECRET_BOUND,
    V1Abort,
    V1Challenge,
    V1Commitment,
    V1EvidenceCode,
    V1ModuleMatrix,
    V1ModuleVector,
    V1Polynomial,
    V1PublicProfile,
    V1Response,
    parse_v1_challenge,
    v1_negacyclic_convolution,
    verify_v1_ref,
)

V1_PSR_EXPERIMENT_ID: Final = "V1-P2-PSR-E1"
V1_PSR_MANIFEST_SCHEMA_VERSION: Final = 1
V1_PSR_SEED_SIZE: Final = 32
V1_PSR_BLOCK_SIZE: Final = 64
V1_PSR_SECRET_DOMAIN: Final = b"CAN-V1-PSR-SECRET-v1\x00"
V1_PSR_MASK_DOMAIN: Final = b"CAN-V1-PSR-MASK-v1\x00"
V1_PSR_CHALLENGE_DOMAIN: Final = b"CAN-V1-PSR-CHALLENGE-v1\x00"
V1_PSR_TARGET_DIGEST_DOMAIN: Final = b"CAN-V1-PSR-TARGET-DIGEST-v1\x00"
V1_PSR_THEORETICAL_EMIT_PROBABILITY: Final = (13 / 17) ** 32


class V1PSRInputError(ValueError):
    """表示 V1-P2-PSR-E1 实验输入不满足固定契约。"""


class V1PSRLifecycleError(RuntimeError):
    """表示调用方试图复用已释放的临时 prover fixture。"""


class V1PSRManifestError(RuntimeError):
    """表示公开向量清单无法按固定策略生成或保存。"""


class V1PSRRetryError(RuntimeError):
    """表示 A3-v2 retry harness 遇到无法继续的可信协议状态。"""


class V1PSROutcome(StrEnum):
    """定义单次 toy prover attempt 的公开结果。"""

    EMIT = "emit"
    ABORT = "abort"


class V1PSRCoordinatorOutcome(StrEnum):
    """定义一个 transcript 经 A3-v2 终结后的公开结果。"""

    PROTECTED = "protected"
    ABORTED = "aborted"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class V1PSRRetryRecord:
    """保存一次 retry 的公开摘要和实验阶段耗时。"""

    retry_index: int
    prover_outcome: V1PSROutcome
    coordinator_outcome: V1PSRCoordinatorOutcome
    nonce_sha256: bytes
    transcript_sha256: bytes
    commitment_sha256: bytes
    challenge_sha256: bytes
    sampler_ns: int
    response_ns: int
    exact_ns: int
    a3_ns: int
    total_ns: int
    replay_denies: int
    concurrent_denies: int

    def __post_init__(self) -> None:
        _validate_counter(self.retry_index, "retry_index")
        if type(self.prover_outcome) is not V1PSROutcome:
            raise V1PSRInputError("retry record prover outcome has the wrong type")
        if type(self.coordinator_outcome) is not V1PSRCoordinatorOutcome:
            raise V1PSRInputError("retry record coordinator outcome has the wrong type")
        for digest in (
            self.nonce_sha256,
            self.transcript_sha256,
            self.commitment_sha256,
            self.challenge_sha256,
        ):
            if type(digest) is not bytes or len(digest) != V1_DIGEST_SIZE:
                raise V1PSRInputError("retry record contains a non-canonical digest")
        for value in (
            self.sampler_ns,
            self.response_ns,
            self.exact_ns,
            self.a3_ns,
            self.total_ns,
            self.replay_denies,
            self.concurrent_denies,
        ):
            if type(value) is not int or value < 0:
                raise V1PSRInputError("retry record metric is outside its fixed range")

    def public_record(self) -> dict[str, object]:
        """返回不含 response、secret 或原始 transcript 的公开记录。"""
        return {
            "a3_ns": self.a3_ns,
            "challenge_sha256": self.challenge_sha256.hex(),
            "commitment_sha256": self.commitment_sha256.hex(),
            "concurrent_denies": self.concurrent_denies,
            "coordinator_outcome": self.coordinator_outcome.value,
            "exact_ns": self.exact_ns,
            "nonce_sha256": self.nonce_sha256.hex(),
            "prover_outcome": self.prover_outcome.value,
            "replay_denies": self.replay_denies,
            "response_ns": self.response_ns,
            "retry_index": self.retry_index,
            "sampler_ns": self.sampler_ns,
            "total_ns": self.total_ns,
            "transcript_sha256": self.transcript_sha256.hex(),
        }


@dataclass(frozen=True, slots=True)
class V1PSRRetryReport:
    """保存一次 trial 的 retry 结果、计数和分阶段耗时摘要。"""

    trial_index: int
    max_attempts: int
    outcome: Literal["protected", "exhausted"]
    attempts: tuple[V1PSRRetryRecord, ...]
    retry_exhausted: bool
    protected_calls: int
    exact_false_rejects: int

    def __post_init__(self) -> None:
        _validate_counter(self.trial_index, "trial_index")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise V1PSRInputError("max_attempts must be a positive exact integer")
        if type(self.outcome) is not str or self.outcome not in ("protected", "exhausted"):
            raise V1PSRInputError("retry report outcome is not canonical")
        if type(self.attempts) is not tuple or not self.attempts:
            raise V1PSRInputError("retry report requires at least one attempt")
        if len(self.attempts) > self.max_attempts:
            raise V1PSRInputError("retry report contains too many attempts")
        if tuple(item.retry_index for item in self.attempts) != tuple(range(len(self.attempts))):
            raise V1PSRInputError("retry report indexes are not contiguous")
        if type(self.retry_exhausted) is not bool or type(self.protected_calls) is not int:
            raise V1PSRInputError("retry report counters have the wrong type")
        if self.protected_calls < 0 or type(self.exact_false_rejects) is not int:
            raise V1PSRInputError("retry report counters are outside their fixed range")
        if self.exact_false_rejects < 0:
            raise V1PSRInputError("retry report false-reject count is negative")
        if self.outcome == "protected" and self.retry_exhausted:
            raise V1PSRInputError("protected retry report cannot be exhausted")
        if self.outcome == "exhausted" and not self.retry_exhausted:
            raise V1PSRInputError("exhausted retry report must be marked exhausted")

    @property
    def attempt_count(self) -> int:
        """返回本 trial 实际创建的 transcript 数量。"""
        return len(self.attempts)

    def public_record(self) -> dict[str, object]:
        """返回只含摘要、计数和 latency 的规范公开报告。"""
        return {
            "attempt_count": self.attempt_count,
            "attempts": [attempt.public_record() for attempt in self.attempts],
            "exact_false_rejects": self.exact_false_rejects,
            "experiment_id": V1_PSR_EXPERIMENT_ID,
            "max_attempts": self.max_attempts,
            "outcome": self.outcome,
            "protected_calls": self.protected_calls,
            "retry_exhausted": self.retry_exhausted,
            "schema_version": V1_PSR_MANIFEST_SCHEMA_VERSION,
            "trial_index": self.trial_index,
        }


def _validate_seed(seed: object) -> bytes:
    if type(seed) is not bytes or len(seed) != V1_PSR_SEED_SIZE:
        raise V1PSRInputError("V1 PSR seed must be exactly 32 bytes")
    return seed


def _validate_counter(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise V1PSRInputError(f"{label} must be a canonical unsigned 64-bit integer")
    return value


def _bind_terminal(
    coordinator: A3V2ProtocolCoordinator,
    operation: Literal["abort", "respond"],
    raw_terminal: bytes,
) -> Callable[[], object]:
    """绑定一个不可变 terminal wire object, 供并发 replay 测量使用。"""
    if operation == "abort":

        def invoke() -> object:
            return coordinator.abort(raw_terminal)
    else:

        def invoke() -> object:
            return coordinator.respond(raw_terminal)

    return invoke


def _shake_stream(
    role_domain: bytes,
    seed: bytes,
    trial_index: int,
    retry_index: int,
) -> Iterator[int]:
    block_index = 0
    while True:
        payload = b"".join(
            (
                role_domain,
                seed,
                trial_index.to_bytes(8, byteorder="big", signed=False),
                retry_index.to_bytes(8, byteorder="big", signed=False),
                block_index.to_bytes(4, byteorder="big", signed=False),
            )
        )
        yield from hashlib.shake_256(payload).digest(V1_PSR_BLOCK_SIZE)
        block_index += 1
        if block_index >= 1 << 32:
            raise V1PSRInputError("V1 PSR SHAKE256 block counter exhausted")


def _sample_rejection_values(
    stream: Iterator[int],
    *,
    count: int,
    acceptance_limit: int,
    modulus: int,
    offset: int,
) -> tuple[int, ...]:
    output: list[int] = []
    while len(output) < count:
        value = next(stream)
        if value < acceptance_limit:
            output.append((value % modulus) + offset)
    return tuple(output)


def _polynomialize(coefficients: Sequence[int]) -> V1ModuleVector:
    return tuple(
        tuple(coefficients[offset : offset + V1_RING_DEGREE])
        for offset in range(0, len(coefficients), V1_RING_DEGREE)
    )


def sample_v1_secret(seed: bytes) -> V1ModuleVector:
    """从固定 secret role stream 均匀采样四个短多项式。"""
    canonical_seed = _validate_seed(seed)
    values = _sample_rejection_values(
        _shake_stream(V1_PSR_SECRET_DOMAIN, canonical_seed, 0, 0),
        count=V1_RESPONSE_POLYNOMIALS * V1_RING_DEGREE,
        acceptance_limit=255,
        modulus=2 * V1_SECRET_BOUND + 1,
        offset=-V1_SECRET_BOUND,
    )
    return _polynomialize(values)


def sample_v1_mask(seed: bytes, trial_index: int, retry_index: int) -> V1ModuleVector:
    """按 trial/retry counter 均匀采样四个 fresh bounded mask 多项式。"""
    canonical_seed = _validate_seed(seed)
    trial = _validate_counter(trial_index, "trial_index")
    retry = _validate_counter(retry_index, "retry_index")
    values = _sample_rejection_values(
        _shake_stream(V1_PSR_MASK_DOMAIN, canonical_seed, trial, retry),
        count=V1_RESPONSE_POLYNOMIALS * V1_RING_DEGREE,
        acceptance_limit=255,
        modulus=2 * V1_MASK_BOUND + 1,
        offset=-V1_MASK_BOUND,
    )
    return _polynomialize(values)


def v1_challenge_set() -> tuple[V1Challenge, ...]:
    """按位置和符号字典序返回固定的 112 个服务端 challenge。"""
    challenges: list[V1Challenge] = []
    for first_position in range(V1_RING_DEGREE):
        for second_position in range(first_position + 1, V1_RING_DEGREE):
            for first_sign in (-1, 1):
                for second_sign in (-1, 1):
                    coefficients = [0] * V1_RING_DEGREE
                    coefficients[first_position] = first_sign
                    coefficients[second_position] = second_sign
                    challenges.append(V1Challenge(V1_PROFILE_ID, coefficients))
    return tuple(challenges)


V1_PSR_CHALLENGE_SET: Final = v1_challenge_set()


def sample_v1_challenge(seed: bytes, trial_index: int, retry_index: int) -> V1Challenge:
    """从可信 challenge role stream 均匀选择固定集合中的一个挑战。"""
    canonical_seed = _validate_seed(seed)
    trial = _validate_counter(trial_index, "trial_index")
    retry = _validate_counter(retry_index, "retry_index")
    index = _sample_rejection_values(
        _shake_stream(V1_PSR_CHALLENGE_DOMAIN, canonical_seed, trial, retry),
        count=1,
        acceptance_limit=224,
        modulus=len(V1_PSR_CHALLENGE_SET),
        offset=0,
    )[0]
    return V1_PSR_CHALLENGE_SET[index]


def _canonical_short_vector(
    values: Iterable[Iterable[int]],
    *,
    bound: int,
    label: str,
) -> V1ModuleVector:
    try:
        canonical = tuple(tuple(polynomial) for polynomial in values)
    except TypeError as error:
        raise V1PSRInputError(f"{label} must contain iterable polynomials") from error
    if len(canonical) != V1_RESPONSE_POLYNOMIALS or any(
        len(polynomial) != V1_RING_DEGREE for polynomial in canonical
    ):
        raise V1PSRInputError(f"{label} has the wrong shape")
    if any(
        type(coefficient) is not int or not -bound <= coefficient <= bound
        for polynomial in canonical
        for coefficient in polynomial
    ):
        raise V1PSRInputError(f"{label} contains a non-canonical coefficient")
    return canonical


def _module_action(
    matrix: V1ModuleMatrix,
    vector: V1ModuleVector,
) -> V1ModuleVector:
    output: list[V1Polynomial] = []
    for row_index in range(V1_MODULE_ROWS):
        coefficients = [0] * V1_RING_DEGREE
        for column_index in range(V1_MODULE_COLUMNS):
            product = v1_negacyclic_convolution(
                matrix[row_index][column_index], vector[column_index]
            )
            coefficients = [
                value + product_value
                for value, product_value in zip(coefficients, product, strict=True)
            ]
        coefficients = [
            value + identity_value
            for value, identity_value in zip(
                coefficients,
                vector[V1_MODULE_COLUMNS + row_index],
                strict=True,
            )
        ]
        output.append(tuple(value % V1_MODULUS for value in coefficients))
    return tuple(output)


def compute_v1_commitment(profile: V1PublicProfile, mask: Iterable[Iterable[int]]) -> V1Commitment:
    """复用公开 matrix 计算 ``u=Abar*y`` 的规范 commitment。"""
    if type(profile) is not V1PublicProfile:
        raise V1PSRInputError("profile must be exactly V1PublicProfile")
    canonical_mask = _canonical_short_vector(mask, bound=V1_MASK_BOUND, label="V1 mask")
    return V1Commitment(profile.profile_id, _module_action(profile.matrix, canonical_mask))


def compute_v1_response(
    secret: Iterable[Iterable[int]],
    mask: Iterable[Iterable[int]],
    challenge: V1Challenge,
) -> V1ModuleVector:
    """按未约减整数语义计算 ``z=y+c*s``。"""
    canonical_secret = _canonical_short_vector(secret, bound=V1_SECRET_BOUND, label="V1 secret")
    canonical_mask = _canonical_short_vector(mask, bound=V1_MASK_BOUND, label="V1 mask")
    if type(challenge) is not V1Challenge:
        raise V1PSRInputError("challenge must be exactly V1Challenge")
    return tuple(
        tuple(
            mask_value + shift_value
            for mask_value, shift_value in zip(mask_polynomial, shift, strict=True)
        )
        for mask_polynomial, secret_polynomial in zip(canonical_mask, canonical_secret, strict=True)
        for shift in (v1_negacyclic_convolution(challenge.coefficients, secret_polynomial),)
    )


def v1_response_emits(response: Iterable[Iterable[int]]) -> bool:
    """返回 response 是否满足固定 ``B=6`` rejection 边界。"""
    try:
        canonical = tuple(tuple(polynomial) for polynomial in response)
    except TypeError as error:
        raise V1PSRInputError("V1 response must contain iterable polynomials") from error
    if len(canonical) != V1_RESPONSE_POLYNOMIALS or any(
        len(polynomial) != V1_RING_DEGREE for polynomial in canonical
    ):
        raise V1PSRInputError("V1 response has the wrong shape")
    if any(type(coefficient) is not int for polynomial in canonical for coefficient in polynomial):
        raise V1PSRInputError("V1 response contains a non-integer coefficient")
    return all(
        abs(coefficient) <= V1_RESPONSE_BOUND
        for polynomial in canonical
        for coefficient in polynomial
    )


def _target_digest(target: V1ModuleVector) -> bytes:
    payload = bytearray(V1_PSR_TARGET_DIGEST_DOMAIN)
    for polynomial in target:
        for coefficient in polynomial:
            payload.extend(coefficient.to_bytes(4, byteorder="big", signed=False))
    return hashlib.sha256(bytes(payload)).digest()


@dataclass(frozen=True, slots=True)
class V1PSRAttempt:
    """保存一次 transcript 的公开对象和可选 emitted response。"""

    trial_index: int
    retry_index: int
    commitment: V1Commitment
    challenge: V1Challenge
    outcome: V1PSROutcome
    response: V1Response | None = field(repr=False)
    profile_sha256: bytes
    seed_sha256: bytes
    target_sha256: bytes

    def __post_init__(self) -> None:
        _validate_counter(self.trial_index, "trial_index")
        _validate_counter(self.retry_index, "retry_index")
        if type(self.commitment) is not V1Commitment or type(self.challenge) is not V1Challenge:
            raise V1PSRInputError("V1 PSR attempt contains non-canonical public objects")
        if type(self.outcome) is not V1PSROutcome:
            raise V1PSRInputError("V1 PSR attempt outcome has the wrong type")
        if (self.outcome is V1PSROutcome.EMIT) is not (type(self.response) is V1Response):
            raise V1PSRInputError("V1 PSR attempt outcome and response disagree")
        for digest in (self.profile_sha256, self.seed_sha256, self.target_sha256):
            if type(digest) is not bytes or len(digest) != V1_DIGEST_SIZE:
                raise V1PSRInputError("V1 PSR attempt contains a non-canonical digest")

    def public_record(self) -> dict[str, object]:
        """返回不含 seed、secret、mask、response 或 transcript 的公开清单记录。"""
        return {
            "challenge_sha256": hashlib.sha256(self.challenge.encode()).hexdigest(),
            "commitment_sha256": hashlib.sha256(self.commitment.encode()).hexdigest(),
            "outcome": self.outcome.value,
            "profile_id": V1_PROFILE_ID,
            "profile_sha256": self.profile_sha256.hex(),
            "protocol_id": V1_PROTOCOL_ID,
            "retry_index": self.retry_index,
            "seed_sha256": self.seed_sha256.hex(),
            "target_sha256": self.target_sha256.hex(),
            "trial_index": self.trial_index,
        }


@dataclass(frozen=True, slots=True)
class _V1PSRPreparedAttempt:
    """保存 commitment 后、challenge 前的 transcript-local mask。"""

    trial_index: int
    retry_index: int
    commitment: V1Commitment
    mask: V1ModuleVector = field(repr=False)
    profile_sha256: bytes
    seed_sha256: bytes
    target_sha256: bytes


class V1GeneratedKeyFixture:
    """持有可显式释放的临时 toy secret 与对应公开 profile。"""

    __slots__ = ("_closed", "_secret", "_seed", "profile", "seed_sha256", "target_sha256")

    def __init__(self, identity_id: bytes, seed: bytes) -> None:
        canonical_seed = _validate_seed(seed)
        secret = sample_v1_secret(canonical_seed)
        target = _module_action(V1_CONFORMANCE_MATRIX, secret)
        self.profile = V1PublicProfile(
            V1_PROFILE_ID,
            identity_id,
            V1_CONFORMANCE_MATRIX,
            target,
        )
        self.seed_sha256 = hashlib.sha256(canonical_seed).digest()
        self.target_sha256 = _target_digest(target)
        self._seed = bytearray(canonical_seed)
        self._secret = [list(polynomial) for polynomial in secret]
        self._closed = False

    @property
    def closed(self) -> bool:
        """返回临时 seed 与 secret 是否已被逻辑释放。"""
        return self._closed

    def prove_attempt(
        self,
        trial_index: int,
        retry_index: int,
        transcript_id: bytes,
    ) -> V1PSRAttempt:
        """生成一个 fresh mask/challenge 并返回 emit 或 abort 结果。"""
        prepared = self._prepare_attempt(trial_index, retry_index)
        challenge = sample_v1_challenge(
            bytes(self._seed), prepared.trial_index, prepared.retry_index
        )
        return self._finish_attempt(prepared, challenge, transcript_id)

    def _prepare_attempt(self, trial_index: int, retry_index: int) -> _V1PSRPreparedAttempt:
        """在 trusted challenge 前生成 fresh mask 和 commitment。"""
        if self._closed:
            raise V1PSRLifecycleError("V1 generated-key fixture is closed")
        trial = _validate_counter(trial_index, "trial_index")
        retry = _validate_counter(retry_index, "retry_index")
        seed = bytes(self._seed)
        mask = sample_v1_mask(seed, trial, retry)
        commitment = compute_v1_commitment(self.profile, mask)
        return _V1PSRPreparedAttempt(
            trial,
            retry,
            commitment,
            mask,
            self.profile.public_key_sha256,
            self.seed_sha256,
            self.target_sha256,
        )

    def _finish_attempt(
        self,
        prepared: _V1PSRPreparedAttempt,
        challenge: V1Challenge,
        transcript_id: bytes,
    ) -> V1PSRAttempt:
        """使用 coordinator 提供的 challenge 计算单次 response。"""
        if self._closed:
            raise V1PSRLifecycleError("V1 generated-key fixture is closed")
        if type(prepared) is not _V1PSRPreparedAttempt:
            raise V1PSRInputError("prepared V1 attempt has the wrong type")
        if type(challenge) is not V1Challenge:
            raise V1PSRInputError("challenge must be exactly V1Challenge")
        if type(transcript_id) is not bytes or len(transcript_id) != V1_DIGEST_SIZE:
            raise V1PSRInputError("transcript_id must be exactly 32 bytes")
        response_polynomials = compute_v1_response(self._secret, prepared.mask, challenge)
        emits = v1_response_emits(response_polynomials)
        response = V1Response(transcript_id, response_polynomials) if emits else None
        return V1PSRAttempt(
            prepared.trial_index,
            prepared.retry_index,
            prepared.commitment,
            challenge,
            V1PSROutcome.EMIT if emits else V1PSROutcome.ABORT,
            response,
            prepared.profile_sha256,
            prepared.seed_sha256,
            prepared.target_sha256,
        )

    def _abort_attempt(
        self,
        prepared: _V1PSRPreparedAttempt,
        challenge: V1Challenge,
    ) -> V1PSRAttempt:
        """在不计算或保留 response 的情况下标记 trusted abort。"""
        if self._closed:
            raise V1PSRLifecycleError("V1 generated-key fixture is closed")
        if type(prepared) is not _V1PSRPreparedAttempt or type(challenge) is not V1Challenge:
            raise V1PSRInputError("abort attempt contains a non-canonical object")
        return V1PSRAttempt(
            prepared.trial_index,
            prepared.retry_index,
            prepared.commitment,
            challenge,
            V1PSROutcome.ABORT,
            None,
            prepared.profile_sha256,
            prepared.seed_sha256,
            prepared.target_sha256,
        )

    def close(self) -> None:
        """覆盖并释放 fixture 持有的临时 seed 与 toy secret。"""
        if self._closed:
            return
        for index in range(len(self._seed)):
            self._seed[index] = 0
        for polynomial in self._secret:
            for index in range(len(polynomial)):
                polynomial[index] = 0
            polynomial.clear()
        self._secret.clear()
        self._closed = True

    def __enter__(self) -> V1GeneratedKeyFixture:
        """进入临时 generated-key fixture 生命周期。"""
        if self._closed:
            raise V1PSRLifecycleError("V1 generated-key fixture is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """无论成功或异常都释放临时 seed 与 toy secret。"""
        self.close()


def build_v1_generated_key_fixture(identity_id: bytes, seed: bytes) -> V1GeneratedKeyFixture:
    """用固定公开 matrix 和临时 toy secret 构造 generated-key fixture。"""
    return V1GeneratedKeyFixture(identity_id, seed)


def _require_challenge_envelope(value: object) -> A3V2ChallengeEnvelope:
    if type(value) is not dict or value.get("status") != "challenge":
        raise V1PSRRetryError("A3-v2 coordinator did not issue a challenge")
    required = {"version", "status", "message", "challenge", "transcript_id"}
    if set(value) != required:
        raise V1PSRRetryError("A3-v2 challenge envelope fields changed")
    if (
        type(value["version"]) is not int
        or value["version"] != 4
        or type(value["message"]) is not bytes
        or type(value["challenge"]) is not bytes
        or type(value["transcript_id"]) is not bytes
    ):
        raise V1PSRRetryError("A3-v2 challenge envelope contains a non-canonical field")
    return cast(A3V2ChallengeEnvelope, value)


def _count_concurrent_terminal_claims(
    operation: Callable[[], object],
    *,
    snapshot: Callable[[], A3V2ProtocolSnapshot],
    expected_terminal_claims: int,
) -> tuple[int, int, A3V2ProtocolSnapshot]:
    """并发竞争同一 pending transcript, 返回 (deny_count, terminal_delta, snapshot)。"""
    before = snapshot()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(operation) for _ in range(2))
        concurrent_results = tuple(future.result() for future in futures)
    deny_count = sum(result == {"version": 4, "status": "deny"} for result in concurrent_results)
    after = snapshot()
    terminal_delta = after.terminal_claims - before.terminal_claims
    if terminal_delta != expected_terminal_claims:
        raise V1PSRRetryError("A3-v2 concurrent terminal claim count changed")
    replay_result = operation()
    if replay_result != {"version": 4, "status": "deny"}:
        raise V1PSRRetryError("A3-v2 replay was not denied")
    return deny_count, terminal_delta, after


def run_v1_a3_v2_retry(
    fixture: V1GeneratedKeyFixture,
    coordinator: A3V2ProtocolCoordinator,
    trusted_input: A3V2TrustedInput,
    *,
    trial_index: int,
    max_attempts: int,
    challenge_for_retry: Callable[[int], V1Challenge],
    forced_abort_prefix: int = 0,
    expiry_indices: Sequence[int] = (),
    expire_hook: Callable[[int], None] | None = None,
) -> V1PSRRetryReport:
    """运行一次本地 toy prover 的 A3-v2 fresh-transcript retry trial。

    retry 每次通过 coordinator 新建 transcript; 旧 transcript 只用于 replay/concurrency 负向验收。
    """
    if type(fixture) is not V1GeneratedKeyFixture:
        raise V1PSRInputError("retry harness requires the exact generated-key fixture")
    if type(coordinator) is not A3V2ProtocolCoordinator:
        raise V1PSRInputError("retry harness requires the exact A3-v2 coordinator")
    if type(trusted_input) is not A3V2TrustedInput:
        raise V1PSRInputError("retry harness requires exact trusted adapter input")
    if not callable(challenge_for_retry):
        raise V1PSRInputError("challenge_for_retry must be a trusted callable")
    trial = _validate_counter(trial_index, "trial_index")
    if type(max_attempts) is not int or max_attempts <= 0:
        raise V1PSRInputError("max_attempts must be a positive exact integer")
    if type(forced_abort_prefix) is not int or not 0 <= forced_abort_prefix <= max_attempts:
        raise V1PSRInputError("forced_abort_prefix is outside the fixed attempt budget")
    if type(expiry_indices) not in (tuple, list):
        raise V1PSRInputError("expiry_indices must be a trusted tuple or list")
    canonical_expiry = tuple(expiry_indices)
    if any(type(index) is not int or not 0 <= index < max_attempts for index in canonical_expiry):
        raise V1PSRInputError("expiry_indices contains a non-canonical attempt index")
    if len(set(canonical_expiry)) != len(canonical_expiry):
        raise V1PSRInputError("expiry_indices contains duplicates")
    if canonical_expiry and expire_hook is None:
        raise V1PSRInputError("expiry_indices require a trusted expire_hook")
    if trusted_input.identity_id != fixture.profile.identity_id:
        raise V1PSRInputError("trusted input identity does not match generated profile")

    before = coordinator.snapshot()
    records: list[V1PSRRetryRecord] = []
    transcript_ids: set[bytes] = set()
    nonce_values: set[bytes] = set()
    for retry_index in range(max_attempts):
        total_start = time.perf_counter_ns()
        sampler_start = time.perf_counter_ns()
        prepared = fixture._prepare_attempt(trial, retry_index)
        sampler_ns = time.perf_counter_ns() - sampler_start
        issued_start = time.perf_counter_ns()
        issued = _require_challenge_envelope(
            coordinator.begin(trusted_input, prepared.commitment.encode())
        )
        a3_begin_ns = time.perf_counter_ns() - issued_start
        message = parse_a3_v2_message(issued["message"])
        transcript_id = issued["transcript_id"]
        challenge = parse_v1_challenge(issued["challenge"])
        expected_challenge = challenge_for_retry(retry_index)
        if type(expected_challenge) is not V1Challenge or challenge != expected_challenge:
            raise V1PSRRetryError("A3-v2 coordinator challenge changed from trusted sampler")
        if transcript_id in transcript_ids or message.nonce in nonce_values:
            raise V1PSRRetryError("A3-v2 retry reused a transcript or nonce")
        transcript_ids.add(transcript_id)
        nonce_values.add(message.nonce)

        if retry_index in canonical_expiry:
            assert expire_hook is not None
            expire_hook(retry_index)
            attempt = fixture._abort_attempt(prepared, challenge)
            raw_terminal = V1Abort(transcript_id).encode()
            terminal_operation = _bind_terminal(coordinator, "abort", raw_terminal)
            expected_successes = 0
            coordinator_outcome = V1PSRCoordinatorOutcome.EXPIRED
        else:
            response_start = time.perf_counter_ns()
            attempt = (
                fixture._abort_attempt(prepared, challenge)
                if retry_index < forced_abort_prefix
                else fixture._finish_attempt(prepared, challenge, transcript_id)
            )
            response_ns = time.perf_counter_ns() - response_start
            if attempt.outcome is V1PSROutcome.EMIT:
                assert attempt.response is not None
                exact_start = time.perf_counter_ns()
                evidence = verify_v1_ref(
                    attempt.commitment.encode(),
                    attempt.challenge.encode(),
                    attempt.response.encode(),
                    transcript_id,
                    fixture.profile,
                )
                exact_ns = time.perf_counter_ns() - exact_start
                if evidence.code is not V1EvidenceCode.RELATION_ACCEPT:
                    raise V1PSRRetryError("honest emitted response was rejected by exact reference")
                raw_terminal = attempt.response.encode()
                terminal_operation = _bind_terminal(coordinator, "respond", raw_terminal)
                expected_successes = 1
                coordinator_outcome = V1PSRCoordinatorOutcome.PROTECTED
            else:
                exact_ns = 0
                raw_terminal = V1Abort(transcript_id).encode()
                terminal_operation = _bind_terminal(coordinator, "abort", raw_terminal)
                expected_successes = 0
                coordinator_outcome = V1PSRCoordinatorOutcome.ABORTED
            terminal_start = time.perf_counter_ns()
            concurrent_denies, terminal_delta, after = _count_concurrent_terminal_claims(
                terminal_operation,
                snapshot=coordinator.snapshot,
                expected_terminal_claims=1,
            )
            a3_terminal_ns = time.perf_counter_ns() - terminal_start
            if expected_successes == 1:
                if after.protected_calls - before.protected_calls != 1:
                    raise V1PSRRetryError("A3-v2 honest emitted response was denied")
            elif after.protected_calls != before.protected_calls:
                raise V1PSRRetryError("A3-v2 abort called the protected operation")
            replay_denies = 1
            records.append(
                V1PSRRetryRecord(
                    retry_index=retry_index,
                    prover_outcome=attempt.outcome,
                    coordinator_outcome=coordinator_outcome,
                    nonce_sha256=hashlib.sha256(message.nonce).digest(),
                    transcript_sha256=hashlib.sha256(transcript_id).digest(),
                    commitment_sha256=hashlib.sha256(prepared.commitment.encode()).digest(),
                    challenge_sha256=hashlib.sha256(challenge.encode()).digest(),
                    sampler_ns=sampler_ns,
                    response_ns=response_ns,
                    exact_ns=exact_ns,
                    a3_ns=a3_begin_ns + a3_terminal_ns,
                    total_ns=time.perf_counter_ns() - total_start,
                    replay_denies=replay_denies,
                    concurrent_denies=concurrent_denies,
                )
            )
            if expected_successes == 1:
                if after.protected_calls - before.protected_calls != 1:
                    raise V1PSRRetryError("A3-v2 protected call count changed")
                return V1PSRRetryReport(
                    trial,
                    max_attempts,
                    "protected",
                    tuple(records),
                    False,
                    after.protected_calls - before.protected_calls,
                    0,
                )
            continue

        terminal_start = time.perf_counter_ns()
        concurrent_denies, terminal_delta, after = _count_concurrent_terminal_claims(
            terminal_operation,
            snapshot=coordinator.snapshot,
            expected_terminal_claims=1,
        )
        del terminal_delta
        a3_terminal_ns = time.perf_counter_ns() - terminal_start
        if after.protected_calls != before.protected_calls:
            raise V1PSRRetryError("A3-v2 expired transcript called the protected operation")
        replay_denies = 1
        records.append(
            V1PSRRetryRecord(
                retry_index=retry_index,
                prover_outcome=attempt.outcome,
                coordinator_outcome=coordinator_outcome,
                nonce_sha256=hashlib.sha256(message.nonce).digest(),
                transcript_sha256=hashlib.sha256(transcript_id).digest(),
                commitment_sha256=hashlib.sha256(prepared.commitment.encode()).digest(),
                challenge_sha256=hashlib.sha256(challenge.encode()).digest(),
                sampler_ns=sampler_ns,
                response_ns=0,
                exact_ns=0,
                a3_ns=a3_begin_ns + a3_terminal_ns,
                total_ns=time.perf_counter_ns() - total_start,
                replay_denies=replay_denies,
                concurrent_denies=concurrent_denies,
            )
        )

    after = coordinator.snapshot()
    protected_calls = after.protected_calls - before.protected_calls
    if protected_calls != 0:
        raise V1PSRRetryError("retry exhaustion called the protected operation")
    return V1PSRRetryReport(
        trial,
        max_attempts,
        "exhausted",
        tuple(records),
        True,
        protected_calls,
        0,
    )


def build_v1_vector_manifest(attempts: Sequence[V1PSRAttempt]) -> bytes:
    """将同一 fixture 的公开 attempt 摘要编码为规范 ASCII JSON。"""
    if type(attempts) not in (tuple, list) or not attempts:
        raise V1PSRManifestError("V1 vector manifest requires a non-empty tuple or list")
    if any(type(attempt) is not V1PSRAttempt for attempt in attempts):
        raise V1PSRManifestError("V1 vector manifest contains a non-canonical attempt")
    first = attempts[0]
    if any(
        attempt.profile_sha256 != first.profile_sha256
        or attempt.seed_sha256 != first.seed_sha256
        or attempt.target_sha256 != first.target_sha256
        for attempt in attempts
    ):
        raise V1PSRManifestError("V1 vector manifest mixes generated-key fixtures")
    records = [attempt.public_record() for attempt in attempts]
    emit_count = sum(attempt.outcome is V1PSROutcome.EMIT for attempt in attempts)
    payload = {
        "abort_count": len(records) - emit_count,
        "emit_count": emit_count,
        "experiment_id": V1_PSR_EXPERIMENT_ID,
        "schema_version": V1_PSR_MANIFEST_SCHEMA_VERSION,
        "vector_count": len(records),
        "vectors": records,
    }
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def write_v1_vector_manifest(path: Path, attempts: Sequence[V1PSRAttempt]) -> Path:
    """以拒绝覆盖策略写入不含 secret 的公开向量清单。"""
    if not isinstance(path, Path):
        raise V1PSRManifestError("V1 vector manifest path must be pathlib.Path")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise V1PSRManifestError("V1 vector manifest parent must be an existing directory")
    payload = build_v1_vector_manifest(attempts)
    try:
        with path.open("xb") as manifest_file:
            manifest_file.write(payload)
    except OSError as error:
        raise V1PSRManifestError("V1 vector manifest cannot be created") from error
    return path


__all__ = [
    "V1_PSR_BLOCK_SIZE",
    "V1_PSR_CHALLENGE_DOMAIN",
    "V1_PSR_CHALLENGE_SET",
    "V1_PSR_EXPERIMENT_ID",
    "V1_PSR_MANIFEST_SCHEMA_VERSION",
    "V1_PSR_MASK_DOMAIN",
    "V1_PSR_SECRET_DOMAIN",
    "V1_PSR_SEED_SIZE",
    "V1_PSR_TARGET_DIGEST_DOMAIN",
    "V1_PSR_THEORETICAL_EMIT_PROBABILITY",
    "V1GeneratedKeyFixture",
    "V1PSRAttempt",
    "V1PSRCoordinatorOutcome",
    "V1PSRInputError",
    "V1PSRLifecycleError",
    "V1PSRManifestError",
    "V1PSROutcome",
    "V1PSRRetryError",
    "V1PSRRetryRecord",
    "V1PSRRetryReport",
    "build_v1_generated_key_fixture",
    "build_v1_vector_manifest",
    "compute_v1_commitment",
    "compute_v1_response",
    "run_v1_a3_v2_retry",
    "sample_v1_challenge",
    "sample_v1_mask",
    "sample_v1_secret",
    "v1_challenge_set",
    "v1_response_emits",
    "write_v1_vector_manifest",
]
