"""V1-P2 exact reference 与 A3-v2 单次提交边界的组合测试。"""

from __future__ import annotations

import hashlib

from _v1_support import (
    V1_TEST_RESPONSE,
    build_v1_accepting_commitment,
    build_v1_coordinator,
    build_v1_trusted_input,
)
from can.access import (
    A3V2Clock,
    A3V2ProtocolCoordinator,
    A3V2TranscriptStore,
    A3V2TrustedInput,
    build_v1_a3_v2_profile,
)
from can.experiments.v1_psr import (
    V1GeneratedKeyFixture,
    V1PSROutcome,
    build_v1_generated_key_fixture,
    run_v1_a3_v2_retry,
    sample_v1_challenge,
)
from can.reference import V1Response

RETRY_SEED = (7495).to_bytes(32, byteorder="big", signed=False)
RETRY_IDENTITY = hashlib.sha256(b"CAN V1 PSR retry integration").digest()
RETRY_INPUT_PROFILE = hashlib.sha256(b"CAN V1 PSR retry input profile").digest()


class _RetryClock:
    def __init__(self) -> None:
        self.wall_ms = 1_700_000_000_000
        self.mono_ns = 5_000_000_000


def _build_retry_coordinator(
    *,
    fixture: V1GeneratedKeyFixture,
    clock: _RetryClock,
    protected_calls: list[object],
) -> tuple[A3V2ProtocolCoordinator, A3V2TrustedInput, list[int]]:
    """构造使用 trusted challenge sequence 的临时 A3-v2 retry route。"""
    challenge_indexes: list[int] = []

    def challenge_sampler(degree: int, weight: int) -> tuple[int, ...]:
        index = len(challenge_indexes)
        challenge_indexes.append(index)
        return sample_v1_challenge(RETRY_SEED, 0, index).coefficients

    input_digest = hashlib.sha256(b"CAN retry trusted input").digest()
    trusted = A3V2TrustedInput(
        model_id=2,
        identity_id=RETRY_IDENTITY,
        scope_id=1,
        input_profile_sha256=RETRY_INPUT_PROFILE,
        input_digest=input_digest,
        snapshot=b"retry snapshot",
    )
    route = build_v1_a3_v2_profile(
        fixture.profile,
        model_id=2,
        scope_id=1,
        input_profile_sha256=RETRY_INPUT_PROFILE,
        protected_operation=lambda snapshot: protected_calls.append(snapshot),
    )
    nonce_counter = [0]

    def nonce_source(size: int) -> bytes:
        nonce_counter[0] += 1
        return nonce_counter[0].to_bytes(size, byteorder="big", signed=False)

    store = A3V2TranscriptStore(
        clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
        random_bytes=nonce_source,
    )
    return (
        A3V2ProtocolCoordinator(
            (route,),
            store=store,
            challenge_sampler=challenge_sampler,
        ),
        trusted,
        challenge_indexes,
    )


def test_psr_retry_uses_fresh_transcripts_and_protects_once() -> None:
    """forced abort 后必须新建 transcript, 首次 emit 只调用一次 protected operation。"""
    protected_calls: list[object] = []
    clock = _RetryClock()
    with build_v1_generated_key_fixture(RETRY_IDENTITY, RETRY_SEED) as fixture:
        coordinator, trusted, challenge_indexes = _build_retry_coordinator(
            fixture=fixture,
            clock=clock,
            protected_calls=protected_calls,
        )
        report = run_v1_a3_v2_retry(
            fixture,
            coordinator,
            trusted,
            trial_index=0,
            max_attempts=4,
            forced_abort_prefix=2,
            challenge_for_retry=lambda retry: sample_v1_challenge(RETRY_SEED, 0, retry),
        )

    assert report.outcome == "protected"
    assert report.attempt_count == 3
    assert [attempt.prover_outcome for attempt in report.attempts] == [
        V1PSROutcome.ABORT,
        V1PSROutcome.ABORT,
        V1PSROutcome.EMIT,
    ]
    assert len({attempt.nonce_sha256 for attempt in report.attempts}) == 3
    assert len({attempt.transcript_sha256 for attempt in report.attempts}) == 3
    assert challenge_indexes == [0, 1, 2]
    assert report.protected_calls == 1
    assert len(protected_calls) == 1
    assert report.exact_false_rejects == 0
    assert all(attempt.replay_denies == 1 for attempt in report.attempts)
    assert [attempt.concurrent_denies for attempt in report.attempts] == [2, 2, 1]


def test_psr_retry_exhaustion_has_zero_verifier_and_protected_calls() -> None:
    """预算耗尽必须返回 exhausted, 且 coordinator 不调用 verifier 或 protected callback。"""
    protected_calls: list[object] = []
    clock = _RetryClock()
    with build_v1_generated_key_fixture(RETRY_IDENTITY, RETRY_SEED) as fixture:
        coordinator, trusted, _ = _build_retry_coordinator(
            fixture=fixture,
            clock=clock,
            protected_calls=protected_calls,
        )
        report = run_v1_a3_v2_retry(
            fixture,
            coordinator,
            trusted,
            trial_index=0,
            max_attempts=2,
            forced_abort_prefix=2,
            challenge_for_retry=lambda retry: sample_v1_challenge(RETRY_SEED, 0, retry),
        )

    snapshot = coordinator.snapshot()
    assert report.outcome == "exhausted"
    assert report.retry_exhausted is True
    assert report.attempt_count == 2
    assert report.protected_calls == 0
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0
    assert protected_calls == []


def test_psr_retry_expiry_is_terminal_and_next_retry_is_fresh() -> None:
    """expiry 只终结当前 transcript, 后续 retry 仍使用新 nonce/transcript。"""
    protected_calls: list[object] = []
    clock = _RetryClock()
    with build_v1_generated_key_fixture(RETRY_IDENTITY, RETRY_SEED) as fixture:
        coordinator, trusted, _ = _build_retry_coordinator(
            fixture=fixture,
            clock=clock,
            protected_calls=protected_calls,
        )

        def expire(_retry_index: int) -> None:
            clock.mono_ns += 60_000 * 1_000_000

        report = run_v1_a3_v2_retry(
            fixture,
            coordinator,
            trusted,
            trial_index=0,
            max_attempts=4,
            expiry_indices=(0,),
            expire_hook=expire,
            challenge_for_retry=lambda retry: sample_v1_challenge(RETRY_SEED, 0, retry),
        )

    assert report.outcome == "protected"
    assert report.attempts[0].coordinator_outcome.value == "expired"
    assert report.attempt_count == 3
    assert report.protected_calls == 1
    assert len(protected_calls) == 1
    snapshot = coordinator.snapshot()
    assert snapshot.expiries == 1
    assert snapshot.verifier_calls == 1


def test_exact_relation_commits_one_a3_v2_transcript_once() -> None:
    """公开 conformance relation accept 经协调器后最多执行一次受保护操作。"""
    coordinator, profile, recorder, _ = build_v1_coordinator()
    issued = coordinator.begin(
        build_v1_trusted_input(),
        build_v1_accepting_commitment(profile).encode(),
    )
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    accepted = coordinator.respond(response)
    replayed = coordinator.respond(response)

    assert accepted == {"version": 4, "status": "protected"}
    assert replayed == {"version": 4, "status": "deny"}
    assert recorder.snapshots == [b"canonical snapshot"]
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1


def test_exact_relation_reject_is_terminal_before_protected_operation() -> None:
    """方程篡改必须终结 transcript, 且不能通过有效重试触发受保护操作。"""
    coordinator, profile, recorder, _ = build_v1_coordinator()
    issued = coordinator.begin(
        build_v1_trusted_input(),
        build_v1_accepting_commitment(profile).encode(),
    )
    assert issued["status"] == "challenge"
    changed = [list(polynomial) for polynomial in V1_TEST_RESPONSE]
    changed[0][0] += 1
    invalid = V1Response(issued["transcript_id"], changed).encode()
    valid = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    rejected = coordinator.respond(invalid)
    retried = coordinator.respond(valid)

    assert rejected == {"version": 4, "status": "deny"}
    assert retried == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.protected_calls == 0
