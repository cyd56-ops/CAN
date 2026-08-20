"""A3-v2 commit-first 协议壳单元测试。"""

from __future__ import annotations

import hashlib

import pytest

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_INPUT_PROFILE,
    V1_TEST_RESPONSE,
    V1ProtectedRecorder,
    build_v1_accepting_commitment,
    build_v1_coordinator,
    build_v1_trusted_input,
)
from can.access import (
    A3_V2_CHALLENGE_TTL_MS,
    A3_V2_MESSAGE_SIZE,
    A3V2ChallengeEnvelope,
    A3V2ExecutionState,
    A3V2InternalResultCode,
    A3V2Message,
    A3V2ProtocolCoordinator,
    A3V2ProtocolInputError,
    A3V2RouteDecision,
    A3V2StateError,
    A3V2TrustedInput,
    compute_a3_v2_binding_digest,
    compute_a3_v2_transcript_id,
    parse_a3_v2_message,
)
from can.reference import V1Abort, V1PublicProfile, V1Response


def _begin() -> tuple[
    A3V2ProtocolCoordinator,
    A3V2ChallengeEnvelope,
    V1ProtectedRecorder,
    V1PublicProfile,
]:
    coordinator, profile, recorder, _ = build_v1_coordinator()
    commitment = build_v1_accepting_commitment(profile)
    issued = coordinator.begin(build_v1_trusted_input(), commitment.encode())
    assert issued["status"] == "challenge"
    return coordinator, issued, recorder, profile


def test_message_round_trip_uses_new_domain_and_fixed_size() -> None:
    """A3-v2 message 必须使用独立 version/domain 且保持固定长度。"""
    message = A3V2Message(
        2,
        2,
        bytes(range(32)),
        1,
        1_700_000_000_000,
        1_700_000_060_000,
        bytes(reversed(range(32))),
        bytes([9]) * 32,
    )

    encoded = message.encode()

    assert len(encoded) == A3_V2_MESSAGE_SIZE
    assert parse_a3_v2_message(encoded) == message
    assert not encoded.startswith(b"CAN-A3-MSG-v1\x00")


def test_begin_binds_trusted_digest_and_issues_commit_first_challenge() -> None:
    """只有 exact trusted input 和 canonical commitment 才签发 server challenge。"""
    coordinator, profile, _, _ = build_v1_coordinator()
    trusted_input = build_v1_trusted_input(b"same immutable snapshot")
    commitment = build_v1_accepting_commitment(profile).encode()

    issued = coordinator.begin(trusted_input, commitment)

    assert issued["status"] == "challenge"
    challenge = issued
    message = parse_a3_v2_message(challenge["message"])
    assert message.input_digest == trusted_input.input_digest
    assert challenge["challenge"] == V1_TEST_CHALLENGE.encode()
    binding = compute_a3_v2_binding_digest(
        profile.public_key_sha256,
        V1_TEST_INPUT_PROFILE,
        challenge["message"],
        commitment,
    )
    assert challenge["transcript_id"] == compute_a3_v2_transcript_id(
        binding, challenge["challenge"]
    )


def test_valid_response_claims_once_and_calls_stored_snapshot_once() -> None:
    """exact accept 只能提交一次并调用一次受保护 callback。"""
    coordinator, issued, recorder, _ = _begin()
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


def test_internal_result_delivers_operation_value_exactly_once() -> None:
    """可信 adapter 只能从成功的内部结果取得一次 operation value。"""
    operation_value = object()
    recorder = V1ProtectedRecorder(result=operation_value)
    coordinator, profile, _, _ = build_v1_coordinator(recorder=recorder)
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    result = coordinator.commit_and_execute(response)

    assert result.route_decision is A3V2RouteDecision.PROTECTED
    assert result.execution_state is A3V2ExecutionState.SUCCEEDED
    assert result.code is A3V2InternalResultCode.PROTECTED_SUCCEEDED
    assert result.consume_operation_value() is operation_value
    with pytest.raises(A3V2StateError, match="already consumed"):
        result.consume_operation_value()
    snapshot = coordinator.snapshot()
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1
    assert snapshot.protected_responses == 0


def test_c1_response_discards_internal_value_and_remains_status_only() -> None:
    """C1 adapter 必须丢弃 callback value 且保持原 version-4 schema。"""
    recorder = V1ProtectedRecorder(result={"logits": [1.0, 2.0]})
    coordinator, profile, _, _ = build_v1_coordinator(recorder=recorder)
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    result = coordinator.respond(response)

    assert result == {"version": 4, "status": "protected"}
    assert tuple(result) == ("version", "status")


def test_relation_reject_is_terminal_and_valid_retry_cannot_run() -> None:
    """同一 transcript 的首个 parsed response 即使无效也必须终结状态。"""
    coordinator, issued, recorder, _ = _begin()
    invalid_values = [list(polynomial) for polynomial in V1_TEST_RESPONSE]
    invalid_values[0][0] += 1
    invalid = V1Response(issued["transcript_id"], invalid_values).encode()
    valid = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    rejected = coordinator.respond(invalid)
    retry = coordinator.respond(valid)

    assert rejected == {"version": 4, "status": "deny"}
    assert retry == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.protected_calls == 0


def test_internal_relation_reject_is_pre_execution_deny() -> None:
    """relation reject 必须表示为尚未开始业务执行的内部 DENY。"""
    coordinator, issued, recorder, _ = _begin()
    invalid_values = [list(polynomial) for polynomial in V1_TEST_RESPONSE]
    invalid_values[0][0] += 1
    invalid = V1Response(issued["transcript_id"], invalid_values).encode()

    result = coordinator.commit_and_execute(invalid)

    assert result.route_decision is A3V2RouteDecision.DENY
    assert result.execution_state is A3V2ExecutionState.NOT_STARTED
    assert result.code is A3V2InternalResultCode.VERIFICATION_REJECTED
    with pytest.raises(A3V2StateError, match="no successful operation value"):
        result.consume_operation_value()
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 0
    assert snapshot.protected_calls == 0


def test_abort_is_terminal_without_verifier_or_protected_call() -> None:
    """规范 abort 必须终结 transcript 且保持零 verifier/model calls。"""
    coordinator, issued, recorder, _ = _begin()
    abort = V1Abort(issued["transcript_id"]).encode()

    aborted = coordinator.abort(abort)
    after_abort = coordinator.respond(
        V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()
    )

    assert aborted == {"version": 4, "status": "deny"}
    assert after_abort == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.aborts == 1
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


def test_expiry_atomically_terminates_pending_transcript() -> None:
    """deadline 边界上的 response 必须 expiry deny 且不能重试。"""
    coordinator, profile, recorder, clock = build_v1_coordinator()
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    challenge = issued
    clock.mono_ns += A3_V2_CHALLENGE_TTL_MS * 1_000_000
    response = V1Response(challenge["transcript_id"], V1_TEST_RESPONSE).encode()

    expired = coordinator.respond(response)
    replayed = coordinator.respond(response)

    assert expired == {"version": 4, "status": "deny"}
    assert replayed == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.expiries == 1
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


def test_same_commitment_cannot_create_a_second_transcript() -> None:
    """commitment 在单个 state epoch 中只能签发一个 challenge。"""
    coordinator, profile, _, _ = build_v1_coordinator()
    commitment = build_v1_accepting_commitment(profile).encode()

    first = coordinator.begin(build_v1_trusted_input(b"first"), commitment)
    second = coordinator.begin(build_v1_trusted_input(b"second"), commitment)

    assert first["status"] == "challenge"
    assert second == {"version": 4, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.challenge_issues == 1
    assert snapshot.challenge_denies == 1


def test_wrong_input_profile_digest_is_rejected_before_state_creation() -> None:
    """跨业务 profile digest 不能进入同一 A3-v2 route。"""
    coordinator, profile, _, _ = build_v1_coordinator()
    trusted = build_v1_trusted_input()
    wrong = A3V2TrustedInput(
        trusted.model_id,
        trusted.identity_id,
        trusted.scope_id,
        hashlib.sha256(b"other input profile").digest(),
        trusted.input_digest,
        trusted.snapshot,
    )

    result = coordinator.begin(wrong, build_v1_accepting_commitment(profile).encode())

    assert result == {"version": 4, "status": "deny"}
    assert coordinator.snapshot().challenge_issues == 0


@pytest.mark.parametrize("trusted", [None, {}, True, b"digest"])
def test_untrusted_objects_cannot_replace_adapter_output(trusted: object) -> None:
    """外部 dict、bool 或 bytes 不能伪装为本地 trusted adapter 结果。"""
    coordinator, profile, _, _ = build_v1_coordinator()

    result = coordinator.begin(trusted, build_v1_accepting_commitment(profile).encode())

    assert result == {"version": 4, "status": "deny"}


def test_post_commit_operation_failure_is_one_call_without_retry() -> None:
    """allow commit 后 callback 异常固定 deny, 但不得回滚或二次调用。"""
    recorder = V1ProtectedRecorder(fail=True)
    coordinator, profile, _, _ = build_v1_coordinator(recorder=recorder)
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    challenge = issued
    response = V1Response(challenge["transcript_id"], V1_TEST_RESPONSE).encode()

    failed = coordinator.respond(response)
    replayed = coordinator.respond(response)

    assert failed == {"version": 4, "status": "deny"}
    assert replayed == {"version": 4, "status": "deny"}
    assert len(recorder.snapshots) == 1
    snapshot = coordinator.snapshot()
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1
    assert snapshot.protected_responses == 0


def test_internal_result_distinguishes_post_commit_execution_failure() -> None:
    """受保护 callback 异常必须保留 PROTECTED/FAILED 内部状态。"""
    recorder = V1ProtectedRecorder(fail=True)
    coordinator, profile, _, _ = build_v1_coordinator(recorder=recorder)
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    result = coordinator.commit_and_execute(response)

    assert result.route_decision is A3V2RouteDecision.PROTECTED
    assert result.execution_state is A3V2ExecutionState.FAILED
    assert result.code is A3V2InternalResultCode.PROTECTED_EXECUTION_ERROR
    with pytest.raises(A3V2StateError, match="no successful operation value"):
        result.consume_operation_value()
    snapshot = coordinator.snapshot()
    assert snapshot.allow_commits == 1
    assert snapshot.deny_commits == 0
    assert snapshot.protected_calls == 1
    assert snapshot.protected_responses == 0


def test_default_coordinator_is_closed_without_a_route() -> None:
    """没有本地 V1 route 时不能创建 transcript。"""
    profile_coordinator, profile, _, _ = build_v1_coordinator()
    del profile_coordinator
    coordinator = A3V2ProtocolCoordinator()

    result = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )

    assert result == {"version": 4, "status": "deny"}


def test_a3_v1_message_is_rejected_by_v2_parser() -> None:
    """A3-v1 message 不能进入 A3-v2 message parser。"""
    with pytest.raises(A3V2ProtocolInputError):
        parse_a3_v2_message(b"CAN-A3-MSG-v1\x00" + bytes(A3_V2_MESSAGE_SIZE - 14))
