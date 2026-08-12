"""V1-C1 neural adapter 与 A3-v2 coordinator 的组合验收。"""

from __future__ import annotations

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_INPUT_PROFILE,
    V1_TEST_MODEL_ID,
    V1_TEST_RESPONSE,
    V1_TEST_SCOPE_ID,
    V1NonceSource,
    V1ProtectedRecorder,
    V1TestClock,
    build_v1_accepting_commitment,
    build_v1_trusted_input,
)
from can.access import (
    A3V2Clock,
    A3V2ProtocolCoordinator,
    A3V2TranscriptStore,
    build_v1_a3_v2_neural_profile,
)
from can.reference import V1PublicProfile, V1Response, build_v1_conformance_profile
from can.verifier import compile_v1_neural_profile


def _build_neural_coordinator() -> tuple[
    A3V2ProtocolCoordinator, V1PublicProfile, V1ProtectedRecorder
]:
    profile = build_v1_conformance_profile(bytes(range(32)))
    neural_profile = compile_v1_neural_profile(profile)
    recorder = V1ProtectedRecorder()
    clock = V1TestClock()
    route = build_v1_a3_v2_neural_profile(
        neural_profile,
        model_id=V1_TEST_MODEL_ID,
        scope_id=V1_TEST_SCOPE_ID,
        input_profile_sha256=V1_TEST_INPUT_PROFILE,
        protected_operation=recorder,
    )
    store = A3V2TranscriptStore(
        clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
        random_bytes=V1NonceSource(),
    )
    coordinator = A3V2ProtocolCoordinator(
        (route,),
        store=store,
        challenge_sampler=lambda degree, weight: V1_TEST_CHALLENGE.coefficients,
    )
    return coordinator, profile, recorder


def test_v1_neural_route_commits_protected_operation_once() -> None:
    """neural accept 只能经 A3-v2 单一协调器产生一次 protected call。"""
    coordinator, profile, recorder = _build_neural_coordinator()
    trusted = build_v1_trusted_input()
    issued = coordinator.begin(trusted, build_v1_accepting_commitment(profile).encode())
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()
    assert coordinator.respond(response) == {"version": 4, "status": "protected"}
    assert len(recorder.snapshots) == 1
    assert coordinator.respond(response) == {"version": 4, "status": "deny"}
    assert len(recorder.snapshots) == 1
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1


def test_v1_neural_route_reject_is_terminal_with_zero_protected_calls() -> None:
    """neural equation reject 必须终结 transcript 且不产生 protected call。"""
    coordinator, profile, recorder = _build_neural_coordinator()
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    changed = [list(polynomial) for polynomial in V1_TEST_RESPONSE]
    changed[0][0] += 1
    invalid = V1Response(issued["transcript_id"], changed).encode()
    valid = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    assert coordinator.respond(invalid) == {"version": 4, "status": "deny"}
    assert coordinator.respond(valid) == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.deny_commits == 2
    assert snapshot.protected_calls == 0


def test_foreign_route_wire_has_zero_neural_or_protected_calls() -> None:
    """V0 wire 不能进入 V1 neural route 或触发受保护操作。"""
    coordinator, _, recorder = _build_neural_coordinator()
    foreign_wire = bytes(23)

    assert coordinator.begin(build_v1_trusted_input(), foreign_wire) == {
        "version": 4,
        "status": "deny",
    }
    assert coordinator.respond(foreign_wire) == {"version": 4, "status": "deny"}
    assert coordinator.abort(foreign_wire) == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.challenge_issues == 0
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0
