"""V1-P2 exact relation 与 A3-v2 的防御性安全边界测试。"""

from __future__ import annotations

import ast
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import pytest

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_IDENTITY,
    V1_TEST_INPUT_PROFILE,
    V1_TEST_MODEL_ID,
    V1_TEST_RESPONSE,
    V1_TEST_SCOPE_ID,
    V1NonceSource,
    V1ProtectedRecorder,
    build_v1_accepting_commitment,
    build_v1_coordinator,
    build_v1_trusted_input,
)
from can.access import (
    A3V2Clock,
    A3V2Evidence,
    A3V2ExecutionState,
    A3V2ProtocolCoordinator,
    A3V2RouteDecision,
    A3V2StateError,
    A3V2TranscriptStore,
    A3V2TrustedInput,
    V1ReferenceAdapter,
    build_v1_a3_v2_profile,
)
from can.reference import (
    V1Abort,
    V1PublicProfile,
    V1ReferenceEvidence,
    V1Response,
    build_v1_conformance_profile,
)


def _issued_response() -> tuple[A3V2ProtocolCoordinator, V1ProtectedRecorder, bytes]:
    coordinator, profile, recorder, _ = build_v1_coordinator()
    issued = coordinator.begin(
        build_v1_trusted_input(),
        build_v1_accepting_commitment(profile).encode(),
    )
    assert issued["status"] == "challenge"
    return (
        coordinator,
        recorder,
        V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode(),
    )


def _replace_route_field(
    trusted: A3V2TrustedInput,
    route_field: str,
    value: object,
) -> A3V2TrustedInput:
    if route_field == "model_id":
        assert type(value) is int
        return replace(trusted, model_id=value)
    if route_field == "identity_id":
        assert type(value) is bytes
        return replace(trusted, identity_id=value)
    if route_field == "scope_id":
        assert type(value) is int
        return replace(trusted, scope_id=value)
    assert route_field == "input_profile_sha256" and type(value) is bytes
    return replace(trusted, input_profile_sha256=value)


def test_public_runtime_objects_have_no_secret_or_authority_fields() -> None:
    """V1 profile、adapter 与 evidence 只能携带公开配置和摘要。"""
    observed = {
        item.name
        for object_type in (
            V1PublicProfile,
            V1ReferenceAdapter,
            V1ReferenceEvidence,
            A3V2Evidence,
        )
        for item in fields(object_type)
    }
    forbidden = {
        "secret",
        "secret_key",
        "private_key",
        "mask",
        "decision",
        "authorization",
        "capability",
        "model",
    }

    assert not observed & forbidden


@pytest.mark.parametrize(
    "route_field,value",
    [
        ("model_id", V1_TEST_MODEL_ID + 1),
        ("identity_id", bytes([255]) * 32),
        ("scope_id", V1_TEST_SCOPE_ID + 1),
        ("input_profile_sha256", hashlib.sha256(b"other profile").digest()),
    ],
)
def test_trusted_route_binding_mismatch_creates_no_transcript(
    route_field: str,
    value: object,
) -> None:
    """identity、model、scope 或业务 profile 不匹配时必须在建态前拒绝。"""
    coordinator, profile, recorder, _ = build_v1_coordinator()
    trusted = _replace_route_field(build_v1_trusted_input(), route_field, value)

    result = coordinator.begin(trusted, build_v1_accepting_commitment(profile).encode())

    assert result == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.challenge_issues == 0
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


@pytest.mark.parametrize("raw_response", [None, True, {}, bytearray(181), b""])
def test_response_type_confusion_has_zero_verifier_and_protected_calls(
    raw_response: object,
) -> None:
    """非规范 response 类型或长度不能 claim transcript 或调用 verifier。"""
    coordinator, profile, recorder, _ = build_v1_coordinator()
    issued = coordinator.begin(
        build_v1_trusted_input(),
        build_v1_accepting_commitment(profile).encode(),
    )
    assert issued["status"] == "challenge"

    result = coordinator.respond(raw_response)

    assert result == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 0
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


@pytest.mark.parametrize(
    "foreign_wire",
    [
        bytes(23),
        b"CAN-A3-MSG-v1\x00" + bytes(119),
        bytes(105),
        b"CAN-V1-SIS-COM-v1\x00" + bytes(64),
    ],
)
def test_foreign_route_bytes_have_no_v1_fallback(foreign_wire: bytes) -> None:
    """V0、A3-v1、A4 与 V1-P1 bytes 不能进入 V1-P2 路线。"""
    coordinator, _, recorder, _ = build_v1_coordinator()

    begin = coordinator.begin(build_v1_trusted_input(), foreign_wire)
    response = coordinator.respond(foreign_wire)
    abort = coordinator.abort(foreign_wire)

    assert begin == {"version": 4, "status": "deny"}
    assert response == {"version": 4, "status": "deny"}
    assert abort == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.challenge_issues == 0
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


def test_concurrent_duplicate_response_commits_and_calls_once() -> None:
    """并发重复 response 只能有一个 allow commit 和一次受保护调用。"""
    coordinator, recorder, response = _issued_response()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _index: coordinator.respond(response), range(32)))

    assert sum(result["status"] == "protected" for result in results) == 1
    assert recorder.snapshots == [b"canonical snapshot"]
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1


def test_concurrent_internal_commit_and_value_delivery_are_both_single_use() -> None:
    """并发内部提交与 value 消费都只能成功一次。"""
    operation_value = object()
    recorder = V1ProtectedRecorder(result=operation_value)
    coordinator, profile, _, _ = build_v1_coordinator(recorder=recorder)
    issued = coordinator.begin(
        build_v1_trusted_input(), build_v1_accepting_commitment(profile).encode()
    )
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(
            executor.map(lambda _index: coordinator.commit_and_execute(response), range(32))
        )

    successful = [
        result
        for result in results
        if result.route_decision is A3V2RouteDecision.PROTECTED
        and result.execution_state is A3V2ExecutionState.SUCCEEDED
    ]
    assert len(successful) == 1
    assert recorder.snapshots == [b"canonical snapshot"]

    def consume_once(_index: int) -> object | None:
        try:
            return successful[0].consume_operation_value()
        except A3V2StateError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        delivered = tuple(executor.map(consume_once, range(32)))

    assert sum(value is operation_value for value in delivered) == 1
    assert sum(value is None for value in delivered) == 31
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1


def test_abort_internal_clock_error_returns_fixed_deny() -> None:
    """abort 的可信时钟异常必须 fail closed, 不泄露异常或调用受保护操作。"""

    class FailingClock:
        wall_ms = 1_700_000_000_000
        mono_ns = 5_000_000_000
        fail = False

        def monotonic_ns(self) -> int:
            if self.fail:
                raise RuntimeError("synthetic trusted clock failure")
            return self.mono_ns

    clock = FailingClock()
    recorder = V1ProtectedRecorder()
    profile = build_v1_conformance_profile(V1_TEST_IDENTITY)
    route = build_v1_a3_v2_profile(
        profile,
        model_id=V1_TEST_MODEL_ID,
        scope_id=V1_TEST_SCOPE_ID,
        input_profile_sha256=V1_TEST_INPUT_PROFILE,
        protected_operation=recorder,
    )
    coordinator = A3V2ProtocolCoordinator(
        (route,),
        store=A3V2TranscriptStore(
            clock=A3V2Clock(lambda: clock.wall_ms, clock.monotonic_ns),
            random_bytes=V1NonceSource(),
        ),
        challenge_sampler=lambda degree, weight: V1_TEST_CHALLENGE.coefficients,
    )
    issued = coordinator.begin(
        build_v1_trusted_input(),
        build_v1_accepting_commitment(profile).encode(),
    )
    assert issued["status"] == "challenge"
    clock.fail = True

    result = coordinator.abort(V1Abort(issued["transcript_id"]).encode())

    assert result == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


def test_reference_module_has_no_prover_model_or_fallback_dependency() -> None:
    """exact reference 不得导入模型、访问层、prover 或旧路线 fallback。"""
    source_path = Path("src/can/reference/v1.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    public_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    assert not any(
        name.startswith(("can.access", "can.model", "can.verifier")) for name in imported_modules
    )
    assert not {"keygen", "prover", "sign", "sample_mask"} & public_names
    assert not any(
        name.startswith(("can.reference.a0", "can.reference.a4")) for name in imported_modules
    )
