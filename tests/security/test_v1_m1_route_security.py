"""V1-M1 输入 adapter 与 A3-v2 route isolation 安全测试。"""

import torch

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_RESPONSE,
    V1NonceSource,
    V1ProtectedRecorder,
    V1TestClock,
    build_v1_accepting_commitment,
)
from can.access import (
    V1_M1_INPUT_PROFILE_SHA256,
    V1_M1_MODEL_ID,
    V1_M1_SCOPE_ID,
    A3V2Clock,
    A3V2ProtocolCoordinator,
    A3V2TranscriptStore,
    A3V2TrustedInput,
    V1M1AccessCoordinator,
    V1M1InputAdapter,
    build_v1_a3_v2_profile,
)
from can.reference import V1PublicProfile, V1Response, build_v1_conformance_profile

IDENTITY = bytes(range(32))


def _coordinator() -> tuple[V1M1AccessCoordinator, V1PublicProfile, V1ProtectedRecorder]:
    profile = build_v1_conformance_profile(IDENTITY)
    recorder = V1ProtectedRecorder()
    route = build_v1_a3_v2_profile(
        profile,
        model_id=V1_M1_MODEL_ID,
        scope_id=V1_M1_SCOPE_ID,
        input_profile_sha256=V1_M1_INPUT_PROFILE_SHA256,
        protected_operation=recorder,
    )
    clock = V1TestClock()
    return (
        V1M1AccessCoordinator(
            V1M1InputAdapter(IDENTITY),
            A3V2ProtocolCoordinator(
                (route,),
                store=A3V2TranscriptStore(
                    clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
                    random_bytes=V1NonceSource(),
                ),
                challenge_sampler=lambda degree, weight: V1_TEST_CHALLENGE.coefficients,
            ),
        ),
        profile,
        recorder,
    )


def test_valid_v1_m1_route_binds_snapshot_and_calls_only_after_evidence_accepts() -> None:
    """valid V1 response 才能将 adapter snapshot 交给受保护 callback 一次。"""
    coordinator, profile, recorder = _coordinator()
    image = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
    issued = coordinator.begin(image, build_v1_accepting_commitment(profile).encode())
    assert issued["status"] == "challenge"

    result = coordinator.respond(V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode())

    assert result == {"version": 4, "status": "protected"}
    assert len(recorder.snapshots) == 1


def test_a2_and_foreign_v1_inputs_cannot_enter_v1_m1_route() -> None:
    """Fashion-MNIST shape 与跨 profile trusted input 均必须零 protected calls。"""
    coordinator, profile, recorder = _coordinator()
    fashion_mnist = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    valid = V1M1InputAdapter(IDENTITY).adapt(torch.zeros((1, 3, 32, 32), dtype=torch.uint8))
    foreign = A3V2TrustedInput(
        model_id=2,
        identity_id=valid.identity_id,
        scope_id=valid.scope_id,
        input_profile_sha256=valid.input_profile_sha256,
        input_digest=valid.input_digest,
        snapshot=valid.snapshot,
    )

    commitment = build_v1_accepting_commitment(profile).encode()
    fashion_result = coordinator.begin(fashion_mnist, commitment)
    result = coordinator.begin(foreign, commitment)

    assert fashion_result == {"version": 4, "status": "deny"}
    assert result == {"version": 4, "status": "deny"}
    assert recorder.snapshots == []
    assert coordinator.snapshot().protected_calls == 0
