"""V1-M1 AuthenticatedR2 reject、route isolation 与单次提交安全测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_RESPONSE,
    V1NonceSource,
    V1TestClock,
    build_v1_accepting_commitment,
)
from can.access import A3V2Clock, A3V2TranscriptStore, AuthenticatedR2
from can.model import V1Cifar100ResNet18
from can.reference import V1PublicProfile, V1Response, build_v1_conformance_profile
from can.verifier import compile_v1_neural_profile

IDENTITY = bytes(range(32))


def _authenticated_r2(
    model: V1Cifar100ResNet18,
) -> tuple[AuthenticatedR2, V1PublicProfile]:
    public_profile = build_v1_conformance_profile(IDENTITY)
    neural_profile = compile_v1_neural_profile(public_profile)
    clock = V1TestClock()
    authenticated = AuthenticatedR2(
        neural_profile,
        model,
        store=A3V2TranscriptStore(
            clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
            random_bytes=V1NonceSource(),
        ),
        challenge_sampler=lambda degree, weight: V1_TEST_CHALLENGE.coefficients,
    )
    return authenticated, public_profile


def test_relation_reject_and_replay_have_zero_r2_forward_calls() -> None:
    """tamper 必须终结 transcript, 后续合法 replay 也不能进入 R2。"""
    model = V1Cifar100ResNet18().eval()
    model_calls: list[None] = []

    def record_call(*_args: object) -> None:
        model_calls.append(None)

    handle = model.register_forward_hook(record_call)
    try:
        authenticated, public_profile = _authenticated_r2(model)
        image = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
        issued = authenticated.begin(
            image,
            build_v1_accepting_commitment(public_profile).encode(),
        )
        assert issued["status"] == "challenge"
        changed = [list(polynomial) for polynomial in V1_TEST_RESPONSE]
        changed[0][0] += 1
        invalid = V1Response(issued["transcript_id"], changed).encode()
        valid = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

        assert authenticated.respond(invalid) == {"version": 4, "status": "deny"}
        assert authenticated.respond(valid) == {"version": 4, "status": "deny"}
    finally:
        handle.remove()

    assert model_calls == []
    snapshot = authenticated.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 0
    assert snapshot.protected_calls == 0


def test_noncanonical_business_input_has_zero_gate_and_r2_calls() -> None:
    """跨模型 shape 必须在业务 adapter 拒绝, 不能进入 neural Gate Layer 或 R2。"""
    model = V1Cifar100ResNet18().eval()
    authenticated, public_profile = _authenticated_r2(model)
    foreign_image = torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    result = authenticated.begin(
        foreign_image,
        build_v1_accepting_commitment(public_profile).encode(),
    )

    assert result == {"version": 4, "status": "deny"}
    snapshot = authenticated.snapshot()
    assert snapshot.challenge_issues == 0
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


def test_concurrent_duplicate_response_commits_only_one_r2_call() -> None:
    """并发重复 response 只能有一个 allow commit 和一次 R2 forward。"""
    model = V1Cifar100ResNet18().eval()
    model_calls: list[None] = []

    def record_call(*_args: object) -> None:
        model_calls.append(None)

    handle = model.register_forward_hook(record_call)
    try:
        authenticated, public_profile = _authenticated_r2(model)
        image = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
        issued = authenticated.begin(
            image,
            build_v1_accepting_commitment(public_profile).encode(),
        )
        assert issued["status"] == "challenge"
        response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(authenticated.respond, (response, response)))
    finally:
        handle.remove()

    assert sorted(result["status"] for result in results) == ["deny", "protected"]
    assert len(model_calls) == 1
    snapshot = authenticated.snapshot()
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1
