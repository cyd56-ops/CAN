"""A4 reference/neural adapter 与 A3 freshness 协调器的组合测试。"""

from __future__ import annotations

import torch

from can.access import (
    A3Clock,
    A3NonceStore,
    A3ProtocolCoordinator,
    build_a4_neural_verification_profile,
    build_a4_verification_profile,
)
from can.model.a2_mlp import A2FashionMNISTMLP
from can.reference import A4PublicProfile
from can.verifier import compile_a4_neural_profile
from conftest import A4ProofFactory


class _Clock:
    wall_ms = 1_700_000_000_000
    mono_ns = 5_000_000_000


def _image() -> torch.Tensor:
    return torch.full((1, 1, 28, 28), 0.25, dtype=torch.float32)


def _coordinator(profile: A4PublicProfile) -> A3ProtocolCoordinator:
    clock = _Clock()
    store = A3NonceStore(
        clock=A3Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
        random_bytes=lambda size: b"\x01" * size,
    )
    return A3ProtocolCoordinator(
        A2FashionMNISTMLP().eval(),
        (build_a4_verification_profile(profile),),
        store=store,
    )


def _neural_coordinator(profile: A4PublicProfile) -> A3ProtocolCoordinator:
    clock = _Clock()
    store = A3NonceStore(
        clock=A3Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
        random_bytes=lambda size: b"\x01" * size,
    )
    return A3ProtocolCoordinator(
        A2FashionMNISTMLP().eval(),
        (build_a4_neural_verification_profile(compile_a4_neural_profile(profile)),),
        store=store,
    )


def _issue(coordinator: A3ProtocolCoordinator, identity_id: bytes) -> bytes:
    response = coordinator.issue_challenge(
        {
            "version": 1,
            "model_id": 1,
            "identity_id": identity_id,
            "scope_id": 1,
            "image": _image(),
        }
    )
    assert response["status"] == "challenge"
    return response["message"]


def test_exact_a4_relation_consumes_one_a3_challenge_once(
    a4_profile: A4PublicProfile,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """A4 exact accept 经 A3 原子 consume 后最多调用一次 protected model。"""
    coordinator = _coordinator(a4_profile)
    message = _issue(coordinator, a4_profile.identity_id)
    proof = a4_proof_factory(message, a4_profile, bytes(range(32, 64)))

    accepted = coordinator.respond(message, proof, _image())
    replayed = coordinator.respond(message, proof, _image())

    assert accepted["status"] == "protected"
    assert replayed == {"version": 3, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_model_calls == 1


def test_invalid_a4_proof_does_not_consume_and_valid_retry_can_commit(
    a4_profile: A4PublicProfile,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """A4 relation reject 保持零模型调用且不消耗 pending challenge。"""
    coordinator = _coordinator(a4_profile)
    message = _issue(coordinator, a4_profile.identity_id)
    proof = a4_proof_factory(message, a4_profile, bytes(range(32, 64)))
    invalid = bytearray(proof)
    invalid[33] = 2

    rejected = coordinator.respond(message, bytes(invalid), _image())
    after_reject = coordinator.snapshot()
    accepted = coordinator.respond(message, proof, _image())

    assert rejected == {"version": 3, "status": "deny"}
    assert after_reject.protected_model_calls == 0
    assert accepted["status"] == "protected"
    assert coordinator.snapshot().protected_model_calls == 1


def test_neural_a4_relation_consumes_one_a3_challenge_once(
    a4_profile: A4PublicProfile,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """A4-C1 accept 必须复用 A3 原子 consume 和单次模型调用边界。"""
    coordinator = _neural_coordinator(a4_profile)
    message = _issue(coordinator, a4_profile.identity_id)
    proof = a4_proof_factory(message, a4_profile, bytes(range(32, 64)))

    accepted = coordinator.respond(message, proof, _image())
    replayed = coordinator.respond(message, proof, _image())

    assert accepted["status"] == "protected"
    assert replayed == {"version": 3, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_model_calls == 1


def test_neural_a4_reject_keeps_zero_calls_and_allows_valid_retry(
    a4_profile: A4PublicProfile,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """A4-C1 reject 不能消费 pending challenge 或进入 protected model。"""
    coordinator = _neural_coordinator(a4_profile)
    message = _issue(coordinator, a4_profile.identity_id)
    proof = a4_proof_factory(message, a4_profile, bytes(range(32, 64)))
    invalid = bytearray(proof)
    invalid[33] = 2

    rejected = coordinator.respond(message, bytes(invalid), _image())
    after_reject = coordinator.snapshot()
    accepted = coordinator.respond(message, proof, _image())

    assert rejected == {"version": 3, "status": "deny"}
    assert after_reject.protected_model_calls == 0
    assert accepted["status"] == "protected"
    assert coordinator.snapshot().protected_model_calls == 1
