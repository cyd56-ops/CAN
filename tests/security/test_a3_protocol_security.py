"""A3-v1 防御性安全边界测试。"""

import hashlib

import pytest
import torch

from can.access import (
    A3Clock,
    A3Evidence,
    A3EvidenceCode,
    A3NonceStore,
    A3ProtocolCoordinator,
    A3VerificationProfile,
)
from can.model.a2_mlp import A2FashionMNISTMLP


class _Clock:
    wall_ms = 1_700_000_000_000
    mono_ns = 5_000_000_000


def _images(value: float = 0.25) -> torch.Tensor:
    return torch.full((1, 1, 28, 28), value, dtype=torch.float32)


def _coordinator() -> tuple[A3ProtocolCoordinator, bytes, _Clock]:
    identity = bytes(range(32))
    clock = _Clock()

    def verifier(message: bytes, proof: bytes) -> object:
        code = A3EvidenceCode.PROOF_ACCEPT if proof == b"valid" else A3EvidenceCode.PROOF_REJECT
        return A3Evidence(code, identity, hashlib.sha256(message).digest(), 11)

    profile = A3VerificationProfile(identity, 11, verifier)
    store = A3NonceStore(
        clock=A3Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
        random_bytes=lambda size: b"\x01" * size,
    )
    coordinator = A3ProtocolCoordinator(A2FashionMNISTMLP().eval(), (profile,), store=store)
    return coordinator, identity, clock


def _request(identity: bytes, image: torch.Tensor) -> dict[str, object]:
    return {
        "version": 1,
        "model_id": 1,
        "identity_id": identity,
        "scope_id": 1,
        "image": image,
    }


def _challenge(coordinator: A3ProtocolCoordinator, identity: bytes) -> bytes:
    response = coordinator.issue_challenge(_request(identity, _images()))
    assert response["status"] == "challenge"
    return response["message"]


def test_request_cannot_inject_route_policy_evidence_or_decision() -> None:
    """请求字段不能替换 A3 entry、policy、profile、evidence 或 decision。"""
    coordinator, identity, _ = _coordinator()
    request = _request(identity, _images())
    request.update(
        {
            "entry": "public",
            "policy": "weak",
            "evidence": {"code": "proof_accept"},
            "decision": True,
        }
    )

    response = coordinator.issue_challenge(request)

    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().challenge_issues == 0


@pytest.mark.parametrize("proof", [None, True, b"", b"x" * 65_536, bytearray(b"valid")])
def test_noncanonical_proof_types_and_lengths_have_zero_protected_calls(proof: object) -> None:
    """proof 类型和边界错误必须在 verifier/model 前拒绝。"""
    coordinator, identity, _ = _coordinator()
    message = _challenge(coordinator, identity)

    response = coordinator.respond(message, proof, _images())

    assert response == {"version": 3, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_model_calls == 0


def test_unknown_identity_cannot_create_a_challenge_or_call_verifier() -> None:
    """未知 identity 不得从本地 profile registry 得到弱回退。"""
    coordinator, _, _ = _coordinator()
    unknown = bytes([255]) * 32

    response = coordinator.issue_challenge(_request(unknown, _images()))

    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().challenge_issues == 0


def test_nonce_collision_never_replaces_an_existing_pending_record() -> None:
    """可信随机源 nonce 冲突必须拒绝且不能覆盖旧 challenge。"""
    coordinator, identity, _ = _coordinator()
    first = coordinator.issue_challenge(_request(identity, _images()))
    second = coordinator.issue_challenge(_request(identity, _images()))

    assert first["status"] == "challenge"
    assert second == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().challenge_issues == 1
    assert coordinator.snapshot().challenge_denies == 1


def test_input_tamper_cannot_reuse_message_for_a_different_snapshot() -> None:
    """业务输入摘要绑定必须阻止跨输入复用 proof message。"""
    coordinator, identity, _ = _coordinator()
    message = _challenge(coordinator, identity)

    response = coordinator.respond(message, b"valid", _images(0.75))

    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().verifier_calls == 0
    assert coordinator.snapshot().protected_model_calls == 0


def test_separate_nonce_stores_do_not_share_authorization_state() -> None:
    """nonce 状态必须属于显式可信 store, 不能落入全局可变状态."""
    first, identity, _ = _coordinator()
    second, _, _ = _coordinator()
    first_message = _challenge(first, identity)
    second_message = _challenge(second, identity)

    assert first_message == second_message
    assert first.respond(first_message, b"valid", _images())["status"] == "protected"
    assert second.respond(second_message, b"valid", _images())["status"] == "protected"
