"""A3-v1 canonical codec、状态机和协调器单元测试。"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from can.access import (
    A3_CHALLENGE_TTL_MS,
    A3_MESSAGE_SIZE,
    A3Clock,
    A3Evidence,
    A3EvidenceCode,
    A3Message,
    A3NonceStore,
    A3ProtocolCoordinator,
    A3ProtocolInputError,
    A3VerificationProfile,
    canonicalize_a3_image,
    parse_a3_message,
)
from can.model.a2_mlp import A2FashionMNISTMLP
from can.verifier import A1Evidence, A1EvidenceCode


class _FakeClock:
    def __init__(self) -> None:
        self.wall_ms = 1_700_000_000_000
        self.mono_ns = 5_000_000_000

    def clock(self) -> A3Clock:
        return A3Clock(lambda: self.wall_ms, lambda: self.mono_ns)


class _NonceSource:
    def __init__(self) -> None:
        self._counter = 0

    def __call__(self, size: int) -> bytes:
        self._counter += 1
        return self._counter.to_bytes(size, byteorder="big", signed=False)


def _image(value: float = 0.25) -> torch.Tensor:
    return torch.full((1, 1, 28, 28), value, dtype=torch.float32)


def _request(identity: bytes, image: torch.Tensor) -> dict[str, object]:
    return {
        "version": 1,
        "model_id": 1,
        "identity_id": identity,
        "scope_id": 1,
        "image": image,
    }


def _make_coordinator(
    *,
    verifier: object | None = None,
    clock: _FakeClock | None = None,
) -> tuple[A3ProtocolCoordinator, bytes, _FakeClock]:
    fake_clock = clock or _FakeClock()
    identity = bytes(range(32))
    if verifier is None:

        def verifier(message: bytes, proof: bytes) -> object:
            digest = hashlib.sha256(message).digest()
            code = A3EvidenceCode.PROOF_ACCEPT if proof == b"valid" else A3EvidenceCode.PROOF_REJECT
            return A3Evidence(code, identity, digest, 7)

    profile = A3VerificationProfile(identity, 7, verifier)  # type: ignore[arg-type]
    store = A3NonceStore(clock=fake_clock.clock(), random_bytes=_NonceSource())
    torch.manual_seed(20_260_808)
    coordinator = A3ProtocolCoordinator(
        A2FashionMNISTMLP().eval(),
        (profile,),
        store=store,
    )
    return coordinator, identity, fake_clock


def _issue(coordinator: A3ProtocolCoordinator, identity: bytes, image: torch.Tensor) -> bytes:
    response = coordinator.issue_challenge(_request(identity, image))
    assert response["status"] == "challenge"
    message = response["message"]
    assert type(message) is bytes
    return message


def test_message_round_trip_is_exactly_133_bytes() -> None:
    """A3 message 编码必须固定长度且解析后无损往返。"""
    message = A3Message(
        1,
        1,
        bytes(range(32)),
        1,
        1_700_000_000_000,
        1_700_000_060_000,
        bytes(reversed(range(32))),
        bytes([9]) * 32,
    )

    encoded = message.encode()

    assert len(encoded) == A3_MESSAGE_SIZE
    assert parse_a3_message(encoded) == message


@pytest.mark.parametrize("tamper_offset", [0, 14, 51, 53, 61])
def test_message_tamper_is_rejected(tamper_offset: int) -> None:
    """canonical message 的域、版本和时间字段篡改必须解析失败。"""
    message = A3Message(
        1,
        1,
        bytes(range(32)),
        1,
        1_700_000_000_000,
        1_700_000_060_000,
        bytes(range(32)),
        bytes([9]) * 32,
    )
    tampered = bytearray(message.encode())
    tampered[tamper_offset] ^= 1

    with pytest.raises(A3ProtocolInputError):
        parse_a3_message(bytes(tampered))


@pytest.mark.parametrize("tamper_offset", [19, 69, 101])
def test_binding_field_tamper_remains_canonical_but_is_rejected_by_coordinator(
    tamper_offset: int,
) -> None:
    """身份, nonce 和摘要篡改保持格式合法, 但必须在绑定层拒绝。"""
    coordinator, identity, _ = _make_coordinator()
    image = _image()
    message = bytearray(_issue(coordinator, identity, image))
    message[tamper_offset] ^= 1

    parsed = parse_a3_message(bytes(message))
    response = coordinator.respond(parsed.encode(), b"valid", image)

    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().verifier_calls == 0


def test_input_digest_is_deterministic_and_negative_zero_is_rejected() -> None:
    """A3 image hash 使用固定快照, negative zero 进入拒绝路径."""
    first_snapshot, first_digest = canonicalize_a3_image(_image())
    second_snapshot, second_digest = canonicalize_a3_image(_image())

    assert torch.equal(first_snapshot, second_snapshot)
    assert first_digest == second_digest
    negative_zero = torch.tensor([[[[-0.0] * 28] * 28]], dtype=torch.float32)
    with pytest.raises(A3ProtocolInputError):
        canonicalize_a3_image(negative_zero)


def test_missing_a4_profile_is_default_closed() -> None:
    """没有本地 A4 profile 时 challenge 和 response 都固定拒绝。"""
    coordinator = A3ProtocolCoordinator(A2FashionMNISTMLP().eval())

    challenge = coordinator.issue_challenge(_request(bytes(range(32)), _image()))
    response = coordinator.respond(bytes(A3_MESSAGE_SIZE), b"valid", _image())

    assert challenge == {"version": 3, "status": "deny"}
    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().protected_model_calls == 0


def test_valid_proof_consumes_once_and_replay_has_zero_model_call() -> None:
    """测试 stub 的 exact accept 只能让一个 nonce 调用一次 protected model。"""
    coordinator, identity, _ = _make_coordinator()
    image = _image()
    message = _issue(coordinator, identity, image)

    first = coordinator.respond(message, b"valid", image)
    second = coordinator.respond(message, b"valid", image)

    assert first["status"] == "protected"
    assert second == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().protected_model_calls == 1
    assert coordinator.snapshot().allow_commits == 1
    assert coordinator.snapshot().deny_commits == 1


def test_invalid_proof_does_not_consume_pending_challenge() -> None:
    """invalid proof 只拒绝, 不消耗 pending nonce, 随后合法 stub 可继续测试."""
    coordinator, identity, image_clock = _make_coordinator()
    image = _image()
    message = _issue(coordinator, identity, image)

    rejected = coordinator.respond(message, b"invalid", image)
    accepted = coordinator.respond(message, b"valid", image)

    assert rejected == {"version": 3, "status": "deny"}
    assert accepted["status"] == "protected"
    assert image_clock.mono_ns == 5_000_000_000
    assert coordinator.snapshot().protected_model_calls == 1


def test_expiry_is_checked_before_and_inside_atomic_consume() -> None:
    """过期 challenge 不调用 verifier 或 protected model。"""
    coordinator, identity, clock = _make_coordinator()
    image = _image()
    message = _issue(coordinator, identity, image)
    clock.mono_ns += A3_CHALLENGE_TTL_MS * 1_000_000

    response = coordinator.respond(message, b"valid", image)

    assert response == {"version": 3, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_model_calls == 0


def test_monotonic_clock_rollback_is_fail_closed() -> None:
    """可信 monotonic clock 回拨必须拒绝, 不能延长 challenge。"""
    coordinator, identity, clock = _make_coordinator()
    image = _image()
    message = _issue(coordinator, identity, image)
    clock.mono_ns -= 1

    response = coordinator.respond(message, b"valid", image)

    assert response == {"version": 3, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_model_calls == 0


def test_input_substitution_is_rejected_before_verifier() -> None:
    """不同业务图像不能复用原 challenge。"""
    coordinator, identity, image_clock = _make_coordinator()
    original = _image(0.25)
    message = _issue(coordinator, identity, original)

    response = coordinator.respond(message, b"valid", _image(0.5))

    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().verifier_calls == 0
    assert coordinator.snapshot().protected_model_calls == 0
    assert image_clock.mono_ns == 5_000_000_000


def test_a1_evidence_is_not_an_a3_accept() -> None:
    """A1 numeric evidence 不能伪造 A3 authentication evidence。"""
    identity = bytes(range(32))

    def verifier(message: bytes, proof: bytes) -> object:
        del message, proof
        return A1Evidence(A1EvidenceCode.NUMERIC_ACCEPT)

    coordinator, _, _ = _make_coordinator(verifier=verifier)
    image = _image()
    message = _issue(coordinator, identity, image)

    response = coordinator.respond(message, b"valid", image)

    assert response == {"version": 3, "status": "deny"}
    assert coordinator.snapshot().protected_model_calls == 0


def test_post_commit_model_failure_is_one_call_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """consume 后模型异常仍计一次调用, 且不回滚或降级重试。"""
    coordinator, identity, _ = _make_coordinator()
    image = _image()
    message = _issue(coordinator, identity, image)

    def fail_forward(self: A2FashionMNISTMLP, images: torch.Tensor) -> torch.Tensor:
        del self, images
        raise RuntimeError("synthetic protected model failure")

    monkeypatch.setattr(A2FashionMNISTMLP, "forward", fail_forward)

    response = coordinator.respond(message, b"valid", image)
    replay = coordinator.respond(message, b"valid", image)

    assert response == {"version": 3, "status": "deny"}
    assert replay == {"version": 3, "status": "deny"}
    snapshot = coordinator.snapshot()
    assert snapshot.protected_model_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.deny_commits == 1


def test_concurrent_valid_replay_has_one_consume_and_one_model_call() -> None:
    """并发合法 replay 只能有一个 atomic consume winner。"""
    _, identity, _ = _make_coordinator()
    barrier = Barrier(6)

    def verifier(bound_message: bytes, proof: bytes) -> object:
        assert proof == b"valid"
        barrier.wait(timeout=5)
        return A3Evidence(
            A3EvidenceCode.PROOF_ACCEPT,
            identity,
            hashlib.sha256(bound_message).digest(),
            7,
        )

    coordinator = _make_coordinator(verifier=verifier)[0]
    image = _image()
    message = _issue(coordinator, identity, image)
    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = tuple(
            executor.submit(coordinator.respond, message, b"valid", image) for _ in range(6)
        )
        results = tuple(future.result(timeout=10) for future in responses)

    assert sum(result["status"] == "protected" for result in results) == 1
    assert sum(result["status"] == "deny" for result in results) == 5
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 6
    assert snapshot.allow_commits == 1
    assert snapshot.deny_commits == 5
    assert snapshot.protected_model_calls == 1
