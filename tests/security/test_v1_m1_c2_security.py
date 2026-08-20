"""V1-M1-C2 route isolation、post-commit failure 与 schema 安全测试。"""

from __future__ import annotations

import torch

from _v1_support import V1_TEST_RESPONSE, V1NonceSource, V1TestClock, build_v1_accepting_commitment
from can.access import A3V2Clock, A3V2TranscriptStore, V1M1C2Coordinator, V1M1C2Cut, V1M1C2Policy
from can.model import V1Cifar100ResNet18
from can.reference import V1PublicProfile, V1Response, build_v1_conformance_profile
from can.verifier import compile_v1_neural_profile

IDENTITY = bytes(range(32))


def _image() -> torch.Tensor:
    return torch.zeros((1, 3, 32, 32), dtype=torch.uint8)


def _build(
    events: list[str] | None = None,
) -> tuple[V1M1C2Coordinator, V1Cifar100ResNet18, V1PublicProfile, V1TestClock]:
    model = V1Cifar100ResNet18().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    profile = build_v1_conformance_profile(IDENTITY)
    clock = V1TestClock()
    coordinator = V1M1C2Coordinator(
        compile_v1_neural_profile(profile),
        model,
        cut=V1M1C2Cut.LAYER2,
        policy=V1M1C2Policy(public_entry_enabled=True),
        store=A3V2TranscriptStore(
            clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
            random_bytes=V1NonceSource(),
        ),
        challenge_sampler=lambda _degree, _weight: (1, 0, 0, 0, 0, 0, 0, -1),
        event_sink=None if events is None else events.append,
    )
    return coordinator, model, profile, clock


def test_malformed_replay_expiry_and_abort_have_no_business_fallback() -> None:
    """所有 pre-execution 失败都不能执行 public head 或 protected suffix。"""
    coordinator, _model, profile, clock = _build()
    commitment = build_v1_accepting_commitment(profile).encode()

    assert coordinator.respond_protected(b"bad") == {"version": 5, "status": "deny"}
    issued = coordinator.begin_protected(_image(), commitment)
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()
    assert coordinator.abort_protected(b"bad") == {"version": 5, "status": "deny"}
    protected = coordinator.respond_protected(response)
    assert protected["version"] == 5
    assert protected["status"] == "protected"
    assert set(protected) == {"version", "status", "class_id"}
    assert coordinator.respond_protected(response) == {"version": 5, "status": "deny"}
    assert coordinator.handle_public(_image(), decision="protected") == {
        "version": 5,
        "status": "deny",
    }

    alternate_response = tuple(
        tuple(
            2 if row_index == 0 and coefficient_index == 0 else value
            for coefficient_index, value in enumerate(row)
        )
        for row_index, row in enumerate(V1_TEST_RESPONSE)
    )
    issued_expiring = coordinator.begin_protected(
        _image(), build_v1_accepting_commitment(profile, response=alternate_response).encode()
    )
    assert issued_expiring["status"] == "challenge"
    clock.mono_ns += 60_000 * 1_000_000
    expired = coordinator.respond_protected(
        V1Response(issued_expiring["transcript_id"], alternate_response).encode()
    )
    assert expired == {"version": 5, "status": "deny"}
    assert coordinator.snapshot().protected_calls == 1


def test_post_commit_suffix_error_is_not_pre_execution_deny() -> None:
    """suffix 已启动后异常必须保留 protected execution error 事件与真实调用边界。"""
    events: list[str] = []
    coordinator, model, profile, _clock = _build(events)

    def fail_suffix(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        raise RuntimeError("synthetic suffix failure")

    handle = model.layer4.register_forward_hook(fail_suffix)
    try:
        issued = coordinator.begin_protected(
            _image(), build_v1_accepting_commitment(profile).encode()
        )
        assert issued["status"] == "challenge"
        result = coordinator.respond_protected(
            V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()
        )
    finally:
        handle.remove()

    assert result == {"version": 5, "status": "deny"}
    assert "suffix_start" in events
    assert "protected_execution_error:suffix" in events
    assert coordinator.snapshot().protected_calls == 1
