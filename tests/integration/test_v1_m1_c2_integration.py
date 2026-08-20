"""V1-M1-C2 双入口与 C1 transcript compatibility 集成测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from _v1_support import V1_TEST_RESPONSE, V1NonceSource, V1TestClock, build_v1_accepting_commitment
from can.access import A3V2Clock, A3V2TranscriptStore, V1M1C2Coordinator, V1M1C2Cut, V1M1C2Policy
from can.model import V1Cifar100ResNet18
from can.reference import V1PublicProfile, V1Response, build_v1_conformance_profile
from can.verifier import compile_v1_neural_profile

IDENTITY = bytes(range(32))


def _image() -> torch.Tensor:
    return torch.zeros((1, 3, 32, 32), dtype=torch.uint8)


def _build() -> tuple[V1M1C2Coordinator, V1Cifar100ResNet18, V1PublicProfile]:
    model = V1Cifar100ResNet18().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    profile = build_v1_conformance_profile(IDENTITY)
    clock = V1TestClock()
    coordinator = V1M1C2Coordinator(
        compile_v1_neural_profile(profile),
        model,
        cut=V1M1C2Cut.LAYER3,
        policy=V1M1C2Policy(public_entry_enabled=True),
        store=A3V2TranscriptStore(
            clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
            random_bytes=V1NonceSource(),
        ),
        challenge_sampler=lambda _degree, _weight: (1, 0, 0, 0, 0, 0, 0, -1),
    )
    return coordinator, model, profile


def test_c2_double_entry_keeps_public_and_protected_schemas_disjoint() -> None:
    """public response 不接受 protected schema, protected challenge 使用 version 5。"""
    coordinator, _model, profile = _build()
    public = coordinator.handle_public(_image())
    issued = coordinator.begin_protected(_image(), build_v1_accepting_commitment(profile).encode())

    assert public["version"] == 5
    assert public["status"] == "public"
    assert set(public) == {"version", "status", "coarse_class_id"}
    assert issued["version"] == 5
    assert issued["status"] == "challenge"
    assert set(issued) == {"version", "status", "message", "challenge", "transcript_id"}


def test_concurrent_duplicate_protected_responses_release_one_result() -> None:
    """同一 transcript 的并发 response 只能有一个 protected release。"""
    coordinator, _model, profile = _build()
    issued = coordinator.begin_protected(_image(), build_v1_accepting_commitment(profile).encode())
    assert issued["status"] == "challenge"
    response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(coordinator.respond_protected, (response,) * 32))

    assert sum(result["status"] == "protected" for result in results) == 1
    assert sum(result["status"] == "deny" for result in results) == 31
    snapshot = coordinator.snapshot()
    assert snapshot.terminal_claims == 1
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1
