"""V1-M1-C2 split composition 与 version-5 entry 单元测试。"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from _v1_support import (
    V1_TEST_RESPONSE,
    V1NonceSource,
    V1TestClock,
    build_v1_accepting_commitment,
)
from can.access import (
    V1_M1_C2_PROTECTED_CLASS_COUNT,
    V1_M1_C2_PUBLIC_CLASS_COUNT,
    A3V2Clock,
    A3V2TranscriptStore,
    V1M1C2Coordinator,
    V1M1C2Cut,
    V1M1C2Policy,
)
from can.access.v1_m1_adapter import normalize_v1_m1_uint8_batch
from can.model import V1Cifar100ResNet18
from can.reference import V1PublicProfile, V1Response, build_v1_conformance_profile
from can.verifier import compile_v1_neural_profile

IDENTITY = bytes(range(32))


def _image() -> Tensor:
    return (
        torch.arange(3 * 32 * 32, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, 3, 32, 32)
    )


def _coordinator(
    cut: V1M1C2Cut,
    *,
    public_enabled: bool = True,
    events: list[str] | None = None,
    model: V1Cifar100ResNet18 | None = None,
) -> tuple[V1M1C2Coordinator, V1Cifar100ResNet18, V1PublicProfile]:
    typed_model = V1Cifar100ResNet18().eval() if model is None else model
    for parameter in typed_model.parameters():
        parameter.requires_grad_(False)
    profile = build_v1_conformance_profile(IDENTITY)
    neural_profile = compile_v1_neural_profile(profile)
    clock = V1TestClock()
    coordinator = V1M1C2Coordinator(
        neural_profile,
        typed_model,
        cut=cut,
        policy=V1M1C2Policy(public_entry_enabled=public_enabled),
        store=A3V2TranscriptStore(
            clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
            random_bytes=V1NonceSource(),
        ),
        challenge_sampler=lambda _degree, _weight: (1, 0, 0, 0, 0, 0, 0, -1),
        event_sink=None if events is None else events.append,
    )
    return coordinator, typed_model, profile


def _hook_counter(module: torch.nn.Module) -> tuple[list[int], Callable[[], None]]:
    calls = [0]

    def hook(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        calls[0] += 1

    handle = module.register_forward_hook(hook)
    return calls, handle.remove


@pytest.mark.parametrize("cut", tuple(V1M1C2Cut))
def test_public_entry_executes_prefix_and_head_only(cut: V1M1C2Cut) -> None:
    """每个完整 stage cut 的 public path 都不调用 verifier 或 protected suffix。"""
    events: list[str] = []
    coordinator, model, _profile = _coordinator(cut, events=events)
    head = coordinator.public_head
    assert head is not None
    prefix_calls, remove_prefix = _hook_counter(model.layer2)
    suffix_module = model.classifier if cut is V1M1C2Cut.LAYER4 else model.layer4
    suffix_calls, remove_suffix = _hook_counter(suffix_module)
    head_calls, remove_head = _hook_counter(head)
    try:
        result = coordinator.handle_public(_image())
    finally:
        remove_prefix()
        remove_suffix()
        remove_head()

    assert result["version"] == 5
    assert result["status"] == "public"
    assert 0 <= result["coarse_class_id"] < V1_M1_C2_PUBLIC_CLASS_COUNT
    assert prefix_calls[0] == 1
    assert suffix_calls[0] == 0
    assert head_calls[0] == 1
    assert coordinator.snapshot().verifier_calls == 0
    assert coordinator.snapshot().protected_calls == 0
    assert "verifier_accept" not in events


@pytest.mark.parametrize("cut", tuple(V1M1C2Cut))
def test_protected_entry_returns_fine_class_and_matches_direct_r2(cut: V1M1C2Cut) -> None:
    """protected split logits 必须与同一 frozen R2 direct logits 逐元素相等。"""
    events: list[str] = []
    coordinator, model, profile = _coordinator(cut, events=events)
    image = _image()
    captured: list[Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: object, output: object) -> None:
        assert type(output) is Tensor
        captured.append(output.detach().clone())

    handle = model.classifier.register_forward_hook(capture)
    try:
        issued = coordinator.begin_protected(image, build_v1_accepting_commitment(profile).encode())
        assert issued["status"] == "challenge"
        response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()
        result = coordinator.respond_protected(response)
    finally:
        handle.remove()

    with torch.inference_mode():
        direct = model(normalize_v1_m1_uint8_batch(image)).detach()

    assert result["version"] == 5
    assert result["status"] == "protected"
    assert 0 <= result["class_id"] < V1_M1_C2_PROTECTED_CLASS_COUNT
    assert len(captured) == 1
    assert torch.equal(captured[0], direct)
    assert events == [
        "verifier_accept",
        "coordinator_commit(PROTECTED)",
        "preprocess_start",
        "prefix_start",
        "suffix_start",
        "internal_result_commit",
        "response_release",
    ]
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1


def test_c2_uses_one_r2_tree_and_separate_public_head_storage() -> None:
    """public head 参数不得复制或共享 accepted R2 state。"""
    coordinator, model, _profile = _coordinator(V1M1C2Cut.LAYER2)
    head = coordinator.public_head
    assert head is not None
    model_storage = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    head_storage = {parameter.untyped_storage().data_ptr() for parameter in head.parameters()}
    assert model_storage.isdisjoint(head_storage)
    names = tuple(name for name, _ in model.named_parameters())
    assert len(names) == len(set(names))


def test_public_and_protected_entries_reject_route_field_injection_without_calls() -> None:
    """entry、cut、head 和 threshold 等请求字段不能改变可信部署路由。"""
    coordinator, model, profile = _coordinator(V1M1C2Cut.LAYER2)
    calls: list[int] = [0]

    def forbidden(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        calls[0] += 1

    handles = [
        model.layer2.register_forward_hook(forbidden),
        model.layer4.register_forward_hook(forbidden),
    ]
    try:
        public = coordinator.handle_public(_image(), cut="layer4", threshold=0)
        protected = coordinator.begin_protected(
            _image(),
            build_v1_accepting_commitment(profile).encode(),
            entry="public",
            head="protected",
        )
    finally:
        for handle in handles:
            handle.remove()

    assert public == {"version": 5, "status": "deny"}
    assert protected == {"version": 5, "status": "deny"}
    assert calls == [0]
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 0
    assert snapshot.protected_calls == 0


def test_relation_reject_never_falls_back_to_public() -> None:
    """canonical relation reject 必须是 zero-business-call C2 deny。"""
    coordinator, model, profile = _coordinator(V1M1C2Cut.LAYER2)
    head = coordinator.public_head
    assert head is not None
    public_calls, remove_public = _hook_counter(head)
    protected_calls, remove_protected = _hook_counter(model.layer4)
    try:
        issued = coordinator.begin_protected(
            _image(), build_v1_accepting_commitment(profile).encode()
        )
        assert issued["status"] == "challenge"
        changed = [list(polynomial) for polynomial in V1_TEST_RESPONSE]
        changed[0][0] += 1
        response = V1Response(issued["transcript_id"], changed).encode()
        result = coordinator.respond_protected(response)
    finally:
        remove_public()
        remove_protected()

    assert result == {"version": 5, "status": "deny"}
    assert public_calls[0] == 0
    assert protected_calls[0] == 0
    snapshot = coordinator.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.protected_calls == 0


def test_public_entry_disabled_is_fail_closed() -> None:
    """未由可信策略启用 public entry 时不能保留或执行 public head。"""
    coordinator, _model, _profile = _coordinator(V1M1C2Cut.LAYER2, public_enabled=False)
    assert coordinator.public_head is None
    assert coordinator.handle_public(_image()) == {"version": 5, "status": "deny"}
