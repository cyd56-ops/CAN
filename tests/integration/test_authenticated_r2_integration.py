"""V1-M1 AuthenticatedR2 神经 Gate Layer 与业务语义组合验收。"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_RESPONSE,
    V1NonceSource,
    V1TestClock,
    build_v1_accepting_commitment,
)
from can.access import (
    A3V2Clock,
    A3V2TranscriptStore,
    AuthenticatedR2,
    normalize_v1_m1_uint8_batch,
)
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


def test_neural_accept_runs_exactly_the_direct_frozen_r2_semantics() -> None:
    """allow 后唯一 R2 forward 的 logits 必须与同一冻结模型直接推理逐元素相等。"""
    torch.manual_seed(0)
    model = V1Cifar100ResNet18().eval()
    image = _image()
    with torch.inference_mode():
        expected = model(normalize_v1_m1_uint8_batch(image)).detach().clone()

    observed: list[Tensor] = []

    def capture_logits(_module: object, _inputs: object, output: object) -> None:
        observed.append(cast(Tensor, output).detach().clone())

    handle = model.register_forward_hook(capture_logits)
    try:
        authenticated, public_profile = _authenticated_r2(model)
        issued = authenticated.begin(
            image,
            build_v1_accepting_commitment(public_profile).encode(),
        )
        assert issued["status"] == "challenge"
        response = V1Response(issued["transcript_id"], V1_TEST_RESPONSE).encode()

        assert authenticated.respond(response) == {"version": 4, "status": "protected"}
        assert authenticated.respond(response) == {"version": 4, "status": "deny"}
    finally:
        handle.remove()

    assert len(observed) == 1
    assert torch.equal(observed[0], expected)
    snapshot = authenticated.snapshot()
    assert snapshot.verifier_calls == 1
    assert snapshot.allow_commits == 1
    assert snapshot.protected_calls == 1
    assert snapshot.protected_responses == 1
