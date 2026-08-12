"""V1-C1 graph 与 coefficient-domain exact relation 的差分测试。"""

from __future__ import annotations

import random

from can.reference import (
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RESPONSE_BOUND,
    V1Challenge,
    V1Commitment,
    V1Response,
    build_v1_conformance_profile,
    v1_negacyclic_convolution,
    verify_v1_ref,
)
from can.verifier.v1 import V1NeuralEvidenceCode, compile_v1_neural_profile, verify_v1_neural


def _commitment(
    profile: object,
    challenge: V1Challenge,
    response: tuple[tuple[int, ...], ...],
) -> V1Commitment:
    assert hasattr(profile, "matrix") and hasattr(profile, "target")
    outputs: list[tuple[int, ...]] = []
    for row_index in range(2):
        lhs = [0] * 8
        for column_index in range(2):
            product = v1_negacyclic_convolution(
                profile.matrix[row_index][column_index], response[column_index]
            )
            lhs = [left + right for left, right in zip(lhs, product, strict=True)]
        lhs = [left + right for left, right in zip(lhs, response[2 + row_index], strict=True)]
        target = v1_negacyclic_convolution(challenge.coefficients, profile.target[row_index])
        outputs.append(
            tuple((left - right) % V1_MODULUS for left, right in zip(lhs, target, strict=True))
        )
    return V1Commitment(V1_PROFILE_ID, outputs)


def test_v1_c1_matches_reference_for_seeded_boundary_and_tamper_vectors() -> None:
    """valid、norm boundary、equation tamper 全部保持单向 soundness。"""
    profile = build_v1_conformance_profile(bytes(range(32)))
    neural = compile_v1_neural_profile(profile)
    rng = random.Random(20260812)
    for index in range(24):
        challenge_values = [0] * 8
        for support in rng.sample(range(8), 2):
            challenge_values[support] = rng.choice((-1, 1))
        challenge = V1Challenge(V1_PROFILE_ID, challenge_values)
        response_values = tuple(
            tuple(rng.randint(-V1_RESPONSE_BOUND, V1_RESPONSE_BOUND) for _ in range(8))
            for _ in range(4)
        )
        transcript = index.to_bytes(32, "big")
        commitment = _commitment(profile, challenge, response_values)
        response = V1Response(transcript, response_values)
        exact = verify_v1_ref(
            commitment.encode(), challenge.encode(), response.encode(), transcript, profile
        )
        actual = verify_v1_neural(
            commitment.encode(), challenge.encode(), response.encode(), transcript, neural
        )
        assert (actual.code is V1NeuralEvidenceCode.NEURAL_ACCEPT) is exact.accepted
