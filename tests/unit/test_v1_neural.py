"""V1-C1 coefficient-domain graph 的确定性单元测试。"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

from _v1_support import (
    V1_TEST_CHALLENGE,
    V1_TEST_RESPONSE,
    build_v1_accepting_commitment,
)
from can.reference import (
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RESPONSE_BOUND,
    V1Challenge,
    V1Commitment,
    V1EvidenceCode,
    V1Response,
    build_v1_conformance_profile,
    verify_v1_ref,
)
from can.verifier.v1 import (
    V1_NEURAL_CANDIDATE_ID,
    V1_NEURAL_LAYER_WIDTHS,
    V1NeuralEvidenceCode,
    compile_v1_neural_profile,
    verify_v1_neural,
)

IDENTITY = bytes(range(32))
TRANSCRIPT = hashlib.sha256(b"V1-C1 transcript").digest()


def test_v1_c1_freezes_graph_and_range_ledger() -> None:
    """V1-C1 topology 与公开 range ledger 必须固定。"""
    profile = compile_v1_neural_profile(build_v1_conformance_profile(IDENTITY))
    assert profile.candidate_id == V1_NEURAL_CANDIDATE_ID
    assert tuple((layer.input_width, layer.output_width) for layer in profile.layers) == (
        (56, V1_NEURAL_LAYER_WIDTHS[0]),
        (V1_NEURAL_LAYER_WIDTHS[0], V1_NEURAL_LAYER_WIDTHS[1]),
        (V1_NEURAL_LAYER_WIDTHS[1], 1),
    )
    assert profile.layers[0].output_width == 11056
    assert profile.layers[1].output_width == 17


def test_v1_c1_accepts_exact_relation_and_rejects_tamper() -> None:
    """valid relation、equation tamper 与 norm 边界必须匹配 exact reference。"""
    reference_profile = build_v1_conformance_profile(IDENTITY)
    neural_profile = compile_v1_neural_profile(reference_profile)
    commitment = build_v1_accepting_commitment(
        reference_profile, V1_TEST_CHALLENGE, V1_TEST_RESPONSE
    )
    response = V1Response(TRANSCRIPT, V1_TEST_RESPONSE)
    assert (
        verify_v1_neural(
            commitment.encode(),
            V1_TEST_CHALLENGE.encode(),
            response.encode(),
            TRANSCRIPT,
            neural_profile,
        ).code
        is V1NeuralEvidenceCode.NEURAL_ACCEPT
    )

    changed = [list(polynomial) for polynomial in commitment.polynomials]
    changed[0][0] = (changed[0][0] + 1) % V1_MODULUS
    mutated = V1Commitment(V1_PROFILE_ID, changed)
    assert (
        verify_v1_neural(
            mutated.encode(),
            V1_TEST_CHALLENGE.encode(),
            response.encode(),
            TRANSCRIPT,
            neural_profile,
        ).code
        is V1NeuralEvidenceCode.NEURAL_REJECT
    )

    oversized = list(V1_TEST_RESPONSE)
    oversized[0] = (V1_RESPONSE_BOUND + 1, *oversized[0][1:])
    oversized_response = V1Response(TRANSCRIPT, oversized)
    assert (
        verify_v1_neural(
            commitment.encode(),
            V1_TEST_CHALLENGE.encode(),
            oversized_response.encode(),
            TRANSCRIPT,
            neural_profile,
        ).code
        is V1NeuralEvidenceCode.NEURAL_REJECT
    )


def test_v1_c1_matches_reference_on_independent_valid_vectors() -> None:
    """独立构造的多组 valid response 必须无 false reject。"""
    reference_profile = build_v1_conformance_profile(IDENTITY)
    neural_profile = compile_v1_neural_profile(reference_profile)
    for index in range(8):
        # fixed weight two, with a second non-overlapping support.
        challenge = V1Challenge(
            V1_PROFILE_ID,
            tuple(1 if item in (index, (index + 1) % 8) else 0 for item in range(8)),
        )
        commitment = build_v1_accepting_commitment(reference_profile, challenge, V1_TEST_RESPONSE)
        response = V1Response(index.to_bytes(32, "big"), V1_TEST_RESPONSE)
        exact = verify_v1_ref(
            commitment.encode(),
            challenge.encode(),
            response.encode(),
            response.transcript_id,
            reference_profile,
        )
        neural = verify_v1_neural(
            commitment.encode(),
            challenge.encode(),
            response.encode(),
            response.transcript_id,
            neural_profile,
        )
        assert exact.code is V1EvidenceCode.RELATION_ACCEPT
        assert neural.code is V1NeuralEvidenceCode.NEURAL_ACCEPT


def test_v1_c1_profile_is_immutable() -> None:
    """compiled profile 不能在构造后被请求方改写。"""
    profile = compile_v1_neural_profile(build_v1_conformance_profile(IDENTITY))
    try:
        profile.__setattr__("layers", ())
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("V1-C1 compiled profile must be frozen")
