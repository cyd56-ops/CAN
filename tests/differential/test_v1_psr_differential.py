"""V1-P2-PSR-E1 与既有 exact relation 的差分测试。"""

from __future__ import annotations

import hashlib

from can.experiments.v1_psr import (
    V1_PSR_CHALLENGE_SET,
    V1PSROutcome,
    build_v1_generated_key_fixture,
    compute_v1_commitment,
    compute_v1_response,
    sample_v1_secret,
)
from can.reference import (
    V1EvidenceCode,
    V1Response,
    v1_negacyclic_convolution,
    verify_v1_ref,
)

SEED = bytes(range(32))
IDENTITY = hashlib.sha256(b"CAN V1 PSR differential identity").digest()


def test_every_challenge_preserves_generated_key_exact_relation() -> None:
    """112 个 challenge 的可发出构造都必须被同一 generated public key 接受。"""
    secret = sample_v1_secret(SEED)
    transcript_id = hashlib.sha256(b"all challenge differential").digest()
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        for challenge in V1_PSR_CHALLENGE_SET:
            mask = tuple(
                tuple(-coefficient for coefficient in shift)
                for secret_polynomial in secret
                for shift in (v1_negacyclic_convolution(challenge.coefficients, secret_polynomial),)
            )
            commitment = compute_v1_commitment(fixture.profile, mask)
            response = V1Response(
                transcript_id,
                compute_v1_response(secret, mask, challenge),
            )

            evidence = verify_v1_ref(
                commitment.encode(),
                challenge.encode(),
                response.encode(),
                transcript_id,
                fixture.profile,
            )

            assert all(coefficient == 0 for row in response.polynomials for coefficient in row)
            assert evidence.code is V1EvidenceCode.RELATION_ACCEPT


def test_first_sampled_emit_has_zero_exact_false_rejects() -> None:
    """固定 sampler 首个 emit 必须保持 signed response 与 exact verifier 一致。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        emitted = None
        for retry_index in range(5_000):
            transcript_id = hashlib.sha256(retry_index.to_bytes(8, "big")).digest()
            attempt = fixture.prove_attempt(0, retry_index, transcript_id)
            if attempt.outcome is V1PSROutcome.EMIT:
                emitted = (retry_index, transcript_id, attempt)
                break

        assert emitted is not None
        retry_index, transcript_id, attempt = emitted
        assert retry_index == 4_447
        assert attempt.response is not None
        evidence = verify_v1_ref(
            attempt.commitment.encode(),
            attempt.challenge.encode(),
            attempt.response.encode(),
            transcript_id,
            fixture.profile,
        )

    assert evidence.code is V1EvidenceCode.RELATION_ACCEPT
