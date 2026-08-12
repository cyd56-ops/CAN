"""V1-P2 negacyclic convolution 与 exact relation 的独立差分测试。"""

from __future__ import annotations

import random

from can.reference import (
    V1_CHALLENGE_WEIGHT,
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RESPONSE_BOUND,
    V1_RING_DEGREE,
    V1Challenge,
    V1Commitment,
    V1EvidenceCode,
    V1Response,
    build_v1_conformance_profile,
    v1_negacyclic_convolution,
    verify_v1_ref,
)


def _independent_convolution(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    output = [0] * V1_RING_DEGREE
    for output_index in range(V1_RING_DEGREE):
        total = 0
        for left_index in range(V1_RING_DEGREE):
            right_index = output_index - left_index
            if right_index >= 0:
                total += left[left_index] * right[right_index]
            else:
                total -= left[left_index] * right[right_index + V1_RING_DEGREE]
        output[output_index] = total
    return tuple(output)


def _independent_commitment(
    matrix: tuple[tuple[tuple[int, ...], ...], ...],
    target: tuple[tuple[int, ...], ...],
    challenge: tuple[int, ...],
    response: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    commitment: list[tuple[int, ...]] = []
    for row_index in range(2):
        lhs = [0] * V1_RING_DEGREE
        for column_index in range(2):
            product = _independent_convolution(
                matrix[row_index][column_index], response[column_index]
            )
            for coefficient_index in range(V1_RING_DEGREE):
                lhs[coefficient_index] += product[coefficient_index]
        for coefficient_index in range(V1_RING_DEGREE):
            lhs[coefficient_index] += response[2 + row_index][coefficient_index]
        challenge_target = _independent_convolution(challenge, target[row_index])
        commitment.append(
            tuple(
                (lhs[index] - challenge_target[index]) % V1_MODULUS
                for index in range(V1_RING_DEGREE)
            )
        )
    return tuple(commitment)


def test_negacyclic_convolution_matches_independent_oracle() -> None:
    """固定种子多项式族必须与不同索引公式的 oracle 完全一致。"""
    generator = random.Random(20_260_811)

    for _ in range(256):
        left = tuple(generator.randint(-257, 257) for _ in range(V1_RING_DEGREE))
        right = tuple(generator.randint(-8, 8) for _ in range(V1_RING_DEGREE))
        assert v1_negacyclic_convolution(left, right) == _independent_convolution(left, right)


def test_exact_relation_accepts_independently_constructed_vectors() -> None:
    """独立 oracle 构造的 canonical relation 向量必须零 false reject。"""
    generator = random.Random(20_260_812)
    profile = build_v1_conformance_profile(bytes(range(32)))

    for vector_index in range(128):
        support = generator.sample(range(V1_RING_DEGREE), V1_CHALLENGE_WEIGHT)
        challenge_values = [0] * V1_RING_DEGREE
        for index in support:
            challenge_values[index] = generator.choice((-1, 1))
        challenge = V1Challenge(V1_PROFILE_ID, challenge_values)
        response_values = tuple(
            tuple(
                generator.randint(-V1_RESPONSE_BOUND, V1_RESPONSE_BOUND)
                for _ in range(V1_RING_DEGREE)
            )
            for _ in range(4)
        )
        commitment_values = _independent_commitment(
            profile.matrix,
            profile.target,
            challenge.coefficients,
            response_values,
        )
        transcript_id = vector_index.to_bytes(32, byteorder="big", signed=False)

        evidence = verify_v1_ref(
            V1Commitment(V1_PROFILE_ID, commitment_values).encode(),
            challenge.encode(),
            V1Response(transcript_id, response_values).encode(),
            transcript_id,
            profile,
        )

        assert evidence.code is V1EvidenceCode.RELATION_ACCEPT
