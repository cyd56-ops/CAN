"""A4-C1 graph 与精确关系的确定性差分测试。"""

from __future__ import annotations

import random

import pytest

from can.reference.a4 import A4_MODULUS, A4_VECTOR_SIZE, A4PublicProfile, verify_a4_ref
from can.verifier.a4 import (
    A4NeuralEvidenceCode,
    A4NeuralProfile,
    _evaluate_a4_core,
    _relu,
    verify_a4_neural,
)
from conftest import A4ProofFactory


def test_point_pulse_is_exact_over_the_complete_shifted_residual_range() -> None:
    """三 ReLU point pulse 在完整证明范围内只能接受整数零点。"""
    for offset in range(-2_377_799, 2_359_369):
        pulse = _relu(offset + 1) - 2 * _relu(offset) + _relu(offset - 1)
        if pulse != int(offset == 0):
            pytest.fail(f"point pulse mismatch at {offset}")


def test_norm_violation_is_exact_over_signed_int8() -> None:
    """两 ReLU norm violation 对全部 signed-int8 系数精确。"""
    for coefficient in range(-128, 128):
        violation = _relu(coefficient - 1) + _relu(-coefficient - 1)
        assert (violation == 0) is (coefficient in (-1, 0, 1))


def test_core_matches_direct_relation_for_seeded_canonical_vectors(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
) -> None:
    """固定种子 canonical 域样本必须逐输入匹配 direct exact predicate。"""
    rng = random.Random(20260812)
    for case_index in range(48):
        if case_index % 3 == 0:
            z = tuple(rng.choice((-1, 0, 1)) for _ in range(A4_VECTOR_SIZE))
        else:
            z = tuple(rng.randint(-128, 127) for _ in range(A4_VECTOR_SIZE))
        actual_y = tuple(
            sum(a * coefficient for a, coefficient in zip(row, z, strict=True)) % A4_MODULUS
            for row in a4_profile.matrix
        )
        if case_index % 2 == 0:
            y = actual_y
        else:
            mutated = list(actual_y)
            mutated[case_index % len(mutated)] = (
                mutated[case_index % len(mutated)] + 1
            ) % A4_MODULUS
            y = tuple(mutated)
        expected = int(max(abs(value) for value in z) <= 1 and y == actual_y)
        assert _evaluate_a4_core(y, z, a4_neural_profile) == expected


def test_raw_neural_and_reference_verifiers_match_valid_proof_family(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """多组公开 conformance proof 必须逐项匹配 reference。"""
    for seed in range(12):
        salt = bytes((seed + index) % 256 for index in range(32))
        proof = a4_proof_factory(a4_message, a4_profile, salt)
        reference = verify_a4_ref(a4_message, proof, a4_profile)
        neural = verify_a4_neural(a4_message, proof, a4_neural_profile)

        assert reference.accepted
        assert neural.code is A4NeuralEvidenceCode.NEURAL_ACCEPT
