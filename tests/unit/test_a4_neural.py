"""A4-C1 canonical graph 的确定性单元测试。"""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError

import pytest

from can.reference.a4 import A4_MODULUS, A4_VECTOR_SIZE, A4Proof, A4PublicProfile, verify_a4_ref
from can.verifier.a4 import (
    A4_MULTIPLES,
    A4_NEURAL_CANDIDATE_ID,
    A4_NEURAL_LAYER_WIDTHS,
    A4NeuralEvaluationError,
    A4NeuralEvidenceCode,
    A4NeuralProfile,
    _evaluate_a4_core,
    _run_a4_graph,
    compile_a4_neural_profile,
    verify_a4_neural,
)
from conftest import A4ProofFactory


def test_a4_c1_freezes_topology_and_multiple_domain(a4_profile: A4PublicProfile) -> None:
    """A4-C1 graph topology 与合法倍数集合必须固定。"""
    profile = compile_a4_neural_profile(a4_profile)

    assert profile.candidate_id == A4_NEURAL_CANDIDATE_ID
    assert profile.scale == 1
    assert A4_MULTIPLES == tuple(range(-72, 72))
    assert tuple((layer.input_width, layer.output_width) for layer in profile.layers) == (
        (80, A4_NEURAL_LAYER_WIDTHS[0]),
        (A4_NEURAL_LAYER_WIDTHS[0], A4_NEURAL_LAYER_WIDTHS[1]),
        (A4_NEURAL_LAYER_WIDTHS[1], A4_NEURAL_LAYER_WIDTHS[2]),
    )
    assert len(profile.layers[0].rows) == 3600
    assert len(profile.layers[1].rows) == 1153
    assert len(profile.layers[2].rows) == 1


def test_a4_c1_accepts_public_gadget_proof(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """公开 gadget fixture 的 valid relation 必须被 exact/neural 同时接受。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))

    reference = verify_a4_ref(a4_message, proof, a4_profile)
    neural = verify_a4_neural(a4_message, proof, a4_neural_profile)

    assert reference.accepted
    assert neural.code is A4NeuralEvidenceCode.NEURAL_ACCEPT
    assert neural.accepted
    assert neural.identity_id == a4_profile.identity_id
    assert neural.message_sha256


def test_a4_c1_rejects_norm_and_equation_tamper(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """norm 越界和 syndrome equation tamper 必须 fail closed。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    parsed = A4Proof(
        1, proof[1:33], tuple(value - 256 if value >= 128 else value for value in proof[33:])
    )
    norm_vector = list(parsed.vector)
    norm_vector[0] = 2
    norm_proof = A4Proof(1, parsed.salt, norm_vector).encode()
    salt = bytearray(parsed.salt)
    salt[0] ^= 1
    equation_proof = A4Proof(1, bytes(salt), parsed.vector).encode()

    assert (
        verify_a4_neural(a4_message, norm_proof, a4_neural_profile).code
        is A4NeuralEvidenceCode.NEURAL_REJECT
    )
    assert (
        verify_a4_neural(a4_message, equation_proof, a4_neural_profile).code
        is A4NeuralEvidenceCode.NEURAL_REJECT
    )


def test_a4_c1_input_parser_fail_closed(
    a4_neural_profile: A4NeuralProfile,
    a4_message: bytes,
) -> None:
    """错误类型、proof 长度和 message 长度不能进入 graph。"""
    assert (
        verify_a4_neural(bytearray(a4_message), bytes(105), a4_neural_profile).code
        is A4NeuralEvidenceCode.INPUT_REJECT
    )
    assert (
        verify_a4_neural(a4_message, bytes(104), a4_neural_profile).code
        is A4NeuralEvidenceCode.INPUT_REJECT
    )
    assert (
        verify_a4_neural(a4_message, bytes(105), object()).code
        is A4NeuralEvidenceCode.CONFIG_REJECT
    )


def test_a4_c1_matches_the_integer_predicate_on_deterministic_vectors(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
) -> None:
    """固定种子生成的 canonical `(y,z)` 向量必须与精确整数谓词一致。"""
    rng = random.Random(20260811)
    for _ in range(24):
        z = tuple(rng.choice((-1, 0, 1)) for _ in range(A4_VECTOR_SIZE))
        valid_y = tuple(
            sum(a * coefficient for a, coefficient in zip(row, z, strict=True)) % A4_MODULUS
            for row in a4_profile.matrix
        )
        assert _evaluate_a4_core(valid_y, z, a4_neural_profile) == 1

        wrong_y = list(valid_y)
        wrong_y[0] = (wrong_y[0] + 1) % A4_MODULUS
        assert _evaluate_a4_core(tuple(wrong_y), z, a4_neural_profile) == 0

        invalid_z = list(z)
        invalid_z[0] = 2
        assert _evaluate_a4_core(valid_y, tuple(invalid_z), a4_neural_profile) == 0


def test_a4_c1_handles_signed_int8_endpoints(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
) -> None:
    """signed-int8 两端必须是规范输入但因 norm violation 被拒绝。"""
    y = tuple(0 for _ in range(8))
    for endpoint in (-128, 127):
        z = (endpoint,) + (0,) * (A4_VECTOR_SIZE - 1)
        assert _evaluate_a4_core(y, z, a4_neural_profile) == 0


def test_a4_c1_trace_is_integer_and_binary(
    a4_profile: A4PublicProfile,
    a4_neural_profile: A4NeuralProfile,
) -> None:
    """三层 trace 的输出必须是固定整数且最终只有 0/1。"""
    z = (0,) * A4_VECTOR_SIZE
    y = tuple(0 for _ in range(8))
    trace = _run_a4_graph(y, z, a4_neural_profile)

    assert all(type(value) is int for layer in trace for value in layer)
    assert trace[-1] in ((0,), (1,))
    assert max(trace[0]) <= 2_359_369
    assert max(trace[1]) <= 9_144


def test_a4_c1_rejects_noncanonical_core_values(a4_neural_profile: A4NeuralProfile) -> None:
    """core 的内部 canonical contract 拒绝错误 shape、类型和范围。"""
    with pytest.raises(A4NeuralEvaluationError):
        _evaluate_a4_core((0,) * 7, (0,) * A4_VECTOR_SIZE, a4_neural_profile)
    with pytest.raises(A4NeuralEvaluationError):
        _evaluate_a4_core((0,) * 8, (0,) * 71, a4_neural_profile)
    with pytest.raises(A4NeuralEvaluationError):
        _evaluate_a4_core((0,) * 7 + (True,), (0,) * A4_VECTOR_SIZE, a4_neural_profile)
    with pytest.raises(A4NeuralEvaluationError):
        _evaluate_a4_core((0,) * 8, (128,) + (0,) * 71, a4_neural_profile)


def test_a4_c1_compilation_does_not_change_public_profile(a4_profile: A4PublicProfile) -> None:
    """graph 编译只能复制可信 profile, 不能改写其公开矩阵。"""
    before = a4_profile.matrix
    compiled = compile_a4_neural_profile(a4_profile)

    assert compiled.public_profile is a4_profile
    assert a4_profile.matrix == before

    with pytest.raises(FrozenInstanceError):
        compiled.__setattr__("layers", ())
