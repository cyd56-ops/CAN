"""A4-C1 neural verifier 的防御性边界测试。"""

from __future__ import annotations

import inspect

import pytest

import can.verifier.a4 as a4_neural_module
from can.access import A3EvidenceCode, A4NeuralAdapter
from can.reference.a4 import A4Proof, A4PublicProfile
from can.verifier.a4 import (
    A4NeuralEvidenceCode,
    compile_a4_neural_profile,
    verify_a4_neural,
)
from conftest import A4ProofFactory


def test_a4_neural_module_has_no_reference_or_authority_fallback() -> None:
    """A4-C1 module 不能导入 reference verifier、模型或授权 API。"""
    source = inspect.getsource(a4_neural_module)
    assert "verify_a4_ref" not in source
    assert "from can.access" not in source
    assert "from can.model" not in source
    assert "verify_a4_ref" not in a4_neural_module.__dict__


def test_a4_neural_profile_is_not_client_selectable(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """错误 profile、key-like proof 前缀和 decision 注入都必须拒绝。"""
    compiled = compile_a4_neural_profile(a4_profile)
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))

    assert verify_a4_neural(a4_message, proof, object()).code is A4NeuralEvidenceCode.CONFIG_REJECT
    assert (
        verify_a4_neural(a4_message, b"key=" + proof, compiled).code
        is A4NeuralEvidenceCode.INPUT_REJECT
    )
    assert (
        verify_a4_neural(a4_message, proof + b"decision=allow", compiled).code
        is A4NeuralEvidenceCode.INPUT_REJECT
    )


def test_a4_neural_adapter_produces_only_a3_evidence(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """neural adapter 只能映射 A3 evidence, 不能返回授权能力。"""
    compiled = compile_a4_neural_profile(a4_profile)
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    adapter = A4NeuralAdapter(compiled)

    evidence = adapter(a4_message, proof)
    assert evidence.code is A3EvidenceCode.PROOF_ACCEPT
    assert not hasattr(evidence, "decision")
    assert not hasattr(evidence, "capability")
    assert not hasattr(evidence, "model")


def test_a4_neural_rejects_norm_invalid_proof_without_reference_call(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """norm invalid proof 必须在 neural graph 内拒绝且不 fallback 到 reference。"""
    compiled = compile_a4_neural_profile(a4_profile)
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    parsed = A4Proof(
        1, proof[1:33], tuple(value - 256 if value >= 128 else value for value in proof[33:])
    )
    vector = list(parsed.vector)
    vector[0] = -2
    invalid = A4Proof(1, parsed.salt, vector).encode()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("reference verifier fallback is forbidden")

    monkeypatch.setattr("can.reference.a4.verify_a4_ref", forbidden)
    result = verify_a4_neural(a4_message, invalid, compiled)

    assert result.code is A4NeuralEvidenceCode.NEURAL_REJECT
