"""A4 GPV-PFDH toy 公钥关系的防御性安全边界测试。"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from can.access import A3Evidence, A4ReferenceAdapter
from can.reference import (
    A4_PROOF_SIZE,
    A4_SYNDROME_SIZE,
    A4_VECTOR_SIZE,
    A4EvidenceCode,
    A4ProfileValidationError,
    A4Proof,
    A4PublicProfile,
    A4ReferenceEvidence,
    verify_a4_ref,
)
from conftest import A4ProofFactory


def test_public_profile_adapter_and_evidence_have_no_private_or_authority_fields(
    a4_profile: A4PublicProfile,
) -> None:
    """A4 运行时对象只能持有公开配置与无授权能力证据。"""
    profile_fields = {item.name for item in fields(A4PublicProfile)}
    adapter_fields = {item.name for item in fields(A4ReferenceAdapter)}
    reference_fields = {item.name for item in fields(A4ReferenceEvidence)}
    a3_fields = {item.name for item in fields(A3Evidence)}
    forbidden = {
        "private_key",
        "secret_key",
        "trapdoor",
        "signer",
        "decision",
        "gate",
        "capability",
        "authorization",
    }

    assert profile_fields == {"profile_id", "identity_id", "matrix", "public_key_sha256"}
    assert adapter_fields == {"profile"}
    assert reference_fields == {"code"}
    assert a3_fields == {"code", "identity_id", "message_sha256", "profile_id"}
    assert not forbidden & (profile_fields | adapter_fields | reference_fields | a3_fields)
    assert len(a4_profile.public_key_sha256) == 32


def test_client_key_profile_and_decision_bytes_cannot_extend_the_proof(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """proof 前后附加矩阵、profile 或 decision 只能得到解析拒绝。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    injected = b"profile=weak;decision=allow;A=" + bytes([1]) * 32

    prefixed = verify_a4_ref(a4_message, injected + proof, a4_profile)
    appended = verify_a4_ref(a4_message, proof + injected, a4_profile)

    assert prefixed.code is A4EvidenceCode.PROOF_PARSE_REJECT
    assert appended.code is A4EvidenceCode.PROOF_PARSE_REJECT
    assert not prefixed.accepted and not appended.accepted


@pytest.mark.parametrize("coefficient", [-128, -2, 2, 127])
def test_all_signed_int8_values_outside_the_norm_bound_reject(
    coefficient: int,
    a4_profile: A4PublicProfile,
    a4_message: bytes,
) -> None:
    """signed-int8 可表示但超出短向量域的输入不得被约减或裁剪。"""
    vector = [0] * A4_VECTOR_SIZE
    vector[0] = coefficient
    proof = A4Proof(1, bytes(32), vector).encode()

    evidence = verify_a4_ref(a4_message, proof, a4_profile)

    assert evidence.code is A4EvidenceCode.NORM_REJECT
    assert not evidence.accepted


@pytest.mark.parametrize("raw_proof", [True, bytearray(A4_PROOF_SIZE), "proof", None])
def test_proof_type_confusion_never_reaches_relation_accept(
    raw_proof: object,
    a4_profile: A4PublicProfile,
    a4_message: bytes,
) -> None:
    """bool、mutable bytes、text 和 null 都必须 fail closed。"""
    evidence = verify_a4_ref(a4_message, raw_proof, a4_profile)

    assert evidence.code is A4EvidenceCode.PROOF_PARSE_REJECT
    assert not evidence.accepted


def test_rank_deficient_public_configuration_cannot_activate() -> None:
    """全零或线性相关矩阵不能进入本地可信 profile。"""
    zero_matrix = [[0] * A4_VECTOR_SIZE for _ in range(A4_SYNDROME_SIZE)]
    repeated_rows = [[1] * A4_VECTOR_SIZE for _ in range(A4_SYNDROME_SIZE)]

    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(1, bytes(32), zero_matrix)
    with pytest.raises(A4ProfileValidationError):
        A4PublicProfile(1, bytes(32), repeated_rows)


def test_message_salt_and_vector_tamper_each_reject_without_fallback(
    a4_profile: A4PublicProfile,
    a4_message: bytes,
    a4_proof_factory: A4ProofFactory,
) -> None:
    """三类外部可控数据的单字节篡改都不能保留 relation accept。"""
    proof = a4_proof_factory(a4_message, a4_profile, bytes(range(32, 64)))
    message_tamper = bytearray(a4_message)
    message_tamper[-1] ^= 1
    salt_tamper = bytearray(proof)
    salt_tamper[1] ^= 1
    vector_tamper = bytearray(proof)
    vector_tamper[-1] = 1

    results = (
        verify_a4_ref(bytes(message_tamper), proof, a4_profile),
        verify_a4_ref(a4_message, bytes(salt_tamper), a4_profile),
        verify_a4_ref(a4_message, bytes(vector_tamper), a4_profile),
    )

    assert all(not evidence.accepted for evidence in results)
    assert all(evidence.code is A4EvidenceCode.EQUATION_REJECT for evidence in results)


def test_reference_module_has_no_signer_secret_fallback_or_model_dependency() -> None:
    """A4 reference 源码不得引入签名器、私钥、A0/A1 fallback 或模型调用。"""
    source_path = Path("src/can/reference/a4.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    public_functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    assert not any(name.startswith(("can.model", "can.verifier")) for name in imported_modules)
    assert not {"sign", "keygen", "sample_preimage", "verify_ref"} & public_functions
    assert "can.reference.a0" not in imported_modules
