"""跨测试类别复用的确定性 A2-E1 门控夹具。"""

from collections.abc import Callable, Sequence

import pytest

from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A4_MESSAGE_DOMAIN,
    A4_PROFILE_ID,
    A4_SYNDROME_SIZE,
    A4_VECTOR_SIZE,
    A0Slot,
    A4Proof,
    A4PublicProfile,
    hash_to_a4_syndrome,
    mod_q,
)
from can.verifier import (
    A1CompiledProfile,
    A1CompiledRegistry,
    A4NeuralProfile,
    compile_a1_profile,
    compile_a4_neural_profile,
)
from can.verifier.a1_torch import A1TorchBackend, compile_a1_torch_backend

A2_GATE_TEST_SLOT_ID = 0xA2E1
A4_TEST_IDENTITY = bytes(range(32))
A4_TEST_SALT = bytes(range(32, 64))
A4ProofFactory = Callable[[bytes, A4PublicProfile, bytes], bytes]


def _a4_gadget_matrix() -> list[list[int]]:
    matrix = [[0] * A4_VECTOR_SIZE for _ in range(A4_SYNDROME_SIZE)]
    for row_index in range(A4_SYNDROME_SIZE):
        for bit_index in range(9):
            matrix[row_index][row_index * 9 + bit_index] = 1 << bit_index
    return matrix


def _encode_a4_message(identity_id: bytes) -> bytes:
    issued_at_ms = 1_700_000_000_000
    return (
        A4_MESSAGE_DOMAIN
        + bytes([1])
        + (1).to_bytes(4, byteorder="big", signed=False)
        + identity_id
        + (1).to_bytes(2, byteorder="big", signed=False)
        + issued_at_ms.to_bytes(8, byteorder="big", signed=False)
        + (issued_at_ms + 60_000).to_bytes(8, byteorder="big", signed=False)
        + b"\x01" * 32
        + b"\x02" * 32
    )


def _build_a4_test_proof(
    message: bytes,
    profile: A4PublicProfile,
    salt: bytes,
) -> bytes:
    target = hash_to_a4_syndrome(message, salt, profile)
    vector = [0] * A4_VECTOR_SIZE
    for row_index, coefficient in enumerate(target):
        for bit_index in range(9):
            vector[row_index * 9 + bit_index] = (coefficient >> bit_index) & 1
    return A4Proof(1, salt, vector).encode()


@pytest.fixture
def a4_profile() -> A4PublicProfile:
    """返回只含公开 gadget 矩阵的弱 A4 conformance profile。"""
    return A4PublicProfile(A4_PROFILE_ID, A4_TEST_IDENTITY, _a4_gadget_matrix())


@pytest.fixture
def a4_message() -> bytes:
    """返回确定性的 canonical A3 message。"""
    return _encode_a4_message(A4_TEST_IDENTITY)


@pytest.fixture
def a4_proof_factory() -> A4ProofFactory:
    """返回不使用私钥、仅适配公开 gadget fixture 的 proof 构造器。"""
    return _build_a4_test_proof


@pytest.fixture
def a4_neural_profile(a4_profile: A4PublicProfile) -> A4NeuralProfile:
    """返回由弱公开 profile 编译的固定 A4-C1 graph。"""
    return compile_a4_neural_profile(a4_profile)


def encode_a2_gate_credential(
    coefficients: Sequence[int],
    *,
    profile_id: int = A0_PROFILE_ID,
    slot_id: int = A2_GATE_TEST_SLOT_ID,
) -> bytes:
    """编码测试专用 A0-v1 credential, 不用于生产凭据。"""
    return (
        bytes([A0_VERSION])
        + profile_id.to_bytes(2, byteorder="big", signed=False)
        + slot_id.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in coefficients)
    )


def build_a2_gate_backend() -> tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes]:
    """在内存中生成固定非生产 profile、backend 和正负 credential。"""
    slot = A0Slot(
        A2_GATE_TEST_SLOT_ID,
        [[row_index + 1] * A0_SECRET_SIZE for row_index in range(A0_COMPONENT_COUNT)],
    )
    profile = compile_a1_profile(slot, (1,) * A0_SECRET_SIZE)
    accepted_coefficients = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    rejected_coefficients = accepted_coefficients.copy()
    rejected_coefficients[0] = mod_q(rejected_coefficients[0] + 9)
    backend = compile_a1_torch_backend(A1CompiledRegistry([profile]))
    return (
        profile,
        backend,
        encode_a2_gate_credential(accepted_coefficients),
        encode_a2_gate_credential(rejected_coefficients),
    )


@pytest.fixture(scope="session")
def a2_gate_fixture() -> tuple[A1CompiledProfile, A1TorchBackend, bytes, bytes]:
    """跨测试文件只读复用一次完整启动认证后的 A1-B1 backend。"""
    return build_a2_gate_backend()
