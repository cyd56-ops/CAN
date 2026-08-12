"""A1-B1 PyTorch backend、dependency-free graph 与 A0 oracle 的全域差分测试。"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

import pytest
import torch

from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_MODULUS,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A0EvidenceCode,
    A0Registry,
    A0Slot,
    mod_q,
    parse_credential,
    verify_ref,
)
from can.verifier import (
    A1CompiledProfile,
    A1CompiledRegistry,
    A1EvidenceCode,
    compile_a1_profile,
    verify_a1,
)
from can.verifier.a1 import _run_graph
from can.verifier.a1_torch import (
    A1TorchBackend,
    _evaluate_torch_with_trace,
    compile_a1_torch_backend,
    verify_a1_torch,
)

TEST_SLOT_ID = 0xB1D1
TEST_SEED = 20260724


def _encode(
    coefficients: Sequence[int],
    *,
    profile_id: int = A0_PROFILE_ID,
    slot_id: int = TEST_SLOT_ID,
) -> bytes:
    return (
        bytes([A0_VERSION])
        + profile_id.to_bytes(2, byteorder="big", signed=False)
        + slot_id.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in coefficients)
    )


@pytest.fixture(scope="module")
def differential_fixture() -> tuple[
    A0Registry,
    A1CompiledRegistry,
    A1TorchBackend,
    A1CompiledProfile,
    tuple[int, ...],
]:
    """构建固定种子的 A0、A1 和 A1-B1 共同 profile。"""
    random = Random(TEST_SEED)
    secret = tuple(random.randrange(2) for _ in range(A0_SECRET_SIZE))
    rows: list[list[int]] = []
    for row_index in range(A0_COMPONENT_COUNT):
        row = [random.randrange(A0_MODULUS) for _ in range(A0_SECRET_SIZE)]
        if all(value == 0 for value in row):
            row[0] = row_index + 1
        rows.append(row)
    slot = A0Slot(TEST_SLOT_ID, rows)
    profile = compile_a1_profile(slot, secret)
    registry = A1CompiledRegistry([profile])
    return A0Registry([slot]), registry, compile_a1_torch_backend(registry), profile, secret


def _credential_for_offsets(profile: A1CompiledProfile, offsets: Sequence[int]) -> bytes:
    return _encode(
        [
            mod_q(profile.anchors[index] + A0_CENTER + offsets[index])
            for index in range(A0_COMPONENT_COUNT)
        ]
    )


def test_every_slot_component_and_coefficient_matches_all_backends(
    differential_fixture: tuple[
        A0Registry,
        A1CompiledRegistry,
        A1TorchBackend,
        A1CompiledProfile,
        tuple[int, ...],
    ],
) -> None:
    """八个位置的全部 b_i=0..256 应逐层匹配且保持单向 soundness。"""
    reference_registry, compiled_registry, backend, profile, secret = differential_fixture
    base = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    false_accepts = 0
    issuer_false_rejects = 0

    for component in range(A0_COMPONENT_COUNT):
        for coefficient in range(A0_MODULUS):
            coefficients = base.copy()
            coefficients[component] = coefficient
            raw = _encode(coefficients)
            canonical = parse_credential(raw).b

            reference = verify_ref(raw, reference_registry, secret)
            dependency_free = verify_a1(raw, compiled_registry)
            torch_evidence = verify_a1_torch(raw, backend)
            torch_trace = _evaluate_torch_with_trace(
                canonical, backend, A0_PROFILE_ID, TEST_SLOT_ID
            )

            assert tuple(tuple(layer.tolist()) for layer in torch_trace.layers) == _run_graph(
                canonical, profile
            )
            assert torch_trace.layers[-1].dtype is torch.int32
            assert torch_trace.layers[-1].item() in (0, 1)
            assert torch_evidence == dependency_free
            assert not torch_evidence.accepted or reference.accepted
            false_accepts += int(torch_evidence.accepted and not reference.accepted)
            issuer_false_rejects += int(
                reference.code is A0EvidenceCode.ISSUER_CORE and not torch_evidence.accepted
            )

    assert false_accepts == 0
    assert issuer_false_rejects == 0


@pytest.mark.parametrize(
    ("offsets", "torch_code", "reference_code"),
    [
        ([0] * 8, A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.ISSUER_CORE),
        ([-4, 4, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.ISSUER_CORE),
        ([5, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.REFERENCE_GUARD),
        ([8, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_ACCEPT, A0EvidenceCode.REFERENCE_GUARD),
        ([9, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REFERENCE_GUARD),
        ([12, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REFERENCE_GUARD),
        ([13, 0, 0, 0, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REJECT),
        ([0, 0, 0, 13, 0, 0, 0, 0], A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REJECT),
        ([128] * 8, A1EvidenceCode.NUMERIC_REJECT, A0EvidenceCode.REJECT),
    ],
)
def test_a0_vector_families_preserve_one_sided_soundness(
    differential_fixture: tuple[
        A0Registry,
        A1CompiledRegistry,
        A1TorchBackend,
        A1CompiledProfile,
        tuple[int, ...],
    ],
    offsets: list[int],
    torch_code: A1EvidenceCode,
    reference_code: A0EvidenceCode,
) -> None:
    """core、guard、reject 和 mixed 向量族应与既有两个事实源一致。"""
    reference_registry, compiled_registry, backend, profile, secret = differential_fixture
    raw = _credential_for_offsets(profile, offsets)

    reference = verify_ref(raw, reference_registry, secret)
    dependency_free = verify_a1(raw, compiled_registry)
    torch_evidence = verify_a1_torch(raw, backend)

    assert reference.code is reference_code
    assert torch_evidence.code is torch_code
    assert torch_evidence == dependency_free
    assert not torch_evidence.accepted or reference.accepted


def test_modular_wrap_and_bit_zero_vectors_match(
    differential_fixture: tuple[
        A0Registry,
        A1CompiledRegistry,
        A1TorchBackend,
        A1CompiledProfile,
        tuple[int, ...],
    ],
) -> None:
    """wrap 与显式 b_i=0 向量应保持逐层一致且不触发路线变化。"""
    reference_registry, compiled_registry, backend, profile, secret = differential_fixture
    wrap = _credential_for_offsets(profile, [4, -4, 0, 1, -1, 2, -2, 3])
    bit_zero_coefficients = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    bit_zero_coefficients[3] = 0
    bit_zero = _encode(bit_zero_coefficients)

    for raw in (wrap, bit_zero):
        canonical = parse_credential(raw).b
        reference = verify_ref(raw, reference_registry, secret)
        dependency_free = verify_a1(raw, compiled_registry)
        torch_evidence = verify_a1_torch(raw, backend)
        trace = _evaluate_torch_with_trace(canonical, backend, A0_PROFILE_ID, TEST_SLOT_ID)

        assert tuple(tuple(layer.tolist()) for layer in trace.layers) == _run_graph(
            canonical, profile
        )
        assert torch_evidence == dependency_free
        assert not torch_evidence.accepted or reference.accepted


def test_malformed_and_unknown_profiles_fail_before_torch_graph(
    differential_fixture: tuple[
        A0Registry,
        A1CompiledRegistry,
        A1TorchBackend,
        A1CompiledProfile,
        tuple[int, ...],
    ],
) -> None:
    """错误编码和未知 profile 应在 Torch graph 前稳定拒绝。"""
    _, _, backend, profile, _ = differential_fixture
    core = _credential_for_offsets(profile, [0] * A0_COMPONENT_COUNT)

    malformed = verify_a1_torch(core + b"\x00", backend)
    unknown = verify_a1_torch(_encode([0] * A0_COMPONENT_COUNT, profile_id=2), backend)

    assert malformed.code is A1EvidenceCode.PARSE_REJECT
    assert unknown.code is A1EvidenceCode.PROFILE_REJECT
    assert backend.active
