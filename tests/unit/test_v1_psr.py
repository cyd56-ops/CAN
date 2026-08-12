"""V1-P2-PSR-E1 generated-key 与确定性采样器单元测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from can.experiments.v1_psr import (
    V1_PSR_CHALLENGE_DOMAIN,
    V1_PSR_CHALLENGE_SET,
    V1_PSR_MASK_DOMAIN,
    V1_PSR_SECRET_DOMAIN,
    V1_PSR_THEORETICAL_EMIT_PROBABILITY,
    V1GeneratedKeyFixture,
    V1PSRCoordinatorOutcome,
    V1PSRInputError,
    V1PSRLifecycleError,
    V1PSRManifestError,
    V1PSROutcome,
    V1PSRRetryRecord,
    V1PSRRetryReport,
    build_v1_generated_key_fixture,
    build_v1_vector_manifest,
    compute_v1_commitment,
    compute_v1_response,
    sample_v1_challenge,
    sample_v1_mask,
    sample_v1_secret,
    v1_response_emits,
    write_v1_vector_manifest,
)
from can.reference import (
    V1_CONFORMANCE_MATRIX,
    V1_CONFORMANCE_TARGET,
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RESPONSE_BOUND,
    V1_RING_DEGREE,
    V1Challenge,
    v1_negacyclic_convolution,
)

SEED = bytes(range(32))
IDENTITY = hashlib.sha256(b"CAN V1 PSR unit identity").digest()

EXPECTED_SECRET = (
    (-1, -1, -1, 1, 1, -1, 0, -1),
    (-1, 1, 1, -1, -1, -1, 1, 0),
    (0, 0, -1, 0, 0, -1, -1, 0),
    (0, -1, 1, 0, 1, 0, -1, 0),
)
EXPECTED_MASK = (
    (-6, -1, 3, -4, 2, -6, 0, 8),
    (-4, -7, -6, 2, -5, 7, 7, -6),
    (-5, 7, -4, 3, 6, 2, -7, -2),
    (-7, 4, 6, 0, -6, 5, 0, 4),
)


def _flatten(polynomials: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(coefficient for polynomial in polynomials for coefficient in polynomial)


def _manual_module_action(
    vector: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    output: list[tuple[int, ...]] = []
    for row_index in range(2):
        coefficients = [0] * V1_RING_DEGREE
        for column_index in range(2):
            product = v1_negacyclic_convolution(
                V1_CONFORMANCE_MATRIX[row_index][column_index], vector[column_index]
            )
            coefficients = [
                value + product_value
                for value, product_value in zip(coefficients, product, strict=True)
            ]
        coefficients = [
            value + identity_value
            for value, identity_value in zip(coefficients, vector[2 + row_index], strict=True)
        ]
        output.append(tuple(value % V1_MODULUS for value in coefficients))
    return tuple(output)


def test_fixed_shake256_vectors_are_polynomial_major_and_deterministic() -> None:
    """固定 seed 必须生成唯一且按多项式优先排列的跨实现向量。"""
    assert sample_v1_secret(SEED) == EXPECTED_SECRET
    assert sample_v1_mask(SEED, 0, 0) == EXPECTED_MASK
    assert sample_v1_challenge(SEED, 0, 0).coefficients == (0, 1, 0, 0, -1, 0, 0, 0)
    assert sample_v1_secret(SEED) == sample_v1_secret(SEED)
    assert sample_v1_mask(SEED, 0, 0) == sample_v1_mask(SEED, 0, 0)


def test_secret_sampler_discards_255_instead_of_reducing_it() -> None:
    """secret byte rejection 必须跳过 255, 不能直接对所有 byte 取模。"""
    seed = (5).to_bytes(32, byteorder="big", signed=False)
    payload = V1_PSR_SECRET_DOMAIN + seed + bytes(8) + bytes(8) + bytes(4)
    block = hashlib.shake_256(payload).digest(64)
    assert block[22] == 255
    expected = tuple((value % 3) - 1 for value in block if value < 255)[:32]
    direct_modulo = tuple((value % 3) - 1 for value in block[:32])

    assert _flatten(sample_v1_secret(seed)) == expected
    assert expected != direct_modulo


def test_mask_sampler_discards_255_instead_of_reducing_it() -> None:
    """mask byte rejection 必须跳过 255, 不能改变后续坐标位置。"""
    seed = (2).to_bytes(32, byteorder="big", signed=False)
    payload = V1_PSR_MASK_DOMAIN + seed + bytes(8) + bytes(8) + bytes(4)
    block = hashlib.shake_256(payload).digest(64)
    assert block[31] == 255
    expected = tuple((value % 17) - 8 for value in block if value < 255)[:32]
    direct_modulo = tuple((value % 17) - 8 for value in block[:32])

    assert _flatten(sample_v1_mask(seed, 0, 0)) == expected
    assert expected != direct_modulo


def test_challenge_sampler_discards_bytes_at_or_above_224() -> None:
    """challenge byte rejection 必须跳过偏置尾部再选择 112 元素集合。"""
    seed = (6).to_bytes(32, byteorder="big", signed=False)
    payload = V1_PSR_CHALLENGE_DOMAIN + seed + bytes(8) + bytes(8) + bytes(4)
    block = hashlib.shake_256(payload).digest(64)
    assert block[:8] == bytes((255, 227, 158, 255, 131, 93, 9, 92))
    first_accepted = next(value for value in block if value < 224)

    assert sample_v1_challenge(seed, 0, 0) == V1_PSR_CHALLENGE_SET[first_accepted % 112]


@pytest.mark.parametrize("bad_seed", [None, True, b"", bytes(31), bytes(33), bytearray(32)])
def test_sampler_rejects_noncanonical_seed(bad_seed: object) -> None:
    """外部 seed 类型与长度混淆必须在 SHAKE 派生前拒绝。"""
    with pytest.raises(V1PSRInputError):
        sample_v1_secret(bad_seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_counter", [True, -1, 1 << 64, 1.0, "0"])
def test_sampler_rejects_noncanonical_counters(bad_counter: object) -> None:
    """trial/retry counter 只接受规范 unsigned-u64 exact int。"""
    with pytest.raises(V1PSRInputError):
        sample_v1_mask(SEED, bad_counter, 0)  # type: ignore[arg-type]
    with pytest.raises(V1PSRInputError):
        sample_v1_challenge(SEED, 0, bad_counter)  # type: ignore[arg-type]


def test_role_and_retry_streams_are_separated() -> None:
    """不同 role 或 retry tuple 不得复用同一采样结果。"""
    assert sample_v1_secret(SEED) != sample_v1_mask(SEED, 0, 0)
    assert sample_v1_mask(SEED, 0, 0) != sample_v1_mask(SEED, 0, 1)
    assert sample_v1_mask(SEED, 0, 0) != sample_v1_mask(SEED, 1, 0)
    assert sample_v1_challenge(SEED, 0, 0) != sample_v1_challenge(SEED, 0, 1)


def test_rejection_domains_map_to_uniform_finite_supports() -> None:
    """三个 accepted byte 域必须对目标集合产生完全相同的原像计数。"""
    secret_counts = tuple(sum(value % 3 == index for value in range(255)) for index in range(3))
    mask_counts = tuple(sum(value % 17 == index for value in range(255)) for index in range(17))
    challenge_counts = tuple(
        sum(value % 112 == index for value in range(224)) for index in range(112)
    )

    assert secret_counts == (85,) * 3
    assert mask_counts == (15,) * 17
    assert challenge_counts == (2,) * 112


def test_theoretical_emit_probability_matches_finite_cube_count() -> None:
    """理论 emit probability 必须来自 32 个独立 13/17 坐标计数。"""
    assert V1_PSR_THEORETICAL_EMIT_PROBABILITY == (13 / 17) ** 32
    assert V1_PSR_THEORETICAL_EMIT_PROBABILITY == pytest.approx(
        0.00018699146739962278,
        abs=1e-18,
    )


def test_challenge_set_has_the_exact_lexicographic_112_elements() -> None:
    """固定 challenge 集必须完整、唯一、等权且顺序不漂移。"""
    coefficient_vectors = tuple(challenge.coefficients for challenge in V1_PSR_CHALLENGE_SET)

    assert len(coefficient_vectors) == 112
    assert len(set(coefficient_vectors)) == 112
    assert coefficient_vectors[:4] == (
        (-1, -1, 0, 0, 0, 0, 0, 0),
        (-1, 1, 0, 0, 0, 0, 0, 0),
        (1, -1, 0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0, 0, 0),
    )
    assert coefficient_vectors[-1] == (0, 0, 0, 0, 0, 0, 1, 1)
    assert all(
        sum(coefficient != 0 for coefficient in coefficients) == 2
        and set(coefficients) <= {-1, 0, 1}
        for coefficients in coefficient_vectors
    )


def test_generated_key_target_is_derived_from_secret_and_not_fixed_fixture() -> None:
    """临时 profile 的公开 target 必须精确等于 Abar*s 且不复用固定 target。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        expected_target = _manual_module_action(EXPECTED_SECRET)

        assert fixture.profile.target == expected_target
        assert fixture.profile.target != V1_CONFORMANCE_TARGET
        assert fixture.profile.profile_id == V1_PROFILE_ID
        assert len(fixture.profile.public_key_sha256) == 32
        assert len(fixture.target_sha256) == 32


@pytest.mark.parametrize(
    "secret",
    [
        ((0,) * 8,) * 4,
        ((-1,) * 8,) * 4,
        ((1,) * 8,) * 4,
        ((-1, 0, 1, -1, 0, 1, -1, 1),) * 4,
    ],
)
def test_secret_domain_boundary_fixtures_derive_public_targets(
    secret: tuple[tuple[int, ...], ...],
) -> None:
    """全零、全正负边界和 mixed secret 都必须保持固定 shape 与 exact target。"""
    mask = ((0,) * 8,) * 4
    challenge = V1Challenge(V1_PROFILE_ID, (-1, 1, 0, 0, 0, 0, 0, 0))
    response = compute_v1_response(secret, mask, challenge)

    assert len(secret) == 4
    assert all(len(polynomial) == 8 for polynomial in secret)
    assert all(-1 <= coefficient <= 1 for polynomial in secret for coefficient in polynomial)
    assert response == tuple(
        v1_negacyclic_convolution(challenge.coefficients, polynomial) for polynomial in secret
    )


@pytest.mark.parametrize(
    "mask",
    [
        ((-8,) * 8,) * 4,
        ((8,) * 8,) * 4,
        ((0,) * 8,) * 4,
        ((-8, 8, 0, -8, 8, 0, -8, 8),) * 4,
    ],
)
def test_mask_domain_boundary_fixtures_preserve_inclusive_range(
    mask: tuple[tuple[int, ...], ...],
) -> None:
    """全负、全正、全零和 mixed mask 必须覆盖闭区间端点而不约减。"""
    response = compute_v1_response(((0,) * 8,) * 4, mask, V1_PSR_CHALLENGE_SET[0])
    assert response == mask
    assert all(-8 <= coefficient <= 8 for polynomial in response for coefficient in polynomial)


def test_commitment_reuses_exact_coefficient_domain_module_action() -> None:
    """commitment 必须是规范 residue ``Abar*y``, 包括负循环 wraparound。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        commitment = compute_v1_commitment(fixture.profile, EXPECTED_MASK)

    assert commitment.polynomials == _manual_module_action(EXPECTED_MASK)
    assert all(
        0 <= coefficient < V1_MODULUS
        for polynomial in commitment.polynomials
        for coefficient in polynomial
    )


def test_response_is_unreduced_integer_negacyclic_sum() -> None:
    """response 只能计算 ``y+c*s``, 不能做模 q、截断或饱和。"""
    secret = (
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0,) * 8,
        (0,) * 8,
        (0,) * 8,
    )
    mask = ((-8,) + (0,) * 7, (0,) * 8, (0,) * 8, (0,) * 8)
    challenge = V1Challenge(V1_PROFILE_ID, (0, 1, 0, 0, 0, 0, 0, 1))
    expected_shift = v1_negacyclic_convolution(challenge.coefficients, secret[0])

    response = compute_v1_response(secret, mask, challenge)

    assert expected_shift == (-1, 0, 0, 0, 0, 0, -1, 0)
    assert response[0] == (-9, 0, 0, 0, 0, 0, -1, 0)
    assert response[0][0] != (-9) % V1_MODULUS


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [(V1_RESPONSE_BOUND - 1, True), (V1_RESPONSE_BOUND, True), (V1_RESPONSE_BOUND + 1, False)],
)
def test_emit_rule_has_exact_5_6_7_boundary(boundary: int, expected: bool) -> None:
    """B=6 应 emit, B+1 应 abort, 且 mixed vector 不得被 clipping。"""
    response = ((-boundary, boundary, 0, 0, 0, 0, 0, 0),) + ((0,) * 8,) * 3
    assert v1_response_emits(response) is expected


def test_fixed_secret_challenge_translation_has_13_to_13_count() -> None:
    """任意固定 toy s,c 的每个 shift 都应把 13 个 mask 值双射到 [-6,6]。"""
    challenge = V1Challenge(V1_PROFILE_ID, (1, 0, 0, -1, 0, 0, 0, 0))
    for secret_polynomial in EXPECTED_SECRET:
        shift = v1_negacyclic_convolution(challenge.coefficients, secret_polynomial)
        assert all(-2 <= coefficient <= 2 for coefficient in shift)
        for coefficient in shift:
            emitted = tuple(
                mask_value + coefficient
                for mask_value in range(-8, 9)
                if abs(mask_value + coefficient) <= V1_RESPONSE_BOUND
            )
            assert emitted == tuple(range(-V1_RESPONSE_BOUND, V1_RESPONSE_BOUND + 1))


def test_attempt_is_deterministic_and_abort_has_no_response() -> None:
    """同一 counter tuple 应复现, abort 不得产生 response wire object。"""
    transcript_id = hashlib.sha256(b"abort transcript").digest()
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        first = fixture.prove_attempt(0, 0, transcript_id)
        second = fixture.prove_attempt(0, 0, transcript_id)
        fresh = fixture.prove_attempt(0, 1, transcript_id)

    assert first == second
    assert first.outcome is V1PSROutcome.ABORT
    assert first.response is None
    assert (fresh.commitment, fresh.challenge) != (first.commitment, first.challenge)


def test_retry_report_requires_contiguous_attempts_and_matching_terminal_state() -> None:
    """retry report 只接受连续 attempt index 与一致的 exhaustion 状态。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        first = fixture.prove_attempt(0, 0, hashlib.sha256(b"retry-0").digest())
    retry_record = V1PSRRetryRecord(
        retry_index=0,
        prover_outcome=first.outcome,
        coordinator_outcome=V1PSRCoordinatorOutcome.ABORTED,
        nonce_sha256=bytes(32),
        transcript_sha256=bytes([1]) * 32,
        commitment_sha256=bytes([2]) * 32,
        challenge_sha256=bytes([3]) * 32,
        sampler_ns=1,
        response_ns=1,
        exact_ns=0,
        a3_ns=1,
        total_ns=1,
        replay_denies=1,
        concurrent_denies=2,
    )
    report = V1PSRRetryReport(0, 1, "exhausted", (retry_record,), True, 0, 0)
    assert report.attempt_count == 1
    assert report.public_record()["protected_calls"] == 0
    with pytest.raises(V1PSRInputError):
        V1PSRRetryReport(0, 1, "exhausted", (retry_record,), False, 0, 0)


def test_attempt_computes_commitment_before_sampling_server_challenge() -> None:
    """prover attempt 必须先固定 commitment, 再请求可信 harness challenge。"""
    call_order: list[str] = []
    original_commitment = compute_v1_commitment
    original_challenge = sample_v1_challenge

    def commitment_wrapper(profile: object, mask: object) -> object:
        call_order.append("commitment")
        return original_commitment(profile, mask)  # type: ignore[arg-type]

    def challenge_wrapper(seed: bytes, trial_index: int, retry_index: int) -> V1Challenge:
        call_order.append("challenge")
        return original_challenge(seed, trial_index, retry_index)

    with (
        build_v1_generated_key_fixture(IDENTITY, SEED) as fixture,
        patch("can.experiments.v1_psr.compute_v1_commitment", side_effect=commitment_wrapper),
        patch("can.experiments.v1_psr.sample_v1_challenge", side_effect=challenge_wrapper),
    ):
        fixture.prove_attempt(0, 0, bytes(32))

    assert call_order == ["commitment", "challenge"]


def test_context_manager_closes_fixture_on_success_and_exception() -> None:
    """fixture 在正常和异常退出后都必须禁止再次生成 attempt。"""
    fixture = V1GeneratedKeyFixture(IDENTITY, SEED)
    with fixture:
        assert not fixture.closed
    assert fixture.closed
    with pytest.raises(V1PSRLifecycleError):
        fixture.prove_attempt(0, 0, bytes(32))

    failing = V1GeneratedKeyFixture(IDENTITY, SEED)
    with pytest.raises(RuntimeError, match="synthetic failure"), failing:
        raise RuntimeError("synthetic failure")
    assert failing.closed


def test_public_vector_manifest_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    """manifest 只写公开摘要、结果和计数, 并拒绝覆盖既有文件。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        attempts = [
            fixture.prove_attempt(0, retry, hashlib.sha256(retry.to_bytes(8, "big")).digest())
            for retry in range(2)
        ]
    first = build_v1_vector_manifest(attempts)
    second = build_v1_vector_manifest(tuple(attempts))
    decoded = json.loads(first)

    assert first == second
    assert decoded["vector_count"] == 2
    assert decoded["abort_count"] == 2
    assert decoded["emit_count"] == 0
    assert set(decoded["vectors"][0]) == {
        "challenge_sha256",
        "commitment_sha256",
        "outcome",
        "profile_id",
        "profile_sha256",
        "protocol_id",
        "retry_index",
        "seed_sha256",
        "target_sha256",
        "trial_index",
    }
    path = tmp_path / "vectors.json"
    assert write_v1_vector_manifest(path, attempts) == path
    assert path.read_bytes() == first
    with pytest.raises(V1PSRManifestError):
        write_v1_vector_manifest(path, attempts)
