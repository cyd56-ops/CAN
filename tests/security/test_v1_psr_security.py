"""V1-P2-PSR-E1 secret hygiene 与 fail-closed 实验边界测试。"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from can.access import A3V2TrustedInput
from can.experiments.v1_psr import (
    V1PSRInputError,
    V1PSRManifestError,
    build_v1_generated_key_fixture,
    build_v1_vector_manifest,
    compute_v1_commitment,
    compute_v1_response,
    sample_v1_mask,
    sample_v1_secret,
    v1_response_emits,
    write_v1_vector_manifest,
)

SEED = bytes(range(32))
IDENTITY = hashlib.sha256(b"CAN V1 PSR security identity").digest()


def test_public_profile_attempt_repr_and_manifest_do_not_serialize_private_values() -> None:
    """公开输出不得包含 seed、secret、mask、response 或 transcript 原文。"""
    transcript_id = hashlib.sha256(b"private transcript").digest()
    secret = sample_v1_secret(SEED)
    mask = sample_v1_mask(SEED, 0, 0)
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        attempt = fixture.prove_attempt(0, 0, transcript_id)
        profile_repr = repr(fixture.profile)
        attempt_repr = repr(attempt)
        manifest = build_v1_vector_manifest((attempt,))

        assert not hasattr(fixture.profile, "secret")
        assert not hasattr(fixture.profile, "seed")

    decoded = json.loads(manifest)
    record = decoded["vectors"][0]
    assert "secret" not in record
    assert "mask" not in record
    assert "response" not in record
    assert "transcript" not in record
    assert SEED.hex().encode("ascii") not in manifest
    assert transcript_id.hex().encode("ascii") not in manifest
    assert repr(secret) not in profile_repr
    assert repr(mask) not in attempt_repr
    assert "response=" not in attempt_repr


@pytest.mark.parametrize(
    "invalid",
    [
        (),
        ((0,) * 8,) * 3,
        ((0,) * 8,) * 5,
        ((0,) * 7,) * 4,
        ((True,) + (0,) * 7,) + ((0,) * 8,) * 3,
        ((9,) + (0,) * 7,) + ((0,) * 8,) * 3,
    ],
)
def test_mask_boundary_rejects_shape_type_and_range_confusion(invalid: object) -> None:
    """实验 arithmetic 入口必须拒绝非规范 mask, 不能截断或取模。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        with pytest.raises(V1PSRInputError):
            compute_v1_commitment(fixture.profile, invalid)  # type: ignore[arg-type]


def test_response_boundary_rejects_non_integer_and_wrong_shape() -> None:
    """rejection predicate 不得依赖 truthy 或接受不完整 response。"""
    with pytest.raises(V1PSRInputError):
        v1_response_emits(((True,) + (0,) * 7,) + ((0,) * 8,) * 3)
    with pytest.raises(V1PSRInputError):
        v1_response_emits(((0,) * 8,) * 3)
    with pytest.raises(V1PSRInputError):
        compute_v1_response(
            ((0,) * 8,) * 4,
            ((0,) * 8,) * 4,
            object(),  # type: ignore[arg-type]
        )


def test_manifest_rejects_mixed_fixtures_before_creating_artifact(tmp_path: Path) -> None:
    """不同 seed/profile 的 attempts 不得混入同一公开向量清单。"""
    with build_v1_generated_key_fixture(IDENTITY, SEED) as first_fixture:
        first = first_fixture.prove_attempt(0, 0, bytes(32))
    with build_v1_generated_key_fixture(IDENTITY, bytes(reversed(SEED))) as second_fixture:
        second = second_fixture.prove_attempt(0, 1, bytes([1]) * 32)
    target = tmp_path / "mixed.json"

    with pytest.raises(V1PSRManifestError):
        write_v1_vector_manifest(target, (first, second))

    assert not target.exists()


def test_manifest_rejects_symlinked_parent(tmp_path: Path) -> None:
    """测试 artifact 不得经 symlink 写到预期临时目录之外。"""
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        attempt = fixture.prove_attempt(0, 0, bytes(32))

    with pytest.raises(V1PSRManifestError):
        write_v1_vector_manifest(linked_parent / "vectors.json", (attempt,))

    assert not (real_parent / "vectors.json").exists()


def test_generated_fixture_never_accepts_client_selected_matrix_or_profile() -> None:
    """generated-key API 不提供 matrix、profile 或 challenge-policy 覆盖入口。"""
    with pytest.raises(TypeError):
        build_v1_generated_key_fixture(  # type: ignore[call-arg]
            IDENTITY,
            SEED,
            matrix=(),
        )
    assert tuple(inspect.signature(build_v1_generated_key_fixture).parameters) == (
        "identity_id",
        "seed",
    )
    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        attempt_parameters = tuple(inspect.signature(fixture.prove_attempt).parameters)
    assert attempt_parameters == ("trial_index", "retry_index", "transcript_id")


def test_retry_harness_rejects_client_selected_budget_and_identity() -> None:
    """retry budget 和 trusted identity 必须来自本地配置, 不能由 payload 降级。"""
    from can.access import A3V2ProtocolCoordinator
    from can.experiments.v1_psr import run_v1_a3_v2_retry, sample_v1_challenge

    with build_v1_generated_key_fixture(IDENTITY, SEED) as fixture:
        with pytest.raises(V1PSRInputError):
            run_v1_a3_v2_retry(
                fixture,
                A3V2ProtocolCoordinator(),
                A3V2TrustedInput(
                    2,
                    IDENTITY,
                    1,
                    bytes(32),
                    bytes([1]) * 32,
                    b"snapshot",
                ),
                trial_index=0,
                max_attempts=0,
                challenge_for_retry=lambda _: sample_v1_challenge(SEED, 0, 0),
            )

        with pytest.raises(V1PSRInputError):
            run_v1_a3_v2_retry(
                fixture,
                A3V2ProtocolCoordinator(),
                A3V2TrustedInput(
                    2,
                    bytes([9]) * 32,
                    1,
                    bytes(32),
                    bytes([1]) * 32,
                    b"snapshot",
                ),
                trial_index=0,
                max_attempts=1,
                challenge_for_retry=lambda _: sample_v1_challenge(SEED, 0, 0),
            )
