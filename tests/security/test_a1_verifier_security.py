"""A1-C1 verifier adapter 和 compiled profile 的防御性安全测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import cast

import pytest

import can.reference.a0 as reference_module
import can.verifier.a1 as a1_module
from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0_VERSION,
    A0Slot,
    mod_q,
)
from can.verifier import (
    A1CompiledRegistry,
    A1Evidence,
    A1EvidenceCode,
    A1ProfileValidationError,
    compile_a1_profile,
    verify_a1,
)

TEST_SLOT_ID = 0xCA11


def _fixture() -> tuple[A1CompiledRegistry, bytes]:
    slot = A0Slot(
        TEST_SLOT_ID,
        [[row + 1] * A0_SECRET_SIZE for row in range(A0_COMPONENT_COUNT)],
    )
    secret = (1,) * A0_SECRET_SIZE
    profile = compile_a1_profile(slot, secret)
    raw = (
        bytes([A0_VERSION])
        + A0_PROFILE_ID.to_bytes(2, byteorder="big", signed=False)
        + TEST_SLOT_ID.to_bytes(4, byteorder="big", signed=False)
        + b"".join(
            mod_q(anchor + A0_CENTER).to_bytes(2, byteorder="big", signed=False)
            for anchor in profile.anchors
        )
    )
    return A1CompiledRegistry([profile]), raw


@pytest.mark.parametrize(
    "raw_value",
    [bytearray(23), memoryview(bytes(23)), "x", True, None],
)
def test_adapter_rejects_raw_credential_type_confusion(raw_value: object) -> None:
    """A1 公共入口必须保留 A0 parser 的 exact-bytes 边界。"""
    registry, _ = _fixture()

    evidence = verify_a1(cast(bytes, raw_value), registry)

    assert evidence.code is A1EvidenceCode.PARSE_REJECT
    assert not evidence.accepted


def test_client_cannot_append_anchor_weights_or_threshold() -> None:
    """请求附加 compiled 参数时只能得到固定解析拒绝。"""
    registry, raw = _fixture()
    injected = b"anchor=0;weights=client;threshold=128;scale=0;candidate=weak" + raw

    prefixed = verify_a1(injected, registry)
    appended = verify_a1(raw + injected, registry)

    assert prefixed.code is A1EvidenceCode.PARSE_REJECT
    assert appended.code is A1EvidenceCode.PARSE_REJECT


def test_evidence_or_claimed_decision_cannot_be_submitted_as_credential() -> None:
    """预计算 evidence 或 claimed allow 不具有 verifier 输入语义。"""
    registry, _ = _fixture()
    evidence = A1Evidence(A1EvidenceCode.NUMERIC_ACCEPT)

    submitted_evidence = verify_a1(cast(bytes, evidence), registry)
    submitted_decision = verify_a1(cast(bytes, {"decision": "allow"}), registry)

    assert submitted_evidence.code is A1EvidenceCode.PARSE_REJECT
    assert submitted_decision.code is A1EvidenceCode.PARSE_REJECT


def test_main_adapter_does_not_call_reference_or_define_exact_ops_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主路径接受时不得执行 reference 或普通算子 fallback。"""
    registry, raw = _fixture()

    def forbidden_reference(*args: object, **kwargs: object) -> object:
        raise AssertionError("reference fallback was called")

    monkeypatch.setattr(reference_module, "verify_ref", forbidden_reference)

    evidence = verify_a1(raw, registry)

    assert evidence.code is A1EvidenceCode.NUMERIC_ACCEPT
    assert "verify_ref" not in a1_module.__dict__
    assert "exact_ops" not in a1_module.__dict__


def test_internal_graph_failure_fails_closed_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """内部异常只能产生配置拒绝且不能切换到 reference 路线。"""
    registry, raw = _fixture()
    reference_calls = 0

    def failing_core(*args: object, **kwargs: object) -> int:
        raise RuntimeError("forced core failure")

    def counted_reference(*args: object, **kwargs: object) -> object:
        nonlocal reference_calls
        reference_calls += 1
        raise AssertionError("reference fallback was called")

    monkeypatch.setattr(a1_module, "_evaluate_core", failing_core)
    monkeypatch.setattr(reference_module, "verify_ref", counted_reference)

    evidence = verify_a1(raw, registry)

    assert evidence.code is A1EvidenceCode.CONFIG_REJECT
    assert not evidence.accepted
    assert reference_calls == 0


def test_core_call_graph_contains_no_ordinary_relation_operators() -> None:
    """固定 graph 模块不得引入测试 exact-ops baseline 或 reference 入口。"""
    core_globals = a1_module._evaluate_core.__globals__

    assert "verify_ref" not in core_globals
    assert "floor" not in core_globals
    assert "exact_ops" not in core_globals


def test_compiled_profile_and_registry_do_not_mutate_on_replay() -> None:
    """重复 credential 只重放相同 evidence 且不创建隐藏权限状态。"""
    registry, raw = _fixture()
    profiles_before = registry.profiles

    first = verify_a1(raw, registry)
    replayed = verify_a1(raw, registry)

    assert first == replayed
    assert first.accepted
    assert registry.profiles == profiles_before
    with pytest.raises(FrozenInstanceError):
        registry_attribute = "_profiles"
        setattr(registry, registry_attribute, {})


def test_evidence_contains_only_a_stable_code() -> None:
    """外部 evidence 不得泄露距离、锚点、权重或授权原语。"""
    evidence_fields = {item.name for item in fields(A1Evidence)}

    assert evidence_fields == {"code"}
    assert not (
        {
            "anchors",
            "weights",
            "distances",
            "gate",
            "decision",
            "authorization",
            "capability",
        }
        & evidence_fields
    )


def test_secret_and_slot_type_confusion_fail_at_compiler_boundary() -> None:
    """bool secret、禁用 slot 和错误对象不能进入 compiled registry。"""
    matrix = [[row + 1] * A0_SECRET_SIZE for row in range(A0_COMPONENT_COUNT)]
    enabled = A0Slot(TEST_SLOT_ID, matrix)
    disabled = A0Slot(TEST_SLOT_ID, matrix, enabled=False)

    with pytest.raises(A1ProfileValidationError):
        compile_a1_profile(enabled, [False] * A0_SECRET_SIZE)
    with pytest.raises(A1ProfileValidationError):
        compile_a1_profile(disabled, (1,) * A0_SECRET_SIZE)
    with pytest.raises(A1ProfileValidationError):
        compile_a1_profile(cast(A0Slot, object()), (1,) * A0_SECRET_SIZE)
