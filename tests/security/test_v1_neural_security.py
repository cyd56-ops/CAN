"""V1-C1 neural verifier 的防御性边界测试。"""

from __future__ import annotations

import inspect

import can.verifier.v1 as neural_module
from can.reference import build_v1_conformance_profile
from can.verifier.v1 import V1NeuralEvidenceCode, compile_v1_neural_profile, verify_v1_neural


def test_v1_neural_has_no_reference_or_authority_fallback() -> None:
    """V1-C1 模块不能导入 exact reference、access 或 model。"""
    source = inspect.getsource(neural_module)
    assert "verify_v1_ref" not in source
    assert "from can.access" not in source
    assert "from can.model" not in source


def test_v1_neural_rejects_foreign_route_and_type_confusion() -> None:
    """foreign bytes、错误 profile 和类型混淆必须 fail closed。"""
    compiled = compile_v1_neural_profile(build_v1_conformance_profile(bytes(range(32))))
    assert (
        verify_v1_neural(bytes(23), bytes(32), bytes(32), bytes(32), compiled).code
        is V1NeuralEvidenceCode.INPUT_REJECT
    )
    assert (
        verify_v1_neural(None, None, None, None, object()).code
        is V1NeuralEvidenceCode.CONFIG_REJECT
    )
