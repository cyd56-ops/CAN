"""A2-E1 门控实验入口的参数与 artifact 安全测试。"""

import pytest

import can.experiments.a2_gate as gate_experiment


def test_gate_cli_does_not_accept_authorization_model_or_backend_overrides() -> None:
    """CLI 不得让请求方选择 evidence、策略、模型、backend 或 credential。"""
    forbidden_arguments = (
        ["--evidence", "numeric_accept"],
        ["--allow"],
        ["--policy", "weak"],
        ["--model", "public"],
        ["--backend", "dependency-free"],
        ["--credential", "00"],
        ["--device", "cuda"],
    )

    for arguments in forbidden_arguments:
        with pytest.raises(SystemExit) as raised:
            gate_experiment.main(arguments)
        assert raised.value.code == 2
