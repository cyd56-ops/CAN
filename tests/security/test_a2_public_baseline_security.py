"""A2-E2 public baseline 的依赖、CLI 与 artifact 防御性测试。"""

import ast
from pathlib import Path

import pytest

import can.experiments.a2_public_baseline as public_baseline

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_public_modules_do_not_import_protected_model_gate_or_baseline() -> None:
    """public 实现不得依赖 protected forward、训练入口或协调器。"""
    paths = (
        REPOSITORY_ROOT / "src/can/model/a2_public_mlp.py",
        REPOSITORY_ROOT / "src/can/experiments/a2_public_baseline.py",
    )
    forbidden_modules = {
        "can.model.a2_mlp",
        "can.experiments.a2_baseline",
        "can.access.a2_gate",
        "can.experiments.a2_gate",
    }

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imported_modules.isdisjoint(forbidden_modules)


def test_public_cli_rejects_training_routing_and_authorization_overrides() -> None:
    """CLI 不得开放模型、训练、路由、凭据或授权参数。"""
    forbidden_arguments = (
        ["--epochs", "1"],
        ["--device", "cuda"],
        ["--model", "protected"],
        ["--head", "protected"],
        ["--backend", "fallback"],
        ["--data-root", "/tmp/data"],
        ["--entry", "protected"],
        ["--capability", "protected"],
        ["--credential", "00"],
        ["--allow", "true"],
        ["--evidence", "numeric_accept"],
    )

    for arguments in forbidden_arguments:
        with pytest.raises(SystemExit) as raised:
            public_baseline.main(arguments)
        assert raised.value.code == 2


def test_public_report_writer_stays_under_fixed_ignored_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """public 报告只能由固定 repeat 映射到本地 artifact 根。"""
    monkeypatch.setattr(public_baseline, "A2_REPORT_ROOT", tmp_path)

    output = public_baseline._write_report({"schema_version": 1}, 1)

    assert output == tmp_path / "public-baseline-repeat-1.json"
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not (tmp_path / "public-baseline-repeat-1.json.tmp").exists()
