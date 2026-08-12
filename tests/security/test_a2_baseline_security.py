"""A2-E1 baseline 的输入与 artifact 防御性测试。"""

from pathlib import Path

import pytest

import can.experiments.a2_baseline as baseline


def test_cli_does_not_accept_model_training_or_authorization_overrides() -> None:
    """CLI 不得开放模型、设备、epoch、数据路径或授权参数。"""
    forbidden_arguments = (
        ["--epochs", "1"],
        ["--device", "cuda"],
        ["--model", "lenet"],
        ["--data-root", "/tmp/data"],
        ["--allow", "true"],
        ["--evidence", "numeric_accept"],
    )

    for arguments in forbidden_arguments:
        with pytest.raises(SystemExit) as raised:
            baseline.main(arguments)
        assert raised.value.code == 2


def test_report_writer_stays_under_fixed_ignored_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """报告只能由固定 repeat 映射到本地 artifact 根。"""
    monkeypatch.setattr(baseline, "A2_REPORT_ROOT", tmp_path)

    output = baseline._write_report({"schema_version": 1}, 1)

    assert output == tmp_path / "baseline-repeat-1.json"
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not (tmp_path / "baseline-repeat-1.json.tmp").exists()
