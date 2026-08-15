"""V1-M1 baseline artifact 路径安全测试。"""

from pathlib import Path

import pytest

from can.experiments.v1_m1_baseline import V1M1BaselineError, _artifact_paths


def test_artifact_root_rejects_a_symlink(tmp_path: Path) -> None:
    """artifact root 不能通过 symlink 指向请求方选择的其他目录。"""
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "artifacts"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(V1M1BaselineError, match="not a canonical directory"):
        _artifact_paths(root, 1)
