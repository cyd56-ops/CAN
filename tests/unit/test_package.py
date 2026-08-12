"""顶层包的最小单元测试。"""

import can


def test_package_version() -> None:
    """包版本应与技术 bootstrap 版本一致。"""
    assert can.__version__ == "0.1.0"
