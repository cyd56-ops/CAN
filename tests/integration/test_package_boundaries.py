"""计划包边界的最小集成测试。"""

from importlib import import_module


def test_planned_package_boundaries_are_importable() -> None:
    """技术 bootstrap 定义的包边界都应可导入。"""
    modules = (
        "can.access",
        "can.experiments",
        "can.model",
        "can.reference",
        "can.verifier",
    )

    for module_name in modules:
        assert import_module(module_name).__name__ == module_name
