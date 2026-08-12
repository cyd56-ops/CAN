"""A2-E2 三态实验入口的训练、参数与报告边界测试。"""

import ast
import inspect
from pathlib import Path

import can.experiments.a2_capability as capability_experiment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_capability_experiment_never_calls_a_training_helper() -> None:
    """三态 checkpoint 的实验入口不得重训任一已验收 baseline。"""
    source_path = REPOSITORY_ROOT / "src/can/experiments/a2_capability.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "_train_model" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes


def test_capability_experiment_api_accepts_no_route_or_authority_overrides() -> None:
    """本地实验 API 只能接收两个模型对象, 不接收路由、策略或 credential。"""
    parameters = inspect.signature(capability_experiment.run_a2_capability_experiment).parameters

    assert tuple(parameters) == ("protected_model", "public_model")
    assert not {
        "entry",
        "policy",
        "capability",
        "backend",
        "credential",
        "evidence",
        "decision",
        "fallback",
        "report_path",
    } & set(parameters)


def test_capability_report_schema_does_not_name_sensitive_request_material() -> None:
    """固定报告构造源码不得加入请求、凭据、logits 或 feature 字段。"""
    source = inspect.getsource(capability_experiment.run_a2_capability_experiment)

    for forbidden_key in (
        '"credential"',
        '"images"',
        '"evidence"',
        '"logits"',
        '"features"',
        '"capability"',
        '"secret"',
    ):
        assert forbidden_key not in source
