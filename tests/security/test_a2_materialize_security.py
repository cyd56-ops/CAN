"""A2 trusted materializer 与 no-training evaluator 的边界测试。"""

import ast
import inspect
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

import can.experiments.a2_capability as capability_experiment
import can.experiments.a2_materialize as materialize
from can.model.a2_mlp import A2FashionMNISTMLP

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _function_node(source_path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


def test_materialized_report_entry_has_no_training_calls() -> None:
    """加载并报告入口不能把训练逻辑混入 no-training evaluator。"""
    source_path = REPOSITORY_ROOT / "src/can/experiments/a2_materialize.py"
    node = _function_node(source_path, "run_a2_materialized_report")
    called_attributes = {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert "_train_model" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert tuple(inspect.signature(materialize.run_a2_materialized_report).parameters) == ()


def test_materializer_cli_has_only_fixed_local_routes() -> None:
    """CLI 不接受 request、策略、报告路径或模型覆盖。"""
    parameters = inspect.signature(materialize.main).parameters
    assert tuple(parameters) == ("argv",)
    source = inspect.getsource(materialize.main)
    for forbidden in (
        "--state-root",
        "--report-path",
        "--credential",
        "--policy",
        "--backend",
        "--model",
        "--fallback",
    ):
        assert forbidden not in source


def test_materializer_uses_weights_only_loading_and_no_full_model_pickle() -> None:
    """本地 artifact 只按受限 state_dict 加载, 不恢复完整模型 pickle。"""
    source = inspect.getsource(materialize)
    assert "weights_only=True" in source
    assert "torch.save(model" not in source
    assert "optimizer" in source
    assert capability_experiment.run_a2_capability_experiment.__name__ == (
        "run_a2_capability_experiment"
    )


def _cloned_state(model: A2FashionMNISTMLP) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in model.state_dict().items()
    )


def test_state_validator_rejects_mapping_dtype_device_layout_and_finiteness() -> None:
    """加载边界拒绝非 canonical mapping、dtype、device、layout 和数值。"""
    model = A2FashionMNISTMLP()
    canonical = _cloned_state(model)
    first_key = next(iter(canonical))

    with pytest.raises(materialize.A2MaterializationError):
        materialize._validate_state_dict(model, dict(canonical))

    replacements = (
        canonical[first_key].to(dtype=torch.float64),
        canonical[first_key].to(device="meta"),
        canonical[first_key].transpose(0, 1).contiguous().transpose(0, 1),
        canonical[first_key].clone(),
    )
    replacements[-1].reshape(-1)[0] = torch.nan
    for replacement in replacements:
        malformed = _cloned_state(model)
        malformed[first_key] = replacement
        with pytest.raises(materialize.A2MaterializationError):
            materialize._validate_state_dict(model, malformed)


def test_state_loader_rejects_symlink(tmp_path: Path) -> None:
    """local state 路径不能通过 symlink 改写信任根。"""
    target = tmp_path / "target.pt"
    torch.save(A2FashionMNISTMLP().state_dict(), target)
    link = tmp_path / "state.pt"
    link.symlink_to(target)

    with pytest.raises(materialize.A2MaterializationError):
        materialize._load_state_file("protected", link, materialize._file_digest(target))
