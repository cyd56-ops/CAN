"""A1-B1 PyTorch adapter、startup gate 和 artifact 边界的安全测试。"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import torch

import can.reference.a0 as reference_module
import can.verifier.a1 as a1_module
import can.verifier.a1_torch as torch_module
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
    A1AffineReluLayer,
    A1CompiledRegistry,
    A1Evidence,
    A1EvidenceCode,
    compile_a1_profile,
)
from can.verifier.a1_torch import (
    A1TorchBackend,
    A1TorchBackendError,
    compile_a1_torch_backend,
    verify_a1_torch,
)

TEST_SLOT_ID = 0xB1EC
_ARTIFACT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".pkl", ".pickle"}


def _profile_and_raw() -> tuple[A1CompiledRegistry, bytes]:
    slot = A0Slot(
        TEST_SLOT_ID,
        [[row + 1] * A0_SECRET_SIZE for row in range(A0_COMPONENT_COUNT)],
    )
    profile = compile_a1_profile(slot, (1,) * A0_SECRET_SIZE)
    coefficients = tuple(mod_q(anchor + A0_CENTER) for anchor in profile.anchors)
    raw = (
        bytes([A0_VERSION])
        + A0_PROFILE_ID.to_bytes(2, byteorder="big", signed=False)
        + TEST_SLOT_ID.to_bytes(4, byteorder="big", signed=False)
        + b"".join(value.to_bytes(2, byteorder="big", signed=False) for value in coefficients)
    )
    return A1CompiledRegistry([profile]), raw


def _backend_and_raw() -> tuple[A1TorchBackend, bytes]:
    registry, raw = _profile_and_raw()
    return compile_a1_torch_backend(registry), raw


@pytest.fixture(scope="module")
def fixed_backend() -> tuple[A1TorchBackend, bytes]:
    """构建只读复用的确定性安全测试 backend。"""
    return _backend_and_raw()


@pytest.mark.parametrize("raw_value", [bytearray(23), memoryview(bytes(23)), "x", True, None])
def test_adapter_rejects_raw_type_confusion(
    fixed_backend: tuple[A1TorchBackend, bytes], raw_value: object
) -> None:
    """公共入口必须保留 A0 parser 的 exact-bytes 边界。"""
    backend, _ = fixed_backend

    evidence = verify_a1_torch(cast(bytes, raw_value), backend)

    assert evidence.code is A1EvidenceCode.PARSE_REJECT
    assert not evidence.accepted
    assert backend.active


def test_client_cannot_submit_tensor_parameters_or_claimed_evidence(
    fixed_backend: tuple[A1TorchBackend, bytes],
) -> None:
    """tensor、参数附加和 claimed allow 均不具有 credential 语义。"""
    backend, raw = fixed_backend
    injected = b"device=cuda;dtype=float;threshold=128;scale=0;weight=client" + raw
    claimed = A1Evidence(A1EvidenceCode.NUMERIC_ACCEPT)

    results = (
        verify_a1_torch(cast(bytes, torch.zeros(8, dtype=torch.int32)), backend),
        verify_a1_torch(injected, backend),
        verify_a1_torch(raw + injected, backend),
        verify_a1_torch(cast(bytes, claimed), backend),
        verify_a1_torch(cast(bytes, {"decision": "allow"}), backend),
    )

    assert all(result.code is A1EvidenceCode.PARSE_REJECT for result in results)
    assert list(inspect.signature(verify_a1_torch).parameters) == ["raw_credential", "backend"]


def test_public_adapter_never_calls_other_relation_backends(
    fixed_backend: tuple[A1TorchBackend, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """成功路径不得调用 dependency-free core、exact-ops 或 A0 reference。"""
    backend, raw = fixed_backend

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("fallback relation was called")

    monkeypatch.setattr(a1_module, "_evaluate_core", forbidden)
    monkeypatch.setattr(a1_module, "_run_graph", forbidden)
    monkeypatch.setattr(reference_module, "verify_ref", forbidden)

    evidence = verify_a1_torch(raw, backend)

    assert evidence.code is A1EvidenceCode.NUMERIC_ACCEPT
    assert "verify_ref" not in torch_module.__dict__
    assert "_evaluate_core" not in torch_module.__dict__
    assert "_run_graph" not in torch_module.__dict__
    assert "exact_ops" not in torch_module.__dict__
    assert "_exact_profile_trace" not in verify_a1_torch.__code__.co_names


def test_operator_exception_disables_backend_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch 异常应永久禁用该实例并只产生 CONFIG_REJECT。"""
    backend, raw = _backend_and_raw()
    reference_calls = 0

    def failing_forward(self: object, activation: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("forced operator failure")

    def counted_reference(*args: object, **kwargs: object) -> object:
        nonlocal reference_calls
        reference_calls += 1
        raise AssertionError("reference fallback was called")

    monkeypatch.setattr(torch_module._TorchProfileModule, "forward", failing_forward)
    monkeypatch.setattr(reference_module, "verify_ref", counted_reference)

    first = verify_a1_torch(raw, backend)
    replay = verify_a1_torch(raw, backend)

    assert first.code is A1EvidenceCode.CONFIG_REJECT
    assert replay.code is A1EvidenceCode.CONFIG_REJECT
    assert not backend.active
    assert reference_calls == 0


def _tamper_module(backend: A1TorchBackend, case: str) -> None:
    module = next(iter(backend._entries.values())).module
    if case == "content":
        module.weight_1[0, 0].add_(1)
    elif case == "shape":
        module.weight_1 = module.weight_1[:, :7].contiguous()
    elif case == "stride":
        module.weight_1 = module.weight_1.t().contiguous().t()
    elif case == "dtype":
        module.weight_1 = module.weight_1.to(dtype=torch.int64)
    elif case == "device":
        module.weight_1 = torch.empty((40, 8), dtype=torch.int32, device="meta")
    elif case == "training":
        module.train()
    elif case == "persistent":
        module._non_persistent_buffers_set.remove("weight_1")
    elif case == "hook":
        module.register_forward_hook(lambda _module, _inputs, output: output)
    else:
        raise AssertionError(f"unknown tamper case: {case}")


@pytest.mark.parametrize(
    "case", ["content", "shape", "stride", "dtype", "device", "training", "persistent", "hook"]
)
def test_module_tamper_fails_closed_and_disables_backend(case: str) -> None:
    """buffer 和 module 元数据漂移不得进入 graph 或触发弱路线。"""
    backend, raw = _backend_and_raw()
    _tamper_module(backend, case)

    evidence = verify_a1_torch(raw, backend)

    assert evidence.code is A1EvidenceCode.CONFIG_REJECT
    assert not evidence.accepted
    assert not backend.active


@pytest.mark.parametrize("case", ["float", "non_bit", "wrong_shape"])
def test_invalid_operator_output_fails_closed(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """输出必须是 shape (1,) 的 exact int32 bit。"""
    backend, raw = _backend_and_raw()

    def invalid_forward(self: object, activation: torch.Tensor) -> torch.Tensor:
        if case == "float":
            return torch.tensor((1.0,), dtype=torch.float32)
        if case == "non_bit":
            return torch.tensor((2,), dtype=torch.int32)
        return torch.tensor((1, 1), dtype=torch.int32)

    monkeypatch.setattr(torch_module._TorchProfileModule, "forward", invalid_forward)

    evidence = verify_a1_torch(raw, backend)

    assert evidence.code is A1EvidenceCode.CONFIG_REJECT
    assert not backend.active


def test_runtime_environment_drift_disables_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """激活后的 torch 版本漂移必须禁用实例且不继续服务。"""
    backend, raw = _backend_and_raw()
    monkeypatch.setattr(torch_module, "package_version", lambda _name: "2.13.1+cpu")

    evidence = verify_a1_torch(raw, backend)

    assert evidence.code is A1EvidenceCode.CONFIG_REJECT
    assert not backend.active


def test_range_certificate_failure_prevents_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    """无法证明 int32/int64 范围时不能产生 active backend。"""
    registry, _ = _profile_and_raw()
    original = torch_module._interval_layer

    def unsafe_interval(
        input_bounds: tuple[tuple[int, int], ...],
        layer: A1AffineReluLayer,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        outputs, affine = original(input_bounds, layer)
        if len(affine) == 40:
            affine = ((-385, 386), *affine[1:])
        return outputs, affine

    monkeypatch.setattr(torch_module, "_interval_layer", unsafe_interval)

    with pytest.raises(A1TorchBackendError):
        compile_a1_torch_backend(registry)


def test_replay_and_concurrent_verification_do_not_mutate_buffers(
    fixed_backend: tuple[A1TorchBackend, bytes],
) -> None:
    """重复与并发提交应只返回相同 evidence, 不创建隐藏权限状态。"""
    backend, raw = fixed_backend
    module = next(iter(backend._entries.values())).module
    before = {name: tensor.clone() for name, tensor in module.named_buffers()}

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(lambda _index: verify_a1_torch(raw, backend), range(32)))

    assert all(result.code is A1EvidenceCode.NUMERIC_ACCEPT for result in results)
    assert backend.active
    assert module.state_dict() == {}
    assert set(before) == {name for name, _ in module.named_buffers()}
    for name, tensor in module.named_buffers():
        assert torch.equal(tensor, before[name])


def _project_artifacts(project_root: Path) -> set[Path]:
    return {
        path.relative_to(project_root)
        for path in project_root.rglob("*")
        if path.is_file()
        and ".venv" not in path.parts
        and path.suffix.lower() in _ARTIFACT_SUFFIXES
    }


def test_backend_writes_no_model_or_secret_bearing_artifact() -> None:
    """backend 构建与验证不得在项目目录生成 checkpoint 或 export。"""
    project_root = Path(__file__).resolve().parents[2]
    before = _project_artifacts(project_root)

    backend, raw = _backend_and_raw()
    evidence = verify_a1_torch(raw, backend)
    module = next(iter(backend._entries.values())).module

    assert evidence.code is A1EvidenceCode.NUMERIC_ACCEPT
    assert module.state_dict() == {}
    assert _project_artifacts(project_root) == before


def test_dependency_free_packages_import_when_torch_is_blocked() -> None:
    """可选模块缺失时 A0/A1 基础包不得隐式导入 torch 或 fallback。"""
    project_root = Path(__file__).resolve().parents[2]
    script = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch blocked by test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import can.reference
import can.verifier
assert "torch" not in can.verifier.__dict__
assert "A1TorchBackend" not in can.verifier.__dict__
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_backend_has_no_global_compiled_profile_or_authorization_state() -> None:
    """模块全局不能保存 profile、credential、decision 或 capability。"""
    forbidden_names = {
        "registry",
        "profile",
        "credential",
        "decision",
        "authorization",
        "capability",
    }
    evidence_fields = {item.name for item in fields(A1Evidence)}

    assert forbidden_names.isdisjoint(torch_module.__dict__)
    assert evidence_fields == {"code"}
