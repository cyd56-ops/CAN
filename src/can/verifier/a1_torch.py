"""A1-B1 PyTorch CPU 精确整数 conformance backend。"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from types import MappingProxyType
from typing import Final

import torch
from torch import Tensor, nn

from can.reference import (
    A0_CENTER,
    A0_COMPONENT_COUNT,
    A0_MODULUS,
    A0_PROFILE_ID,
    CredentialParseError,
    mod_q,
    parse_credential,
)
from can.verifier.a1 import (
    A1_CANDIDATE_ID,
    A1_INPUT_WIDTH,
    A1_LAYER_WIDTHS,
    A1_SCALE,
    A1AffineReluLayer,
    A1CompiledProfile,
    A1CompiledRegistry,
    A1Evidence,
    A1EvidenceCode,
    A1ProfileLookupError,
)

A1_TORCH_BACKEND_ID: Final = "CAN-TORCH-CPU-EXACT-v1"
A1_TORCH_VERSION: Final = "2.13.0+cpu"

_INT32_MIN: Final = -(2**31)
_INT32_MAX: Final = 2**31 - 1
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1
_BUFFER_NAMES: Final = (
    "weight_1",
    "bias_1",
    "weight_2",
    "bias_2",
    "weight_3",
    "bias_3",
)


class A1TorchBackendError(RuntimeError):
    """表示 A1-B1 环境、构造或执行未通过 fail-closed gate。"""


@dataclass(frozen=True, slots=True)
class _LayerExecution:
    products: Tensor
    accumulator: Tensor
    preactivation: Tensor
    relu: Tensor
    output: Tensor


@dataclass(frozen=True, slots=True)
class _TorchTrace:
    layers: tuple[Tensor, Tensor, Tensor]
    executions: tuple[_LayerExecution, _LayerExecution, _LayerExecution]


def _execute_layer(activation: Tensor, weight: Tensor, bias: Tensor) -> _LayerExecution:
    expanded_input = activation.unsqueeze(0).expand_as(weight)
    products_i32 = torch.mul(weight, expanded_input)
    accumulator_i64 = torch.sum(products_i32, dim=1, dtype=torch.int64)
    preactivation_i64 = torch.add(accumulator_i64, bias.to(dtype=torch.int64))
    relu_i64 = torch.clamp(preactivation_i64, min=0)
    next_activation_i32 = relu_i64.to(dtype=torch.int32)
    return _LayerExecution(
        products_i32,
        accumulator_i64,
        preactivation_i64,
        relu_i64,
        next_activation_i32,
    )


class _TorchProfileModule(nn.Module):
    weight_1: Tensor
    bias_1: Tensor
    weight_2: Tensor
    bias_2: Tensor
    weight_3: Tensor
    bias_3: Tensor

    def __init__(self, profile: A1CompiledProfile) -> None:
        super().__init__()
        for index, layer in enumerate(profile.layers, start=1):
            weight = torch.tensor(layer.weights, dtype=torch.int32, device="cpu")
            bias = torch.tensor(layer.bias, dtype=torch.int32, device="cpu")
            self.register_buffer(f"weight_{index}", weight, persistent=False)
            self.register_buffer(f"bias_{index}", bias, persistent=False)
        self.eval()

    def forward(self, activation: Tensor) -> Tensor:
        """按固定三层整数算子序列返回单元素张量。"""
        first = _execute_layer(activation, self.weight_1, self.bias_1).output
        second = _execute_layer(first, self.weight_2, self.bias_2).output
        return _execute_layer(second, self.weight_3, self.bias_3).output

    def run_with_trace(self, activation: Tensor) -> _TorchTrace:
        """执行与公共路径相同的三层序列并保留测试用中间量。"""
        first = _execute_layer(activation, self.weight_1, self.bias_1)
        second = _execute_layer(first.output, self.weight_2, self.bias_2)
        third = _execute_layer(second.output, self.weight_3, self.bias_3)
        return _TorchTrace(
            (first.output, second.output, third.output),
            (first, second, third),
        )


@dataclass(frozen=True, slots=True)
class _BackendEntry:
    profile: A1CompiledProfile
    module: _TorchProfileModule


class A1TorchBackend:
    """持有通过 startup gate 的本地 A1-B1 profile modules。"""

    __slots__ = ("_active", "_entries")

    def __init__(self, registry: A1CompiledRegistry) -> None:
        if type(registry) is not A1CompiledRegistry:
            raise A1TorchBackendError("registry must be exactly A1CompiledRegistry")
        if not registry.profiles:
            raise A1TorchBackendError("backend requires at least one compiled profile")

        self._active = False
        _validate_environment()
        _operator_microprobe()

        entries: dict[tuple[int, int], _BackendEntry] = {}
        for profile in registry.profiles:
            module = _TorchProfileModule(profile)
            entry = _BackendEntry(profile, module)
            _validate_profile_contract(profile)
            _validate_module(entry)
            _certify_profile(entry)
            entries[(profile.profile_id, profile.slot_id)] = entry

        self._entries: Mapping[tuple[int, int], _BackendEntry] = MappingProxyType(entries)
        self._active = True

    @property
    def backend_id(self) -> str:
        """返回固定且不可由请求方选择的 backend 标识。"""
        return A1_TORCH_BACKEND_ID

    @property
    def active(self) -> bool:
        """返回当前实例是否仍通过 fail-closed gate。"""
        return self._active

    def _lookup(self, profile_id: int, slot_id: int) -> _BackendEntry:
        if type(profile_id) is not int or profile_id != A0_PROFILE_ID:
            raise A1ProfileLookupError("unknown A1 torch profile")
        if type(slot_id) is not int:
            raise A1ProfileLookupError("slot_id must be exactly int")
        entry = self._entries.get((profile_id, slot_id))
        if entry is None:
            raise A1ProfileLookupError("unknown A1 torch slot")
        return entry

    def _disable(self) -> None:
        self._active = False


def _validate_environment() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise A1TorchBackendError("unsupported operating system or architecture")
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise A1TorchBackendError("unsupported Python runtime")
    try:
        installed_version = package_version("torch")
    except PackageNotFoundError as error:
        raise A1TorchBackendError("torch distribution is unavailable") from error
    if installed_version != A1_TORCH_VERSION or str(torch.__version__) != A1_TORCH_VERSION:
        raise A1TorchBackendError("unsupported torch version")
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise A1TorchBackendError("accelerator-enabled torch build is unsupported")
    if torch.cuda.is_available():
        raise A1TorchBackendError("CUDA runtime must remain unavailable")


def _expected_buffer_contract(
    profile: A1CompiledProfile, name: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[object, ...]]:
    kind, raw_index = name.split("_", maxsplit=1)
    layer = profile.layers[int(raw_index) - 1]
    if kind == "weight":
        values: tuple[object, ...] = tuple(tuple(row) for row in layer.weights)
        return (layer.output_width, layer.input_width), (layer.input_width, 1), values
    values = tuple(layer.bias)
    return (layer.output_width,), (1,), values


def _canonical_tensor_values(tensor: Tensor) -> tuple[object, ...]:
    values = tensor.tolist()
    if tensor.ndim == 2:
        return tuple(tuple(row) for row in values)
    return tuple(values)


def _validate_module(entry: _BackendEntry) -> None:
    module = entry.module
    named_buffers = dict(module.named_buffers())
    if tuple(named_buffers) != _BUFFER_NAMES:
        raise A1TorchBackendError("module has an unexpected buffer set")
    if tuple(module.parameters()):
        raise A1TorchBackendError("module must not contain parameters")
    if tuple(module.children()):
        raise A1TorchBackendError("module must not contain parameters or child modules")
    if module.training:
        raise A1TorchBackendError("module must remain in evaluation mode")
    if module.state_dict():
        raise A1TorchBackendError("non-persistent verifier buffers entered state_dict")
    if module._non_persistent_buffers_set != set(_BUFFER_NAMES):
        raise A1TorchBackendError("verifier buffer persistence changed")
    if any(
        hooks
        for hooks in (
            module._forward_hooks,
            module._forward_pre_hooks,
            module._backward_hooks,
            module._backward_pre_hooks,
        )
    ):
        raise A1TorchBackendError("module hooks are unsupported")

    for name, tensor in named_buffers.items():
        expected_shape, expected_stride, expected = _expected_buffer_contract(entry.profile, name)
        if (
            type(tensor) is not Tensor
            or tensor.dtype is not torch.int32
            or tensor.device.type != "cpu"
            or tensor.layout is not torch.strided
            or tuple(tensor.shape) != expected_shape
            or tensor.stride() != expected_stride
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise A1TorchBackendError("module buffer metadata changed")
        if _canonical_tensor_values(tensor) != expected:
            raise A1TorchBackendError("module buffer content changed")


def _validate_profile_contract(profile: A1CompiledProfile) -> None:
    if (
        type(profile) is not A1CompiledProfile
        or profile.candidate_id != A1_CANDIDATE_ID
        or profile.profile_id != A0_PROFILE_ID
        or profile.scale != A1_SCALE
        or len(profile.layers) != 3
    ):
        raise A1TorchBackendError("compiled profile contract changed")
    topology = tuple((layer.input_width, layer.output_width) for layer in profile.layers)
    if topology != (
        (A1_INPUT_WIDTH, A1_LAYER_WIDTHS[0]),
        (A1_LAYER_WIDTHS[0], A1_LAYER_WIDTHS[1]),
        (A1_LAYER_WIDTHS[1], A1_LAYER_WIDTHS[2]),
    ):
        raise A1TorchBackendError("compiled profile topology changed")

    bounds = tuple((0, A0_MODULUS - 1) for _ in range(A1_INPUT_WIDTH))
    layer1_bounds, layer1_affine = _interval_layer(bounds, profile.layers[0])
    if min(low for low, _ in layer1_affine) < -384 or max(high for _, high in layer1_affine) > 385:
        raise A1TorchBackendError("layer 1 range certificate failed")
    layer2_bounds, layer2_affine = _interval_layer(layer1_bounds, profile.layers[1])
    if min(low for low, _ in layer2_affine) < -1145 or max(high for _, high in layer2_affine) > 907:
        raise A1TorchBackendError("layer 2 range certificate failed")
    if any(not 0 <= low <= high <= _INT32_MAX for low, high in layer2_bounds):
        raise A1TorchBackendError("layer 2 cast range certificate failed")


def _interval_layer(
    input_bounds: tuple[tuple[int, int], ...],
    layer: A1AffineReluLayer,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    affine_bounds: list[tuple[int, int]] = []
    output_bounds: list[tuple[int, int]] = []
    for row, bias in zip(layer.weights, layer.bias, strict=True):
        low = bias
        high = bias
        for weight, (input_low, input_high) in zip(row, input_bounds, strict=True):
            endpoints = (weight * input_low, weight * input_high)
            if any(not _INT32_MIN <= product <= _INT32_MAX for product in endpoints):
                raise A1TorchBackendError("int32 product range certificate failed")
            low += min(endpoints)
            high += max(endpoints)
        if not _INT64_MIN <= low <= high <= _INT64_MAX:
            raise A1TorchBackendError("int64 accumulator range certificate failed")
        affine_bounds.append((low, high))
        output_bounds.append((max(0, low), max(0, high)))
    return tuple(output_bounds), tuple(affine_bounds)


def _validate_input_tensor(activation: Tensor) -> None:
    if (
        type(activation) is not Tensor
        or activation.dtype is not torch.int32
        or activation.device.type != "cpu"
        or activation.layout is not torch.strided
        or tuple(activation.shape) != (A1_INPUT_WIDTH,)
        or activation.stride() != (1,)
        or not activation.is_contiguous()
        or activation.requires_grad
    ):
        raise A1TorchBackendError("canonical credential tensor metadata changed")
    values = activation.tolist()
    if any(type(value) is not int or not 0 <= value < A0_MODULUS for value in values):
        raise A1TorchBackendError("canonical credential tensor range changed")


def _validate_execution(trace: _TorchTrace) -> None:
    expected_widths = A1_LAYER_WIDTHS
    for index, (activation, execution, width) in enumerate(
        zip(trace.layers, trace.executions, expected_widths, strict=True), start=1
    ):
        weight_width = A1_INPUT_WIDTH if index == 1 else expected_widths[index - 2]
        if (
            execution.products.dtype is not torch.int32
            or execution.products.device.type != "cpu"
            or execution.products.layout is not torch.strided
            or tuple(execution.products.shape) != (width, weight_width)
            or execution.products.stride() != (weight_width, 1)
            or not execution.products.is_contiguous()
            or execution.products.requires_grad
        ):
            raise A1TorchBackendError("product tensor contract changed")
        for intermediate in (
            execution.accumulator,
            execution.preactivation,
            execution.relu,
        ):
            if (
                intermediate.dtype is not torch.int64
                or intermediate.device.type != "cpu"
                or tuple(intermediate.shape) != (width,)
                or intermediate.stride() != (1,)
                or not intermediate.is_contiguous()
                or intermediate.requires_grad
            ):
                raise A1TorchBackendError("int64 intermediate tensor contract changed")
        if (
            activation.dtype is not torch.int32
            or activation.device.type != "cpu"
            or tuple(activation.shape) != (width,)
            or activation.stride() != (1,)
            or not activation.is_contiguous()
            or activation.requires_grad
        ):
            raise A1TorchBackendError("activation tensor contract changed")
        relu_values = execution.relu.tolist()
        if any(type(value) is not int or not 0 <= value <= _INT32_MAX for value in relu_values):
            raise A1TorchBackendError("int32 cast range certificate failed")


def _operator_microprobe() -> None:
    activation = torch.tensor((0, 256), dtype=torch.int32, device="cpu")
    weight = torch.tensor(((-1, 1), (1, -1)), dtype=torch.int32, device="cpu")
    bias = torch.tensor((3, -4), dtype=torch.int32, device="cpu")
    execution = _execute_layer(activation, weight, bias)
    if (
        execution.products.dtype is not torch.int32
        or execution.accumulator.dtype is not torch.int64
        or execution.preactivation.dtype is not torch.int64
        or execution.relu.dtype is not torch.int64
        or execution.output.dtype is not torch.int32
        or execution.products.tolist() != [[0, 256], [0, -256]]
        or execution.accumulator.tolist() != [256, -256]
        or execution.preactivation.tolist() != [259, -260]
        or execution.relu.tolist() != [259, 0]
        or execution.output.tolist() != [259, 0]
    ):
        raise A1TorchBackendError("fixed operator microprobe failed")


def _exact_layer(inputs: tuple[int, ...], layer: A1AffineReluLayer) -> tuple[int, ...]:
    return tuple(
        max(0, bias + sum(weight * value for weight, value in zip(row, inputs, strict=True)))
        for row, bias in zip(layer.weights, layer.bias, strict=True)
    )


def _exact_profile_trace(
    coefficients: tuple[int, ...], profile: A1CompiledProfile
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    first = _exact_layer(coefficients, profile.layers[0])
    second = _exact_layer(first, profile.layers[1])
    third = _exact_layer(second, profile.layers[2])
    return first, second, third


def _tensor_trace(coefficients: tuple[int, ...], module: _TorchProfileModule) -> _TorchTrace:
    activation = torch.tensor(coefficients, dtype=torch.int32, device="cpu")
    _validate_input_tensor(activation)
    with torch.inference_mode():
        trace = module.run_with_trace(activation)
        runtime_output = module(activation)
    _validate_execution(trace)
    runtime_value = _validate_output_tensor(runtime_output)
    if trace.layers[-1].tolist() != [runtime_value]:
        raise A1TorchBackendError("runtime and trace paths diverged")
    return trace


def _validate_output_tensor(output: Tensor) -> int:
    if (
        type(output) is not Tensor
        or output.dtype is not torch.int32
        or output.device.type != "cpu"
        or output.layout is not torch.strided
        or tuple(output.shape) != (1,)
        or output.stride() != (1,)
        or not output.is_contiguous()
        or output.requires_grad
    ):
        raise A1TorchBackendError("backend output metadata changed")
    output_value = output.item()
    if type(output_value) is not int or output_value not in (0, 1):
        raise A1TorchBackendError("backend output is not an exact bit")
    return output_value


def _trace_values(
    trace: _TorchTrace,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return tuple(tuple(layer.tolist()) for layer in trace.layers)  # type: ignore[return-value]


def _certify_profile(entry: _BackendEntry) -> None:
    profile = entry.profile
    base = [mod_q(anchor + A0_CENTER) for anchor in profile.anchors]
    layer2_min = [_INT32_MAX] * A1_LAYER_WIDTHS[1]
    layer2_max = [_INT32_MIN] * A1_LAYER_WIDTHS[1]

    for component in range(A0_COMPONENT_COUNT):
        for coefficient in range(A0_MODULUS):
            coefficients = base.copy()
            coefficients[component] = coefficient
            canonical = tuple(coefficients)
            expected = _exact_profile_trace(canonical, profile)
            observed = _trace_values(_tensor_trace(canonical, entry.module))
            if observed != expected:
                raise A1TorchBackendError("profile differential startup gate failed")
            for index, value in enumerate(observed[1]):
                layer2_min[index] = min(layer2_min[index], value)
                layer2_max[index] = max(layer2_max[index], value)

    expected_layer2_max = [9 if index % 2 == 0 else 8 for index in range(A1_LAYER_WIDTHS[1])]
    if layer2_min != [0] * A1_LAYER_WIDTHS[1] or layer2_max != expected_layer2_max:
        raise A1TorchBackendError("layer 2 semantic range certificate failed")
    semantic_bounds = tuple((0, maximum) for maximum in expected_layer2_max)
    _, layer3_affine = _interval_layer(semantic_bounds, profile.layers[2])
    if layer3_affine != ((-71, 65),):
        raise A1TorchBackendError("layer 3 range certificate failed")

    for pass_count in range(A0_COMPONENT_COUNT + 1):
        offsets = (0,) * pass_count + (-9,) * (A0_COMPONENT_COUNT - pass_count)
        and_coefficients = tuple(
            mod_q(profile.anchors[index] + A0_CENTER + offsets[index])
            for index in range(A0_COMPONENT_COUNT)
        )
        observed = _trace_values(_tensor_trace(and_coefficients, entry.module))
        if observed != _exact_profile_trace(and_coefficients, profile):
            raise A1TorchBackendError("AND differential startup gate failed")
        if observed[-1] != (int(pass_count == A0_COMPONENT_COUNT),):
            raise A1TorchBackendError("AND semantic startup gate failed")


def _evaluate_torch_with_trace(
    coefficients: tuple[int, ...],
    backend: A1TorchBackend,
    profile_id: int,
    slot_id: int,
) -> _TorchTrace:
    if type(backend) is not A1TorchBackend or not backend.active:
        raise A1TorchBackendError("backend is not active")
    if type(coefficients) is not tuple or len(coefficients) != A1_INPUT_WIDTH:
        raise A1TorchBackendError("trace input has the wrong shape")
    if any(type(value) is not int or not 0 <= value < A0_MODULUS for value in coefficients):
        raise A1TorchBackendError("trace input contains a non-canonical coefficient")
    _validate_environment()
    entry = backend._lookup(profile_id, slot_id)
    _validate_module(entry)
    return _tensor_trace(coefficients, entry.module)


def compile_a1_torch_backend(registry: A1CompiledRegistry) -> A1TorchBackend:
    """从可信 compiled registry 构建并完整激活内存态 CPU backend。"""
    return A1TorchBackend(registry)


def verify_a1_torch(raw_credential: bytes, backend: A1TorchBackend) -> A1Evidence:
    """用 A1-B1 验证原始 credential, 并只返回无授权能力的 evidence。"""
    try:
        credential = parse_credential(raw_credential)
    except CredentialParseError:
        return A1Evidence(A1EvidenceCode.PARSE_REJECT)

    if type(backend) is not A1TorchBackend or not backend.active:
        return A1Evidence(A1EvidenceCode.CONFIG_REJECT)
    try:
        _validate_environment()
        entry = backend._lookup(credential.profile_id, credential.slot_id)
    except A1ProfileLookupError:
        return A1Evidence(A1EvidenceCode.PROFILE_REJECT)
    except Exception:
        backend._disable()
        return A1Evidence(A1EvidenceCode.CONFIG_REJECT)

    try:
        _validate_module(entry)
        activation = torch.tensor(credential.b, dtype=torch.int32, device="cpu")
        _validate_input_tensor(activation)
        with torch.inference_mode():
            output = entry.module(activation)
        output_value = _validate_output_tensor(output)
    except Exception:
        backend._disable()
        return A1Evidence(A1EvidenceCode.CONFIG_REJECT)

    code = A1EvidenceCode.NUMERIC_ACCEPT if output_value == 1 else A1EvidenceCode.NUMERIC_REJECT
    return A1Evidence(code)
