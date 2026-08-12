"""A1-C1 固定整数 ReLU verifier 的 dependency-free conformance backend。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from can.reference import (
    A0_COMPONENT_COUNT,
    A0_MODULUS,
    A0_PROFILE_ID,
    A0_SECRET_SIZE,
    A0Slot,
    CredentialParseError,
    mod_q,
    parse_credential,
)

A1_CANDIDATE_ID: Final = "CAN-RELU-EXACT-v1"
A1_SCALE: Final = 1
A1_INPUT_WIDTH: Final = A0_COMPONENT_COUNT
A1_LAYER_WIDTHS: Final = (40, 16, 1)
A1_DISTANCE_THRESHOLD: Final = 8

_INT32_MIN: Final = -(2**31)
_INT32_MAX: Final = 2**31 - 1
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1
_DISTANCE_UNITS_PER_COMPONENT: Final = 5
_THRESHOLD_UNITS_PER_COMPONENT: Final = 2

Matrix = tuple[tuple[int, ...], ...]


class A1ProfileValidationError(ValueError):
    """表示本地 A1 compiled profile 不满足固定构造约束。"""


class A1ProfileLookupError(LookupError):
    """表示 credential 未解析到唯一启用的 A1 compiled profile。"""


class A1EvaluationError(RuntimeError):
    """表示固定整数 graph 无法在已证明的数值边界内执行。"""


@dataclass(frozen=True, slots=True, init=False)
class A1AffineReluLayer:
    """保存一个经过精确类型和 shape 校验的不可变 affine/ReLU 层。"""

    weights: Matrix
    bias: tuple[int, ...]

    def __init__(
        self,
        weights: Iterable[Iterable[int]],
        bias: Iterable[int],
    ) -> None:
        try:
            canonical_weights = tuple(tuple(row) for row in weights)
            canonical_bias = tuple(bias)
        except TypeError as error:
            raise A1ProfileValidationError("layer values must be iterable") from error

        if not canonical_weights or not canonical_weights[0]:
            raise A1ProfileValidationError("layer must have non-empty input and output widths")
        input_width = len(canonical_weights[0])
        if any(len(row) != input_width for row in canonical_weights):
            raise A1ProfileValidationError("layer weight rows must have one input width")
        if len(canonical_bias) != len(canonical_weights):
            raise A1ProfileValidationError("layer bias has the wrong output width")
        if any(
            type(value) is not int or not _INT32_MIN <= value <= _INT32_MAX
            for row in canonical_weights
            for value in row
        ):
            raise A1ProfileValidationError("layer weights must be exact int32 values")
        if any(
            type(value) is not int or not _INT32_MIN <= value <= _INT32_MAX
            for value in canonical_bias
        ):
            raise A1ProfileValidationError("layer bias must contain exact int32 values")

        object.__setattr__(self, "weights", canonical_weights)
        object.__setattr__(self, "bias", canonical_bias)

    @property
    def input_width(self) -> int:
        """返回该层固定的输入宽度。"""
        return len(self.weights[0])

    @property
    def output_width(self) -> int:
        """返回该层固定的输出宽度。"""
        return len(self.weights)


def _single_input_row(width: int, index: int, coefficient: int) -> tuple[int, ...]:
    return tuple(coefficient if column == index else 0 for column in range(width))


def _build_fixed_layers(anchors: tuple[int, ...]) -> tuple[A1AffineReluLayer, ...]:
    layer1_weights: list[tuple[int, ...]] = []
    layer1_bias: list[int] = []
    for component, anchor in enumerate(anchors):
        layer1_weights.extend(
            (
                _single_input_row(A1_INPUT_WIDTH, component, -1),
                _single_input_row(A1_INPUT_WIDTH, component, 1),
                _single_input_row(A1_INPUT_WIDTH, component, 1),
                _single_input_row(A1_INPUT_WIDTH, component, 1),
                _single_input_row(A1_INPUT_WIDTH, component, 1),
            )
        )
        layer1_bias.extend((anchor, 129 - anchor, 1 - anchor, -anchor, -128 - anchor))

    distance_coefficients = (-1, -2, 1, 2, -2)
    layer2_weights: list[tuple[int, ...]] = []
    layer2_bias: list[int] = []
    for component in range(A0_COMPONENT_COUNT):
        start = component * _DISTANCE_UNITS_PER_COMPONENT
        row = tuple(
            distance_coefficients[column - start] if start <= column < start + 5 else 0
            for column in range(A1_LAYER_WIDTHS[0])
        )
        layer2_weights.extend((row, row))
        layer2_bias.extend((138, 137))

    layer3_weights = (tuple(1 if column % 2 == 0 else -1 for column in range(A1_LAYER_WIDTHS[1])),)
    layer3_bias = (-7,)

    return (
        A1AffineReluLayer(layer1_weights, layer1_bias),
        A1AffineReluLayer(layer2_weights, layer2_bias),
        A1AffineReluLayer(layer3_weights, layer3_bias),
    )


@dataclass(frozen=True, slots=True, init=False)
class A1CompiledProfile:
    """保存一个固定候选、不可训练且不可变的本地 A1 profile。"""

    candidate_id: str
    profile_id: int
    slot_id: int
    scale: int
    anchors: tuple[int, ...] = field(repr=False)
    layers: tuple[A1AffineReluLayer, ...] = field(repr=False)

    def __init__(self, slot_id: int, anchors: Iterable[int]) -> None:
        if type(slot_id) is not int or not 0 <= slot_id <= 0xFFFFFFFF:
            raise A1ProfileValidationError("slot_id is not a canonical uint32")
        try:
            canonical_anchors = tuple(anchors)
        except TypeError as error:
            raise A1ProfileValidationError("anchors must be iterable") from error
        if len(canonical_anchors) != A0_COMPONENT_COUNT:
            raise A1ProfileValidationError("anchors have the wrong component count")
        if any(
            type(anchor) is not int or not 0 <= anchor < A0_MODULUS for anchor in canonical_anchors
        ):
            raise A1ProfileValidationError("anchors contain a non-canonical coefficient")

        layers = _build_fixed_layers(canonical_anchors)
        if tuple((layer.input_width, layer.output_width) for layer in layers) != (
            (A1_INPUT_WIDTH, A1_LAYER_WIDTHS[0]),
            (A1_LAYER_WIDTHS[0], A1_LAYER_WIDTHS[1]),
            (A1_LAYER_WIDTHS[1], A1_LAYER_WIDTHS[2]),
        ):
            raise A1ProfileValidationError("compiled graph has the wrong topology")

        object.__setattr__(self, "candidate_id", A1_CANDIDATE_ID)
        object.__setattr__(self, "profile_id", A0_PROFILE_ID)
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "scale", A1_SCALE)
        object.__setattr__(self, "anchors", canonical_anchors)
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True, slots=True, init=False)
class A1CompiledRegistry:
    """提供不可变、无 profile 或 slot 回退的 A1 compiled profile 查询。"""

    _profiles: Mapping[tuple[int, int], A1CompiledProfile] = field(repr=False)

    def __init__(self, profiles: Iterable[A1CompiledProfile]) -> None:
        try:
            profile_iterator = iter(profiles)
        except TypeError as error:
            raise A1ProfileValidationError("profiles must be iterable") from error

        indexed_profiles: dict[tuple[int, int], A1CompiledProfile] = {}
        for profile in profile_iterator:
            if type(profile) is not A1CompiledProfile:
                raise A1ProfileValidationError("registry entries must be exactly A1CompiledProfile")
            key = (profile.profile_id, profile.slot_id)
            if key in indexed_profiles:
                raise A1ProfileValidationError("registry contains a duplicate profile and slot")
            indexed_profiles[key] = profile

        object.__setattr__(self, "_profiles", MappingProxyType(indexed_profiles))

    @property
    def profiles(self) -> tuple[A1CompiledProfile, ...]:
        """返回不会暴露内部映射的不可变 compiled profile 快照。"""
        return tuple(self._profiles.values())

    def lookup(self, profile_id: int, slot_id: int) -> A1CompiledProfile:
        """按固定 profile 与 slot 查找本地构造。未知项不执行回退。"""
        if type(profile_id) is not int or profile_id != A0_PROFILE_ID:
            raise A1ProfileLookupError("unknown A1 profile")
        if type(slot_id) is not int:
            raise A1ProfileLookupError("slot_id must be exactly int")
        profile = self._profiles.get((profile_id, slot_id))
        if profile is None:
            raise A1ProfileLookupError("unknown A1 slot")
        return profile


class A1EvidenceCode(StrEnum):
    """定义 A1 adapter 的稳定、无授权能力证据码。"""

    PARSE_REJECT = "parse_reject"
    PROFILE_REJECT = "profile_reject"
    CONFIG_REJECT = "config_reject"
    NUMERIC_ACCEPT = "numeric_accept"
    NUMERIC_REJECT = "numeric_reject"


@dataclass(frozen=True, slots=True)
class A1Evidence:
    """记录不携带距离、gate、decision 或 capability 的 A1 数值判定。"""

    code: A1EvidenceCode

    def __post_init__(self) -> None:
        if type(self.code) is not A1EvidenceCode:
            raise TypeError("code must be exactly A1EvidenceCode")

    @property
    def accepted(self) -> bool:
        """返回不提交任何授权的数值 relation 结果。"""
        return self.code is A1EvidenceCode.NUMERIC_ACCEPT


@dataclass(frozen=True, slots=True)
class _A1Trace:
    layer1: tuple[int, ...]
    layer2: tuple[int, ...]
    distances: tuple[int, ...]
    gates: tuple[int, ...]
    output: int


def compile_a1_profile(slot: A0Slot, s_test: Sequence[int]) -> A1CompiledProfile:
    """从可信 A0 slot 和 toy secret 编译固定且不可变的 A1 profile。"""
    if type(slot) is not A0Slot or not slot.enabled:
        raise A1ProfileValidationError("slot must be an enabled exact A0Slot")
    try:
        secret = tuple(s_test)
    except TypeError as error:
        raise A1ProfileValidationError("s_test must be an iterable binary vector") from error
    if len(secret) != A0_SECRET_SIZE or any(
        type(coefficient) is not int or coefficient not in (0, 1) for coefficient in secret
    ):
        raise A1ProfileValidationError("s_test must contain exactly 32 binary integers")

    anchors: list[int] = []
    for row in slot.matrix:
        accumulator = 0
        for coefficient, secret_bit in zip(row, secret, strict=True):
            accumulator += coefficient * secret_bit
            if not _INT64_MIN <= accumulator <= _INT64_MAX:
                raise A1ProfileValidationError("profile compiler accumulator exceeds int64")
        anchors.append(mod_q(accumulator))
    return A1CompiledProfile(slot.slot_id, anchors)


def _canonical_coefficients(coefficients: tuple[int, ...]) -> tuple[int, ...]:
    if type(coefficients) is not tuple or len(coefficients) != A1_INPUT_WIDTH:
        raise A1EvaluationError("core input must be one exact eight-component tuple")
    if any(type(value) is not int or not 0 <= value < A0_MODULUS for value in coefficients):
        raise A1EvaluationError("core input contains a non-canonical coefficient")
    return coefficients


def _require_int32(value: int) -> int:
    if type(value) is not int or not _INT32_MIN <= value <= _INT32_MAX:
        raise A1EvaluationError("affine value exceeds exact int32 semantics")
    return value


def _relu(value: int) -> int:
    return value if value > 0 else 0


def _affine_relu(
    inputs: tuple[int, ...],
    layer: A1AffineReluLayer,
) -> tuple[int, ...]:
    if len(inputs) != layer.input_width:
        raise A1EvaluationError("layer input has the wrong width")

    outputs: list[int] = []
    for row, bias in zip(layer.weights, layer.bias, strict=True):
        accumulator = _require_int32(bias)
        for weight, value in zip(row, inputs, strict=True):
            product = _require_int32(weight * value)
            accumulator = _require_int32(accumulator + product)
        outputs.append(_relu(accumulator))
    return tuple(outputs)


def _run_graph(
    coefficients: tuple[int, ...],
    profile: A1CompiledProfile,
) -> tuple[tuple[int, ...], ...]:
    if type(profile) is not A1CompiledProfile:
        raise A1EvaluationError("profile must be exactly A1CompiledProfile")
    values = _canonical_coefficients(coefficients)
    outputs: list[tuple[int, ...]] = []
    for layer in profile.layers:
        values = _affine_relu(values, layer)
        outputs.append(values)
    return tuple(outputs)


def _evaluate_core(coefficients: tuple[int, ...], profile: A1CompiledProfile) -> int:
    output = _run_graph(coefficients, profile)[-1]
    if len(output) != 1 or output[0] not in (0, 1):
        raise A1EvaluationError("fixed graph output is not an exact decision bit")
    return output[0]


def _evaluate_with_trace(
    coefficients: tuple[int, ...],
    profile: A1CompiledProfile,
) -> _A1Trace:
    layer1, layer2, layer3 = _run_graph(coefficients, profile)
    distances = tuple(
        _require_int32(
            -129
            + layer1[index]
            + 2 * layer1[index + 1]
            - layer1[index + 2]
            - 2 * layer1[index + 3]
            + 2 * layer1[index + 4]
        )
        for index in range(0, A1_LAYER_WIDTHS[0], _DISTANCE_UNITS_PER_COMPONENT)
    )
    gates = tuple(
        layer2[index] - layer2[index + 1]
        for index in range(0, A1_LAYER_WIDTHS[1], _THRESHOLD_UNITS_PER_COMPONENT)
    )
    if any(not 0 <= distance <= 128 for distance in distances):
        raise A1EvaluationError("distance trace is outside the proved range")
    if any(gate not in (0, 1) for gate in gates):
        raise A1EvaluationError("threshold trace is not an exact bit")
    if len(layer3) != 1 or layer3[0] not in (0, 1):
        raise A1EvaluationError("AND trace is not an exact bit")
    return _A1Trace(layer1, layer2, distances, gates, layer3[0])


def verify_a1(raw_credential: bytes, registry: A1CompiledRegistry) -> A1Evidence:
    """验证 A1 credential 并只返回不可提交授权的结构化 evidence。"""
    try:
        credential = parse_credential(raw_credential)
    except CredentialParseError:
        return A1Evidence(A1EvidenceCode.PARSE_REJECT)

    if type(registry) is not A1CompiledRegistry:
        return A1Evidence(A1EvidenceCode.CONFIG_REJECT)
    try:
        profile = registry.lookup(credential.profile_id, credential.slot_id)
    except A1ProfileLookupError:
        return A1Evidence(A1EvidenceCode.PROFILE_REJECT)

    try:
        output = _evaluate_core(credential.b, profile)
    except Exception:
        return A1Evidence(A1EvidenceCode.CONFIG_REJECT)
    code = A1EvidenceCode.NUMERIC_ACCEPT if output == 1 else A1EvidenceCode.NUMERIC_REJECT
    return A1Evidence(code)
