"""A4-C1 canonical `(y,z)` 固定整数 ReLU verifier。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from can.reference.a4 import (
    A4_MODULUS,
    A4_SYNDROME_SIZE,
    A4_VECTOR_SIZE,
    A4MessageValidationError,
    A4ProofParseError,
    A4PublicProfile,
    hash_to_a4_syndrome,
    parse_a4_proof,
)

A4_NEURAL_CANDIDATE_ID: Final = "CAN-RELU-A4-PFDH-TOY-v1"
A4_NEURAL_SCALE: Final = 1
A4_NEURAL_INPUT_WIDTH: Final = A4_SYNDROME_SIZE + A4_VECTOR_SIZE
A4_NEURAL_LAYER_WIDTHS: Final = (3600, 1153, 1)
A4_MULTIPLE_MIN: Final = -72
A4_MULTIPLE_MAX: Final = 71
A4_MULTIPLES: Final = tuple(range(A4_MULTIPLE_MIN, A4_MULTIPLE_MAX + 1))

_NORM_UNIT_COUNT: Final = 2 * A4_VECTOR_SIZE
_PULSE_COUNT: Final = A4_SYNDROME_SIZE * len(A4_MULTIPLES)
_H1_PULSE_OFFSET: Final = _NORM_UNIT_COUNT
_H2_NORM_INDEX: Final = _PULSE_COUNT
_INT32_MIN: Final = -(2**31)
_INT32_MAX: Final = 2**31 - 1
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1


class A4NeuralProfileError(ValueError):
    """表示 A4-C1 compiled profile 不满足固定 graph 契约。"""


class A4NeuralEvaluationError(RuntimeError):
    """表示 A4-C1 graph 无法在精确范围内执行。"""


class A4NeuralEvidenceCode(StrEnum):
    """定义 A4-C1 verifier 的稳定、无授权能力 evidence 码。"""

    INPUT_REJECT = "input_reject"
    CONFIG_REJECT = "config_reject"
    NEURAL_REJECT = "neural_reject"
    NEURAL_ACCEPT = "neural_accept"


@dataclass(frozen=True, slots=True)
class _SparseRow:
    terms: tuple[tuple[int, int], ...]
    bias: int


@dataclass(frozen=True, slots=True)
class A4SparseAffineReluLayer:
    """保存 A4-C1 固定 sparse affine/ReLU 层的数学结构。"""

    input_width: int
    rows: tuple[_SparseRow, ...] = field(repr=False)

    @property
    def output_width(self) -> int:
        """返回该层的固定输出宽度。"""
        return len(self.rows)


@dataclass(frozen=True, slots=True, init=False)
class A4NeuralProfile:
    """保存由可信 A4 public profile 编译出的不可变 A4-C1 graph。"""

    candidate_id: str
    profile_id: int
    identity_id: bytes
    scale: int
    public_profile: A4PublicProfile = field(repr=False)
    layers: tuple[A4SparseAffineReluLayer, ...] = field(repr=False)

    def __init__(self, public_profile: A4PublicProfile) -> None:
        if type(public_profile) is not A4PublicProfile:
            raise A4NeuralProfileError("A4-C1 profile requires the exact public profile type")
        layers = _build_layers(public_profile)
        object.__setattr__(self, "candidate_id", A4_NEURAL_CANDIDATE_ID)
        object.__setattr__(self, "profile_id", public_profile.profile_id)
        object.__setattr__(self, "identity_id", public_profile.identity_id)
        object.__setattr__(self, "scale", A4_NEURAL_SCALE)
        object.__setattr__(self, "public_profile", public_profile)
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True, slots=True)
class A4NeuralEvidence:
    """保存 A4-C1 判定, 不携带 gate、decision 或 capability。"""

    code: A4NeuralEvidenceCode
    identity_id: bytes
    profile_id: int
    message_sha256: bytes

    def __post_init__(self) -> None:
        if type(self.code) is not A4NeuralEvidenceCode:
            raise TypeError("A4 neural evidence code must use the exact enum type")
        if type(self.identity_id) is not bytes or len(self.identity_id) != 32:
            raise TypeError("A4 neural evidence identity must be exactly 32 bytes")
        if type(self.profile_id) is not int or self.profile_id != 1:
            raise TypeError("A4 neural evidence profile must be exactly 1")
        if type(self.message_sha256) is not bytes or len(self.message_sha256) != 32:
            raise TypeError("A4 neural evidence digest must be exactly 32 bytes")

    @property
    def accepted(self) -> bool:
        """返回 neural relation 是否接受, 不提交任何授权。"""
        return self.code is A4NeuralEvidenceCode.NEURAL_ACCEPT


def _require_int32(value: int) -> int:
    if type(value) is not int or not _INT32_MIN <= value <= _INT32_MAX:
        raise A4NeuralEvaluationError("A4-C1 value exceeds exact int32 semantics")
    return value


def _require_int64(value: int) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        raise A4NeuralEvaluationError("A4-C1 accumulator exceeds exact int64 semantics")
    return value


def _relu(value: int) -> int:
    return value if value > 0 else 0


def _validate_sparse_layer(layer: A4SparseAffineReluLayer) -> None:
    if type(layer) is not A4SparseAffineReluLayer:
        raise A4NeuralProfileError("A4 layers must use the exact sparse layer type")
    if type(layer.input_width) is not int or layer.input_width <= 0:
        raise A4NeuralProfileError("A4 layer input width is not positive")
    if not layer.rows:
        raise A4NeuralProfileError("A4 layer must have at least one row")
    for row in layer.rows:
        if type(row) is not _SparseRow:
            raise A4NeuralProfileError("A4 layer rows are not canonical")
        if type(row.bias) is not int or not _INT32_MIN <= row.bias <= _INT32_MAX:
            raise A4NeuralProfileError("A4 layer bias is not exact int32")
        previous_index = -1
        for index, weight in row.terms:
            if type(index) is not int or not 0 <= index < layer.input_width:
                raise A4NeuralProfileError("A4 sparse term index is outside the layer")
            if index <= previous_index:
                raise A4NeuralProfileError("A4 sparse term indexes must be unique and ordered")
            if type(weight) is not int or not _INT32_MIN <= weight <= _INT32_MAX or weight == 0:
                raise A4NeuralProfileError("A4 sparse term weight is not canonical int32")
            previous_index = index


def _make_row(terms: Iterable[tuple[int, int]], bias: int) -> _SparseRow:
    canonical_terms = tuple(terms)
    if type(bias) is not int or not _INT32_MIN <= bias <= _INT32_MAX:
        raise A4NeuralProfileError("A4 compiled bias is outside int32")
    return _SparseRow(canonical_terms, bias)


def _build_layers(profile: A4PublicProfile) -> tuple[A4SparseAffineReluLayer, ...]:
    layer1_rows: list[_SparseRow] = []

    for index in range(A4_VECTOR_SIZE):
        input_index = A4_SYNDROME_SIZE + index
        layer1_rows.append(_make_row(((input_index, 1),), -1))
        layer1_rows.append(_make_row(((input_index, -1),), -1))

    residual_terms_by_row: list[tuple[tuple[int, int], ...]] = []
    for row_index, row in enumerate(profile.matrix):
        residual_terms_by_row.append(
            (
                (row_index, -1),
                *(
                    (A4_SYNDROME_SIZE + column, coefficient)
                    for column, coefficient in enumerate(row)
                    if coefficient != 0
                ),
            )
        )

    for row_index in range(A4_SYNDROME_SIZE):
        residual_terms = residual_terms_by_row[row_index]
        for multiple in A4_MULTIPLES:
            center = multiple * A4_MODULUS
            layer1_rows.append(_make_row(residual_terms, -center + 1))
            layer1_rows.append(_make_row(residual_terms, -center))
            layer1_rows.append(_make_row(residual_terms, -center - 1))

    layer2_rows: list[_SparseRow] = []
    for pulse_index in range(_PULSE_COUNT):
        start = _H1_PULSE_OFFSET + pulse_index * 3
        layer2_rows.append(_make_row(((start, 1), (start + 1, -2), (start + 2, 1)), 0))
    layer2_rows.append(_make_row(((index, 1) for index in range(_NORM_UNIT_COUNT)), 0))

    layer3_rows = (
        _make_row(
            (*(tuple((index, 1) for index in range(_PULSE_COUNT))), (_H2_NORM_INDEX, -1)),
            -7,
        ),
    )
    layers = (
        A4SparseAffineReluLayer(A4_NEURAL_INPUT_WIDTH, tuple(layer1_rows)),
        A4SparseAffineReluLayer(A4_NEURAL_LAYER_WIDTHS[0], tuple(layer2_rows)),
        A4SparseAffineReluLayer(A4_NEURAL_LAYER_WIDTHS[1], layer3_rows),
    )
    for layer in layers:
        _validate_sparse_layer(layer)
    if tuple((layer.input_width, layer.output_width) for layer in layers) != (
        (A4_NEURAL_INPUT_WIDTH, A4_NEURAL_LAYER_WIDTHS[0]),
        (A4_NEURAL_LAYER_WIDTHS[0], A4_NEURAL_LAYER_WIDTHS[1]),
        (A4_NEURAL_LAYER_WIDTHS[1], A4_NEURAL_LAYER_WIDTHS[2]),
    ):
        raise A4NeuralProfileError("A4-C1 graph topology is not frozen")
    return layers


def compile_a4_neural_profile(profile: A4PublicProfile) -> A4NeuralProfile:
    """从可信公开 profile 编译不可变的 A4-C1 exact graph。"""
    if type(profile) is not A4PublicProfile:
        raise A4NeuralProfileError("A4-C1 compiler requires the exact public profile type")
    return A4NeuralProfile(profile)


def _canonical_core_input(y: tuple[int, ...], z: tuple[int, ...]) -> tuple[int, ...]:
    if type(y) is not tuple or len(y) != A4_SYNDROME_SIZE:
        raise A4NeuralEvaluationError("A4-C1 y input has the wrong shape")
    if type(z) is not tuple or len(z) != A4_VECTOR_SIZE:
        raise A4NeuralEvaluationError("A4-C1 z input has the wrong shape")
    if any(type(value) is not int or not 0 <= value < A4_MODULUS for value in y):
        raise A4NeuralEvaluationError("A4-C1 y input is not canonical int32")
    if any(type(value) is not int or not -128 <= value <= 127 for value in z):
        raise A4NeuralEvaluationError("A4-C1 z input is not canonical signed int8")
    return y + z


def _affine_relu(inputs: tuple[int, ...], layer: A4SparseAffineReluLayer) -> tuple[int, ...]:
    if len(inputs) != layer.input_width:
        raise A4NeuralEvaluationError("A4-C1 layer input has the wrong width")
    outputs: list[int] = []
    for row in layer.rows:
        accumulator = _require_int64(row.bias)
        for index, weight in row.terms:
            accumulator = _require_int64(accumulator + weight * inputs[index])
        output = _relu(accumulator)
        outputs.append(_require_int32(output))
    return tuple(outputs)


def _run_a4_graph(
    y: tuple[int, ...],
    z: tuple[int, ...],
    profile: A4NeuralProfile,
) -> tuple[tuple[int, ...], ...]:
    if type(profile) is not A4NeuralProfile:
        raise A4NeuralEvaluationError("A4-C1 graph requires the exact compiled profile type")
    values = _canonical_core_input(y, z)
    outputs: list[tuple[int, ...]] = []
    for layer in profile.layers:
        values = _affine_relu(values, layer)
        outputs.append(values)
    return tuple(outputs)


def _evaluate_a4_core(
    y: tuple[int, ...],
    z: tuple[int, ...],
    profile: A4NeuralProfile,
) -> int:
    output = _run_a4_graph(y, z, profile)[-1]
    if len(output) != 1 or output[0] not in (0, 1):
        raise A4NeuralEvaluationError("A4-C1 output is not an exact decision bit")
    return output[0]


def verify_a4_neural(
    raw_message: object,
    raw_proof: object,
    profile: object,
) -> A4NeuralEvidence:
    """执行 A4-C1 neural relation 并只返回绑定 evidence。"""
    if type(profile) is not A4NeuralProfile:
        return A4NeuralEvidence(
            A4NeuralEvidenceCode.CONFIG_REJECT,
            bytes(32),
            1,
            bytes(32),
        )
    identity_id = profile.identity_id
    message_digest = (
        hashlib.sha256(raw_message).digest() if type(raw_message) is bytes else bytes(32)
    )
    try:
        proof = parse_a4_proof(raw_proof)
        target = hash_to_a4_syndrome(raw_message, proof.salt, profile.public_profile)
    except (A4MessageValidationError, A4ProofParseError, TypeError, ValueError):
        return A4NeuralEvidence(
            A4NeuralEvidenceCode.INPUT_REJECT,
            identity_id,
            profile.profile_id,
            message_digest,
        )
    try:
        output = _evaluate_a4_core(target, proof.vector, profile)
    except A4NeuralEvaluationError:
        return A4NeuralEvidence(
            A4NeuralEvidenceCode.CONFIG_REJECT,
            identity_id,
            profile.profile_id,
            message_digest,
        )
    code = A4NeuralEvidenceCode.NEURAL_ACCEPT if output == 1 else A4NeuralEvidenceCode.NEURAL_REJECT
    return A4NeuralEvidence(code, identity_id, profile.profile_id, message_digest)


__all__ = [
    "A4_MULTIPLES",
    "A4_NEURAL_CANDIDATE_ID",
    "A4_NEURAL_INPUT_WIDTH",
    "A4_NEURAL_LAYER_WIDTHS",
    "A4_NEURAL_SCALE",
    "A4NeuralEvaluationError",
    "A4NeuralEvidence",
    "A4NeuralEvidenceCode",
    "A4NeuralProfile",
    "A4NeuralProfileError",
    "A4SparseAffineReluLayer",
    "compile_a4_neural_profile",
    "verify_a4_neural",
]
