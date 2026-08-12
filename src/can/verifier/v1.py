"""V1-C1 coefficient-domain Module-SIS 固定整数 ReLU verifier。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from can.reference.v1 import (
    V1_CHALLENGE_WEIGHT,
    V1_MODULE_COLUMNS,
    V1_MODULE_ROWS,
    V1_MODULUS,
    V1_RESPONSE_BOUND,
    V1_RESPONSE_POLYNOMIALS,
    V1_RING_DEGREE,
    V1Challenge,
    V1Commitment,
    V1PublicProfile,
    V1Response,
    V1WireParseError,
    parse_v1_challenge,
    parse_v1_commitment,
    parse_v1_response,
)

V1_NEURAL_CANDIDATE_ID: Final = "CAN-RELU-V1-MSIS-COEFF-v1"
V1_NEURAL_SCALE: Final = 1
V1_NEURAL_RESPONSE_INPUT_BOUND: Final = V1_RESPONSE_BOUND + 1
V1_NEURAL_INPUT_WIDTH: Final = (
    V1_MODULE_ROWS * V1_RING_DEGREE + V1_RING_DEGREE + V1_RESPONSE_POLYNOMIALS * V1_RING_DEGREE
)
V1_NEURAL_NORM_UNITS: Final = V1_RESPONSE_POLYNOMIALS * V1_RING_DEGREE * 2
V1_NEURAL_RESIDUAL_COUNT: Final = V1_MODULE_ROWS * V1_RING_DEGREE

# The complete canonical input range gives a finite, auditable ledger. The
# convolution sums use A coefficients in [0,q-1], z in [-B,B] and N terms.
V1_CONVOLUTION_ACCUMULATOR_BOUND: Final = (
    V1_RING_DEGREE * (V1_MODULUS - 1) * V1_NEURAL_RESPONSE_INPUT_BOUND
)
V1_RESIDUAL_BOUND: Final = (
    V1_MODULE_COLUMNS * V1_CONVOLUTION_ACCUMULATOR_BOUND
    + V1_NEURAL_RESPONSE_INPUT_BOUND
    + V1_MODULUS
    - 1
    + V1_CHALLENGE_WEIGHT * (V1_MODULUS - 1)
)
V1_RESIDUAL_MULTIPLE_MIN: Final = -(V1_RESIDUAL_BOUND // V1_MODULUS)
V1_RESIDUAL_MULTIPLE_MAX: Final = V1_RESIDUAL_BOUND // V1_MODULUS
V1_RESIDUAL_MULTIPLES: Final = tuple(range(V1_RESIDUAL_MULTIPLE_MIN, V1_RESIDUAL_MULTIPLE_MAX + 1))
V1_RESIDUAL_PULSE_COUNT: Final = V1_NEURAL_RESIDUAL_COUNT * len(V1_RESIDUAL_MULTIPLES)
V1_NEURAL_PULSE_COUNT: Final = V1_RESIDUAL_PULSE_COUNT
V1_NEURAL_LAYER_WIDTHS: Final = (
    V1_RESIDUAL_PULSE_COUNT * 3 + V1_NEURAL_NORM_UNITS,
    V1_NEURAL_RESIDUAL_COUNT + 1,
    1,
)
V1_NEURAL_LAYER2_WIDTH: Final = V1_NEURAL_LAYER_WIDTHS[1]
V1_CONVOLUTION_ACCUMULATOR_MAX: Final = V1_MODULE_COLUMNS * V1_CONVOLUTION_ACCUMULATOR_BOUND
V1_RESIDUAL_ABS_MAX: Final = V1_RESIDUAL_BOUND
V1_PULSE_INPUT_MAX: Final = V1_RESIDUAL_BOUND + V1_RESIDUAL_MULTIPLE_MAX * V1_MODULUS + 1
V1_PULSE_ACCUMULATOR_MAX: Final = 4 * V1_PULSE_INPUT_MAX
V1_NORM_VIOLATION_MAX: Final = V1_NEURAL_RESPONSE_INPUT_BOUND - V1_RESPONSE_BOUND
V1_FINAL_ACCUMULATOR_MIN: Final = -(
    (V1_NEURAL_NORM_UNITS // 2) * V1_NORM_VIOLATION_MAX + V1_NEURAL_RESIDUAL_COUNT - 1
)
V1_FINAL_ACCUMULATOR_MAX: Final = 1

_INT32_MIN: Final = -(2**31)
_INT32_MAX: Final = 2**31 - 1
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1


class V1NeuralProfileError(ValueError):
    """表示 V1-C1 compiled graph 不满足固定契约。"""


class V1NeuralEvaluationError(RuntimeError):
    """表示 V1-C1 graph 无法在精确范围内执行。"""


class V1NeuralEvidenceCode(StrEnum):
    """定义 V1-C1 的稳定、无授权能力 evidence 码。"""

    INPUT_REJECT = "input_reject"
    CONFIG_REJECT = "config_reject"
    NEURAL_REJECT = "neural_reject"
    NEURAL_ACCEPT = "neural_accept"


@dataclass(frozen=True, slots=True)
class V1SparseRow:
    """保存固定 sparse affine 行。"""

    terms: tuple[tuple[int, int], ...]
    bias: int


@dataclass(frozen=True, slots=True)
class V1SparseAffineReluLayer:
    """保存 V1-C1 固定 sparse affine/ReLU 层。"""

    input_width: int
    rows: tuple[V1SparseRow, ...] = field(repr=False)

    @property
    def output_width(self) -> int:
        """返回固定输出宽度。"""
        return len(self.rows)


@dataclass(frozen=True, slots=True, init=False)
class V1NeuralProfile:
    """保存由可信 V1 public profile 编译出的不可变 graph。"""

    candidate_id: str
    profile_id: int
    identity_id: bytes
    scale: int
    public_profile: V1PublicProfile = field(repr=False)
    layers: tuple[V1SparseAffineReluLayer, ...] = field(repr=False)

    def __init__(self, public_profile: V1PublicProfile) -> None:
        if type(public_profile) is not V1PublicProfile:
            raise V1NeuralProfileError("V1-C1 requires the exact public profile type")
        layers = _build_layers(public_profile)
        object.__setattr__(self, "candidate_id", V1_NEURAL_CANDIDATE_ID)
        object.__setattr__(self, "profile_id", public_profile.profile_id)
        object.__setattr__(self, "identity_id", public_profile.identity_id)
        object.__setattr__(self, "scale", V1_NEURAL_SCALE)
        object.__setattr__(self, "public_profile", public_profile)
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True, slots=True)
class V1NeuralEvidence:
    """保存不具有授权能力的 V1-C1 判定摘要。"""

    code: V1NeuralEvidenceCode
    identity_id: bytes
    profile_id: int
    transcript_id: bytes

    def __post_init__(self) -> None:
        if type(self.code) is not V1NeuralEvidenceCode:
            raise TypeError("V1 neural evidence code must use the exact enum type")
        if type(self.identity_id) is not bytes or len(self.identity_id) != 32:
            raise TypeError("V1 neural evidence identity must be exactly 32 bytes")
        if type(self.profile_id) is not int or self.profile_id != 1:
            raise TypeError("V1 neural evidence profile must be exactly 1")
        if type(self.transcript_id) is not bytes or len(self.transcript_id) != 32:
            raise TypeError("V1 neural evidence transcript must be exactly 32 bytes")

    @property
    def accepted(self) -> bool:
        """返回 neural relation 是否接受, 不提交授权。"""
        return self.code is V1NeuralEvidenceCode.NEURAL_ACCEPT


def _require_int32(value: int) -> int:
    if type(value) is not int or not _INT32_MIN <= value <= _INT32_MAX:
        raise V1NeuralEvaluationError("V1-C1 value exceeds exact int32 semantics")
    return value


def _require_int64(value: int) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        raise V1NeuralEvaluationError("V1-C1 accumulator exceeds exact int64 semantics")
    return value


def _relu(value: int) -> int:
    return value if value > 0 else 0


def _canonical_row(terms: Iterable[tuple[int, int]], bias: int) -> V1SparseRow:
    combined: dict[int, int] = {}
    for index, weight in terms:
        combined[index] = combined.get(index, 0) + weight
    canonical = tuple((index, weight) for index, weight in sorted(combined.items()) if weight != 0)
    if type(bias) is not int or not _INT32_MIN <= bias <= _INT32_MAX:
        raise V1NeuralProfileError("V1-C1 bias is outside int32")
    for index, weight in canonical:
        if type(index) is not int or index < 0:
            raise V1NeuralProfileError("V1-C1 sparse indexes are not ordered")
        if type(weight) is not int or weight == 0 or not _INT32_MIN <= weight <= _INT32_MAX:
            raise V1NeuralProfileError("V1-C1 sparse weight is not canonical int32")
    return V1SparseRow(canonical, bias)


def _build_layers(profile: V1PublicProfile) -> tuple[V1SparseAffineReluLayer, ...]:
    layer1_rows: list[V1SparseRow] = []
    # Input is u (row-major residues), c, then z (vector-major coefficients).
    u_offset = 0
    c_offset = V1_MODULE_ROWS * V1_RING_DEGREE
    z_offset = c_offset + V1_RING_DEGREE
    for row_index in range(V1_MODULE_ROWS):
        for coefficient_index in range(V1_RING_DEGREE):
            residual_terms: list[tuple[int, int]] = []
            for column_index in range(V1_MODULE_COLUMNS):
                matrix_poly = profile.matrix[row_index][column_index]
                for power in range(V1_RING_DEGREE):
                    degree = coefficient_index - power
                    sign = 1
                    if degree < 0:
                        degree += V1_RING_DEGREE
                        sign = -1
                    residual_terms.append(
                        (
                            z_offset + column_index * V1_RING_DEGREE + degree,
                            sign * matrix_poly[power],
                        )
                    )
            residual_terms.append(
                (
                    z_offset + (V1_MODULE_COLUMNS + row_index) * V1_RING_DEGREE + coefficient_index,
                    1,
                )
            )
            residual_terms.append((u_offset + row_index * V1_RING_DEGREE + coefficient_index, -1))
            for power in range(V1_RING_DEGREE):
                degree = coefficient_index - power
                sign = 1
                if degree < 0:
                    degree += V1_RING_DEGREE
                    sign = -1
                residual_terms.append((c_offset + degree, -sign * profile.target[row_index][power]))
            # One row per k and offset produces a point pulse at residual == k*q.
            for multiple in V1_RESIDUAL_MULTIPLES:
                center = multiple * V1_MODULUS
                layer1_rows.extend(
                    (
                        _canonical_row(residual_terms, -center + 1),
                        _canonical_row(residual_terms, -center),
                        _canonical_row(residual_terms, -center - 1),
                    )
                )
    # Norm check uses two ReLUs per response coefficient: max(0,z-B)+max(0,-z-B).
    for index in range(V1_NEURAL_NORM_UNITS // 2):
        layer1_rows.append(_canonical_row(((z_offset + index, 1),), -V1_RESPONSE_BOUND))
        layer1_rows.append(_canonical_row(((z_offset + index, -1),), -V1_RESPONSE_BOUND))
    layer2_rows: list[V1SparseRow] = []
    for residual_index in range(V1_NEURAL_RESIDUAL_COUNT):
        start = residual_index * len(V1_RESIDUAL_MULTIPLES) * 3
        pulse_terms: list[tuple[int, int]] = []
        for multiple_index in range(len(V1_RESIDUAL_MULTIPLES)):
            pulse_start = start + multiple_index * 3
            pulse_terms.extend(((pulse_start, 1), (pulse_start + 1, -2), (pulse_start + 2, 1)))
        layer2_rows.append(_canonical_row(pulse_terms, 0))
    norm_start = V1_RESIDUAL_PULSE_COUNT * 3
    layer2_rows.append(
        _canonical_row(
            ((norm_start + index, 1) for index in range(V1_NEURAL_NORM_UNITS)),
            0,
        )
    )
    # Subtract the norm accumulator: only zero residuals and zero norm violation survive.
    layer3_rows = (
        _canonical_row(
            (
                *tuple((index, 1) for index in range(V1_NEURAL_RESIDUAL_COUNT)),
                (V1_NEURAL_RESIDUAL_COUNT, -1),
            ),
            -V1_NEURAL_RESIDUAL_COUNT + 1,
        ),
    )
    layers = (
        V1SparseAffineReluLayer(V1_NEURAL_INPUT_WIDTH, tuple(layer1_rows)),
        V1SparseAffineReluLayer(len(layer1_rows), tuple(layer2_rows)),
        V1SparseAffineReluLayer(len(layer2_rows), layer3_rows),
    )
    expected = (
        V1_NEURAL_INPUT_WIDTH,
        V1_NEURAL_LAYER_WIDTHS[0],
        V1_NEURAL_LAYER_WIDTHS[1],
    )
    if tuple(layer.input_width for layer in layers) != expected or tuple(
        layer.output_width for layer in layers
    ) != (V1_NEURAL_LAYER_WIDTHS[0], V1_NEURAL_LAYER_WIDTHS[1], 1):
        raise V1NeuralProfileError("V1-C1 graph topology is not frozen")
    return layers


def compile_v1_neural_profile(profile: V1PublicProfile) -> V1NeuralProfile:
    """从可信 V1 public profile 编译不可变 coefficient-domain graph。"""
    return V1NeuralProfile(profile)


def _affine_relu(inputs: tuple[int, ...], layer: V1SparseAffineReluLayer) -> tuple[int, ...]:
    if len(inputs) != layer.input_width:
        raise V1NeuralEvaluationError("V1-C1 layer input has the wrong width")
    outputs: list[int] = []
    for row in layer.rows:
        accumulator = _require_int64(row.bias)
        for index, weight in row.terms:
            accumulator = _require_int64(accumulator + weight * inputs[index])
        outputs.append(_require_int32(_relu(accumulator)))
    return tuple(outputs)


def _canonical_core_input(
    commitment: V1Commitment, challenge: V1Challenge, response: V1Response
) -> tuple[int, ...]:
    if commitment.profile_id != challenge.profile_id:
        raise V1NeuralEvaluationError("V1-C1 profile ids do not match")
    return (
        *tuple(value for polynomial in commitment.polynomials for value in polynomial),
        *challenge.coefficients,
        *tuple(value for polynomial in response.polynomials for value in polynomial),
    )


def _run_v1_graph(
    commitment: V1Commitment,
    challenge: V1Challenge,
    response: V1Response,
    profile: V1NeuralProfile,
) -> tuple[tuple[int, ...], ...]:
    if type(profile) is not V1NeuralProfile:
        raise V1NeuralEvaluationError("V1-C1 graph requires exact compiled profile")
    values = _canonical_core_input(commitment, challenge, response)
    outputs: list[tuple[int, ...]] = []
    for layer in profile.layers:
        values = _affine_relu(values, layer)
        outputs.append(values)
    return tuple(outputs)


def verify_v1_neural(
    raw_commitment: object,
    raw_challenge: object,
    raw_response: object,
    expected_transcript_id: object,
    profile: object,
) -> V1NeuralEvidence:
    """执行 V1-C1 graph 并返回只含公开绑定摘要的 evidence。"""
    if type(profile) is not V1NeuralProfile:
        return V1NeuralEvidence(V1NeuralEvidenceCode.CONFIG_REJECT, bytes(32), 1, bytes(32))
    try:
        commitment = parse_v1_commitment(raw_commitment)
        challenge = parse_v1_challenge(raw_challenge)
        response = parse_v1_response(raw_response)
        if type(expected_transcript_id) is not bytes or len(expected_transcript_id) != 32:
            raise V1WireParseError("V1 transcript id is not canonical")
        if response.transcript_id != expected_transcript_id:
            raise V1WireParseError("V1 transcript id is not bound")
        if (
            commitment.profile_id != profile.profile_id
            or challenge.profile_id != profile.profile_id
        ):
            raise V1WireParseError("V1 profile id is not bound")
        if any(
            abs(value) > V1_NEURAL_RESPONSE_INPUT_BOUND
            for polynomial in response.polynomials
            for value in polynomial
        ):
            return V1NeuralEvidence(
                V1NeuralEvidenceCode.NEURAL_REJECT,
                profile.identity_id,
                profile.profile_id,
                response.transcript_id,
            )
        output = _run_v1_graph(commitment, challenge, response, profile)[-1][0]
    except V1WireParseError:
        return V1NeuralEvidence(
            V1NeuralEvidenceCode.INPUT_REJECT,
            profile.identity_id,
            profile.profile_id,
            (
                expected_transcript_id
                if type(expected_transcript_id) is bytes and len(expected_transcript_id) == 32
                else bytes(32)
            ),
        )
    except V1NeuralEvaluationError:
        return V1NeuralEvidence(
            V1NeuralEvidenceCode.CONFIG_REJECT,
            profile.identity_id,
            profile.profile_id,
            response.transcript_id if "response" in locals() else bytes(32),
        )
    code = V1NeuralEvidenceCode.NEURAL_ACCEPT if output == 1 else V1NeuralEvidenceCode.NEURAL_REJECT
    return V1NeuralEvidence(code, profile.identity_id, profile.profile_id, response.transcript_id)


__all__ = [
    "V1_CONVOLUTION_ACCUMULATOR_BOUND",
    "V1_CONVOLUTION_ACCUMULATOR_MAX",
    "V1_FINAL_ACCUMULATOR_MAX",
    "V1_FINAL_ACCUMULATOR_MIN",
    "V1_NEURAL_CANDIDATE_ID",
    "V1_NEURAL_INPUT_WIDTH",
    "V1_NEURAL_LAYER2_WIDTH",
    "V1_NEURAL_LAYER_WIDTHS",
    "V1_NEURAL_NORM_UNITS",
    "V1_NEURAL_PULSE_COUNT",
    "V1_NEURAL_RESIDUAL_COUNT",
    "V1_NEURAL_SCALE",
    "V1_NORM_VIOLATION_MAX",
    "V1_RESIDUAL_ABS_MAX",
    "V1_RESIDUAL_BOUND",
    "V1_RESIDUAL_MULTIPLES",
    "V1_RESIDUAL_MULTIPLE_MAX",
    "V1_RESIDUAL_MULTIPLE_MIN",
    "V1_RESIDUAL_PULSE_COUNT",
    "V1NeuralEvaluationError",
    "V1NeuralEvidence",
    "V1NeuralEvidenceCode",
    "V1NeuralProfile",
    "V1NeuralProfileError",
    "V1SparseAffineReluLayer",
    "V1SparseRow",
    "compile_v1_neural_profile",
    "verify_v1_neural",
]
