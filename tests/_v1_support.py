"""V1-P2/A3-v2 测试使用的确定性公开 conformance helpers。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from can.access import (
    A3V2Clock,
    A3V2ProtocolCoordinator,
    A3V2TranscriptStore,
    A3V2TrustedInput,
    build_v1_a3_v2_profile,
)
from can.reference import (
    V1_MODULUS,
    V1_PROFILE_ID,
    V1_RING_DEGREE,
    V1Challenge,
    V1Commitment,
    V1PublicProfile,
    build_v1_conformance_profile,
    v1_negacyclic_convolution,
)

V1_TEST_IDENTITY = bytes(range(32))
V1_TEST_MODEL_ID = 2
V1_TEST_SCOPE_ID = 1
V1_TEST_INPUT_PROFILE = hashlib.sha256(b"CAN-TEST-V1-INPUT-PROFILE-v1").digest()
V1_TEST_CHALLENGE = V1Challenge(V1_PROFILE_ID, (1, 0, 0, 0, 0, 0, 0, -1))
V1_TEST_RESPONSE = (
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0),
    (-1, 0, 1, 0, 0, 0, 0, 0),
    (0, -1, 0, 1, 0, 0, 0, 0),
)


@dataclass(slots=True)
class V1TestClock:
    """提供测试可控的 wall 与 monotonic time。"""

    wall_ms: int = 1_700_000_000_000
    mono_ns: int = 5_000_000_000


@dataclass(slots=True)
class V1ProtectedRecorder:
    """记录受保护 callback 的调用次数与 snapshot。"""

    fail: bool = False
    result: object = None
    snapshots: list[object] = field(default_factory=list)

    def __call__(self, snapshot: object) -> object:
        """记录一次调用并按测试配置返回或抛出。"""
        self.snapshots.append(snapshot)
        if self.fail:
            raise RuntimeError("synthetic protected operation failure")
        return self.result


class V1NonceSource:
    """产生不重复的确定性 32 字节测试 nonce。"""

    __slots__ = ("_counter",)

    def __init__(self) -> None:
        self._counter = 0

    def __call__(self, size: int) -> bytes:
        """返回固定宽度递增 nonce。"""
        self._counter += 1
        return self._counter.to_bytes(size, byteorder="big", signed=False)


def build_v1_accepting_commitment(
    profile: V1PublicProfile,
    challenge: V1Challenge = V1_TEST_CHALLENGE,
    response: tuple[tuple[int, ...], ...] = V1_TEST_RESPONSE,
) -> V1Commitment:
    """从公开 relation 直接构造测试 commitment, 不生成或保存 secret。"""
    output: list[tuple[int, ...]] = []
    for row_index in range(2):
        lhs = [0] * V1_RING_DEGREE
        for column_index in range(2):
            product = v1_negacyclic_convolution(
                profile.matrix[row_index][column_index], response[column_index]
            )
            lhs = [value + product_value for value, product_value in zip(lhs, product, strict=True)]
        lhs = [
            value + identity_value
            for value, identity_value in zip(lhs, response[2 + row_index], strict=True)
        ]
        target = v1_negacyclic_convolution(challenge.coefficients, profile.target[row_index])
        output.append(
            tuple(
                (value - target_value) % V1_MODULUS
                for value, target_value in zip(lhs, target, strict=True)
            )
        )
    return V1Commitment(V1_PROFILE_ID, output)


def build_v1_trusted_input(snapshot: object = b"canonical snapshot") -> A3V2TrustedInput:
    """模拟独立业务 adapter 产生的摘要与同一 snapshot。"""
    if type(snapshot) is bytes:
        snapshot_bytes = snapshot
    else:
        snapshot_bytes = repr(snapshot).encode("utf-8")
    input_digest = hashlib.sha256(
        b"CAN-TEST-V1-INPUT-v1\x00" + V1_TEST_INPUT_PROFILE + snapshot_bytes
    ).digest()
    return A3V2TrustedInput(
        model_id=V1_TEST_MODEL_ID,
        identity_id=V1_TEST_IDENTITY,
        scope_id=V1_TEST_SCOPE_ID,
        input_profile_sha256=V1_TEST_INPUT_PROFILE,
        input_digest=input_digest,
        snapshot=snapshot,
    )


def build_v1_coordinator(
    *,
    recorder: V1ProtectedRecorder | None = None,
) -> tuple[A3V2ProtocolCoordinator, V1PublicProfile, V1ProtectedRecorder, V1TestClock]:
    """构造只含 V1 exact adapter 的确定性 A3-v2 coordinator。"""
    profile = build_v1_conformance_profile(V1_TEST_IDENTITY)
    protected = V1ProtectedRecorder() if recorder is None else recorder
    clock = V1TestClock()
    route = build_v1_a3_v2_profile(
        profile,
        model_id=V1_TEST_MODEL_ID,
        scope_id=V1_TEST_SCOPE_ID,
        input_profile_sha256=V1_TEST_INPUT_PROFILE,
        protected_operation=protected,
    )
    store = A3V2TranscriptStore(
        clock=A3V2Clock(lambda: clock.wall_ms, lambda: clock.mono_ns),
        random_bytes=V1NonceSource(),
    )
    coordinator = A3V2ProtocolCoordinator(
        (route,),
        store=store,
        challenge_sampler=lambda degree, weight: V1_TEST_CHALLENGE.coefficients,
    )
    return coordinator, profile, protected, clock
