"""A3-v1 请求绑定、challenge freshness 与单次消费协议壳。"""

from __future__ import annotations

import hashlib
import secrets
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock, RLock
from typing import Final, Literal, TypeAlias, TypedDict, cast

import torch
from torch import Tensor, nn

from can.model.a2_mlp import A2_CLASS_COUNT, A2FashionMNISTMLP, validate_a2_images

A3_PROTOCOL_VERSION: Final = 1
A3_RESPONSE_VERSION: Final = 3
A3_MODEL_ID: Final = 1
A3_SCOPE_ID: Final = 1
A3_CHALLENGE_TTL_MS: Final = 60_000
A3_IDENTITY_SIZE: Final = 32
A3_NONCE_SIZE: Final = 32
A3_DIGEST_SIZE: Final = 32
A3_PROOF_MAX_SIZE: Final = 65_535
A3_MESSAGE_SIZE: Final = 133
A3_MESSAGE_DOMAIN: Final = b"CAN-A3-MSG-v1\x00"
A3_INPUT_DOMAIN: Final = b"CAN-A3-INPUT-v1\x00"

_EXPECTED_MODULE_TYPES: Final = (
    A2FashionMNISTMLP,
    nn.Sequential,
    nn.Flatten,
    nn.Linear,
    nn.ReLU,
    nn.Linear,
    nn.ReLU,
    nn.Linear,
)
_EXPECTED_PARAMETERS: Final = {
    "_network.1.weight": (256, 784),
    "_network.1.bias": (256,),
    "_network.3.weight": (128, 256),
    "_network.3.bias": (128,),
    "_network.5.weight": (A2_CLASS_COUNT, 128),
    "_network.5.bias": (A2_CLASS_COUNT,),
}


class A3DenyEnvelope(TypedDict):
    """定义 A3 固定拒绝响应。"""

    version: Literal[3]
    status: Literal["deny"]


class A3ChallengeEnvelope(TypedDict):
    """定义只返回 canonical message 的 challenge 响应。"""

    version: Literal[3]
    status: Literal["challenge"]
    message: bytes


class A3ProtectedEnvelope(TypedDict):
    """定义只返回 protected top-1 类别的响应。"""

    version: Literal[3]
    status: Literal["protected"]
    class_id: int


A3Envelope: TypeAlias = A3DenyEnvelope | A3ChallengeEnvelope | A3ProtectedEnvelope


class A3EvidenceCode(Enum):
    """表示 verifier 产生的无授权能力证据码。"""

    PROOF_ACCEPT = "proof_accept"
    PROOF_REJECT = "proof_reject"
    CONFIG_REJECT = "config_reject"


class A3ProtocolConfigError(ValueError):
    """表示本地可信 A3 配置不满足固定协议契约。"""


class A3ProtocolInputError(ValueError):
    """表示不可信 A3 请求不满足规范输入契约。"""


class A3StateError(RuntimeError):
    """表示可信 nonce 状态或时钟发生不可恢复错误。"""


@dataclass(frozen=True, slots=True)
class A3Clock:
    """提供可注入的可信 wall/monotonic 时钟。"""

    wall_time_ms: Callable[[], int]
    monotonic_ns: Callable[[], int]


def _default_wall_time_ms() -> int:
    return time.time_ns() // 1_000_000


def _default_monotonic_ns() -> int:
    return time.monotonic_ns()


@dataclass(frozen=True, slots=True)
class A3Evidence:
    """保存与精确 A3 message 绑定的 evidence, 不具有授权能力。"""

    code: A3EvidenceCode
    identity_id: bytes
    message_sha256: bytes
    profile_id: int

    def __post_init__(self) -> None:
        if type(self.code) is not A3EvidenceCode:
            raise A3ProtocolConfigError("A3 evidence code must use the exact enum type")
        if type(self.identity_id) is not bytes or len(self.identity_id) != A3_IDENTITY_SIZE:
            raise A3ProtocolConfigError("A3 evidence identity_id must be exactly 32 bytes")
        if type(self.message_sha256) is not bytes or len(self.message_sha256) != A3_DIGEST_SIZE:
            raise A3ProtocolConfigError("A3 evidence message digest must be exactly 32 bytes")
        if (
            type(self.profile_id) is not int
            or isinstance(self.profile_id, bool)
            or self.profile_id < 1
        ):
            raise A3ProtocolConfigError("A3 evidence profile_id must be a positive exact int")


A3Verifier: TypeAlias = Callable[[bytes, bytes], object]


@dataclass(frozen=True, slots=True)
class A3VerificationProfile:
    """绑定一个本地 identity 与未来 A4 evidence-only verifier。"""

    identity_id: bytes
    profile_id: int
    verifier: A3Verifier

    def __post_init__(self) -> None:
        if type(self.identity_id) is not bytes or len(self.identity_id) != A3_IDENTITY_SIZE:
            raise A3ProtocolConfigError("A3 profile identity_id must be exactly 32 bytes")
        if (
            type(self.profile_id) is not int
            or isinstance(self.profile_id, bool)
            or self.profile_id < 1
        ):
            raise A3ProtocolConfigError("A3 profile_id must be a positive exact int")
        if not callable(self.verifier):
            raise A3ProtocolConfigError("A3 profile verifier must be callable")


@dataclass(frozen=True, slots=True)
class A3Message:
    """表示规范 A3-v1 proof message 的解析结果。"""

    version: int
    model_id: int
    identity_id: bytes
    scope_id: int
    issued_at_ms: int
    expires_at_ms: int
    nonce: bytes
    input_digest: bytes

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or isinstance(self.version, bool)
            or self.version != A3_PROTOCOL_VERSION
        ):
            raise A3ProtocolInputError("A3 message version is not canonical")
        if (
            type(self.model_id) is not int
            or isinstance(self.model_id, bool)
            or self.model_id != A3_MODEL_ID
        ):
            raise A3ProtocolInputError("A3 message model_id is not locally trusted")
        if type(self.identity_id) is not bytes or len(self.identity_id) != A3_IDENTITY_SIZE:
            raise A3ProtocolInputError("A3 identity_id must be exactly 32 bytes")
        if (
            type(self.scope_id) is not int
            or isinstance(self.scope_id, bool)
            or self.scope_id != A3_SCOPE_ID
        ):
            raise A3ProtocolInputError("A3 message scope_id is not locally trusted")
        if (
            type(self.issued_at_ms) is not int
            or isinstance(self.issued_at_ms, bool)
            or self.issued_at_ms < 0
        ):
            raise A3ProtocolInputError("A3 issued_at_ms must be a non-negative exact int")
        if (
            type(self.expires_at_ms) is not int
            or isinstance(self.expires_at_ms, bool)
            or self.expires_at_ms != self.issued_at_ms + A3_CHALLENGE_TTL_MS
        ):
            raise A3ProtocolInputError("A3 expires_at_ms does not match the fixed TTL")
        if type(self.nonce) is not bytes or len(self.nonce) != A3_NONCE_SIZE:
            raise A3ProtocolInputError("A3 nonce must be exactly 32 bytes")
        if type(self.input_digest) is not bytes or len(self.input_digest) != A3_DIGEST_SIZE:
            raise A3ProtocolInputError("A3 input digest must be exactly 32 bytes")
        for value in (self.model_id, self.scope_id, self.issued_at_ms, self.expires_at_ms):
            if value >= 1 << 64:
                raise A3ProtocolInputError("A3 message integer exceeds the canonical wire width")

    def encode(self) -> bytes:
        """将 A3 message 编码为唯一的 133 字节大端字节串。"""
        return b"".join(
            (
                A3_MESSAGE_DOMAIN,
                self.version.to_bytes(1, byteorder="big", signed=False),
                self.model_id.to_bytes(4, byteorder="big", signed=False),
                self.identity_id,
                self.scope_id.to_bytes(2, byteorder="big", signed=False),
                self.issued_at_ms.to_bytes(8, byteorder="big", signed=False),
                self.expires_at_ms.to_bytes(8, byteorder="big", signed=False),
                self.nonce,
                self.input_digest,
            )
        )


def parse_a3_message(raw_message: object) -> A3Message:
    """严格解析固定长度的 A3-v1 message。"""
    if type(raw_message) is not bytes or len(raw_message) != A3_MESSAGE_SIZE:
        raise A3ProtocolInputError("A3 message must be exactly 133 bytes")
    if raw_message[: len(A3_MESSAGE_DOMAIN)] != A3_MESSAGE_DOMAIN:
        raise A3ProtocolInputError("A3 message domain separator is not canonical")
    offset = len(A3_MESSAGE_DOMAIN)
    message = A3Message(
        version=raw_message[offset],
        model_id=int.from_bytes(
            raw_message[offset + 1 : offset + 5], byteorder="big", signed=False
        ),
        identity_id=raw_message[offset + 5 : offset + 37],
        scope_id=int.from_bytes(
            raw_message[offset + 37 : offset + 39], byteorder="big", signed=False
        ),
        issued_at_ms=int.from_bytes(
            raw_message[offset + 39 : offset + 47], byteorder="big", signed=False
        ),
        expires_at_ms=int.from_bytes(
            raw_message[offset + 47 : offset + 55], byteorder="big", signed=False
        ),
        nonce=raw_message[offset + 55 : offset + 87],
        input_digest=raw_message[offset + 87 : offset + 119],
    )
    if message.encode() != raw_message:
        raise A3ProtocolInputError("A3 message has a non-canonical encoding")
    return message


def canonicalize_a3_image(images: object) -> tuple[Tensor, bytes]:
    """验证、复制单张 A3 图像并返回其 canonical snapshot 与 SHA-256。"""
    if type(images) is not torch.Tensor:
        raise A3ProtocolInputError("A3 image must be exactly torch.Tensor")
    validate_a2_images(images)
    if images.shape[0] != 1:
        raise A3ProtocolInputError("A3 challenge binding accepts exactly one image")
    if bool(torch.signbit(images).any().item()):
        raise A3ProtocolInputError("A3 image rejects negative zero and negative values")
    snapshot = images.detach().clone(memory_format=torch.contiguous_format)
    values = snapshot.reshape(-1).tolist()
    payload = bytearray(A3_INPUT_DOMAIN)
    payload.extend(struct.pack(">BHHH", 3, 1, 28, 28))
    for value in values:
        payload.extend(struct.pack(">f", float(value)))
    return snapshot, hashlib.sha256(bytes(payload)).digest()


@dataclass(frozen=True, slots=True)
class _A3ChallengeRecord:
    """保存 nonce store 内部的不可变绑定记录。"""

    message_sha256: bytes
    identity_id: bytes
    model_id: int
    scope_id: int
    input_digest: bytes
    profile_id: int
    issued_at_ms: int
    expires_at_ms: int
    deadline_ns: int
    state: Literal["PENDING", "CONSUMED"]


class A3NonceStore:
    """提供单进程线程安全的 A3 challenge 状态与原子单次消费。"""

    __slots__ = (
        "_clock",
        "_clock_lock",
        "_last_monotonic_ns",
        "_lock",
        "_random_bytes",
        "_records",
        "_used_nonces",
    )

    def __init__(
        self,
        *,
        clock: A3Clock | None = None,
        random_bytes: Callable[[int], bytes] | None = None,
    ) -> None:
        self._clock = (
            A3Clock(_default_wall_time_ms, _default_monotonic_ns) if clock is None else clock
        )
        if type(self._clock) is not A3Clock:
            raise A3ProtocolConfigError("A3 clock must use the exact trusted clock type")
        if not callable(self._clock.wall_time_ms) or not callable(self._clock.monotonic_ns):
            raise A3ProtocolConfigError("A3 clock callbacks must be callable")
        self._random_bytes = secrets.token_bytes if random_bytes is None else random_bytes
        if not callable(self._random_bytes):
            raise A3ProtocolConfigError("A3 random source must be callable")
        self._clock_lock = RLock()
        self._last_monotonic_ns: int | None = None
        self._lock = RLock()
        self._records: dict[bytes, _A3ChallengeRecord] = {}
        self._used_nonces: set[bytes] = set()

    def _wall_time_ms(self) -> int:
        value = self._clock.wall_time_ms()
        if type(value) is not int or isinstance(value, bool) or value < 0 or value >= 1 << 64:
            raise A3StateError("trusted wall clock returned a non-canonical value")
        return value

    def _monotonic_ns(self) -> int:
        value = self._clock.monotonic_ns()
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise A3StateError("trusted monotonic clock returned a non-canonical value")
        with self._clock_lock:
            if self._last_monotonic_ns is not None and value < self._last_monotonic_ns:
                raise A3StateError("trusted monotonic clock moved backwards")
            self._last_monotonic_ns = value
        return value

    def _allocate_nonce(self) -> bytes:
        candidate = self._random_bytes(A3_NONCE_SIZE)
        if type(candidate) is not bytes or len(candidate) != A3_NONCE_SIZE:
            raise A3StateError("trusted random source returned a non-canonical nonce")
        with self._lock:
            if candidate in self._used_nonces:
                raise A3StateError("nonce collision in the current state epoch")
            self._used_nonces.add(candidate)
        return candidate

    def _insert_pending(self, nonce: bytes, record: _A3ChallengeRecord) -> None:
        if type(nonce) is not bytes or len(nonce) != A3_NONCE_SIZE:
            raise A3StateError("nonce store key is not canonical")
        if type(record) is not _A3ChallengeRecord or record.state != "PENDING":
            raise A3StateError("nonce store accepts only pending records")
        with self._lock:
            if nonce in self._records:
                raise A3StateError("nonce record replacement is forbidden")
            self._records[nonce] = record

    def _lookup(self, nonce: bytes) -> _A3ChallengeRecord | None:
        with self._lock:
            return self._records.get(nonce)

    def _consume_if_pending(
        self,
        *,
        nonce: bytes,
        message_sha256: bytes,
        identity_id: bytes,
        model_id: int,
        scope_id: int,
        input_digest: bytes,
        profile_id: int,
    ) -> bool:
        with self._lock:
            now_ns = self._monotonic_ns()
            record = self._records.get(nonce)
            if record is None or record.state != "PENDING":
                return False
            if (
                record.message_sha256 != message_sha256
                or record.identity_id != identity_id
                or record.model_id != model_id
                or record.scope_id != scope_id
                or record.input_digest != input_digest
                or record.profile_id != profile_id
                or now_ns >= record.deadline_ns
            ):
                return False
            self._records[nonce] = replace(record, state="CONSUMED")
            return True


@dataclass(frozen=True, slots=True)
class A3ProtocolSnapshot:
    """提供不含输入、proof、nonce 或 evidence 的 A3 计数快照。"""

    challenge_issues: int
    challenge_denies: int
    verifier_calls: int
    coordinator_commits: int
    allow_commits: int
    deny_commits: int
    protected_model_calls: int
    protected_responses: int
    deny_responses: int


def _deny_envelope() -> A3DenyEnvelope:
    return {"version": A3_RESPONSE_VERSION, "status": "deny"}


def _validate_protected_model(model: object) -> A2FashionMNISTMLP:
    if type(model) is not A2FashionMNISTMLP:
        raise A3ProtocolConfigError("A3 protected model must be exactly A2FashionMNISTMLP")
    typed_model = model
    if typed_model.training:
        raise A3ProtocolConfigError("A3 protected model must remain in evaluation mode")
    if tuple(type(module) for module in typed_model.modules()) != _EXPECTED_MODULE_TYPES:
        raise A3ProtocolConfigError("A3 protected model topology changed")
    if tuple(typed_model.buffers()) or any(
        hooks
        for module in typed_model.modules()
        for hooks in (
            module._forward_hooks,
            module._forward_pre_hooks,
            module._backward_hooks,
            module._backward_pre_hooks,
        )
    ):
        raise A3ProtocolConfigError("A3 protected model hooks/buffers are unsupported")
    parameters = dict(typed_model.named_parameters())
    if set(parameters) != set(_EXPECTED_PARAMETERS):
        raise A3ProtocolConfigError("A3 protected model parameter set changed")
    for name, expected_shape in _EXPECTED_PARAMETERS.items():
        parameter = parameters[name]
        if (
            type(parameter) is not nn.Parameter
            or parameter.dtype is not torch.float32
            or parameter.device.type != "cpu"
            or parameter.device.index is not None
            or parameter.layout is not torch.strided
            or tuple(parameter.shape) != expected_shape
            or not parameter.is_contiguous()
            or not bool(torch.isfinite(parameter).all().item())
        ):
            raise A3ProtocolConfigError("A3 protected model parameter contract changed")
    return typed_model


class A3ProtocolCoordinator:
    """以唯一协调器执行 A3 request binding、evidence 检查和 protected 调用。"""

    __slots__ = (
        "_allow_commits",
        "_challenge_denies",
        "_challenge_issues",
        "_coordinator_commits",
        "_deny_commits",
        "_deny_responses",
        "_lock",
        "_profiles",
        "_protected_model",
        "_protected_model_calls",
        "_protected_responses",
        "_store",
        "_verifier_calls",
    )

    def __init__(
        self,
        protected_model: object | None = None,
        profiles: Sequence[A3VerificationProfile] = (),
        *,
        store: A3NonceStore | None = None,
    ) -> None:
        if protected_model is not None:
            _validate_protected_model(protected_model)
        if type(profiles) not in (tuple, list):
            raise A3ProtocolConfigError("A3 profiles must be a trusted tuple or list")
        profile_map: dict[bytes, A3VerificationProfile] = {}
        for profile in profiles:
            if type(profile) is not A3VerificationProfile:
                raise A3ProtocolConfigError("A3 profiles must use the exact profile type")
            if profile.identity_id in profile_map:
                raise A3ProtocolConfigError("A3 profile identity is duplicated")
            profile_map[profile.identity_id] = profile
        if store is not None and type(store) is not A3NonceStore:
            raise A3ProtocolConfigError("A3 store must use the exact trusted store type")
        self._protected_model = protected_model
        self._profiles = profile_map
        self._store = A3NonceStore() if store is None else store
        self._lock = Lock()
        self._challenge_issues = 0
        self._challenge_denies = 0
        self._verifier_calls = 0
        self._coordinator_commits = 0
        self._allow_commits = 0
        self._deny_commits = 0
        self._protected_model_calls = 0
        self._protected_responses = 0
        self._deny_responses = 0

    def snapshot(self) -> A3ProtocolSnapshot:
        """返回不含请求内容、proof、nonce 或 evidence 的内部计数。"""
        with self._lock:
            return A3ProtocolSnapshot(
                challenge_issues=self._challenge_issues,
                challenge_denies=self._challenge_denies,
                verifier_calls=self._verifier_calls,
                coordinator_commits=self._coordinator_commits,
                allow_commits=self._allow_commits,
                deny_commits=self._deny_commits,
                protected_model_calls=self._protected_model_calls,
                protected_responses=self._protected_responses,
                deny_responses=self._deny_responses,
            )

    def _record_challenge(self, *, issued: bool) -> None:
        with self._lock:
            if issued:
                self._challenge_issues += 1
            else:
                self._challenge_denies += 1

    def _record_verifier_call(self) -> None:
        with self._lock:
            self._verifier_calls += 1

    def _commit(self, *, allow: bool) -> None:
        with self._lock:
            self._coordinator_commits += 1
            if allow:
                self._allow_commits += 1
            else:
                self._deny_commits += 1

    def _record_response(self, *, protected: bool) -> None:
        with self._lock:
            if protected:
                self._protected_responses += 1
            else:
                self._deny_responses += 1

    def issue_challenge(self, request: object) -> A3Envelope:
        """为可信 identity 创建一次绑定 challenge, 配置或输入失败固定拒绝。"""
        try:
            if type(request) is not dict or set(request) != {
                "version",
                "model_id",
                "identity_id",
                "scope_id",
                "image",
            }:
                raise A3ProtocolInputError("A3 challenge request fields are not closed")
            if type(request["version"]) is not int or request["version"] != A3_PROTOCOL_VERSION:
                raise A3ProtocolInputError("A3 challenge version is not canonical")
            if type(request["model_id"]) is not int or request["model_id"] != A3_MODEL_ID:
                raise A3ProtocolInputError("A3 challenge model is not locally bound")
            if type(request["scope_id"]) is not int or request["scope_id"] != A3_SCOPE_ID:
                raise A3ProtocolInputError("A3 challenge scope is not locally bound")
            identity_id = request["identity_id"]
            if type(identity_id) is not bytes or len(identity_id) != A3_IDENTITY_SIZE:
                raise A3ProtocolInputError("A3 challenge identity is not canonical")
            profile = self._profiles.get(identity_id)
            if self._protected_model is None or profile is None:
                raise A3ProtocolConfigError("A3 protected entry has no active A4 profile")
            _, input_digest = canonicalize_a3_image(request["image"])
            issued_at_ms = self._store._wall_time_ms()
            monotonic_now = self._store._monotonic_ns()
            nonce = self._store._allocate_nonce()
            message = A3Message(
                A3_PROTOCOL_VERSION,
                A3_MODEL_ID,
                identity_id,
                A3_SCOPE_ID,
                issued_at_ms,
                issued_at_ms + A3_CHALLENGE_TTL_MS,
                nonce,
                input_digest,
            )
            self._store._insert_pending(
                nonce,
                _A3ChallengeRecord(
                    hashlib.sha256(message.encode()).digest(),
                    identity_id,
                    A3_MODEL_ID,
                    A3_SCOPE_ID,
                    input_digest,
                    profile.profile_id,
                    issued_at_ms,
                    message.expires_at_ms,
                    monotonic_now + A3_CHALLENGE_TTL_MS * 1_000_000,
                    "PENDING",
                ),
            )
            self._record_challenge(issued=True)
            return {
                "version": A3_RESPONSE_VERSION,
                "status": "challenge",
                "message": message.encode(),
            }
        except Exception:
            self._record_challenge(issued=False)
            return _deny_envelope()

    def respond(self, message: object, proof: object, images: object) -> A3Envelope:
        """验证绑定 proof 并以原子 nonce consume 决定是否调用 protected model。"""
        try:
            if self._protected_model is None:
                raise A3ProtocolConfigError("A3 protected entry is disabled")
            if type(proof) is not bytes or not 1 <= len(proof) <= A3_PROOF_MAX_SIZE:
                raise A3ProtocolInputError("A3 proof bytes are not canonical")
            parsed = parse_a3_message(message)
            snapshot, input_digest = canonicalize_a3_image(images)
            profile = self._profiles.get(parsed.identity_id)
            record = self._store._lookup(parsed.nonce)
            if profile is None or record is None:
                raise A3ProtocolInputError("A3 identity or nonce is unknown")
            now_ns = self._store._monotonic_ns()
            message_bytes = parsed.encode()
            message_sha256 = hashlib.sha256(message_bytes).digest()
            if (
                record.state != "PENDING"
                or now_ns >= record.deadline_ns
                or record.message_sha256 != message_sha256
                or record.identity_id != parsed.identity_id
                or record.model_id != parsed.model_id
                or record.scope_id != parsed.scope_id
                or record.input_digest != input_digest
                or record.input_digest != parsed.input_digest
                or record.profile_id != profile.profile_id
            ):
                raise A3ProtocolInputError("A3 request binding or freshness check failed")
            self._record_verifier_call()
            evidence = profile.verifier(message_bytes, proof)
            if (
                type(evidence) is not A3Evidence
                or evidence.code is not A3EvidenceCode.PROOF_ACCEPT
                or evidence.identity_id != parsed.identity_id
                or evidence.message_sha256 != message_sha256
                or evidence.profile_id != profile.profile_id
            ):
                raise A3ProtocolInputError("A3 evidence is not an exact bound accept")
            # 受保护模型在构造时已校验并冻结; 按威胁模型不在请求路径重复校验可信不可变配置
            # (参见 AGENTS.md Threat-model alignment)。
            model = cast(A2FashionMNISTMLP, self._protected_model)
            if not self._store._consume_if_pending(
                nonce=parsed.nonce,
                message_sha256=message_sha256,
                identity_id=parsed.identity_id,
                model_id=parsed.model_id,
                scope_id=parsed.scope_id,
                input_digest=input_digest,
                profile_id=profile.profile_id,
            ):
                raise A3ProtocolInputError("A3 atomic consume rejected the request")
            self._commit(allow=True)
            with self._lock:
                self._protected_model_calls += 1
            try:
                with torch.inference_mode():
                    logits = model(snapshot)
                    if (
                        type(logits) is not torch.Tensor
                        or logits.dtype is not torch.float32
                        or logits.device.type != "cpu"
                        or logits.device.index is not None
                        or logits.layout is not torch.strided
                        or not logits.is_contiguous()
                        or tuple(logits.shape) != (1, A2_CLASS_COUNT)
                        or not bool(torch.isfinite(logits).all().item())
                    ):
                        raise RuntimeError("A3 protected model output is not canonical")
                    class_id = logits.argmax(dim=1).item()
                if type(class_id) is not int or not 0 <= class_id < A2_CLASS_COUNT:
                    raise RuntimeError("A3 protected class id is not canonical")
                response: A3ProtectedEnvelope = {
                    "version": A3_RESPONSE_VERSION,
                    "status": "protected",
                    "class_id": class_id,
                }
                self._record_response(protected=True)
                return response
            except Exception:
                self._record_response(protected=False)
                return _deny_envelope()
        except Exception:
            self._commit(allow=False)
            self._record_response(protected=False)
            return _deny_envelope()


__all__ = [
    "A3_CHALLENGE_TTL_MS",
    "A3_DIGEST_SIZE",
    "A3_IDENTITY_SIZE",
    "A3_MESSAGE_DOMAIN",
    "A3_MESSAGE_SIZE",
    "A3_MODEL_ID",
    "A3_NONCE_SIZE",
    "A3_PROTOCOL_VERSION",
    "A3_RESPONSE_VERSION",
    "A3_SCOPE_ID",
    "A3ChallengeEnvelope",
    "A3Clock",
    "A3DenyEnvelope",
    "A3Envelope",
    "A3Evidence",
    "A3EvidenceCode",
    "A3Message",
    "A3NonceStore",
    "A3ProtectedEnvelope",
    "A3ProtocolConfigError",
    "A3ProtocolCoordinator",
    "A3ProtocolInputError",
    "A3ProtocolSnapshot",
    "A3StateError",
    "A3VerificationProfile",
    "canonicalize_a3_image",
    "parse_a3_message",
]
