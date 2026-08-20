"""V1-M1-C2 固定 R2 module tree 的 public/protected 硬路由。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, Literal, TypeAlias, TypedDict, cast

import torch
from torch import Tensor, nn

from can.access.a3_v2 import (
    A3_V2_RESPONSE_VERSION,
    A3V2ChallengeSampler,
    A3V2Evidence,
    A3V2EvidenceCode,
    A3V2ExecutionState,
    A3V2InternalResultCode,
    A3V2ProtectedExecutionError,
    A3V2ProtocolConfigError,
    A3V2ProtocolCoordinator,
    A3V2ProtocolSnapshot,
    A3V2RouteDecision,
    A3V2TranscriptStore,
    A3V2VerificationProfile,
)
from can.access.v1_adapter import build_v1_a3_v2_neural_profile
from can.access.v1_m1_adapter import (
    V1_M1_INPUT_PROFILE_SHA256,
    V1_M1_MODEL_ID,
    V1_M1_SCOPE_ID,
    V1M1InputAdapter,
    V1M1InputError,
    _preprocess_v1_m1_snapshot,
    _V1M1ImageSnapshot,
)
from can.model.v1_cifar100_resnet import V1Cifar100ResNet18
from can.verifier.v1 import V1NeuralProfile

V1_M1_C2_RESPONSE_VERSION: Final = 5
V1_M1_C2_PUBLIC_CLASS_COUNT: Final = 20
V1_M1_C2_PROTECTED_CLASS_COUNT: Final = 100


class V1M1C2Cut(Enum):
    """定义只能位于完整 residual stage 末端的 C2 cut。"""

    LAYER2 = "layer2"
    LAYER3 = "layer3"
    LAYER4 = "layer4"

    @property
    def channels(self) -> int:
        """返回该 cut 的 frozen feature channel 数。"""
        return {
            V1M1C2Cut.LAYER2: 128,
            V1M1C2Cut.LAYER3: 256,
            V1M1C2Cut.LAYER4: 512,
        }[self]


class V1M1C2ConfigError(ValueError):
    """表示本地可信 C2 配置不满足固定契约。"""


class V1M1C2InputError(ValueError):
    """表示 C2 业务输入或外部入口字段不满足固定契约。"""


class V1M1C2DenyEnvelope(TypedDict):
    """定义 C2 固定拒绝响应。"""

    version: Literal[5]
    status: Literal["deny"]


class V1M1C2ChallengeEnvelope(TypedDict):
    """定义 C2 version-5 challenge 响应。"""

    version: Literal[5]
    status: Literal["challenge"]
    message: bytes
    challenge: bytes
    transcript_id: bytes


class V1M1C2PublicEnvelope(TypedDict):
    """定义 C2 public coarse-class 响应。"""

    version: Literal[5]
    status: Literal["public"]
    coarse_class_id: int


class V1M1C2ProtectedEnvelope(TypedDict):
    """定义 C2 protected fine-class 响应。"""

    version: Literal[5]
    status: Literal["protected"]
    class_id: int


V1M1C2Envelope: TypeAlias = (
    V1M1C2DenyEnvelope | V1M1C2ChallengeEnvelope | V1M1C2PublicEnvelope | V1M1C2ProtectedEnvelope
)
V1M1C2EventSink: TypeAlias = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class V1M1C2Policy:
    """定义构造时固定的可信 public entry 开关。"""

    public_entry_enabled: bool = False

    def __post_init__(self) -> None:
        if type(self.public_entry_enabled) is not bool:
            raise V1M1C2ConfigError("public_entry_enabled must be exactly bool")


class V1M1C2PublicHead(nn.Module):
    """实现统一的 AdaptiveAvgPool2d -> Flatten -> Linear public head。"""

    __slots__ = ("channels", "classifier", "flatten", "pool")

    def __init__(self, channels: int) -> None:
        super().__init__()
        if type(channels) is not int or channels not in (128, 256, 512):
            raise V1M1C2ConfigError("C2 public head channels are not canonical")
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(channels, V1_M1_C2_PUBLIC_CLASS_COUNT)

    def forward(self, features: Tensor) -> Tensor:
        """将 cut feature 映射为 20 类 coarse logits。"""
        return cast(Tensor, self.classifier(self.flatten(self.pool(features))))


def _deny_envelope() -> V1M1C2DenyEnvelope:
    return {"version": V1_M1_C2_RESPONSE_VERSION, "status": "deny"}


def _map_challenge(envelope: object) -> V1M1C2Envelope:
    if type(envelope) is not dict:
        return _deny_envelope()
    if envelope.get("status") != "challenge":
        return _deny_envelope()
    required = {"version", "status", "message", "challenge", "transcript_id"}
    if set(envelope) != required:
        return _deny_envelope()
    if envelope.get("version") != A3_V2_RESPONSE_VERSION:
        return _deny_envelope()
    if not all(type(envelope[key]) is bytes for key in required - {"version", "status"}):
        return _deny_envelope()
    return {
        "version": V1_M1_C2_RESPONSE_VERSION,
        "status": "challenge",
        "message": cast(bytes, envelope["message"]),
        "challenge": cast(bytes, envelope["challenge"]),
        "transcript_id": cast(bytes, envelope["transcript_id"]),
    }


def _validate_logits(logits: object, class_count: int) -> Tensor:
    if (
        type(logits) is not Tensor
        or logits.dtype is not torch.float32
        or logits.layout is not torch.strided
        or not logits.is_contiguous()
        or tuple(logits.shape) != (1, class_count)
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise V1M1C2InputError("C2 operation returned non-canonical logits")
    return logits


def _class_id(logits: object, class_count: int) -> int:
    value = _validate_logits(logits, class_count).argmax(dim=1).item()
    if type(value) is not int or not 0 <= value < class_count:
        raise V1M1C2InputError("C2 class index is not canonical")
    return value


def _validate_r2_model(model: object) -> V1Cifar100ResNet18:
    if type(model) is not V1Cifar100ResNet18:
        raise V1M1C2ConfigError("C2 requires the exact accepted R2 model type")
    typed_model = model
    if typed_model.training:
        raise V1M1C2ConfigError("C2 accepted R2 model must remain in eval mode")
    parameters = tuple(typed_model.parameters())
    if not parameters or any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise V1M1C2ConfigError("C2 accepted R2 parameters must use float32")
    if any(parameter.requires_grad for parameter in parameters):
        raise V1M1C2ConfigError("C2 accepted R2 parameters must be frozen")
    return typed_model


def _validate_public_head(head: V1M1C2PublicHead, cut: V1M1C2Cut, device: torch.device) -> None:
    if type(head) is not V1M1C2PublicHead or head.channels != cut.channels:
        raise V1M1C2ConfigError("C2 public head does not match the trusted cut")
    if head.training:
        raise V1M1C2ConfigError("C2 public head must remain in eval mode")
    parameter = next(head.parameters(), None)
    if parameter is None or parameter.dtype is not torch.float32 or parameter.device != device:
        raise V1M1C2ConfigError("C2 public head device or dtype is not canonical")


class V1M1C2Coordinator:
    """组合唯一 frozen R2 module tree、public head 和 A3-v2 protected entry。"""

    __slots__ = (
        "_adapter",
        "_coordinator",
        "_cut",
        "_emit",
        "_model",
        "_policy",
        "_public_head",
    )

    def __init__(
        self,
        neural_profile: V1NeuralProfile,
        model: V1Cifar100ResNet18,
        *,
        cut: V1M1C2Cut,
        public_head: V1M1C2PublicHead | None = None,
        policy: V1M1C2Policy | None = None,
        store: A3V2TranscriptStore | None = None,
        challenge_sampler: A3V2ChallengeSampler | None = None,
        event_sink: V1M1C2EventSink | None = None,
    ) -> None:
        if type(neural_profile) is not V1NeuralProfile:
            raise V1M1C2ConfigError("C2 requires the exact V1 neural profile")
        if type(cut) is not V1M1C2Cut:
            raise V1M1C2ConfigError("C2 cut must use the exact V1M1C2Cut enum")
        selected_policy = V1M1C2Policy() if policy is None else policy
        if type(selected_policy) is not V1M1C2Policy:
            raise V1M1C2ConfigError("C2 policy must use the exact V1M1C2Policy type")
        if event_sink is not None and not callable(event_sink):
            raise V1M1C2ConfigError("C2 event sink must be callable")
        typed_model = _validate_r2_model(model)
        device = next(typed_model.parameters()).device
        if selected_policy.public_entry_enabled:
            selected_head = (
                V1M1C2PublicHead(cut.channels).to(device).eval()
                if public_head is None
                else public_head
            )
            _validate_public_head(selected_head, cut, device)
        elif public_head is not None:
            raise V1M1C2ConfigError("disabled C2 public entry must not retain a public head")
        else:
            selected_head = None

        self._model = typed_model
        self._cut = cut
        self._policy = selected_policy
        self._public_head = selected_head
        self._adapter = V1M1InputAdapter(neural_profile.identity_id)
        self._emit = (lambda _event: None) if event_sink is None else event_sink

        protected_operation = self._protected_operation
        base_route = build_v1_a3_v2_neural_profile(
            neural_profile,
            model_id=V1_M1_MODEL_ID,
            scope_id=V1_M1_SCOPE_ID,
            input_profile_sha256=V1_M1_INPUT_PROFILE_SHA256,
            protected_operation=protected_operation,
        )
        route = replace(base_route, verifier=self._verifier_with_events(base_route))
        self._coordinator = A3V2ProtocolCoordinator(
            (route,),
            store=store,
            challenge_sampler=challenge_sampler,
        )

    @property
    def cut(self) -> V1M1C2Cut:
        """返回构造时固定的 stage-boundary cut。"""
        return self._cut

    @property
    def public_head(self) -> V1M1C2PublicHead | None:
        """返回可信部署绑定的 public head。"""
        return self._public_head

    def snapshot(self) -> A3V2ProtocolSnapshot:
        """返回不含业务输入或模型输出的 protected transcript 计数。"""
        return self._coordinator.snapshot()

    def _verifier_with_events(self, route: A3V2VerificationProfile) -> Callable[..., object]:
        verifier = route.verifier

        def verify(*args: bytes) -> object:
            evidence = verifier(*args)
            if type(evidence) is A3V2Evidence and evidence.code is A3V2EvidenceCode.RELATION_ACCEPT:
                self._emit("verifier_accept")
            return evidence

        return verify

    def _prefix(self, images: Tensor) -> Tensor:
        output = self._model.stem(images)
        output = self._model.layer1(output)
        if self._cut is V1M1C2Cut.LAYER2:
            return cast(Tensor, self._model.layer2(output))
        output = self._model.layer2(output)
        if self._cut is V1M1C2Cut.LAYER3:
            return cast(Tensor, self._model.layer3(output))
        output = self._model.layer3(output)
        return cast(Tensor, self._model.layer4(output))

    def _suffix(self, features: Tensor) -> Tensor:
        output = features
        if self._cut is V1M1C2Cut.LAYER2:
            output = self._model.layer3(output)
            output = self._model.layer4(output)
        elif self._cut is V1M1C2Cut.LAYER3:
            output = self._model.layer4(output)
        return cast(
            Tensor,
            self._model.classifier(self._model.flatten(self._model.average_pool(output))),
        )

    def _protected_operation(self, snapshot: object) -> Tensor:
        self._emit("coordinator_commit(PROTECTED)")
        try:
            self._emit("preprocess_start")
            if not isinstance(snapshot, _V1M1ImageSnapshot):
                raise V1M1C2InputError("C2 protected snapshot has the wrong type")
            with torch.inference_mode():
                images = _preprocess_v1_m1_snapshot(snapshot).to(
                    next(self._model.parameters()).device
                )
        except Exception as error:
            raise A3V2ProtectedExecutionError("preprocess") from error
        try:
            self._emit("prefix_start")
            with torch.inference_mode():
                features = self._prefix(images)
        except Exception as error:
            raise A3V2ProtectedExecutionError("prefix") from error
        try:
            self._emit("suffix_start")
            with torch.inference_mode():
                logits = self._suffix(features)
        except Exception as error:
            raise A3V2ProtectedExecutionError("suffix") from error
        self._emit("internal_result_commit")
        return logits

    def _public_logits(self, snapshot: object) -> Tensor:
        try:
            self._emit("preprocess_start")
            if not isinstance(snapshot, _V1M1ImageSnapshot):
                raise V1M1C2InputError("C2 public snapshot has the wrong type")
            with torch.inference_mode():
                images = _preprocess_v1_m1_snapshot(snapshot).to(
                    next(self._model.parameters()).device
                )
            self._emit("prefix_start")
            with torch.inference_mode():
                features = self._prefix(images)
            head = self._public_head
            if head is None:
                raise V1M1C2ConfigError("C2 public entry is disabled")
            self._emit("public_head_start")
            with torch.inference_mode():
                return cast(Tensor, head(features))
        except Exception:
            self._emit("public_execution_error")
            raise

    def handle_public(self, image: object, **untrusted_fields: object) -> V1M1C2Envelope:
        """执行可信绑定的 public entry, 拒绝任何请求方 route 字段。"""
        if untrusted_fields:
            return _deny_envelope()
        if not self._policy.public_entry_enabled or self._public_head is None:
            return _deny_envelope()
        try:
            trusted_input = self._adapter.adapt(image)
            class_id = _class_id(
                self._public_logits(trusted_input.snapshot),
                V1_M1_C2_PUBLIC_CLASS_COUNT,
            )
            self._emit("response_release")
            return {
                "version": V1_M1_C2_RESPONSE_VERSION,
                "status": "public",
                "coarse_class_id": class_id,
            }
        except Exception:
            return _deny_envelope()

    def begin_protected(
        self,
        image: object,
        commitment: object,
        **untrusted_fields: object,
    ) -> V1M1C2Envelope:
        """创建 protected challenge, 失败时不调用 verifier 或 public entry。"""
        if untrusted_fields:
            return _deny_envelope()
        try:
            trusted_input = self._adapter.adapt(image)
            return _map_challenge(self._coordinator.begin(trusted_input, commitment))
        except (V1M1InputError, A3V2ProtocolConfigError):
            return _deny_envelope()
        except Exception:
            return _deny_envelope()

    def respond_protected(self, response: object, **untrusted_fields: object) -> V1M1C2Envelope:
        """验证 protected response, 仅成功时释放一个 fine class id。"""
        if untrusted_fields:
            return _deny_envelope()
        result = self._coordinator.commit_and_execute(response)
        if (
            result.route_decision is not A3V2RouteDecision.PROTECTED
            or result.execution_state is not A3V2ExecutionState.SUCCEEDED
            or result.code is not A3V2InternalResultCode.PROTECTED_SUCCEEDED
        ):
            if result.execution_state is A3V2ExecutionState.FAILED:
                self._emit(f"protected_execution_error:{result.failure_stage or 'unknown'}")
            return _deny_envelope()
        try:
            class_id = _class_id(
                result.consume_operation_value(),
                V1_M1_C2_PROTECTED_CLASS_COUNT,
            )
        except Exception:
            self._emit("protected_result_extraction_error")
            return _deny_envelope()
        self._emit("response_release")
        return {
            "version": V1_M1_C2_RESPONSE_VERSION,
            "status": "protected",
            "class_id": class_id,
        }

    def abort_protected(self, abort: object, **untrusted_fields: object) -> V1M1C2DenyEnvelope:
        """终结 protected transcript, 固定返回 C2 deny。"""
        if untrusted_fields:
            return _deny_envelope()
        try:
            self._coordinator.abort(abort)
        except Exception:
            pass
        return _deny_envelope()


__all__ = [
    "V1_M1_C2_PROTECTED_CLASS_COUNT",
    "V1_M1_C2_PUBLIC_CLASS_COUNT",
    "V1_M1_C2_RESPONSE_VERSION",
    "V1M1C2ChallengeEnvelope",
    "V1M1C2ConfigError",
    "V1M1C2Coordinator",
    "V1M1C2Cut",
    "V1M1C2DenyEnvelope",
    "V1M1C2Envelope",
    "V1M1C2InputError",
    "V1M1C2Policy",
    "V1M1C2ProtectedEnvelope",
    "V1M1C2PublicEnvelope",
    "V1M1C2PublicHead",
]
