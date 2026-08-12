"""A2-E2 本地绑定的 public/protected 三态协调器。"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Final, Literal, TypeAlias, TypedDict, cast

import torch
from torch import Tensor, nn

from can.model.a2_mlp import A2_CLASS_COUNT, A2FashionMNISTMLP, validate_a2_images
from can.model.a2_public_mlp import (
    A2_PUBLIC_CLASS_COUNT,
    A2FashionMNISTPublicMLP,
    validate_a2_public_images,
)
from can.verifier import A1Evidence, A1EvidenceCode
from can.verifier.a1_torch import A1TorchBackend, verify_a1_torch

A2_CAPABILITY_RESPONSE_VERSION: Final = 2
A2_CAPABILITY_POLICY_VERSION: Final = "CAN-A2-CAPABILITY-POLICY-v1"
A2_CAPABILITY_TIMING_SAMPLE_LIMIT: Final = 4_096

_EXPECTED_PROTECTED_MODULE_TYPES: Final = (
    A2FashionMNISTMLP,
    nn.Sequential,
    nn.Flatten,
    nn.Linear,
    nn.ReLU,
    nn.Linear,
    nn.ReLU,
    nn.Linear,
)
_EXPECTED_PROTECTED_PARAMETERS: Final = {
    "_network.1.weight": (256, 784),
    "_network.1.bias": (256,),
    "_network.3.weight": (128, 256),
    "_network.3.bias": (128,),
    "_network.5.weight": (A2_CLASS_COUNT, 128),
    "_network.5.bias": (A2_CLASS_COUNT,),
}
_EXPECTED_PUBLIC_MODULE_TYPES: Final = (
    A2FashionMNISTPublicMLP,
    nn.Sequential,
    nn.Flatten,
    nn.Linear,
    nn.ReLU,
    nn.Linear,
)
_EXPECTED_PUBLIC_PARAMETERS: Final = {
    "_network.1.weight": (64, 784),
    "_network.1.bias": (64,),
    "_network.3.weight": (A2_PUBLIC_CLASS_COUNT, 64),
    "_network.3.bias": (A2_PUBLIC_CLASS_COUNT,),
}


class A2CapabilityDenyEnvelope(TypedDict):
    """定义 A2-E2 不泄露失败原因的固定 deny 响应。"""

    version: Literal[2]
    status: Literal["deny"]


class A2PublicEnvelope(TypedDict):
    """定义只释放一个 public coarse class 的固定响应。"""

    version: Literal[2]
    status: Literal["public"]
    coarse_class_id: int


class A2ProtectedEnvelope(TypedDict):
    """定义只释放一个 protected top-1 class 的固定响应。"""

    version: Literal[2]
    status: Literal["protected"]
    class_id: int


A2CapabilityEnvelope: TypeAlias = A2CapabilityDenyEnvelope | A2PublicEnvelope | A2ProtectedEnvelope


class A2CapabilityDecision(Enum):
    """表示协调器唯一提交点的内部互斥三态决定。"""

    DENY = "deny"
    PUBLIC = "public"
    PROTECTED = "protected"


class A2CapabilityConfigError(ValueError):
    """表示本地可信 A2-E2 配置不满足固定契约。"""


@dataclass(frozen=True, slots=True)
class A2CapabilityPolicy:
    """定义默认关闭且只能在构造时设置的本地 public entry 策略。"""

    public_entry_enabled: bool = False

    def __post_init__(self) -> None:
        if type(self.public_entry_enabled) is not bool:
            raise A2CapabilityConfigError("public_entry_enabled must be exactly bool")


@dataclass(frozen=True, slots=True)
class A2CapabilityStartupEvent:
    """记录不含请求内容的本地策略启动审计事件。"""

    event_version: int
    policy_version: str
    event_code: Literal["PUBLIC_ENTRY_DISABLED", "PUBLIC_ENTRY_ENABLED"]
    public_entry_enabled: bool


@dataclass(frozen=True, slots=True)
class A2CapabilitySnapshot:
    """提供不含输入、凭据、evidence 或模型输出的三态计数。"""

    verifier_calls: int
    coordinator_commits: int
    deny_commits: int
    public_commits: int
    protected_commits: int
    public_model_calls: int
    protected_model_calls: int
    deny_responses: int
    public_responses: int
    protected_responses: int


@dataclass(frozen=True, slots=True)
class A2CapabilityTimingSample:
    """记录单次三态请求的不敏感内部阶段耗时。"""

    entry: Literal["public", "protected"]
    response_status: Literal["deny", "public", "protected"]
    committed_decision: A2CapabilityDecision
    validation_ns: int
    verifier_ns: int
    coordinator_ns: int
    public_model_ns: int
    protected_model_ns: int
    total_ns: int


def _deny_envelope() -> A2CapabilityDenyEnvelope:
    return {"version": 2, "status": "deny"}


def _public_envelope(coarse_class_id: int) -> A2PublicEnvelope:
    if type(coarse_class_id) is not int or not 0 <= coarse_class_id < A2_PUBLIC_CLASS_COUNT:
        raise RuntimeError("public model returned a non-canonical coarse class index")
    return {"version": 2, "status": "public", "coarse_class_id": coarse_class_id}


def _protected_envelope(class_id: int) -> A2ProtectedEnvelope:
    if type(class_id) is not int or not 0 <= class_id < A2_CLASS_COUNT:
        raise RuntimeError("protected model returned a non-canonical class index")
    return {"version": 2, "status": "protected", "class_id": class_id}


def _has_module_hooks(module: nn.Module) -> bool:
    return any(
        hooks
        for hooks in (
            module._forward_hooks,
            module._forward_pre_hooks,
            module._backward_hooks,
            module._backward_pre_hooks,
        )
    )


def _validate_model_parameters(
    model: nn.Module,
    expected_parameters: dict[str, tuple[int, ...]],
    *,
    check_values: bool,
    model_name: str,
) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(expected_parameters):
        raise A2CapabilityConfigError(f"{model_name} parameter set changed")
    for name, expected_shape in expected_parameters.items():
        parameter = parameters[name]
        if (
            type(parameter) is not nn.Parameter
            or parameter.dtype is not torch.float32
            or parameter.device.type != "cpu"
            or parameter.device.index is not None
            or parameter.layout is not torch.strided
            or tuple(parameter.shape) != expected_shape
            or not parameter.is_contiguous()
        ):
            raise A2CapabilityConfigError(f"{model_name} parameter metadata changed")
        if check_values and not bool(torch.isfinite(parameter).all().item()):
            raise A2CapabilityConfigError(f"{model_name} parameters must be finite")


def _validate_protected_model(model: object, *, check_values: bool) -> A2FashionMNISTMLP:
    if type(model) is not A2FashionMNISTMLP:
        raise A2CapabilityConfigError("protected model must be exactly A2FashionMNISTMLP")
    typed_model = model
    if typed_model.training:
        raise A2CapabilityConfigError("protected model must remain in evaluation mode")
    if tuple(type(module) for module in typed_model.modules()) != _EXPECTED_PROTECTED_MODULE_TYPES:
        raise A2CapabilityConfigError("protected model topology changed")
    if tuple(typed_model.buffers()):
        raise A2CapabilityConfigError("protected model must not contain buffers")
    if any(_has_module_hooks(module) for module in typed_model.modules()):
        raise A2CapabilityConfigError("protected model hooks are unsupported")
    _validate_model_parameters(
        typed_model,
        _EXPECTED_PROTECTED_PARAMETERS,
        check_values=check_values,
        model_name="protected model",
    )
    return typed_model


def _validate_public_model(model: object, *, check_values: bool) -> A2FashionMNISTPublicMLP:
    if type(model) is not A2FashionMNISTPublicMLP:
        raise A2CapabilityConfigError("public model must be exactly A2FashionMNISTPublicMLP")
    typed_model = model
    if typed_model.training:
        raise A2CapabilityConfigError("public model must remain in evaluation mode")
    if tuple(type(module) for module in typed_model.modules()) != _EXPECTED_PUBLIC_MODULE_TYPES:
        raise A2CapabilityConfigError("public model topology changed")
    if tuple(typed_model.buffers()):
        raise A2CapabilityConfigError("public model must not contain buffers")
    if any(_has_module_hooks(module) for module in typed_model.modules()):
        raise A2CapabilityConfigError("public model hooks are unsupported")
    _validate_model_parameters(
        typed_model,
        _EXPECTED_PUBLIC_PARAMETERS,
        check_values=check_values,
        model_name="public model",
    )
    return typed_model


def _validate_independent_storage(
    protected_model: A2FashionMNISTMLP, public_model: A2FashionMNISTPublicMLP
) -> None:
    protected_storage = {
        parameter.untyped_storage().data_ptr() for parameter in protected_model.parameters()
    }
    public_storage = {
        parameter.untyped_storage().data_ptr() for parameter in public_model.parameters()
    }
    if not protected_storage.isdisjoint(public_storage):
        raise A2CapabilityConfigError("public and protected model parameters share storage")


def _validate_logits(logits: object, class_count: int, *, model_name: str) -> Tensor:
    if (
        type(logits) is not Tensor
        or logits.dtype is not torch.float32
        or logits.device.type != "cpu"
        or logits.device.index is not None
        or logits.layout is not torch.strided
        or not logits.is_contiguous()
        or tuple(logits.shape) != (1, class_count)
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise RuntimeError(f"{model_name} returned non-canonical logits")
    return logits


def _canonicalize_public_images(images: object) -> Tensor:
    validate_a2_public_images(cast(Tensor, images))
    typed_images = cast(Tensor, images)
    if typed_images.shape[0] != 1:
        raise ValueError("the A2-E2 public entry supports exactly one image")
    snapshot = typed_images.detach().clone(memory_format=torch.contiguous_format)
    validate_a2_public_images(snapshot)
    return snapshot


def _canonicalize_protected_images(images: object) -> Tensor:
    validate_a2_images(cast(Tensor, images))
    typed_images = cast(Tensor, images)
    if typed_images.shape[0] != 1:
        raise ValueError("the A2-E2 protected entry supports exactly one image")
    snapshot = typed_images.detach().clone(memory_format=torch.contiguous_format)
    validate_a2_images(snapshot)
    return snapshot


class A2CapabilityCoordinator:
    """作为唯一提交点处理本地绑定的 public/protected entries。"""

    __slots__ = (
        "_backend",
        "_coordinator_commits",
        "_deny_commits",
        "_deny_responses",
        "_lock",
        "_policy",
        "_protected_commits",
        "_protected_model",
        "_protected_model_calls",
        "_protected_responses",
        "_public_commits",
        "_public_model",
        "_public_model_calls",
        "_public_responses",
        "_startup_event",
        "_timings",
        "_verifier_calls",
    )

    def __init__(
        self,
        backend: A1TorchBackend,
        protected_model: A2FashionMNISTMLP,
        *,
        public_model: A2FashionMNISTPublicMLP | None = None,
        policy: A2CapabilityPolicy | None = None,
    ) -> None:
        if type(backend) is not A1TorchBackend or not backend.active:
            raise A2CapabilityConfigError("coordinator requires one active A1-B1 backend")
        selected_policy = A2CapabilityPolicy() if policy is None else policy
        if type(selected_policy) is not A2CapabilityPolicy:
            raise A2CapabilityConfigError("policy must be exactly A2CapabilityPolicy")
        typed_protected_model = _validate_protected_model(protected_model, check_values=True)
        if selected_policy.public_entry_enabled:
            typed_public_model = _validate_public_model(public_model, check_values=True)
            _validate_independent_storage(typed_protected_model, typed_public_model)
        elif public_model is not None:
            raise A2CapabilityConfigError("disabled public entry must not retain a public model")
        else:
            typed_public_model = None

        self._backend = backend
        self._protected_model = typed_protected_model
        self._public_model = typed_public_model
        self._policy = selected_policy
        event_code: Literal["PUBLIC_ENTRY_DISABLED", "PUBLIC_ENTRY_ENABLED"] = (
            "PUBLIC_ENTRY_ENABLED"
            if selected_policy.public_entry_enabled
            else "PUBLIC_ENTRY_DISABLED"
        )
        self._startup_event = A2CapabilityStartupEvent(
            event_version=1,
            policy_version=A2_CAPABILITY_POLICY_VERSION,
            event_code=event_code,
            public_entry_enabled=selected_policy.public_entry_enabled,
        )
        self._lock = Lock()
        self._verifier_calls = 0
        self._coordinator_commits = 0
        self._deny_commits = 0
        self._public_commits = 0
        self._protected_commits = 0
        self._public_model_calls = 0
        self._protected_model_calls = 0
        self._deny_responses = 0
        self._public_responses = 0
        self._protected_responses = 0
        self._timings: deque[A2CapabilityTimingSample] = deque(
            maxlen=A2_CAPABILITY_TIMING_SAMPLE_LIMIT
        )

    def startup_audit_event(self) -> A2CapabilityStartupEvent:
        """返回构造时固定且不含请求内容的本地策略事件。"""
        return self._startup_event

    def snapshot(self) -> A2CapabilitySnapshot:
        """返回线程安全且不影响决定的三态计数快照。"""
        with self._lock:
            return A2CapabilitySnapshot(
                verifier_calls=self._verifier_calls,
                coordinator_commits=self._coordinator_commits,
                deny_commits=self._deny_commits,
                public_commits=self._public_commits,
                protected_commits=self._protected_commits,
                public_model_calls=self._public_model_calls,
                protected_model_calls=self._protected_model_calls,
                deny_responses=self._deny_responses,
                public_responses=self._public_responses,
                protected_responses=self._protected_responses,
            )

    def timing_snapshot(self) -> tuple[A2CapabilityTimingSample, ...]:
        """返回有界且不含输入、凭据或模型输出的计时样本。"""
        with self._lock:
            return tuple(self._timings)

    def _increment_verifier_calls(self) -> None:
        with self._lock:
            self._verifier_calls += 1

    def _commit_selected_model(self, decision: A2CapabilityDecision) -> None:
        if type(decision) is not A2CapabilityDecision:
            raise RuntimeError("selected model decision type changed")
        with self._lock:
            self._coordinator_commits += 1
            if decision is A2CapabilityDecision.PUBLIC:
                self._public_commits += 1
                self._public_model_calls += 1
            elif decision is A2CapabilityDecision.PROTECTED:
                self._protected_commits += 1
                self._protected_model_calls += 1
            else:
                raise RuntimeError("deny cannot invoke a model")

    def _commit(self, decision: A2CapabilityDecision) -> None:
        if type(decision) is not A2CapabilityDecision:
            raise RuntimeError("coordinator decision type changed")
        with self._lock:
            self._coordinator_commits += 1
            if decision is A2CapabilityDecision.DENY:
                self._deny_commits += 1
            elif decision is A2CapabilityDecision.PUBLIC:
                self._public_commits += 1
            elif decision is A2CapabilityDecision.PROTECTED:
                self._protected_commits += 1

    def _record_completion(
        self,
        *,
        entry: Literal["public", "protected"],
        response_status: Literal["deny", "public", "protected"],
        committed_decision: A2CapabilityDecision,
        validation_ns: int,
        verifier_ns: int,
        public_model_ns: int,
        protected_model_ns: int,
        total_start: int,
    ) -> None:
        total_ns = time.perf_counter_ns() - total_start
        coordinator_ns = max(
            0,
            total_ns - validation_ns - verifier_ns - public_model_ns - protected_model_ns,
        )
        sample = A2CapabilityTimingSample(
            entry=entry,
            response_status=response_status,
            committed_decision=committed_decision,
            validation_ns=validation_ns,
            verifier_ns=verifier_ns,
            coordinator_ns=coordinator_ns,
            public_model_ns=public_model_ns,
            protected_model_ns=protected_model_ns,
            total_ns=total_ns,
        )
        with self._lock:
            if response_status == "deny":
                self._deny_responses += 1
            elif response_status == "public":
                self._public_responses += 1
            else:
                self._protected_responses += 1
            self._timings.append(sample)

    def _finish_deny(
        self,
        *,
        entry: Literal["public", "protected"],
        total_start: int,
        validation_ns: int,
        verifier_ns: int,
        committed_decision: A2CapabilityDecision | None = None,
        public_model_ns: int = 0,
        protected_model_ns: int = 0,
    ) -> A2CapabilityDenyEnvelope:
        decision = committed_decision
        if decision is None:
            decision = A2CapabilityDecision.DENY
            self._commit(decision)
        self._record_completion(
            entry=entry,
            response_status="deny",
            committed_decision=decision,
            validation_ns=validation_ns,
            verifier_ns=verifier_ns,
            public_model_ns=public_model_ns,
            protected_model_ns=protected_model_ns,
            total_start=total_start,
        )
        return _deny_envelope()

    def _validate_runtime_policy(self) -> A2CapabilityPolicy:
        if type(self._policy) is not A2CapabilityPolicy:
            raise A2CapabilityConfigError("runtime policy type changed")
        expected_code: Literal["PUBLIC_ENTRY_DISABLED", "PUBLIC_ENTRY_ENABLED"] = (
            "PUBLIC_ENTRY_ENABLED" if self._policy.public_entry_enabled else "PUBLIC_ENTRY_DISABLED"
        )
        if self._startup_event != A2CapabilityStartupEvent(
            event_version=1,
            policy_version=A2_CAPABILITY_POLICY_VERSION,
            event_code=expected_code,
            public_entry_enabled=self._policy.public_entry_enabled,
        ):
            raise A2CapabilityConfigError("runtime policy changed after startup")
        return self._policy

    def handle_public(
        self,
        images: object = None,
        **untrusted_fields: object,
    ) -> A2CapabilityEnvelope:
        """处理本地绑定的 public entry, payload 额外字段或任何失败均 deny。"""
        total_start = time.perf_counter_ns()
        validation_ns = 0

        if untrusted_fields:
            return self._finish_deny(
                entry="public",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=0,
            )

        validation_start = time.perf_counter_ns()
        try:
            canonical_images = _canonicalize_public_images(images)
        except Exception:
            validation_ns = time.perf_counter_ns() - validation_start
            return self._finish_deny(
                entry="public",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=0,
            )
        validation_ns = time.perf_counter_ns() - validation_start

        try:
            policy = self._validate_runtime_policy()
            if not policy.public_entry_enabled:
                raise A2CapabilityConfigError("public entry is disabled")
            # 两个模型在构造时已校验, 冻结并确认存储独立; 按威胁模型不在请求路径重复校验
            # 可信不可变配置 (参见 AGENTS.md Threat-model alignment)。
            public_model = cast(A2FashionMNISTPublicMLP, self._public_model)
        except Exception:
            return self._finish_deny(
                entry="public",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=0,
            )

        decision = A2CapabilityDecision.PUBLIC
        self._commit_selected_model(decision)
        model_start = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                logits = _validate_logits(
                    public_model(canonical_images),
                    A2_PUBLIC_CLASS_COUNT,
                    model_name="public model",
                )
                class_value = logits.argmax(dim=1).item()
            public_model_ns = time.perf_counter_ns() - model_start
            if type(class_value) is not int:
                raise RuntimeError("public model class index type changed")
            response = _public_envelope(class_value)
        except Exception:
            public_model_ns = time.perf_counter_ns() - model_start
            return self._finish_deny(
                entry="public",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=0,
                committed_decision=decision,
                public_model_ns=public_model_ns,
            )

        self._record_completion(
            entry="public",
            response_status="public",
            committed_decision=decision,
            validation_ns=validation_ns,
            verifier_ns=0,
            public_model_ns=public_model_ns,
            protected_model_ns=0,
            total_start=total_start,
        )
        return response

    def handle_protected(
        self,
        images: object = None,
        raw_credential: object = None,
        **untrusted_fields: object,
    ) -> A2CapabilityEnvelope:
        """处理本地绑定的 protected entry, 验证失败绝不降级到 public。"""
        total_start = time.perf_counter_ns()
        validation_ns = 0
        verifier_ns = 0

        if untrusted_fields:
            return self._finish_deny(
                entry="protected",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )

        validation_start = time.perf_counter_ns()
        try:
            canonical_images = _canonicalize_protected_images(images)
        except Exception:
            validation_ns = time.perf_counter_ns() - validation_start
            return self._finish_deny(
                entry="protected",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )
        validation_ns = time.perf_counter_ns() - validation_start

        verifier_start = time.perf_counter_ns()
        self._increment_verifier_calls()
        try:
            evidence = verify_a1_torch(cast(bytes, raw_credential), self._backend)
        except Exception:
            verifier_ns = time.perf_counter_ns() - verifier_start
            return self._finish_deny(
                entry="protected",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )
        verifier_ns = time.perf_counter_ns() - verifier_start

        try:
            self._validate_runtime_policy()
            # 受保护模型在构造时已校验并冻结; 按威胁模型不在请求路径重复校验可信不可变配置
            # (参见 AGENTS.md Threat-model alignment)。
            protected_model = self._protected_model
        except Exception:
            return self._finish_deny(
                entry="protected",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )
        if type(evidence) is not A1Evidence or evidence.code is not A1EvidenceCode.NUMERIC_ACCEPT:
            return self._finish_deny(
                entry="protected",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )

        decision = A2CapabilityDecision.PROTECTED
        self._commit_selected_model(decision)
        model_start = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                logits = _validate_logits(
                    protected_model(canonical_images),
                    A2_CLASS_COUNT,
                    model_name="protected model",
                )
                class_value = logits.argmax(dim=1).item()
            protected_model_ns = time.perf_counter_ns() - model_start
            if type(class_value) is not int:
                raise RuntimeError("protected model class index type changed")
            response = _protected_envelope(class_value)
        except Exception:
            protected_model_ns = time.perf_counter_ns() - model_start
            return self._finish_deny(
                entry="protected",
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
                committed_decision=decision,
                protected_model_ns=protected_model_ns,
            )

        self._record_completion(
            entry="protected",
            response_status="protected",
            committed_decision=decision,
            validation_ns=validation_ns,
            verifier_ns=verifier_ns,
            public_model_ns=0,
            protected_model_ns=protected_model_ns,
            total_start=total_start,
        )
        return response
