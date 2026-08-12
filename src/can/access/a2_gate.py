"""A2-E1 单一协调器与二元前置硬门控。"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Final, Literal, TypeAlias, TypedDict, cast

import torch
from torch import Tensor, nn

from can.model.a2_mlp import A2_CLASS_COUNT, A2FashionMNISTMLP, validate_a2_images
from can.verifier import A1Evidence, A1EvidenceCode
from can.verifier.a1_torch import A1TorchBackend, verify_a1_torch

A2_ACCESS_RESPONSE_VERSION: Final = 1
A2_TIMING_SAMPLE_LIMIT: Final = 4_096

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


class A2DenyEnvelope(TypedDict):
    """定义不泄露内部拒绝原因的固定外部响应。"""

    version: Literal[1]
    status: Literal["deny"]


class A2AllowEnvelope(TypedDict):
    """定义只释放一个 top-1 类别的固定外部响应。"""

    version: Literal[1]
    status: Literal["ok"]
    class_id: int


A2AccessEnvelope: TypeAlias = A2DenyEnvelope | A2AllowEnvelope


class A2CoordinatorConfigError(ValueError):
    """表示本地可信协调器配置不满足 A2-E1 固定契约。"""


@dataclass(frozen=True, slots=True)
class A2AccessSnapshot:
    """提供不含凭据、图像或 evidence 的内部调用计数快照。"""

    verifier_calls: int
    coordinator_commits: int
    allow_commits: int
    deny_commits: int
    protected_model_calls: int
    ok_responses: int
    deny_responses: int


@dataclass(frozen=True, slots=True)
class A2TimingSample:
    """记录单次请求的不敏感内部阶段耗时。"""

    response_ok: bool
    allow_committed: bool
    validation_ns: int
    verifier_ns: int
    coordinator_ns: int
    protected_model_ns: int
    total_ns: int


def _deny_envelope() -> A2DenyEnvelope:
    return {"version": 1, "status": "deny"}


def _allow_envelope(class_id: int) -> A2AllowEnvelope:
    if type(class_id) is not int or not 0 <= class_id < A2_CLASS_COUNT:
        raise RuntimeError("protected model returned a non-canonical class index")
    return {"version": 1, "status": "ok", "class_id": class_id}


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


def _validate_protected_model(model: object, *, check_values: bool) -> A2FashionMNISTMLP:
    if type(model) is not A2FashionMNISTMLP:
        raise A2CoordinatorConfigError("protected model must be exactly A2FashionMNISTMLP")
    typed_model = model
    if typed_model.training:
        raise A2CoordinatorConfigError("protected model must remain in evaluation mode")
    if tuple(type(module) for module in typed_model.modules()) != _EXPECTED_MODULE_TYPES:
        raise A2CoordinatorConfigError("protected model topology changed")
    if tuple(typed_model.buffers()):
        raise A2CoordinatorConfigError("protected model must not contain buffers")
    if any(_has_module_hooks(module) for module in typed_model.modules()):
        raise A2CoordinatorConfigError("protected model hooks are unsupported")

    parameters = dict(typed_model.named_parameters())
    if set(parameters) != set(_EXPECTED_PARAMETERS):
        raise A2CoordinatorConfigError("protected model parameter set changed")
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
        ):
            raise A2CoordinatorConfigError("protected model parameter metadata changed")
        if check_values and not bool(torch.isfinite(parameter).all().item()):
            raise A2CoordinatorConfigError("protected model parameters must be finite")
    return typed_model


def _canonicalize_public_images(images: object) -> Tensor:
    validate_a2_images(cast(Tensor, images))
    typed_images = cast(Tensor, images)
    if typed_images.shape[0] != 1:
        raise ValueError("the public A2-E1 response supports exactly one image")
    snapshot = typed_images.detach().clone(memory_format=torch.contiguous_format)
    validate_a2_images(snapshot)
    return snapshot


class A2AccessCoordinator:
    """以唯一提交点把固定 A1-B1 evidence 转换为一次 MLP 调用。"""

    __slots__ = (
        "_allow_commits",
        "_backend",
        "_coordinator_commits",
        "_deny_commits",
        "_deny_responses",
        "_lock",
        "_model",
        "_ok_responses",
        "_protected_model_calls",
        "_timings",
        "_verifier_calls",
    )

    def __init__(self, backend: A1TorchBackend, model: A2FashionMNISTMLP) -> None:
        if type(backend) is not A1TorchBackend or not backend.active:
            raise A2CoordinatorConfigError("coordinator requires one active A1-B1 backend")
        self._backend = backend
        self._model = _validate_protected_model(model, check_values=True)
        self._lock = Lock()
        self._verifier_calls = 0
        self._coordinator_commits = 0
        self._allow_commits = 0
        self._deny_commits = 0
        self._protected_model_calls = 0
        self._ok_responses = 0
        self._deny_responses = 0
        self._timings: deque[A2TimingSample] = deque(maxlen=A2_TIMING_SAMPLE_LIMIT)

    def snapshot(self) -> A2AccessSnapshot:
        """返回线程安全且不影响授权的内部计数快照。"""
        with self._lock:
            return A2AccessSnapshot(
                verifier_calls=self._verifier_calls,
                coordinator_commits=self._coordinator_commits,
                allow_commits=self._allow_commits,
                deny_commits=self._deny_commits,
                protected_model_calls=self._protected_model_calls,
                ok_responses=self._ok_responses,
                deny_responses=self._deny_responses,
            )

    def timing_snapshot(self) -> tuple[A2TimingSample, ...]:
        """返回有界、不含请求内容的内部计时样本。"""
        with self._lock:
            return tuple(self._timings)

    def _increment_verifier_calls(self) -> None:
        with self._lock:
            self._verifier_calls += 1

    def _commit(self, allow: bool) -> None:
        if type(allow) is not bool:
            raise RuntimeError("coordinator decision must be exactly bool")
        with self._lock:
            self._coordinator_commits += 1
            if allow:
                self._allow_commits += 1
            else:
                self._deny_commits += 1

    def _increment_protected_calls(self) -> None:
        with self._lock:
            self._protected_model_calls += 1

    def _record_response(self, *, ok: bool) -> None:
        with self._lock:
            if ok:
                self._ok_responses += 1
            else:
                self._deny_responses += 1

    def _record_timing(
        self,
        *,
        response_ok: bool,
        allow_committed: bool,
        validation_ns: int,
        verifier_ns: int,
        protected_model_ns: int,
        total_ns: int,
    ) -> None:
        coordinator_ns = max(0, total_ns - validation_ns - verifier_ns - protected_model_ns)
        sample = A2TimingSample(
            response_ok=response_ok,
            allow_committed=allow_committed,
            validation_ns=validation_ns,
            verifier_ns=verifier_ns,
            coordinator_ns=coordinator_ns,
            protected_model_ns=protected_model_ns,
            total_ns=total_ns,
        )
        with self._lock:
            self._timings.append(sample)

    def _finish_deny(
        self,
        *,
        total_start: int,
        validation_ns: int,
        verifier_ns: int,
        allow_committed: bool = False,
        protected_model_ns: int = 0,
    ) -> A2DenyEnvelope:
        if not allow_committed:
            self._commit(False)
        self._record_response(ok=False)
        total_ns = time.perf_counter_ns() - total_start
        self._record_timing(
            response_ok=False,
            allow_committed=allow_committed,
            validation_ns=validation_ns,
            verifier_ns=verifier_ns,
            protected_model_ns=protected_model_ns,
            total_ns=total_ns,
        )
        return _deny_envelope()

    def handle(
        self,
        images: object = None,
        raw_credential: object = None,
        **untrusted_fields: object,
    ) -> A2AccessEnvelope:
        """处理原始图像与 credential, 额外字段或任何失败均返回固定 deny。"""
        total_start = time.perf_counter_ns()
        validation_ns = 0
        verifier_ns = 0

        if untrusted_fields:
            return self._finish_deny(
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )

        validation_start = time.perf_counter_ns()
        try:
            canonical_images = _canonicalize_public_images(images)
        except Exception:
            validation_ns = time.perf_counter_ns() - validation_start
            return self._finish_deny(
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
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )
        verifier_ns = time.perf_counter_ns() - verifier_start

        # 受保护模型在构造时已校验并冻结 (见 __init__), 且属于可信不可变配置;
        # 按威胁模型不在请求路径重复校验其拓扑/权重 (参见 AGENTS.md Threat-model alignment)。
        model = self._model
        if type(evidence) is not A1Evidence or evidence.code is not A1EvidenceCode.NUMERIC_ACCEPT:
            return self._finish_deny(
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
            )

        self._commit(True)
        self._increment_protected_calls()
        model_start = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                logits = model(canonical_images)
                class_value = logits.argmax(dim=1).item()
            protected_model_ns = time.perf_counter_ns() - model_start
            if type(class_value) is not int:
                raise RuntimeError("protected model class index type changed")
            response = _allow_envelope(class_value)
        except Exception:
            protected_model_ns = time.perf_counter_ns() - model_start
            return self._finish_deny(
                total_start=total_start,
                validation_ns=validation_ns,
                verifier_ns=verifier_ns,
                allow_committed=True,
                protected_model_ns=protected_model_ns,
            )

        self._record_response(ok=True)
        total_ns = time.perf_counter_ns() - total_start
        self._record_timing(
            response_ok=True,
            allow_committed=True,
            validation_ns=validation_ns,
            verifier_ns=verifier_ns,
            protected_model_ns=protected_model_ns,
            total_ns=total_ns,
        )
        return response
