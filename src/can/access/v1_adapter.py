"""把 V1-P2 exact relation 适配为 A3-v2 transcript-bound evidence。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from can.access.a3_v2 import (
    A3V2Evidence,
    A3V2EvidenceCode,
    A3V2ProtectedOperation,
    A3V2ProtocolConfigError,
    A3V2VerificationProfile,
)
from can.reference.v1 import (
    V1EvidenceCode,
    V1PublicProfile,
    verify_v1_ref,
)
from can.verifier.v1 import V1NeuralEvidenceCode, V1NeuralProfile, verify_v1_neural


@dataclass(frozen=True, slots=True)
class V1ReferenceAdapter:
    """绑定一个无 secret 的 V1-P2 公开 profile 并产生 A3-v2 evidence。"""

    profile: V1PublicProfile

    def __post_init__(self) -> None:
        if type(self.profile) is not V1PublicProfile:
            raise A3V2ProtocolConfigError("V1 exact adapter requires the exact public profile type")

    def __call__(
        self,
        commitment: bytes,
        challenge: bytes,
        response: bytes,
        transcript_id: bytes,
    ) -> A3V2Evidence:
        """执行 exact relation 并返回不具有授权能力的公开摘要证据。"""
        reference = verify_v1_ref(
            commitment,
            challenge,
            response,
            transcript_id,
            self.profile,
        )
        if reference.code is V1EvidenceCode.RELATION_ACCEPT:
            code = A3V2EvidenceCode.RELATION_ACCEPT
        elif reference.code is V1EvidenceCode.CONFIG_REJECT:
            code = A3V2EvidenceCode.CONFIG_REJECT
        else:
            code = A3V2EvidenceCode.RELATION_REJECT
        return A3V2Evidence(
            code=code,
            identity_id=self.profile.identity_id,
            verification_profile_sha256=self.profile.public_key_sha256,
            commitment_sha256=hashlib.sha256(commitment).digest(),
            challenge_sha256=hashlib.sha256(challenge).digest(),
            response_sha256=hashlib.sha256(response).digest(),
            transcript_id=transcript_id,
        )


@dataclass(frozen=True, slots=True)
class V1NeuralAdapter:
    """绑定一个 V1-C1 compiled profile 并产生 A3-v2 evidence。"""

    profile: V1NeuralProfile

    def __post_init__(self) -> None:
        if type(self.profile) is not V1NeuralProfile:
            raise A3V2ProtocolConfigError(
                "V1 neural adapter requires the exact compiled profile type"
            )

    def __call__(
        self,
        commitment: bytes,
        challenge: bytes,
        response: bytes,
        transcript_id: bytes,
    ) -> A3V2Evidence:
        """执行固定 neural relation 并返回不具有授权能力的绑定证据。"""
        neural = verify_v1_neural(commitment, challenge, response, transcript_id, self.profile)
        if neural.code is V1NeuralEvidenceCode.NEURAL_ACCEPT:
            code = A3V2EvidenceCode.RELATION_ACCEPT
        elif neural.code is V1NeuralEvidenceCode.CONFIG_REJECT:
            code = A3V2EvidenceCode.CONFIG_REJECT
        else:
            code = A3V2EvidenceCode.RELATION_REJECT
        return A3V2Evidence(
            code=code,
            identity_id=neural.identity_id,
            verification_profile_sha256=self.profile.public_profile.public_key_sha256,
            commitment_sha256=hashlib.sha256(commitment).digest(),
            challenge_sha256=hashlib.sha256(challenge).digest(),
            response_sha256=hashlib.sha256(response).digest(),
            transcript_id=transcript_id,
        )


def build_v1_a3_v2_profile(
    profile: V1PublicProfile,
    *,
    model_id: int,
    scope_id: int,
    input_profile_sha256: bytes,
    protected_operation: A3V2ProtectedOperation,
) -> A3V2VerificationProfile:
    """从本地 V1 公开参数和业务 profile 构造 A3-v2 route。"""
    if type(profile) is not V1PublicProfile:
        raise A3V2ProtocolConfigError("V1 route builder requires the exact public profile")
    return A3V2VerificationProfile(
        identity_id=profile.identity_id,
        verification_profile_sha256=profile.public_key_sha256,
        model_id=model_id,
        scope_id=scope_id,
        input_profile_sha256=input_profile_sha256,
        verifier=V1ReferenceAdapter(profile),
        protected_operation=protected_operation,
    )


def build_v1_a3_v2_neural_profile(
    profile: V1NeuralProfile,
    *,
    model_id: int,
    scope_id: int,
    input_profile_sha256: bytes,
    protected_operation: A3V2ProtectedOperation,
) -> A3V2VerificationProfile:
    """从本地 V1-C1 profile 构造独立的 A3-v2 neural route。"""
    if type(profile) is not V1NeuralProfile:
        raise A3V2ProtocolConfigError("V1 neural route builder requires the exact compiled profile")
    return A3V2VerificationProfile(
        identity_id=profile.identity_id,
        verification_profile_sha256=profile.public_profile.public_key_sha256,
        model_id=model_id,
        scope_id=scope_id,
        input_profile_sha256=input_profile_sha256,
        verifier=V1NeuralAdapter(profile),
        protected_operation=protected_operation,
    )


__all__ = [
    "V1NeuralAdapter",
    "V1ReferenceAdapter",
    "build_v1_a3_v2_neural_profile",
    "build_v1_a3_v2_profile",
]
