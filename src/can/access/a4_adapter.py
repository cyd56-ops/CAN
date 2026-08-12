"""把 A4 GPV 精确关系适配为 A3 所需的 message-bound evidence。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from can.access.a3_protocol import (
    A3_DIGEST_SIZE,
    A3Evidence,
    A3EvidenceCode,
    A3ProtocolConfigError,
    A3VerificationProfile,
)
from can.reference.a4 import (
    A4EvidenceCode,
    A4PublicProfile,
    verify_a4_ref,
)
from can.verifier.a4 import (
    A4NeuralEvidenceCode,
    A4NeuralProfile,
    verify_a4_neural,
)


@dataclass(frozen=True, slots=True)
class A4ReferenceAdapter:
    """绑定一个无私钥公开 profile 并产生 A3 evidence。"""

    profile: A4PublicProfile

    def __post_init__(self) -> None:
        if type(self.profile) is not A4PublicProfile:
            raise A3ProtocolConfigError("A4 adapter requires the exact public profile type")

    def __call__(self, message: bytes, proof: bytes) -> A3Evidence:
        """验证 exact relation 并返回不具有授权能力的绑定证据。"""
        reference = verify_a4_ref(message, proof, self.profile)
        if reference.code is A4EvidenceCode.RELATION_ACCEPT:
            code = A3EvidenceCode.PROOF_ACCEPT
        elif reference.code is A4EvidenceCode.CONFIG_REJECT:
            code = A3EvidenceCode.CONFIG_REJECT
        else:
            code = A3EvidenceCode.PROOF_REJECT
        message_sha256 = (
            hashlib.sha256(message).digest() if type(message) is bytes else bytes(A3_DIGEST_SIZE)
        )
        return A3Evidence(
            code,
            self.profile.identity_id,
            message_sha256,
            self.profile.profile_id,
        )


@dataclass(frozen=True, slots=True)
class A4NeuralAdapter:
    """绑定一个 A4-C1 compiled profile 并产生 A3 evidence。"""

    profile: A4NeuralProfile

    def __post_init__(self) -> None:
        if type(self.profile) is not A4NeuralProfile:
            raise A3ProtocolConfigError(
                "A4 neural adapter requires the exact compiled profile type"
            )

    def __call__(self, message: bytes, proof: bytes) -> A3Evidence:
        """执行固定 neural relation 并返回无授权能力的绑定证据。"""
        neural = verify_a4_neural(message, proof, self.profile)
        if neural.code is A4NeuralEvidenceCode.NEURAL_ACCEPT:
            code = A3EvidenceCode.PROOF_ACCEPT
        elif neural.code is A4NeuralEvidenceCode.CONFIG_REJECT:
            code = A3EvidenceCode.CONFIG_REJECT
        else:
            code = A3EvidenceCode.PROOF_REJECT
        return A3Evidence(
            code,
            neural.identity_id,
            neural.message_sha256,
            neural.profile_id,
        )


def build_a4_verification_profile(profile: A4PublicProfile) -> A3VerificationProfile:
    """从本地 A4 公共参数构造 A3 identity verification profile。"""
    if type(profile) is not A4PublicProfile:
        raise A3ProtocolConfigError("A4 profile builder requires the exact public profile type")
    return A3VerificationProfile(
        profile.identity_id,
        profile.profile_id,
        A4ReferenceAdapter(profile),
    )


def build_a4_neural_verification_profile(profile: A4NeuralProfile) -> A3VerificationProfile:
    """从本地 A4-C1 compiled profile 构造 A3 verification profile。"""
    if type(profile) is not A4NeuralProfile:
        raise A3ProtocolConfigError(
            "A4 neural profile builder requires the exact compiled profile type"
        )
    return A3VerificationProfile(
        profile.identity_id,
        profile.profile_id,
        A4NeuralAdapter(profile),
    )


__all__ = [
    "A4NeuralAdapter",
    "A4ReferenceAdapter",
    "build_a4_neural_verification_profile",
    "build_a4_verification_profile",
]
