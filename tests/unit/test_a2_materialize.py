"""A2 trusted materializer 的 state/manifest 生命周期单元测试。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import BinaryIO, cast

import pytest
import torch

import can.experiments.a2_baseline as protected_baseline
import can.experiments.a2_capability as capability_experiment
import can.experiments.a2_materialize as materialize
import can.experiments.a2_public_baseline as public_baseline
from can.model.a2_mlp import A2FashionMNISTMLP
from can.model.a2_public_mlp import A2FashionMNISTPublicMLP


def _write_accepted_local_states(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[materialize.A2StatePaths, A2FashionMNISTMLP, A2FashionMNISTPublicMLP]:
    """在临时目录创建两份测试 state, 并把其摘要固定为本测试模型。"""
    torch.manual_seed(20260808)
    protected = A2FashionMNISTMLP()
    public = A2FashionMNISTPublicMLP()
    protected_digest = protected_baseline._hash_model_state(protected)
    public_digest = public_baseline._hash_model_state(public)
    monkeypatch.setattr(
        capability_experiment, "A2_EXPECTED_PROTECTED_STATE_SHA256", protected_digest
    )
    monkeypatch.setattr(capability_experiment, "A2_EXPECTED_PUBLIC_STATE_SHA256", public_digest)
    paths = materialize._state_paths(root)
    materialize._atomic_save_state(paths.protected, protected.state_dict())
    materialize._atomic_save_state(paths.public, public.state_dict())
    return paths, protected, public


def _canonical_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise AssertionError("test manifest root changed")
    return cast(dict[str, object], value)


def test_materialized_states_round_trip_with_canonical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两个 local state 必须按 manifest 摘要加载并保持模型状态。"""
    paths, protected, public = _write_accepted_local_states(tmp_path, monkeypatch)
    monkeypatch.setattr(materialize, "A2_STATE_ROOT", tmp_path)

    observed_manifest = materialize._write_manifest(paths)
    loaded_protected, loaded_public = materialize.load_materialized_states(tmp_path)

    assert observed_manifest == paths.manifest
    assert loaded_protected.training is False
    assert loaded_public.training is False
    assert protected_baseline._hash_model_state(
        loaded_protected
    ) == protected_baseline._hash_model_state(protected)
    assert public_baseline._hash_model_state(loaded_public) == public_baseline._hash_model_state(
        public
    )
    assert paths.manifest.read_bytes() == (
        json.dumps(_canonical_manifest(paths.manifest), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def test_manifest_rejects_duplicate_nonfinite_and_protocol_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """manifest 的重复字段、NaN 和协议漂移都必须 fail closed。"""
    paths, _, _ = _write_accepted_local_states(tmp_path, monkeypatch)
    materialize._write_manifest(paths)
    original_manifest = paths.manifest.read_text(encoding="utf-8")

    paths.manifest.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(materialize.A2MaterializationError):
        materialize._load_manifest(paths.manifest)

    paths.manifest.write_text(original_manifest, encoding="utf-8")
    manifest = _canonical_manifest(paths.manifest)
    manifest["protocol"]["torch"] = "other"  # type: ignore[index]
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(materialize.A2MaterializationError):
        materialize._load_manifest(paths.manifest)

    paths.manifest.write_text('{"schema_version": NaN}', encoding="utf-8")
    with pytest.raises(materialize.A2MaterializationError):
        materialize._load_manifest(paths.manifest)


def test_manifest_rejects_data_and_state_entry_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """数据摘要和 state entry 不能被本地 manifest 覆盖。"""
    paths, _, _ = _write_accepted_local_states(tmp_path, monkeypatch)
    materialize._write_manifest(paths)
    manifest = _canonical_manifest(paths.manifest)

    manifest["data"]["root"] = "data/other"  # type: ignore[index]
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(materialize.A2MaterializationError):
        materialize.load_materialized_states(tmp_path)

    manifest = _canonical_manifest(paths.manifest)
    manifest["data"]["root"] = str(protected_baseline.A2_DATA_ROOT)  # type: ignore[index]
    manifest["states"]["public"]["parameter_count"] = True  # type: ignore[index]
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(materialize.A2MaterializationError):
        materialize.load_materialized_states(tmp_path)


def test_state_file_tamper_and_malformed_payload_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """state 文件摘要、pickle payload 和 topology 漂移必须拒绝。"""
    paths, _, _ = _write_accepted_local_states(tmp_path, monkeypatch)
    materialize._write_manifest(paths)

    original = paths.protected.read_bytes()
    paths.protected.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(materialize.A2MaterializationError):
        materialize.load_materialized_states(tmp_path)

    paths.protected.write_bytes(b"not a torch state")
    with pytest.raises(materialize.A2MaterializationError):
        materialize._load_state_file(
            "protected", paths.protected, materialize._file_digest(paths.protected)
        )

    malformed: OrderedDict[str, torch.Tensor] = OrderedDict(
        [("wrong", torch.zeros(1, dtype=torch.float32))]
    )
    torch.save(malformed, paths.protected)
    with pytest.raises(materialize.A2MaterializationError):
        materialize._load_state_file(
            "protected", paths.protected, materialize._file_digest(paths.protected)
        )


def test_state_paths_and_materialization_routes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """入口只接受 canonical kind, 并拒绝覆盖已有 artifact。"""
    with pytest.raises(materialize.A2MaterializationError):
        materialize._state_paths(tmp_path.as_posix())  # type: ignore[arg-type]
    with pytest.raises(materialize.A2MaterializationError):
        materialize.materialize_a2_states("other")  # type: ignore[arg-type]

    protected = tmp_path / "protected-state.pt"
    monkeypatch.setattr(materialize, "_materialize_protected_state", lambda: protected)
    monkeypatch.setattr(
        materialize, "_materialize_public_state", lambda: tmp_path / "public-state.pt"
    )
    assert materialize.materialize_a2_states("protected") == protected
    assert materialize.materialize_a2_states("public") == tmp_path / "public-state.pt"


def test_atomic_state_writer_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    """state writer 使用一次性创建, 避免静默替换既有本地状态。"""
    path = tmp_path / "state.pt"
    path.write_bytes(b"existing")
    with pytest.raises(materialize.A2MaterializationError):
        materialize._atomic_save_state(path, OrderedDict())
    assert path.read_bytes() == b"existing"


def test_concurrent_state_writers_publish_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """并发重复物化只能发布一个 state, 另一调用必须拒绝。"""
    path = tmp_path / "state.pt"
    state: OrderedDict[str, torch.Tensor] = OrderedDict(
        [("weight", torch.ones(2, dtype=torch.float32))]
    )
    barrier = Barrier(2)
    original_save = torch.save

    def synchronized_save(value: object, stream: BinaryIO) -> None:
        original_save(value, stream)
        barrier.wait()

    def attempt() -> str | materialize.A2MaterializationError:
        try:
            return materialize._atomic_save_state(path, state)
        except materialize.A2MaterializationError as error:
            return error

    monkeypatch.setattr(torch, "save", synchronized_save)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(attempt) for _ in range(2))
        results = tuple(future.result() for future in futures)

    assert sum(type(result) is str for result in results) == 1
    assert sum(type(result) is materialize.A2MaterializationError for result in results) == 1
    assert path.is_file()


def test_file_digest_is_sha256(tmp_path: Path) -> None:
    """manifest file digest 使用标准 SHA-256。"""
    path = tmp_path / "payload"
    payload = b"a2-state"
    path.write_bytes(payload)
    assert materialize._file_digest(path) == hashlib.sha256(payload).hexdigest()


def test_full_materialization_uses_fixed_isolated_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """完整入口按两个固定 seed 物化, 再校验 manifest 并报告。"""
    calls: list[tuple[list[str], str]] = []

    def record_run(
        command: list[str], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        assert check is True
        calls.append((command, env["PYTHONHASHSEED"]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", record_run)
    monkeypatch.setattr(
        capability_experiment, "A2_CAPABILITY_REPORT_PATH", tmp_path / "report.json"
    )

    assert materialize.run_a2_materialization() == tmp_path / "report.json"
    assert calls == [
        (
            [
                sys.executable,
                "-m",
                materialize.A2_MATERIALIZER_MODULE,
                "--materialize",
                "protected",
            ],
            str(protected_baseline.GLOBAL_SEED),
        ),
        (
            [
                sys.executable,
                "-m",
                materialize.A2_MATERIALIZER_MODULE,
                "--materialize",
                "public",
            ],
            str(public_baseline.PUBLIC_GLOBAL_SEED),
        ),
        (
            [sys.executable, "-m", materialize.A2_MATERIALIZER_MODULE, "--manifest"],
            str(protected_baseline.GLOBAL_SEED),
        ),
        (
            [sys.executable, "-m", materialize.A2_MATERIALIZER_MODULE, "--report"],
            str(protected_baseline.GLOBAL_SEED),
        ),
    ]
