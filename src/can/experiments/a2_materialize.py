"""A2 已验收 baseline 的确定性物化与本地 state 生命周期。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NoReturn, cast

import torch

from can.experiments import a2_baseline as protected_baseline
from can.experiments import a2_capability as capability_experiment
from can.experiments import a2_public_baseline as public_baseline
from can.model.a2_mlp import A2_PARAMETER_COUNT, A2FashionMNISTMLP
from can.model.a2_public_mlp import A2_PUBLIC_PARAMETER_COUNT, A2FashionMNISTPublicMLP

A2_MATERIALIZATION_ID: Final = "CAN-A2-ACCEPTED-STATE-MATERIALIZATION-v1"
A2_MATERIALIZER_MODULE: Final = "can.experiments.a2_materialize"
A2_STATE_ROOT: Final = protected_baseline.A2_REPORT_ROOT / "local-states"
A2_PROTECTED_STATE_FILENAME: Final = "protected-state.pt"
A2_PUBLIC_STATE_FILENAME: Final = "public-state.pt"
A2_MANIFEST_FILENAME: Final = "manifest.json"
A2_MAX_STATE_BYTES: Final = 4_000_000
_KINDS: Final = ("protected", "public")
StateKind = Literal["protected", "public"]


class A2MaterializationError(RuntimeError):
    """表示 A2 state 物化、保存或加载不满足固定契约。"""


@dataclass(frozen=True, slots=True)
class A2StatePaths:
    """保存两个 local state 和 manifest 的固定路径。"""

    root: Path
    protected: Path
    public: Path
    manifest: Path


def _state_paths(root: Path) -> A2StatePaths:
    if not isinstance(root, Path):
        raise A2MaterializationError("state root must be a pathlib.Path")
    return A2StatePaths(
        root=root,
        protected=root / A2_PROTECTED_STATE_FILENAME,
        public=root / A2_PUBLIC_STATE_FILENAME,
        manifest=root / A2_MANIFEST_FILENAME,
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise A2MaterializationError("state file cannot be read") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_state(kind: StateKind) -> tuple[str, str, int, type[torch.nn.Module]]:
    if kind == "protected":
        return (
            capability_experiment.A2_EXPECTED_PROTECTED_STATE_SHA256,
            "784->256->128->10",
            A2_PARAMETER_COUNT,
            A2FashionMNISTMLP,
        )
    if kind == "public":
        return (
            capability_experiment.A2_EXPECTED_PUBLIC_STATE_SHA256,
            "784->64->2",
            A2_PUBLIC_PARAMETER_COUNT,
            A2FashionMNISTPublicMLP,
        )
    raise A2MaterializationError("state kind is not canonical")


def _model_digest(kind: StateKind, model: object) -> str:
    expected, _, _, model_type = _expected_state(kind)
    if type(model) is not model_type:
        raise A2MaterializationError(f"{kind} model class changed")
    if kind == "protected":
        observed = protected_baseline._hash_model_state(cast(A2FashionMNISTMLP, model))
    else:
        observed = public_baseline._hash_model_state(cast(A2FashionMNISTPublicMLP, model))
    if observed != expected:
        raise A2MaterializationError(f"{kind} model state digest changed")
    return observed


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise A2MaterializationError(f"refusing to overwrite existing local artifact: {path.name}")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o600)
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise A2MaterializationError(
                f"refusing to overwrite existing local artifact: {path.name}"
            ) from error
    except OSError as error:
        raise A2MaterializationError("local artifact write failed") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_state(path: Path, state: Mapping[str, torch.Tensor]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise A2MaterializationError(f"refusing to overwrite existing local state: {path.name}")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary_path.stat().st_size > A2_MAX_STATE_BYTES:
            raise A2MaterializationError("serialized state is oversized")
        temporary_path.chmod(0o600)
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise A2MaterializationError(
                f"refusing to overwrite existing local state: {path.name}"
            ) from error
    except A2MaterializationError:
        raise
    except (OSError, RuntimeError) as error:
        raise A2MaterializationError("local state write failed") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return _file_digest(path)


def _validate_state_dict(model: torch.nn.Module, value: object) -> OrderedDict[str, torch.Tensor]:
    if type(value) is not OrderedDict:
        raise A2MaterializationError("state artifact must contain an OrderedDict")
    state = cast(OrderedDict[str, torch.Tensor], value)
    expected_keys = tuple(model.state_dict())
    if tuple(state) != expected_keys:
        raise A2MaterializationError("state artifact keys changed")
    for key, tensor in state.items():
        if type(key) is not str or type(tensor) is not torch.Tensor:
            raise A2MaterializationError("state artifact has invalid key or tensor type")
        if tensor.dtype is not torch.float32:
            raise A2MaterializationError("state artifact tensors must use float32")
        if tensor.device.type != "cpu" or tensor.device.index is not None:
            raise A2MaterializationError("state artifact tensors must remain on CPU")
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise A2MaterializationError("state artifact tensors must be contiguous")
        if not bool(torch.isfinite(tensor).all().item()):
            raise A2MaterializationError("state artifact tensors must be finite")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise A2MaterializationError("state artifact does not match model topology") from error
    model.eval()
    return state


def _load_state_file(kind: StateKind, path: Path, expected_file_sha256: str) -> torch.nn.Module:
    if path.is_symlink() or not path.is_file():
        raise A2MaterializationError(f"{kind} state file is missing or symlinked")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise A2MaterializationError("state file metadata cannot be read") from error
    if size < 1 or size > A2_MAX_STATE_BYTES:
        raise A2MaterializationError("state file size is outside the fixed bound")
    observed_file_sha256 = _file_digest(path)
    if observed_file_sha256 != expected_file_sha256:
        raise A2MaterializationError(f"{kind} state file digest changed")
    _, _, _, model_type = _expected_state(kind)
    model = model_type().to(device="cpu", dtype=torch.float32)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise A2MaterializationError(f"{kind} state artifact cannot be loaded") from error
    _validate_state_dict(model, value)
    _model_digest(kind, model)
    return model


def _manifest_for(paths: A2StatePaths) -> dict[str, object]:
    protected_state, protected_topology, protected_count, _ = _expected_state("protected")
    public_state, public_topology, public_count, _ = _expected_state("public")
    if not paths.protected.is_file() or not paths.public.is_file():
        raise A2MaterializationError("both local states are required before writing manifest")
    resources = protected_baseline._validate_data_resources(protected_baseline.A2_DATA_ROOT)
    return {
        "schema_version": 1,
        "materialization_id": A2_MATERIALIZATION_ID,
        "state_dict_only": True,
        "optimizer_state_saved": False,
        "protocol": {
            "protected_python_hash_seed": str(protected_baseline.GLOBAL_SEED),
            "public_python_hash_seed": str(public_baseline.PUBLIC_GLOBAL_SEED),
            "epochs": protected_baseline.EPOCH_COUNT,
            "torch": protected_baseline.TORCH_VERSION,
            "torchvision": protected_baseline.TORCHVISION_VERSION,
            "numpy": protected_baseline.NUMPY_VERSION,
            "pillow": protected_baseline.PILLOW_VERSION,
        },
        "data": {
            "root": str(protected_baseline.A2_DATA_ROOT),
            "resources": resources,
        },
        "states": {
            "protected": {
                "filename": paths.protected.name,
                "state_sha256": protected_state,
                "file_sha256": _file_digest(paths.protected),
                "topology": protected_topology,
                "parameter_count": protected_count,
            },
            "public": {
                "filename": paths.public.name,
                "state_sha256": public_state,
                "file_sha256": _file_digest(paths.public),
                "topology": public_topology,
                "parameter_count": public_count,
            },
        },
    }


def _write_manifest(paths: A2StatePaths) -> Path:
    manifest = _manifest_for(paths)
    payload = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )
    if paths.manifest.exists() or paths.manifest.is_symlink():
        raise A2MaterializationError("refusing to overwrite existing state manifest")
    _atomic_write_bytes(paths.manifest, payload)
    return paths.manifest


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise A2MaterializationError("state manifest has duplicate or invalid fields")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise A2MaterializationError(f"state manifest contains non-finite JSON constant: {value}")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000:
            raise A2MaterializationError("state manifest is missing, symlinked or oversized")
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except A2MaterializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise A2MaterializationError("state manifest is not canonical JSON") from error
    if type(value) is not dict:
        raise A2MaterializationError("state manifest root must be an object")
    expected_keys = {
        "schema_version",
        "materialization_id",
        "state_dict_only",
        "optimizer_state_saved",
        "protocol",
        "data",
        "states",
    }
    if set(value) != expected_keys:
        raise A2MaterializationError("state manifest fields changed")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["materialization_id"] != A2_MATERIALIZATION_ID
        or type(value["state_dict_only"]) is not bool
        or value["state_dict_only"] is not True
        or type(value["optimizer_state_saved"]) is not bool
        or value["optimizer_state_saved"] is not False
    ):
        raise A2MaterializationError("state manifest protocol changed")

    protocol = value["protocol"]
    expected_protocol = {
        "protected_python_hash_seed": str(protected_baseline.GLOBAL_SEED),
        "public_python_hash_seed": str(public_baseline.PUBLIC_GLOBAL_SEED),
        "epochs": protected_baseline.EPOCH_COUNT,
        "torch": protected_baseline.TORCH_VERSION,
        "torchvision": protected_baseline.TORCHVISION_VERSION,
        "numpy": protected_baseline.NUMPY_VERSION,
        "pillow": protected_baseline.PILLOW_VERSION,
    }
    if type(protocol) is not dict or protocol != expected_protocol:
        raise A2MaterializationError("state manifest runtime protocol changed")

    data = value["data"]
    if type(data) is not dict or set(data) != {"root", "resources"}:
        raise A2MaterializationError("state manifest data fields changed")
    expected_root = str(protected_baseline.A2_DATA_ROOT)
    if data["root"] != expected_root:
        raise A2MaterializationError("state manifest data root changed")
    try:
        expected_resources = protected_baseline._validate_data_resources(
            protected_baseline.A2_DATA_ROOT
        )
    except (OSError, protected_baseline.A2BaselineError) as error:
        raise A2MaterializationError("current A2 data resources are not accepted") from error
    if data["resources"] != expected_resources:
        raise A2MaterializationError("state manifest data resources changed")

    canonical = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )
    if raw.encode("utf-8") != canonical:
        raise A2MaterializationError("state manifest is not canonical JSON")
    return value


def _state_entry(manifest: dict[str, object], kind: StateKind) -> dict[str, object]:
    states = manifest["states"]
    if type(states) is not dict or set(states) != set(_KINDS):
        raise A2MaterializationError("state manifest state entries changed")
    entry = states[kind]
    if type(entry) is not dict:
        raise A2MaterializationError("state manifest entry must be an object")
    typed_entry = cast(dict[str, object], entry)
    if set(typed_entry) != {
        "filename",
        "state_sha256",
        "file_sha256",
        "topology",
        "parameter_count",
    }:
        raise A2MaterializationError("state manifest entry fields changed")
    expected_state, expected_topology, expected_count, _ = _expected_state(kind)
    expected_filename = (
        A2_PROTECTED_STATE_FILENAME if kind == "protected" else A2_PUBLIC_STATE_FILENAME
    )
    if (
        type(typed_entry["filename"]) is not str
        or typed_entry["filename"] != expected_filename
        or type(typed_entry["state_sha256"]) is not str
        or typed_entry["state_sha256"] != expected_state
        or type(typed_entry["topology"]) is not str
        or typed_entry["topology"] != expected_topology
        or type(typed_entry["parameter_count"]) is not int
        or typed_entry["parameter_count"] != expected_count
        or not _is_sha256(typed_entry["file_sha256"])
    ):
        raise A2MaterializationError(f"{kind} state manifest entry changed")
    return typed_entry


def load_materialized_states(
    state_root: Path = A2_STATE_ROOT,
) -> tuple[A2FashionMNISTMLP, A2FashionMNISTPublicMLP]:
    """加载并严格校验本地 accepted protected/public state。"""
    paths = _state_paths(state_root)
    manifest = _load_manifest(paths.manifest)
    protected_entry = _state_entry(manifest, "protected")
    public_entry = _state_entry(manifest, "public")
    protected_model = cast(
        A2FashionMNISTMLP,
        _load_state_file("protected", paths.protected, cast(str, protected_entry["file_sha256"])),
    )
    public_model = cast(
        A2FashionMNISTPublicMLP,
        _load_state_file("public", paths.public, cast(str, public_entry["file_sha256"])),
    )
    return protected_model, public_model


def _accepted_reference(kind: StateKind) -> tuple[str, str]:
    references = capability_experiment._load_accepted_baseline_references()
    reference = references[kind]
    if type(reference) is not dict:
        raise A2MaterializationError("accepted baseline reference is malformed")
    model = reference["model"]
    test = reference["test"]
    if type(model) is not dict or type(test) is not dict:
        raise A2MaterializationError("accepted baseline reference is malformed")
    state_sha256 = model.get("state_sha256")
    predictions_sha256 = test.get("predictions_sha256")
    if not _is_sha256(state_sha256) or not _is_sha256(predictions_sha256):
        raise A2MaterializationError("accepted baseline digest is malformed")
    return cast(str, state_sha256), cast(str, predictions_sha256)


def _materialize_protected_state() -> Path:
    protected_baseline._validate_environment()
    protected_baseline._configure_determinism()
    accepted_state, accepted_predictions = _accepted_reference("protected")
    protected_baseline._validate_data_resources(protected_baseline.A2_DATA_ROOT)
    data = protected_baseline._load_data(protected_baseline.A2_DATA_ROOT)
    model, _ = protected_baseline._train_model(data, emit_progress=True)
    metrics = protected_baseline._evaluate(model, data.test_loader)
    if protected_baseline._hash_model_state(model) != accepted_state:
        raise A2MaterializationError("deterministic protected state digest changed")
    if metrics.predictions_sha256 != accepted_predictions:
        raise A2MaterializationError("deterministic protected predictions changed")
    path = _state_paths(A2_STATE_ROOT).protected
    _atomic_save_state(path, model.state_dict())
    return path


def _materialize_public_state() -> Path:
    public_baseline._validate_environment()
    public_baseline._configure_determinism()
    accepted_state, accepted_predictions = _accepted_reference("public")
    public_baseline._validate_data_resources(public_baseline.A2_DATA_ROOT)
    data = public_baseline._load_data(public_baseline.A2_DATA_ROOT)
    model, _ = public_baseline._train_model(data, emit_progress=True)
    metrics = public_baseline._evaluate(model, data.test_loader)
    if public_baseline._hash_model_state(model) != accepted_state:
        raise A2MaterializationError("deterministic public state digest changed")
    if metrics.predictions_sha256 != accepted_predictions:
        raise A2MaterializationError("deterministic public predictions changed")
    path = _state_paths(A2_STATE_ROOT).public
    _atomic_save_state(path, model.state_dict())
    return path


def materialize_a2_states(kind: StateKind) -> Path:
    """按固定协议物化一个 accepted baseline 的本地 state_dict。"""
    if type(kind) is not str or kind not in _KINDS:
        raise A2MaterializationError("materialization kind must be protected or public")
    if kind == "protected":
        return _materialize_protected_state()
    return _materialize_public_state()


def write_a2_state_manifest() -> Path:
    """写入同时覆盖两个已校验 local state 的 manifest。"""
    paths = _state_paths(A2_STATE_ROOT)
    load_materialized_states_without_manifest(paths)
    return _write_manifest(paths)


def load_materialized_states_without_manifest(paths: A2StatePaths) -> None:
    """在 manifest 生成前验证两个 state 文件的 canonical digest。"""
    for kind, path in (("protected", paths.protected), ("public", paths.public)):
        state_kind = cast(StateKind, kind)
        if path.is_symlink() or not path.is_file():
            raise A2MaterializationError(f"{kind} state file is missing or symlinked")
        _load_state_file(state_kind, path, _file_digest(path))


def run_a2_materialized_report() -> Path:
    """加载 local states 并调用不训练的 A2-E2 三态报告入口。"""
    protected_model, public_model = load_materialized_states()
    return capability_experiment.run_a2_capability_experiment(protected_model, public_model)


def run_a2_materialization() -> Path:
    """在隔离的固定 seed 子进程中物化两个 state 并生成三态报告。"""
    commands = (
        ("protected", protected_baseline.GLOBAL_SEED),
        ("public", public_baseline.PUBLIC_GLOBAL_SEED),
    )
    for kind, seed in commands:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(seed)
        subprocess.run(
            [sys.executable, "-m", A2_MATERIALIZER_MODULE, "--materialize", kind],
            check=True,
            env=environment,
        )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(protected_baseline.GLOBAL_SEED)
    subprocess.run(
        [sys.executable, "-m", A2_MATERIALIZER_MODULE, "--manifest"],
        check=True,
        env=environment,
    )
    subprocess.run(
        [sys.executable, "-m", A2_MATERIALIZER_MODULE, "--report"],
        check=True,
        env=environment,
    )
    return capability_experiment.A2_CAPABILITY_REPORT_PATH


def main(argv: list[str] | None = None) -> int:
    """解析受信 materialization 子命令。"""
    parser = argparse.ArgumentParser(description="Materialize accepted CAN A2 model states")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", choices=_KINDS)
    mode.add_argument("--manifest", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--run", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.run:
        output = run_a2_materialization()
    elif arguments.manifest:
        output = write_a2_state_manifest()
    elif arguments.report:
        output = run_a2_materialized_report()
    else:
        output = materialize_a2_states(cast(StateKind, arguments.materialize))
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
