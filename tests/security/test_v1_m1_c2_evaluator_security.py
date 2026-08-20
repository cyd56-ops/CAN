"""V1-M1-C2 evaluator artifact 的 fail-closed 文件边界。"""

from pathlib import Path

import pytest
import torch

from can.access import V1M1C2Cut, V1M1C2PublicHead
from can.experiments import v1_m1_c2 as training
from can.experiments import v1_m1_c2_evaluator as evaluator


def test_evaluator_json_writer_refuses_overwrite_and_symlink(tmp_path: Path) -> None:
    """correction/report 只能追加一次, 不跟随已有 symlink。"""
    path = tmp_path / "report.json"
    evaluator._atomic_write_json(path, {"schema_version": 1})
    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="overwrite"):
        evaluator._atomic_write_json(path, {"schema_version": 1})

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="overwrite"):
        evaluator._atomic_write_json(link, {"schema_version": 1})


def test_evaluator_rejects_symlinked_artifact_root(tmp_path: Path) -> None:
    """请求方控制的 symlink root 不能替代可信 C2 artifact 目录。"""
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "c2"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="symlinked"):
        evaluator.load_v1_m1_c2_accepted_public_head(link, torch.device("cpu"))


def test_evaluator_rejects_symlinked_public_head_state(tmp_path: Path) -> None:
    """public-head loader 不跟随 artifact 目录内的 state symlink。"""
    target = tmp_path / "target.pt"
    head = V1M1C2PublicHead(V1M1C2Cut.LAYER2.channels)
    state = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
    torch.save(state, target)
    link = tmp_path / "accepted-public-head.pt"
    link.symlink_to(target)

    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="symlinked"):
        evaluator._load_head_state(
            link,
            V1M1C2Cut.LAYER2,
            training._head_state_digest(state),
            torch.device("cpu"),
        )


def test_evaluator_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    """artifact JSON 的重复安全字段不能以后值覆盖前值。"""
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")

    with pytest.raises(evaluator.V1M1C2EvaluatorError, match="canonical JSON"):
        evaluator._read_json_object(path, "manifest")
