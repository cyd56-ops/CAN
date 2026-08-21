"""V1-M1-C2 public-head runner 的本地 preflight 与选择测试。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from can.access import V1M1C2Cut
from can.access.v1_m1_adapter import V1_M1_INPUT_PROFILE_SHA256
from can.experiments import v1_m1_baseline as baseline
from can.experiments import v1_m1_c2


def _metrics(top1_percent: float) -> v1_m1_c2.V1M1C2HeadMetrics:
    return v1_m1_c2.V1M1C2HeadMetrics(
        loss=1.0,
        top1_percent=top1_percent,
        correct_top1=int(top1_percent),
        total=100,
        predictions_sha256="a" * 64,
    )


def _run(
    run_name: v1_m1_c2.V1M1C2RunName,
    cut: V1M1C2Cut,
    top1_percent: float,
) -> v1_m1_c2.V1M1C2HeadRunResult:
    config = v1_m1_c2.V1M1C2HeadTrainingConfig(
        run_name=run_name,
        cut=cut,
        seed=v1_m1_c2.V1_M1_C2_RUN_SEEDS[run_name],
    )
    state = {
        "classifier.weight": torch.zeros((20, cut.channels)),
        "classifier.bias": torch.zeros(20),
    }
    validation = _metrics(top1_percent)
    return v1_m1_c2.V1M1C2HeadRunResult(
        config=config,
        selected_epoch=1,
        validation=validation,
        epochs=(v1_m1_c2.V1M1C2EpochMetrics(1, validation),),
        state_sha256=v1_m1_c2._head_state_digest(state),
        state=state,
    )


def test_preflight_is_no_training_no_download_and_checks_head_sizes(tmp_path: Path) -> None:
    """本地 preflight 只验证冻结配置, 不创建数据或 artifact。"""
    artifact_root = tmp_path / "c2"
    result = v1_m1_c2.preflight_v1_m1_c2(artifact_root)

    assert result["writes_artifact"] is False
    assert result["downloads_data"] is False
    assert result["trains"] is False
    assert result["threshold_percent"] == 75.0
    assert result["candidate_heads"] == [
        {"cut": "layer2", "channels": 128, "parameter_count": 2_580},
        {"cut": "layer3", "channels": 256, "parameter_count": 5_140},
        {"cut": "layer4", "channels": 512, "parameter_count": 10_260},
    ]
    assert not artifact_root.exists()


def test_training_config_rejects_seed_or_topology_changes() -> None:
    """H1/H2 seed、epoch 和 optimizer 改动必须 fail closed。"""
    with pytest.raises(v1_m1_c2.V1M1C2ExperimentError, match="seed"):
        v1_m1_c2.V1M1C2HeadTrainingConfig("H1", V1M1C2Cut.LAYER2, 1730)
    with pytest.raises(v1_m1_c2.V1M1C2ExperimentError, match="configuration"):
        v1_m1_c2.V1M1C2HeadTrainingConfig(
            "H1",
            V1M1C2Cut.LAYER2,
            1729,
            epoch_count=49,
        )


def test_batch_progress_uses_fixed_width_bar_and_public_counts() -> None:
    """C2 batch 进度条必须显示固定宽度、run、cut 和公开计数。"""
    config = v1_m1_c2.V1M1C2HeadTrainingConfig("H2", V1M1C2Cut.LAYER4, 1730)

    initial = v1_m1_c2._format_v1_m1_c2_batch_progress(config, 0, 4, "train", 0, 0, 2)
    completed = v1_m1_c2._format_v1_m1_c2_batch_progress(config, 4, 4, "complete", 50, 1, 1)

    assert initial == (
        "V1-M1-C2 progress [------------------------------]   0.00% "
        "run=H2 seed=1730 cut=layer4 stage=train epoch=0/50 batch=0/2"
    )
    assert completed == (
        "V1-M1-C2 progress [##############################] 100.00% "
        "run=H2 seed=1730 cut=layer4 stage=complete epoch=50/50 batch=1/1"
    )


def test_batch_progress_reports_one_start_and_artifact_gated_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """C2 全流程只输出一次开始; 仅在调用 finish 后输出完成。"""
    h1 = v1_m1_c2.V1M1C2HeadTrainingConfig("H1", V1M1C2Cut.LAYER2, 1729)
    h2 = v1_m1_c2.V1M1C2HeadTrainingConfig("H2", V1M1C2Cut.LAYER4, 1730)
    reporter = v1_m1_c2._V1M1C2ProgressReporter(3)

    reporter.start(h1, first_train_batch_count=1)
    reporter.complete_batch(h1, "train", 1, 1, 1)
    reporter.start(h2, first_train_batch_count=1)
    reporter.complete_batch(h2, "selected_validation", 50, 1, 1)
    reporter.complete_batch(h2, "test", 50, 1, 1)
    before_finish = capsys.readouterr().out

    assert before_finish.count("V1-M1-C2 training started") == 1
    assert "V1-M1-C2 training completed" not in before_finish
    assert "V1-M1-C2 progress [##############################] 100.00%" in before_finish
    assert "run=H1 seed=1729 cut=layer2 stage=train" in before_finish
    assert "run=H2 seed=1730 cut=layer4 stage=test" in before_finish

    reporter.finish(h2, final_test_batch_count=1)
    completion = capsys.readouterr().out

    assert "V1-M1-C2 progress [##############################] 100.00%" in completion
    assert completion.endswith(
        "V1-M1-C2 training completed accepted_run=H2 cut=layer4 completed_batches=3/3\n"
    )


def test_h1_h2_determinism_uses_independent_rng_seeds_in_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 runner 进程可以按 H1/H2 seed 重置 RNG, 不动态修改 PYTHONHASHSEED。"""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    v1_m1_c2._configure_c2_determinism(1729)
    first = torch.rand(1)
    v1_m1_c2._configure_c2_determinism(1730)
    second = torch.rand(1)
    v1_m1_c2._configure_c2_determinism(1729)
    assert torch.equal(first, torch.rand(1))
    assert not torch.equal(first, second)


def test_coarse_digest_is_ordered_and_range_checked() -> None:
    """coarse digest 绑定 train/test archive 顺序并拒绝越界 label。"""
    train = torch.tensor([0, 19, 3], dtype=torch.int64)
    test = torch.tensor([4, 7], dtype=torch.int64)
    first = v1_m1_c2._coarse_labels_digest(train, test)
    second = v1_m1_c2._coarse_labels_digest(train, test)
    assert first == second
    assert first != v1_m1_c2._coarse_labels_digest(test, train)
    with pytest.raises(v1_m1_c2.V1M1C2ExperimentError, match="range"):
        v1_m1_c2._coarse_labels_digest(torch.tensor([20], dtype=torch.int64), test)


def test_h1_selection_uses_first_shallow_passing_cut() -> None:
    """H1 达标时必须选择 layer2, 即使更深 cut 得分更高。"""
    candidates = tuple(
        _run("H1", cut, score)
        for cut, score in (
            (V1M1C2Cut.LAYER2, 75.0),
            (V1M1C2Cut.LAYER3, 99.0),
            (V1M1C2Cut.LAYER4, 99.0),
        )
    )
    assert v1_m1_c2._select_h1_cut(candidates) is V1M1C2Cut.LAYER2


def test_accepted_head_tie_breaks_to_h1() -> None:
    """H1/H2 validation 平局必须固定选择 H1。"""
    h1 = _run("H1", V1M1C2Cut.LAYER3, 80.0)
    h2 = _run("H2", V1M1C2Cut.LAYER3, 80.0)
    assert v1_m1_c2._select_accepted_head(h1, h2) is h1


def test_runner_stops_before_cuda_or_training_when_archive_missing(tmp_path: Path) -> None:
    """正式 runner 在数据缺失时不得创建 artifact 或初始化训练。"""
    artifact_root = tmp_path / "c2"
    with pytest.raises(v1_m1_c2.V1M1C2ExperimentError, match=r"archive|unavailable"):
        v1_m1_c2.run_v1_m1_c2(
            tmp_path / "data",
            tmp_path / "accepted",
            torch.device("cuda:0"),
            artifact_root,
        )
    assert not artifact_root.exists()


def test_artifact_paths_refuse_existing_outputs(tmp_path: Path) -> None:
    """accepted artifact 不能覆盖已有 state。"""
    root = tmp_path / "c2"
    root.mkdir()
    (root / v1_m1_c2.V1_M1_C2_STATE_FILENAME).write_bytes(b"existing")
    with pytest.raises(v1_m1_c2.V1M1C2ExperimentError, match="already exists"):
        v1_m1_c2._artifact_paths(root)


def test_artifact_writer_saves_only_public_head_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """public artifact 只能保存 head state 和 accepted R2 digest, 不能复制 R2 state。"""
    root = tmp_path / "c2"
    paths = v1_m1_c2._artifact_paths(root)
    h1_candidates = tuple(_run("H1", cut, 80.0) for cut in V1M1C2Cut)
    h1 = h1_candidates[1]
    h2 = _run("H2", V1M1C2Cut.LAYER3, 81.0)
    test = _metrics(80.0)
    archive = baseline.V1M1ArchiveManifest(
        filename="cifar-100-python.tar.gz",
        byte_size=169_001_437,
        sha256="a" * 64,
        md5="b" * 32,
    )
    empty_dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], [])
    empty_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(empty_dataset)
    data = v1_m1_c2._V1M1C2DataBundle(
        archive=archive,
        decoded_sha256="c" * 64,
        coarse_labels_sha256="d" * 64,
        train_loader=empty_loader,
        validation_loader=empty_loader,
        test_loader=empty_loader,
    )
    monkeypatch.setattr(
        baseline,
        "_environment_report",
        lambda _device: {"device": "cpu"},
    )

    v1_m1_c2._write_c2_artifacts(
        paths,
        h1,
        h1_candidates,
        h2,
        test,
        data,
        "e" * 64,
        torch.device("cpu"),
    )

    state = torch.load(paths.state, weights_only=True)
    assert set(state) == {"classifier.weight", "classifier.bias"}
    manifest = paths.manifest.read_text(encoding="utf-8")
    expected_digest = '"accepted_r2_state_sha256": "' + "e" * 64 + '"'
    assert expected_digest in manifest
    assert '"input_profile_sha256": "' + V1_M1_INPUT_PROFILE_SHA256.hex() + '"' in manifest
    assert "R2 state" not in manifest
