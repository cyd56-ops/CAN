"""V1-M1 无下载 archive contract 与训练计划单元测试。"""

import json
import pickle
from pathlib import Path

import pytest
import torch

from can.experiments.v1_m1_baseline import (
    V1_M1_EPOCH_COUNT,
    V1_M1_FINE_LABEL_NAMES,
    V1_M1_RUN_SEEDS,
    V1M1ArchiveManifest,
    V1M1BaselineError,
    V1M1BaselinePlan,
    V1M1BaselineResult,
    V1M1EpochMetrics,
    V1M1EvaluationMetrics,
    V1M1TrainingConfig,
    _artifact_paths,
    _build_v1_m1_split_indices,
    _decode_meta,
    _format_v1_m1_batch_progress,
    _format_v1_m1_epoch_progress,
    _V1M1ProgressReporter,
    _write_v1_m1_artifacts,
    build_v1_m1_baseline_plan,
    run_v1_m1_baseline,
    verify_v1_m1_archive,
)


def test_training_config_selects_only_the_two_pre_registered_runs() -> None:
    """训练配置只能选择 1729/1730 的两个预注册 baseline run。"""
    first = V1M1TrainingConfig(run_index=1, seed=V1_M1_RUN_SEEDS[0])
    second = V1M1TrainingConfig(run_index=2, seed=V1_M1_RUN_SEEDS[1])

    assert first.epoch_count == V1_M1_EPOCH_COUNT
    assert second.seed == 1730


@pytest.mark.parametrize(
    ("run_index", "seed"),
    [(0, 1729), (3, 1730), (True, 1729), (1, 1730)],
)
def test_training_config_rejects_unregistered_run_or_seed(run_index: object, seed: object) -> None:
    """run index、bool/int 混淆和 seed 覆盖必须 fail closed。"""
    with pytest.raises(V1M1BaselineError):
        V1M1TrainingConfig(run_index=run_index, seed=seed)  # type: ignore[arg-type]


def test_archive_verification_and_plan_never_download_missing_data(tmp_path: Path) -> None:
    """缺少 archive 时只能报错, 不能隐式连接网络、下载或创建数据。"""
    with pytest.raises(V1M1BaselineError, match="unavailable"):
        verify_v1_m1_archive(tmp_path)
    with pytest.raises(V1M1BaselineError, match="unavailable"):
        build_v1_m1_baseline_plan(tmp_path, 1)


def test_split_uses_the_first_fifty_source_order_examples_per_fine_label() -> None:
    """每个 fine label 的前 50 个 source-order 样本必须构成 validation。"""
    labels = torch.arange(100, dtype=torch.int64).repeat_interleave(500)

    training, validation = _build_v1_m1_split_indices(labels)

    assert tuple(training.shape) == (45_000,)
    assert tuple(validation.shape) == (5_000,)
    assert validation[:50].tolist() == list(range(50))
    assert training[:450].tolist() == list(range(50, 500))
    assert validation[50:100].tolist() == list(range(500, 550))


def test_split_rejects_an_unbalanced_cifar100_training_label_set() -> None:
    """不是每类 500 条的 decoded train split 必须拒绝。"""
    labels = torch.arange(100, dtype=torch.int64).repeat_interleave(500)
    labels[0] = 1

    with pytest.raises(V1M1BaselineError, match="class-balanced"):
        _build_v1_m1_split_indices(labels)


def test_epoch_progress_uses_only_stable_aggregate_metrics() -> None:
    """每个 epoch 的进度行应含可读聚合指标而不含样本或权重。"""
    metrics = V1M1EpochMetrics(
        epoch=7,
        training_loss=1.25,
        validation=V1M1EvaluationMetrics(
            loss=0.5,
            top1_percent=72.5,
            top5_percent=91.25,
            correct_top1=3_625,
            correct_top5=4_563,
            total=5_000,
            predictions_sha256="a" * 64,
        ),
    )

    progress = _format_v1_m1_epoch_progress(
        V1M1TrainingConfig(run_index=1, seed=1729),
        metrics,
        best_epoch=6,
        best_validation_top1=71.25,
    )

    assert progress == (
        "V1-M1 progress run=1 seed=1729 epoch=7/200 train_loss=1.250000 "
        "validation_loss=0.500000 validation_top1_percent=72.5000 best_epoch=6 "
        "best_validation_top1_percent=71.2500"
    )


def test_batch_progress_uses_fixed_width_bar_and_public_counts() -> None:
    """batch 进度条必须以公开计数同步到 0% 和 100%。"""
    config = V1M1TrainingConfig(run_index=1, seed=1729)

    initial = _format_v1_m1_batch_progress(config, 0, 4, "train", 0, 0, 2)
    completed = _format_v1_m1_batch_progress(config, 4, 4, "complete", 200, 1, 1)

    assert initial == (
        "V1-M1 progress [------------------------------]   0.00% run=1 seed=1729 "
        "stage=train epoch=0/200 batch=0/2"
    )
    assert completed == (
        "V1-M1 progress [##############################] 100.00% run=1 seed=1729 "
        "stage=complete epoch=200/200 batch=1/1"
    )


def test_batch_progress_reports_start_updates_and_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """训练器必须输出开始、实时进度和 artifact 成功后的结束提示。"""
    reporter = _V1M1ProgressReporter(V1M1TrainingConfig(run_index=1, seed=1729), 3)

    reporter.start(first_train_batch_count=1)
    reporter.complete_batch("train", 1, 1, 1)
    reporter.complete_batch("validation", 1, 1, 1)
    reporter.complete_batch("test", 200, 1, 1)
    reporter.finish(final_test_batch_count=1)

    output = capsys.readouterr().out
    assert output.startswith("V1-M1 training started run=1 seed=1729 epochs=200 total_batches=3\n")
    assert "\rV1-M1 progress [------------------------------]   0.00%" in output
    assert "\rV1-M1 progress [##############################] 100.00%" in output
    assert output.endswith("V1-M1 training completed run=1 seed=1729 completed_batches=3/3\n")


def test_runner_stops_at_missing_archive_before_cuda_or_training(tmp_path: Path) -> None:
    """正式 runner 对缺失资源 fail closed 且不下载或初始化训练。"""
    artifact_root = tmp_path / "artifacts"
    with pytest.raises(V1M1BaselineError, match="unavailable"):
        run_v1_m1_baseline(tmp_path, 1, torch.device("cuda"), artifact_root)
    assert not artifact_root.exists()


def test_metadata_decoder_rejects_unknown_fields(tmp_path: Path) -> None:
    """已解压 metadata 的未知字段不能绕过固定 CIFAR-100 契约。"""
    path = tmp_path / "meta"
    with path.open("wb") as stream:
        pickle.dump(
            {
                "fine_label_names": list(V1_M1_FINE_LABEL_NAMES),
                "coarse_label_names": ["coarse"] * 20,
                "unexpected": "reject",
            },
            stream,
        )

    with pytest.raises(V1M1BaselineError, match="fields changed"):
        _decode_meta(path)


def _artifact_result(artifact_root: Path) -> V1M1BaselineResult:
    metrics = V1M1EvaluationMetrics(
        loss=1.0,
        top1_percent=70.0,
        top5_percent=90.0,
        correct_top1=7,
        correct_top5=9,
        total=10,
        predictions_sha256="a" * 64,
    )
    config = V1M1TrainingConfig(run_index=1, seed=1729)
    return V1M1BaselineResult(
        plan=V1M1BaselinePlan(
            archive=V1M1ArchiveManifest(
                filename="cifar-100-python.tar.gz",
                byte_size=169_001_437,
                sha256="b" * 64,
                md5="c" * 32,
            ),
            training=config,
        ),
        dataset_sha256="d" * 64,
        selected_epoch=1,
        epochs=(V1M1EpochMetrics(epoch=1, training_loss=2.0, validation=metrics),),
        test=metrics,
        state_sha256="e" * 64,
        artifacts=_artifact_paths(artifact_root, 1),
    )


def test_artifact_writer_persists_cpu_state_and_structured_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """writer 必须只写 CPU state、manifest 和不含 state 内容的 report。"""
    result = _artifact_result(tmp_path / "artifacts")
    monkeypatch.setattr(
        "can.experiments.v1_m1_baseline._environment_report",
        lambda device: {"device": str(device), "python_hash_seed": "1729"},
    )

    _write_v1_m1_artifacts(result, {"weight": torch.ones(1)}, torch.device("cuda"))

    assert result.artifacts.root.is_dir()
    assert result.artifacts.state.stat().st_mode & 0o777 == 0o600
    assert result.artifacts.manifest.stat().st_mode & 0o777 == 0o600
    assert result.artifacts.report.stat().st_mode & 0o777 == 0o600
    assert torch.load(result.artifacts.state, weights_only=True) == {"weight": torch.ones(1)}
    manifest = json.loads(result.artifacts.manifest.read_text(encoding="utf-8"))
    report = json.loads(result.artifacts.report.read_text(encoding="utf-8"))
    assert manifest["state"]["canonical_state_sha256"] == "e" * 64
    assert manifest["state"]["optimizer_state_saved"] is False
    assert report["state_sha256"] == "e" * 64
    assert "weight" not in result.artifacts.report.read_text(encoding="utf-8")

    with pytest.raises(V1M1BaselineError, match="already exists"):
        _write_v1_m1_artifacts(result, {"weight": torch.ones(1)}, torch.device("cuda"))
