"""CPU-only tests for publishing an already-computed analysis stage."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from test_part1_analysis import _fixture_source


def _preserved_stage(tmp_path: Path):
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    output, manifest = publish_analysis(source, bootstrap_replicates=2)
    stage = output.with_name(f".{output.name}.stage-preserved")
    output.rename(stage)
    return stage, output, manifest


def test_finalize_preserved_analysis_stage_validates_and_atomically_publishes(
    tmp_path: Path,
) -> None:
    from part1_analysis_stage_recovery import finalize_preserved_analysis_stage

    stage, target, manifest = _preserved_stage(tmp_path)
    stage_inode = stage.stat().st_ino

    published = finalize_preserved_analysis_stage(
        stage=stage,
        target_name=target.name,
        expected_model_run_id=manifest["model_run_id"],
        expected_analysis_id=manifest["analysis_id"],
        expected_bootstrap_replicates=2,
    )

    assert published == manifest
    assert target.stat().st_ino == stage_inode
    assert not stage.exists()


def test_finalize_preserved_analysis_stage_rejects_identity_mismatch(
    tmp_path: Path,
) -> None:
    from part1_analysis_stage_recovery import finalize_preserved_analysis_stage

    stage, target, manifest = _preserved_stage(tmp_path)

    with pytest.raises(ValueError, match="identity differs"):
        finalize_preserved_analysis_stage(
            stage=stage,
            target_name=target.name,
            expected_model_run_id=manifest["model_run_id"],
            expected_analysis_id="f" * 64,
            expected_bootstrap_replicates=2,
        )

    assert stage.is_dir()
    assert not target.exists()


def test_finalize_preserved_analysis_stage_never_overwrites_existing_target(
    tmp_path: Path,
) -> None:
    from part1_analysis_stage_recovery import finalize_preserved_analysis_stage

    stage, target, manifest = _preserved_stage(tmp_path)
    target.mkdir()
    marker = target / "user-owned"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        finalize_preserved_analysis_stage(
            stage=stage,
            target_name=target.name,
            expected_model_run_id=manifest["model_run_id"],
            expected_analysis_id=manifest["analysis_id"],
            expected_bootstrap_replicates=2,
        )

    assert marker.read_text() == "keep"
    assert stage.is_dir()


def test_finalize_analysis_stage_cli_reports_published_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from finalize_part1_analysis_stage import main

    stage, target, manifest = _preserved_stage(tmp_path)
    result = main(
        [
            "--stage",
            str(stage),
            "--target-name",
            target.name,
            "--expected-model-run-id",
            manifest["model_run_id"],
            "--expected-analysis-id",
            manifest["analysis_id"],
            "--expected-bootstrap-replicates",
            "2",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "analysis_id": manifest["analysis_id"],
        "analysis_manifest_hash": manifest["analysis_manifest_hash"],
        "bootstrap_replicates": 2,
        "paper_analysis_ready": True,
        "status": "published",
        "target": str(target),
    }


def test_analysis_stage_recovery_job_is_short_cpu_only_and_exactly_parameterized() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "part1_finalize_analysis_stage.sh"
    ).read_text()

    assert "#SBATCH --time=1:00:00" in job
    assert "#SBATCH --gpus" not in job
    assert ': "${ANALYSIS_STAGE:?' in job
    assert ': "${EXPECTED_ANALYSIS_ID:?' in job
    assert "--expected-bootstrap-replicates 5000" in job
    assert "srun --cpu-bind=none" in job
