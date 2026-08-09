"""Synthetic CPU-only tests for the Part 1 analysis artifact lifecycle."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import pytest

from part1_contract import FIXED_CHECKPOINT_FRACTIONS, load_config


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fixture_source(repository_root: Path):
    from part1_analysis import AnalysisSource
    from test_part1_trajectories import _checkpoint, _natural

    naturals: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    question_frame: list[dict[str, str]] = []
    for sample_index in range(2):
        question_id = _hex(f"question-{sample_index}")
        question_frame.append(
            {"subject": "high_school_mathematics", "question_id": question_id}
        )
        for run_id, answer in enumerate(("A", "C")):
            natural = _natural(
                sample_index=sample_index,
                run_id=run_id,
                answer=answer,
                confidence=0.2 if answer == "A" else 0.8,
            )
            naturals.append(natural)
            checkpoints.extend(
                _checkpoint(natural, index, answer="C") for index in range(11)
            )

    model_run_id = naturals[0]["model_run_id"]
    model_manifest = {
        "study_id": naturals[0]["study_id"],
        "study_manifest_hash": "5" * 64,
        "question_manifest_hash": naturals[0]["question_manifest_hash"],
        "model_run_id": model_run_id,
        "model_run_manifest_hash": naturals[0]["model_run_manifest_hash"],
        "output_paths": {
            "analysis": f"results/part1/{model_run_id}/analysis",
        },
    }
    coverage = {
        "validation_report_id": "6" * 64,
        "paper_analysis_ready": True,
        "summary": {
            "natural_partition": {"terminal_infrastructure_failure": 0},
            "checkpoint_partition": {"terminal_infrastructure_failure": 0},
        },
    }
    merge = {
        "merge_id": "7" * 64,
        "merge_manifest_hash": "8" * 64,
        "study_id": model_manifest["study_id"],
        "study_manifest_hash": model_manifest["study_manifest_hash"],
        "question_manifest_hash": model_manifest["question_manifest_hash"],
        "model_run_id": model_run_id,
        "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
        "coverage_report_id": coverage["validation_report_id"],
    }
    revalidations: list[str] = []
    return AnalysisSource(
        repository_root=repository_root,
        model_manifest=model_manifest,
        merge_manifest=merge,
        coverage_report=coverage,
        analysis_config=load_config("analysis"),
        question_frame=tuple(question_frame),
        natural_rows=tuple(naturals),
        checkpoint_rows=tuple(checkpoints),
        small_fixture=True,
        revalidate_inputs=lambda: revalidations.append("checked"),
    ), revalidations


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return list(reader.fieldnames or ()), list(reader), data


def _strict_loader_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Build tiny real files while replacing only upstream accepted validators."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import part1_analysis as analysis

    repository = tmp_path / "repository"
    model_run_id = "a" * 64
    run_root = repository / "results" / "part1" / model_run_id
    merged = run_root / "merged"
    validation = run_root / "validation"
    config_path = repository / "configs" / "part1" / "analysis.json"
    merged.mkdir(parents=True)
    validation.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "configs" / "part1" / "analysis.json",
        config_path,
    )

    table_rows = {
        "natural_results": ({"row_kind": "natural"},),
        "checkpoint_results": ({"row_kind": "checkpoint"},),
    }
    outputs: dict[str, dict[str, Any]] = {}
    for kind, filename in analysis.MERGE_TABLE_FILENAMES.items():
        if kind not in table_rows:
            continue
        table = pa.table({"fixture": [kind]})
        path = merged / filename
        pq.write_table(table, path)
        data = path.read_bytes()
        outputs[kind] = {
            "relative_path": path.relative_to(repository).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
            "row_count": 1,
        }

    model = {
        "schema_version": "1.1.0",
        "production": True,
        "execution_scope": "production",
        "clean_tracked_worktree": True,
        "study_id": "1" * 64,
        "study_manifest_hash": "2" * 64,
        "question_manifest_hash": "3" * 64,
        "model_run_id": model_run_id,
        "model_run_manifest_hash": "4" * 64,
        "output_paths": {
            "raw_shards": f"results/part1/{model_run_id}/raw_shards",
            "validation": f"results/part1/{model_run_id}/validation",
            "merged": f"results/part1/{model_run_id}/merged",
            "analysis": f"results/part1/{model_run_id}/analysis",
        },
    }
    model_path = run_root / "model_run_manifest.json"
    model_path.write_bytes(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    coverage = {
        "validation_report_id": "5" * 64,
        "paper_analysis_ready": True,
        "validation_started_at": "2026-01-01T00:00:00Z",
        "validation_completed_at": "2026-01-01T00:01:00Z",
        "source_files": [],
        "summary": {
            "observed_git_commit": "6" * 40,
            "clean_tracked_worktree": True,
            "natural_partition": {"terminal_infrastructure_failure": 0},
            "checkpoint_partition": {"terminal_infrastructure_failure": 0},
        },
    }
    coverage_path = validation / "coverage_report.json"

    def write_coverage() -> bytes:
        data = json.dumps(coverage, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        coverage_path.write_bytes(data)
        return data

    coverage_bytes = write_coverage()
    merge_manifest = {
        "study_id": model["study_id"],
        "study_manifest_hash": model["study_manifest_hash"],
        "question_manifest_hash": model["question_manifest_hash"],
        "model_run_id": model["model_run_id"],
        "model_run_manifest_hash": model["model_run_manifest_hash"],
        "merge_id": "7" * 64,
        "merge_manifest_hash": "8" * 64,
        "coverage_report_id": coverage["validation_report_id"],
        "coverage_report": {
            "relative_path": coverage_path.relative_to(repository).as_posix(),
            "sha256": hashlib.sha256(coverage_bytes).hexdigest(),
            "byte_size": len(coverage_bytes),
        },
        "outputs": outputs,
    }
    records = tuple(
        {
            "sample_index": index,
            "subject": "high_school_mathematics",
            "question_id": f"{index:064x}",
        }
        for index in range(500)
    )
    bundle = SimpleNamespace(records=records, study_manifest={"study_id": model["study_id"]})
    calls: dict[str, Any] = {
        "descriptor_reads": [],
        "merge_validations": 0,
        "compatibility": 0,
        "coverage_semantics": 0,
        "snapshot_errors": [],
    }

    monkeypatch.setattr(analysis, "validate_instance", lambda *_args: None)
    monkeypatch.setattr(analysis, "validate_fixed_model_requested_contract", lambda *_args: None)
    monkeypatch.setattr(analysis, "model_run_id", lambda value: value["model_run_id"])
    monkeypatch.setattr(
        analysis,
        "model_run_manifest_hash",
        lambda value: value["model_run_manifest_hash"],
    )

    def validate_merge(_path: Path, *, expected_manifest: Any = None):
        calls["merge_validations"] += 1
        if expected_manifest is not None:
            assert expected_manifest == merge_manifest
        return copy.deepcopy(merge_manifest)

    monkeypatch.setattr(analysis, "validate_merge_directory", validate_merge)
    monkeypatch.setattr(analysis, "validate_merge_directory_at", lambda *_args, **_kwargs: None)
    original_read_at = analysis._read_regular_file_at

    def read_at(descriptor: int, filename: str) -> bytes:
        calls["descriptor_reads"].append(filename)
        return original_read_at(descriptor, filename)

    monkeypatch.setattr(analysis, "_read_regular_file_at", read_at)
    monkeypatch.setattr(
        analysis,
        "decode_merge_table",
        lambda kind, _table: list(copy.deepcopy(table_rows[kind])),
    )
    monkeypatch.setattr(analysis, "coverage_report_id", lambda value: value["validation_report_id"])

    def validate_coverage(_value: Any) -> None:
        calls["coverage_semantics"] += 1

    monkeypatch.setattr(analysis, "validate_coverage_report_semantics", validate_coverage)
    monkeypatch.setattr(analysis, "build_coverage_report", lambda **_kwargs: copy.deepcopy(coverage))
    monkeypatch.setattr(analysis, "load_manifest_bundle", lambda **_kwargs: bundle)

    def compatible(_study: Any, _model: Any) -> None:
        calls["compatibility"] += 1

    monkeypatch.setattr(analysis, "validate_manifest_compatibility", compatible)
    monkeypatch.setattr(
        analysis,
        "_global_snapshot_errors",
        lambda **_kwargs: list(calls["snapshot_errors"]),
    )
    return {
        "analysis": analysis,
        "repository": repository,
        "model": model,
        "model_path": model_path,
        "merge": merge_manifest,
        "coverage": coverage,
        "coverage_path": coverage_path,
        "write_coverage": write_coverage,
        "bundle": bundle,
        "calls": calls,
    }


def test_analysis_config_uses_an_independent_exact_json_typed_oracle() -> None:
    from part1_analysis import analysis_config_hash, validate_analysis_config

    config = load_config("analysis")
    validate_analysis_config(config)
    first_hash = analysis_config_hash(config)
    assert len(first_hash) == 64

    for field, value in (
        ("bootstrap_seed", 42.0),
        ("final_bootstrap_replicates", 1000),
        ("primary_target", "checkpoint_local_correct"),
    ):
        drifted = copy.deepcopy(config)
        drifted[field] = value
        with pytest.raises(ValueError, match="analysis config"):
            validate_analysis_config(drifted)


def test_tracked_pretty_printed_analysis_config_loads_as_exact_json_typed_input() -> None:
    from part1_analysis import _read_analysis_config

    repository_root = Path(__file__).resolve().parents[1]
    config, data, path = _read_analysis_config(repository_root)
    assert config == load_config("analysis")
    assert data == path.read_bytes()
    assert data.startswith(b"{\n")


def test_strict_production_loader_reads_canonical_descriptor_bound_inputs_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    source = fixture["analysis"].load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
    )

    assert source.small_fixture is False
    assert source.natural_rows == ({"row_kind": "natural"},)
    assert source.checkpoint_rows == ({"row_kind": "checkpoint"},)
    assert source.question_frame[0]["question_id"] == f"{0:064x}"
    assert source.question_frame[-1]["question_id"] == f"{499:064x}"
    assert fixture["calls"]["descriptor_reads"] == [
        fixture["analysis"].MERGE_TABLE_FILENAMES["natural_results"],
        fixture["analysis"].MERGE_TABLE_FILENAMES["checkpoint_results"],
    ]
    assert fixture["calls"]["coverage_semantics"] == 1
    assert fixture["calls"]["compatibility"] == 1
    assert fixture["calls"]["merge_validations"] >= 3

    fixture["calls"]["snapshot_errors"].append("tracked source changed")
    with pytest.raises(ValueError, match="source/Git snapshot changed"):
        source.revalidate_inputs()


def test_strict_production_loader_rejects_merged_table_hash_drift_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    natural_path = (
        fixture["repository"]
        / fixture["model"]["output_paths"]["merged"]
        / fixture["analysis"].MERGE_TABLE_FILENAMES["natural_results"]
    )
    natural_path.write_bytes(natural_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="natural_results bytes differ"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production", False),
        ("execution_scope", "smoke"),
        ("schema_version", "1.0.0"),
    ],
)
def test_strict_production_loader_rejects_smoke_or_legacy_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    fixture["model"][field] = value
    fixture["model_path"].write_bytes(
        json.dumps(fixture["model"], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    with pytest.raises(ValueError, match="canonical production"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_rejects_noncanonical_path_and_mixed_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    alternate = fixture["repository"] / "alternate-model-run-manifest.json"
    alternate.write_bytes(fixture["model_path"].read_bytes())
    with pytest.raises(ValueError, match="path is not canonical"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"], model_run_manifest_path=alternate
        )

    fixture["merge"]["question_manifest_hash"] = "9" * 64
    with pytest.raises(ValueError, match="model/merge provenance differs"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_rejects_coverage_identity_readiness_and_rebuild_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    fixture["coverage"]["paper_analysis_ready"] = False
    data = fixture["write_coverage"]()
    fixture["merge"]["coverage_report"]["sha256"] = hashlib.sha256(data).hexdigest()
    fixture["merge"]["coverage_report"]["byte_size"] = len(data)
    with pytest.raises(ValueError, match="paper-final analysis requires"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )

    fixture = _strict_loader_fixture(tmp_path / "identity", monkeypatch)
    fixture["merge"]["coverage_report_id"] = "f" * 64
    with pytest.raises(ValueError, match="coverage bytes or identity"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )

    fixture = _strict_loader_fixture(tmp_path / "rebuild", monkeypatch)
    monkeypatch.setattr(
        fixture["analysis"],
        "build_coverage_report",
        lambda **_kwargs: {"different": True},
    )
    with pytest.raises(ValueError, match="current immutable source/Git snapshot"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_rejects_tracked_bundle_or_dependency_incompatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture["analysis"],
        "load_manifest_bundle",
        lambda **_kwargs: SimpleNamespace(
            records=fixture["bundle"].records[:-1],
            study_manifest=fixture["bundle"].study_manifest,
        ),
    )
    with pytest.raises(ValueError, match="fixed ordered 500-question frame"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )

    fixture = _strict_loader_fixture(tmp_path / "dependency", monkeypatch)

    def incompatible(*_args: Any) -> None:
        raise ValueError("dependency compatibility drift")

    monkeypatch.setattr(fixture["analysis"], "validate_manifest_compatibility", incompatible)
    with pytest.raises(ValueError, match="dependency compatibility drift"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_revalidation_detects_config_or_coverage_byte_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    source = fixture["analysis"].load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
    )
    fixture["coverage"]["validation_completed_at"] = "2026-01-01T00:02:00Z"
    fixture["coverage_path"].write_bytes(
        json.dumps(
            fixture["coverage"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    with pytest.raises(ValueError, match="coverage report changed"):
        source.revalidate_inputs()

    fixture = _strict_loader_fixture(tmp_path / "config", monkeypatch)
    source = fixture["analysis"].load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
    )
    config_path = fixture["repository"] / "configs" / "part1" / "analysis.json"
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="analysis config changed"):
        source.revalidate_inputs()


def test_one_trajectory_build_and_one_shared_plan_feed_every_fixed_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    original_builder = analysis.build_trajectory_rows
    builder_calls = 0
    plan_ids: list[int] = []

    def build_once(*args: Any, **kwargs: Any):
        nonlocal builder_calls
        builder_calls += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(analysis, "build_trajectory_rows", build_once)
    for name in (
        "primary_auroc_analysis",
        "natural_calibration_analysis",
        "checkpoint_calibration_analysis",
        "secondary_checkpoint_auroc_analysis",
        "within_question_analysis",
    ):
        original = getattr(analysis, name)

        def wrapper(rows: Any, plan: Any, *args: Any, _original=original, **kwargs: Any):
            plan_ids.append(id(plan))
            return _original(rows, plan, *args, **kwargs)

        monkeypatch.setattr(analysis, name, wrapper)

    computation = analysis.compute_analysis(source, bootstrap_replicates=3)
    assert builder_calls == 1
    assert len(set(plan_ids)) == 1
    assert len(computation.trajectory_rows) == len(source.natural_rows)
    assert computation.summary["repetition_filter_applied"] is False
    assert computation.summary["successful_abnormal_output_policy"] == "preserved"
    checkpoint = computation.analyses["checkpoint_calibration"]
    assert checkpoint["predictors"] == [
        "checkpoint_normalized_confidence",
        "checkpoint_maximum_ad_probability",
    ]
    assert {
        row["requested_fraction"] for row in checkpoint["metric_rows"]
    } == set(FIXED_CHECKPOINT_FRACTIONS)
    assert {
        row["requested_fraction"]
        for row in checkpoint["metric_rows"]
        if row["is_main_checkpoint"]
    } == {0.0, 0.5, 1.0}
    assert all(row["target"] == "natural_correct" for row in computation.tables["primary_auroc"])
    assert all(
        row["target"] == "checkpoint_local_correct"
        for row in computation.tables["secondary_checkpoint_auroc"]
    )
    assert all("entropy" not in row["predictor"] for row in computation.tables["calibration_metrics"])


def test_switching_stabilization_summary_preserves_unavailable_and_not_applicable(
    tmp_path: Path,
) -> None:
    from part1_analysis import compute_analysis

    source, _ = _fixture_source(tmp_path)
    computation = compute_analysis(source, bootstrap_replicates=2)
    pooled = computation.tables["trajectory_events"][0]
    assert pooled["grouping"] == "pooled"
    assert pooled["trajectory_count"] == 4
    assert pooled["switch_count_available"] == 4
    assert pooled["switch_count_unavailable"] == 0
    assert pooled["first_appearance_found"] == 2
    assert pooled["first_appearance_not_found"] == 2
    assert pooled["later_recovery_not_applicable"] == 4
    assert pooled["endpoint_agreement_true"] == 2
    assert pooled["endpoint_agreement_false"] == 2
    assert pooled["stabilization_computed"] == 4
    assert pooled["stabilization_unavailable"] == 0


def test_deterministic_artifacts_sidecars_summary_and_plot_series(tmp_path: Path) -> None:
    from part1_analysis import EXPECTED_ARTIFACT_NAMES, publish_analysis

    source, revalidations = _fixture_source(tmp_path)
    output, manifest = publish_analysis(source, bootstrap_replicates=3)
    assert output.name == "development-r3"
    assert set(path.name for path in output.iterdir()) == set(EXPECTED_ARTIFACT_NAMES)
    assert revalidations == ["checked", "checked"]
    assert manifest == json.loads((output / "analysis_manifest.json").read_text())
    summary = json.loads((output / "analysis_summary.json").read_text())
    assert summary["analysis_id"] == manifest["analysis_id"]
    assert summary["paper_analysis_ready"] is True
    assert summary["terminal_infrastructure_failure_count"] == 0
    assert summary["all_checkpoint_fractions_table"] == "calibration_metrics.csv"
    assert len(summary["plot_series"]["checkpoint_ece"]) == 22
    assert {
        row["requested_fraction"]
        for row in summary["plot_series"]["checkpoint_ece"]
        if row["is_main_checkpoint"]
    } == {0.0, 0.5, 1.0}

    for table_name, metadata_name in manifest["tables"].items():
        columns, rows, data = _read_csv(output / table_name)
        assert not data.startswith(b"#")
        assert data.count(b"\n") == len(rows) + 1
        metadata = json.loads((output / metadata_name).read_text())
        assert metadata["ordered_columns"] == columns
        assert metadata["row_count"] == len(rows)
        assert metadata["table_sha256"] == hashlib.sha256(data).hexdigest()
        assert metadata["analysis_config_hash"] == manifest["analysis_config_hash"]
        assert metadata["source_provenance"]["merge_manifest_hash"] == "8" * 64
    _, trajectory_rows, _ = _read_csv(output / "trajectory_features.csv")
    assert isinstance(json.loads(trajectory_rows[0]["checkpoint_calibration"]), list)
    assert isinstance(json.loads(trajectory_rows[0]["feature_missing_reasons"]), dict)
    for plot in manifest["plots"]:
        payload = (output / plot).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) > 100


def test_identity_is_relocation_invariant_and_sensitive_to_bootstrap_or_source(
    tmp_path: Path,
) -> None:
    from part1_analysis import compute_analysis

    first, _ = _fixture_source(tmp_path / "first")
    second, _ = _fixture_source(tmp_path / "second")
    one = compute_analysis(first, bootstrap_replicates=2)
    relocated = compute_analysis(second, bootstrap_replicates=2)
    changed_replicates = compute_analysis(first, bootstrap_replicates=3)
    changed_source = copy.deepcopy(first.merge_manifest)
    changed_source["merge_manifest_hash"] = "9" * 64
    altered = copy.copy(first)
    object.__setattr__(altered, "merge_manifest", changed_source)
    changed_merge = compute_analysis(altered, bootstrap_replicates=2)
    assert one.analysis_id == relocated.analysis_id
    assert one.analysis_id != changed_replicates.analysis_id
    assert one.analysis_id != changed_merge.analysis_id


def test_identical_rerun_keeps_inode_and_malformed_collision_fails(tmp_path: Path) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    inode = output.stat().st_ino
    same, _ = publish_analysis(source, bootstrap_replicates=2)
    assert same.stat().st_ino == inode
    (output / "unexpected.txt").write_text("collision")
    with pytest.raises(ValueError, match="missing or extra"):
        publish_analysis(source, bootstrap_replicates=2)


def test_manifest_standalone_identity_and_json_types_reject_rehashed_tampering(
    tmp_path: Path,
) -> None:
    from part1_analysis import (
        _analysis_manifest_hash,
        publish_analysis,
        validate_analysis_manifest,
    )

    source, _ = _fixture_source(tmp_path)
    _output, manifest = publish_analysis(source, bootstrap_replicates=2)
    validate_analysis_manifest(manifest)
    for field, value in (
        ("merge_manifest_hash", "9" * 64),
        ("bootstrap_seed", 42.0),
        ("bootstrap_replicates", 2.0),
    ):
        tampered = copy.deepcopy(manifest)
        tampered[field] = value
        tampered["analysis_manifest_hash"] = _analysis_manifest_hash(tampered)
        with pytest.raises(ValueError):
            validate_analysis_manifest(tampered)


@pytest.mark.parametrize(
    "boundary",
    [
        "stage_created",
        "artifacts_written",
        "stage_fsynced",
        "reload_complete",
        "before_input_revalidation",
        "before_exclusive_rename",
    ],
)
def test_prepublication_faults_leave_no_final_and_clean_own_stage(
    tmp_path: Path, boundary: str
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)

    def fail(observed: str) -> None:
        if observed == boundary:
            raise RuntimeError(boundary)

    with pytest.raises(RuntimeError, match=boundary):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=fail)
    analysis_root = (
        tmp_path / source.model_manifest["output_paths"]["analysis"]
    )
    assert not (analysis_root / "development-r2").exists()
    assert not list(analysis_root.glob(".development-r2.stage-*"))


def test_postrename_fault_rolls_back_without_claiming_success(tmp_path: Path) -> None:
    from part1_analysis import PublicationDurabilityError, publish_analysis

    source, _ = _fixture_source(tmp_path)

    def fail(boundary: str) -> None:
        if boundary == "after_exclusive_rename":
            raise OSError("injected post-rename durability failure")

    with pytest.raises(PublicationDurabilityError, match="rolled back"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=fail)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    assert not (root / "development-r2").exists()
    assert not list(root.glob(".development-r2.stage-*"))


def test_stage_cleanup_quarantines_by_open_identity_and_restores_a_substitution(
    tmp_path: Path,
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    analysis_root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def fault(boundary: str) -> None:
        nonlocal held
        if boundary == "artifacts_written":
            raise RuntimeError("enter cleanup")
        if boundary == "before_stage_cleanup_identity_move":
            stage = next(analysis_root.glob(".development-r2.stage-*"))
            held = stage.with_name(f"{stage.name}.held")
            stage.rename(held)
            stage.mkdir()
            (stage / "replacement-marker").write_text("unrelated", encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity changed"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=fault)
    assert held is not None and held.is_dir()
    replacement = next(
        path
        for path in analysis_root.glob(".development-r2.stage-*")
        if path != held
    )
    assert (replacement / "replacement-marker").read_text() == "unrelated"
    assert not (analysis_root / "development-r2").exists()


def test_two_publishers_converge_on_one_identical_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    (tmp_path / source.model_manifest["output_paths"]["analysis"]).mkdir(
        parents=True
    )
    barrier = Barrier(2)
    render_lock = Lock()
    original_render = analysis._render_plots

    # Matplotlib's process-global state is not thread-safe. Serialize only plot
    # rendering so the test isolates the publication race it is intended to cover.
    def render(*args: Any, **kwargs: Any) -> None:
        with render_lock:
            original_render(*args, **kwargs)

    monkeypatch.setattr(analysis, "_render_plots", render)

    def publish():
        def hook(boundary: str) -> None:
            if boundary == "before_exclusive_rename":
                barrier.wait(timeout=30)

        return analysis.publish_analysis(source, bootstrap_replicates=2, fault_hook=hook)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result() for future in (executor.submit(publish), executor.submit(publish))]
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]
    assert not list(results[0][0].parent.glob(".development-r2.stage-*"))


@pytest.mark.parametrize("mutation", ["missing", "content", "symlink"])
def test_existing_partial_divergent_or_symlinked_artifact_is_incompatible(
    tmp_path: Path, mutation: str
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    target = output / "primary_auroc.csv"
    if mutation == "missing":
        target.unlink()
    elif mutation == "content":
        target.write_bytes(target.read_bytes() + b"corrupt\n")
    else:
        target.unlink()
        target.symlink_to(output / "analysis_summary.json")
    with pytest.raises(ValueError):
        publish_analysis(source, bootstrap_replicates=2)


def test_false_paper_readiness_or_any_terminal_failure_blocks_publication(
    tmp_path: Path,
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    for mutator in (
        lambda coverage: coverage.update(paper_analysis_ready=False),
        lambda coverage: coverage["summary"]["natural_partition"].update(
            terminal_infrastructure_failure=1
        ),
        lambda coverage: coverage["summary"]["checkpoint_partition"].update(
            terminal_infrastructure_failure=1
        ),
    ):
        changed = copy.deepcopy(source.coverage_report)
        mutator(changed)
        invalid = copy.copy(source)
        object.__setattr__(invalid, "coverage_report", changed)
        with pytest.raises(ValueError, match="paper-final analysis"):
            publish_analysis(invalid, bootstrap_replicates=2)


def test_analysis_publication_rejects_symlinked_final_directory(tmp_path: Path) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    analysis_root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    analysis_root.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (analysis_root / "development-r2").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked|non-directory"):
        publish_analysis(source, bootstrap_replicates=2)


def test_fixture_evidence_is_explicitly_limited(tmp_path: Path) -> None:
    from part1_analysis import FIXTURE_EVIDENCE_LIMITS, compute_analysis

    source, _ = _fixture_source(tmp_path)
    result = compute_analysis(source, bootstrap_replicates=1)
    assert result.summary["evidence_scope"] == FIXTURE_EVIDENCE_LIMITS
    assert "real SmolLM3" in FIXTURE_EVIDENCE_LIMITS
