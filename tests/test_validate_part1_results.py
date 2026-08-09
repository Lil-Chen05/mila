"""Publication and CLI tests for the Part 1 production coverage report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from part1_contract import validate_instance


def _coverage_report(
    *,
    ready: bool = True,
    observed_git_commit: str = "5" * 40,
    authoritative_inventory: bool = True,
    shard_count: int = 500,
) -> dict:
    from part1_coverage import (
        CHECKPOINT_ATTRIBUTE_VALUES,
        NATURAL_ATTRIBUTE_VALUES,
        coverage_report_id,
    )

    def matrix(dimensions: dict, count: int) -> dict:
        import itertools

        rows = []
        for index, combination in enumerate(itertools.product(*dimensions.values())):
            rows.append(
                {
                    **dict(zip(dimensions, combination, strict=True)),
                    "count": count if index == 0 else 0,
                }
            )
        return {"dimensions": list(dimensions), "rows": rows}

    model_run_id = "b" * 64

    def source(
        relative_path: str,
        kind: str,
        *,
        shard_id: str | None = None,
        state: str = "regular_file",
        sha256: str = "a" * 64,
        byte_size: int = 1,
    ) -> dict:
        return {
            "relative_path": relative_path,
            "shard_id": shard_id,
            "kind": kind,
            "state": state,
            "sha256": sha256,
            "byte_size": byte_size,
        }

    if authoritative_inventory:
        source_files = [
            source("manifests/part1/questions.jsonl", "questions"),
            source(
                "manifests/part1/questions.manifest.json", "question_manifest"
            ),
            source("manifests/part1/study_manifest.json", "study_manifest"),
            source(
                f"results/part1/{model_run_id}/model_run_manifest.json",
                "model_run_manifest",
            ),
            source("uv.lock", "dependency_lock", sha256="6" * 64),
        ]
        core_sources = (
            (".shard-provenance.json", "shard_provenance"),
            ("natural_results.jsonl", "natural_results"),
            ("checkpoint_results.jsonl", "checkpoint_results"),
            ("audit_events.jsonl", "audit_events"),
            (".finalized", "finalization_marker"),
        )
        for shard_index in range(shard_count):
            shard_id = f"shard-{shard_index:03d}"
            shard_root = f"results/part1/{model_run_id}/raw_shards/{shard_id}"
            source_files.extend(
                source(
                    f"{shard_root}/{filename}",
                    kind,
                    shard_id=shard_id,
                )
                for filename, kind in core_sources
            )
    else:
        source_files = [
            source(
                f"results/part1/{model_run_id}/raw_shards/shard-000/natural_results.jsonl",
                "natural_results",
                shard_id="shard-000",
            )
        ]
    source_files.sort(key=lambda item: item["relative_path"])
    report = {
        "schema_name": "part1_validation_report",
        "schema_version": "1.1.0",
        "validation_report_id": "",
        "study_id": "1" * 64,
        "model_run_id": model_run_id,
        "model_run_manifest_hash": "2" * 64,
        "validated_artifact_kind": "production_coverage",
        "validated_artifact_identity": "b" * 64,
        "validation_started_at": "2026-08-05T00:00:00Z",
        "validation_completed_at": "2026-08-05T00:00:01Z",
        "validator_version": "part1-production-coverage-v1",
        "is_valid": ready,
        "structurally_valid": ready,
        "coverage_complete": True,
        "paper_analysis_ready": ready,
        "checks": [
            {
                "name": "provenance_paths_and_sources",
                "outcome": "passed" if ready else "failed",
                "details": {},
            },
            {"name": "logical_coverage", "outcome": "passed", "details": {}},
            {
                "name": "paper_analysis_readiness",
                "outcome": "passed" if ready else "failed",
                "details": {},
            },
        ],
        "error_count": 0 if ready else 2,
        "warning_count": 0,
        "summary": {
            "question_manifest_hash": "3" * 64,
            "study_manifest_hash": "4" * 64,
            "final_production_git_commit": observed_git_commit,
            "observed_git_commit": observed_git_commit,
            "dependency_lock_sha256": "6" * 64,
            "clean_tracked_worktree": True,
            "expected": {
                "questions": 500,
                "shards": 500,
                "natural_logical_keys": 5000,
                "checkpoint_logical_keys": 55000,
            },
            "observed": {
                "shards": shard_count,
                "natural_physical_records": 5000,
                "checkpoint_physical_records": 55000,
                "source_files": len(source_files),
            },
            "historical_layout_detected": None,
            "natural_partition": {
                "complete": 5000,
                "terminal_infrastructure_failure": 0,
                "retryable_incomplete": 0,
                "missing": 0,
                "duplicate": 0,
                "schema_incompatible": 0,
                "manifest_incompatible": 0,
            },
            "checkpoint_partition": {
                "complete": 55000,
                "terminal_infrastructure_failure": 0,
                "retryable_incomplete": 0,
                "ineligible": 0,
                "missing": 0,
                "duplicate": 0,
            },
            "outcome_counts": {
                "natural_execution_complete": 5000,
                "natural_terminal_infrastructure_failure": 0,
                "checkpoint_execution_complete": 55000,
                "checkpoint_terminal_infrastructure_failure": 0,
                "checkpoint_ineligible": 0,
            },
            "unexpected_physical_record_count": 0,
            "structural_error_count": 0 if ready else 1,
            "structural_warning_count": 0,
            "structural_errors": [] if ready else ["synthetic not-ready report"],
            "structural_warnings": [],
            "natural_model_output_matrix": matrix(NATURAL_ATTRIBUTE_VALUES, 5000),
            "checkpoint_model_output_matrix": matrix(
                CHECKPOINT_ATTRIBUTE_VALUES, 55000
            ),
        },
        "source_files": source_files,
    }
    report["validation_report_id"] = coverage_report_id(report)
    validate_instance("validation_report", report)
    return report


def _rehash(report: dict) -> None:
    from part1_coverage import coverage_report_id

    report["validation_report_id"] = coverage_report_id(report)


def _ignore_repository_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import part1_coverage

    monkeypatch.setattr(part1_coverage, "_global_snapshot_errors", lambda **_kwargs: [])


def test_coverage_schema_preserves_v1_and_accepts_v11_variant() -> None:
    from part1_store_fixtures import MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, STUDY_ID
    from part1_store import Part1ShardStore

    legacy = Part1ShardStore(
        Path("unused"),
        shard_id="shard-000",
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    ).validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-08-05T00:00:00Z",
        completed_at="2026-08-05T00:00:01Z",
    )
    assert legacy["schema_version"] == "1.0.0"
    validate_instance("validation_report", legacy)
    validate_instance("validation_report", _coverage_report())

    legacy_coverage = copy.deepcopy(legacy)
    legacy_coverage["validated_artifact_kind"] = "production_coverage"
    with pytest.raises(ValueError):
        validate_instance("validation_report", legacy_coverage)

    invalid_kind = _coverage_report()
    invalid_kind["source_files"][0]["kind"] = "arbitrary_source"
    with pytest.raises(ValueError):
        validate_instance("validation_report", invalid_kind)

    invalid_shard = _coverage_report()
    shard_source = next(
        item for item in invalid_shard["source_files"] if item["shard_id"] is not None
    )
    shard_source["shard_id"] = "shard-500"
    with pytest.raises(ValueError):
        validate_instance("validation_report", invalid_shard)


def test_coverage_schema_requires_exact_summary_partitions_and_matrices() -> None:
    report = _coverage_report()
    missing_partition_key = copy.deepcopy(report)
    del missing_partition_key["summary"]["natural_partition"]["missing"]
    with pytest.raises(ValueError):
        validate_instance("validation_report", missing_partition_key)

    short_matrix = copy.deepcopy(report)
    short_matrix["summary"]["checkpoint_model_output_matrix"]["rows"].pop()
    with pytest.raises(ValueError):
        validate_instance("validation_report", short_matrix)

    extra_summary = copy.deepcopy(report)
    extra_summary["summary"]["uncontracted"] = True
    with pytest.raises(ValueError):
        validate_instance("validation_report", extra_summary)


def test_report_identity_ignores_timestamps_and_mutable_status_but_hashes_sources() -> None:
    from part1_coverage import coverage_report_id

    first = _coverage_report()
    status_changed = copy.deepcopy(first)
    status_changed.update(
        validation_started_at="2027-01-01T00:00:00Z",
        validation_completed_at="2027-01-01T00:00:01Z",
        is_valid=False,
        structurally_valid=False,
        coverage_complete=False,
        paper_analysis_ready=False,
        error_count=9,
        summary={**status_changed["summary"], "display_status": "FAILED"},
    )
    assert coverage_report_id(status_changed) == first["validation_report_id"]

    changed_source = copy.deepcopy(first)
    changed_source["source_files"][0]["sha256"] = "f" * 64
    assert coverage_report_id(changed_source) != first["validation_report_id"]


def test_atomic_publication_replaces_only_with_valid_complete_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_coverage

    _ignore_repository_snapshot(monkeypatch)
    target = tmp_path / "validation" / "coverage_report.json"
    first = _coverage_report()
    part1_coverage.publish_coverage_report(
        first, target, repository_root=tmp_path
    )
    assert json.loads(target.read_text(encoding="utf-8")) == first
    assert not list(target.parent.glob(".coverage_report.json.*.tmp"))

    before = target.read_bytes()
    second = copy.deepcopy(first)
    second["validation_completed_at"] = "2026-08-05T00:00:02Z"
    real_replace = part1_coverage.os.replace

    def fail_replace(source, destination):
        if Path(destination) == target:
            raise OSError("synthetic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(part1_coverage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        part1_coverage.publish_coverage_report(
            second, target, repository_root=tmp_path
        )
    assert target.read_bytes() == before
    assert not list(target.parent.glob(".coverage_report.json.*.tmp"))


def test_atomic_publication_rejects_symlink_and_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    outside = tmp_path / "outside.json"
    outside.write_text("owned", encoding="utf-8")
    target = tmp_path / "validation" / "coverage_report.json"
    target.parent.mkdir()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        publish_coverage_report(
            _coverage_report(), target, repository_root=tmp_path
        )
    assert outside.read_text(encoding="utf-8") == "owned"

    target.unlink()
    invalid = _coverage_report()
    invalid["model_run_id"] = "bad"
    with pytest.raises(ValueError):
        publish_coverage_report(invalid, target, repository_root=tmp_path)
    assert not target.exists()

    target.parent.rmdir()
    target.parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory|parent"):
        publish_coverage_report(
            _coverage_report(), target, repository_root=tmp_path
        )


def test_atomic_publication_rejects_inconsistent_readiness_booleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    report = _coverage_report()
    report["structurally_valid"] = False
    with pytest.raises(ValueError, match="is_valid|structurally"):
        publish_coverage_report(
            report,
            tmp_path / "coverage_report.json",
            repository_root=tmp_path,
        )
    assert not (tmp_path / "coverage_report.json").exists()

    report = _coverage_report()
    report["coverage_complete"] = False
    with pytest.raises(ValueError, match="paper_analysis_ready"):
        publish_coverage_report(
            report,
            tmp_path / "coverage_report.json",
            repository_root=tmp_path,
        )


def test_atomic_publication_enforces_partition_matrix_and_source_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    wrong_partition = _coverage_report()
    wrong_partition["summary"]["natural_partition"]["complete"] = 4_999
    with pytest.raises(ValueError, match="natural partition"):
        publish_coverage_report(
            wrong_partition,
            tmp_path / "coverage_report.json",
            repository_root=tmp_path,
        )

    wrong_matrix = _coverage_report()
    wrong_matrix["summary"]["natural_model_output_matrix"]["rows"][0]["count"] = 4_999
    with pytest.raises(ValueError, match="matrix"):
        publish_coverage_report(
            wrong_matrix,
            tmp_path / "coverage_report.json",
            repository_root=tmp_path,
        )

    wrong_source = _coverage_report()
    wrong_source["source_files"][0].update(
        state="absent", sha256="a" * 64, byte_size=1
    )
    with pytest.raises(ValueError, match="absent"):
        publish_coverage_report(
            wrong_source,
            tmp_path / "coverage_report.json",
            repository_root=tmp_path,
        )


def test_publish_rejects_one_source_forged_ready_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    forged = _coverage_report(authoritative_inventory=False)

    with pytest.raises(ValueError, match="global|inventory|authoritative"):
        publish_coverage_report(
            forged,
            tmp_path / "forged.json",
            repository_root=tmp_path,
        )
    assert not (tmp_path / "forged.json").exists()


def test_publish_rejects_kind_shard_and_path_inconsistencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    mutations = (
        lambda item: item.update(kind="checkpoint_results"),
        lambda item: item.update(shard_id="shard-001"),
        lambda item: item.update(
            relative_path=item["relative_path"].replace(
                "natural_results.jsonl", "misnamed-natural.jsonl"
            )
        ),
    )
    for index, mutate in enumerate(mutations):
        report = _coverage_report()
        item = next(
            source
            for source in report["source_files"]
            if source["kind"] == "natural_results"
            and source["shard_id"] == "shard-000"
        )
        mutate(item)
        report["source_files"].sort(key=lambda source: source["relative_path"])
        _rehash(report)
        target = tmp_path / f"mismatch-{index}.json"

        with pytest.raises(ValueError, match="canonical|inventory|shard|kind|path"):
            publish_coverage_report(
                report,
                target,
                repository_root=tmp_path,
            )
        assert not target.exists()


def test_publish_rejects_missing_global_and_ready_core_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    missing_global = _coverage_report()
    missing_global["source_files"] = [
        item
        for item in missing_global["source_files"]
        if item["kind"] != "questions"
    ]
    missing_global["summary"]["observed"]["source_files"] -= 1
    _rehash(missing_global)
    with pytest.raises(ValueError, match="global|questions|inventory"):
        publish_coverage_report(
            missing_global,
            tmp_path / "missing-global.json",
            repository_root=tmp_path,
        )

    missing_core = _coverage_report()
    missing_core["source_files"] = [
        item
        for item in missing_core["source_files"]
        if not (
            item["kind"] == "audit_events"
            and item["shard_id"] == "shard-499"
        )
    ]
    missing_core["summary"]["observed"]["source_files"] -= 1
    _rehash(missing_core)
    with pytest.raises(ValueError, match="core|audit|inventory|shard-499"):
        publish_coverage_report(
            missing_core,
            tmp_path / "missing-core.json",
            repository_root=tmp_path,
        )


def test_publish_rejects_surplus_ready_physical_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    report = _coverage_report()
    report["summary"]["observed"]["natural_physical_records"] += 1

    with pytest.raises(ValueError, match="natural physical-record|surplus"):
        publish_coverage_report(
            report,
            tmp_path / "surplus.json",
            repository_root=tmp_path,
        )


def test_publish_requires_repository_root(tmp_path: Path) -> None:
    from part1_coverage import publish_coverage_report

    with pytest.raises(TypeError, match="repository_root"):
        publish_coverage_report(_coverage_report(), tmp_path / "coverage.json")


def test_invalid_partial_shard_inventory_remains_publishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_coverage import publish_coverage_report

    _ignore_repository_snapshot(monkeypatch)
    report = _coverage_report(ready=False, shard_count=1)
    target = tmp_path / "partial.json"
    publish_coverage_report(report, target, repository_root=tmp_path)
    assert json.loads(target.read_text(encoding="utf-8")) == report


def test_cli_publishes_report_and_returns_readiness_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import validate_part1_results

    _ignore_repository_snapshot(monkeypatch)
    model_manifest = tmp_path / "model_run_manifest.json"
    model_manifest.write_text(
        json.dumps(
            {
                "output_paths": {
                    "validation": (
                        "results/part1/" + "b" * 64 + "/validation"
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    reports = iter(
        (
            _coverage_report(ready=True),
            _coverage_report(ready=False, shard_count=1),
        )
    )
    monkeypatch.setattr(
        validate_part1_results,
        "build_coverage_report",
        lambda **_kwargs: next(reports),
    )

    args = [
        "--repository-root",
        str(tmp_path),
        "--model-run-manifest",
        str(model_manifest),
    ]
    assert validate_part1_results.main(args) == 0
    target = (
        tmp_path
        / "results"
        / "part1"
        / ("b" * 64)
        / "validation"
        / "coverage_report.json"
    )
    assert target.exists()
    assert validate_part1_results.main(args) == 2
    assert json.loads(target.read_text(encoding="utf-8"))["paper_analysis_ready"] is False
    assert "coverage_report.json" in capsys.readouterr().out


def test_cli_failure_never_leaves_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import validate_part1_results

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        validate_part1_results,
        "build_coverage_report",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("unsafe input")),
    )
    assert validate_part1_results.main(
        [
            "--repository-root",
            str(tmp_path),
            "--model-run-manifest",
            str(manifest),
        ]
    ) == 2
    assert not list(tmp_path.rglob("coverage_report.json"))
