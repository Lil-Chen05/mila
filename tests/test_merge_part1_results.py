"""Coverage gate, atomic publication, and CLI tests for the Part 1 merge."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest


def _prepared(tmp_path: Path):
    from part1_merge import MergeInputs
    from part1_store import Part1ShardStore
    from test_part1_coverage import _populate_shard, _production_fixture, _raw_bytes

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture)
    store = Part1ShardStore(
        shard_root,
        shard_id="shard-000",
        study_id=fixture["manifest"]["study_id"],
        model_run_id=fixture["manifest"]["model_run_id"],
        model_run_manifest_hash=fixture["manifest"]["model_run_manifest_hash"],
    )
    inspection = store.inspect()
    report = {
        "validation_report_id": "c" * 64,
        "paper_analysis_ready": True,
        "summary": {
            "study_manifest_hash": fixture["bundle"].study_manifest[
                "study_manifest_hash"
            ],
            "question_manifest_hash": fixture["manifest"]["question_manifest_hash"],
            "observed_git_commit": fixture["manifest"]["final_production_git_commit"],
            "clean_tracked_worktree": True,
        },
        "source_files": [],
    }
    report_bytes = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    report_path = fixture["repository"] / "coverage_report.json"
    report_path.write_bytes(report_bytes)
    inputs = MergeInputs(
        repository_root=fixture["repository"],
        model_manifest=fixture["manifest"],
        coverage_report=report,
        coverage_report_path=report_path,
        coverage_report_bytes=report_bytes,
        source_files=(),
        natural_records=inspection.natural_results,
        checkpoint_records=inspection.checkpoint_results,
        audit_events=inspection.audit_events,
    )
    return fixture, shard_root, inputs, _raw_bytes(shard_root)


def _disable_snapshot_recheck(monkeypatch: pytest.MonkeyPatch) -> None:
    import part1_merge

    monkeypatch.setattr(part1_merge, "revalidate_merge_inputs", lambda _inputs: None)


def test_coverage_gate_allows_only_structural_complete_reports_and_terminal_diagnostics() -> None:
    from part1_merge import require_mergeable_coverage
    from test_validate_part1_results import _coverage_report, _rehash

    ready = _coverage_report()
    require_mergeable_coverage(ready)

    diagnostic = copy.deepcopy(ready)
    diagnostic["paper_analysis_ready"] = False
    diagnostic["summary"]["natural_partition"]["complete"] -= 1
    diagnostic["summary"]["natural_partition"]["terminal_infrastructure_failure"] += 1
    diagnostic["summary"]["checkpoint_partition"]["complete"] -= 11
    diagnostic["summary"]["checkpoint_partition"]["ineligible"] += 11
    diagnostic["summary"]["outcome_counts"].update(
        natural_execution_complete=4999,
        natural_terminal_infrastructure_failure=1,
        checkpoint_execution_complete=54989,
        checkpoint_ineligible=11,
    )
    diagnostic["summary"]["natural_model_output_matrix"]["rows"][0]["count"] -= 1
    diagnostic["summary"]["checkpoint_model_output_matrix"]["rows"][0]["count"] -= 11
    diagnostic["summary"]["observed"]["checkpoint_physical_records"] -= 11
    diagnostic["checks"][2]["outcome"] = "warning"
    diagnostic["warning_count"] = 1
    _rehash(diagnostic)
    require_mergeable_coverage(diagnostic)

    for field in ("structurally_valid", "coverage_complete"):
        invalid = copy.deepcopy(ready)
        invalid[field] = False
        with pytest.raises(ValueError, match=field):
            require_mergeable_coverage(invalid)

    incomplete = _coverage_report(ready=False)
    with pytest.raises(ValueError):
        require_mergeable_coverage(incomplete)


@pytest.mark.parametrize("shard_count", [20, 200, 499])
def test_coverage_gate_rejects_historical_and_partial_source_inventories(
    shard_count: int,
) -> None:
    from part1_merge import require_mergeable_coverage
    from test_validate_part1_results import _coverage_report

    with pytest.raises(ValueError, match="500|source inventory|shard"):
        require_mergeable_coverage(_coverage_report(shard_count=shard_count))


def test_snapshot_restatement_detects_changed_appeared_and_disappeared_sources(
    tmp_path: Path,
) -> None:
    from part1_merge import require_source_snapshot

    regular = tmp_path / "regular"
    regular.write_bytes(b"original")
    absent = tmp_path / "absent"
    entries = [
        {
            "relative_path": "regular",
            "state": "regular_file",
            "sha256": hashlib.sha256(b"original").hexdigest(),
            "byte_size": 8,
        },
        {
            "relative_path": "absent",
            "state": "absent",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "byte_size": 0,
        },
    ]
    require_source_snapshot(tmp_path, entries)
    regular.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        require_source_snapshot(tmp_path, entries)
    regular.unlink()
    with pytest.raises(ValueError, match="absent"):
        require_source_snapshot(tmp_path, entries)
    regular.write_bytes(b"original")
    absent.write_bytes(b"appeared")
    with pytest.raises(ValueError, match="present"):
        require_source_snapshot(tmp_path, entries)


def test_snapshot_restatement_rejects_symlinked_path_ancestors(tmp_path: Path) -> None:
    from part1_merge import require_source_snapshot

    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "source").write_bytes(b"bytes")
    (tmp_path / "linked").symlink_to(actual, target_is_directory=True)
    entries = [
        {
            "relative_path": "linked/source",
            "state": "regular_file",
            "sha256": hashlib.sha256(b"bytes").hexdigest(),
            "byte_size": 5,
        }
    ]
    with pytest.raises(ValueError, match="symlink"):
        require_source_snapshot(tmp_path, entries)


def test_identical_rerun_is_noop_and_divergent_partial_extra_symlink_targets_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_merge import publish_merge

    fixture, shard_root, inputs, raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]
    assert publish_merge(inputs) == target
    first_bytes = {path.name: path.read_bytes() for path in target.iterdir()}
    assert publish_merge(inputs) == target
    assert {path.name: path.read_bytes() for path in target.iterdir()} == first_bytes

    for mutation in ("divergent", "partial", "extra"):
        case_root = tmp_path / mutation
        case_fixture, case_shard, case_inputs, case_raw = _prepared(case_root)
        case_target = (
            case_fixture["repository"]
            / case_fixture["manifest"]["output_paths"]["merged"]
        )
        publish_merge(case_inputs)
        if mutation == "divergent":
            (case_target / "natural_results.parquet").write_bytes(b"changed")
        elif mutation == "partial":
            (case_target / "audit_events.parquet").unlink()
        else:
            (case_target / "extra.bin").write_bytes(b"extra")
        with pytest.raises(ValueError):
            publish_merge(case_inputs)
        assert __import__("test_part1_coverage")._raw_bytes(case_shard) == case_raw

    symlink_root = tmp_path / "symlink"
    symlink_fixture, symlink_shard, symlink_inputs, symlink_raw = _prepared(symlink_root)
    symlink_target = (
        symlink_fixture["repository"]
        / symlink_fixture["manifest"]["output_paths"]["merged"]
    )
    symlink_target.parent.mkdir(parents=True, exist_ok=True)
    symlink_target.symlink_to(symlink_shard, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        publish_merge(symlink_inputs)
    assert __import__("test_part1_coverage")._raw_bytes(symlink_shard) == symlink_raw
    assert __import__("test_part1_coverage")._raw_bytes(shard_root) == raw_before


@pytest.mark.parametrize(
    "boundary",
    ["stage_created", "table_writes_complete", "manifest_written", "reload_complete", "before_rename"],
)
def test_every_staging_failure_leaves_no_final_output_and_raw_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    from part1_merge import publish_merge

    fixture, shard_root, inputs, raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]

    def fail(observed: str) -> None:
        if observed == boundary:
            raise RuntimeError(boundary)

    with pytest.raises(RuntimeError, match=boundary):
        publish_merge(inputs, fault_hook=fail)
    assert not os.path.lexists(target)
    assert not list(target.parent.glob(f".{target.name}.stage-*"))
    assert __import__("test_part1_coverage")._raw_bytes(shard_root) == raw_before


def test_racing_identical_winner_is_validated_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]
    real_rename = part1_merge.os.rename

    def race(source: Path, destination: Path) -> None:
        winner = Path(str(source) + "-winner")
        __import__("shutil").copytree(source, winner)
        real_rename(winner, destination)
        raise FileExistsError("racing writer won")

    monkeypatch.setattr(part1_merge.os, "rename", race)
    assert part1_merge.publish_merge(inputs) == target
    assert target.is_dir()


def test_cli_publishes_structural_diagnostic_and_reports_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import merge_part1_results
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    inputs.coverage_report["paper_analysis_ready"] = False
    monkeypatch.setattr(
        merge_part1_results, "load_validated_merge_inputs", lambda **_kwargs: inputs
    )
    _disable_snapshot_recheck(monkeypatch)
    code = merge_part1_results.main(
        [
            "--repository-root",
            str(fixture["repository"]),
            "--model-run-manifest",
            str(fixture["manifest_path"]),
            "--coverage-report",
            str(inputs.coverage_report_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "merged_diagnostic"
    assert payload["paper_analysis_ready"] is False


def test_merge_job_wrapper_stays_cpu_only_and_invokes_merge_cli() -> None:
    text = Path("jobs/part1_merge.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-task" not in text
    assert "scripts/merge_part1_results.py" in text
    assert "--model-run-manifest" in text
