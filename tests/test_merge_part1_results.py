"""Coverage gate, atomic publication, and CLI tests for the Part 1 merge."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest


def _complete_source_files(model_run_id: str) -> list[dict]:
    def regular(relative_path: str, kind: str, shard_id: str | None) -> dict:
        payload = relative_path.encode()
        return {
            "relative_path": relative_path,
            "shard_id": shard_id,
            "kind": kind,
            "state": "regular_file",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        }

    global_paths = {
        "questions": "manifests/part1/questions.jsonl",
        "question_manifest": "manifests/part1/questions.manifest.json",
        "study_manifest": "manifests/part1/study_manifest.json",
        "model_run_manifest": f"results/part1/{model_run_id}/model_run_manifest.json",
        "dependency_lock": "uv.lock",
    }
    entries = [
        regular(relative_path, kind, None)
        for kind, relative_path in global_paths.items()
    ]
    core_names = {
        "shard_provenance": ".shard-provenance.json",
        "natural_results": "natural_results.jsonl",
        "checkpoint_results": "checkpoint_results.jsonl",
        "audit_events": "audit_events.jsonl",
        "finalization_marker": ".finalized",
    }
    for shard_index in range(500):
        shard_id = f"shard-{shard_index:03d}"
        prefix = f"results/part1/{model_run_id}/raw_shards/{shard_id}"
        for kind, filename in core_names.items():
            relative_path = f"{prefix}/{filename}"
            if kind == "checkpoint_results":
                entries.append(
                    {
                        "relative_path": relative_path,
                        "shard_id": shard_id,
                        "kind": kind,
                        "state": "absent",
                        "sha256": hashlib.sha256(b"").hexdigest(),
                        "byte_size": 0,
                    }
                )
            else:
                entries.append(regular(relative_path, kind, shard_id))
    return sorted(entries, key=lambda item: item["relative_path"])


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
    source_files = _complete_source_files(fixture["manifest"]["model_run_id"])
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
        "source_files": source_files,
    }
    report_bytes = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    report_path = (
        fixture["repository"]
        / fixture["manifest"]["output_paths"]["validation"]
        / "coverage_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)
    inputs = MergeInputs(
        repository_root=fixture["repository"],
        model_manifest=fixture["manifest"],
        coverage_report=report,
        coverage_report_path=report_path,
        coverage_report_bytes=report_bytes,
        source_files=tuple(source_files),
        natural_records=inspection.natural_results,
        checkpoint_records=inspection.checkpoint_results,
        audit_events=inspection.audit_events,
    )
    return fixture, shard_root, inputs, _raw_bytes(shard_root)


def _disable_snapshot_recheck(monkeypatch: pytest.MonkeyPatch) -> None:
    import part1_merge

    monkeypatch.setattr(part1_merge, "revalidate_merge_inputs", lambda _inputs: None)
    monkeypatch.setattr(part1_merge, "EXPECTED_NATURAL_COUNT", 1)
    monkeypatch.setattr(part1_merge, "CHECKPOINTS_PER_NATURAL", 1, raising=False)


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


def test_descriptor_bound_read_cannot_parse_swapped_path_bytes(tmp_path: Path) -> None:
    from part1_merge import _read_inventory_entry_at

    source = tmp_path / "source.jsonl"
    original = b'{"source":"coverage-bound"}\n'
    transient = b'{"source":"transient-attacker"}\n'
    source.write_bytes(original)
    entry = {
        "relative_path": "source.jsonl",
        "shard_id": "shard-000",
        "kind": "natural_results",
        "state": "regular_file",
        "sha256": hashlib.sha256(original).hexdigest(),
        "byte_size": len(original),
    }
    root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )

    def swap_after_open() -> None:
        source.rename(tmp_path / "held-original.jsonl")
        source.write_bytes(transient)

    try:
        observed = _read_inventory_entry_at(
            root_descriptor, entry, after_open=swap_after_open
        )
    finally:
        os.close(root_descriptor)
    assert observed == original
    assert source.read_bytes() == transient


def test_revalidation_requires_exact_coverage_source_inventory(
    tmp_path: Path,
) -> None:
    from part1_merge import revalidate_merge_inputs

    _fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    typed_lookalike = [dict(item) for item in inputs.coverage_report["source_files"]]
    typed_lookalike[0]["byte_size"] = float(typed_lookalike[0]["byte_size"])
    inputs.source_files = tuple(typed_lookalike)
    with pytest.raises(ValueError, match="exactly equal"):
        revalidate_merge_inputs(inputs)

    inputs.source_files = (
        {
            "relative_path": "unexpected",
            "shard_id": None,
            "kind": "dependency_lock",
            "state": "absent",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "byte_size": 0,
        },
    )
    with pytest.raises(ValueError, match="exactly equal"):
        revalidate_merge_inputs(inputs)


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


def test_existing_merge_reload_uses_opened_directory_and_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_merge import (
        _validate_merge_directory_descriptor,
        publish_merge,
        validate_merge_directory,
    )

    _fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = publish_merge(inputs)
    rows = {
        "natural_results": inputs.natural_records,
        "checkpoint_results": inputs.checkpoint_records,
        "audit_events": inputs.audit_events,
    }
    manifest = validate_merge_directory(target, expected_rows=rows)
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    held = target.with_name("held-merge")
    target.rename(held)
    target.mkdir()
    (target / "attacker").write_bytes(b"transient")
    try:
        assert _validate_merge_directory_descriptor(
            descriptor, expected_manifest=manifest, expected_rows=rows
        ) == manifest
    finally:
        os.close(descriptor)


def test_expected_rows_use_json_typed_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_merge import publish_merge, validate_merge_directory

    _fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    generated_count = len(inputs.natural_records[0]["generated_token_ids"])
    inputs.natural_records[0]["generated_token_count"] = float(generated_count)
    target = publish_merge(inputs)
    expected_rows = {
        "natural_results": copy.deepcopy(inputs.natural_records),
        "checkpoint_results": copy.deepcopy(inputs.checkpoint_records),
        "audit_events": copy.deepcopy(inputs.audit_events),
    }
    expected_rows["natural_results"][0]["generated_token_count"] = generated_count
    with pytest.raises(ValueError, match="not lossless"):
        validate_merge_directory(target, expected_rows=expected_rows)


def test_cleanup_never_deletes_a_substituted_stage_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]
    held_stage: list[Path] = []
    replacement: list[Path] = []

    def substitute(boundary: str) -> None:
        if boundary == "table_writes_complete":
            raise RuntimeError("injected before cleanup")
        if boundary != "before_stage_cleanup_identity_move":
            return
        stage = next(target.parent.glob(".merged.stage-*"))
        held = stage.with_name("held-owned-stage")
        stage.rename(held)
        stage.mkdir()
        (stage / "replacement-marker").write_bytes(b"must survive")
        held_stage.append(held)
        replacement.append(stage)

    with pytest.raises(RuntimeError):
        part1_merge.publish_merge(inputs, fault_hook=substitute)
    assert held_stage[0].is_dir()
    assert (replacement[0] / "replacement-marker").read_bytes() == b"must survive"


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


@pytest.mark.parametrize("racing_target", ["empty", "partial", "symlink"])
def test_exclusive_race_never_replaces_incompatible_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, racing_target: str
) -> None:
    import part1_merge

    fixture, shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]

    def publish_racer(boundary: str) -> None:
        if boundary != "before_exclusive_rename":
            return
        if racing_target == "symlink":
            target.symlink_to(shard_root, target_is_directory=True)
            return
        target.mkdir()
        if racing_target == "partial":
            (target / "winner-marker").write_bytes(b"partial")

    with pytest.raises(ValueError):
        part1_merge.publish_merge(inputs, fault_hook=publish_racer)
    if racing_target == "symlink":
        assert target.is_symlink()
        assert target.resolve() == shard_root.resolve()
    else:
        assert target.is_dir()
        assert set(path.name for path in target.iterdir()) == (
            {"winner-marker"} if racing_target == "partial" else set()
        )


def test_racing_identical_winner_is_validated_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]

    def publish_racer(boundary: str) -> None:
        if boundary != "before_exclusive_rename":
            return
        stage = next(target.parent.glob(f".{target.name}.stage-*"))
        __import__("shutil").copytree(stage, target)

    assert part1_merge.publish_merge(inputs, fault_hook=publish_racer) == target
    assert target.is_dir()


def test_post_rename_parent_fsync_failure_rolls_back_without_reporting_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]
    real_fsync = part1_merge._fsync_directory_descriptor
    calls = 0

    def fail_first_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected post-rename parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        part1_merge, "_fsync_directory_descriptor", fail_first_parent_fsync
    )
    with pytest.raises(part1_merge.PublicationDurabilityError, match="rolled back"):
        part1_merge.publish_merge(inputs)
    assert not os.path.lexists(target)
    assert not list(target.parent.glob(f".{target.name}.stage-*"))


def test_publication_parent_descriptor_closes_when_stage_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]
    real_open = part1_merge.os.open
    real_close = part1_merge.os.close
    parent_descriptors: list[int] = []
    closed_descriptors: list[int] = []

    def observe_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == target.parent:
            parent_descriptors.append(descriptor)
        return descriptor

    def observe_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(part1_merge.os, "open", observe_open)
    monkeypatch.setattr(part1_merge.os, "close", observe_close)
    monkeypatch.setattr(
        part1_merge.tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("injected stage failure")),
    )
    with pytest.raises(OSError, match="injected stage failure"):
        part1_merge.publish_merge(inputs)
    assert len(parent_descriptors) == 1
    assert parent_descriptors[0] in closed_descriptors


def test_stage_is_removed_when_opening_its_descriptor_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target = fixture["repository"] / fixture["manifest"]["output_paths"]["merged"]
    real_open = part1_merge.os.open

    def fail_stage_open(path, flags, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".merged.stage-"):
            raise OSError("injected stage descriptor failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(part1_merge.os, "open", fail_stage_open)
    with pytest.raises(OSError, match="injected stage descriptor failure"):
        part1_merge.publish_merge(inputs)
    assert not list(target.parent.glob(".merged.stage-*"))


def test_stage_identity_stat_failure_closes_parent_and_reports_indeterminate_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    _fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    real_stat = part1_merge.os.stat
    real_open = part1_merge.os.open
    real_close = part1_merge.os.close
    parent_descriptors: list[int] = []
    closed_descriptors: list[int] = []

    def observe_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == (
            inputs.repository_root / inputs.model_manifest["output_paths"]["merged"]
        ).parent:
            parent_descriptors.append(descriptor)
        return descriptor

    def fail_initial_stage_stat(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".merged.stage-"):
            raise OSError("injected stage identity stat failure")
        return real_stat(path, *args, **kwargs)

    def observe_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(part1_merge.os, "open", observe_open)
    monkeypatch.setattr(part1_merge.os, "stat", fail_initial_stage_stat)
    monkeypatch.setattr(part1_merge.os, "close", observe_close)
    with pytest.raises(
        part1_merge.PublicationStateIndeterminateError,
        match="stage=.*identity",
    ):
        part1_merge.publish_merge(inputs)
    assert parent_descriptors[0] in closed_descriptors


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


def test_cli_revalidates_manifest_after_post_publication_target_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import merge_part1_results
    import part1_merge

    fixture, _shard_root, inputs, _raw_before = _prepared(tmp_path)
    monkeypatch.setattr(
        merge_part1_results, "load_validated_merge_inputs", lambda **_kwargs: inputs
    )
    _disable_snapshot_recheck(monkeypatch)
    real_publish = part1_merge.publish_merge

    published_manifest: dict = {}

    def publish_then_swap(publish_inputs, **kwargs):
        publication = real_publish(publish_inputs, **kwargs)
        assert isinstance(publication, tuple)
        target, manifest = publication
        published_manifest.update(manifest)
        target.rename(target.with_name("held-valid-merge"))
        target.mkdir()
        (target / "merge_manifest.json").write_text(
            json.dumps(
                {"merge_id": "a" * 64, "merge_manifest_hash": "b" * 64}
            ),
            encoding="utf-8",
        )
        return target, manifest

    monkeypatch.setattr(merge_part1_results, "publish_merge", publish_then_swap)
    code = merge_part1_results.main(
        [
            "--repository-root",
            str(fixture["repository"]),
            "--model-run-manifest",
            str(fixture["manifest_path"]),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["merge_id"] == published_manifest["merge_id"]
    assert payload["merge_id"] != "a" * 64


def test_merge_job_wrapper_stays_cpu_only_and_invokes_merge_cli() -> None:
    text = Path("jobs/part1_merge.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-task" not in text
    assert "scripts/merge_part1_results.py" in text
    assert "--model-run-manifest" in text
