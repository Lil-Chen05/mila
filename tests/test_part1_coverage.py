"""Synthetic, login-safe tests for complete Part 1 coverage accounting."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from part1_contract import validate_instance

def _question_ids() -> tuple[str, ...]:
    return tuple(f"{index:064x}" for index in range(500))


def _natural_record(
    *,
    outcome: str = "complete",
    stop_reason: str = "eos",
    reasoning_status: str = "closed",
    answer_parse_status: str = "parsed",
    confidence_parse_status: str = "parsed",
) -> dict:
    return {
        "natural_execution_outcome": outcome,
        "stop_reason": "error" if outcome != "complete" else stop_reason,
        "reasoning_status": (
            "malformed" if outcome != "complete" else reasoning_status
        ),
        "answer_parse_status": (
            "missing" if outcome != "complete" else answer_parse_status
        ),
        "confidence_parse_status": (
            "missing" if outcome != "complete" else confidence_parse_status
        ),
    }


def _checkpoint_record(
    *,
    outcome: str = "complete",
    checkpoint_model_output_status: str = "valid",
    answer_parse_status: str = "parsed",
    confidence_parse_status: str = "parsed",
    answer_token_status: str = "located",
    entropy_status: str = "computed",
) -> dict:
    if outcome != "complete":
        checkpoint_model_output_status = "invalid"
        answer_parse_status = "missing"
        confidence_parse_status = "missing"
        answer_token_status = "unsupported"
        entropy_status = "unavailable"
    return {
        "checkpoint_execution_outcome": outcome,
        "checkpoint_model_output_status": checkpoint_model_output_status,
        "answer_parse_status": answer_parse_status,
        "confidence_parse_status": confidence_parse_status,
        "answer_token_status": answer_token_status,
        "entropy_status": entropy_status,
    }


def _nominal_observations():
    from part1_coverage import RecordObservation

    natural = {}
    checkpoints = {}
    for question_id in _question_ids():
        for run_id in range(10):
            natural[(question_id, run_id)] = (
                RecordObservation.valid(_natural_record(), source="natural.jsonl:1"),
            )
            for checkpoint_index in range(11):
                checkpoints[(question_id, run_id, f"cp-{checkpoint_index:02d}")] = (
                    RecordObservation.valid(
                        _checkpoint_record(), source="checkpoint.jsonl:1"
                    ),
                )
    return natural, checkpoints


def test_nominal_partitions_are_exact_exhaustive_and_paper_ready() -> None:
    from part1_coverage import classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )

    assert report["natural_partition"] == {
        "complete": 5_000,
        "terminal_infrastructure_failure": 0,
        "retryable_incomplete": 0,
        "missing": 0,
        "duplicate": 0,
        "schema_incompatible": 0,
        "manifest_incompatible": 0,
    }
    assert report["checkpoint_partition"] == {
        "complete": 55_000,
        "terminal_infrastructure_failure": 0,
        "retryable_incomplete": 0,
        "ineligible": 0,
        "missing": 0,
        "duplicate": 0,
    }
    assert sum(report["natural_partition"].values()) == 5_000
    assert sum(report["checkpoint_partition"].values()) == 55_000
    assert report["coverage_complete"] is True
    assert report["paper_analysis_ready"] is True


def test_natural_failure_makes_exactly_eleven_children_ineligible() -> None:
    from part1_coverage import RecordObservation, classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    key = (_question_ids()[0], 0)
    natural[key] = (
        RecordObservation.valid(
            _natural_record(outcome="terminal_infrastructure_failure"),
            source="natural.jsonl:1",
        ),
    )
    for checkpoint_index in range(11):
        checkpoints.pop((*key, f"cp-{checkpoint_index:02d}"))

    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )

    assert report["natural_partition"]["terminal_infrastructure_failure"] == 1
    assert report["checkpoint_partition"]["ineligible"] == 11
    assert report["checkpoint_partition"]["missing"] == 0
    assert report["coverage_complete"] is True
    assert report["paper_analysis_ready"] is False


def test_abnormal_success_is_complete_and_checkpoint_eligible() -> None:
    from part1_coverage import RecordObservation, classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    key = (_question_ids()[0], 0)
    natural[key] = (
        RecordObservation.valid(
            _natural_record(
                stop_reason="max_new_tokens",
                reasoning_status="missing_close",
                answer_parse_status="missing",
                confidence_parse_status="malformed",
            ),
            source="natural.jsonl:1",
        ),
    )

    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )

    assert report["natural_partition"]["complete"] == 5_000
    assert report["checkpoint_partition"]["complete"] == 55_000
    row = next(
        row
        for row in report["natural_model_output_matrix"]["rows"]
        if row["stop_reason"] == "max_new_tokens"
        and row["reasoning_status"] == "missing_close"
        and row["answer_parse_status"] == "missing"
        and row["confidence_parse_status"] == "malformed"
    )
    assert row["count"] == 1


def test_partition_precedence_lifecycle_and_hierarchy_are_deterministic() -> None:
    from part1_coverage import RecordObservation, classify_logical_coverage

    question_ids = _question_ids()
    natural, checkpoints = _nominal_observations()
    duplicate_key = (question_ids[0], 0)
    schema_key = (question_ids[0], 1)
    manifest_key = (question_ids[0], 2)
    retry_key = (question_ids[0], 3)
    missing_key = (question_ids[0], 4)
    natural[duplicate_key] = (
        RecordObservation.valid(_natural_record(), source="n:1"),
        RecordObservation.schema_incompatible("bad schema", source="n:2"),
    )
    natural[schema_key] = (
        RecordObservation.schema_incompatible("bad schema", source="n:3"),
    )
    natural[manifest_key] = (
        RecordObservation.manifest_incompatible("wrong run", source="n:4"),
    )
    natural.pop(retry_key)
    natural.pop(missing_key)
    natural_lifecycle = {retry_key}

    # Physical checkpoint data beneath invalid/missing parents is never accepted.
    for parent_key in (duplicate_key, schema_key, manifest_key, retry_key, missing_key):
        for checkpoint_index in range(11):
            checkpoints.pop((*parent_key, f"cp-{checkpoint_index:02d}"))
        checkpoints[(*parent_key, "cp-00")] = (
            RecordObservation.valid(_checkpoint_record(), source="c:1"),
        )

    report = classify_logical_coverage(
        question_ids=question_ids,
        natural_observations=natural,
        checkpoint_observations=checkpoints,
        natural_lifecycle_keys=natural_lifecycle,
    )

    assert report["natural_partition"]["duplicate"] == 1
    assert report["natural_partition"]["schema_incompatible"] == 1
    assert report["natural_partition"]["manifest_incompatible"] == 1
    assert report["natural_partition"]["retryable_incomplete"] == 1
    assert report["natural_partition"]["missing"] == 1
    assert report["checkpoint_partition"]["complete"] == 54_945
    assert report["checkpoint_partition"]["missing"] == 55
    assert report["unexpected_physical_record_count"] == 5
    assert len(report["structural_errors"]) >= 5
    assert report["coverage_complete"] is False
    assert report["structurally_valid"] is False


def test_checkpoint_failure_retryable_duplicate_and_missing_are_separate() -> None:
    from part1_coverage import RecordObservation, classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    parent = (_question_ids()[0], 0)
    failed = (*parent, "cp-00")
    retry = (*parent, "cp-01")
    missing = (*parent, "cp-02")
    duplicate = (*parent, "cp-03")
    checkpoints[failed] = (
        RecordObservation.valid(
            _checkpoint_record(outcome="terminal_infrastructure_failure"), source="c:1"
        ),
    )
    checkpoints.pop(retry)
    checkpoints.pop(missing)
    checkpoints[duplicate] = (
        RecordObservation.valid(_checkpoint_record(), source="c:2"),
        RecordObservation.valid(_checkpoint_record(), source="c:3"),
    )

    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
        checkpoint_lifecycle_keys={retry},
    )

    assert report["checkpoint_partition"]["terminal_infrastructure_failure"] == 1
    assert report["checkpoint_partition"]["retryable_incomplete"] == 1
    assert report["checkpoint_partition"]["missing"] == 1
    assert report["checkpoint_partition"]["duplicate"] == 1
    assert report["coverage_complete"] is False
    assert report["paper_analysis_ready"] is False


def test_model_output_matrices_are_full_cartesian_and_include_zeros() -> None:
    from part1_coverage import (
        CHECKPOINT_ATTRIBUTE_VALUES,
        NATURAL_ATTRIBUTE_VALUES,
        RecordObservation,
        classify_logical_coverage,
    )

    natural, checkpoints = _nominal_observations()
    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )

    natural_matrix = report["natural_model_output_matrix"]
    checkpoint_matrix = report["checkpoint_model_output_matrix"]
    expected_natural_rows = 1
    for values in NATURAL_ATTRIBUTE_VALUES.values():
        expected_natural_rows *= len(values)
    expected_checkpoint_rows = 1
    for values in CHECKPOINT_ATTRIBUTE_VALUES.values():
        expected_checkpoint_rows *= len(values)
    assert len(natural_matrix["rows"]) == expected_natural_rows
    assert len(checkpoint_matrix["rows"]) == expected_checkpoint_rows
    assert sum(row["count"] for row in natural_matrix["rows"]) == 5_000
    assert sum(row["count"] for row in checkpoint_matrix["rows"]) == 55_000
    assert any(row["count"] == 0 for row in natural_matrix["rows"])
    assert any(row["count"] == 0 for row in checkpoint_matrix["rows"])


def test_readiness_booleans_are_independent() -> None:
    from part1_coverage import RecordObservation, classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    terminal_parent = (_question_ids()[0], 0)
    natural[terminal_parent] = (
        RecordObservation.valid(
            _natural_record(outcome="terminal_infrastructure_failure"), source="n:1"
        ),
    )
    for index in range(11):
        checkpoints.pop((*terminal_parent, f"cp-{index:02d}"))
    terminal = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )
    assert (
        terminal["structurally_valid"],
        terminal["coverage_complete"],
        terminal["paper_analysis_ready"],
    ) == (True, True, False)

    structurally_bad = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations={**natural, ("a" * 64, 0): natural[terminal_parent]},
        checkpoint_observations=checkpoints,
    )
    assert structurally_bad["structurally_valid"] is False


def test_unexpected_physical_data_invalidates_structure_not_logical_coverage() -> None:
    from part1_coverage import classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
        unexpected_physical_record_count=1,
    )

    assert report["coverage_complete"] is True
    assert report["structurally_valid"] is False
    assert report["paper_analysis_ready"] is False


def test_checkpoint_terminal_failure_is_covered_but_not_paper_ready() -> None:
    from part1_coverage import RecordObservation, classify_logical_coverage

    natural, checkpoints = _nominal_observations()
    key = (_question_ids()[0], 0, "cp-00")
    checkpoints[key] = (
        RecordObservation.valid(
            _checkpoint_record(outcome="terminal_infrastructure_failure"),
            source="checkpoint.jsonl:1",
        ),
    )
    report = classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )

    assert report["checkpoint_partition"]["terminal_infrastructure_failure"] == 1
    assert (
        report["structurally_valid"],
        report["coverage_complete"],
        report["paper_analysis_ready"],
    ) == (True, True, False)


def test_matrix_enum_values_and_cartesian_order_are_independently_pinned() -> None:
    import itertools

    from part1_coverage import CHECKPOINT_ATTRIBUTE_VALUES, NATURAL_ATTRIBUTE_VALUES

    assert NATURAL_ATTRIBUTE_VALUES == {
        "stop_reason": ("eos", "max_new_tokens", "stopping_criterion", "error", "other"),
        "reasoning_status": ("closed", "missing_close", "no_reasoning", "malformed"),
        "answer_parse_status": ("parsed", "missing", "malformed", "out_of_domain"),
        "confidence_parse_status": ("parsed", "missing", "malformed", "out_of_range"),
    }
    assert CHECKPOINT_ATTRIBUTE_VALUES == {
        "checkpoint_model_output_status": ("valid", "invalid"),
        "answer_parse_status": ("parsed", "missing", "malformed", "out_of_domain"),
        "confidence_parse_status": ("parsed", "missing", "malformed", "out_of_range"),
        "answer_token_status": ("located", "missing", "ambiguous", "unsupported"),
        "entropy_status": ("computed", "unavailable", "invalid"),
    }
    natural, checkpoints = _nominal_observations()
    report = __import__("part1_coverage").classify_logical_coverage(
        question_ids=_question_ids(),
        natural_observations=natural,
        checkpoint_observations=checkpoints,
    )
    for matrix_name, dimensions in (
        ("natural_model_output_matrix", NATURAL_ATTRIBUTE_VALUES),
        ("checkpoint_model_output_matrix", CHECKPOINT_ATTRIBUTE_VALUES),
    ):
        matrix = report[matrix_name]
        observed_order = [
            tuple(row[name] for name in dimensions) for row in matrix["rows"]
        ]
        assert observed_order == list(itertools.product(*dimensions.values()))


def _production_fixture(tmp_path: Path) -> dict:
    from test_run_part1_shard import production_fixture

    return production_fixture(tmp_path)


def _populate_shard(fixture: dict, *, terminal_naturals: bool = False) -> Path:
    from part1_checkpoints import build_checkpoint_probe_plans
    from part1_contract import derive_generation_seed
    from part1_generation import build_natural_infrastructure_failure_result
    from part1_store import Part1ShardStore
    from part1_store_fixtures import attempt_event
    from test_run_part1_shard import _fake_checkpoint, _fake_natural

    root = (
        fixture["repository"]
        / fixture["manifest"]["output_paths"]["raw_shards"]
        / "shard-000"
    )
    manifest = fixture["manifest"]
    question = fixture["bundle"].records[0]
    token_contract = fixture["preflight"]["token_contract"]
    store = Part1ShardStore(
        root,
        shard_id="shard-000",
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
        unsafe_for_tests=True,
    )
    store.initialize_provenance_header()
    for run_id in range(1):
        seed = derive_generation_seed(
            base_seed=manifest["base_generation_seed"],
            canonical_model_identity=manifest["canonical_model_identity"],
            question_id=question["question_id"],
            run_id=run_id,
        )
        if terminal_naturals:
            natural = build_natural_infrastructure_failure_result(
                identity={
                    "study_id": manifest["study_id"],
                    "model_run_id": manifest["model_run_id"],
                    "model_run_manifest_hash": manifest["model_run_manifest_hash"],
                    "question_manifest_hash": manifest["question_manifest_hash"],
                    "question_id": question["question_id"],
                    "sample_index": question["sample_index"],
                    "subject": question["subject"],
                    "gold_letter": question["gold_letter"],
                },
                run_id=run_id,
                generation_seed=seed,
                terminal_attempt_number=1,
                prompt_hash=manifest["prompt_hash"],
                failure_category="invalid_configuration",
                infrastructure_failure_reference=f"synthetic:natural:{run_id}",
                error_details={"category": "invalid_configuration", "synthetic": True},
            )
        else:
            natural = _fake_natural(
                model=object(),
                tokenizer=object(),
                question=question,
                run_id=run_id,
                seed=seed,
                attempt_number=1,
                model_manifest=manifest,
                token_contract=token_contract,
            )
        started = attempt_event(natural, "attempt_started", 0)
        completed = attempt_event(natural, "attempt_completed", 1)
        store.append_audit_event(started)
        store.commit_terminal_result(natural, completed)
        if natural["natural_execution_outcome"] != "complete":
            continue
        for plan in build_checkpoint_probe_plans(
            natural,
            inducer_token_ids=token_contract["inducer_token_ids"],
            inducer_version=manifest["inducer_version"],
        )[:2]:
            checkpoint = _fake_checkpoint(
                model=object(),
                tokenizer=object(),
                parent=natural,
                plan=plan,
                token_contract=token_contract,
                gold_letter=question["gold_letter"],
                attempt_number=1,
            )
            started = attempt_event(checkpoint, "attempt_started", 0)
            completed = attempt_event(checkpoint, "attempt_completed", 1)
            store.append_audit_event(started)
            store.commit_terminal_result(checkpoint, completed)
    store.finalize()
    return root


def _raw_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_real_finalized_store_scan_is_read_only_and_hashes_every_merge_source(
    tmp_path: Path,
) -> None:
    from part1_coverage import scan_production_shard

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture)
    (shard_root / ".writer.guard").write_bytes(b"")
    before = _raw_bytes(shard_root)
    scan = scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=shard_root,
        shard_index=0,
        question=fixture["bundle"].records[0],
        model_manifest=fixture["manifest"],
    )

    assert len(scan.natural_observations) == 1
    assert len(scan.checkpoint_observations) == 2
    assert scan.structural_errors == ()
    assert {item["kind"] for item in scan.source_files} == {
        "shard_provenance",
        "natural_results",
        "checkpoint_results",
        "audit_events",
        "finalization_marker",
        "runtime_guard",
    }
    assert all(item["state"] == "regular_file" for item in scan.source_files)
    assert _raw_bytes(shard_root) == before


def test_terminal_natural_store_has_absent_checkpoint_stream_and_is_valid(
    tmp_path: Path,
) -> None:
    from part1_coverage import scan_production_shard

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture, terminal_naturals=True)
    scan = scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=shard_root,
        shard_index=0,
        question=fixture["bundle"].records[0],
        model_manifest=fixture["manifest"],
    )

    assert len(scan.natural_observations) == 1
    assert not scan.checkpoint_observations
    assert scan.structural_errors == ()
    checkpoint_source = next(
        item for item in scan.source_files if item["kind"] == "checkpoint_results"
    )
    assert checkpoint_source == {
        "relative_path": (
            fixture["manifest"]["output_paths"]["raw_shards"]
            + "/shard-000/checkpoint_results.jsonl"
        ),
        "shard_id": "shard-000",
        "kind": "checkpoint_results",
        "state": "absent",
        "sha256": hashlib.sha256(b"").hexdigest(),
        "byte_size": 0,
    }


def test_build_report_is_schema_valid_for_safe_incomplete_production(
    tmp_path: Path,
) -> None:
    from part1_coverage import build_coverage_report

    fixture = _production_fixture(tmp_path)
    _populate_shard(fixture, terminal_naturals=True)
    report = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
        validation_started_at="2026-08-05T00:00:00Z",
        validation_completed_at="2026-08-05T00:00:01Z",
    )

    validate_instance("validation_report", report)
    assert report["summary"]["expected"] == {
        "questions": 500,
        "shards": 500,
        "natural_logical_keys": 5_000,
        "checkpoint_logical_keys": 55_000,
    }
    assert report["summary"]["observed"]["shards"] == 1
    assert report["summary"]["natural_partition"]["terminal_infrastructure_failure"] == 1
    assert report["summary"]["natural_partition"]["missing"] == 4_999
    assert report["summary"]["checkpoint_partition"]["ineligible"] == 11
    assert report["summary"]["checkpoint_partition"]["missing"] == 54_989
    assert report["structurally_valid"] is False
    assert report["coverage_complete"] is False
    assert report["paper_analysis_ready"] is False
    assert report["summary"]["clean_tracked_worktree"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate"),
        ("schema", "schema"),
        ("manifest", "manifest"),
        ("seed", "seed"),
        ("record_id", "identity"),
        ("lifecycle", "lifecycle|completion"),
        ("alias", "alias|hierarchy"),
        ("finalization", "finalization"),
    ],
)
def test_scanner_detects_record_lifecycle_hierarchy_and_marker_defects(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from part1_coverage import scan_production_shard

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture)
    natural_path = shard_root / "natural_results.jsonl"
    checkpoint_path = shard_root / "checkpoint_results.jsonl"
    audit_path = shard_root / "audit_events.jsonl"
    if mutation == "duplicate":
        first = natural_path.read_text(encoding="utf-8").splitlines()[0]
        with natural_path.open("a", encoding="utf-8") as handle:
            handle.write(first + "\n")
    elif mutation in {"schema", "manifest", "seed", "record_id"}:
        records = [json.loads(line) for line in natural_path.read_text(encoding="utf-8").splitlines()]
        if mutation == "schema":
            records[0]["schema_version"] = "9.9.9"
        elif mutation == "manifest":
            records[0]["model_run_manifest_hash"] = "9" * 64
        elif mutation == "seed":
            records[0]["generation_seed"] += 1
        else:
            records[0]["raw_record_id"] = "9" * 64
        natural_path.write_text(
            "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
    elif mutation == "lifecycle":
        events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        events = [
            event
            for event in events
            if not (
                event["event_type"] == "attempt_completed"
                and event["run_id"] == 0
                and event["checkpoint_id"] is None
            )
        ]
        audit_path.write_text(
            "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
            encoding="utf-8",
        )
    elif mutation == "alias":
        records = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines()]
        alias = next(record for record in records if record["is_alias"])
        alias["alias_metadata"]["owner_checkpoint_id"] = "cp-10"
        checkpoint_path.write_text(
            "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        marker = json.loads((shard_root / ".finalized").read_text(encoding="utf-8"))
        marker["finalized_at"] = marker["finalized_at"].replace("Z", "+00:00")
        (shard_root / ".finalized").write_text(json.dumps(marker), encoding="utf-8")

    scan = scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=shard_root,
        shard_index=0,
        question=fixture["bundle"].records[0],
        model_manifest=fixture["manifest"],
    )
    assert any(__import__("re").search(message, error) for error in scan.structural_errors)


def test_scanner_rejects_unexpected_files_and_symlinks(tmp_path: Path) -> None:
    from part1_coverage import scan_production_shard

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture)
    (shard_root / "unexpected.bin").write_bytes(b"unexpected")
    (shard_root / "linked-stream").symlink_to(shard_root / "natural_results.jsonl")
    scan = scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=shard_root,
        shard_index=0,
        question=fixture["bundle"].records[0],
        model_manifest=fixture["manifest"],
    )
    assert any("unexpected" in error for error in scan.structural_errors)
    assert any("symlink" in error for error in scan.structural_errors)


@pytest.mark.parametrize("historical_count", [20, 200])
def test_build_report_explicitly_rejects_historical_shard_layouts(
    tmp_path: Path, historical_count: int
) -> None:
    from part1_coverage import build_coverage_report

    fixture = _production_fixture(tmp_path)
    raw_root = fixture["repository"] / fixture["manifest"]["output_paths"]["raw_shards"]
    for index in range(historical_count):
        (raw_root / f"shard-{index:03d}").mkdir(parents=True)
    report = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    assert report["structurally_valid"] is False
    assert report["summary"]["historical_layout_detected"] == historical_count
    assert any(
        "historical" in check["details"].get("error", "")
        for check in report["checks"]
    )


def test_build_report_rejects_mixed_root_and_changed_source_bytes(tmp_path: Path) -> None:
    from part1_coverage import build_coverage_report

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture, terminal_naturals=True)
    first = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    header = json.loads((shard_root / ".shard-provenance.json").read_text(encoding="utf-8"))
    header["model_run_id"] = "9" * 64
    (shard_root / ".shard-provenance.json").write_text(json.dumps(header), encoding="utf-8")
    second = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    assert second["validation_report_id"] != first["validation_report_id"]
    assert second["structurally_valid"] is False
    assert any(
        "provenance" in error or "model-run" in error
        for error in second["summary"]["structural_errors"]
    )


def test_malformed_json_is_unassignable_and_mixed_hashes_are_manifest_incompatible(
    tmp_path: Path,
) -> None:
    from part1_coverage import scan_production_shard

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture, terminal_naturals=True)
    natural_path = shard_root / "natural_results.jsonl"
    natural_path.write_bytes(natural_path.read_bytes() + b"{malformed\n")
    scan = scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=shard_root,
        shard_index=0,
        question=fixture["bundle"].records[0],
        model_manifest=fixture["manifest"],
    )
    assert scan.unassignable_physical_record_count == 1
    assert any("malformed JSON" in error for error in scan.structural_errors)

    record = json.loads(natural_path.read_text(encoding="utf-8").splitlines()[0])
    from part1_coverage import _natural_compatibility

    for field in ("study_id", "model_run_id", "model_run_manifest_hash", "question_manifest_hash"):
        changed = copy.deepcopy(record)
        changed[field] = "9" * 64
        errors = _natural_compatibility(
            changed,
            question=fixture["bundle"].records[0],
            model_manifest=fixture["manifest"],
        )
        assert any(field in error for error in errors)


def test_raw_root_symlink_and_dirty_tracked_worktree_fail_their_independent_gates(
    tmp_path: Path,
) -> None:
    from part1_coverage import build_coverage_report

    fixture = _production_fixture(tmp_path)
    raw_root = fixture["repository"] / fixture["manifest"]["output_paths"]["raw_shards"]
    outside = tmp_path / "outside"
    outside.mkdir()
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    raw_root.symlink_to(outside, target_is_directory=True)
    symlink_report = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    assert symlink_report["structurally_valid"] is False
    assert any(
        "symlink" in error for error in symlink_report["summary"]["structural_errors"]
    )

    raw_root.unlink()
    (fixture["repository"] / ".gitignore").write_text(
        "results/part1/\nresults/part1-smoke/\n# dirty\n", encoding="utf-8"
    )
    dirty_report = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    assert dirty_report["summary"]["clean_tracked_worktree"] is False
    assert dirty_report["structurally_valid"] is False


def test_shard_scan_detects_source_mutation_during_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_coverage

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture, terminal_naturals=True)
    natural_path = shard_root / "natural_results.jsonl"
    original_validate = part1_coverage.Part1ShardStore.validate_shard
    mutated = False

    def mutate_after_store_validation(store, *args, **kwargs):
        nonlocal mutated
        result = original_validate(store, *args, **kwargs)
        if not mutated:
            mutated = True
            natural_path.write_bytes(natural_path.read_bytes() + b"{mid-scan-mutation\n")
        return result

    monkeypatch.setattr(
        part1_coverage.Part1ShardStore,
        "validate_shard",
        mutate_after_store_validation,
    )
    scan = part1_coverage.scan_production_shard(
        repository_root=fixture["repository"],
        shard_root=shard_root,
        shard_index=0,
        question=fixture["bundle"].records[0],
        model_manifest=fixture["manifest"],
    )

    assert any(
        "source bytes changed during validation" in error
        for error in scan.structural_errors
    )


def test_build_detects_source_mutation_after_shard_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_coverage

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture, terminal_naturals=True)
    original_scan = part1_coverage.scan_production_shard
    mutated = False

    def mutate_after_scan(**kwargs):
        nonlocal mutated
        scan = original_scan(**kwargs)
        if not mutated:
            mutated = True
            natural_path = shard_root / "natural_results.jsonl"
            natural_path.write_bytes(natural_path.read_bytes() + b"{late-mutation\n")
        return scan

    monkeypatch.setattr(part1_coverage, "scan_production_shard", mutate_after_scan)
    report = part1_coverage.build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )

    assert report["structurally_valid"] is False
    assert any(
        "changed after inventory" in error or "changed during validation" in error
        for error in report["summary"]["structural_errors"]
    )


def test_publication_revalidates_all_inventoried_sources(
    tmp_path: Path,
) -> None:
    from part1_coverage import build_coverage_report, publish_coverage_report

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture, terminal_naturals=True)
    report = build_coverage_report(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["manifest_path"],
    )
    natural_path = shard_root / "natural_results.jsonl"
    natural_path.write_bytes(natural_path.read_bytes() + b"{post-build-mutation\n")
    target = fixture["repository"] / "coverage_report.json"

    with pytest.raises(ValueError, match="snapshot|changed after inventory"):
        publish_coverage_report(
            report,
            target,
            repository_root=fixture["repository"],
        )
    assert not target.exists()
