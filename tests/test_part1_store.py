"""Synthetic, login-safe tests for Part 1 append-only storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from part1_store import (
    DuplicateTerminalResultError,
    FinalizedShardError,
    InjectedCrash,
    MalformedMiddleError,
    Part1ShardStore,
    Part1StoreError,
    StreamTailError,
)
from part1_store_fixtures import (
    MODEL_RUN_ID,
    MODEL_RUN_MANIFEST_HASH,
    QUESTION_ID,
    SHARD_ID,
    STUDY_ID,
    attempt_event,
    checkpoint_result,
    natural_result,
    shard_event,
)


def store(tmp_path: Path) -> Part1ShardStore:
    return Part1ShardStore(
        tmp_path / "shard",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )


def test_separate_streams_durably_preserve_full_precision_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    started = attempt_event(result, "attempt_started", 0)
    fsync_calls = 0

    def observed_fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1

    monkeypatch.setattr("part1_store.os.fsync", observed_fsync)
    shard.append_audit_event(started)
    shard.append_terminal_result(result)

    assert fsync_calls == 2
    assert shard.natural_results_path.exists()
    assert shard.audit_events_path.exists()
    assert not shard.checkpoint_results_path.exists()
    stored = json.loads(shard.natural_results_path.read_text(encoding="utf-8"))
    assert stored["per_token_entropy_nats"] == result["per_token_entropy_nats"]
    assert "1.1234567890123457" in shard.natural_results_path.read_text(encoding="utf-8")

    misaligned = natural_result()
    misaligned["per_token_entropy_nats"] = [1.0]
    with pytest.raises(ValueError, match="aligned"):
        store(tmp_path / "bad").append_terminal_result(misaligned)

    checkpoint = checkpoint_result()
    checkpoint["ad_logits_float32"] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="four|too short"):
        store(tmp_path / "bad-checkpoint").append_terminal_result(checkpoint)


def test_duplicate_conflict_and_premature_terminal_failure_are_rejected(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_terminal_result(result)
    with pytest.raises(DuplicateTerminalResultError, match="already has a terminal result"):
        shard.append_terminal_result(result)

    conflicting = dict(result)
    conflicting["natural_correct"] = False
    with pytest.raises(DuplicateTerminalResultError, match="already has a terminal result"):
        shard.append_terminal_result(conflicting)

    failure_store = store(tmp_path / "failure")
    for attempt_number in (1, 2):
        attempt = natural_result(attempt_number=attempt_number)
        failure_store.append_audit_event(attempt_event(attempt, "attempt_started", 0))
    failure = natural_result(attempt_number=2, outcome="terminal_infrastructure_failure")
    with pytest.raises(ValueError, match="attempt exhaustion"):
        failure_store.append_terminal_result(failure)

    exhausted = natural_result(attempt_number=3, outcome="terminal_infrastructure_failure")
    exhausted_store = store(tmp_path / "exhausted")
    for attempt_number in (1, 2, 3):
        attempt = natural_result(attempt_number=attempt_number)
        exhausted_store.append_audit_event(attempt_event(attempt, "attempt_started", 0))
    exhausted_store.append_terminal_result(exhausted)


@pytest.mark.parametrize(
    ("boundary", "terminal_count", "completion_count", "has_tail"),
    [
        ("before_result_append", 0, 0, False),
        ("during_result_append", 0, 0, True),
        ("after_result_fsync_before_completion_event", 1, 0, False),
        ("during_completion_event_append", 1, 0, True),
        ("after_both_fsyncs", 1, 1, False),
    ],
)
def test_controlled_crashes_cover_all_durable_append_boundaries(
    tmp_path: Path,
    boundary: str,
    terminal_count: int,
    completion_count: int,
    has_tail: bool,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    with pytest.raises(InjectedCrash, match=boundary):
        shard.commit_terminal_result(
            result,
            attempt_event(result, "attempt_completed", 1),
            fault_at=boundary,
        )

    inspection = shard.inspect()
    assert len(inspection.natural_results) == terminal_count
    assert sum(event["event_type"] == "attempt_completed" for event in inspection.audit_events) == completion_count
    assert bool(inspection.trailing_tails) is has_tail


def test_terminal_result_without_completion_is_authoritative_and_reconciliation_idempotent(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_terminal_result(result)

    before = shard.build_index()
    key = (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0)
    assert key in before.completed_keys
    assert before.missing_completion_record_ids == {result["raw_record_id"]}

    first = shard.reconcile(
        event_timestamp="2026-07-31T00:00:01Z",
        execution_context={"hostname": "node", "pid": 456},
    )
    second = shard.reconcile(
        event_timestamp="2026-07-31T00:00:02Z",
        execution_context={"hostname": "node", "pid": 456},
    )
    assert [event["event_type"] for event in first] == ["terminal_result_recovered"]
    assert second == []
    assert shard.natural_results_path.read_bytes().count(b"\n") == 1


def test_completion_without_result_and_orphan_started_attempt_are_interrupted_and_consumed(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    missing_result = natural_result()
    shard.append_audit_event(attempt_event(missing_result, "attempt_started", 0))
    shard.append_audit_event(attempt_event(missing_result, "attempt_completed", 1))

    checkpoint = checkpoint_result(attempt_number=2)
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    index = shard.build_index()
    natural_key = (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0)
    checkpoint_key = (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, "cp-05")
    assert index.attempts_consumed[natural_key] == frozenset({1})
    assert index.attempts_consumed[checkpoint_key] == frozenset({2})
    assert len(index.inconsistent_completion_attempt_ids) == 1
    assert len(index.orphaned_attempt_ids) == 1
    assert natural_key not in index.completed_keys

    appended = shard.reconcile(
        event_timestamp="2026-07-31T00:00:03Z",
        execution_context={"hostname": "node", "pid": 789},
    )
    assert [event["event_type"] for event in appended] == [
        "attempt_interrupted",
        "attempt_interrupted",
    ]
    assert shard.reconcile(
        event_timestamp="2026-07-31T00:00:04Z", execution_context={}
    ) == []


def test_partial_tail_recovery_quarantines_exact_bytes_and_preserves_valid_prefix(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_terminal_result(result)
    valid_prefix = shard.natural_results_path.read_bytes()
    partial = b'{"schema_name":"part1_natural_terminal_result","raw_record_id":"cut'
    with shard.natural_results_path.open("ab") as handle:
        handle.write(partial)

    recovered = shard.recover_trailing_line(
        "natural_results",
        event_sequence=0,
        event_timestamp="2026-07-31T00:00:05Z",
        execution_context={"hostname": "node", "pid": 123},
    )
    assert recovered is not None
    assert recovered.read_bytes() == partial
    assert shard.natural_results_path.read_bytes() == valid_prefix
    assert shard.inspect().audit_events[-1]["event_type"] == "trailing_line_recovered"
    assert shard.recover_trailing_line(
        "natural_results",
        event_sequence=1,
        event_timestamp="2026-07-31T00:00:06Z",
        execution_context={},
    ) is None


def test_malformed_middle_is_rejected_and_never_truncated(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    valid_line = json.dumps(result, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    original = valid_line + b"not-json\n" + valid_line
    shard.root.mkdir(parents=True)
    shard.natural_results_path.write_bytes(original)
    with pytest.raises(MalformedMiddleError, match="middle"):
        shard.recover_trailing_line(
            "natural_results",
            event_sequence=0,
            event_timestamp="2026-07-31T00:00:05Z",
            execution_context={},
        )
    assert shard.natural_results_path.read_bytes() == original


def test_append_refuses_unrecovered_tail_and_finalized_shard_is_immutable(tmp_path: Path) -> None:
    shard = store(tmp_path)
    shard.root.mkdir(parents=True)
    shard.audit_events_path.write_bytes(b'{"incomplete"')
    with pytest.raises(StreamTailError, match="recover"):
        shard.append_audit_event(attempt_event(natural_result(), "attempt_started", 0))

    clean = store(tmp_path / "clean")
    clean_result = natural_result()
    clean.append_audit_event(attempt_event(clean_result, "attempt_started", 0))
    clean.commit_terminal_result(
        clean_result, attempt_event(clean_result, "attempt_completed", 1)
    )
    clean.finalize()
    before = clean.natural_results_path.read_bytes()
    with pytest.raises(FinalizedShardError):
        clean.append_terminal_result(checkpoint_result())
    with pytest.raises(FinalizedShardError):
        clean.recover_trailing_line(
            "natural_results",
            event_sequence=0,
            event_timestamp="2026-07-31T00:00:05Z",
            execution_context={},
        )
    assert clean.natural_results_path.read_bytes() == before


def test_validation_report_covers_required_checks_without_mutating_raw_streams(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    natural = natural_result()
    checkpoint = checkpoint_result()
    shard.append_audit_event(attempt_event(natural, "attempt_started", 0))
    shard.commit_terminal_result(natural, attempt_event(natural, "attempt_completed", 1))
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    shard.commit_terminal_result(checkpoint, attempt_event(checkpoint, "attempt_completed", 1))
    raw_before = {path: path.read_bytes() for path in shard.stream_paths.values() if path.exists()}

    report_path = tmp_path / "reports" / "validation.json"
    report = shard.write_validation_report(
        report_path,
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is True
    assert {check["name"] for check in report["checks"]} == {
        "json_syntax",
        "schema_validity",
        "duplicates_conflicts",
        "malformed_middle",
        "trailing_tail_state",
        "array_alignment",
        "terminal_event_consistency",
        "outcome_nullability",
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert raw_before == {path: path.read_bytes() for path in raw_before}


def test_checkpoint_and_natural_indexes_remain_independent(tmp_path: Path) -> None:
    shard = store(tmp_path)
    natural = natural_result()
    checkpoint = checkpoint_result()
    shard.append_terminal_result(natural)
    shard.append_terminal_result(checkpoint)
    index = shard.build_index()
    assert len(index.natural_terminal_by_key) == 1
    assert len(index.checkpoint_terminal_by_key) == 1
    assert (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0) in index.completed_keys
    assert (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, "cp-05") in index.completed_keys


def test_valid_json_at_eof_without_newline_is_preserved_and_repaired(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    shard.root.mkdir(parents=True)
    shard.natural_results_path.write_bytes(encoded)

    inspection = shard.inspect()
    assert inspection.natural_results == (result,)
    assert inspection.trailing_tails == {}
    assert inspection.unterminated_streams == frozenset({"natural_results"})

    assert shard.recover_trailing_line(
        "natural_results",
        event_sequence=0,
        event_timestamp="2026-07-31T00:00:07Z",
        execution_context={"hostname": "node", "pid": 123},
    ) is None
    assert shard.natural_results_path.read_bytes() == encoded + b"\n"
    assert not (shard.root / "quarantine").exists()
    assert shard.inspect().audit_events[-1]["event_type"] == "trailing_line_recovered"


def test_commit_requires_started_attempt_and_validation_flags_raw_result_without_start(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    with pytest.raises(ValueError, match="attempt_started"):
        shard.commit_terminal_result(result, attempt_event(result, "attempt_completed", 1))

    shard.append_terminal_result(result)
    report = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is False
    consistency = next(
        check for check in report["checks"] if check["name"] == "terminal_event_consistency"
    )
    assert consistency["outcome"] == "failed"


def test_commit_preflights_completion_before_writing_terminal_bytes(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    started = attempt_event(result, "attempt_started", 0)
    completion = attempt_event(result, "attempt_completed", 1)
    shard.append_audit_event(started)
    shard.append_audit_event(completion)

    with pytest.raises(ValueError, match="duplicate audit event"):
        shard.commit_terminal_result(result, completion)
    assert not shard.natural_results_path.exists()


def test_validation_rejects_completion_reference_to_a_different_attempt_result(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    natural = natural_result()
    checkpoint = checkpoint_result()
    shard.append_audit_event(attempt_event(natural, "attempt_started", 0))
    shard.append_terminal_result(natural)
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    shard.append_terminal_result(checkpoint)
    mismatched = attempt_event(checkpoint, "attempt_completed", 1)
    mismatched["terminal_record_id"] = natural["raw_record_id"]
    shard.append_audit_event(mismatched)

    report = shard.validate_shard(
        artifact_kind="checkpoint_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is False
    consistency = next(
        check for check in report["checks"] if check["name"] == "terminal_event_consistency"
    )
    assert consistency["outcome"] == "failed"


def test_validation_distinguishes_valid_json_with_invalid_schema(tmp_path: Path) -> None:
    shard = store(tmp_path)
    shard.root.mkdir(parents=True)
    shard.natural_results_path.write_text(
        json.dumps({"schema_name": "part1_natural_terminal_result"}) + "\n",
        encoding="utf-8",
    )
    report = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    checks = {check["name"]: check["outcome"] for check in report["checks"]}
    assert checks["json_syntax"] == "passed"
    assert checks["schema_validity"] == "failed"
    assert checks["duplicates_conflicts"] == "warning"


def test_finalization_requires_reconciliation_of_missing_completion_evidence(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_terminal_result(result)
    with pytest.raises(Part1StoreError, match="reconciliation"):
        shard.finalize()

    shard.reconcile(
        event_timestamp="2026-07-31T00:00:01Z",
        execution_context={"hostname": "node", "pid": 123},
    )
    shard.finalize()


def test_recovery_event_is_preflighted_before_trailing_bytes_are_changed(tmp_path: Path) -> None:
    shard = store(tmp_path)
    duplicate_event = shard_event("trailing_line_recovered", 0)
    shard.append_audit_event(duplicate_event)
    shard.root.mkdir(parents=True, exist_ok=True)
    partial = b'{"schema_name":"part1_natural_terminal_result"'
    shard.natural_results_path.write_bytes(partial)

    with pytest.raises(ValueError, match="duplicate audit event"):
        shard.recover_trailing_line(
            "natural_results",
            event_sequence=0,
            event_timestamp="2026-07-31T00:00:01Z",
            execution_context={"hostname": "node", "pid": 123},
        )
    assert shard.natural_results_path.read_bytes() == partial
    assert not (shard.root / "quarantine").exists()
