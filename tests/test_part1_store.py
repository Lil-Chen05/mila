"""Synthetic, login-safe tests for Part 1 append-only storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from part1_contract import shared_probe_id
from part1_store import (
    DuplicateTerminalResultError,
    FinalizedShardError,
    InjectedCrash,
    InvalidRecordError,
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
    shard = Part1ShardStore(
        tmp_path / "shard",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        unsafe_for_tests=True,
    )
    shard.initialize_provenance_header()
    return shard


def test_build_index_from_validated_snapshot_never_reopens_store_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.commit_terminal_result(
        result, attempt_event(result, "attempt_completed", 1)
    )
    inspection = shard.inspect()
    expected = shard.build_index()
    monkeypatch.setattr(
        shard,
        "inspect",
        lambda: (_ for _ in ()).throw(AssertionError("stream path reopened")),
    )
    monkeypatch.setattr(
        shard,
        "_load_recovery_journal_events",
        lambda: (_ for _ in ()).throw(AssertionError("journal path reopened")),
    )

    observed = shard.build_index_from_snapshot(
        inspection, recovery_journal_events=()
    )
    assert observed.natural_terminal_by_key == expected.natural_terminal_by_key
    assert observed.checkpoint_terminal_by_key == expected.checkpoint_terminal_by_key
    assert observed.pending_recovery_event_ids == frozenset()


def test_recovery_snapshot_rejects_duplicate_effective_journal_names(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    with pytest.raises(InvalidRecordError, match="duplicate recovery journal filename"):
        shard.recovery_journal_events_from_snapshot(
            {
                "recovery_journal/first/event.json": b"{}",
                "recovery_journal/second/event.json": b"{}",
            }
        )


def _write_synthetic_raw_terminal(shard: Part1ShardStore, result: dict) -> None:
    """Bypass public publication only to construct explicitly corrupt disk state."""

    shard.root.mkdir(parents=True, exist_ok=True)
    path = (
        shard.natural_results_path
        if result["schema_name"] == "part1_natural_terminal_result"
        else shard.checkpoint_results_path
    )
    path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _aliased_checkpoint(
    *, checkpoint_index: int, prefix_hash: str, inducer_version: str, inducer_text: str
) -> dict:
    checkpoint = checkpoint_result(checkpoint_id=f"cp-{checkpoint_index:02d}")
    checkpoint.update(
        requested_checkpoint_index=checkpoint_index,
        requested_fraction=checkpoint_index / 10,
        k_keep=0,
        actual_fraction=0.0,
        prefix_hash=prefix_hash,
        inducer_version=inducer_version,
        inducer_text=inducer_text,
        is_alias=checkpoint_index != 0,
        alias_metadata={
            "owner_checkpoint_id": "cp-00",
            "members": [f"cp-{index:02d}" for index in range(6)],
        },
    )
    checkpoint["shared_probe_id"] = shared_probe_id(
        checkpoint["study_id"],
        checkpoint["model_run_id"],
        checkpoint["question_id"],
        checkpoint["run_id"],
        prefix_hash=prefix_hash,
        inducer_version=inducer_version,
    )
    return checkpoint


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

    assert fsync_calls == 4
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


def test_first_stream_entries_and_final_artifacts_fsync_their_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = store(tmp_path)
    synced_directories: list[Path] = []
    monkeypatch.setattr(
        shard,
        "_fsync_directory",
        lambda path: synced_directories.append(Path(path)),
    )
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    assert synced_directories == [shard.root]
    synced_directories.clear()
    shard.append_terminal_result(result)
    assert synced_directories == [shard.root]
    synced_directories.clear()
    report_path = shard.root / "reports" / "report.json"
    shard.write_validation_report(
        report_path,
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report_path.parent in synced_directories

    finalized = store(tmp_path / "finalized")
    finalized_synced: list[Path] = []
    monkeypatch.setattr(
        finalized,
        "_fsync_directory",
        lambda path: finalized_synced.append(Path(path)),
    )
    finalized.finalize()
    assert finalized.root in finalized_synced


@pytest.mark.parametrize("is_checkpoint", [False, True])
def test_nonretryable_interruption_cannot_be_published_or_validate(
    tmp_path: Path, is_checkpoint: bool
) -> None:
    shard = store(tmp_path)
    record = checkpoint_result() if is_checkpoint else natural_result()
    if is_checkpoint:
        parent = natural_result()
        shard.append_audit_event(attempt_event(parent, "attempt_started", 0))
        shard.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    shard.append_audit_event(attempt_event(record, "attempt_started", 0))
    interrupted = attempt_event(record, "attempt_interrupted", 1)
    interrupted.update(
        outcome_category="invalid_configuration",
        retry_classification="terminal",
        retry_decision="do_not_retry",
        backoff_seconds=None,
    )
    with pytest.raises(ValueError, match="interrupted|terminalization"):
        shard.append_audit_event(interrupted)


def test_handwritten_nonretryable_interruption_cannot_validate_or_finalize(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    record = natural_result()
    started = attempt_event(record, "attempt_started", 0)
    interrupted = attempt_event(record, "attempt_interrupted", 1)
    interrupted.update(
        outcome_category="invalid_configuration",
        retry_classification="terminal",
        retry_decision="do_not_retry",
        backoff_seconds=None,
    )
    shard.audit_events_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in (started, interrupted)
        ),
        encoding="utf-8",
    )
    report = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is False
    with pytest.raises(Part1StoreError, match="invalid shard"):
        shard.finalize()


@pytest.mark.parametrize("is_checkpoint", [False, True])
@pytest.mark.parametrize("attempt_number", [1, 2])
def test_early_exhausted_interruption_is_rejected_for_every_key_kind(
    tmp_path: Path, is_checkpoint: bool, attempt_number: int
) -> None:
    shard = store(tmp_path)
    if is_checkpoint:
        parent = natural_result()
        shard.append_audit_event(attempt_event(parent, "attempt_started", 0))
        shard.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    factory = checkpoint_result if is_checkpoint else natural_result
    if attempt_number == 2:
        first = factory(attempt_number=1)
        shard.append_audit_event(attempt_event(first, "attempt_started", 0))
        shard.append_audit_event(attempt_event(first, "attempt_failed", 1))
    record = factory(attempt_number=attempt_number)
    shard.append_audit_event(attempt_event(record, "attempt_started", 0))
    interrupted = attempt_event(record, "attempt_interrupted", 1)
    interrupted.update(retry_decision="exhausted", backoff_seconds=None)
    with pytest.raises(ValueError, match="retry decision|policy"):
        shard.append_audit_event(interrupted)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k_keep", 1),
        ("actual_fraction", 0.5),
        ("shared_probe_id", "2" * 64),
        ("is_alias", False),
        ("alias_metadata", {"owner_checkpoint_id": "cp-05", "members": ["cp-05"]}),
    ],
)
def test_checkpoint_parent_recomputes_placement_probe_and_aliases(
    tmp_path: Path, field: str, value
) -> None:
    shard = store(tmp_path)
    parent = natural_result()
    shard.append_audit_event(attempt_event(parent, "attempt_started", 0))
    shard.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    checkpoint = checkpoint_result()
    checkpoint[field] = value
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    with pytest.raises(ValueError, match="placement|fraction|probe|alias"):
        shard.append_terminal_result(checkpoint)


@pytest.mark.parametrize("reasoning_count", [0, 1, 2, 3])
def test_checkpoint_parent_accepts_exact_zero_short_and_ties_even_placements(
    tmp_path: Path, reasoning_count: int
) -> None:
    shard = store(tmp_path)
    parent = natural_result()
    parent["reasoning_token_count"] = reasoning_count
    if reasoning_count == 0:
        parent.update(
            reasoning_status="no_reasoning",
            reasoning_text="",
            mean_reasoning_entropy_nats=None,
            tail_reasoning_entropy_nats=None,
        )
    shard.append_audit_event(attempt_event(parent, "attempt_started", 0))
    shard.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))

    checkpoint = checkpoint_result()
    k_keep = round(0.5 * reasoning_count)
    placements = [round((index / 10) * reasoning_count) for index in range(11)]
    members = [
        f"cp-{index:02d}"
        for index, placement in enumerate(placements)
        if placement == k_keep
    ]
    checkpoint.update(
        k_keep=k_keep,
        actual_fraction=(k_keep / reasoning_count if reasoning_count else None),
        is_alias=checkpoint["checkpoint_id"] != members[0],
        alias_metadata={"owner_checkpoint_id": members[0], "members": members},
    )
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    shard.append_terminal_result(checkpoint)


def test_alias_group_rejects_cross_record_physical_probe_disagreement_on_append(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    parent = natural_result()
    shard.append_audit_event(attempt_event(parent, "attempt_started", 0))
    shard.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    owner = _aliased_checkpoint(
        checkpoint_index=0,
        prefix_hash="3" * 64,
        inducer_version="smollm3-forced-close-v1",
        inducer_text="</think>\nAnswer:",
    )
    member = _aliased_checkpoint(
        checkpoint_index=1,
        prefix_hash="4" * 64,
        inducer_version="smollm3-forced-close-v2",
        inducer_text="</think>\nDifferent:",
    )
    shard.append_audit_event(attempt_event(owner, "attempt_started", 0))
    shard.append_terminal_result(owner)
    shard.append_audit_event(attempt_event(member, "attempt_started", 0))
    with pytest.raises(ValueError, match="alias.*physical|physical.*alias"):
        shard.append_terminal_result(member)


def test_alias_group_disagreement_is_reported_when_read_from_disk(tmp_path: Path) -> None:
    shard = store(tmp_path)
    parent = natural_result()
    shard.append_audit_event(attempt_event(parent, "attempt_started", 0))
    shard.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    owner = _aliased_checkpoint(
        checkpoint_index=0,
        prefix_hash="3" * 64,
        inducer_version="smollm3-forced-close-v1",
        inducer_text="</think>\nAnswer:",
    )
    member = _aliased_checkpoint(
        checkpoint_index=1,
        prefix_hash="4" * 64,
        inducer_version="smollm3-forced-close-v2",
        inducer_text="</think>\nDifferent:",
    )
    shard.checkpoint_results_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in (owner, member)
        ),
        encoding="utf-8",
    )
    index = shard.build_index()
    assert any("alias" in error and "physical" in error for error in index.hierarchy_errors)
    report = shard.validate_shard(
        artifact_kind="checkpoint_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is False


def test_unlocked_store_mutation_is_impossible_by_default(tmp_path: Path) -> None:
    shard = Part1ShardStore(
        tmp_path / "read-only",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )
    with pytest.raises(Part1StoreError, match="lock capability|unsafe_for_tests"):
        shard.initialize_provenance_header()


def test_duplicate_conflict_and_premature_terminal_failure_are_rejected(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
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
        if attempt_number == 1:
            failure_store.append_audit_event(attempt_event(attempt, "attempt_failed", 1))
    failure = natural_result(attempt_number=2, outcome="terminal_infrastructure_failure")
    with pytest.raises(ValueError, match="exhaustion"):
        failure_store.append_terminal_result(failure)

    exhausted = natural_result(attempt_number=3, outcome="terminal_infrastructure_failure")
    exhausted_store = store(tmp_path / "exhausted")
    for attempt_number in (1, 2, 3):
        attempt = natural_result(attempt_number=attempt_number)
        exhausted_store.append_audit_event(attempt_event(attempt, "attempt_started", 0))
        if attempt_number < 3:
            exhausted_store.append_audit_event(attempt_event(attempt, "attempt_failed", 1))
    exhausted_store.append_terminal_result(exhausted)


def test_nonretryable_failure_terminalizes_current_attempt_without_fabricated_retries(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    failure = natural_result(attempt_number=1, outcome="terminal_infrastructure_failure")
    failure["terminal_error_details"]["category"] = "invalid_configuration"
    shard.append_audit_event(attempt_event(failure, "attempt_started", 0))
    completion = attempt_event(failure, "attempt_completed", 1)
    shard.commit_terminal_result(failure, completion)
    assert len(shard.inspect().natural_results) == 1
    stored_completion = shard.inspect().audit_events[-1]
    assert stored_completion["outcome_category"] == "invalid_configuration"
    assert stored_completion["retry_classification"] == "terminal"
    assert stored_completion["retry_decision"] == "do_not_retry"


def test_retryable_terminal_failure_requires_real_sequential_exhaustion(tmp_path: Path) -> None:
    shard = store(tmp_path)
    first = natural_result(attempt_number=1)
    shard.append_audit_event(attempt_event(first, "attempt_started", 0))
    premature = natural_result(attempt_number=1, outcome="terminal_infrastructure_failure")
    with pytest.raises(ValueError, match="retryable.*attempt 3|exhaustion"):
        shard.append_terminal_result(premature)

    failed = attempt_event(first, "attempt_failed", 1)
    failed.update(
        outcome_category="temporary_filesystem_failure",
        retry_classification="retryable",
        retry_decision="retry",
    )
    shard.append_audit_event(failed)
    third = natural_result(attempt_number=3)
    with pytest.raises(ValueError, match="sequential|attempt 2"):
        shard.append_audit_event(attempt_event(third, "attempt_started", 0))


def test_failure_event_rejects_policy_metadata_inconsistent_with_category(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    failed = attempt_event(result, "attempt_failed", 1)
    failed.update(
        outcome_category="invalid_configuration",
        retry_classification="retryable",
        retry_decision="retry",
    )
    with pytest.raises(ValueError, match="retry classification|retry decision|policy"):
        shard.append_audit_event(failed)


def test_read_only_index_flags_persisted_failure_policy_mismatch(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    started = attempt_event(result, "attempt_started", 0)
    failed = attempt_event(result, "attempt_failed", 1)
    failed.update(
        outcome_category="invalid_configuration",
        retry_classification="retryable",
        retry_decision="retry",
    )
    shard.root.mkdir(parents=True, exist_ok=True)
    shard.audit_events_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in (started, failed)
        ),
        encoding="utf-8",
    )
    index = shard.build_index()
    assert any("failure policy" in error for error in index.lifecycle_errors)
    report = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is False


def test_read_only_index_flags_terminal_completion_policy_mismatch(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result(attempt_number=1, outcome="terminal_infrastructure_failure")
    result["terminal_error_details"]["category"] = "invalid_configuration"
    started = attempt_event(result, "attempt_started", 0)
    completion = attempt_event(result, "attempt_completed", 1)
    completion.update(
        outcome_category="temporary_filesystem_failure",
        retry_classification="retryable",
        retry_decision="retry",
    )
    _write_synthetic_raw_terminal(shard, result)
    shard.audit_events_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in (started, completion)
        ),
        encoding="utf-8",
    )
    index = shard.build_index()
    assert any("completion failure policy" in error for error in index.lifecycle_errors)


def test_read_only_index_flags_nonsequential_persisted_attempt_numbers(tmp_path: Path) -> None:
    shard = store(tmp_path)
    third = natural_result(attempt_number=3)
    shard.root.mkdir(parents=True, exist_ok=True)
    shard.audit_events_path.write_text(
        json.dumps(
            attempt_event(third, "attempt_started", 0),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    index = shard.build_index()
    assert any("nonsequential" in error for error in index.lifecycle_errors)


def test_retry_failure_backoff_must_match_shared_schedule(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    bad = attempt_event(result, "attempt_failed", 1)
    bad["backoff_seconds"] = 99
    with pytest.raises(ValueError, match="backoff"):
        shard.append_audit_event(bad)


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

    checkpoint = checkpoint_result(attempt_number=1)
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    index = shard.build_index()
    natural_key = (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0)
    checkpoint_key = (STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, "cp-05")
    assert index.attempts_consumed[natural_key] == frozenset({1})
    assert index.attempts_consumed[checkpoint_key] == frozenset({1})
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
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
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
    shard.root.mkdir(parents=True, exist_ok=True)
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
    shard.root.mkdir(parents=True, exist_ok=True)
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
    shard.append_audit_event(attempt_event(natural, "attempt_started", 0))
    shard.append_terminal_result(natural)
    shard.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
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
    shard.root.mkdir(parents=True, exist_ok=True)
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


def test_public_terminal_append_requires_start_and_writes_no_authoritative_bytes(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    with pytest.raises(ValueError, match="attempt_started"):
        shard.commit_terminal_result(result, attempt_event(result, "attempt_completed", 1))

    with pytest.raises(ValueError, match="attempt_started"):
        shard.append_terminal_result(result)
    assert not shard.natural_results_path.exists()
    assert shard.reconcile(
        event_timestamp="2026-07-31T00:00:01Z",
        execution_context={"hostname": "node", "pid": 123},
    ) == []
    assert shard.inspect().natural_results == ()
    assert shard.inspect().audit_events == ()


def test_commit_preflights_completion_before_writing_terminal_bytes(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    started = attempt_event(result, "attempt_started", 0)
    completion = attempt_event(result, "attempt_completed", 1)
    shard.append_audit_event(started)
    shard.append_audit_event(completion)

    with pytest.raises(ValueError, match="duplicate audit event|terminal lifecycle"):
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
    shard.root.mkdir(parents=True, exist_ok=True)
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


@pytest.mark.parametrize(
    "fault_at",
    [
        "before_recovery_evidence",
        "after_recovery_evidence_before_mutation",
        "after_recovery_mutation_before_audit_append",
    ],
)
def test_result_tail_recovery_resumes_from_every_durable_boundary(
    tmp_path: Path, fault_at: str
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_terminal_result(result)
    valid_prefix = shard.natural_results_path.read_bytes()
    partial = b'{"schema_name":"part1_natural_terminal_result","cut"'
    with shard.natural_results_path.open("ab") as handle:
        handle.write(partial)

    with pytest.raises(InjectedCrash, match=fault_at):
        shard.recover_trailing_line(
            "natural_results",
            event_sequence=7,
            event_timestamp="2026-07-31T00:00:08Z",
            execution_context={"hostname": "node", "pid": 123},
            fault_at=fault_at,
        )

    quarantine_files = list((shard.root / "quarantine").glob("*.bin"))
    assert len(quarantine_files) == 1
    assert quarantine_files[0].read_bytes() == partial
    journal_files = list(shard.recovery_journal_directory.glob("*.json"))
    if fault_at == "before_recovery_evidence":
        assert journal_files == []
        assert shard.natural_results_path.read_bytes() == valid_prefix + partial
    else:
        assert len(journal_files) == 1
        if fault_at == "after_recovery_evidence_before_mutation":
            assert shard.natural_results_path.read_bytes() == valid_prefix + partial
        else:
            assert shard.natural_results_path.read_bytes() == valid_prefix
            report = shard.validate_shard(
                artifact_kind="natural_shard",
                started_at="2026-07-31T00:00:00Z",
                completed_at="2026-07-31T00:00:01Z",
            )
            assert report["is_valid"] is False
            with pytest.raises(Part1StoreError, match="invalid|recovery"):
                shard.finalize()

    recovered_path = shard.recover_trailing_line(
        "natural_results",
        event_sequence=7,
        event_timestamp="2026-07-31T00:00:08Z",
        execution_context={"hostname": "node", "pid": 123},
    )
    assert recovered_path == quarantine_files[0]
    assert shard.natural_results_path.read_bytes() == valid_prefix
    recovery_events = [
        event
        for event in shard.inspect().audit_events
        if event["event_type"] == "trailing_line_recovered"
    ]
    assert len(recovery_events) == 1
    assert shard.recover_trailing_line(
        "natural_results",
        event_sequence=7,
        event_timestamp="2026-07-31T00:00:08Z",
        execution_context={"hostname": "node", "pid": 123},
    ) == quarantine_files[0]
    assert len(
        [event for event in shard.inspect().audit_events if event["event_type"] == "trailing_line_recovered"]
    ) == 1


@pytest.mark.parametrize(
    "fault_at",
    [
        "after_recovery_evidence_before_mutation",
        "after_recovery_mutation_before_audit_append",
    ],
)
def test_audit_tail_recovery_uses_external_durable_evidence(
    tmp_path: Path, fault_at: str
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    valid_prefix = shard.audit_events_path.read_bytes()
    partial = b'{"schema_name":"part1_audit_event","cut"'
    with shard.audit_events_path.open("ab") as handle:
        handle.write(partial)

    with pytest.raises(InjectedCrash, match=fault_at):
        shard.recover_trailing_line(
            "audit_events",
            event_sequence=8,
            event_timestamp="2026-07-31T00:00:09Z",
            execution_context={"hostname": "node", "pid": 123},
            fault_at=fault_at,
        )
    assert len(list(shard.recovery_journal_directory.glob("*.json"))) == 1

    shard.recover_trailing_line(
        "audit_events",
        event_sequence=8,
        event_timestamp="2026-07-31T00:00:09Z",
        execution_context={"hostname": "node", "pid": 123},
    )
    assert shard.audit_events_path.read_bytes().startswith(valid_prefix)
    assert partial not in shard.audit_events_path.read_bytes()
    assert sum(
        event["event_type"] == "trailing_line_recovered"
        for event in shard.inspect().audit_events
    ) == 1


def test_pending_recovery_journals_can_be_finished_without_reconstructing_arguments(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    shard.root.mkdir(parents=True, exist_ok=True)
    partial = b'{"schema_name":"part1_natural_terminal_result","cut"'
    shard.natural_results_path.write_bytes(partial)
    with pytest.raises(InjectedCrash):
        shard.recover_trailing_line(
            "natural_results",
            event_sequence=11,
            event_timestamp="2026-07-31T00:00:10Z",
            execution_context={"hostname": "node", "pid": 123},
            fault_at="after_recovery_mutation_before_audit_append",
        )

    assert shard.finish_pending_recoveries() == [
        next(shard.recovery_journal_directory.glob("*.json")).stem
    ]
    assert shard.finish_pending_recoveries() == []


def test_attempt_terminal_lifecycle_rejects_results_after_failure_or_interruption(
    tmp_path: Path,
) -> None:
    for terminal_type in ("attempt_failed", "attempt_interrupted"):
        shard = store(tmp_path / terminal_type)
        result = natural_result()
        shard.append_audit_event(attempt_event(result, "attempt_started", 0))
        shard.append_audit_event(attempt_event(result, terminal_type, 1))
        with pytest.raises(ValueError, match="terminal lifecycle|already.*terminal"):
            shard.append_terminal_result(result)
        with pytest.raises(ValueError, match="terminal lifecycle|already.*terminal"):
            shard.commit_terminal_result(result, attempt_event(result, "attempt_completed", 2))


def test_attempt_events_require_start_and_reject_conflicting_terminal_states(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    with pytest.raises(ValueError, match="attempt_started"):
        shard.append_audit_event(attempt_event(result, "attempt_failed", 1))

    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_audit_event(attempt_event(result, "attempt_failed", 1))
    with pytest.raises(ValueError, match="terminal lifecycle|already.*terminal"):
        shard.append_audit_event(attempt_event(result, "attempt_interrupted", 2))
    with pytest.raises(ValueError, match="terminal lifecycle|already.*terminal"):
        shard.append_audit_event(attempt_event(result, "attempt_completed", 2))


@pytest.mark.parametrize("event_type", ["attempt_failed", "attempt_interrupted"])
def test_failure_or_interruption_cannot_follow_an_existing_terminal_result(
    tmp_path: Path, event_type: str
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_terminal_result(result)
    with pytest.raises(ValueError, match="terminal result|terminal lifecycle"):
        shard.append_audit_event(attempt_event(result, event_type, 1))


def test_start_cannot_be_appended_retroactively_after_a_terminal_result(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    _write_synthetic_raw_terminal(shard, result)
    with pytest.raises(ValueError, match="attempt_started.*terminal result|terminal result.*start"):
        shard.append_audit_event(attempt_event(result, "attempt_started", 0))


def test_attempt_terminal_event_record_references_follow_lifecycle_contract(
    tmp_path: Path,
) -> None:
    result = natural_result()
    shard = store(tmp_path / "failed")
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    failed = attempt_event(result, "attempt_failed", 1)
    failed["terminal_record_id"] = result["raw_record_id"]
    with pytest.raises(ValueError, match="terminal_record_id"):
        shard.append_audit_event(failed)

    shard = store(tmp_path / "completed")
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    completion = attempt_event(result, "attempt_completed", 1)
    completion["terminal_record_id"] = None
    with pytest.raises(ValueError, match="terminal_record_id"):
        shard.append_audit_event(completion)


def test_recovered_result_may_receive_one_matching_completion(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_terminal_result(result)
    appended = shard.reconcile(
        event_timestamp="2026-07-31T00:00:01Z",
        execution_context={"hostname": "node", "pid": 123},
        append_missing_completion=True,
    )
    assert [event["event_type"] for event in appended] == [
        "terminal_result_recovered",
        "attempt_completed",
    ]
    assert shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )["is_valid"] is True
    with pytest.raises(ValueError, match="terminal lifecycle|duplicate"):
        shard.append_audit_event(attempt_event(result, "attempt_completed", 3))


def test_read_only_validation_rejects_handwritten_contradictory_lifecycle(tmp_path: Path) -> None:
    shard = store(tmp_path)
    result = natural_result()
    events = [
        attempt_event(result, "attempt_started", 0),
        attempt_event(result, "attempt_failed", 1),
        attempt_event(result, "attempt_completed", 2),
    ]
    shard.root.mkdir(parents=True, exist_ok=True)
    shard.natural_results_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    shard.audit_events_path.write_text(
        "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
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
    assert consistency["details"]["lifecycle_errors"]


def test_completion_without_result_can_be_classified_interrupted_and_then_validated(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.append_audit_event(attempt_event(result, "attempt_completed", 1))
    shard.reconcile(
        event_timestamp="2026-07-31T00:00:01Z",
        execution_context={"hostname": "node", "pid": 123},
    )
    report = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is True


def test_validation_report_identity_excludes_mutable_report_state_and_is_domain_separated(
    tmp_path: Path,
) -> None:
    shard = store(tmp_path)
    first = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    second = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2030-01-01T00:00:00Z",
        completed_at="2030-01-01T00:00:01Z",
    )
    assert first["validation_report_id"] == second["validation_report_id"]

    shard.root.mkdir(parents=True, exist_ok=True)
    shard.natural_results_path.write_bytes(b'{"incomplete"')
    invalid = shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2040-01-01T00:00:00Z",
        completed_at="2040-01-01T00:00:01Z",
    )
    assert invalid["is_valid"] is False
    assert invalid["validation_report_id"] == first["validation_report_id"]
    checkpoint_kind = shard.validate_shard(
        artifact_kind="checkpoint_shard",
        started_at="2040-01-01T00:00:00Z",
        completed_at="2040-01-01T00:00:01Z",
    )
    assert checkpoint_kind["validation_report_id"] != first["validation_report_id"]

    different_shard = Part1ShardStore(
        tmp_path / "other",
        shard_id="shard-001",
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        unsafe_for_tests=True,
    )
    different_shard.initialize_provenance_header()
    different = different_shard.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert different["validation_report_id"] != first["validation_report_id"]
