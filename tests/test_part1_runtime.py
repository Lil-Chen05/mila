"""Synthetic, login-safe tests for Part 1 runtime safety and resumability."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from part1_store import Part1ShardStore
from part1_store_fixtures import (
    MODEL_RUN_ID,
    MODEL_RUN_MANIFEST_HASH,
    QUESTION_ID,
    SHARD_ID,
    STUDY_ID,
    attempt_event,
    checkpoint_result,
    natural_result,
)


def test_runtime_contract_module_is_available() -> None:
    assert importlib.util.find_spec("part1_runtime") is not None


def test_store_checks_current_lock_ownership_before_each_mutation(tmp_path: Path) -> None:
    owned = True

    def assert_owned() -> None:
        if not owned:
            raise RuntimeError("writer lease was displaced")

    shard = Part1ShardStore(
        tmp_path / "shard",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        mutation_guard=assert_owned,
    )
    shard.initialize_provenance_header()
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))

    owned = False
    with pytest.raises(RuntimeError, match="displaced"):
        shard.append_terminal_result(result)


@pytest.mark.parametrize("operation", ["validation_report", "finalize"])
def test_store_rechecks_ownership_immediately_before_file_creation(
    tmp_path: Path, operation: str
) -> None:
    checks = 0

    def displaced_after_preflight() -> None:
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RuntimeError("writer lease was displaced")

    shard = Part1ShardStore(
        tmp_path / "shard",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        mutation_guard=displaced_after_preflight,
    )
    shard.initialize_provenance_header()
    if operation == "validation_report":
        output = tmp_path / "report.json"
        with pytest.raises(RuntimeError, match="displaced"):
            shard.write_validation_report(
                output,
                artifact_kind="natural_shard",
                started_at="2026-07-31T00:00:00Z",
                completed_at="2026-07-31T00:00:01Z",
            )
        assert not output.exists()
    else:
        with pytest.raises(RuntimeError, match="displaced"):
            shard.finalize()
        assert not shard.finalization_path.exists()


def _owner(*, pid: int, acquired_at: str = "2026-07-31T00:00:00Z"):
    from part1_runtime import LockMetadata

    return LockMetadata.new(
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        shard_id=SHARD_ID,
        hostname="synthetic-node",
        pid=pid,
        slurm_job_id="job-17",
        slurm_array_task_id="3",
        acquired_at=acquired_at,
    )


def _acquire(tmp_path: Path, *, pid: int = 101):
    from part1_runtime import LockedShardSession

    return LockedShardSession.acquire(
        tmp_path / "shard",
        owner=_owner(pid=pid),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )


def test_atomic_lock_rejects_second_writer_and_prevents_concurrent_append(tmp_path: Path) -> None:
    from part1_runtime import LockHeldError, LockedShardSession

    first = _acquire(tmp_path)
    result = natural_result()
    first.store.append_audit_event(attempt_event(result, "attempt_started", 0))

    with pytest.raises(LockHeldError, match="already locked"):
        LockedShardSession.acquire(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        )

    first.store.append_terminal_result(result)
    assert len(first.store.inspect().natural_results) == 1
    first.close()


def test_first_shard_root_creation_fsyncs_each_new_parent_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_runtime import LockedShardSession

    synced: list[Path] = []
    monkeypatch.setattr(
        "part1_runtime._fsync_directory", lambda path: synced.append(Path(path))
    )
    root = tmp_path / "level-one" / "level-two" / "shard"
    session = LockedShardSession.acquire(
        root,
        owner=_owner(pid=101),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )
    assert tmp_path in synced
    assert root.parent.parent in synced
    assert root.parent in synced
    session.close()


def test_verified_stale_recovery_is_race_safe_audited_and_displaces_old_writer(
    tmp_path: Path,
) -> None:
    from part1_runtime import Liveness, LockedShardSession, LostLockOwnershipError

    old = _acquire(tmp_path, pid=101)
    replacement = LockedShardSession.recover_stale(
        tmp_path / "shard",
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        worker_liveness=lambda metadata: Liveness.DEAD,
        slurm_liveness=lambda metadata: Liveness.DEAD,
        event_timestamp="2026-07-31T00:01:00Z",
        execution_context={"hostname": "recovery-node", "pid": 202},
    )

    event = replacement.store.inspect().audit_events[-1]
    assert event["event_type"] == "stale_lock_recovered"
    assert event["related_lock_owner"]["lock_id"] == old.owner.lock_id
    history = list((replacement.store.root / ".lock_history").glob("*.json"))
    assert len(history) == 2
    claim_history = next(path for path in history if path.name.endswith(".claim.json"))
    assert json.loads(claim_history.read_text(encoding="utf-8"))["previous_lock"]["pid"] == 101

    with pytest.raises(LostLockOwnershipError, match="no longer owns"):
        old.store.append_audit_event(attempt_event(natural_result(), "attempt_started", 0))
    replacement.close()


def test_takeover_crash_after_replacement_is_resumable_and_idempotent(tmp_path: Path) -> None:
    from part1_runtime import (
        InjectedTakeoverCrash,
        Liveness,
        LockHeldError,
        LockedShardSession,
    )

    old = _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash, match="after_lock_replacement"):
        LockedShardSession.recover_stale(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:01:00Z",
            execution_context={"hostname": "recovery-node", "pid": 202},
            fault_at="after_lock_replacement_before_event",
        )
    with pytest.raises(LockHeldError, match="recovery"):
        LockedShardSession.acquire(
            tmp_path / "shard",
            owner=_owner(pid=303),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        )

    recovered = LockedShardSession.finish_pending_takeover(tmp_path / "shard")
    events = [
        event
        for event in recovered.store.inspect().audit_events
        if event["event_type"] == "stale_lock_recovered"
    ]
    assert len(events) == 1
    assert events[0]["related_lock_owner"]["lock_id"] == old.owner.lock_id
    assert not (recovered.store.root / ".writer-lock-recovery.claim").exists()
    recovered.close()


def test_partial_takeover_event_append_recovers_without_sequence_collision(tmp_path: Path) -> None:
    from part1_runtime import InjectedTakeoverCrash, Liveness, LockedShardSession

    _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash, match="during_takeover_event_append"):
        LockedShardSession.recover_stale(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:01:00Z",
            execution_context={"hostname": "recovery-node", "pid": 202},
            fault_at="during_takeover_event_append",
        )
    recovered = LockedShardSession.finish_pending_takeover(tmp_path / "shard")
    shard_events = [
        event
        for event in recovered.store.inspect().audit_events
        if event["event_scope"] == "shard"
    ]
    assert [event["event_type"] for event in shard_events] == [
        "trailing_line_recovered",
        "stale_lock_recovered",
    ]
    assert len({event["event_sequence"] for event in shard_events}) == 2
    recovered.close()


def test_takeover_finishes_pending_storage_recovery_before_its_shard_event(tmp_path: Path) -> None:
    from part1_store import InjectedCrash
    from part1_runtime import LockedShardSession

    old = _acquire(tmp_path, pid=101)
    result = natural_result()
    old.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    with pytest.raises(InjectedCrash):
        old.store.commit_terminal_result(
            result,
            attempt_event(result, "attempt_completed", 1),
            fault_at="during_result_append",
        )
    with pytest.raises(InjectedCrash):
        old.store.recover_trailing_line(
            "natural_results",
            event_sequence=0,
            event_timestamp="2026-07-31T00:00:30Z",
            execution_context={"hostname": "old-node", "pid": 101},
            fault_at="after_recovery_evidence_before_mutation",
        )

    replacement = LockedShardSession.operator_unlock(
        tmp_path / "shard",
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        operator_reason="allocation ended during recovery",
        event_timestamp="2026-07-31T00:02:00Z",
        execution_context={"hostname": "operator-node", "pid": 202},
    )
    shard_events = [
        event
        for event in replacement.store.inspect().audit_events
        if event["event_scope"] == "shard"
    ]
    assert [event["event_type"] for event in shard_events] == [
        "trailing_line_recovered",
        "operator_unlock",
    ]
    assert len({event["event_sequence"] for event in shard_events}) == 2
    replacement.close()


def test_acquire_and_takeover_reject_finalized_shards(tmp_path: Path) -> None:
    from part1_runtime import FinalizedRuntimeShardError, Liveness, LockedShardSession

    session = _acquire(tmp_path, pid=101)
    session.store.finalize()
    with pytest.raises(FinalizedRuntimeShardError, match="finalized"):
        LockedShardSession.recover_stale(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:01:00Z",
            execution_context={"hostname": "recovery-node", "pid": 202},
        )
    session.close()
    with pytest.raises(FinalizedRuntimeShardError, match="finalized"):
        LockedShardSession.acquire(
            tmp_path / "shard",
            owner=_owner(pid=303),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        )


@pytest.mark.parametrize(
    ("worker_state", "slurm_state", "message"),
    [
        ("LIVE", "DEAD", "live"),
        ("DEAD", "LIVE", "live"),
        ("DEAD", "UNKNOWN", "uncertain"),
    ],
)
def test_stale_recovery_refuses_live_or_uncertain_liveness_and_never_uses_age(
    tmp_path: Path, worker_state: str, slurm_state: str, message: str
) -> None:
    from part1_runtime import Liveness, LockedShardSession, StaleRecoveryRefused

    old = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=_owner(pid=101, acquired_at="2000-01-01T00:00:00Z"),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )
    with pytest.raises(StaleRecoveryRefused, match=message):
        LockedShardSession.recover_stale(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness[worker_state],
            slurm_liveness=lambda metadata: Liveness[slurm_state],
            event_timestamp="2026-07-31T00:01:00Z",
            execution_context={"hostname": "recovery-node", "pid": 202},
        )
    old.assert_owned()
    old.close()


def test_default_remote_and_slurm_liveness_probes_fail_closed(tmp_path: Path) -> None:
    from part1_runtime import (
        Liveness,
        LockedShardSession,
        StaleRecoveryRefused,
        default_worker_liveness,
    )

    old = _acquire(tmp_path, pid=999999)
    with pytest.raises(StaleRecoveryRefused, match="uncertain"):
        LockedShardSession.recover_stale(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=default_worker_liveness,
            slurm_liveness=lambda metadata: Liveness.UNKNOWN,
            event_timestamp="2026-07-31T00:01:00Z",
            execution_context={"hostname": "recovery-node", "pid": 202},
        )
    old.close()


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "job-17_3\n", "LIVE"),
        (0, "", "DEAD"),
        (1, "", "UNKNOWN"),
        (0, "job-17_3\nother\n", "UNKNOWN"),
        (0, "other\n", "UNKNOWN"),
    ],
)
def test_slurm_probe_uses_exact_array_selector_and_fails_closed(
    returncode: int, stdout: str, expected: str
) -> None:
    from part1_runtime import Liveness, probe_slurm_liveness

    observed: dict = {}

    def runner(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return type("Result", (), {"returncode": returncode, "stdout": stdout})()

    assert probe_slurm_liveness(_owner(pid=101), runner=runner) is Liveness[expected]
    assert observed["command"] == [
        "squeue",
        "--jobs=job-17_3",
        "--noheader",
        "--format=%i",
    ]
    assert observed["kwargs"]["timeout"] > 0


def test_slurm_probe_treats_timeout_and_missing_command_as_unknown() -> None:
    import subprocess

    from part1_runtime import Liveness, probe_slurm_liveness

    for error in (subprocess.TimeoutExpired("squeue", 1), FileNotFoundError("squeue")):
        def runner(command, **kwargs):
            raise error

        assert probe_slurm_liveness(_owner(pid=101), runner=runner) is Liveness.UNKNOWN


def test_operator_unlock_requires_reason_audits_takeover_and_invalidates_old_lease(
    tmp_path: Path,
) -> None:
    from part1_runtime import LockedShardSession, LostLockOwnershipError

    old = _acquire(tmp_path, pid=101)
    with pytest.raises(ValueError, match="reason"):
        LockedShardSession.operator_unlock(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            operator_reason="  ",
            event_timestamp="2026-07-31T00:02:00Z",
            execution_context={"hostname": "operator-node", "pid": 202},
        )

    replacement = LockedShardSession.operator_unlock(
        tmp_path / "shard",
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        operator_reason="verified abandoned allocation",
        event_timestamp="2026-07-31T00:02:00Z",
        execution_context={"hostname": "operator-node", "pid": 202},
    )
    event = replacement.store.inspect().audit_events[-1]
    assert event["event_type"] == "operator_unlock"
    assert event["operator_reason"] == "verified abandoned allocation"
    assert event["related_lock_owner"]["lock_id"] == old.owner.lock_id
    with pytest.raises(LostLockOwnershipError):
        old.assert_owned()
    replacement.close()


@pytest.mark.parametrize(
    "category",
    [
        "interrupted_process",
        "temporary_filesystem_failure",
        "transient_worker_failure",
        "transient_cuda_runtime_failure",
    ],
)
def test_retryable_failure_categories_preserve_identity_and_seed(category: str) -> None:
    from part1_runtime import WorkSpec, plan_retry

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    plan = plan_retry(work, category=category, attempts_consumed=1)
    assert plan.decision == "retry"
    assert plan.next_attempt_number == 2
    assert plan.work == work
    assert plan.work.seed == 123


@pytest.mark.parametrize(
    "category",
    [
        "invalid_configuration",
        "schema_incompatibility",
        "manifest_incompatibility",
        "tokenizer_preflight_incompatibility",
        "deterministic_context_overflow",
        "reproducible_cuda_oom",
        "unsupported_model_or_tokenizer_behaviour",
        "corrupt_immutable_manifest",
    ],
)
def test_terminal_failure_categories_never_retry(category: str) -> None:
    from part1_runtime import WorkSpec, plan_retry

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    plan = plan_retry(work, category=category, attempts_consumed=1)
    assert plan.classification == "terminal"
    assert plan.decision == "do_not_retry"
    assert plan.next_attempt_number is None


def test_attempt_limit_is_exactly_three_total_attempts() -> None:
    from part1_runtime import WorkSpec, plan_retry

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    exhausted = plan_retry(work, category="interrupted_process", attempts_consumed=3)
    assert exhausted.decision == "exhausted"
    assert exhausted.next_attempt_number is None
    with pytest.raises(ValueError, match="0 through 3"):
        plan_retry(work, category="interrupted_process", attempts_consumed=4)


def test_retry_config_must_equal_both_shared_failure_category_sets() -> None:
    from part1_contract import load_config
    from part1_runtime import CompatibilityError, validate_retry_policy_config

    config = load_config("retries")
    validate_retry_policy_config(config)
    changed = dict(config)
    changed["terminal_categories"] = config["terminal_categories"][:-1]
    with pytest.raises(CompatibilityError, match="terminal_categories"):
        validate_retry_policy_config(changed)


def test_cuda_retry_requires_worker_exit_and_forbids_in_process_execution() -> None:
    from part1_runtime import FreshProcessRequired, WorkSpec, assert_in_process_retry_allowed, plan_retry

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    plan = plan_retry(work, category="transient_cuda_runtime_failure", attempts_consumed=1)
    assert plan.requires_fresh_process is True
    assert plan.terminate_current_process is True
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        assert_in_process_retry_allowed(plan)


def _work_specs():
    from part1_runtime import WorkSpec

    natural = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    checkpoints = [
        WorkSpec.checkpoint(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, cp, seed=123)
        for cp in ("cp-00", "cp-05")
    ]
    return natural, checkpoints


def test_resume_keeps_successful_natural_independent_from_missing_checkpoints(tmp_path: Path) -> None:
    from part1_runtime import plan_resume

    shard = _acquire(tmp_path).store
    result = natural_result()
    shard.append_audit_event(attempt_event(result, "attempt_started", 0))
    shard.commit_terminal_result(result, attempt_event(result, "attempt_completed", 1))
    natural, checkpoints = _work_specs()

    first = plan_resume(shard.build_index(), [natural, *checkpoints])
    second = plan_resume(shard.build_index(), [natural, *checkpoints])
    assert first == second
    assert first[natural].status == "completed"
    assert all(first[item].status == "retryable" for item in checkpoints)
    assert first[natural].reason == "terminal_result_exists"


def test_checkpoint_resume_skips_only_completed_checkpoint_identity(tmp_path: Path) -> None:
    from part1_runtime import plan_resume

    session = _acquire(tmp_path)
    natural = natural_result()
    session.store.append_audit_event(attempt_event(natural, "attempt_started", 0))
    session.store.commit_terminal_result(natural, attempt_event(natural, "attempt_completed", 1))
    checkpoint = checkpoint_result(checkpoint_id="cp-05")
    session.store.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    session.store.commit_terminal_result(
        checkpoint, attempt_event(checkpoint, "attempt_completed", 1)
    )
    natural_work, checkpoints = _work_specs()
    decisions = plan_resume(session.store.build_index(), [natural_work, *checkpoints])
    assert decisions[checkpoints[0]].status == "retryable"
    assert decisions[checkpoints[1]].status == "completed"


def test_checkpoint_is_ineligible_without_complete_parent_natural(tmp_path: Path) -> None:
    from part1_runtime import plan_resume

    natural_work, checkpoints = _work_specs()
    empty = _acquire(tmp_path / "missing")
    missing = plan_resume(empty.store.build_index(), [natural_work, *checkpoints])
    assert all(missing[item].status == "ineligible" for item in checkpoints)

    failed = _acquire(tmp_path / "failed")
    for number in (1, 2, 3):
        attempt = natural_result(attempt_number=number)
        failed.store.append_audit_event(attempt_event(attempt, "attempt_started", 0))
        if number < 3:
            event = attempt_event(attempt, "attempt_failed", 1)
            event["outcome_category"] = "interrupted_process"
            event["retry_classification"] = "retryable"
            event["retry_decision"] = "retry"
            failed.store.append_audit_event(event)
    terminal = natural_result(attempt_number=3, outcome="terminal_infrastructure_failure")
    failed.store.commit_terminal_result(
        terminal, attempt_event(terminal, "attempt_completed", 1)
    )
    failed_plan = plan_resume(failed.store.build_index(), [natural_work, *checkpoints])
    assert failed_plan[natural_work].status == "terminal"
    assert all(failed_plan[item].status == "ineligible" for item in checkpoints)


def test_resume_reconciles_orphan_once_counts_attempt_and_is_idempotent(tmp_path: Path) -> None:
    from part1_runtime import prepare_resume

    session = _acquire(tmp_path)
    result = natural_result()
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    natural, _ = _work_specs()

    first = prepare_resume(
        session.store,
        [natural],
        event_timestamp="2026-07-31T00:03:00Z",
        execution_context={"hostname": "resume-node", "pid": 101},
    )
    assert first[natural].attempts_consumed == 1
    assert first[natural].status == "retryable"
    assert first[natural].next_attempt_number == 2
    assert [event["event_type"] for event in session.store.inspect().audit_events] == [
        "attempt_started",
        "attempt_interrupted",
    ]
    second = prepare_resume(
        session.store,
        [natural],
        event_timestamp="2026-07-31T00:04:00Z",
        execution_context={"hostname": "resume-node", "pid": 101},
    )
    assert second == first
    assert len(session.store.inspect().audit_events) == 2


def test_resume_classifies_terminal_failure_and_exhaustion(tmp_path: Path) -> None:
    from part1_runtime import WorkSpec, plan_resume

    session = _acquire(tmp_path)
    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    for number in (1, 2):
        record = natural_result(attempt_number=number)
        started = attempt_event(record, "attempt_started", 0)
        session.store.append_audit_event(started)
        failed = attempt_event(record, "attempt_failed", 1)
        failed["outcome_category"] = "interrupted_process"
        failed["retry_classification"] = "retryable"
        failed["retry_decision"] = "retry"
        session.store.append_audit_event(failed)
    third = natural_result(attempt_number=3)
    session.store.append_audit_event(attempt_event(third, "attempt_started", 0))
    session.store.append_audit_event(attempt_event(third, "attempt_interrupted", 1))
    decision = plan_resume(session.store.build_index(), [work])[work]
    assert decision.status == "terminalization_required"
    assert decision.reason == "terminal_result_required"


@pytest.mark.parametrize(
    ("category", "classification", "expected_status"),
    [
        ("temporary_filesystem_failure", "retryable", "retryable"),
        ("invalid_configuration", "terminal", "terminal"),
    ],
)
def test_resume_honors_failed_attempt_classification_before_exhaustion(
    tmp_path: Path, category: str, classification: str, expected_status: str
) -> None:
    from part1_runtime import WorkSpec, plan_resume

    session = _acquire(tmp_path)
    result = natural_result(
        outcome="terminal_infrastructure_failure" if classification == "terminal" else "complete"
    )
    if classification == "terminal":
        result["terminal_error_details"]["category"] = category
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    if classification == "terminal":
        session.store.commit_terminal_result(
            result, attempt_event(result, "attempt_completed", 1)
        )
        work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
        assert plan_resume(session.store.build_index(), [work])[work].status == expected_status
        return
    failed = attempt_event(result, "attempt_failed", 1)
    failed["outcome_category"] = category
    failed["retry_classification"] = classification
    failed["retry_decision"] = "retry" if classification == "retryable" else "do_not_retry"
    session.store.append_audit_event(failed)
    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    assert plan_resume(session.store.build_index(), [work])[work].status == expected_status


def test_resume_rejects_seed_or_logical_provenance_mismatch(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, plan_resume

    session = _acquire(tmp_path)
    result = natural_result()
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    session.store.commit_terminal_result(result, attempt_event(result, "attempt_completed", 1))
    wrong_seed = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=999)
    with pytest.raises(CompatibilityError, match="seed"):
        plan_resume(session.store.build_index(), [wrong_seed])

    wrong_checkpoint_seed = WorkSpec.checkpoint(
        STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, "cp-05", seed=999
    )
    with pytest.raises(CompatibilityError, match="seed"):
        plan_resume(session.store.build_index(), [wrong_checkpoint_seed])


def test_resume_rejects_requested_study_or_model_run_incompatible_with_shard(
    tmp_path: Path,
) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, plan_resume, prepare_resume

    session = _acquire(tmp_path)
    result = natural_result()
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    wrong = WorkSpec.natural(STUDY_ID, "0" * 64, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    with pytest.raises(CompatibilityError, match="study/model-run|shard provenance"):
        plan_resume(session.store.build_index(), [wrong])
    with pytest.raises(CompatibilityError, match="study/model-run|shard provenance"):
        prepare_resume(
            session.store,
            [wrong],
            event_timestamp="2026-07-31T00:05:00Z",
            execution_context={"hostname": "resume-node", "pid": 101},
        )


def test_resume_refuses_index_with_lifecycle_corruption(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, plan_resume

    session = _acquire(tmp_path)
    third = natural_result(attempt_number=3)
    session.store.root.mkdir(parents=True, exist_ok=True)
    session.store.audit_events_path.write_text(
        json.dumps(
            attempt_event(third, "attempt_started", 0),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, MODEL_RUN_MANIFEST_HASH, QUESTION_ID, 0, seed=123)
    with pytest.raises(CompatibilityError, match="lifecycle"):
        plan_resume(session.store.build_index(), [work])


def _compatible_manifests() -> tuple[dict, dict]:
    from part1_contract import (
        FIXED_MODEL_REQUESTED_CONTRACT,
        FIXED_STUDY_CONTRACT,
        model_run_id,
        model_run_manifest_hash,
        study_id,
        study_manifest_hash,
    )

    subjects = [
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "high_school_psychology",
    ]
    study = {
        "schema_name": "part1_study_manifest",
        "schema_version": "1.0.0",
        "question_manifest_hash": "e" * 64,
        **FIXED_STUDY_CONTRACT,
    }
    study["study_id"] = study_id(study)
    study["study_manifest_hash"] = study_manifest_hash(study)
    model = {
        "schema_name": "part1_model_run_manifest",
        "schema_version": "1.0.0",
        "study_id": study["study_id"],
        "study_manifest_hash": study["study_manifest_hash"],
        "question_manifest_hash": study["question_manifest_hash"],
        **FIXED_MODEL_REQUESTED_CONTRACT,
        "model_revision": "model-commit",
        "tokenizer_revision": "tokenizer-commit",
        "canonical_model_identity": "hf:HuggingFaceTB/SmolLM3-3B@model-commit",
        "adapter_version": "smollm3-v1",
        "prompt_version": "part1-prompt-v1",
        "prompt_hash": "d" * 64,
        "parser_version": "part1-parser-v1",
        "inducer_version": "part1-inducer-v1",
        "inducer_text": "</think>\nAnswer:",
        "inducer_token_ids": [1, 2],
        "reasoning_open_tag": "<think>",
        "reasoning_open_token_ids": [3],
        "reasoning_close_tag": "</think>",
        "reasoning_close_token_ids": [4],
        "effective_natural_generation": dict(
            FIXED_MODEL_REQUESTED_CONTRACT["requested_natural_generation"]
        ),
        "effective_checkpoint_generation": dict(
            FIXED_MODEL_REQUESTED_CONTRACT["requested_checkpoint_generation"]
        ),
        "ad_token_convention": "single-token",
        "ad_raw_token_sequences": {"A": [65], "B": [66], "C": [67], "D": [68]},
        "ad_token_ids": [65, 66, 67, 68],
        "environment_versions": {"python": "3.12"},
        "final_production_git_commit": None,
        "production": False,
        "smoke_git_provenance": {"base_commit": "e" * 40, "diff_hash": "1" * 64},
    }
    model["model_run_id"] = model_run_id(model)
    model["model_run_manifest_hash"] = model_run_manifest_hash(model)
    return study, model


def test_manifest_compatibility_rejects_hash_schema_and_hierarchy_mismatches() -> None:
    from part1_runtime import CompatibilityError, validate_manifest_compatibility

    study, model = _compatible_manifests()
    result = validate_manifest_compatibility(study, model)
    assert result["study_id"] == study["study_id"]

    mismatched = dict(model, question_manifest_hash="0" * 64)
    with pytest.raises(CompatibilityError, match="question_manifest_hash"):
        validate_manifest_compatibility(study, mismatched)
    bad_hash = dict(model, model_run_manifest_hash="0" * 64)
    with pytest.raises(CompatibilityError, match="model_run_manifest_hash"):
        validate_manifest_compatibility(study, bad_hash)
    incompatible = dict(study, compatible_raw_record_schema_versions=["2.0.0"])
    incompatible["study_id"] = study["study_id"]
    incompatible["study_manifest_hash"] = study["study_manifest_hash"]
    with pytest.raises(CompatibilityError, match="raw record schema"):
        validate_manifest_compatibility(incompatible, model)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("study", "scientific_protocol_version", "drifted"),
        ("study", "primary_auroc_feature_registry", ["one-feature"]),
        ("model", "model_repository", "other/model"),
        ("model", "base_generation_seed", 7),
        ("model", "requested_natural_generation", {"do_sample": True}),
        ("model", "requested_checkpoint_generation", {"do_sample": False}),
    ],
)
def test_manifest_compatibility_rejects_fixed_contract_drift(
    target: str, field: str, value
) -> None:
    from part1_contract import model_run_id, model_run_manifest_hash, study_id, study_manifest_hash
    from part1_runtime import CompatibilityError, validate_manifest_compatibility

    study, model = _compatible_manifests()
    if target == "study":
        study[field] = value
        study["study_id"] = study_id(study)
        study["study_manifest_hash"] = study_manifest_hash(study)
        model.update(
            study_id=study["study_id"],
            study_manifest_hash=study["study_manifest_hash"],
        )
    else:
        model[field] = value
    model["model_run_id"] = model_run_id(model)
    model["model_run_manifest_hash"] = model_run_manifest_hash(model)
    with pytest.raises(CompatibilityError, match="fixed Part 1|requested"):
        validate_manifest_compatibility(study, model)


def test_dry_run_default_and_templates_are_read_only_and_login_safe(tmp_path: Path) -> None:
    from part1_runtime import run_dry_run

    output_root = tmp_path / "must-not-be-created"
    report = run_dry_run(mode="smoke", persistent_root=output_root, allow_root_override=True)
    assert report["is_valid"] is True
    assert report["would_create_production_manifest"] is False
    assert report["imports_model_or_data_libraries"] is False
    assert not output_root.exists()

    study, model = _compatible_manifests()
    nonexistent_shard = tmp_path / "unmaterialized-shard"
    shard_report = run_dry_run(
        mode="smoke",
        persistent_root=output_root,
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=nonexistent_shard,
    )
    assert shard_report["is_valid"] is False
    assert shard_report["shard_plan"]["mutation_performed"] is False
    assert not nonexistent_shard.exists()


def test_smoke_and_production_paths_are_separate_and_production_fails_closed(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, run_dry_run

    with pytest.raises(ValueError, match="production execution is forbidden"):
        run_dry_run(mode="production", persistent_root=tmp_path / "production")
    with pytest.raises(CompatibilityError, match="separate"):
        run_dry_run(
            mode="smoke",
            persistent_root=tmp_path / "same",
            smoke_root=tmp_path / "same",
            production_root=tmp_path / "same",
            allow_root_override=True,
        )


def test_dry_run_cli_help_and_default_template_validation_do_not_create_outputs() -> None:
    repository = Path(__file__).resolve().parents[1]
    output_root = repository / "results" / "part1-smoke"
    existed_before = output_root.exists()
    help_result = subprocess.run(
        [sys.executable, "scripts/part1_dry_run.py", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    default_result = subprocess.run(
        [sys.executable, "scripts/part1_dry_run.py"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert default_result.returncode == 0, default_result.stderr
    assert json.loads(default_result.stdout)["is_valid"] is True
    assert output_root.exists() is existed_before


def test_operator_unlock_cli_requires_reason_and_uses_no_model_or_data_imports(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    old = _acquire(tmp_path, pid=101)
    base = [
        sys.executable,
        "scripts/part1_operator_unlock.py",
        "--shard-root",
        str(old.store.root),
        "--study-id",
        STUDY_ID,
        "--model-run-id",
        MODEL_RUN_ID,
        "--model-run-manifest-hash",
        MODEL_RUN_MANIFEST_HASH,
        "--shard-id",
        SHARD_ID,
    ]
    missing = subprocess.run(base, cwd=repository, capture_output=True, text=True, check=False)
    assert missing.returncode != 0
    assert "--reason" in missing.stderr
    unlocked = subprocess.run(
        [*base, "--reason", "operator verified allocation ended"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unlocked.returncode == 0, unlocked.stderr
    payload = json.loads(unlocked.stdout)
    assert payload["event_type"] == "operator_unlock"
    source = (repository / "scripts" / "part1_operator_unlock.py").read_text(encoding="utf-8")
    for forbidden in ("torch", "transformers", "datasets", "AutoModel", "AutoTokenizer"):
        assert forbidden not in source


def test_mutation_and_takeover_share_one_cross_process_critical_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_runtime import Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    result = natural_result()
    entered = threading.Event()
    release = threading.Event()
    original = old.store._durable_append

    def paused_append(path: Path, payload: bytes) -> None:
        entered.set()
        assert release.wait(5)
        original(path, payload)

    monkeypatch.setattr(old.store, "_durable_append", paused_append)
    append_thread = threading.Thread(
        target=lambda: old.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    )
    append_thread.start()
    assert entered.wait(5)
    takeover_result: list = []

    def takeover() -> None:
        takeover_result.append(
            LockedShardSession.recover_stale(
                old.store.root,
                owner=_owner(pid=202),
                model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
                worker_liveness=lambda metadata: Liveness.DEAD,
                slurm_liveness=lambda metadata: Liveness.DEAD,
                event_timestamp="2026-07-31T00:10:00Z",
                execution_context={"hostname": "node", "pid": 202},
            )
        )

    takeover_thread = threading.Thread(target=takeover)
    takeover_thread.start()
    time.sleep(0.05)
    assert takeover_result == []
    release.set()
    append_thread.join(5)
    takeover_thread.join(5)
    assert len(takeover_result) == 1
    assert len(takeover_result[0].store.inspect().audit_events) == 2
    takeover_result[0].close()


def test_close_cannot_remove_replacement_lock_during_race(tmp_path: Path) -> None:
    from part1_runtime import Liveness, LockedShardSession, LostLockOwnershipError

    old = _acquire(tmp_path, pid=101)
    replacement = LockedShardSession.recover_stale(
        old.store.root,
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        worker_liveness=lambda metadata: Liveness.DEAD,
        slurm_liveness=lambda metadata: Liveness.DEAD,
        event_timestamp="2026-07-31T00:10:00Z",
        execution_context={"hostname": "node", "pid": 202},
    )
    with pytest.raises(LostLockOwnershipError):
        old.close()
    replacement.assert_owned()
    replacement.close()


def test_takeover_waits_for_full_trailing_recovery_critical_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_runtime import Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    old.store.natural_results_path.write_bytes(b'{"cut"')
    entered = threading.Event()
    release = threading.Event()
    original = old.store._durable_append

    def paused_append(path: Path, payload: bytes) -> None:
        if path == old.store.audit_events_path:
            entered.set()
            assert release.wait(5)
        original(path, payload)

    monkeypatch.setattr(old.store, "_durable_append", paused_append)
    recovery = threading.Thread(
        target=lambda: old.store.recover_trailing_line(
            "natural_results",
            event_sequence=0,
            event_timestamp="2026-07-31T00:09:00Z",
            execution_context={},
        )
    )
    recovery.start()
    assert entered.wait(5)
    replacements: list = []
    takeover = threading.Thread(
        target=lambda: replacements.append(
            LockedShardSession.recover_stale(
                old.store.root,
                owner=_owner(pid=202),
                model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
                worker_liveness=lambda metadata: Liveness.DEAD,
                slurm_liveness=lambda metadata: Liveness.DEAD,
                event_timestamp="2026-07-31T00:10:00Z",
                execution_context={},
            )
        )
    )
    takeover.start()
    time.sleep(0.05)
    assert replacements == []
    release.set()
    recovery.join(5)
    takeover.join(5)
    assert len(replacements) == 1
    replacements[0].close()


@pytest.mark.parametrize("operation", ["report", "finalize"])
def test_takeover_serializes_report_and_finalization_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from part1_runtime import FinalizedRuntimeShardError, Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    entered = threading.Event()
    release = threading.Event()
    original = old.store.validate_shard

    def paused_validation(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(old.store, "validate_shard", paused_validation)
    mutation_errors: list[Exception] = []

    def mutate() -> None:
        try:
            if operation == "report":
                old.store.write_validation_report(
                    tmp_path / "report.json",
                    artifact_kind="natural_shard",
                    started_at="2026-07-31T00:00:00Z",
                    completed_at="2026-07-31T00:00:01Z",
                )
            else:
                old.store.finalize()
        except Exception as exc:  # pragma: no cover - asserted below
            mutation_errors.append(exc)

    mutation = threading.Thread(target=mutate)
    mutation.start()
    assert entered.wait(5)
    replacements: list = []
    takeover_errors: list[Exception] = []

    def takeover() -> None:
        try:
            replacements.append(
                LockedShardSession.recover_stale(
                    old.store.root,
                    owner=_owner(pid=202),
                    model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
                    worker_liveness=lambda metadata: Liveness.DEAD,
                    slurm_liveness=lambda metadata: Liveness.DEAD,
                    event_timestamp="2026-07-31T00:10:00Z",
                    execution_context={},
                )
            )
        except Exception as exc:
            takeover_errors.append(exc)

    takeover_thread = threading.Thread(target=takeover)
    takeover_thread.start()
    time.sleep(0.05)
    assert replacements == [] and takeover_errors == []
    release.set()
    mutation.join(5)
    takeover_thread.join(5)
    assert mutation_errors == []
    if operation == "report":
        assert len(replacements) == 1
        replacements[0].close()
    else:
        assert len(takeover_errors) == 1
        assert isinstance(takeover_errors[0], FinalizedRuntimeShardError)
        old.close()


@pytest.mark.parametrize(
    "boundary",
    [
        "after_claim_creation_before_liveness",
        "after_verified_evidence_before_replacement",
        "after_lock_replacement_before_event",
        "after_event_before_claim_cleanup",
    ],
)
def test_every_takeover_claim_state_is_resumable(tmp_path: Path, boundary: str) -> None:
    from part1_runtime import InjectedTakeoverCrash, Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash, match=boundary):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:11:00Z",
            execution_context={"hostname": "node", "pid": 202},
            fault_at=boundary,
        )
    completed = LockedShardSession.finish_pending_takeover(
        old.store.root,
        worker_liveness=lambda metadata: Liveness.DEAD,
        slurm_liveness=lambda metadata: Liveness.DEAD,
    )
    takeover_events = [
        event
        for event in completed.store.inspect().audit_events
        if event["event_type"] == "stale_lock_recovered"
    ]
    assert len(takeover_events) == 1
    history = completed.store.root / ".lock_history"
    assert list(history.glob("*.claim.json"))
    assert list(history.glob("*.event.json"))
    completed.close()


def test_atomic_exclusive_create_never_publishes_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_runtime import _exclusive_create

    target = tmp_path / "claim.json"
    payload = b'{"complete":true}'

    def crash_before_publish(source: Path, destination: Path) -> None:
        assert destination == target
        assert Path(source).read_bytes() == payload
        raise OSError("synthetic publish crash")

    monkeypatch.setattr("part1_runtime.os.link", crash_before_publish)
    with pytest.raises(OSError, match="synthetic publish crash"):
        _exclusive_create(target, payload)
    assert not target.exists()
    orphan_temps = list(tmp_path.glob(".claim.json.*.tmp"))
    assert orphan_temps and all(path.read_bytes() == payload for path in orphan_temps)


def test_pending_replacement_file_is_reused_after_crash(tmp_path: Path) -> None:
    from part1_runtime import InjectedTakeoverCrash, Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash, match="after_pending"):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:11:00Z",
            execution_context={},
            fault_at="after_pending_replacement_durable_before_replace",
        )
    assert list(old.store.root.glob(".*.pending"))
    completed = LockedShardSession.finish_pending_takeover(
        old.store.root,
        worker_liveness=lambda metadata: Liveness.DEAD,
        slurm_liveness=lambda metadata: Liveness.DEAD,
    )
    assert completed.store.inspect().audit_events[-1]["event_type"] == "stale_lock_recovered"
    completed.close()


def test_corrupt_pending_replacement_requires_operator_quarantine(
    tmp_path: Path,
) -> None:
    from part1_runtime import (
        InjectedTakeoverCrash,
        Liveness,
        LockedShardSession,
        Part1RuntimeError,
    )

    old = _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:11:00Z",
            execution_context={},
            fault_at="after_pending_replacement_durable_before_replace",
        )
    pending = next(old.store.root.glob(".*.pending"))
    pending.write_bytes(b"partial-conflict")
    with pytest.raises(Part1RuntimeError, match="pending replacement"):
        LockedShardSession.finish_pending_takeover(
            old.store.root,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
        )
    assert pending.read_bytes() == b"partial-conflict"
    completed = LockedShardSession.finish_pending_takeover(
        old.store.root,
        operator_override_reason="operator quarantined conflicting pending bytes",
    )
    event = completed.store.inspect().audit_events[-1]
    assert event["event_type"] == "operator_unlock"
    assert list((old.store.root / ".lock_history").glob("*.pending-quarantine"))
    completed.close()


def test_ordinary_post_replacement_error_preserves_resumable_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from part1_runtime import Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    original = Part1ShardStore.finish_pending_recoveries
    calls = 0

    def fail_once(store):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ordinary post-replacement validation error")
        return original(store)

    monkeypatch.setattr(Part1ShardStore, "finish_pending_recoveries", fail_once)
    with pytest.raises(RuntimeError, match="ordinary post-replacement"):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:11:00Z",
            execution_context={},
        )
    assert (old.store.root / ".writer-lock-recovery.claim").exists()
    completed = LockedShardSession.finish_pending_takeover(old.store.root)
    events = [
        event
        for event in completed.store.inspect().audit_events
        if event["event_type"] == "stale_lock_recovered"
    ]
    assert len(events) == 1
    completed.close()


def test_unjournaled_raw_tail_is_refused_before_irreversible_replacement(
    tmp_path: Path,
) -> None:
    from part1_runtime import Liveness, LockedShardSession, Part1RuntimeError

    old = _acquire(tmp_path, pid=101)
    old.store.natural_results_path.write_bytes(b'{"partial"')
    with pytest.raises(Part1RuntimeError, match="lack durable recovery evidence"):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:11:00Z",
            execution_context={},
        )
    old.assert_owned()
    assert not (old.store.root / ".writer-lock-recovery.claim").exists()
    assert not list(old.store.root.glob(".*.pending"))
    old.close()


def test_pending_stale_claim_can_be_operator_overridden_with_new_reason(tmp_path: Path) -> None:
    from part1_runtime import InjectedTakeoverCrash, Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:11:00Z",
            execution_context={"hostname": "node", "pid": 202},
            fault_at="after_claim_creation_before_liveness",
        )
    completed = LockedShardSession.finish_pending_takeover(
        old.store.root,
        operator_override_reason="operator confirmed allocation ended",
    )
    event = completed.store.inspect().audit_events[-1]
    assert event["event_type"] == "operator_unlock"
    assert event["operator_reason"] == "operator confirmed allocation ended"
    completed.close()


def test_remote_slurm_dead_is_conclusive_without_remote_pid_probe(tmp_path: Path) -> None:
    from part1_runtime import Liveness, LockedShardSession

    old = _acquire(tmp_path, pid=101)
    replacement = LockedShardSession.recover_stale(
        old.store.root,
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        worker_liveness=lambda metadata: Liveness.UNKNOWN,
        slurm_liveness=lambda metadata: Liveness.DEAD,
        event_timestamp="2026-07-31T00:12:00Z",
        execution_context={"hostname": "other", "pid": 202},
    )
    replacement.close()


@pytest.mark.parametrize(
    ("with_slurm", "hostname_is_local", "expected_success"),
    [
        (True, True, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_pid_dead_only_establishes_staleness_for_same_host(
    tmp_path: Path,
    with_slurm: bool,
    hostname_is_local: bool,
    expected_success: bool,
) -> None:
    import socket

    from part1_runtime import (
        Liveness,
        LockMetadata,
        LockedShardSession,
        StaleRecoveryRefused,
    )

    prior = LockMetadata.new(
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        shard_id=SHARD_ID,
        hostname=socket.gethostname() if hostname_is_local else "remote-node",
        pid=999999,
        slurm_job_id="job-17" if with_slurm else None,
        slurm_array_task_id=None,
        acquired_at="2026-07-31T00:00:00Z",
    )
    old = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=prior,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )
    call = lambda: LockedShardSession.recover_stale(
        old.store.root,
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        worker_liveness=lambda metadata: Liveness.DEAD,
        slurm_liveness=lambda metadata: (
            Liveness.UNKNOWN if with_slurm else Liveness.NOT_APPLICABLE
        ),
        event_timestamp="2026-07-31T00:12:00Z",
        execution_context={},
    )
    if expected_success:
        call().close()
    else:
        with pytest.raises(StaleRecoveryRefused, match="uncertain"):
            call()
        old.close()


def test_attempt_failed_rejects_terminal_or_exhausted_policy(tmp_path: Path) -> None:
    session = _acquire(tmp_path)
    result = natural_result()
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    failed = attempt_event(result, "attempt_failed", 1)
    failed.update(
        outcome_category="invalid_configuration",
        retry_classification="terminal",
        retry_decision="do_not_retry",
        backoff_seconds=None,
    )
    with pytest.raises(ValueError, match="terminalization|required terminal"):
        session.store.append_audit_event(failed)


def test_final_orphan_requires_terminalization_until_result_exists(tmp_path: Path) -> None:
    from part1_runtime import WorkSpec, prepare_resume

    session = _acquire(tmp_path)
    for number in (1, 2):
        attempt = natural_result(attempt_number=number)
        session.store.append_audit_event(attempt_event(attempt, "attempt_started", 0))
        session.store.append_audit_event(attempt_event(attempt, "attempt_failed", 1))
    third = natural_result(attempt_number=3)
    session.store.append_audit_event(attempt_event(third, "attempt_started", 0))
    work = WorkSpec.natural(
        STUDY_ID,
        MODEL_RUN_ID,
        MODEL_RUN_MANIFEST_HASH,
        QUESTION_ID,
        0,
        seed=123,
    )
    decision = prepare_resume(
        session.store,
        [work],
        event_timestamp="2026-07-31T00:13:00Z",
        execution_context={"hostname": "node", "pid": 101},
    )[work]
    assert decision.status == "terminalization_required"
    assert decision.terminalization_required is True
    assert decision.failure_category == "interrupted_process"
    report = session.store.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )
    assert report["is_valid"] is False


def test_exhausted_interruption_can_be_terminalized_exactly_once(tmp_path: Path) -> None:
    session = _acquire(tmp_path)
    for number in (1, 2):
        record = natural_result(attempt_number=number)
        session.store.append_audit_event(attempt_event(record, "attempt_started", 0))
        session.store.append_audit_event(attempt_event(record, "attempt_failed", 1))
    terminal = natural_result(
        attempt_number=3, outcome="terminal_infrastructure_failure"
    )
    terminal["terminal_error_details"]["category"] = "interrupted_process"
    session.store.append_audit_event(attempt_event(terminal, "attempt_started", 0))
    session.store.append_audit_event(attempt_event(terminal, "attempt_interrupted", 1))
    session.store.commit_terminal_result(
        terminal, attempt_event(terminal, "attempt_completed", 2)
    )
    assert session.store.validate_shard(
        artifact_kind="natural_shard",
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:00:01Z",
    )["is_valid"] is True
    with pytest.raises(Exception, match="already has a terminal result"):
        session.store.append_terminal_result(terminal)


def test_checkpoint_publication_requires_complete_matching_parent(tmp_path: Path) -> None:
    session = _acquire(tmp_path)
    checkpoint = checkpoint_result()
    session.store.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    with pytest.raises(ValueError, match="parent natural"):
        session.store.append_terminal_result(checkpoint)

    parent = natural_result()
    other = _acquire(tmp_path / "valid-parent")
    other.store.append_audit_event(attempt_event(parent, "attempt_started", 0))
    other.store.commit_terminal_result(parent, attempt_event(parent, "attempt_completed", 1))
    wrong = checkpoint_result()
    wrong["natural_seed"] = 999
    other.store.append_audit_event(attempt_event(wrong, "attempt_started", 0))
    with pytest.raises(ValueError, match="seed"):
        other.store.append_terminal_result(wrong)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_raw_record_id", "0" * 64),
        ("sample_index", 1),
        ("subject", "high_school_physics"),
        ("question_manifest_hash", "0" * 64),
    ],
)
def test_checkpoint_parent_requires_complete_matching_hierarchy(
    tmp_path: Path, field: str, value
) -> None:
    session = _acquire(tmp_path)
    parent = natural_result()
    session.store.append_audit_event(attempt_event(parent, "attempt_started", 0))
    session.store.commit_terminal_result(
        parent, attempt_event(parent, "attempt_completed", 1)
    )
    checkpoint = checkpoint_result()
    checkpoint[field] = value
    session.store.append_audit_event(attempt_event(checkpoint, "attempt_started", 0))
    with pytest.raises(ValueError, match="parent|differs"):
        session.store.append_terminal_result(checkpoint)


def test_handwritten_orphan_checkpoint_fails_index_and_resume(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, plan_resume

    session = _acquire(tmp_path)
    checkpoint = checkpoint_result()
    session.store.checkpoint_results_path.write_text(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    index = session.store.build_index()
    assert index.hierarchy_errors
    work = WorkSpec.checkpoint(
        STUDY_ID,
        MODEL_RUN_ID,
        MODEL_RUN_MANIFEST_HASH,
        QUESTION_ID,
        0,
        "cp-05",
        seed=123,
    )
    with pytest.raises(CompatibilityError, match="hierarchy"):
        plan_resume(index, [work])


def test_shard_provenance_header_binds_complete_manifest_hash(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, LockedShardSession

    first = _acquire(tmp_path)
    header_path = first.store.provenance_header_path
    header = json.loads(header_path.read_text(encoding="utf-8"))
    assert header["model_run_manifest_hash"] == MODEL_RUN_MANIFEST_HASH
    first.close()
    reopened = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=_owner(pid=202),
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )
    reopened.close()
    with pytest.raises(CompatibilityError, match="provenance|manifest hash"):
        LockedShardSession.acquire(
            tmp_path / "shard",
            owner=_owner(pid=303),
            model_run_manifest_hash="0" * 64,
        )


def test_missing_or_corrupt_provenance_header_blocks_all_store_use(tmp_path: Path) -> None:
    bare = Part1ShardStore(
        tmp_path / "bare",
        shard_id=SHARD_ID,
        study_id=STUDY_ID,
        model_run_id=MODEL_RUN_ID,
        model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
    )
    with pytest.raises(Exception, match="provenance header is missing"):
        bare.append_audit_event(attempt_event(natural_result(), "attempt_started", 0))
    assert not bare.root.exists()

    session = _acquire(tmp_path / "corrupt")
    session.close()
    session.store.provenance_header_path.write_text("{cut", encoding="utf-8")
    from part1_runtime import CompatibilityError, LockedShardSession

    with pytest.raises(CompatibilityError, match="provenance"):
        LockedShardSession.acquire(
            session.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lock_id", 7),
        ("study_id", ""),
        ("model_run_id", None),
        ("shard_id", []),
        ("hostname", " "),
        ("pid", True),
        ("slurm_job_id", 3),
        ("slurm_array_task_id", ""),
        ("acquired_at", "not-a-time"),
    ],
)
def test_lock_metadata_rejects_invalid_types_and_values(field: str, value) -> None:
    metadata = _owner(pid=101).to_dict()
    metadata[field] = value
    from part1_runtime import LockMetadata

    with pytest.raises((TypeError, ValueError), match=field):
        LockMetadata.from_mapping(metadata)


def test_work_spec_complete_manifest_hash_is_resume_identity(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, prepare_resume

    session = _acquire(tmp_path)
    wrong = WorkSpec.natural(
        STUDY_ID,
        MODEL_RUN_ID,
        "0" * 64,
        QUESTION_ID,
        0,
        seed=123,
    )
    with pytest.raises(CompatibilityError, match="manifest hash"):
        prepare_resume(
            session.store,
            [wrong],
            event_timestamp="2026-07-31T00:00:00Z",
            execution_context={},
        )


def test_complete_retry_config_and_backoff_are_enforced() -> None:
    from part1_contract import load_config
    from part1_runtime import CompatibilityError, WorkSpec, plan_retry, validate_retry_policy_config

    config = load_config("retries")
    validate_retry_policy_config(config)
    for field in (
        "config_version",
        "attempt_numbers",
        "cuda_retry_requires_fresh_process",
        "preserve_seed_and_logical_identity",
        "backoff_seconds",
    ):
        changed = dict(config)
        changed[field] = None
        with pytest.raises(CompatibilityError, match=field):
            validate_retry_policy_config(changed)
    work = WorkSpec.natural(
        STUDY_ID,
        MODEL_RUN_ID,
        MODEL_RUN_MANIFEST_HASH,
        QUESTION_ID,
        0,
        seed=123,
    )
    assert plan_retry(work, category="interrupted_process", attempts_consumed=1).backoff_seconds == 30
    assert plan_retry(work, category="interrupted_process", attempts_consumed=2).backoff_seconds == 120
    assert plan_retry(work, category="interrupted_process", attempts_consumed=3).backoff_seconds is None


def test_operator_cli_finishes_pending_takeover_with_reason(tmp_path: Path) -> None:
    from part1_runtime import InjectedTakeoverCrash, Liveness, LockedShardSession

    repository = Path(__file__).resolve().parents[1]
    old = _acquire(tmp_path, pid=101)
    with pytest.raises(InjectedTakeoverCrash):
        LockedShardSession.recover_stale(
            old.store.root,
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
            worker_liveness=lambda metadata: Liveness.DEAD,
            slurm_liveness=lambda metadata: Liveness.DEAD,
            event_timestamp="2026-07-31T00:15:00Z",
            execution_context={},
            fault_at="after_claim_creation_before_liveness",
        )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/part1_operator_unlock.py",
            "--finish-pending",
            "--shard-root",
            str(old.store.root),
            "--reason",
            "operator verified stale allocation",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["event_type"] == "operator_unlock"


def test_dry_run_cli_reports_real_resume_and_fails_on_corrupt_shard(tmp_path: Path) -> None:
    from part1_runtime import LockMetadata, LockedShardSession, WorkSpec

    repository = Path(__file__).resolve().parents[1]
    study, model = _compatible_manifests()
    session = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=LockMetadata.new(
            study_id=study["study_id"],
            model_run_id=model["model_run_id"],
            shard_id=SHARD_ID,
            hostname="synthetic-node",
            pid=101,
            slurm_job_id="job-17",
            slurm_array_task_id="3",
            acquired_at="2026-07-31T00:00:00Z",
        ),
        model_run_manifest_hash=model["model_run_manifest_hash"],
    )
    session.close()
    study_path = tmp_path / "study.json"
    model_path = tmp_path / "model.json"
    work_path = tmp_path / "work.json"
    retry_path = tmp_path / "retry.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    model_path.write_text(json.dumps(model), encoding="utf-8")
    work = WorkSpec.natural(
        study["study_id"],
        model["model_run_id"],
        model["model_run_manifest_hash"],
        QUESTION_ID,
        0,
        seed=123,
    )
    work_path.write_text(json.dumps([work.to_dict()]), encoding="utf-8")
    retry_path.write_text(
        json.dumps(
            {
                "work": work.to_dict(),
                "category": "temporary_filesystem_failure",
                "attempts_consumed": 0,
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "scripts/part1_dry_run.py",
        "--persistent-root",
        str(tmp_path / "configured-smoke"),
        "--study-manifest",
        str(study_path),
        "--model-run-manifest",
        str(model_path),
        "--shard-root",
        str(session.store.root),
        "--work-specs",
        str(work_path),
        "--retry-request",
        str(retry_path),
    ]
    valid = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert valid.returncode == 0, valid.stderr
    payload = json.loads(valid.stdout)
    assert payload["resume_plan"][0]["status"] == "retryable"
    assert payload["is_valid"] is True
    assert payload["retry_plan"]["backoff_seconds"] == 0

    incompatible_work = WorkSpec.natural(
        "0" * 64,
        "1" * 64,
        model["model_run_manifest_hash"],
        QUESTION_ID,
        0,
        seed=123,
    )
    work_path.write_text(json.dumps([incompatible_work.to_dict()]), encoding="utf-8")
    incompatible = subprocess.run(
        command, cwd=repository, capture_output=True, text=True, check=False
    )
    assert incompatible.returncode != 0
    assert json.loads(incompatible.stdout)["resume_plan"][0]["status"] == "ineligible"
    work_path.write_text(json.dumps([work.to_dict()]), encoding="utf-8")

    retry_path.write_text(
        json.dumps(
            {
                "work": work.to_dict(),
                "category": "invalid_configuration",
                "attempts_consumed": 0,
            }
        ),
        encoding="utf-8",
    )
    terminal_retry = subprocess.run(
        command, cwd=repository, capture_output=True, text=True, check=False
    )
    assert terminal_retry.returncode != 0
    assert json.loads(terminal_retry.stdout)["retry_plan"]["eligible"] is False
    retry_path.write_text(
        json.dumps(
            {
                "work": work.to_dict(),
                "category": "temporary_filesystem_failure",
                "attempts_consumed": 0,
            }
        ),
        encoding="utf-8",
    )
    session.store.audit_events_path.write_bytes(b'{"cut"')
    corrupt = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert corrupt.returncode != 0
    assert json.loads(corrupt.stdout)["is_valid"] is False


def test_dry_run_rejects_work_or_retry_without_manifest_bound_shard(
    tmp_path: Path,
) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, run_dry_run

    work = WorkSpec.natural(
        STUDY_ID,
        MODEL_RUN_ID,
        MODEL_RUN_MANIFEST_HASH,
        QUESTION_ID,
        0,
        seed=123,
    )
    with pytest.raises(CompatibilityError, match="work.*manifests.*shard"):
        run_dry_run(
            persistent_root=tmp_path / "smoke",
            allow_root_override=True,
            work_items=[work],
        )


def test_dry_run_binds_empty_shard_work_to_manifest_identity(tmp_path: Path) -> None:
    from part1_runtime import LockMetadata, LockedShardSession, WorkSpec, run_dry_run

    study, model = _compatible_manifests()
    session = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=LockMetadata.new(
            study_id=study["study_id"],
            model_run_id=model["model_run_id"],
            shard_id=SHARD_ID,
            hostname="node",
            pid=101,
            slurm_job_id=None,
            slurm_array_task_id=None,
            acquired_at="2026-07-31T00:00:00Z",
        ),
        model_run_manifest_hash=model["model_run_manifest_hash"],
    )
    session.close()
    wrong = WorkSpec.natural(
        "0" * 64,
        "1" * 64,
        model["model_run_manifest_hash"],
        QUESTION_ID,
        0,
        seed=123,
    )
    report = run_dry_run(
        persistent_root=tmp_path / "smoke",
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=session.store.root,
        work_items=[wrong],
    )
    assert report["is_valid"] is False
    assert report["resume_plan"][0]["status"] == "ineligible"


def test_dry_run_retry_requires_retry_decision_and_exact_persisted_attempt_count(
    tmp_path: Path,
) -> None:
    from part1_runtime import (
        CompatibilityError,
        LockMetadata,
        LockedShardSession,
        WorkSpec,
        run_dry_run,
    )

    study, model = _compatible_manifests()
    session = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=LockMetadata.new(
            study_id=study["study_id"],
            model_run_id=model["model_run_id"],
            shard_id=SHARD_ID,
            hostname="node",
            pid=101,
            slurm_job_id=None,
            slurm_array_task_id=None,
            acquired_at="2026-07-31T00:00:00Z",
        ),
        model_run_manifest_hash=model["model_run_manifest_hash"],
    )
    work = WorkSpec.natural(
        study["study_id"],
        model["model_run_id"],
        model["model_run_manifest_hash"],
        QUESTION_ID,
        0,
        seed=123,
    )
    active = run_dry_run(
        persistent_root=tmp_path / "smoke",
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=session.store.root,
        retry_request={
            "work": work.to_dict(),
            "category": "temporary_filesystem_failure",
            "attempts_consumed": 0,
        },
    )
    assert active["retry_plan"]["eligible"] is False
    assert active["is_valid"] is False
    session.close()
    common = dict(
        persistent_root=tmp_path / "smoke",
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=session.store.root,
    )
    terminal = run_dry_run(
        **common,
        retry_request={
            "work": work.to_dict(),
            "category": "invalid_configuration",
            "attempts_consumed": 0,
        },
    )
    assert terminal["retry_plan"]["decision"] == "do_not_retry"
    assert terminal["retry_plan"]["eligible"] is False
    assert terminal["is_valid"] is False
    wrong_count = run_dry_run(
        **common,
        retry_request={
            "work": work.to_dict(),
            "category": "temporary_filesystem_failure",
            "attempts_consumed": 1,
        },
    )
    assert wrong_count["retry_plan"]["eligible"] is False
    assert wrong_count["is_valid"] is False
    with pytest.raises(CompatibilityError, match="retry.*manifests.*shard"):
        run_dry_run(
            persistent_root=tmp_path / "smoke",
            allow_root_override=True,
            retry_request={
                "work": work.to_dict(),
                "category": "temporary_filesystem_failure",
                "attempts_consumed": 1,
            },
        )


def test_finalized_dry_run_keeps_completed_but_rejects_missing_work(
    tmp_path: Path,
) -> None:
    from part1_contract import attempt_id, natural_record_id
    from part1_runtime import LockMetadata, LockedShardSession, WorkSpec, run_dry_run

    study, model = _compatible_manifests()
    owner = LockMetadata.new(
        study_id=study["study_id"],
        model_run_id=model["model_run_id"],
        shard_id=SHARD_ID,
        hostname="synthetic-node",
        pid=101,
        slurm_job_id="job-17",
        slurm_array_task_id="3",
        acquired_at="2026-07-31T00:00:00Z",
    )
    session = LockedShardSession.acquire(
        tmp_path / "shard",
        owner=owner,
        model_run_manifest_hash=model["model_run_manifest_hash"],
    )
    record = natural_result()
    record.update(
        study_id=study["study_id"],
        model_run_id=model["model_run_id"],
        model_run_manifest_hash=model["model_run_manifest_hash"],
        question_manifest_hash=study["question_manifest_hash"],
    )
    record["raw_record_id"] = natural_record_id(
        record["study_id"],
        record["model_run_id"],
        record["question_id"],
        record["run_id"],
    )
    record["terminal_attempt_id"] = attempt_id(
        record["study_id"],
        record["model_run_id"],
        record["question_id"],
        record["run_id"],
        record["terminal_attempt_number"],
    )
    session.store.append_audit_event(attempt_event(record, "attempt_started", 0))
    session.store.commit_terminal_result(
        record, attempt_event(record, "attempt_completed", 1)
    )
    session.close()
    completed_work = WorkSpec.natural(
        record["study_id"],
        record["model_run_id"],
        record["model_run_manifest_hash"],
        record["question_id"],
        record["run_id"],
        seed=record["generation_seed"],
    )
    completed_retry = run_dry_run(
        persistent_root=tmp_path / "smoke",
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=session.store.root,
        retry_request={
            "work": completed_work.to_dict(),
            "category": "temporary_filesystem_failure",
            "attempts_consumed": 1,
        },
    )
    assert completed_retry["retry_plan"]["eligible"] is False
    assert completed_retry["is_valid"] is False
    session = LockedShardSession.acquire(
        session.store.root,
        owner=LockMetadata.new(
            study_id=study["study_id"],
            model_run_id=model["model_run_id"],
            shard_id=SHARD_ID,
            hostname="synthetic-node",
            pid=202,
            slurm_job_id="job-18",
            slurm_array_task_id="4",
            acquired_at="2026-07-31T00:01:00Z",
        ),
        model_run_manifest_hash=model["model_run_manifest_hash"],
    )
    session.store.finalize()
    session.close()
    completed = completed_work
    missing = WorkSpec.natural(
        record["study_id"],
        record["model_run_id"],
        record["model_run_manifest_hash"],
        "1" * 64,
        0,
        seed=456,
    )
    completed_only = run_dry_run(
        persistent_root=tmp_path / "smoke",
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=session.store.root,
        work_items=[completed],
    )
    assert completed_only["is_valid"] is True
    assert completed_only["resume_plan"][0]["status"] == "completed"
    report = run_dry_run(
        persistent_root=tmp_path / "smoke",
        allow_root_override=True,
        study_manifest=study,
        model_run_manifest=model,
        shard_root=session.store.root,
        work_items=[completed, missing],
    )
    decisions = {item["work"]["question_id"]: item for item in report["resume_plan"]}
    assert decisions[completed.question_id]["status"] == "completed"
    assert decisions[missing.question_id]["status"] == "ineligible"
    assert decisions[missing.question_id]["reason"] == "finalized_shard"
    assert report["is_valid"] is False
    study_path = tmp_path / "study.json"
    model_path = tmp_path / "model.json"
    work_path = tmp_path / "finalized-work.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    model_path.write_text(json.dumps(model), encoding="utf-8")
    work_path.write_text(json.dumps([missing.to_dict()]), encoding="utf-8")
    cli = subprocess.run(
        [
            sys.executable,
            "scripts/part1_dry_run.py",
            "--persistent-root",
            str(tmp_path / "smoke"),
            "--study-manifest",
            str(study_path),
            "--model-run-manifest",
            str(model_path),
            "--shard-root",
            str(session.store.root),
            "--work-specs",
            str(work_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode != 0
    assert json.loads(cli.stdout)["is_valid"] is False


def test_posix_flock_blocks_takeover_in_a_second_process(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    old = _acquire(tmp_path, pid=101)
    ready = tmp_path / "holder-ready"
    release = tmp_path / "holder-release"
    completed = tmp_path / "takeover-completed"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "scripts")
    holder_code = """
from pathlib import Path
import sys, time
from part1_runtime import MutationController
root, ready, release = map(Path, sys.argv[1:])
with MutationController(root).section():
    ready.write_text('ready', encoding='utf-8')
    deadline = time.monotonic() + 5
    while not release.exists():
        if time.monotonic() >= deadline:
            raise SystemExit(3)
        time.sleep(0.01)
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(old.store.root), str(ready), str(release)],
        cwd=repository,
        env=environment,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    takeover_code = """
from pathlib import Path
import sys
from part1_runtime import Liveness, LockMetadata, LockedShardSession
root, completed = map(Path, sys.argv[1:3])
study_id, model_run_id, manifest_hash, shard_id = sys.argv[3:7]
owner = LockMetadata.new(
    study_id=study_id, model_run_id=model_run_id, shard_id=shard_id,
    hostname='replacement-node', pid=202, slurm_job_id='job-18',
    slurm_array_task_id='4', acquired_at='2026-07-31T00:10:00Z')
session = LockedShardSession.recover_stale(
    root, owner=owner, model_run_manifest_hash=manifest_hash,
    worker_liveness=lambda metadata: Liveness.DEAD,
    slurm_liveness=lambda metadata: Liveness.DEAD,
    event_timestamp='2026-07-31T00:10:00Z', execution_context={})
completed.write_text('completed', encoding='utf-8')
session.close()
"""
    takeover = subprocess.Popen(
        [
            sys.executable,
            "-c",
            takeover_code,
            str(old.store.root),
            str(completed),
            STUDY_ID,
            MODEL_RUN_ID,
            MODEL_RUN_MANIFEST_HASH,
            SHARD_ID,
        ],
        cwd=repository,
        env=environment,
    )
    time.sleep(0.2)
    assert takeover.poll() is None
    assert not completed.exists()
    release.write_text("release", encoding="utf-8")
    assert holder.wait(timeout=5) == 0
    assert takeover.wait(timeout=5) == 0
    assert completed.read_text(encoding="utf-8") == "completed"
