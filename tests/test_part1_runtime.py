"""Synthetic, login-safe tests for Part 1 runtime safety and resumability."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    assert len(history) == 1
    assert json.loads(history[0].read_text(encoding="utf-8"))["previous_lock"]["pid"] == 101

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
        ("UNKNOWN", "DEAD", "uncertain"),
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
    from part1_runtime import LockedShardSession, StaleRecoveryRefused

    old = _acquire(tmp_path, pid=999999)
    with pytest.raises(StaleRecoveryRefused, match="uncertain"):
        LockedShardSession.recover_stale(
            tmp_path / "shard",
            owner=_owner(pid=202),
            model_run_manifest_hash=MODEL_RUN_MANIFEST_HASH,
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
        "--jobs",
        "job-17_3",
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

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
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

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
    plan = plan_retry(work, category=category, attempts_consumed=1)
    assert plan.classification == "terminal"
    assert plan.decision == "do_not_retry"
    assert plan.next_attempt_number is None


def test_attempt_limit_is_exactly_three_total_attempts() -> None:
    from part1_runtime import WorkSpec, plan_retry

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
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

    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
    plan = plan_retry(work, category="transient_cuda_runtime_failure", attempts_consumed=1)
    assert plan.requires_fresh_process is True
    assert plan.terminate_current_process is True
    with pytest.raises(FreshProcessRequired, match="fresh CUDA process"):
        assert_in_process_retry_allowed(plan)


def _work_specs():
    from part1_runtime import WorkSpec

    natural = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
    checkpoints = [
        WorkSpec.checkpoint(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, cp, seed=123)
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
    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
    for number in (1, 2, 3):
        record = natural_result(attempt_number=number)
        started = attempt_event(record, "attempt_started", 0)
        session.store.append_audit_event(started)
        failed = attempt_event(record, "attempt_failed", 1)
        failed["outcome_category"] = "interrupted_process"
        failed["retry_classification"] = "retryable"
        failed["retry_decision"] = "retry" if number < 3 else "exhausted"
        session.store.append_audit_event(failed)
    decision = plan_resume(session.store.build_index(), [work])[work]
    assert decision.status == "terminal"
    assert decision.reason == "attempts_exhausted"


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
    result = natural_result()
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    failed = attempt_event(result, "attempt_failed", 1)
    failed["outcome_category"] = category
    failed["retry_classification"] = classification
    failed["retry_decision"] = "retry" if classification == "retryable" else "do_not_retry"
    session.store.append_audit_event(failed)
    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
    assert plan_resume(session.store.build_index(), [work])[work].status == expected_status


def test_resume_rejects_seed_or_logical_provenance_mismatch(tmp_path: Path) -> None:
    from part1_runtime import CompatibilityError, WorkSpec, plan_resume

    session = _acquire(tmp_path)
    result = natural_result()
    session.store.append_audit_event(attempt_event(result, "attempt_started", 0))
    session.store.commit_terminal_result(result, attempt_event(result, "attempt_completed", 1))
    wrong_seed = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=999)
    with pytest.raises(CompatibilityError, match="seed"):
        plan_resume(session.store.build_index(), [wrong_seed])

    wrong_checkpoint_seed = WorkSpec.checkpoint(
        STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, "cp-05", seed=999
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
    wrong = WorkSpec.natural(STUDY_ID, "0" * 64, QUESTION_ID, 0, seed=123)
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
    work = WorkSpec.natural(STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, seed=123)
    with pytest.raises(CompatibilityError, match="lifecycle"):
        plan_resume(session.store.build_index(), [work])


def _compatible_manifests() -> tuple[dict, dict]:
    from part1_contract import model_run_id, model_run_manifest_hash, study_id, study_manifest_hash

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
        "subjects": subjects,
        "subject_quotas": {subject: 100 for subject in subjects},
        "question_sampling_seed": 42,
        "scientific_protocol_version": "part1-science-v1",
        "checkpoint_fractions": [index / 10 for index in range(11)],
        "checkpoint_placement_contract": {"version": "ties-even-v1"},
        "entropy_contract": {"version": "entropy-v1"},
        "natural_answer_validity_rule": {"version": "post-close-v1"},
        "status_contract_version": "part1-status-v1",
        "calibration_contract": {"version": "calibration-v1"},
        "bootstrap_contract": {"version": "bootstrap-v1"},
        "primary_auroc_feature_registry": ["negative_mean_reasoning_entropy"],
        "within_question_analysis": {"version": "paired-v1"},
        "switching_stabilization_contract": {"version": "trajectory-v1"},
        "repetition_policy": "preserve-successful-output",
        "compatible_raw_record_schema_versions": ["1.0.0"],
        "analysis_contract_version": "part1-analysis-v1",
    }
    study["study_id"] = study_id(study)
    study["study_manifest_hash"] = study_manifest_hash(study)
    model = {
        "schema_name": "part1_model_run_manifest",
        "schema_version": "1.0.0",
        "study_id": study["study_id"],
        "study_manifest_hash": study["study_manifest_hash"],
        "question_manifest_hash": study["question_manifest_hash"],
        "model_repository": "HuggingFaceTB/SmolLM3-3B",
        "model_revision": "model-commit",
        "tokenizer_repository": "HuggingFaceTB/SmolLM3-3B",
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
        "requested_natural_generation": {"do_sample": True},
        "effective_natural_generation": {"do_sample": True},
        "requested_checkpoint_generation": {"do_sample": False},
        "effective_checkpoint_generation": {"do_sample": False},
        "ad_token_convention": "single-token",
        "ad_raw_token_sequences": {"A": [65], "B": [66], "C": [67], "D": [68]},
        "ad_token_ids": [65, 66, 67, 68],
        "seed_algorithm_version": "part1-seed-v1",
        "base_generation_seed": 42,
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
    assert shard_report["shard_plan"]["validation"]["is_valid"] is True
    assert shard_report["shard_plan"]["completed_logical_keys"] == 0
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
