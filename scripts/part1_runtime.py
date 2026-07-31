"""Login-safe runtime controls for Part 1 shard execution.

This module deliberately imports no model, tokenizer, dataset, torch, or CUDA
libraries. It coordinates storage ownership, retry decisions, compatibility,
and resume planning using only persisted metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Callable, Iterable, Mapping
import uuid

from part1_contract import (
    CONFIG_NAMES,
    SCHEMA_NAMES,
    audit_event_id,
    checkpoint_record_id,
    load_config,
    load_schema,
    model_run_id,
    model_run_manifest_hash,
    natural_record_id,
    study_id,
    study_manifest_hash,
    validate_instance,
    validate_phase1_config,
)
from part1_failure_policy import (
    MAX_TOTAL_ATTEMPTS,
    RETRYABLE_CATEGORIES,
    TERMINAL_CATEGORIES,
)
from part1_store import CheckpointKey, LogicalKey, NaturalKey, Part1ShardStore, ShardIndex


LOCK_FILENAME = ".writer.lock"
RECOVERY_CLAIM_FILENAME = ".writer-lock-recovery.claim"
TAKEOVER_EVENT_JOURNAL_FILENAME = ".writer-lock-takeover-event.json"
LOCK_HISTORY_DIRECTORY = ".lock_history"


class Part1RuntimeError(RuntimeError):
    """Base class for fail-closed runtime contract violations."""


class LockHeldError(Part1RuntimeError):
    """A shard already has an exclusive owner."""


class LostLockOwnershipError(Part1RuntimeError):
    """A former writer no longer owns the shard lock."""


class StaleRecoveryRefused(Part1RuntimeError):
    """Automatic lock recovery lacked conclusive stale evidence."""


class FreshProcessRequired(Part1RuntimeError):
    """A CUDA retry was incorrectly requested inside the failed process."""


class CompatibilityError(Part1RuntimeError):
    """Persisted provenance is incompatible with requested work."""


class InjectedTakeoverCrash(Part1RuntimeError):
    """Synthetic crash after durable lock replacement."""


class FinalizedRuntimeShardError(Part1RuntimeError):
    """A runtime lock operation targeted an immutable shard."""


class Liveness(Enum):
    LIVE = "live"
    DEAD = "dead"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


LivenessProbe = Callable[["LockMetadata"], Liveness]


@dataclass(frozen=True)
class LockMetadata:
    lock_id: str
    study_id: str
    model_run_id: str
    shard_id: str
    hostname: str
    pid: int
    slurm_job_id: str | None
    slurm_array_task_id: str | None
    acquired_at: str

    @classmethod
    def new(
        cls,
        *,
        study_id: str,
        model_run_id: str,
        shard_id: str,
        hostname: str,
        pid: int,
        slurm_job_id: str | None,
        slurm_array_task_id: str | None,
        acquired_at: str,
    ) -> "LockMetadata":
        if pid < 1:
            raise ValueError("lock PID must be positive")
        if not shard_id:
            raise ValueError("shard_id must be nonblank")
        return cls(
            lock_id=uuid.uuid4().hex,
            study_id=study_id,
            model_run_id=model_run_id,
            shard_id=shard_id,
            hostname=hostname,
            pid=pid,
            slurm_job_id=slurm_job_id,
            slurm_array_task_id=slurm_array_task_id,
            acquired_at=acquired_at,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LockMetadata":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != required:
            missing = sorted(required.difference(value))
            extra = sorted(set(value).difference(required))
            raise Part1RuntimeError(
                f"invalid lock metadata fields; missing={missing}, extra={extra}"
            )
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise Part1RuntimeError(f"invalid lock metadata: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_create(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_lock(path: Path) -> tuple[LockMetadata, bytes]:
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise LockHeldError("shard is not currently locked") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Part1RuntimeError(f"lock metadata is corrupt: {exc}") from exc
    if not isinstance(parsed, dict):
        raise Part1RuntimeError("lock metadata must be a JSON object")
    return LockMetadata.from_mapping(parsed), raw


def _assert_not_finalized(root: Path) -> None:
    if (root / ".finalized").exists():
        raise FinalizedRuntimeShardError("finalized shard is immutable and cannot be locked")


def default_worker_liveness(metadata: LockMetadata) -> Liveness:
    if metadata.hostname != socket.gethostname():
        return Liveness.UNKNOWN
    try:
        os.kill(metadata.pid, 0)
    except ProcessLookupError:
        return Liveness.DEAD
    except PermissionError:
        return Liveness.UNKNOWN
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return Liveness.DEAD
        return Liveness.UNKNOWN
    return Liveness.LIVE


def probe_slurm_liveness(
    metadata: LockMetadata,
    *,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: float = 5.0,
) -> Liveness:
    if metadata.slurm_job_id is None:
        return Liveness.NOT_APPLICABLE
    selector = metadata.slurm_job_id
    if metadata.slurm_array_task_id is not None:
        selector = f"{selector}_{metadata.slurm_array_task_id}"
    try:
        result = runner(
            [
                "squeue",
                "--jobs",
                selector,
                "--noheader",
                "--format=%i",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return Liveness.UNKNOWN
    if result.returncode != 0:
        return Liveness.UNKNOWN
    identifiers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not identifiers:
        return Liveness.DEAD
    if identifiers == [selector]:
        return Liveness.LIVE
    return Liveness.UNKNOWN


def default_slurm_liveness(metadata: LockMetadata) -> Liveness:
    return probe_slurm_liveness(metadata)


class LockedShardSession:
    """An exclusive shard lease coupled directly to guarded storage writes."""

    def __init__(
        self,
        root: Path,
        owner: LockMetadata,
        *,
        model_run_manifest_hash: str,
    ) -> None:
        self.root = Path(root)
        self.owner = owner
        self.lock_path = self.root / LOCK_FILENAME
        self.claim_path = self.root / RECOVERY_CLAIM_FILENAME
        self.store = Part1ShardStore(
            self.root,
            shard_id=owner.shard_id,
            study_id=owner.study_id,
            model_run_id=owner.model_run_id,
            model_run_manifest_hash=model_run_manifest_hash,
            mutation_guard=self.assert_owned,
        )

    @classmethod
    def acquire(
        cls,
        root: Path,
        *,
        owner: LockMetadata,
        model_run_manifest_hash: str,
    ) -> "LockedShardSession":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        _assert_not_finalized(root)
        claim_path = root / RECOVERY_CLAIM_FILENAME
        if claim_path.exists():
            raise LockHeldError("shard lock recovery is in progress")
        lock_path = root / LOCK_FILENAME
        try:
            _exclusive_create(lock_path, _json_bytes(owner.to_dict()))
        except FileExistsError as exc:
            raise LockHeldError("shard is already locked by another writer") from exc
        if claim_path.exists():
            current, _ = _read_lock(lock_path)
            if current.lock_id == owner.lock_id:
                lock_path.unlink()
                _fsync_directory(root)
            raise LockHeldError("shard lock recovery raced with acquisition")
        return cls(root, owner, model_run_manifest_hash=model_run_manifest_hash)

    def assert_owned(self) -> None:
        try:
            current, _ = _read_lock(self.lock_path)
        except LockHeldError as exc:
            raise LostLockOwnershipError("writer no longer owns the shard lock") from exc
        if current != self.owner:
            raise LostLockOwnershipError("writer no longer owns the shard lock")

    def close(self) -> None:
        self.assert_owned()
        self.lock_path.unlink()
        _fsync_directory(self.root)

    def __enter__(self) -> "LockedShardSession":
        self.assert_owned()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @classmethod
    def recover_stale(
        cls,
        root: Path,
        *,
        owner: LockMetadata,
        model_run_manifest_hash: str,
        event_timestamp: str,
        execution_context: Mapping[str, Any],
        worker_liveness: LivenessProbe = default_worker_liveness,
        slurm_liveness: LivenessProbe = default_slurm_liveness,
        fault_at: str | None = None,
    ) -> "LockedShardSession":
        return cls._takeover(
            root,
            owner=owner,
            model_run_manifest_hash=model_run_manifest_hash,
            event_type="stale_lock_recovered",
            event_timestamp=event_timestamp,
            execution_context=execution_context,
            operator_reason=None,
            worker_liveness=worker_liveness,
            slurm_liveness=slurm_liveness,
            fault_at=fault_at,
        )

    @classmethod
    def operator_unlock(
        cls,
        root: Path,
        *,
        owner: LockMetadata,
        model_run_manifest_hash: str,
        operator_reason: str,
        event_timestamp: str,
        execution_context: Mapping[str, Any],
    ) -> "LockedShardSession":
        if not operator_reason.strip():
            raise ValueError("operator unlock requires a nonblank reason")
        return cls._takeover(
            root,
            owner=owner,
            model_run_manifest_hash=model_run_manifest_hash,
            event_type="operator_unlock",
            event_timestamp=event_timestamp,
            execution_context=execution_context,
            operator_reason=operator_reason.strip(),
            worker_liveness=None,
            slurm_liveness=None,
            fault_at=None,
        )

    @classmethod
    def _takeover(
        cls,
        root: Path,
        *,
        owner: LockMetadata,
        model_run_manifest_hash: str,
        event_type: str,
        event_timestamp: str,
        execution_context: Mapping[str, Any],
        operator_reason: str | None,
        worker_liveness: LivenessProbe | None,
        slurm_liveness: LivenessProbe | None,
        fault_at: str | None,
    ) -> "LockedShardSession":
        if fault_at not in {
            None,
            "after_lock_replacement_before_event",
            "during_takeover_event_append",
        }:
            raise ValueError(f"unknown takeover fault boundary: {fault_at}")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        _assert_not_finalized(root)
        claim_path = root / RECOVERY_CLAIM_FILENAME
        if claim_path.exists():
            raise LockHeldError("another lock recovery is already in progress")
        lock_path = root / LOCK_FILENAME
        previous, previous_raw = _read_lock(lock_path)
        if (
            previous.study_id != owner.study_id
            or previous.model_run_id != owner.model_run_id
            or previous.shard_id != owner.shard_id
        ):
            raise CompatibilityError("replacement lock identity differs from prior lock")
        claim = {
            "requested_event_type": event_type,
            "previous_lock": previous.to_dict(),
            "previous_lock_sha256": hashlib.sha256(previous_raw).hexdigest(),
            "replacement_lock": owner.to_dict(),
            "model_run_manifest_hash": model_run_manifest_hash,
            "event_timestamp": event_timestamp,
            "execution_context": dict(execution_context),
            "operator_reason": operator_reason,
        }
        try:
            _exclusive_create(claim_path, _json_bytes(claim))
        except FileExistsError as exc:
            raise LockHeldError("another lock recovery is already in progress") from exc

        replaced = False
        try:
            current, current_raw = _read_lock(lock_path)
            if current != previous or current_raw != previous_raw:
                raise LockHeldError("lock owner changed during recovery claim acquisition")

            if event_type == "stale_lock_recovered":
                assert worker_liveness is not None and slurm_liveness is not None
                worker_state = worker_liveness(previous)
                if worker_state is Liveness.LIVE:
                    raise StaleRecoveryRefused("automatic recovery refused: prior owner is live")
                if worker_state is Liveness.UNKNOWN:
                    raise StaleRecoveryRefused(
                        "automatic recovery refused: liveness is uncertain"
                    )
                slurm_state = slurm_liveness(previous)
                if slurm_state is Liveness.LIVE:
                    raise StaleRecoveryRefused("automatic recovery refused: prior owner is live")
                if slurm_state is Liveness.UNKNOWN:
                    raise StaleRecoveryRefused(
                        "automatic recovery refused: liveness is uncertain"
                    )
                if worker_state is not Liveness.DEAD or slurm_state not in {
                    Liveness.DEAD,
                    Liveness.NOT_APPLICABLE,
                }:
                    raise StaleRecoveryRefused(
                        "automatic recovery refused without conclusive stale evidence"
                    )

            current, current_raw = _read_lock(lock_path)
            if current != previous or current_raw != previous_raw:
                raise LockHeldError("lock owner changed during recovery verification")

            history_directory = root / LOCK_HISTORY_DIRECTORY
            history_directory.mkdir(parents=True, exist_ok=True)
            _fsync_directory(root)
            history_path = history_directory / f"{previous.lock_id}.{event_type}.json"
            history = {
                "recovery_event_type": event_type,
                "recovered_at": event_timestamp,
                "previous_lock": previous.to_dict(),
                "replacement_lock": owner.to_dict(),
                "operator_reason": operator_reason,
            }
            try:
                _exclusive_create(history_path, _json_bytes(history))
            except FileExistsError as exc:
                if history_path.read_bytes() != _json_bytes(history):
                    raise Part1RuntimeError("lock recovery history already exists") from exc

            pending_path = root / f".{LOCK_FILENAME}.{owner.lock_id}.pending"
            _exclusive_create(pending_path, _json_bytes(owner.to_dict()))
            os.replace(pending_path, lock_path)
            _fsync_directory(root)
            replaced = True
            if fault_at == "after_lock_replacement_before_event":
                raise InjectedTakeoverCrash(fault_at)
            return cls._finish_pending_takeover(root, fault_at=fault_at)
        except Exception:
            # If takeover already replaced the lock, keep the claim as durable
            # fail-closed evidence. An operator must inspect the archived owner
            # and complete recovery; no third writer can acquire meanwhile.
            if not replaced and claim_path.exists():
                claim_path.unlink()
                _fsync_directory(root)
            raise

    @classmethod
    def finish_pending_takeover(cls, root: Path) -> "LockedShardSession":
        """Finish durable takeover evidence after a crash, without redoing liveness."""

        return cls._finish_pending_takeover(root, fault_at=None)

    @classmethod
    def _finish_pending_takeover(
        cls, root: Path, *, fault_at: str | None
    ) -> "LockedShardSession":
        """Internal completion with a synthetic partial-event crash boundary."""

        root = Path(root)
        claim_path = root / RECOVERY_CLAIM_FILENAME
        try:
            claim_value = json.loads(claim_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LockHeldError("no pending lock recovery exists") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Part1RuntimeError(f"pending lock recovery claim is corrupt: {exc}") from exc
        if not isinstance(claim_value, dict):
            raise Part1RuntimeError("pending lock recovery claim must be an object")
        owner = LockMetadata.from_mapping(claim_value["replacement_lock"])
        previous = LockMetadata.from_mapping(claim_value["previous_lock"])
        current, _ = _read_lock(root / LOCK_FILENAME)
        if current != owner:
            raise LostLockOwnershipError("replacement writer no longer owns pending takeover")
        session = cls(
            root,
            owner,
            model_run_manifest_hash=claim_value["model_run_manifest_hash"],
        )

        journal_path = root / TAKEOVER_EVENT_JOURNAL_FILENAME
        event: dict[str, Any] | None = None
        if journal_path.exists():
            try:
                event = json.loads(journal_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Part1RuntimeError(f"takeover event journal is corrupt: {exc}") from exc
            if not isinstance(event, dict):
                raise Part1RuntimeError("takeover event journal must contain an object")

        session.store.finish_pending_recoveries()
        inspection = session.store.inspect()
        audit_needs_repair = (
            "audit_events" in inspection.trailing_tails
            or "audit_events" in inspection.unterminated_streams
        )
        if audit_needs_repair and event is not None:
            recovery_sequence = int(event["event_sequence"]) - 1
            if recovery_sequence < 0:
                raise Part1RuntimeError("takeover journal did not reserve a recovery sequence")
            session.store.recover_trailing_line(
                "audit_events",
                event_sequence=recovery_sequence,
                event_timestamp=claim_value["event_timestamp"],
                execution_context=claim_value["execution_context"],
            )
            inspection = session.store.inspect()
        if inspection.trailing_tails or inspection.unterminated_streams:
            raise Part1RuntimeError(
                "pending shard stream recovery must be completed before lock takeover audit"
            )

        if event is None:
            sequence = max(
                (
                    event["event_sequence"]
                    for event in inspection.audit_events
                    if event["event_scope"] == "shard"
                ),
                default=-1,
            ) + 2
            event_type = claim_value["requested_event_type"]
            event = {
                "schema_name": "part1_audit_event",
                "schema_version": "1.0.0",
                "event_id": audit_event_id(
                    None,
                    event_type,
                    sequence,
                    study_id_value=owner.study_id,
                    model_run_id_value=owner.model_run_id,
                    shard_id=owner.shard_id,
                ),
                "event_scope": "shard",
                "study_id": owner.study_id,
                "model_run_id": owner.model_run_id,
                "shard_id": owner.shard_id,
                "question_id": None,
                "run_id": None,
                "checkpoint_id": None,
                "attempt_id": None,
                "attempt_number": None,
                "event_sequence": sequence,
                "event_type": event_type,
                "event_timestamp": claim_value["event_timestamp"],
                "execution_context": claim_value["execution_context"],
                "outcome_category": "verified_stale_lock"
                if event_type == "stale_lock_recovered"
                else "operator_authorized_takeover",
                "error_details": None,
                "retry_classification": None,
                "retry_decision": None,
                "backoff_seconds": None,
                "related_lock_owner": previous.to_dict(),
                "terminal_record_id": None,
                "operator_reason": claim_value["operator_reason"],
            }
            validate_instance("audit_event", event)
            _exclusive_create(journal_path, _json_bytes(event))

        matching = [
            existing
            for existing in session.store.inspect().audit_events
            if existing["event_id"] == event["event_id"]
        ]
        if matching:
            if matching != [event]:
                raise Part1RuntimeError("takeover audit event conflicts with durable journal")
        else:
            if fault_at == "during_takeover_event_append":
                session.store._preflight_audit_event(event)
                session.assert_owned()
                session.store._durable_partial_append(
                    session.store.audit_events_path,
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n",
                )
                raise InjectedTakeoverCrash(fault_at)
            session.store.append_audit_event(event)
        journal_path.unlink()
        claim_path.unlink()
        _fsync_directory(root)
        return session


@dataclass(frozen=True)
class WorkSpec:
    study_id: str
    model_run_id: str
    question_id: str
    run_id: int
    checkpoint_id: str | None
    seed: int

    @classmethod
    def natural(
        cls,
        study_id: str,
        model_run_id: str,
        question_id: str,
        run_id: int,
        *,
        seed: int,
    ) -> "WorkSpec":
        return cls(study_id, model_run_id, question_id, run_id, None, seed)

    @classmethod
    def checkpoint(
        cls,
        study_id: str,
        model_run_id: str,
        question_id: str,
        run_id: int,
        checkpoint_id: str,
        *,
        seed: int,
    ) -> "WorkSpec":
        return cls(study_id, model_run_id, question_id, run_id, checkpoint_id, seed)

    @property
    def key(self) -> LogicalKey:
        prefix: NaturalKey = (
            self.study_id,
            self.model_run_id,
            self.question_id,
            self.run_id,
        )
        return prefix if self.checkpoint_id is None else (*prefix, self.checkpoint_id)

    @property
    def terminal_record_id(self) -> str:
        if self.checkpoint_id is None:
            return natural_record_id(
                self.study_id, self.model_run_id, self.question_id, self.run_id
            )
        return checkpoint_record_id(
            self.study_id,
            self.model_run_id,
            self.question_id,
            self.run_id,
            self.checkpoint_id,
        )


@dataclass(frozen=True)
class RetryPlan:
    work: WorkSpec
    category: str
    classification: str
    decision: str
    attempts_consumed: int
    next_attempt_number: int | None
    requires_fresh_process: bool
    terminate_current_process: bool


def plan_retry(work: WorkSpec, *, category: str, attempts_consumed: int) -> RetryPlan:
    if not 0 <= attempts_consumed <= MAX_TOTAL_ATTEMPTS:
        raise ValueError("attempts_consumed must be 0 through 3")
    if category in TERMINAL_CATEGORIES:
        classification = "terminal"
        decision = "do_not_retry"
        next_attempt = None
    elif category in RETRYABLE_CATEGORIES:
        classification = "retryable"
        if attempts_consumed >= MAX_TOTAL_ATTEMPTS:
            decision = "exhausted"
            next_attempt = None
        else:
            decision = "retry"
            next_attempt = attempts_consumed + 1
    else:
        raise ValueError(f"unknown failure category: {category}")
    fresh = category == "transient_cuda_runtime_failure" and decision == "retry"
    return RetryPlan(
        work=work,
        category=category,
        classification=classification,
        decision=decision,
        attempts_consumed=attempts_consumed,
        next_attempt_number=next_attempt,
        requires_fresh_process=fresh,
        terminate_current_process=fresh,
    )


def assert_in_process_retry_allowed(plan: RetryPlan) -> None:
    if plan.requires_fresh_process:
        raise FreshProcessRequired(
            "transient CUDA retry requires worker termination and a fresh CUDA process"
        )


@dataclass(frozen=True)
class ResumeDecision:
    work: WorkSpec
    status: str
    reason: str
    attempts_consumed: int
    next_attempt_number: int | None


def _events_for_key(index: ShardIndex, key: LogicalKey) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for attempt_events in index.events_by_attempt.values():
        if not attempt_events:
            continue
        first = attempt_events[0]
        event_key: LogicalKey = (
            first["study_id"],
            first["model_run_id"],
            first["question_id"],
            first["run_id"],
        )
        if first["checkpoint_id"] is not None:
            event_key = (*event_key, first["checkpoint_id"])
        if event_key == key:
            events.extend(attempt_events)
    return sorted(events, key=lambda event: (event["attempt_number"], event["event_sequence"]))


def _terminal_for_work(index: ShardIndex, work: WorkSpec) -> Mapping[str, Any] | None:
    if work.checkpoint_id is None:
        return index.natural_terminal_by_key.get(work.key)  # type: ignore[arg-type]
    return index.checkpoint_terminal_by_key.get(work.key)  # type: ignore[arg-type]


def _completed_natural(index: ShardIndex, work: WorkSpec) -> bool:
    key: NaturalKey = (work.study_id, work.model_run_id, work.question_id, work.run_id)
    record = index.natural_terminal_by_key.get(key)
    return record is not None and record["natural_execution_outcome"] == "complete"


def plan_resume(
    index: ShardIndex, work_items: Iterable[WorkSpec]
) -> dict[WorkSpec, ResumeDecision]:
    if index.lifecycle_errors:
        raise CompatibilityError("cannot resume from lifecycle-corrupt shard state")
    decisions: dict[WorkSpec, ResumeDecision] = {}
    known_provenance = {
        (record["study_id"], record["model_run_id"])
        for record in index.terminal_by_id.values()
    }
    known_provenance.update(
        (event["study_id"], event["model_run_id"])
        for events in index.events_by_attempt.values()
        for event in events
    )
    for work in work_items:
        if known_provenance and (work.study_id, work.model_run_id) not in known_provenance:
            raise CompatibilityError("requested work study/model-run identity is incompatible")
        terminal = _terminal_for_work(index, work)
        events = _events_for_key(index, work.key)
        started_numbers = {
            int(event["attempt_number"])
            for event in events
            if event["event_type"] == "attempt_started"
        }
        attempts = len(started_numbers)

        if terminal is not None:
            persisted_seed = (
                terminal["generation_seed"]
                if work.checkpoint_id is None
                else terminal["natural_seed"]
            )
            if persisted_seed != work.seed:
                raise CompatibilityError("persisted terminal seed differs from requested seed")
            if (
                (work.checkpoint_id is None and terminal["raw_record_id"] != work.terminal_record_id)
                or (
                    work.checkpoint_id is not None
                    and terminal["checkpoint_record_id"] != work.terminal_record_id
                )
            ):
                raise CompatibilityError("persisted terminal logical identity is incompatible")
            outcome = terminal[
                "natural_execution_outcome"
                if work.checkpoint_id is None
                else "checkpoint_execution_outcome"
            ]
            status = "completed" if outcome == "complete" else "terminal"
            reason = (
                "terminal_result_exists"
                if outcome == "complete"
                else "terminal_infrastructure_failure"
            )
            decisions[work] = ResumeDecision(work, status, reason, attempts, None)
            continue

        if work.checkpoint_id is not None:
            natural_key: NaturalKey = (
                work.study_id,
                work.model_run_id,
                work.question_id,
                work.run_id,
            )
            parent_natural = index.natural_terminal_by_key.get(natural_key)
            if parent_natural is None or not _completed_natural(index, work):
                decisions[work] = ResumeDecision(
                    work, "ineligible", "parent_natural_not_complete", attempts, None
                )
                continue
            if parent_natural["generation_seed"] != work.seed:
                raise CompatibilityError(
                    "checkpoint natural seed differs from persisted parent natural seed"
                )

        failed_events = [event for event in events if event["event_type"] == "attempt_failed"]
        if failed_events and failed_events[-1]["retry_classification"] == "terminal":
            decisions[work] = ResumeDecision(
                work, "terminal", "nonretryable_failure", attempts, None
            )
            continue
        if attempts >= MAX_TOTAL_ATTEMPTS:
            decisions[work] = ResumeDecision(
                work, "terminal", "attempts_exhausted", attempts, None
            )
            continue
        reason = "not_started" if attempts == 0 else "retryable_interruption_or_failure"
        decisions[work] = ResumeDecision(work, "retryable", reason, attempts, attempts + 1)
    return decisions


def prepare_resume(
    store: Part1ShardStore,
    work_items: Iterable[WorkSpec],
    *,
    event_timestamp: str,
    execution_context: Mapping[str, Any],
) -> dict[WorkSpec, ResumeDecision]:
    work = tuple(work_items)
    for item in work:
        if item.study_id != store.study_id or item.model_run_id != store.model_run_id:
            raise CompatibilityError("requested work is incompatible with shard provenance")
    store.reconcile(
        event_timestamp=event_timestamp,
        execution_context=execution_context,
    )
    return plan_resume(store.build_index(), work)


def validate_manifest_compatibility(
    study_manifest: Mapping[str, Any], model_manifest: Mapping[str, Any]
) -> dict[str, str]:
    try:
        validate_instance("study_manifest", study_manifest)
        validate_instance("model_run_manifest", model_manifest)
    except ValueError as exc:
        raise CompatibilityError(f"schema incompatibility: {exc}") from exc

    compatible_raw = study_manifest["compatible_raw_record_schema_versions"]
    if "1.0.0" not in compatible_raw:
        raise CompatibilityError("study does not allow Phase 1 raw record schema 1.0.0")
    expected_study_id = study_id(study_manifest)
    if study_manifest["study_id"] != expected_study_id:
        raise CompatibilityError("study_id does not match recomputed identity")
    expected_study_hash = study_manifest_hash(study_manifest)
    if study_manifest["study_manifest_hash"] != expected_study_hash:
        raise CompatibilityError("study_manifest_hash does not match recomputed hash")
    for field in ("study_id", "study_manifest_hash", "question_manifest_hash"):
        if model_manifest[field] != study_manifest[field]:
            raise CompatibilityError(f"model-run {field} differs from study manifest")
    expected_model_id = model_run_id(model_manifest)
    if model_manifest["model_run_id"] != expected_model_id:
        raise CompatibilityError("model_run_id does not match recomputed identity")
    expected_model_hash = model_run_manifest_hash(model_manifest)
    if model_manifest["model_run_manifest_hash"] != expected_model_hash:
        raise CompatibilityError("model_run_manifest_hash does not match recomputed hash")
    return {
        "study_id": expected_study_id,
        "study_manifest_hash": expected_study_hash,
        "model_run_id": expected_model_id,
        "model_run_manifest_hash": expected_model_hash,
        "question_manifest_hash": str(study_manifest["question_manifest_hash"]),
    }


def validate_retry_policy_config(config: Mapping[str, Any]) -> None:
    if config.get("max_total_attempts") != MAX_TOTAL_ATTEMPTS:
        raise CompatibilityError("retry max_total_attempts differs from shared policy")
    configured_retryable = config.get("retryable_categories")
    if not isinstance(configured_retryable, list) or (
        len(configured_retryable) != len(RETRYABLE_CATEGORIES)
        or set(configured_retryable) != RETRYABLE_CATEGORIES
    ):
        raise CompatibilityError("retryable_categories differ from shared policy")
    configured_terminal = config.get("terminal_categories")
    if not isinstance(configured_terminal, list) or (
        len(configured_terminal) != len(TERMINAL_CATEGORIES)
        or set(configured_terminal) != TERMINAL_CATEGORIES
    ):
        raise CompatibilityError("terminal_categories differ from shared policy")


def run_dry_run(
    *,
    mode: str = "smoke",
    persistent_root: Path | None = None,
    smoke_root: Path | None = None,
    production_root: Path | None = None,
    allow_root_override: bool = False,
    study_manifest: Mapping[str, Any] | None = None,
    model_run_manifest: Mapping[str, Any] | None = None,
    shard_root: Path | None = None,
) -> dict[str, Any]:
    configs = {name: load_config(name) for name in CONFIG_NAMES}
    storage = dict(configs["storage"])
    configured_smoke = Path(storage["smoke_root"] if smoke_root is None else smoke_root)
    configured_production = Path(
        storage["production_root"] if production_root is None else production_root
    )
    if configured_smoke.expanduser().resolve() == configured_production.expanduser().resolve():
        raise CompatibilityError("smoke and production output roots must be separate")
    if allow_root_override:
        storage["smoke_root"] = str(configured_smoke if smoke_root is not None else persistent_root)
        storage["production_root"] = str(configured_production)
        configs["storage"] = storage
    selected_root = Path(storage["smoke_root"]) if persistent_root is None else Path(persistent_root)
    config_report = validate_phase1_config(
        configs,
        mode=mode,
        persistent_root=selected_root,
    )
    validate_retry_policy_config(configs["retries"])
    for name in SCHEMA_NAMES:
        load_schema(name)

    manifest_report: dict[str, str] | None = None
    if (study_manifest is None) != (model_run_manifest is None):
        raise CompatibilityError("study and model-run manifests must be supplied together")
    if study_manifest is not None and model_run_manifest is not None:
        manifest_report = validate_manifest_compatibility(study_manifest, model_run_manifest)

    shard_report: dict[str, Any] | None = None
    if shard_root is not None:
        if manifest_report is None:
            raise CompatibilityError("shard inspection requires compatible manifest fixtures")
        shard = Part1ShardStore(
            shard_root,
            shard_id=Path(shard_root).name,
            study_id=manifest_report["study_id"],
            model_run_id=manifest_report["model_run_id"],
            model_run_manifest_hash=manifest_report["model_run_manifest_hash"],
        )
        inspection = shard.inspect()
        validation = shard.validate_shard(
            artifact_kind="natural_shard",
            started_at="1970-01-01T00:00:00Z",
            completed_at="1970-01-01T00:00:00Z",
        )
        index = None
        if not inspection.trailing_tails and not inspection.unterminated_streams:
            index = shard.build_index()
        shard_report = {
            "trailing_line_recovery_required": sorted(
                set(inspection.trailing_tails) | set(inspection.unterminated_streams)
            ),
            "pending_storage_recovery_event_ids": sorted(
                index.pending_recovery_event_ids if index is not None else ()
            ),
            "completed_logical_keys": len(index.completed_keys) if index is not None else None,
            "attempts_consumed": sum(
                len(attempts) for attempts in index.attempts_consumed.values()
            )
            if index is not None
            else None,
            "orphaned_attempt_ids": sorted(index.orphaned_attempt_ids)
            if index is not None
            else [],
            "validation": validation,
            "mutation_performed": False,
        }

    return {
        "is_valid": True,
        "mode": mode,
        "config": config_report,
        "schemas_validated": sorted(SCHEMA_NAMES),
        "manifest_compatibility": manifest_report,
        "shard_plan": shard_report,
        "retry_policy": {
            "max_total_attempts": MAX_TOTAL_ATTEMPTS,
            "retryable_categories": sorted(RETRYABLE_CATEGORIES),
            "terminal_categories": sorted(TERMINAL_CATEGORIES),
            "cuda_retry_requires_fresh_process": True,
        },
        "resume_plan": [],
        "would_create_production_manifest": False,
        "imports_model_or_data_libraries": False,
        "mutation_performed": False,
    }
