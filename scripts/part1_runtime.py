"""Login-safe runtime controls for Part 1 shard execution.

This module deliberately imports no model, tokenizer, dataset, torch, or CUDA
libraries. It coordinates storage ownership, retry decisions, compatibility,
and resume planning using only persisted metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
from typing import Any, Callable, Iterable, Iterator, Mapping
import uuid
import re

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
    validate_fixed_model_requested_contract,
    validate_fixed_study_contract,
    validate_phase1_config,
)
from part1_failure_policy import (
    ATTEMPT_NUMBERS,
    BACKOFF_SECONDS,
    MAX_TOTAL_ATTEMPTS,
    RETRYABLE_CATEGORIES,
    RETRYABLE_CATEGORY_ORDER,
    TERMINAL_CATEGORIES,
    TERMINAL_CATEGORY_ORDER,
)
from part1_store import CheckpointKey, LogicalKey, NaturalKey, Part1ShardStore, ShardIndex


LOCK_FILENAME = ".writer.lock"
RECOVERY_CLAIM_FILENAME = ".writer-lock-recovery.claim"
TAKEOVER_EVENT_JOURNAL_FILENAME = ".writer-lock-takeover-event.json"
LOCK_HISTORY_DIRECTORY = ".lock_history"
MUTATION_GUARD_FILENAME = ".writer.guard"

_LOCAL_GUARDS: dict[str, tuple[threading.RLock, threading.local]] = {}
_LOCAL_GUARDS_MUTEX = threading.Lock()


class MutationController:
    """Reentrant process/thread serialization on one stable per-shard inode."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / MUTATION_GUARD_FILENAME
        key = str(self.path.resolve())
        with _LOCAL_GUARDS_MUTEX:
            self._local_lock, self._state = _LOCAL_GUARDS.setdefault(
                key, (threading.RLock(), threading.local())
            )

    @contextmanager
    def section(self) -> Iterator[None]:
        _durable_mkdirs(self.root)
        with self._local_lock:
            depth = getattr(self._state, "depth", 0)
            if depth:
                self._state.depth = depth + 1
                try:
                    yield
                finally:
                    self._state.depth -= 1
                return
            guard_created = not self.path.exists()
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if guard_created:
                    _fsync_directory(self.root)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._state.depth = 1
                yield
            finally:
                self._state.depth = 0
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


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

    def __post_init__(self) -> None:
        hex32 = re.compile(r"^[0-9a-f]{32}$")
        hex64 = re.compile(r"^[0-9a-f]{64}$")
        if not isinstance(self.lock_id, str) or not hex32.fullmatch(self.lock_id):
            raise ValueError("lock_id must be 32 lowercase hexadecimal characters")
        for field, value in (("study_id", self.study_id), ("model_run_id", self.model_run_id)):
            if not isinstance(value, str) or not hex64.fullmatch(value):
                raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
        for field, value in (
            ("shard_id", self.shard_id),
            ("hostname", self.hostname),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonblank string")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1:
            raise ValueError("pid must be a positive integer")
        for field, value in (
            ("slurm_job_id", self.slurm_job_id),
            ("slurm_array_task_id", self.slurm_array_task_id),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be null or a nonblank string")
        if self.slurm_array_task_id is not None and self.slurm_job_id is None:
            raise ValueError("slurm_array_task_id requires slurm_job_id")
        rfc3339 = re.compile(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
        )
        if not isinstance(self.acquired_at, str) or not rfc3339.fullmatch(
            self.acquired_at
        ):
            raise ValueError("acquired_at must be an RFC 3339 timestamp")
        try:
            parsed = datetime.fromisoformat(self.acquired_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("acquired_at must be an RFC 3339 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("acquired_at must include a timezone")

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


def _durable_mkdirs(path: Path) -> None:
    """Create each missing directory and persist its parent entry."""

    missing: list[Path] = []
    cursor = Path(path)
    while not cursor.exists():
        missing.append(cursor)
        if cursor == cursor.parent:
            raise Part1RuntimeError(f"cannot find an existing parent for {path}")
        cursor = cursor.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        _fsync_directory(directory.parent)


def _exclusive_create(path: Path, payload: bytes) -> None:
    """Publish complete bytes without ever exposing a partial target path."""

    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink()
        _fsync_directory(path.parent)
        raise
    _fsync_directory(path.parent)
    temporary.unlink()
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
                f"--jobs={selector}",
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
        self.controller = MutationController(self.root)
        self.store = Part1ShardStore(
            self.root,
            shard_id=owner.shard_id,
            study_id=owner.study_id,
            model_run_id=owner.model_run_id,
            model_run_manifest_hash=model_run_manifest_hash,
            mutation_guard=self.assert_owned,
            mutation_section=self.controller.section,
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
        _durable_mkdirs(root)
        controller = MutationController(root)
        with controller.section():
            _assert_not_finalized(root)
            claim_path = root / RECOVERY_CLAIM_FILENAME
            if claim_path.exists():
                raise LockHeldError("shard lock recovery is in progress")
            lock_path = root / LOCK_FILENAME
            try:
                _exclusive_create(lock_path, _json_bytes(owner.to_dict()))
            except FileExistsError as exc:
                raise LockHeldError("shard is already locked by another writer") from exc
            session = cls(root, owner, model_run_manifest_hash=model_run_manifest_hash)
            try:
                session.store.initialize_provenance_header()
            except Exception as exc:
                lock_path.unlink()
                _fsync_directory(root)
                if "provenance" in str(exc) or "manifest hash" in str(exc):
                    raise CompatibilityError(str(exc)) from exc
                raise
            return session

    def assert_owned(self) -> None:
        try:
            current, _ = _read_lock(self.lock_path)
        except LockHeldError as exc:
            raise LostLockOwnershipError("writer no longer owns the shard lock") from exc
        if current != self.owner:
            raise LostLockOwnershipError("writer no longer owns the shard lock")

    def close(self) -> None:
        with self.controller.section():
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
            "after_claim_creation_before_liveness",
            "after_verified_evidence_before_replacement",
            "after_pending_replacement_durable_before_replace",
            "after_lock_replacement_before_event",
            "during_takeover_event_append",
            "after_event_before_claim_cleanup",
        }:
            raise ValueError(f"unknown takeover fault boundary: {fault_at}")
        root = Path(root)
        _durable_mkdirs(root)
        controller = MutationController(root)
        with controller.section():
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
            try:
                Part1ShardStore(
                    root,
                    shard_id=owner.shard_id,
                    study_id=owner.study_id,
                    model_run_id=owner.model_run_id,
                    model_run_manifest_hash=model_run_manifest_hash,
                )._assert_provenance_header()
            except Exception as exc:
                raise CompatibilityError(
                    f"replacement manifest hash differs from shard provenance: {exc}"
                ) from exc
            history_directory = root / LOCK_HISTORY_DIRECTORY
            _durable_mkdirs(history_directory)
            claim_id = uuid.uuid4().hex
            claim = {
                "claim_id": claim_id,
                "requested_event_type": event_type,
                "previous_lock": previous.to_dict(),
                "previous_lock_sha256": hashlib.sha256(previous_raw).hexdigest(),
                "replacement_lock": owner.to_dict(),
                "model_run_manifest_hash": model_run_manifest_hash,
                "event_timestamp": event_timestamp,
                "execution_context": dict(execution_context),
                "operator_reason": operator_reason,
            }
            payload = _json_bytes(claim)
            _exclusive_create(history_directory / f"{claim_id}.claim.json", payload)
            _exclusive_create(claim_path, payload)
            if fault_at == "after_claim_creation_before_liveness":
                raise InjectedTakeoverCrash(fault_at)
            try:
                return cls._finish_pending_takeover_locked(
                    root,
                    worker_liveness=worker_liveness,
                    slurm_liveness=slurm_liveness,
                    operator_override_reason=None,
                    fault_at=fault_at,
                )
            except InjectedTakeoverCrash:
                raise
            except Exception:
                # Only a wholly reversible refusal may release the active claim.
                # Once pending replacement bytes or a replacement lock exist,
                # the claim is the durable handle needed to finish the takeover.
                pending_path = root / f".{LOCK_FILENAME}.{owner.lock_id}.pending"
                reversible = False
                if not pending_path.exists():
                    try:
                        current, current_raw = _read_lock(lock_path)
                        reversible = current == previous and current_raw == previous_raw
                    except Part1RuntimeError:
                        reversible = False
                if reversible and claim_path.exists():
                    claim_path.unlink()
                    _fsync_directory(root)
                raise

    @classmethod
    def finish_pending_takeover(
        cls,
        root: Path,
        *,
        worker_liveness: LivenessProbe = default_worker_liveness,
        slurm_liveness: LivenessProbe = default_slurm_liveness,
        operator_override_reason: str | None = None,
    ) -> "LockedShardSession":
        """Resume any durable pre- or post-replacement takeover state."""

        root = Path(root)
        with MutationController(root).section():
            return cls._finish_pending_takeover_locked(
                root,
                worker_liveness=worker_liveness,
                slurm_liveness=slurm_liveness,
                operator_override_reason=operator_override_reason,
                fault_at=None,
            )

    @classmethod
    def _finish_pending_takeover_locked(
        cls,
        root: Path,
        *,
        worker_liveness: LivenessProbe | None,
        slurm_liveness: LivenessProbe | None,
        operator_override_reason: str | None,
        fault_at: str | None,
    ) -> "LockedShardSession":
        """Complete a claim while the stable mutation section is held."""

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
        required_claim_fields = {
            "claim_id",
            "requested_event_type",
            "previous_lock",
            "previous_lock_sha256",
            "replacement_lock",
            "model_run_manifest_hash",
            "event_timestamp",
            "execution_context",
            "operator_reason",
        }
        if set(claim_value) != required_claim_fields:
            raise Part1RuntimeError("pending lock recovery claim has invalid fields")
        if not isinstance(claim_value["claim_id"], str) or not re.fullmatch(
            r"[0-9a-f]{32}", claim_value["claim_id"]
        ):
            raise Part1RuntimeError("pending lock recovery claim_id is invalid")
        if claim_value["requested_event_type"] not in {
            "stale_lock_recovered",
            "operator_unlock",
        }:
            raise Part1RuntimeError("pending lock recovery event type is invalid")
        for field in ("previous_lock_sha256", "model_run_manifest_hash"):
            if not isinstance(claim_value[field], str) or not re.fullmatch(
                r"[0-9a-f]{64}", claim_value[field]
            ):
                raise Part1RuntimeError(f"pending lock recovery {field} is invalid")
        if not isinstance(claim_value["execution_context"], dict):
            raise Part1RuntimeError("pending lock recovery execution_context is invalid")
        if claim_value["requested_event_type"] == "operator_unlock" and (
            not isinstance(claim_value["operator_reason"], str)
            or not claim_value["operator_reason"].strip()
        ):
            raise Part1RuntimeError("pending operator recovery reason is invalid")
        owner = LockMetadata.from_mapping(claim_value["replacement_lock"])
        previous = LockMetadata.from_mapping(claim_value["previous_lock"])
        current, current_raw = _read_lock(root / LOCK_FILENAME)
        event_type = claim_value["requested_event_type"]
        operator_reason = claim_value["operator_reason"]
        if operator_override_reason is not None:
            if not operator_override_reason.strip():
                raise ValueError("operator override reason must be nonblank")
            event_type = "operator_unlock"
            operator_reason = operator_override_reason.strip()
        if current == previous:
            if hashlib.sha256(current_raw).hexdigest() != claim_value["previous_lock_sha256"]:
                raise LockHeldError("lock bytes changed after takeover claim")
            if event_type == "stale_lock_recovered":
                assert worker_liveness is not None and slurm_liveness is not None
                worker_state = worker_liveness(previous)
                slurm_state = slurm_liveness(previous)
                if worker_state is Liveness.LIVE or slurm_state is Liveness.LIVE:
                    raise StaleRecoveryRefused("automatic recovery refused: prior owner is live")
                conclusive = False
                if previous.slurm_job_id is not None:
                    conclusive = slurm_state is Liveness.DEAD or (
                        slurm_state is Liveness.UNKNOWN
                        and previous.hostname == socket.gethostname()
                        and worker_state is Liveness.DEAD
                    )
                else:
                    conclusive = (
                        previous.hostname == socket.gethostname()
                        and worker_state is Liveness.DEAD
                    )
                if not conclusive:
                    raise StaleRecoveryRefused(
                        "automatic recovery refused: liveness is uncertain"
                    )
            preflight_store = Part1ShardStore(
                root,
                shard_id=owner.shard_id,
                study_id=owner.study_id,
                model_run_id=owner.model_run_id,
                model_run_manifest_hash=claim_value["model_run_manifest_hash"],
            )
            preflight_inspection = preflight_store.inspect()
            incomplete_streams = set(preflight_inspection.trailing_tails) | set(
                preflight_inspection.unterminated_streams
            )
            if incomplete_streams:
                journal_streams = {
                    event["error_details"]["stream"]
                    for event in preflight_store._load_recovery_journal_events()
                }
                uncovered = incomplete_streams.difference(journal_streams)
                if uncovered:
                    raise Part1RuntimeError(
                        "unrecovered raw tails lack durable recovery evidence before "
                        f"takeover: {sorted(uncovered)}"
                    )
            if fault_at == "after_verified_evidence_before_replacement":
                raise InjectedTakeoverCrash(fault_at)
            pending_path = root / f".{LOCK_FILENAME}.{owner.lock_id}.pending"
            expected_pending = _json_bytes(owner.to_dict())
            if pending_path.exists():
                actual_pending = pending_path.read_bytes()
                if actual_pending != expected_pending:
                    if operator_override_reason is None:
                        raise Part1RuntimeError(
                            "pending replacement bytes conflict with durable claim; "
                            "operator override is required"
                        )
                    history_directory = root / LOCK_HISTORY_DIRECTORY
                    quarantine_path = (
                        history_directory
                        / f"{claim_value['claim_id']}.pending-quarantine"
                    )
                    if quarantine_path.exists():
                        if quarantine_path.read_bytes() != actual_pending:
                            raise Part1RuntimeError(
                                "pending replacement quarantine conflicts with current bytes"
                            )
                    else:
                        _exclusive_create(quarantine_path, actual_pending)
                    pending_path.unlink()
                    _fsync_directory(root)
                    _exclusive_create(pending_path, expected_pending)
            else:
                _exclusive_create(pending_path, expected_pending)
            if fault_at == "after_pending_replacement_durable_before_replace":
                raise InjectedTakeoverCrash(fault_at)
            os.replace(pending_path, root / LOCK_FILENAME)
            _fsync_directory(root)
            current = owner
        if current != owner:
            raise LostLockOwnershipError("replacement writer no longer owns pending takeover")
        if fault_at == "after_lock_replacement_before_event":
            raise InjectedTakeoverCrash(fault_at)
        session = cls(
            root,
            owner,
            model_run_manifest_hash=claim_value["model_run_manifest_hash"],
        )

        history_directory = root / LOCK_HISTORY_DIRECTORY
        journal_path = history_directory / f"{claim_value['claim_id']}.event.json"
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
                "operator_reason": operator_reason,
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
        if fault_at == "after_event_before_claim_cleanup":
            raise InjectedTakeoverCrash(fault_at)
        claim_path.unlink()
        _fsync_directory(root)
        return session


@dataclass(frozen=True)
class WorkSpec:
    study_id: str
    model_run_id: str
    model_run_manifest_hash: str
    question_id: str
    run_id: int
    checkpoint_id: str | None
    seed: int

    @classmethod
    def natural(
        cls,
        study_id: str,
        model_run_id: str,
        *identity: Any,
        seed: int,
    ) -> "WorkSpec":
        if len(identity) == 2:
            model_run_manifest_hash_value, question_id, run_id = "", *identity
        elif len(identity) == 3:
            model_run_manifest_hash_value, question_id, run_id = identity
        else:
            raise TypeError("natural requires question/run and optional manifest hash")
        return cls(
            study_id,
            model_run_id,
            model_run_manifest_hash_value,
            question_id,
            run_id,
            None,
            seed,
        )

    @classmethod
    def checkpoint(
        cls,
        study_id: str,
        model_run_id: str,
        *identity: Any,
        seed: int,
    ) -> "WorkSpec":
        if len(identity) == 3:
            model_run_manifest_hash_value, question_id, run_id, checkpoint_id = "", *identity
        elif len(identity) == 4:
            model_run_manifest_hash_value, question_id, run_id, checkpoint_id = identity
        else:
            raise TypeError("checkpoint requires question/run/checkpoint and optional manifest hash")
        return cls(
            study_id,
            model_run_id,
            model_run_manifest_hash_value,
            question_id,
            run_id,
            checkpoint_id,
            seed,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkSpec":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != required:
            raise ValueError("work spec fields are incomplete or unknown")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
    backoff_seconds: int | None


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
        backoff_seconds=(
            BACKOFF_SECONDS[next_attempt - 1]
            if decision == "retry" and next_attempt is not None
            else None
        ),
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
    terminalization_required: bool = False
    failure_category: str | None = None


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
    if index.hierarchy_errors:
        raise CompatibilityError("cannot resume from hierarchy-corrupt shard state")
    decisions: dict[WorkSpec, ResumeDecision] = {}
    for work in work_items:
        if work.model_run_manifest_hash != index.model_run_manifest_hash:
            raise CompatibilityError("requested work manifest hash is incompatible")
        if (work.study_id, work.model_run_id) != (index.study_id, index.model_run_id):
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
        if work.key in index.terminalization_required:
            decisions[work] = ResumeDecision(
                work,
                "terminalization_required",
                "terminal_result_required",
                attempts,
                None,
                True,
                index.terminalization_required[work.key],
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
        if item.model_run_manifest_hash != store.model_run_manifest_hash:
            raise CompatibilityError("requested work manifest hash differs from shard provenance")
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
        validate_fixed_study_contract(study_manifest)
        validate_fixed_model_requested_contract(model_manifest)
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
    exact_scalars = {
        "config_version": "1.0.0",
        "max_total_attempts": MAX_TOTAL_ATTEMPTS,
        "attempt_numbers": list(ATTEMPT_NUMBERS),
        "cuda_retry_requires_fresh_process": True,
        "preserve_seed_and_logical_identity": True,
        "backoff_seconds": list(BACKOFF_SECONDS),
    }
    for field, expected in exact_scalars.items():
        if config.get(field) != expected:
            raise CompatibilityError(f"retry {field} differs from shared policy")
    configured_retryable = config.get("retryable_categories")
    if configured_retryable != list(RETRYABLE_CATEGORY_ORDER):
        raise CompatibilityError("retryable_categories differ from shared policy")
    configured_terminal = config.get("terminal_categories")
    if configured_terminal != list(TERMINAL_CATEGORY_ORDER):
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
    work_items: Iterable[WorkSpec] = (),
    retry_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested_work = tuple(work_items)
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
    if requested_work and (manifest_report is None or shard_root is None):
        raise CompatibilityError(
            "work specs require compatible manifests and a manifest-bound shard"
        )
    if retry_request is not None and (manifest_report is None or shard_root is None):
        raise CompatibilityError(
            "retry request requires compatible manifests and a manifest-bound shard"
        )

    shard_report: dict[str, Any] | None = None
    if shard_root is not None:
        if manifest_report is None:
            raise CompatibilityError("shard inspection requires compatible manifest fixtures")
        header_path = Path(shard_root) / ".shard-provenance.json"
        if not header_path.exists():
            return {
                "is_valid": False,
                "mode": mode,
                "config": config_report,
                "schemas_validated": sorted(SCHEMA_NAMES),
                "manifest_compatibility": manifest_report,
                "shard_plan": {
                    "validation": {
                        "is_valid": False,
                        "error": "shard provenance header is missing",
                    },
                    "completed_logical_keys": None,
                    "mutation_performed": False,
                    "lock_present": False,
                    "takeover_pending": False,
                    "finalized": False,
                },
                "retry_policy": {
                    "max_total_attempts": MAX_TOTAL_ATTEMPTS,
                    "retryable_categories": sorted(RETRYABLE_CATEGORIES),
                    "terminal_categories": sorted(TERMINAL_CATEGORIES),
                    "cuda_retry_requires_fresh_process": True,
                },
                "resume_plan": [],
                "retry_plan": None,
                "would_create_production_manifest": False,
                "imports_model_or_data_libraries": False,
                "mutation_performed": False,
            }
        try:
            header_value = json.loads(header_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompatibilityError(f"shard provenance header is unavailable: {exc}") from exc
        if not isinstance(header_value, dict) or not isinstance(
            header_value.get("shard_id"), str
        ):
            raise CompatibilityError("shard provenance header is invalid")
        shard = Part1ShardStore(
            shard_root,
            shard_id=header_value["shard_id"],
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
        decisions: dict[WorkSpec, ResumeDecision] = {}
        compatible_work: list[WorkSpec] = []
        work_identity_invalid = False
        for work in requested_work:
            if (
                work.study_id != manifest_report["study_id"]
                or work.model_run_id != manifest_report["model_run_id"]
                or work.model_run_manifest_hash
                != manifest_report["model_run_manifest_hash"]
            ):
                work_identity_invalid = True
                decisions[work] = ResumeDecision(
                    work=work,
                    status="ineligible",
                    reason="manifest_or_shard_identity_mismatch",
                    attempts_consumed=0,
                    next_attempt_number=None,
                )
            else:
                compatible_work.append(work)
        if index is not None:
            decisions.update(plan_resume(index, compatible_work))
        lock_path = Path(shard_root) / LOCK_FILENAME
        lock_owner = _read_lock(lock_path)[0].to_dict() if lock_path.exists() else None
        history_directory = Path(shard_root) / LOCK_HISTORY_DIRECTORY
        finalized = shard.finalization_path.exists()
        finalized_blocked_work = False
        if finalized:
            for work, decision in tuple(decisions.items()):
                if decision.status not in {"completed", "terminal"}:
                    finalized_blocked_work = True
                    decisions[work] = ResumeDecision(
                        work=work,
                        status="ineligible",
                        reason="finalized_shard",
                        attempts_consumed=decision.attempts_consumed,
                        next_attempt_number=None,
                    )
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
            "lock_present": (Path(shard_root) / LOCK_FILENAME).exists(),
            "lock_owner": lock_owner,
            "takeover_pending": (
                Path(shard_root) / RECOVERY_CLAIM_FILENAME
            ).exists(),
            "takeover_history": sorted(
                path.name for path in history_directory.glob("*.json")
            )
            if history_directory.exists()
            else [],
            "finalized": finalized,
            "finalized_blocked_work": finalized_blocked_work,
            "work_identity_invalid": work_identity_invalid,
        }
    else:
        decisions = {}
        index = None

    retry_plan = None
    if retry_request is not None:
        retry_work = WorkSpec.from_mapping(retry_request["work"])
        assert manifest_report is not None and shard_root is not None
        retry_identity_matches = not (
            retry_work.study_id != manifest_report["study_id"]
            or retry_work.model_run_id != manifest_report["model_run_id"]
            or retry_work.model_run_manifest_hash
            != manifest_report["model_run_manifest_hash"]
        )
        requested_attempts = int(retry_request["attempts_consumed"])
        persisted_attempt_numbers = (
            sorted(index.attempts_consumed.get(retry_work.key, ()))
            if index is not None and retry_identity_matches
            else []
        )
        latest_persisted_attempt = (
            persisted_attempt_numbers[-1] if persisted_attempt_numbers else None
        )
        persisted_attempts = latest_persisted_attempt or 0
        persisted_failure_category = None
        latest_retry_closures: list[Mapping[str, Any]] = []
        if index is not None and retry_identity_matches:
            latest_retry_closures = [
                event
                for event in _events_for_key(index, retry_work.key)
                if event["attempt_number"] == latest_persisted_attempt
                and event["event_type"] in {"attempt_failed", "attempt_interrupted"}
                and event.get("retry_classification") == "retryable"
                and event.get("retry_decision") == "retry"
            ]
            if len(latest_retry_closures) == 1:
                persisted_failure_category = latest_retry_closures[0].get(
                    "outcome_category"
                )
        requested_category = str(retry_request["category"])
        retry_policy_plan = (
            plan_retry(
                retry_work,
                category=persisted_failure_category,
                attempts_consumed=persisted_attempts,
            )
            if persisted_failure_category is not None
            else None
        )
        retry_plan = (
            asdict(retry_policy_plan)
            if retry_policy_plan is not None
            else {
                "work": retry_work.to_dict(),
                "category": None,
                "classification": None,
                "decision": None,
                "attempts_consumed": persisted_attempts,
                "next_attempt_number": None,
                "requires_fresh_process": False,
                "terminate_current_process": False,
                "backoff_seconds": None,
            }
        )
        retry_plan["requested_category"] = requested_category
        retry_plan["requested_attempts_consumed"] = requested_attempts
        retry_plan["latest_persisted_attempt"] = latest_persisted_attempt
        retry_plan["persisted_failure_category"] = persisted_failure_category
        retry_plan["latest_retry_closure_count"] = len(latest_retry_closures)
        retry_resume = (
            plan_resume(index, [retry_work])[retry_work]
            if index is not None and retry_identity_matches
            else None
        )
        blockers: list[str] = []
        if not retry_identity_matches:
            blockers.append("manifest_or_shard_identity_mismatch")
        if requested_attempts != persisted_attempts:
            blockers.append("attempt_count_mismatch")
        if persisted_failure_category is None:
            blockers.append("missing_persisted_failure_category")
        if (
            persisted_failure_category is not None
            and requested_category != persisted_failure_category
        ):
            blockers.append("failure_category_mismatch")
        if retry_policy_plan is None or retry_policy_plan.decision != "retry":
            blockers.append("retry_policy_does_not_allow_retry")
        if retry_resume is None or retry_resume.status != "retryable":
            blockers.append("persisted_state_is_not_retryable")
        if shard_report is not None and shard_report["lock_present"]:
            blockers.append("active_writer_lock")
        if shard_report is not None and shard_report["takeover_pending"]:
            blockers.append("lock_takeover_pending")
        if shard_report is not None and shard_report["finalized"]:
            blockers.append("finalized_shard")
        retry_plan["persisted_attempts_consumed"] = persisted_attempts
        retry_plan["eligible"] = not blockers
        retry_plan["ineligibility_reasons"] = blockers

    return {
        "is_valid": (
            shard_report is None
            or (
                bool(shard_report["validation"]["is_valid"])
                and not shard_report["finalized_blocked_work"]
                and not shard_report["work_identity_invalid"]
                and not (retry_plan is not None and not retry_plan["eligible"])
            )
        ),
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
        "resume_plan": [asdict(decision) for decision in decisions.values()],
        "retry_plan": retry_plan,
        "would_create_production_manifest": False,
        "imports_model_or_data_libraries": False,
        "mutation_performed": False,
    }
