"""Crash-consistent, append-only storage for Part 1 terminal results and events.

This module is deliberately login-safe: it imports no model, tokenizer,
dataset, torch, or CUDA code. Exclusive writer locking and retry orchestration
belong to the separate Phase 1 runtime layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from part1_contract import (
    attempt_id,
    audit_event_id,
    canonical_json_bytes,
    checkpoint_record_id,
    natural_record_id,
    validate_instance,
)
from part1_failure_policy import (
    MAX_TOTAL_ATTEMPTS,
    RETRYABLE_CATEGORIES,
    classify_failure,
    validate_failure_event_policy,
)


STORE_VERSION = "part1-store-v1"
VALIDATOR_VERSION = "part1-shard-validator-v1"
VALIDATION_REPORT_IDENTITY_VERSION = "part1-validation-report-identity-v1"
STREAM_FILES = {
    "natural_results": "natural_results.jsonl",
    "checkpoint_results": "checkpoint_results.jsonl",
    "audit_events": "audit_events.jsonl",
}
STREAM_SCHEMAS = {
    "natural_results": "natural_terminal_result",
    "checkpoint_results": "checkpoint_terminal_result",
    "audit_events": "audit_event",
}
FAULT_BOUNDARIES = {
    "before_result_append",
    "during_result_append",
    "after_result_fsync_before_completion_event",
    "during_completion_event_append",
    "after_both_fsyncs",
}
RECOVERY_FAULT_BOUNDARIES = {
    "before_recovery_evidence",
    "after_recovery_evidence_before_mutation",
    "after_recovery_mutation_before_audit_append",
}

NaturalKey = tuple[str, str, str, int]
CheckpointKey = tuple[str, str, str, int, str]
LogicalKey = NaturalKey | CheckpointKey


class Part1StoreError(RuntimeError):
    """Base class for deterministic storage contract failures."""


class DuplicateTerminalResultError(Part1StoreError):
    """A logical key or immutable result ID already has a terminal record."""


class FinalizedShardError(Part1StoreError):
    """An append or recovery was requested after shard finalization."""


class MalformedMiddleError(Part1StoreError):
    """A malformed JSON line appeared before the final physical line."""


class InvalidRecordError(Part1StoreError):
    """A complete JSON object does not satisfy its stream schema."""


class StreamTailError(Part1StoreError):
    """An append was attempted before repairing an incomplete stream tail."""


class InjectedCrash(Part1StoreError):
    """Deterministic test-only simulation of a process crash boundary."""


@dataclass(frozen=True)
class StreamInspection:
    natural_results: tuple[dict[str, Any], ...]
    checkpoint_results: tuple[dict[str, Any], ...]
    audit_events: tuple[dict[str, Any], ...]
    trailing_tails: dict[str, bytes]
    unterminated_streams: frozenset[str]


@dataclass(frozen=True)
class ShardIndex:
    natural_terminal_by_key: dict[NaturalKey, dict[str, Any]]
    checkpoint_terminal_by_key: dict[CheckpointKey, dict[str, Any]]
    terminal_by_id: dict[str, dict[str, Any]]
    events_by_attempt: dict[str, tuple[dict[str, Any], ...]]
    attempts_consumed: dict[LogicalKey, frozenset[int]]
    completed_keys: frozenset[LogicalKey]
    missing_completion_record_ids: frozenset[str]
    missing_started_attempt_ids: frozenset[str]
    inconsistent_completion_attempt_ids: frozenset[str]
    orphaned_attempt_ids: frozenset[str]
    lifecycle_errors: tuple[str, ...]
    pending_recovery_event_ids: frozenset[str]


@dataclass(frozen=True)
class _ParsedStream:
    records: tuple[dict[str, Any], ...]
    trailing_tail: bytes | None
    trailing_offset: int | None
    unterminated_valid_record: bool


def _normalize_json(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSONL records require finite floating-point values")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSONL object keys must be strings")
            normalized_key = _normalize_json(key)
            if normalized_key in normalized:
                raise ValueError("line-ending normalization produced duplicate keys")
            normalized[normalized_key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported JSONL value type: {type(value).__name__}")


def _json_line_bytes(record: Mapping[str, Any]) -> bytes:
    rendered = json.dumps(
        _normalize_json(record),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return rendered.encode("utf-8") + b"\n"


def _record_id(record: Mapping[str, Any]) -> str:
    if record["schema_name"] == "part1_natural_terminal_result":
        return str(record["raw_record_id"])
    return str(record["checkpoint_record_id"])


def _logical_key(record: Mapping[str, Any]) -> LogicalKey:
    prefix: NaturalKey = (
        str(record["study_id"]),
        str(record["model_run_id"]),
        str(record["question_id"]),
        int(record["run_id"]),
    )
    if record["schema_name"] == "part1_checkpoint_terminal_result":
        return (*prefix, str(record["checkpoint_id"]))
    return prefix


def _event_logical_key(event: Mapping[str, Any]) -> LogicalKey:
    prefix: NaturalKey = (
        str(event["study_id"]),
        str(event["model_run_id"]),
        str(event["question_id"]),
        int(event["run_id"]),
    )
    checkpoint_id = event.get("checkpoint_id")
    if checkpoint_id is not None:
        return (*prefix, str(checkpoint_id))
    return prefix


class Part1ShardStore:
    """One active shard containing normalized terminal and event streams."""

    def __init__(
        self,
        root: Path,
        *,
        shard_id: str,
        study_id: str,
        model_run_id: str,
        model_run_manifest_hash: str,
        mutation_guard: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.shard_id = shard_id
        self.study_id = study_id
        self.model_run_id = model_run_id
        self.model_run_manifest_hash = model_run_manifest_hash
        self._mutation_guard = mutation_guard
        self.natural_results_path = self.root / STREAM_FILES["natural_results"]
        self.checkpoint_results_path = self.root / STREAM_FILES["checkpoint_results"]
        self.audit_events_path = self.root / STREAM_FILES["audit_events"]
        self.recovery_journal_directory = self.root / "recovery_journal"
        self.finalization_path = self.root / ".finalized"
        self.stream_paths = {
            "natural_results": self.natural_results_path,
            "checkpoint_results": self.checkpoint_results_path,
            "audit_events": self.audit_events_path,
        }

    def _assert_active(self) -> None:
        if self.finalization_path.exists():
            raise FinalizedShardError(f"shard {self.shard_id} is finalized and immutable")

    def _assert_mutation_authorized(self) -> None:
        if self._mutation_guard is not None:
            self._mutation_guard()

    def _assert_provenance(self, record: Mapping[str, Any]) -> None:
        if record["study_id"] != self.study_id:
            raise ValueError("record study_id differs from shard study_id")
        if record["model_run_id"] != self.model_run_id:
            raise ValueError("record model_run_id differs from shard model_run_id")
        manifest_hash = record.get("model_run_manifest_hash")
        if manifest_hash is not None and manifest_hash != self.model_run_manifest_hash:
            raise ValueError("record model_run_manifest_hash differs from shard provenance")

    def _parse_stream(self, stream_name: str) -> _ParsedStream:
        if stream_name not in self.stream_paths:
            raise ValueError(f"unknown Part 1 shard stream: {stream_name}")
        path = self.stream_paths[stream_name]
        if not path.exists():
            return _ParsedStream((), None, None, False)
        data = path.read_bytes()
        if not data:
            return _ParsedStream((), None, None, False)
        physical_lines = data.splitlines(keepends=True)
        records: list[dict[str, Any]] = []
        offset = 0
        schema_name = STREAM_SCHEMAS[stream_name]
        for index, physical_line in enumerate(physical_lines):
            final = index == len(physical_lines) - 1
            terminated = physical_line.endswith(b"\n")
            payload = physical_line[:-1] if terminated else physical_line
            try:
                decoded = payload.decode("utf-8")
                record = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if final:
                    return _ParsedStream(tuple(records), physical_line, offset, False)
                raise MalformedMiddleError(
                    f"malformed JSON in middle of {path} at physical line {index + 1}"
                ) from exc
            if not isinstance(record, dict):
                raise InvalidRecordError(
                    f"complete record in {path} line {index + 1} is not a JSON object"
                )
            try:
                validate_instance(schema_name, record)
            except ValueError as exc:
                raise InvalidRecordError(
                    f"schema-invalid complete record in {path} line {index + 1}: {exc}"
                ) from exc
            records.append(record)
            if final and not terminated:
                return _ParsedStream(tuple(records), None, None, True)
            offset += len(physical_line)
        return _ParsedStream(tuple(records), None, None, False)

    def inspect(self) -> StreamInspection:
        parsed = {name: self._parse_stream(name) for name in STREAM_FILES}
        return StreamInspection(
            natural_results=parsed["natural_results"].records,
            checkpoint_results=parsed["checkpoint_results"].records,
            audit_events=parsed["audit_events"].records,
            trailing_tails={
                name: stream.trailing_tail
                for name, stream in parsed.items()
                if stream.trailing_tail is not None
            },
            unterminated_streams=frozenset(
                name for name, stream in parsed.items() if stream.unterminated_valid_record
            ),
        )

    def _assert_stream_appendable(self, stream_name: str) -> _ParsedStream:
        parsed = self._parse_stream(stream_name)
        if parsed.trailing_tail is not None or parsed.unterminated_valid_record:
            raise StreamTailError(
                f"{stream_name} has an unrecovered trailing line; recover it before append"
            )
        return parsed

    @staticmethod
    def _durable_append(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _durable_partial_append(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_length = max(1, len(payload) // 2)
        with path.open("ab") as handle:
            handle.write(payload[:partial_length])
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _durable_create(cls, path: Path, payload: bytes) -> None:
        parent_existed = path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            cls._fsync_directory(path.parent.parent)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        cls._fsync_directory(path.parent)

    def _verify_terminal_identity(self, record: Mapping[str, Any]) -> None:
        checkpoint_id = record.get("checkpoint_id")
        if checkpoint_id is None:
            expected_record_id = natural_record_id(
                record["study_id"], record["model_run_id"], record["question_id"], record["run_id"]
            )
        else:
            expected_record_id = checkpoint_record_id(
                record["study_id"],
                record["model_run_id"],
                record["question_id"],
                record["run_id"],
                checkpoint_id,
            )
        if _record_id(record) != expected_record_id:
            raise ValueError("terminal record ID does not match its logical identity")
        expected_attempt_id = attempt_id(
            record["study_id"],
            record["model_run_id"],
            record["question_id"],
            record["run_id"],
            record["terminal_attempt_number"],
            checkpoint_id=checkpoint_id,
        )
        if record["terminal_attempt_id"] != expected_attempt_id:
            raise ValueError("terminal attempt ID does not match its logical identity")

    def _verify_event_identity(self, event: Mapping[str, Any]) -> None:
        if event["event_scope"] == "attempt":
            expected_attempt_id = attempt_id(
                event["study_id"],
                event["model_run_id"],
                event["question_id"],
                event["run_id"],
                event["attempt_number"],
                checkpoint_id=event.get("checkpoint_id"),
            )
            if event["attempt_id"] != expected_attempt_id:
                raise ValueError("audit attempt ID does not match its logical identity")
            expected_event_id = audit_event_id(
                event["attempt_id"], event["event_type"], event["event_sequence"]
            )
        else:
            expected_event_id = audit_event_id(
                None,
                event["event_type"],
                event["event_sequence"],
                study_id_value=event["study_id"],
                model_run_id_value=event["model_run_id"],
                shard_id=event["shard_id"],
            )
        if event["event_id"] != expected_event_id:
            raise ValueError("audit event ID does not match its identity payload")

    def _recovery_journal_path(self, event_id_value: str) -> Path:
        return self.recovery_journal_directory / f"{event_id_value}.json"

    def _load_recovery_journal_event(self, path: Path) -> dict[str, Any]:
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidRecordError(f"invalid recovery journal {path}: {exc}") from exc
        if not isinstance(event, dict):
            raise InvalidRecordError(f"recovery journal {path} is not a JSON object")
        try:
            validate_instance("audit_event", event)
            self._assert_provenance(event)
            self._verify_event_identity(event)
        except (KeyError, ValueError) as exc:
            raise InvalidRecordError(f"invalid recovery journal {path}: {exc}") from exc
        if (
            event["event_type"] != "trailing_line_recovered"
            or event["event_scope"] != "shard"
            or event["shard_id"] != self.shard_id
            or path.name != f"{event['event_id']}.json"
        ):
            raise InvalidRecordError(f"recovery journal {path} has incompatible identity")
        return event

    def _load_recovery_journal_events(self) -> tuple[dict[str, Any], ...]:
        if not self.recovery_journal_directory.exists():
            return ()
        return tuple(
            self._load_recovery_journal_event(path)
            for path in sorted(self.recovery_journal_directory.glob("*.json"))
        )

    def _persist_recovery_journal_event(self, event: Mapping[str, Any]) -> Path:
        path = self._recovery_journal_path(event["event_id"])
        payload = _json_line_bytes(event)[:-1]
        if path.exists():
            existing = self._load_recovery_journal_event(path)
            if existing != dict(event):
                raise Part1StoreError("existing recovery journal has conflicting evidence")
            return path
        self._assert_mutation_authorized()
        self._durable_create(path, payload)
        return path

    @staticmethod
    def _validate_scientific_alignment(record: Mapping[str, Any]) -> None:
        if record["schema_name"] == "part1_natural_terminal_result":
            generated = record["generated_token_ids"]
            entropies = record["per_token_entropy_nats"]
            if generated is not None and len(generated) != len(entropies):
                raise ValueError("generated token IDs and entropy trace must be aligned")
            return
        if record["entropy_status"] != "computed":
            return
        vector_fields = (
            "ad_token_ids",
            "ad_logits_float32",
            "ad_probabilities_float32",
        )
        if any(len(record[field]) != 4 for field in vector_fields):
            raise ValueError("A-D token IDs, logits, and probabilities must contain four aligned values")
        probabilities = record["ad_probabilities_float32"]
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("A-D probabilities must sum to one")
        if not math.isclose(
            record["maximum_ad_probability"], max(probabilities), rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError("maximum_ad_probability must match the A-D probabilities")
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
        if not math.isclose(record["answer_entropy_nats"], entropy, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("answer_entropy_nats must match the A-D probabilities")
        answer_index = record["answer_token_index"]
        forced_ids = record["forced_generated_token_ids"]
        if answer_index >= len(forced_ids) or forced_ids[answer_index] != record["answer_token_id"]:
            raise ValueError("answer-token index and ID must align with forced generated token IDs")

    def _attempt_started(self, attempt_id_value: str) -> bool:
        parsed = self._assert_stream_appendable("audit_events")
        return any(
            event["attempt_id"] == attempt_id_value and event["event_type"] == "attempt_started"
            for event in parsed.records
        )

    def _terminal_records(self) -> tuple[dict[str, Any], ...]:
        natural = self._assert_stream_appendable("natural_results").records
        checkpoint = self._assert_stream_appendable("checkpoint_results").records
        return (*natural, *checkpoint)

    @staticmethod
    def _matching_terminal_record(
        event: Mapping[str, Any], terminal_records: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any] | None:
        for record in terminal_records:
            if (
                _record_id(record) == event.get("terminal_record_id")
                and record["terminal_attempt_id"] == event["attempt_id"]
                and _logical_key(record) == _event_logical_key(event)
            ):
                return record
        return None

    @staticmethod
    def _validate_completion_event_policy(
        event: Mapping[str, Any], record: Mapping[str, Any]
    ) -> None:
        outcome = record.get(
            "natural_execution_outcome", record.get("checkpoint_execution_outcome")
        )
        if outcome == "terminal_infrastructure_failure":
            details = record.get("terminal_error_details")
            category = details.get("category") if isinstance(details, Mapping) else None
            if not isinstance(category, str):
                raise ValueError("terminal completion requires a failure category")
            expected = classify_failure(category, int(event["attempt_number"]))
            if event.get("outcome_category") != category:
                raise ValueError("completion failure category differs from terminal result")
            if event.get("retry_classification") != expected.classification:
                raise ValueError("completion failure policy classification is inconsistent")
            if event.get("retry_decision") != expected.retry_decision:
                raise ValueError("completion failure policy retry decision is inconsistent")
            return
        if any(
            event.get(field) is not None
            for field in ("outcome_category", "retry_classification", "retry_decision")
        ):
            raise ValueError("successful completion must not carry failure policy fields")

    def _consumed_attempts(self, logical_key: LogicalKey) -> set[int]:
        parsed = self._assert_stream_appendable("audit_events")
        return {
            event["attempt_number"]
            for event in parsed.records
            if event["event_scope"] == "attempt"
            and _event_logical_key(event) == logical_key
            and event["event_type"] == "attempt_started"
        }

    def _preflight_terminal_result(self, record: Mapping[str, Any]) -> str:
        self._assert_active()
        schema_name_value = record.get("schema_name")
        if schema_name_value == "part1_natural_terminal_result":
            stream_name = "natural_results"
            schema_name = "natural_terminal_result"
            outcome = record.get("natural_execution_outcome")
        elif schema_name_value == "part1_checkpoint_terminal_result":
            stream_name = "checkpoint_results"
            schema_name = "checkpoint_terminal_result"
            outcome = record.get("checkpoint_execution_outcome")
        else:
            raise ValueError("terminal result has an unsupported schema_name")
        validate_instance(schema_name, record)
        self._validate_scientific_alignment(record)
        self._assert_provenance(record)
        self._verify_terminal_identity(record)
        parsed = self._assert_stream_appendable(stream_name)
        logical_key = _logical_key(record)
        record_id_value = _record_id(record)
        for existing in parsed.records:
            if _logical_key(existing) == logical_key or _record_id(existing) == record_id_value:
                raise DuplicateTerminalResultError(
                    f"logical key {logical_key!r} already has a terminal result"
                )
        audit_records = self._assert_stream_appendable("audit_events").records
        matching_attempt_events = [
            event
            for event in audit_records
            if event["event_scope"] == "attempt"
            and event["attempt_id"] == record["terminal_attempt_id"]
        ]
        if not any(
            event["event_type"] == "attempt_started" for event in matching_attempt_events
        ):
            raise ValueError("terminal publication requires durable matching attempt_started evidence")
        terminal_lifecycle_events = [
            event
            for event in matching_attempt_events
            if event["event_type"]
            in {
                "attempt_failed",
                "attempt_interrupted",
                "attempt_completed",
                "terminal_result_recovered",
            }
        ]
        if terminal_lifecycle_events:
            raise ValueError(
                "attempt already has a terminal lifecycle event; result publication is forbidden"
            )
        if outcome == "terminal_infrastructure_failure":
            consumed = self._consumed_attempts(logical_key)
            terminal_attempt = int(record["terminal_attempt_number"])
            expected_consumed = set(range(1, terminal_attempt + 1))
            if consumed != expected_consumed:
                raise ValueError(
                    "terminal infrastructure failure requires sequential attempt_started "
                    f"events 1 through {terminal_attempt}"
                )
            details = record.get("terminal_error_details")
            category = details.get("category") if isinstance(details, Mapping) else None
            if not isinstance(category, str):
                raise ValueError("terminal infrastructure failure requires a failure category")
            classify_failure(category, terminal_attempt)
            if category in RETRYABLE_CATEGORIES and terminal_attempt != MAX_TOTAL_ATTEMPTS:
                raise ValueError("retryable terminal failure requires exhaustion at attempt 3")

            for earlier_number in range(1, terminal_attempt):
                earlier_events = [
                    event
                    for event in audit_records
                    if event["event_scope"] == "attempt"
                    and _event_logical_key(event) == logical_key
                    and event["attempt_number"] == earlier_number
                ]
                terminal_events = [
                    event
                    for event in earlier_events
                    if event["event_type"] in {"attempt_failed", "attempt_interrupted"}
                ]
                if len(terminal_events) != 1:
                    raise ValueError(
                        f"attempt {earlier_number} must be coherently closed before terminalization"
                    )
                validate_failure_event_policy(terminal_events[0])
                if terminal_events[0]["retry_decision"] != "retry":
                    raise ValueError(
                        f"attempt {earlier_number} did not authorize a subsequent retry"
                    )
        return stream_name

    def append_terminal_result(self, record: Mapping[str, Any]) -> None:
        self._assert_mutation_authorized()
        stream_name = self._preflight_terminal_result(record)
        self._assert_mutation_authorized()
        self._durable_append(self.stream_paths[stream_name], _json_line_bytes(record))

    def _validate_audit_event_against_records(
        self,
        event: Mapping[str, Any],
        existing_records: tuple[dict[str, Any], ...],
        *,
        terminal_records: tuple[Mapping[str, Any], ...] = (),
        prospective_terminal: Mapping[str, Any] | None = None,
    ) -> None:
        self._assert_active()
        validate_instance("audit_event", event)
        self._assert_provenance(event)
        if event["shard_id"] != self.shard_id:
            raise ValueError("audit event shard_id differs from the target shard")
        self._verify_event_identity(event)
        for existing in existing_records:
            if existing["event_id"] == event["event_id"]:
                raise ValueError(f"duplicate audit event ID: {event['event_id']}")
        if event["event_scope"] == "attempt":
            same_attempt = [
                existing
                for existing in existing_records
                if existing["attempt_id"] == event["attempt_id"]
            ]
            if same_attempt and event["event_sequence"] <= max(
                existing["event_sequence"] for existing in same_attempt
            ):
                raise ValueError("audit event_sequence must increase within an attempt")

            event_type = event["event_type"]
            started = [item for item in same_attempt if item["event_type"] == "attempt_started"]
            candidate_records: tuple[Mapping[str, Any], ...] = terminal_records
            if prospective_terminal is not None:
                candidate_records = (*candidate_records, prospective_terminal)
            attempt_has_terminal_result = any(
                record["terminal_attempt_id"] == event["attempt_id"]
                for record in candidate_records
            )
            if event_type in {
                "attempt_started",
                "attempt_failed",
                "attempt_interrupted",
            } and event["terminal_record_id"] is not None:
                raise ValueError(f"{event_type} requires terminal_record_id to be null")
            if event_type in {
                "attempt_completed",
                "terminal_result_recovered",
            } and event["terminal_record_id"] is None:
                raise ValueError(f"{event_type} requires terminal_record_id")
            if event_type == "attempt_started":
                if attempt_has_terminal_result:
                    raise ValueError("attempt_started cannot be appended after a terminal result")
                logical_key = _event_logical_key(event)
                if any(_logical_key(record) == logical_key for record in candidate_records):
                    raise ValueError("attempt_started cannot follow a terminal result for the key")
                if same_attempt:
                    raise ValueError("attempt_started must be the first and only start event")
                if event["event_sequence"] != 0:
                    raise ValueError("attempt_started must use event_sequence 0")
                prior_attempts = {
                    int(existing["attempt_number"])
                    for existing in existing_records
                    if existing["event_scope"] == "attempt"
                    and _event_logical_key(existing) == logical_key
                    and existing["event_type"] == "attempt_started"
                }
                expected_prior = set(range(1, int(event["attempt_number"])))
                if prior_attempts != expected_prior:
                    raise ValueError(
                        "attempt_started numbers must be sequential; "
                        f"attempt {event['attempt_number']} requires prior attempts "
                        f"{sorted(expected_prior)}"
                    )
                for prior_number in sorted(prior_attempts):
                    prior_events = [
                        existing
                        for existing in existing_records
                        if existing["event_scope"] == "attempt"
                        and _event_logical_key(existing) == logical_key
                        and existing["attempt_number"] == prior_number
                    ]
                    closures = [
                        existing
                        for existing in prior_events
                        if existing["event_type"] in {"attempt_failed", "attempt_interrupted"}
                    ]
                    if len(closures) != 1:
                        raise ValueError(
                            f"attempt {prior_number} must be coherently closed before retry"
                        )
                    validate_failure_event_policy(closures[0])
                    if closures[0]["retry_decision"] != "retry":
                        raise ValueError(
                            f"attempt {prior_number} policy does not authorize retry"
                        )
                return
            if not started:
                raise ValueError(f"{event_type} requires prior attempt_started evidence")

            terminal_types = {
                "attempt_failed",
                "attempt_interrupted",
                "attempt_completed",
                "terminal_result_recovered",
            }
            prior_terminal = [
                item for item in same_attempt if item["event_type"] in terminal_types
            ]
            matching_record = self._matching_terminal_record(event, candidate_records)

            if event_type == "terminal_result_recovered":
                if prior_terminal:
                    raise ValueError("attempt already has a terminal lifecycle event")
                if matching_record is None:
                    raise ValueError("terminal_result_recovered requires its matching terminal result")
            elif event_type == "attempt_completed":
                if prior_terminal:
                    if (
                        len(prior_terminal) != 1
                        or prior_terminal[0]["event_type"] != "terminal_result_recovered"
                        or self._matching_terminal_record(prior_terminal[0], candidate_records)
                        is None
                        or matching_record is None
                    ):
                        raise ValueError("attempt already has a conflicting terminal lifecycle event")
                # A completion without a result is retained as corrupt evidence so
                # reconciliation can append attempt_interrupted. Normal commit passes
                # prospective_terminal and therefore requires an exact match.
                if prospective_terminal is not None and matching_record is None:
                    raise ValueError("attempt_completed must match the proposed terminal result")
                if matching_record is not None:
                    self._validate_completion_event_policy(event, matching_record)
            elif event_type == "attempt_interrupted":
                validate_failure_event_policy(dict(event))
                if attempt_has_terminal_result:
                    raise ValueError("attempt interruption cannot follow a terminal result")
                if prior_terminal:
                    completion_without_result = (
                        len(prior_terminal) == 1
                        and prior_terminal[0]["event_type"] == "attempt_completed"
                        and self._matching_terminal_record(prior_terminal[0], terminal_records)
                        is None
                    )
                    if not completion_without_result:
                        raise ValueError("attempt already has a conflicting terminal lifecycle event")
            elif event_type == "attempt_failed":
                validate_failure_event_policy(dict(event))
                if attempt_has_terminal_result:
                    raise ValueError("attempt failure cannot follow a terminal result")
                if prior_terminal:
                    raise ValueError("attempt already has a conflicting terminal lifecycle event")

    def _preflight_audit_event(
        self,
        event: Mapping[str, Any],
        *,
        prospective_terminal: Mapping[str, Any] | None = None,
    ) -> None:
        parsed = self._assert_stream_appendable("audit_events")
        terminal_records = self._terminal_records() if event.get("event_scope") == "attempt" else ()
        self._validate_audit_event_against_records(
            event,
            parsed.records,
            terminal_records=terminal_records,
            prospective_terminal=prospective_terminal,
        )

    def append_audit_event(self, event: Mapping[str, Any]) -> None:
        self._assert_mutation_authorized()
        self._preflight_audit_event(event)
        self._assert_mutation_authorized()
        self._durable_append(self.audit_events_path, _json_line_bytes(event))

    def _validate_commit_pair(
        self, record: Mapping[str, Any], completion_event: Mapping[str, Any]
    ) -> None:
        schema_name = (
            "natural_terminal_result"
            if record.get("schema_name") == "part1_natural_terminal_result"
            else "checkpoint_terminal_result"
        )
        validate_instance(schema_name, record)
        self._validate_scientific_alignment(record)
        self._assert_provenance(record)
        self._verify_terminal_identity(record)
        validate_instance("audit_event", completion_event)
        self._assert_provenance(completion_event)
        self._verify_event_identity(completion_event)
        if completion_event["event_type"] != "attempt_completed":
            raise ValueError("commit requires an attempt_completed event")
        if completion_event["attempt_id"] != record["terminal_attempt_id"]:
            raise ValueError("completion event attempt does not match terminal result")
        if completion_event["terminal_record_id"] != _record_id(record):
            raise ValueError("completion event must reference the committed terminal result")
        if not self._attempt_started(record["terminal_attempt_id"]):
            raise ValueError("commit requires a durable matching attempt_started event")
        self._preflight_terminal_result(record)
        self._preflight_audit_event(completion_event, prospective_terminal=record)

    def commit_terminal_result(
        self,
        record: Mapping[str, Any],
        completion_event: Mapping[str, Any],
        *,
        fault_at: str | None = None,
    ) -> None:
        """Publish a terminal record before its completion event, fsyncing both."""

        self._assert_mutation_authorized()
        if fault_at is not None and fault_at not in FAULT_BOUNDARIES:
            raise ValueError(f"unknown fault boundary: {fault_at}")
        self._assert_active()
        self._validate_commit_pair(record, completion_event)
        if fault_at == "before_result_append":
            raise InjectedCrash(fault_at)

        result_stream = (
            "natural_results"
            if record["schema_name"] == "part1_natural_terminal_result"
            else "checkpoint_results"
        )
        if fault_at == "during_result_append":
            self._assert_mutation_authorized()
            self._durable_partial_append(self.stream_paths[result_stream], _json_line_bytes(record))
            raise InjectedCrash(fault_at)

        self.append_terminal_result(record)
        if fault_at == "after_result_fsync_before_completion_event":
            raise InjectedCrash(fault_at)

        if fault_at == "during_completion_event_append":
            self._assert_stream_appendable("audit_events")
            self._assert_mutation_authorized()
            self._durable_partial_append(self.audit_events_path, _json_line_bytes(completion_event))
            raise InjectedCrash(fault_at)

        self.append_audit_event(completion_event)
        if fault_at == "after_both_fsyncs":
            raise InjectedCrash(fault_at)

    def build_index(self) -> ShardIndex:
        inspection = self.inspect()
        if inspection.trailing_tails or inspection.unterminated_streams:
            raise StreamTailError("active shard contains an unrecovered trailing line")

        natural_by_key: dict[NaturalKey, dict[str, Any]] = {}
        checkpoint_by_key: dict[CheckpointKey, dict[str, Any]] = {}
        terminal_by_id: dict[str, dict[str, Any]] = {}
        terminal_attempt_ids: dict[str, str] = {}
        for record in (*inspection.natural_results, *inspection.checkpoint_results):
            self._assert_provenance(record)
            self._verify_terminal_identity(record)
            key = _logical_key(record)
            record_id_value = _record_id(record)
            target = checkpoint_by_key if len(key) == 5 else natural_by_key
            if key in target or record_id_value in terminal_by_id:
                raise DuplicateTerminalResultError(
                    f"duplicate/conflicting terminal result for logical key {key!r}"
                )
            target[key] = record
            terminal_by_id[record_id_value] = record
            terminal_attempt_ids[record["terminal_attempt_id"]] = record_id_value

        events_by_attempt_lists: dict[str, list[dict[str, Any]]] = {}
        event_ids: set[str] = set()
        attempts_consumed_sets: dict[LogicalKey, set[int]] = {}
        for event in inspection.audit_events:
            self._assert_provenance(event)
            if event["shard_id"] != self.shard_id:
                raise ValueError("audit event shard_id differs from the indexed shard")
            self._verify_event_identity(event)
            if event["event_id"] in event_ids:
                raise ValueError(f"duplicate audit event ID: {event['event_id']}")
            event_ids.add(event["event_id"])
            if event["event_scope"] != "attempt":
                continue
            attempt_events = events_by_attempt_lists.setdefault(event["attempt_id"], [])
            if attempt_events and event["event_sequence"] <= attempt_events[-1]["event_sequence"]:
                raise ValueError("event_sequence must increase in physical audit order")
            attempt_events.append(event)
            if event["event_type"] == "attempt_started":
                attempts_consumed_sets.setdefault(_event_logical_key(event), set()).add(
                    event["attempt_number"]
                )

        events_by_attempt = {
            attempt_id_value: tuple(sorted(events, key=lambda item: item["event_sequence"]))
            for attempt_id_value, events in events_by_attempt_lists.items()
        }
        def event_matches_terminal(event: Mapping[str, Any]) -> bool:
            terminal_record_id = event["terminal_record_id"]
            if terminal_record_id is None or terminal_record_id not in terminal_by_id:
                return False
            record = terminal_by_id[terminal_record_id]
            return (
                record["terminal_attempt_id"] == event["attempt_id"]
                and _logical_key(record) == _event_logical_key(event)
            )

        completed_terminal_ids = {
            event["terminal_record_id"]
            for event in inspection.audit_events
            if event["event_type"] in {"attempt_completed", "terminal_result_recovered"}
            and event_matches_terminal(event)
        }
        missing_completion = frozenset(set(terminal_by_id).difference(completed_terminal_ids))
        missing_started = frozenset(
            attempt_id_value
            for attempt_id_value in set(terminal_attempt_ids) | set(events_by_attempt)
            if not any(
                event["event_type"] == "attempt_started"
                for event in events_by_attempt.get(attempt_id_value, ())
            )
        )
        inconsistent_completion: set[str] = set()
        orphaned: set[str] = set()
        lifecycle_errors: list[str] = []
        for logical_key, attempt_numbers in attempts_consumed_sets.items():
            expected = set(range(1, max(attempt_numbers) + 1))
            if attempt_numbers != expected:
                lifecycle_errors.append(
                    f"logical key {logical_key!r} has nonsequential started attempts "
                    f"{sorted(attempt_numbers)}"
                )
        for attempt_id_value, events in events_by_attempt.items():
            event_types = {event["event_type"] for event in events}
            interrupted = "attempt_interrupted" in event_types
            completion_events = [
                event
                for event in events
                if event["event_type"] in {"attempt_completed", "terminal_result_recovered"}
            ]
            if completion_events and not interrupted:
                if any(not event_matches_terminal(event) for event in completion_events):
                    inconsistent_completion.add(attempt_id_value)
            if (
                "attempt_started" in event_types
                and not event_types.intersection(
                    {"attempt_failed", "attempt_interrupted", "attempt_completed", "terminal_result_recovered"}
                )
                and attempt_id_value not in terminal_attempt_ids
            ):
                orphaned.add(attempt_id_value)

            starts = [event for event in events if event["event_type"] == "attempt_started"]
            terminals = [
                event
                for event in events
                if event["event_type"]
                in {
                    "attempt_failed",
                    "attempt_interrupted",
                    "attempt_completed",
                    "terminal_result_recovered",
                }
            ]
            attempt_record = (
                terminal_by_id.get(terminal_attempt_ids[attempt_id_value])
                if attempt_id_value in terminal_attempt_ids
                else None
            )
            if len(starts) != 1 or events[0]["event_type"] != "attempt_started":
                lifecycle_errors.append(
                    f"attempt {attempt_id_value} must have exactly one leading attempt_started"
                )
            for event in events:
                if event["event_type"] in {
                    "attempt_started",
                    "attempt_failed",
                    "attempt_interrupted",
                } and event["terminal_record_id"] is not None:
                    lifecycle_errors.append(
                        f"attempt {attempt_id_value} event {event['event_type']} "
                        "must not reference a terminal record"
                    )
                if event["event_type"] in {
                    "attempt_completed",
                    "terminal_result_recovered",
                } and event["terminal_record_id"] is None:
                    lifecycle_errors.append(
                        f"attempt {attempt_id_value} event {event['event_type']} "
                        "requires a terminal record reference"
                    )
                if event["event_type"] in {"attempt_failed", "attempt_interrupted"}:
                    try:
                        validate_failure_event_policy(event)
                    except ValueError as exc:
                        lifecycle_errors.append(
                            f"attempt {attempt_id_value} failure policy is inconsistent: {exc}"
                        )
                if (
                    event["event_type"] == "attempt_completed"
                    and attempt_record is not None
                    and event_matches_terminal(event)
                ):
                    try:
                        self._validate_completion_event_policy(event, attempt_record)
                    except ValueError as exc:
                        lifecycle_errors.append(
                            f"attempt {attempt_id_value} completion failure policy is inconsistent: {exc}"
                        )
            terminal_types_in_order = [event["event_type"] for event in terminals]
            matching_terminals = [event_matches_terminal(event) for event in terminals]
            coherent = False
            if not terminals:
                coherent = attempt_record is None or attempt_id_value in terminal_attempt_ids
            elif terminal_types_in_order in (["attempt_failed"], ["attempt_interrupted"]):
                coherent = attempt_record is None
            elif terminal_types_in_order == ["attempt_completed"]:
                coherent = matching_terminals == [True]
            elif terminal_types_in_order == ["terminal_result_recovered"]:
                coherent = matching_terminals == [True]
            elif terminal_types_in_order == [
                "terminal_result_recovered",
                "attempt_completed",
            ]:
                coherent = matching_terminals == [True, True]
            elif terminal_types_in_order == ["attempt_completed", "attempt_interrupted"]:
                # Required corruption-classification exception: completion was
                # durable without its result, then resume classified interruption.
                coherent = matching_terminals == [False, False] and attempt_record is None
            if not coherent:
                lifecycle_errors.append(
                    f"attempt {attempt_id_value} has contradictory terminal lifecycle: "
                    f"{terminal_types_in_order!r}"
                )

        completed_keys: set[LogicalKey] = set(natural_by_key) | set(checkpoint_by_key)
        journal_events = self._load_recovery_journal_events()
        audit_event_ids = {event["event_id"] for event in inspection.audit_events}
        pending_recovery_event_ids = frozenset(
            event["event_id"] for event in journal_events if event["event_id"] not in audit_event_ids
        )
        return ShardIndex(
            natural_terminal_by_key=natural_by_key,
            checkpoint_terminal_by_key=checkpoint_by_key,
            terminal_by_id=terminal_by_id,
            events_by_attempt=events_by_attempt,
            attempts_consumed={
                key: frozenset(numbers) for key, numbers in attempts_consumed_sets.items()
            },
            completed_keys=frozenset(completed_keys),
            missing_completion_record_ids=missing_completion,
            missing_started_attempt_ids=missing_started,
            inconsistent_completion_attempt_ids=frozenset(inconsistent_completion),
            orphaned_attempt_ids=frozenset(orphaned),
            lifecycle_errors=tuple(lifecycle_errors),
            pending_recovery_event_ids=pending_recovery_event_ids,
        )

    def _make_attempt_event(
        self,
        source_event: Mapping[str, Any],
        *,
        event_type: str,
        event_sequence: int,
        event_timestamp: str,
        execution_context: Mapping[str, Any],
        terminal_record_id: str | None,
        outcome_category: str | None,
    ) -> dict[str, Any]:
        failure_policy = (
            classify_failure("interrupted_process", int(source_event["attempt_number"]))
            if event_type == "attempt_interrupted"
            else None
        )
        return {
            "schema_name": "part1_audit_event",
            "schema_version": "1.0.0",
            "event_id": audit_event_id(source_event["attempt_id"], event_type, event_sequence),
            "event_scope": "attempt",
            "study_id": self.study_id,
            "model_run_id": self.model_run_id,
            "shard_id": self.shard_id,
            "question_id": source_event["question_id"],
            "run_id": source_event["run_id"],
            "checkpoint_id": source_event["checkpoint_id"],
            "attempt_id": source_event["attempt_id"],
            "attempt_number": source_event["attempt_number"],
            "event_sequence": event_sequence,
            "event_type": event_type,
            "event_timestamp": event_timestamp,
            "execution_context": dict(execution_context),
            "outcome_category": "interrupted_process" if failure_policy else outcome_category,
            "error_details": {"interruption_reason": outcome_category}
            if failure_policy
            else None,
            "retry_classification": failure_policy.classification if failure_policy else None,
            "retry_decision": failure_policy.retry_decision if failure_policy else None,
            "backoff_seconds": None,
            "related_lock_owner": None,
            "terminal_record_id": terminal_record_id,
            "operator_reason": None,
        }

    def reconcile(
        self,
        *,
        event_timestamp: str,
        execution_context: Mapping[str, Any],
        append_missing_completion: bool = False,
    ) -> list[dict[str, Any]]:
        """Append recovery evidence for authoritative results and interrupted attempts."""

        self._assert_mutation_authorized()
        self._assert_active()
        index = self.build_index()
        appended: list[dict[str, Any]] = []

        for record_id_value in sorted(index.missing_completion_record_ids):
            record = index.terminal_by_id[record_id_value]
            attempt_events = index.events_by_attempt.get(record["terminal_attempt_id"], ())
            if not attempt_events:
                continue
            source = attempt_events[0]
            sequence = max(event["event_sequence"] for event in attempt_events) + 1
            recovered = self._make_attempt_event(
                source,
                event_type="terminal_result_recovered",
                event_sequence=sequence,
                event_timestamp=event_timestamp,
                execution_context=execution_context,
                terminal_record_id=record_id_value,
                outcome_category="authoritative_terminal_result_recovery",
            )
            self.append_audit_event(recovered)
            appended.append(recovered)
            if append_missing_completion:
                completion = self._make_attempt_event(
                    source,
                    event_type="attempt_completed",
                    event_sequence=sequence + 1,
                    event_timestamp=event_timestamp,
                    execution_context=execution_context,
                    terminal_record_id=record_id_value,
                    outcome_category=None,
                )
                outcome = record.get(
                    "natural_execution_outcome", record.get("checkpoint_execution_outcome")
                )
                if outcome == "terminal_infrastructure_failure":
                    category = record["terminal_error_details"]["category"]
                    policy = classify_failure(category, int(record["terminal_attempt_number"]))
                    completion.update(
                        outcome_category=category,
                        retry_classification=policy.classification,
                        retry_decision=policy.retry_decision,
                    )
                self.append_audit_event(completion)
                appended.append(completion)

        index = self.build_index()
        interrupted_ids = sorted(
            index.inconsistent_completion_attempt_ids | index.orphaned_attempt_ids
        )
        for attempt_id_value in interrupted_ids:
            events = index.events_by_attempt[attempt_id_value]
            source = events[0]
            category = (
                "completion_without_terminal_result"
                if attempt_id_value in index.inconsistent_completion_attempt_ids
                else "orphaned_started_attempt"
            )
            interrupted = self._make_attempt_event(
                source,
                event_type="attempt_interrupted",
                event_sequence=max(event["event_sequence"] for event in events) + 1,
                event_timestamp=event_timestamp,
                execution_context=execution_context,
                terminal_record_id=None,
                outcome_category=category,
            )
            self.append_audit_event(interrupted)
            appended.append(interrupted)
        return appended

    def recover_trailing_line(
        self,
        stream_name: str,
        *,
        event_sequence: int,
        event_timestamp: str,
        execution_context: Mapping[str, Any],
        fault_at: str | None = None,
    ) -> Path | None:
        """Durably journal and idempotently finish one final-line repair."""

        self._assert_mutation_authorized()
        self._assert_active()
        if fault_at is not None and fault_at not in RECOVERY_FAULT_BOUNDARIES:
            raise ValueError(f"unknown recovery fault boundary: {fault_at}")
        if stream_name not in self.stream_paths:
            raise ValueError(f"unknown Part 1 shard stream: {stream_name}")

        event_id_value = audit_event_id(
            None,
            "trailing_line_recovered",
            event_sequence,
            study_id_value=self.study_id,
            model_run_id_value=self.model_run_id,
            shard_id=self.shard_id,
        )
        journal_path = self._recovery_journal_path(event_id_value)
        path = self.stream_paths[stream_name]

        if journal_path.exists():
            event = self._load_recovery_journal_event(journal_path)
            if event["error_details"].get("stream") != stream_name:
                raise Part1StoreError("recovery journal stream differs from requested stream")
        else:
            parsed = self._parse_stream(stream_name)
            if parsed.trailing_tail is None and not parsed.unterminated_valid_record:
                return None
            original_bytes = path.read_bytes()
            if parsed.unterminated_valid_record:
                recovery_kind = "valid_record_missing_newline"
                recovered_bytes = 0
                recovered_hash = hashlib.sha256(b"").hexdigest()
                valid_prefix = original_bytes
                quarantine_path = None
            else:
                assert parsed.trailing_tail is not None
                assert parsed.trailing_offset is not None
                recovery_kind = "invalid_final_line"
                recovered_bytes = len(parsed.trailing_tail)
                recovered_hash = hashlib.sha256(parsed.trailing_tail).hexdigest()
                valid_prefix = original_bytes[: parsed.trailing_offset]
                quarantine_path = self.root / "quarantine" / (
                    f"{stream_name}.{recovered_hash}.trailing-bytes.bin"
                )
            event = {
                "schema_name": "part1_audit_event",
                "schema_version": "1.0.0",
                "event_id": event_id_value,
                "event_scope": "shard",
                "study_id": self.study_id,
                "model_run_id": self.model_run_id,
                "shard_id": self.shard_id,
                "question_id": None,
                "run_id": None,
                "checkpoint_id": None,
                "attempt_id": None,
                "attempt_number": None,
                "event_sequence": event_sequence,
                "event_type": "trailing_line_recovered",
                "event_timestamp": event_timestamp,
                "execution_context": dict(execution_context),
                "outcome_category": recovery_kind,
                "error_details": {
                    "stream": stream_name,
                    "recovered_byte_count": recovered_bytes,
                    "recovered_bytes_sha256": recovered_hash,
                    "quarantine_artifact": quarantine_path.name if quarantine_path else None,
                    "original_size": len(original_bytes),
                    "valid_prefix_size": len(valid_prefix),
                    "valid_prefix_sha256": hashlib.sha256(valid_prefix).hexdigest(),
                },
                "retry_classification": None,
                "retry_decision": None,
                "backoff_seconds": None,
                "related_lock_owner": None,
                "terminal_record_id": None,
                "operator_reason": None,
            }
            if stream_name == "audit_events":
                existing_audit_records = parsed.records
            else:
                existing_audit_records = self._assert_stream_appendable("audit_events").records
            self._validate_audit_event_against_records(event, existing_audit_records)

            if quarantine_path is not None:
                tail = parsed.trailing_tail
                assert tail is not None
                if quarantine_path.exists():
                    if quarantine_path.read_bytes() != tail:
                        raise Part1StoreError("existing quarantine artifact has conflicting bytes")
                else:
                    self._assert_mutation_authorized()
                    self._durable_create(quarantine_path, tail)
            if fault_at == "before_recovery_evidence":
                raise InjectedCrash(fault_at)
            self._persist_recovery_journal_event(event)

        details = event["error_details"]
        quarantine_name = details.get("quarantine_artifact")
        quarantine_path = (
            self.root / "quarantine" / quarantine_name if quarantine_name is not None else None
        )
        if fault_at == "after_recovery_evidence_before_mutation":
            raise InjectedCrash(fault_at)

        valid_prefix_size = details["valid_prefix_size"]
        valid_prefix_hash = details["valid_prefix_sha256"]
        current_bytes = path.read_bytes()
        if event["outcome_category"] == "invalid_final_line":
            if quarantine_path is None or not quarantine_path.exists():
                raise Part1StoreError("recovery quarantine artifact is missing")
            quarantined = quarantine_path.read_bytes()
            if (
                len(quarantined) != details["recovered_byte_count"]
                or hashlib.sha256(quarantined).hexdigest()
                != details["recovered_bytes_sha256"]
            ):
                raise Part1StoreError("recovery quarantine bytes do not match durable evidence")
            prefix = current_bytes[:valid_prefix_size]
            prefix_matches = (
                len(prefix) == valid_prefix_size
                and hashlib.sha256(prefix).hexdigest() == valid_prefix_hash
            )
            if not prefix_matches:
                raise Part1StoreError("raw stream valid prefix differs from recovery evidence")
            if current_bytes == prefix + quarantined:
                self._assert_mutation_authorized()
                with path.open("r+b") as handle:
                    handle.truncate(valid_prefix_size)
                    handle.flush()
                    os.fsync(handle.fileno())
            elif current_bytes != prefix:
                raise Part1StoreError("raw stream is neither pre- nor post-recovery state")
        elif event["outcome_category"] == "valid_record_missing_newline":
            if (
                len(current_bytes) == valid_prefix_size
                and hashlib.sha256(current_bytes).hexdigest() == valid_prefix_hash
            ):
                self._assert_mutation_authorized()
                self._durable_append(path, b"\n")
            elif not (
                len(current_bytes) == valid_prefix_size + 1
                and current_bytes.endswith(b"\n")
                and hashlib.sha256(current_bytes[:-1]).hexdigest() == valid_prefix_hash
            ):
                raise Part1StoreError("newline repair state differs from durable evidence")
        else:
            raise Part1StoreError("recovery journal has an unsupported recovery kind")

        if fault_at == "after_recovery_mutation_before_audit_append":
            raise InjectedCrash(fault_at)

        audit_parsed = self._assert_stream_appendable("audit_events")
        existing = [
            candidate for candidate in audit_parsed.records if candidate["event_id"] == event_id_value
        ]
        if existing:
            if len(existing) != 1 or existing[0] != event:
                raise Part1StoreError("main audit recovery evidence conflicts with journal")
        else:
            self._validate_audit_event_against_records(event, audit_parsed.records)
            self._assert_mutation_authorized()
            self._durable_append(self.audit_events_path, _json_line_bytes(event))
        return quarantine_path

    def finish_pending_recoveries(self) -> list[str]:
        """Idempotently finish every durable recovery journal not yet in main audit."""

        self._assert_mutation_authorized()
        self._assert_active()
        finished: list[str] = []
        for event in self._load_recovery_journal_events():
            audit_records = self._parse_stream("audit_events").records
            if any(record["event_id"] == event["event_id"] for record in audit_records):
                continue
            self.recover_trailing_line(
                event["error_details"]["stream"],
                event_sequence=event["event_sequence"],
                event_timestamp=event["event_timestamp"],
                execution_context=event["execution_context"],
            )
            finished.append(event["event_id"])
        return finished

    @staticmethod
    def _check(name: str, outcome: str, **details: Any) -> dict[str, Any]:
        return {"name": name, "outcome": outcome, "details": details}

    def validate_shard(
        self,
        *,
        artifact_kind: str,
        started_at: str,
        completed_at: str,
    ) -> dict[str, Any]:
        """Return a schema-valid validation report without mutating raw streams."""

        checks: list[dict[str, Any]] = []
        inspection: StreamInspection | None = None
        parse_error: Exception | None = None
        journal_error: Exception | None = None
        journal_events: tuple[dict[str, Any], ...] = ()
        try:
            journal_events = self._load_recovery_journal_events()
        except InvalidRecordError as exc:
            journal_error = exc
        try:
            inspection = self.inspect()
        except (MalformedMiddleError, InvalidRecordError) as exc:
            parse_error = exc

        if parse_error is None:
            assert inspection is not None
            audit_event_ids = {event["event_id"] for event in inspection.audit_events}
            pending_recovery_ids = sorted(
                event["event_id"]
                for event in journal_events
                if event["event_id"] not in audit_event_ids
            )
            tail_count = (
                len(inspection.trailing_tails)
                + len(inspection.unterminated_streams)
                + len(pending_recovery_ids)
            )
            checks.extend(
                [
                    self._check(
                        "json_syntax",
                        "failed" if inspection.trailing_tails else "passed",
                        invalid_trailing_streams=sorted(inspection.trailing_tails),
                    ),
                    self._check(
                        "schema_validity",
                        "failed" if journal_error else "passed",
                        records_validated=sum(
                            len(records)
                            for records in (
                                inspection.natural_results,
                                inspection.checkpoint_results,
                                inspection.audit_events,
                            )
                        ),
                        recovery_journal_error=str(journal_error) if journal_error else None,
                    ),
                    self._check("malformed_middle", "passed", malformed_middle_count=0),
                    self._check(
                        "trailing_tail_state",
                        "failed" if tail_count else "passed",
                        invalid_tails=sorted(inspection.trailing_tails),
                        valid_records_missing_newline=sorted(inspection.unterminated_streams),
                        pending_recovery_event_ids=pending_recovery_ids,
                    ),
                ]
            )
        else:
            malformed = isinstance(parse_error, MalformedMiddleError)
            invalid_schema = isinstance(parse_error, InvalidRecordError)
            checks.extend(
                [
                    self._check(
                        "json_syntax",
                        "failed" if malformed else "passed",
                        **({"error": str(parse_error)} if malformed else {}),
                    ),
                    self._check(
                        "schema_validity",
                        "failed" if invalid_schema else "warning",
                        error=str(parse_error),
                    ),
                    self._check(
                        "malformed_middle",
                        "failed" if malformed else "passed",
                        error=str(parse_error) if malformed else None,
                    ),
                    self._check("trailing_tail_state", "warning", state="not_evaluated"),
                ]
            )

        index: ShardIndex | None = None
        index_error: Exception | None = None
        if parse_error is None and inspection is not None and not (
            inspection.trailing_tails or inspection.unterminated_streams
        ):
            try:
                index = self.build_index()
            except (Part1StoreError, ValueError) as exc:
                index_error = exc
        index_not_evaluated = (
            parse_error is not None
            or inspection is None
            or bool(inspection.trailing_tails)
            or bool(inspection.unterminated_streams)
        )
        if index_not_evaluated:
            checks.append(
                self._check("duplicates_conflicts", "warning", state="not_evaluated")
            )
        elif index_error is None:
            checks.append(self._check("duplicates_conflicts", "passed"))
        else:
            checks.append(self._check("duplicates_conflicts", "failed", error=str(index_error)))

        alignment_error: Exception | None = None
        if inspection is not None:
            try:
                for record in (*inspection.natural_results, *inspection.checkpoint_results):
                    self._validate_scientific_alignment(record)
            except ValueError as exc:
                alignment_error = exc
        checks.append(
            self._check(
                "array_alignment",
                "failed" if alignment_error else "passed",
                **({"error": str(alignment_error)} if alignment_error else {}),
            )
        )
        checks.append(
            self._check(
                "outcome_nullability",
                "failed" if parse_error else "passed",
                **({"error": str(parse_error)} if parse_error else {}),
            )
        )

        if index is None:
            consistency_outcome = "failed" if parse_error or index_error else "warning"
            consistency_details: dict[str, Any] = {"state": "not_evaluated"}
        else:
            failures = (
                len(index.missing_started_attempt_ids)
                + len(index.inconsistent_completion_attempt_ids)
                + len(index.orphaned_attempt_ids)
                + len(index.lifecycle_errors)
            )
            warnings = len(index.missing_completion_record_ids)
            consistency_outcome = "failed" if failures else ("warning" if warnings else "passed")
            consistency_details = {
                "missing_started_attempt_ids": sorted(index.missing_started_attempt_ids),
                "inconsistent_completion_attempt_ids": sorted(
                    index.inconsistent_completion_attempt_ids
                ),
                "orphaned_attempt_ids": sorted(index.orphaned_attempt_ids),
                "lifecycle_errors": list(index.lifecycle_errors),
                "authoritative_results_missing_completion": sorted(
                    index.missing_completion_record_ids
                ),
            }
        checks.append(
            self._check("terminal_event_consistency", consistency_outcome, **consistency_details)
        )

        ordered_names = [
            "json_syntax",
            "schema_validity",
            "duplicates_conflicts",
            "malformed_middle",
            "trailing_tail_state",
            "array_alignment",
            "terminal_event_consistency",
            "outcome_nullability",
        ]
        checks_by_name = {check["name"]: check for check in checks}
        checks = [checks_by_name[name] for name in ordered_names]
        error_count = sum(check["outcome"] == "failed" for check in checks)
        warning_count = sum(check["outcome"] == "warning" for check in checks)
        summary = {
            "shard_id": self.shard_id,
            "natural_terminal_records": len(inspection.natural_results) if inspection else 0,
            "checkpoint_terminal_records": len(inspection.checkpoint_results) if inspection else 0,
            "audit_events": len(inspection.audit_events) if inspection else 0,
            "store_version": STORE_VERSION,
        }
        report_without_id = {
            "schema_name": "part1_validation_report",
            "schema_version": "1.0.0",
            "study_id": self.study_id,
            "model_run_id": self.model_run_id,
            "model_run_manifest_hash": self.model_run_manifest_hash,
            "validated_artifact_kind": artifact_kind,
            "validated_artifact_identity": self.shard_id,
            "validation_started_at": started_at,
            "validation_completed_at": completed_at,
            "validator_version": VALIDATOR_VERSION,
            "is_valid": error_count == 0,
            "checks": checks,
            "error_count": error_count,
            "warning_count": warning_count,
            "summary": summary,
        }
        validation_target_identity = {
            "identity_type": "validation_report_id",
            "identity_version": VALIDATION_REPORT_IDENTITY_VERSION,
            "payload": {
                "study_id": self.study_id,
                "model_run_id": self.model_run_id,
                "model_run_manifest_hash": self.model_run_manifest_hash,
                "shard_id": self.shard_id,
                "validated_artifact_kind": artifact_kind,
                "validator_version": VALIDATOR_VERSION,
                "store_contract_version": STORE_VERSION,
            },
        }
        report_id = hashlib.sha256(
            canonical_json_bytes(validation_target_identity)
        ).hexdigest()
        report = {
            **report_without_id,
            "validation_report_id": report_id,
        }
        validate_instance("validation_report", report)
        return report

    def write_validation_report(
        self,
        path: Path,
        *,
        artifact_kind: str,
        started_at: str,
        completed_at: str,
    ) -> dict[str, Any]:
        self._assert_mutation_authorized()
        report = self.validate_shard(
            artifact_kind=artifact_kind,
            started_at=started_at,
            completed_at=completed_at,
        )
        report_path = Path(path)
        self._assert_mutation_authorized()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as handle:
            handle.write(_json_line_bytes(report)[:-1])
            handle.flush()
            os.fsync(handle.fileno())
        return report

    def finalize(self) -> None:
        self._assert_mutation_authorized()
        self._assert_active()
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        report = self.validate_shard(
            artifact_kind="natural_shard",
            started_at=now,
            completed_at=now,
        )
        if not report["is_valid"]:
            raise Part1StoreError("cannot finalize an invalid shard")
        consistency = next(
            check for check in report["checks"] if check["name"] == "terminal_event_consistency"
        )
        if consistency["outcome"] != "passed":
            raise Part1StoreError(
                "cannot finalize until terminal/event reconciliation is complete"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        marker = {
            "store_version": STORE_VERSION,
            "shard_id": self.shard_id,
            "study_id": self.study_id,
            "model_run_id": self.model_run_id,
            "finalized_at": now,
        }
        self._assert_mutation_authorized()
        with self.finalization_path.open("xb") as handle:
            handle.write(_json_line_bytes(marker)[:-1])
            handle.flush()
            os.fsync(handle.fileno())
