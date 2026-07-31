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
from typing import Any, Mapping

from part1_contract import (
    attempt_id,
    audit_event_id,
    canonical_json_bytes,
    checkpoint_record_id,
    natural_record_id,
    validate_instance,
)


STORE_VERSION = "part1-store-v1"
VALIDATOR_VERSION = "part1-shard-validator-v1"
MAX_TOTAL_ATTEMPTS = 3

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
    ) -> None:
        self.root = Path(root)
        self.shard_id = shard_id
        self.study_id = study_id
        self.model_run_id = model_run_id
        self.model_run_manifest_hash = model_run_manifest_hash
        self.natural_results_path = self.root / STREAM_FILES["natural_results"]
        self.checkpoint_results_path = self.root / STREAM_FILES["checkpoint_results"]
        self.audit_events_path = self.root / STREAM_FILES["audit_events"]
        self.finalization_path = self.root / ".finalized"
        self.stream_paths = {
            "natural_results": self.natural_results_path,
            "checkpoint_results": self.checkpoint_results_path,
            "audit_events": self.audit_events_path,
        }

    def _assert_active(self) -> None:
        if self.finalization_path.exists():
            raise FinalizedShardError(f"shard {self.shard_id} is finalized and immutable")

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
        if outcome == "terminal_infrastructure_failure":
            consumed = self._consumed_attempts(logical_key)
            if consumed != set(range(1, MAX_TOTAL_ATTEMPTS + 1)):
                raise ValueError(
                    "terminal infrastructure failure requires actual attempt exhaustion "
                    "with attempt_started events 1, 2, and 3"
                )
        return stream_name

    def append_terminal_result(self, record: Mapping[str, Any]) -> None:
        stream_name = self._preflight_terminal_result(record)
        self._durable_append(self.stream_paths[stream_name], _json_line_bytes(record))

    def _validate_audit_event_against_records(
        self,
        event: Mapping[str, Any],
        existing_records: tuple[dict[str, Any], ...],
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

    def _preflight_audit_event(self, event: Mapping[str, Any]) -> None:
        parsed = self._assert_stream_appendable("audit_events")
        self._validate_audit_event_against_records(event, parsed.records)

    def append_audit_event(self, event: Mapping[str, Any]) -> None:
        self._preflight_audit_event(event)
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
        self._preflight_audit_event(completion_event)

    def commit_terminal_result(
        self,
        record: Mapping[str, Any],
        completion_event: Mapping[str, Any],
        *,
        fault_at: str | None = None,
    ) -> None:
        """Publish a terminal record before its completion event, fsyncing both."""

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
            self._durable_partial_append(self.stream_paths[result_stream], _json_line_bytes(record))
            raise InjectedCrash(fault_at)

        self.append_terminal_result(record)
        if fault_at == "after_result_fsync_before_completion_event":
            raise InjectedCrash(fault_at)

        if fault_at == "during_completion_event_append":
            self._assert_stream_appendable("audit_events")
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

        completed_keys: set[LogicalKey] = set(natural_by_key) | set(checkpoint_by_key)
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
        outcome_category: str,
    ) -> dict[str, Any]:
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
            "outcome_category": outcome_category,
            "error_details": None,
            "retry_classification": None,
            "retry_decision": None,
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
                    outcome_category="recovered_completion",
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
    ) -> Path | None:
        """Repair only the final line, preserving every valid record byte."""

        self._assert_active()
        parsed = self._parse_stream(stream_name)
        if parsed.trailing_tail is None and not parsed.unterminated_valid_record:
            return None
        path = self.stream_paths[stream_name]
        quarantine_path: Path | None = None
        if parsed.unterminated_valid_record:
            recovery_kind = "valid_record_missing_newline"
            recovered_bytes = 0
            recovered_hash = hashlib.sha256(b"").hexdigest()
        else:
            assert parsed.trailing_tail is not None
            assert parsed.trailing_offset is not None
            tail = parsed.trailing_tail
            recovery_kind = "invalid_final_line"
            recovered_bytes = len(tail)
            recovered_hash = hashlib.sha256(tail).hexdigest()
            quarantine_directory = self.root / "quarantine"
            quarantine_path = quarantine_directory / (
                f"{stream_name}.{recovered_hash}.trailing-bytes.bin"
            )

        event = {
            "schema_name": "part1_audit_event",
            "schema_version": "1.0.0",
            "event_id": audit_event_id(
                None,
                "trailing_line_recovered",
                event_sequence,
                study_id_value=self.study_id,
                model_run_id_value=self.model_run_id,
                shard_id=self.shard_id,
            ),
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

        if parsed.unterminated_valid_record:
            self._durable_append(path, b"\n")
        else:
            assert parsed.trailing_tail is not None
            assert parsed.trailing_offset is not None
            tail = parsed.trailing_tail
            assert quarantine_path is not None
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            if quarantine_path.exists():
                if quarantine_path.read_bytes() != tail:
                    raise Part1StoreError("existing quarantine artifact has conflicting bytes")
            else:
                with quarantine_path.open("xb") as handle:
                    handle.write(tail)
                    handle.flush()
                    os.fsync(handle.fileno())
            with path.open("r+b") as handle:
                handle.truncate(parsed.trailing_offset)
                handle.flush()
                os.fsync(handle.fileno())
        self._durable_append(self.audit_events_path, _json_line_bytes(event))
        return quarantine_path

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
        try:
            inspection = self.inspect()
        except (MalformedMiddleError, InvalidRecordError) as exc:
            parse_error = exc

        if parse_error is None:
            assert inspection is not None
            tail_count = len(inspection.trailing_tails) + len(inspection.unterminated_streams)
            checks.extend(
                [
                    self._check(
                        "json_syntax",
                        "failed" if inspection.trailing_tails else "passed",
                        invalid_trailing_streams=sorted(inspection.trailing_tails),
                    ),
                    self._check("schema_validity", "passed", records_validated=sum(
                        len(records)
                        for records in (
                            inspection.natural_results,
                            inspection.checkpoint_results,
                            inspection.audit_events,
                        )
                    )),
                    self._check("malformed_middle", "passed", malformed_middle_count=0),
                    self._check(
                        "trailing_tail_state",
                        "failed" if tail_count else "passed",
                        invalid_tails=sorted(inspection.trailing_tails),
                        valid_records_missing_newline=sorted(inspection.unterminated_streams),
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
            )
            warnings = len(index.missing_completion_record_ids)
            consistency_outcome = "failed" if failures else ("warning" if warnings else "passed")
            consistency_details = {
                "missing_started_attempt_ids": sorted(index.missing_started_attempt_ids),
                "inconsistent_completion_attempt_ids": sorted(
                    index.inconsistent_completion_attempt_ids
                ),
                "orphaned_attempt_ids": sorted(index.orphaned_attempt_ids),
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
        report_id = hashlib.sha256(canonical_json_bytes(report_without_id)).hexdigest()
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
        report = self.validate_shard(
            artifact_kind=artifact_kind,
            started_at=started_at,
            completed_at=completed_at,
        )
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as handle:
            handle.write(_json_line_bytes(report)[:-1])
            handle.flush()
            os.fsync(handle.fileno())
        return report

    def finalize(self) -> None:
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
        with self.finalization_path.open("xb") as handle:
            handle.write(_json_line_bytes(marker)[:-1])
            handle.flush()
            os.fsync(handle.fileno())
