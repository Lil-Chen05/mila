"""Read-only, login-safe production coverage validation for Part 1.

This module imports no dataset, model, tokenizer, torch, or CUDA code.  The
pure classifier is intentionally separate from the on-disk scanner so its
fixed 5,000/55,000 accounting can be tested without manufacturing sixty
thousand raw records for every edge case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from part1_contract import (
    attempt_id,
    canonical_json_bytes,
    checkpoint_record_id,
    derive_generation_seed,
    model_run_id,
    model_run_manifest_hash,
    natural_record_id,
    validate_fixed_model_requested_contract,
    validate_instance,
)
from part1_manifests import load_manifest_bundle
from part1_runtime import validate_manifest_compatibility
from part1_store import Part1ShardStore, STORE_VERSION


COVERAGE_VALIDATOR_VERSION = "part1-production-coverage-v1"
COVERAGE_IDENTITY_VERSION = "part1-production-coverage-identity-v1"
EXPECTED_QUESTION_COUNT = 500
EXPECTED_NATURAL_COUNT = 5_000
EXPECTED_CHECKPOINT_COUNT = 55_000
CHECKPOINT_IDS = tuple(f"cp-{index:02d}" for index in range(11))

NATURAL_PARTITION_KEYS = (
    "complete",
    "terminal_infrastructure_failure",
    "retryable_incomplete",
    "missing",
    "duplicate",
    "schema_incompatible",
    "manifest_incompatible",
)
CHECKPOINT_PARTITION_KEYS = (
    "complete",
    "terminal_infrastructure_failure",
    "retryable_incomplete",
    "ineligible",
    "missing",
    "duplicate",
)
NATURAL_ATTRIBUTE_VALUES = {
    "stop_reason": ("eos", "max_new_tokens", "stopping_criterion", "error", "other"),
    "reasoning_status": ("closed", "missing_close", "no_reasoning", "malformed"),
    "answer_parse_status": ("parsed", "missing", "malformed", "out_of_domain"),
    "confidence_parse_status": ("parsed", "missing", "malformed", "out_of_range"),
}
CHECKPOINT_ATTRIBUTE_VALUES = {
    "checkpoint_model_output_status": ("valid", "invalid"),
    "answer_parse_status": ("parsed", "missing", "malformed", "out_of_domain"),
    "confidence_parse_status": ("parsed", "missing", "malformed", "out_of_range"),
    "answer_token_status": ("located", "missing", "ambiguous", "unsupported"),
    "entropy_status": ("computed", "unavailable", "invalid"),
}
GLOBAL_SOURCE_SPECS = {
    "questions": "manifests/part1/questions.jsonl",
    "question_manifest": "manifests/part1/questions.manifest.json",
    "study_manifest": "manifests/part1/study_manifest.json",
    "model_run_manifest": None,
    "dependency_lock": "uv.lock",
}
CORE_SHARD_SOURCE_SPECS = {
    "shard_provenance": ".shard-provenance.json",
    "natural_results": "natural_results.jsonl",
    "checkpoint_results": "checkpoint_results.jsonl",
    "audit_events": "audit_events.jsonl",
    "finalization_marker": ".finalized",
}
AUXILIARY_SHARD_SOURCE_SPECS = {
    "runtime_guard": (".writer.guard",),
    "takeover_evidence": (".writer-lock-takeover-event.json",),
    "recovery_evidence": ("recovery_journal",),
    "quarantined_bytes": ("quarantine",),
    "lock_history": (".lock_history",),
}
UNEXPECTED_SOURCE_KINDS = frozenset({"unexpected_raw_entry", "unexpected_file"})

NaturalLogicalKey = tuple[str, int]
CheckpointLogicalKey = tuple[str, int, str]


@dataclass(frozen=True)
class RecordObservation:
    """One assignable physical record and its validation disposition."""

    record: Mapping[str, Any] | None
    defect: str | None
    reason: str | None
    source: str

    @classmethod
    def valid(cls, record: Mapping[str, Any], *, source: str) -> "RecordObservation":
        return cls(record=dict(record), defect=None, reason=None, source=source)

    @classmethod
    def schema_incompatible(cls, reason: str, *, source: str) -> "RecordObservation":
        return cls(record=None, defect="schema_incompatible", reason=reason, source=source)

    @classmethod
    def manifest_incompatible(
        cls, reason: str, *, source: str
    ) -> "RecordObservation":
        return cls(record=None, defect="manifest_incompatible", reason=reason, source=source)


@dataclass(frozen=True)
class ShardScan:
    natural_observations: Mapping[NaturalLogicalKey, tuple[RecordObservation, ...]]
    checkpoint_observations: Mapping[
        CheckpointLogicalKey, tuple[RecordObservation, ...]
    ]
    natural_lifecycle_keys: frozenset[NaturalLogicalKey]
    checkpoint_lifecycle_keys: frozenset[CheckpointLogicalKey]
    structural_errors: tuple[str, ...]
    structural_warnings: tuple[str, ...]
    unassignable_physical_record_count: int
    source_files: tuple[dict[str, Any], ...]


def _matrix(
    records: Iterable[Mapping[str, Any]], dimensions: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    names = tuple(dimensions)
    counts: dict[tuple[str, ...], int] = {}
    for record in records:
        key = tuple(str(record[name]) for name in names)
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for combination in itertools.product(*(dimensions[name] for name in names)):
        rows.append(
            {
                **dict(zip(names, combination, strict=True)),
                "count": counts.get(tuple(combination), 0),
            }
        )
    return {"dimensions": list(names), "rows": rows}


def _natural_disposition(observations: Sequence[RecordObservation], attempted: bool) -> str:
    if len(observations) > 1:
        return "duplicate"
    if not observations:
        return "retryable_incomplete" if attempted else "missing"
    observation = observations[0]
    if observation.defect == "schema_incompatible":
        return "schema_incompatible"
    if observation.defect == "manifest_incompatible":
        return "manifest_incompatible"
    assert observation.record is not None
    outcome = observation.record.get("natural_execution_outcome")
    if outcome not in {"complete", "terminal_infrastructure_failure"}:
        return "schema_incompatible"
    return str(outcome)


def _checkpoint_disposition(
    observations: Sequence[RecordObservation], attempted: bool
) -> str:
    if len(observations) > 1:
        return "duplicate"
    if not observations:
        return "retryable_incomplete" if attempted else "missing"
    observation = observations[0]
    if observation.defect is not None or observation.record is None:
        return "retryable_incomplete" if attempted else "missing"
    outcome = observation.record.get("checkpoint_execution_outcome")
    if outcome not in {"complete", "terminal_infrastructure_failure"}:
        return "retryable_incomplete" if attempted else "missing"
    return str(outcome)


def classify_logical_coverage(
    *,
    question_ids: Sequence[str],
    natural_observations: Mapping[NaturalLogicalKey, Sequence[RecordObservation]],
    checkpoint_observations: Mapping[
        CheckpointLogicalKey, Sequence[RecordObservation]
    ],
    natural_lifecycle_keys: Iterable[NaturalLogicalKey] = (),
    checkpoint_lifecycle_keys: Iterable[CheckpointLogicalKey] = (),
    structural_errors: Iterable[str] = (),
    structural_warnings: Iterable[str] = (),
    unexpected_physical_record_count: int = 0,
) -> dict[str, Any]:
    """Partition every fixed logical key with deterministic defect precedence."""

    if len(question_ids) != EXPECTED_QUESTION_COUNT or len(set(question_ids)) != len(
        question_ids
    ):
        raise ValueError("coverage classification requires exactly 500 unique question IDs")
    expected_natural = {
        (question_id, run_id) for question_id in question_ids for run_id in range(10)
    }
    expected_checkpoints = {
        (question_id, run_id, checkpoint_id)
        for question_id in question_ids
        for run_id in range(10)
        for checkpoint_id in CHECKPOINT_IDS
    }
    natural_lifecycle = set(natural_lifecycle_keys)
    checkpoint_lifecycle = set(checkpoint_lifecycle_keys)
    errors = list(structural_errors)
    warnings = list(structural_warnings)
    unexpected = int(unexpected_physical_record_count)
    if unexpected < 0:
        raise ValueError("unexpected physical record count cannot be negative")

    for key, observations in natural_observations.items():
        if key not in expected_natural:
            errors.append(f"unexpected natural logical key: {key!r}")
            unexpected += len(observations)
    for key, observations in checkpoint_observations.items():
        if key not in expected_checkpoints:
            errors.append(f"unexpected checkpoint logical key: {key!r}")
            unexpected += len(observations)

    natural_partition = {key: 0 for key in NATURAL_PARTITION_KEYS}
    natural_categories: dict[NaturalLogicalKey, str] = {}
    natural_complete_records: list[Mapping[str, Any]] = []
    for key in sorted(expected_natural):
        observations = tuple(natural_observations.get(key, ()))
        category = _natural_disposition(observations, key in natural_lifecycle)
        natural_categories[key] = category
        natural_partition[category] += 1
        if category == "complete":
            assert len(observations) == 1 and observations[0].record is not None
            natural_complete_records.append(observations[0].record)
        elif category in {
            "duplicate",
            "schema_incompatible",
            "manifest_incompatible",
        }:
            errors.append(f"natural {key!r} is {category}")

    checkpoint_partition = {key: 0 for key in CHECKPOINT_PARTITION_KEYS}
    checkpoint_complete_records: list[Mapping[str, Any]] = []
    for key in sorted(expected_checkpoints):
        parent_key = key[:2]
        parent_category = natural_categories[parent_key]
        observations = tuple(checkpoint_observations.get(key, ()))
        if parent_category == "terminal_infrastructure_failure":
            category = "ineligible"
            if observations:
                errors.append(
                    f"checkpoint {key!r} has physical data beneath an ineligible parent"
                )
                unexpected += len(observations)
        elif parent_category != "complete":
            category = (
                "duplicate"
                if len(observations) > 1
                else "retryable_incomplete"
                if key in checkpoint_lifecycle
                else "missing"
            )
            if observations:
                errors.append(
                    f"checkpoint {key!r} has physical data beneath a nonterminal or incompatible parent"
                )
                unexpected += len(observations)
        else:
            category = _checkpoint_disposition(
                observations, key in checkpoint_lifecycle
            )
            if len(observations) > 1:
                errors.append(f"checkpoint {key!r} is duplicate")
            elif observations and observations[0].defect is not None:
                errors.append(
                    f"checkpoint {key!r} is {observations[0].defect}: "
                    f"{observations[0].reason}"
                )
                unexpected += 1
        checkpoint_partition[category] += 1
        if category == "complete":
            assert len(observations) == 1 and observations[0].record is not None
            checkpoint_complete_records.append(observations[0].record)

    if sum(natural_partition.values()) != EXPECTED_NATURAL_COUNT:
        raise AssertionError("natural coverage partition is not exhaustive")
    if sum(checkpoint_partition.values()) != EXPECTED_CHECKPOINT_COUNT:
        raise AssertionError("checkpoint coverage partition is not exhaustive")

    natural_coverage_defects = sum(
        natural_partition[key]
        for key in (
            "retryable_incomplete",
            "missing",
            "duplicate",
            "schema_incompatible",
            "manifest_incompatible",
        )
    )
    checkpoint_coverage_defects = sum(
        checkpoint_partition[key]
        for key in ("retryable_incomplete", "missing", "duplicate")
    )
    if unexpected:
        errors.append(f"unexpected physical record count is {unexpected}")
    coverage_complete = natural_coverage_defects == checkpoint_coverage_defects == 0
    structurally_valid = not errors and coverage_complete and unexpected == 0
    terminal_failures = (
        natural_partition["terminal_infrastructure_failure"]
        + checkpoint_partition["terminal_infrastructure_failure"]
    )
    paper_analysis_ready = structurally_valid and coverage_complete and terminal_failures == 0
    return {
        "natural_partition": natural_partition,
        "checkpoint_partition": checkpoint_partition,
        "natural_model_output_matrix": _matrix(
            natural_complete_records, NATURAL_ATTRIBUTE_VALUES
        ),
        "checkpoint_model_output_matrix": _matrix(
            checkpoint_complete_records, CHECKPOINT_ATTRIBUTE_VALUES
        ),
        "natural_execution_complete_count": len(natural_complete_records),
        "checkpoint_execution_complete_count": len(checkpoint_complete_records),
        "unexpected_physical_record_count": unexpected,
        "structural_errors": errors,
        "structural_warnings": warnings,
        "structurally_valid": structurally_valid,
        "coverage_complete": coverage_complete,
        "paper_analysis_ready": paper_analysis_ready,
    }


def coverage_report_id(report: Mapping[str, Any]) -> str:
    """Hash only immutable input identities and exact source-file states."""

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("coverage report summary is missing")
    payload = {
        "identity_type": "production_coverage_report_id",
        "identity_version": COVERAGE_IDENTITY_VERSION,
        "payload": {
            "study_id": report.get("study_id"),
            "study_manifest_hash": summary.get("study_manifest_hash"),
            "question_manifest_hash": summary.get("question_manifest_hash"),
            "model_run_id": report.get("model_run_id"),
            "model_run_manifest_hash": report.get("model_run_manifest_hash"),
            "source_files": sorted(
                (dict(item) for item in report.get("source_files", ())),
                key=lambda item: item["relative_path"],
            ),
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _validate_partition(
    value: Mapping[str, Any],
    *,
    expected_keys: Sequence[str],
    expected_total: int,
    label: str,
) -> None:
    if set(value) != set(expected_keys):
        raise ValueError(f"{label} partition keys differ")
    total = sum(
        _nonnegative_integer(value[key], label=f"{label} partition {key}")
        for key in expected_keys
    )
    if total != expected_total:
        raise ValueError(
            f"{label} partition must total {expected_total}, observed {total}"
        )


def _validate_matrix_semantics(
    value: Mapping[str, Any],
    *,
    dimensions: Mapping[str, Sequence[str]],
    expected_total: int,
    label: str,
) -> None:
    dimension_names = tuple(dimensions)
    if value["dimensions"] != list(dimension_names):
        raise ValueError(f"{label} matrix dimensions or deterministic order differ")
    expected_combinations = list(
        itertools.product(*(dimensions[name] for name in dimension_names))
    )
    rows = value["rows"]
    if len(rows) != len(expected_combinations):
        raise ValueError(f"{label} matrix does not contain the full Cartesian product")
    count_total = 0
    expected_fields = {*dimension_names, "count"}
    for index, (row, combination) in enumerate(
        zip(rows, expected_combinations, strict=True)
    ):
        if set(row) != expected_fields:
            raise ValueError(f"{label} matrix row {index} fields differ")
        observed_combination = tuple(row[name] for name in dimension_names)
        if observed_combination != combination:
            raise ValueError(
                f"{label} matrix row {index} is not in deterministic Cartesian order"
            )
        count_total += _nonnegative_integer(
            row["count"], label=f"{label} matrix row {index} count"
        )
    if count_total != expected_total:
        raise ValueError(
            f"{label} matrix must total {expected_total}, observed {count_total}"
        )


def _authoritative_source_inventory_defects(
    report: Mapping[str, Any],
) -> list[str]:
    """Validate authoritative merge-source names and return structural defects."""

    source_files = report["source_files"]
    model_run_id_value = report["model_run_id"]
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    expected_globals = dict(GLOBAL_SOURCE_SPECS)
    expected_globals["model_run_manifest"] = (
        f"results/part1/{model_run_id_value}/model_run_manifest.json"
    )
    for kind, expected_path in expected_globals.items():
        matches = [item for item in source_files if item["kind"] == kind]
        if len(matches) != 1:
            raise ValueError(
                f"authoritative global source inventory requires exactly one {kind}"
            )
        item = matches[0]
        if item["relative_path"] != expected_path or item["shard_id"] is not None:
            raise ValueError(
                f"authoritative global source {kind} has a noncanonical path or shard ID"
            )
        if (
            item["state"] != "regular_file"
            or item["byte_size"] == 0
            or item["sha256"] == empty_sha256
        ):
            raise ValueError(
                f"authoritative global source {kind} must be a nonempty regular file"
            )
    dependency_source = next(
        item for item in source_files if item["kind"] == "dependency_lock"
    )
    if dependency_source["sha256"] != report["summary"]["dependency_lock_sha256"]:
        raise ValueError(
            "authoritative dependency-lock source hash differs from report provenance"
        )

    raw_prefix = (
        "results",
        "part1",
        model_run_id_value,
        "raw_shards",
    )
    global_kinds = frozenset(expected_globals)
    core_by_shard: dict[str, dict[str, Mapping[str, Any]]] = {}
    inventory_shard_ids: set[str] = set()
    defects: list[str] = []
    for item in source_files:
        kind = item["kind"]
        if kind in global_kinds:
            continue
        relative_path = item["relative_path"]
        parts = Path(relative_path).parts
        if tuple(parts[:4]) != raw_prefix:
            raise ValueError(
                f"source {kind} is outside the canonical production raw root: {relative_path}"
            )
        if kind == "unexpected_raw_entry":
            defects.append(f"unexpected raw-root source is inventoried: {relative_path}")
            continue

        shard_id_value = item["shard_id"]
        if not isinstance(shard_id_value, str):
            raise ValueError(f"shard source {kind} has no canonical shard ID")
        expected_shard_prefix = (*raw_prefix, shard_id_value)
        if tuple(parts[:5]) != expected_shard_prefix:
            raise ValueError(
                f"source path and shard ID differ for {kind}: {relative_path}"
            )
        inventory_shard_ids.add(shard_id_value)
        suffix = tuple(parts[5:])
        if kind in CORE_SHARD_SOURCE_SPECS:
            expected_suffix = (CORE_SHARD_SOURCE_SPECS[kind],)
            if suffix != expected_suffix:
                raise ValueError(
                    f"core shard source {kind} has a noncanonical path: {relative_path}"
                )
            core_by_shard.setdefault(shard_id_value, {})[kind] = item
            continue
        if kind in AUXILIARY_SHARD_SOURCE_SPECS:
            expected_prefix = AUXILIARY_SHARD_SOURCE_SPECS[kind]
            exact_file = kind in {"runtime_guard", "takeover_evidence"}
            if (
                (exact_file and suffix != expected_prefix)
                or (
                    not exact_file
                    and (len(suffix) < 2 or suffix[:1] != expected_prefix)
                )
            ):
                raise ValueError(
                    f"auxiliary shard source {kind} has a noncanonical path: {relative_path}"
                )
            if item["state"] != "regular_file":
                defects.append(
                    f"auxiliary shard source is not regular: {relative_path}"
                )
            continue
        if kind == "unexpected_file":
            if not suffix:
                raise ValueError(
                    f"unexpected shard source has no in-shard path: {relative_path}"
                )
            defects.append(f"unexpected shard source is inventoried: {relative_path}")
            continue
        raise ValueError(f"unsupported source kind in authoritative inventory: {kind}")

    if report["summary"]["observed"]["shards"] != len(inventory_shard_ids):
        defects.append(
            "observed shard count differs from canonical shard IDs in source inventory"
        )
    required_core_kinds = frozenset(CORE_SHARD_SOURCE_SPECS)
    regular_core_kinds = required_core_kinds.difference({"checkpoint_results"})
    for shard_id_value in sorted(inventory_shard_ids):
        shard_sources = core_by_shard.get(shard_id_value, {})
        missing_kinds = sorted(required_core_kinds.difference(shard_sources))
        if missing_kinds:
            defects.append(
                f"core source inventory for {shard_id_value} is missing {missing_kinds}"
            )
        for kind in regular_core_kinds.intersection(shard_sources):
            if shard_sources[kind]["state"] != "regular_file":
                defects.append(
                    f"core source {kind} for {shard_id_value} is not regular"
                )

    if report["structurally_valid"]:
        expected_shard_ids = {f"shard-{index:03d}" for index in range(500)}
        missing_shards = expected_shard_ids.difference(inventory_shard_ids)
        extra_shards = inventory_shard_ids.difference(expected_shard_ids)
        if missing_shards or extra_shards:
            defects.append(
                "structurally valid source inventory does not contain exactly 500 "
                "canonical shard IDs"
            )
        for shard_id_value in expected_shard_ids.intersection(inventory_shard_ids):
            checkpoint_source = core_by_shard.get(shard_id_value, {}).get(
                "checkpoint_results"
            )
            if checkpoint_source is not None and checkpoint_source["state"] not in {
                "regular_file",
                "absent",
            }:
                defects.append(
                    f"checkpoint source state is invalid for {shard_id_value}"
                )
    return defects


def validate_coverage_report_semantics(report: Mapping[str, Any]) -> None:
    """Recompute production-coverage arithmetic not expressible in JSON Schema."""

    summary = report["summary"]
    natural = summary["natural_partition"]
    checkpoints = summary["checkpoint_partition"]
    _validate_partition(
        natural,
        expected_keys=NATURAL_PARTITION_KEYS,
        expected_total=EXPECTED_NATURAL_COUNT,
        label="natural",
    )
    _validate_partition(
        checkpoints,
        expected_keys=CHECKPOINT_PARTITION_KEYS,
        expected_total=EXPECTED_CHECKPOINT_COUNT,
        label="checkpoint",
    )
    if checkpoints["ineligible"] != 11 * natural["terminal_infrastructure_failure"]:
        raise ValueError(
            "checkpoint ineligible count must equal eleven per terminal natural run"
        )

    outcomes = summary["outcome_counts"]
    expected_outcomes = {
        "natural_execution_complete": natural["complete"],
        "natural_terminal_infrastructure_failure": natural[
            "terminal_infrastructure_failure"
        ],
        "checkpoint_execution_complete": checkpoints["complete"],
        "checkpoint_terminal_infrastructure_failure": checkpoints[
            "terminal_infrastructure_failure"
        ],
        "checkpoint_ineligible": checkpoints["ineligible"],
    }
    if outcomes != expected_outcomes:
        raise ValueError("coverage outcome counts differ from logical partitions")

    _validate_matrix_semantics(
        summary["natural_model_output_matrix"],
        dimensions=NATURAL_ATTRIBUTE_VALUES,
        expected_total=natural["complete"],
        label="natural model-output",
    )
    _validate_matrix_semantics(
        summary["checkpoint_model_output_matrix"],
        dimensions=CHECKPOINT_ATTRIBUTE_VALUES,
        expected_total=checkpoints["complete"],
        label="checkpoint model-output",
    )

    structural_errors = summary["structural_errors"]
    structural_warnings = summary["structural_warnings"]
    if summary["structural_error_count"] != len(structural_errors):
        raise ValueError("structural error count differs from structural error list")
    if summary["structural_warning_count"] != len(structural_warnings):
        raise ValueError("structural warning count differs from structural warning list")

    source_files = report["source_files"]
    relative_paths = [item["relative_path"] for item in source_files]
    if relative_paths != sorted(relative_paths):
        raise ValueError("coverage source inventory is not in deterministic path order")
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("coverage source inventory contains duplicate paths")
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    for item in source_files:
        relative_path = item["relative_path"]
        parsed_path = Path(relative_path)
        if (
            parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or parsed_path.as_posix() != relative_path
        ):
            raise ValueError(
                f"coverage source inventory path is unsafe or noncanonical: {relative_path}"
            )
        if item["state"] == "absent" and (
            item["sha256"] != empty_sha256 or item["byte_size"] != 0
        ):
            raise ValueError(
                f"absent source must use the empty hash and zero bytes: {relative_path}"
            )
    regular_source_count = sum(
        item["state"] == "regular_file" for item in source_files
    )
    if summary["observed"]["source_files"] != regular_source_count:
        raise ValueError("observed source-file count differs from source inventory")
    inventory_defects = _authoritative_source_inventory_defects(report)
    if report["structurally_valid"] and inventory_defects:
        raise ValueError(
            "structurally valid authoritative source inventory has defects: "
            + "; ".join(inventory_defects[:5])
        )

    minimum_natural_physical = (
        natural["complete"]
        + natural["terminal_infrastructure_failure"]
        + natural["schema_incompatible"]
        + natural["manifest_incompatible"]
        + 2 * natural["duplicate"]
    )
    minimum_checkpoint_physical = (
        checkpoints["complete"]
        + checkpoints["terminal_infrastructure_failure"]
        + 2 * checkpoints["duplicate"]
    )
    if summary["observed"]["natural_physical_records"] < minimum_natural_physical:
        raise ValueError("observed natural physical-record count is arithmetically impossible")
    if (
        summary["observed"]["checkpoint_physical_records"]
        < minimum_checkpoint_physical
    ):
        raise ValueError(
            "observed checkpoint physical-record count is arithmetically impossible"
        )
    natural_terminal_physical = (
        natural["complete"] + natural["terminal_infrastructure_failure"]
    )
    checkpoint_terminal_physical = (
        checkpoints["complete"] + checkpoints["terminal_infrastructure_failure"]
    )
    observed_natural_physical = summary["observed"]["natural_physical_records"]
    observed_checkpoint_physical = summary["observed"][
        "checkpoint_physical_records"
    ]
    if report["structurally_valid"] and (
        observed_natural_physical != natural_terminal_physical
    ):
        raise ValueError(
            "structurally valid natural physical-record count must exactly equal "
            "complete plus terminal-failure outcomes"
        )
    if report["structurally_valid"] and (
        observed_checkpoint_physical != checkpoint_terminal_physical
    ):
        raise ValueError(
            "structurally valid checkpoint physical-record count must exactly equal "
            "complete plus terminal-failure outcomes"
        )
    if observed_natural_physical > natural_terminal_physical and not (
        natural["duplicate"]
        or summary["unexpected_physical_record_count"]
        or structural_errors
    ):
        raise ValueError(
            "surplus natural physical-record count requires duplicate, unexpected, "
            "or structural defects"
        )
    if observed_checkpoint_physical > checkpoint_terminal_physical and not (
        checkpoints["duplicate"]
        or summary["unexpected_physical_record_count"]
        or structural_errors
    ):
        raise ValueError(
            "surplus checkpoint physical-record count requires duplicate, unexpected, "
            "or structural defects"
        )

    natural_defects = sum(
        natural[key]
        for key in (
            "retryable_incomplete",
            "missing",
            "duplicate",
            "schema_incompatible",
            "manifest_incompatible",
        )
    )
    checkpoint_defects = sum(
        checkpoints[key] for key in ("retryable_incomplete", "missing", "duplicate")
    )
    expected_coverage = natural_defects == checkpoint_defects == 0
    if report["coverage_complete"] != expected_coverage:
        raise ValueError("coverage_complete differs from logical partition arithmetic")

    unexpected = summary["unexpected_physical_record_count"]
    provenance_consistent = (
        summary["observed_git_commit"] == summary["final_production_git_commit"]
        and summary["clean_tracked_worktree"]
        and summary["historical_layout_detected"] is None
        and summary["observed"]["shards"] == summary["expected"]["shards"]
        and not inventory_defects
    )
    if (unexpected or not provenance_consistent or inventory_defects) and not structural_errors:
        raise ValueError(
            "structural errors must explain unexpected data or provenance defects"
        )
    expected_structural = (
        expected_coverage
        and not structural_errors
        and unexpected == 0
        and provenance_consistent
    )
    if report["structurally_valid"] != expected_structural:
        raise ValueError("structurally_valid differs from coverage/provenance arithmetic")
    if report["is_valid"] != expected_structural:
        raise ValueError("coverage report is_valid must equal structurally_valid")

    terminal_failures = (
        natural["terminal_infrastructure_failure"]
        + checkpoints["terminal_infrastructure_failure"]
    )
    expected_paper_ready = expected_structural and terminal_failures == 0
    if report["paper_analysis_ready"] != expected_paper_ready:
        raise ValueError("paper_analysis_ready differs from terminal-failure arithmetic")

    expected_checks = (
        (
            "provenance_paths_and_sources",
            "passed" if expected_structural else "failed",
        ),
        ("logical_coverage", "passed" if expected_coverage else "failed"),
        (
            "paper_analysis_readiness",
            "passed"
            if expected_paper_ready
            else "warning"
            if expected_structural and expected_coverage
            else "failed",
        ),
    )
    observed_checks = tuple(
        (check["name"], check["outcome"]) for check in report["checks"]
    )
    if observed_checks != expected_checks:
        raise ValueError("coverage checks or their deterministic outcomes differ")
    failed_count = sum(check["outcome"] == "failed" for check in report["checks"])
    warning_count = sum(check["outcome"] == "warning" for check in report["checks"])
    if report["error_count"] != failed_count:
        raise ValueError("validation error_count differs from failed checks")
    if report["warning_count"] != warning_count:
        raise ValueError("validation warning_count differs from warning checks")
    if report["validated_artifact_identity"] != report["model_run_id"]:
        raise ValueError("coverage artifact identity must equal model_run_id")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_safe_existing_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    components = [absolute]
    cursor = absolute
    while cursor != cursor.parent:
        cursor = cursor.parent
        components.append(cursor)
    for component in reversed(components):
        if not os.path.lexists(component):
            continue
        mode = component.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"publication path contains a symlink: {component}")
        if component != absolute and not stat.S_ISDIR(mode):
            raise ValueError(f"publication path parent is not a directory: {component}")


def _ensure_safe_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    _require_safe_existing_components(absolute)
    missing: list[Path] = []
    cursor = absolute
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)
    _require_safe_existing_components(absolute)
    if not absolute.is_dir() or absolute.is_symlink():
        raise ValueError(f"publication directory is unsafe: {absolute}")


def publish_coverage_report(
    report: Mapping[str, Any],
    path: Path,
    *,
    repository_root: Path,
) -> None:
    """Fsync a same-directory stage and atomically replace the report."""

    value = dict(report)
    validate_instance("validation_report", value)
    if value["is_valid"] != value["structurally_valid"]:
        raise ValueError("coverage report is_valid must equal structurally_valid")
    if value["paper_analysis_ready"] and not (
        value["structurally_valid"] and value["coverage_complete"]
    ):
        raise ValueError(
            "paper_analysis_ready requires structural validity and complete coverage"
        )
    validate_coverage_report_semantics(value)
    expected_id = coverage_report_id(value)
    if value["validation_report_id"] != expected_id:
        raise ValueError("coverage validation report ID differs from immutable inputs")
    target = Path(os.path.abspath(path))
    normalized_repository_root = Path(os.path.abspath(repository_root))
    _relative_path(normalized_repository_root, target)
    snapshot_errors = _global_snapshot_errors(
        repository_root=normalized_repository_root,
        source_files=value["source_files"],
        expected_git_commit=value["summary"]["observed_git_commit"],
        expected_clean_tracked=value["summary"]["clean_tracked_worktree"],
    )
    if snapshot_errors:
        raise ValueError(
            "coverage input snapshot changed before publication: "
            + "; ".join(snapshot_errors)
        )
    _ensure_safe_directory(target.parent)
    if os.path.lexists(target):
        mode = target.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"coverage report target is a symlink: {target}")
        if not stat.S_ISREG(mode):
            raise ValueError(f"coverage report target is not a regular file: {target}")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        snapshot_errors = _global_snapshot_errors(
            repository_root=normalized_repository_root,
            source_files=value["source_files"],
            expected_git_commit=value["summary"]["observed_git_commit"],
            expected_clean_tracked=value["summary"]["clean_tracked_worktree"],
        )
        if snapshot_errors:
            raise ValueError(
                "coverage input snapshot changed during publication: "
                + "; ".join(snapshot_errors)
            )
        os.replace(staged, target)
        _fsync_directory(target.parent)
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _relative_path(repository_root: Path, path: Path) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(
            Path(os.path.abspath(repository_root))
        ).as_posix()
    except ValueError as exc:
        raise ValueError(f"source path escapes the repository: {path}") from exc


def _read_source(
    *,
    repository_root: Path,
    path: Path,
    shard_id: str | None,
    kind: str,
    allow_absent: bool,
) -> tuple[dict[str, Any], bytes | None, str | None]:
    relative = _relative_path(repository_root, path)
    if not os.path.lexists(path):
        if not allow_absent:
            return (
                {
                    "relative_path": relative,
                    "shard_id": shard_id,
                    "kind": kind,
                    "state": "absent",
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "byte_size": 0,
                },
                None,
                f"required {kind} is missing: {relative}",
            )
        return (
            {
                "relative_path": relative,
                "shard_id": shard_id,
                "kind": kind,
                "state": "absent",
                "sha256": hashlib.sha256(b"").hexdigest(),
                "byte_size": 0,
            },
            None,
            None,
        )
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return (
            {
                "relative_path": relative,
                "shard_id": shard_id,
                "kind": kind,
                "state": "absent",
                "sha256": hashlib.sha256(b"").hexdigest(),
                "byte_size": 0,
            },
            None,
            f"{kind} is a symlink: {relative}",
        )
    if not stat.S_ISREG(mode):
        return (
            {
                "relative_path": relative,
                "shard_id": shard_id,
                "kind": kind,
                "state": "absent",
                "sha256": hashlib.sha256(b"").hexdigest(),
                "byte_size": 0,
            },
            None,
            f"{kind} is not a regular file: {relative}",
        )
    data = path.read_bytes()
    return (
        {
            "relative_path": relative,
            "shard_id": shard_id,
            "kind": kind,
            "state": "regular_file",
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        },
        data,
        None,
    )


def _json_object(data: bytes | None, *, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if data is None:
        return None, None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{label} is invalid JSON: {exc}"
    if not isinstance(value, dict):
        return None, f"{label} is not a JSON object"
    return value, None


def _parse_jsonl(data: bytes | None, *, label: str) -> tuple[list[dict[str, Any]], list[str], int]:
    if data is None or data == b"":
        return [], [], 0
    errors: list[str] = []
    unassignable = 0
    if not data.endswith(b"\n"):
        errors.append(f"{label} is missing its terminal newline")
    records: list[dict[str, Any]] = []
    for line_number, payload in enumerate(data.splitlines(), start=1):
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{label}:{line_number} is malformed JSON: {exc}")
            unassignable += 1
            continue
        if not isinstance(value, dict):
            errors.append(f"{label}:{line_number} is not a JSON object")
            unassignable += 1
            continue
        value["__coverage_source"] = f"{label}:{line_number}"
        records.append(value)
    return records, errors, unassignable


def _observation_key(record: Mapping[str, Any], *, checkpoint: bool) -> tuple[Any, ...] | None:
    question_id = record.get("question_id")
    run_id = record.get("run_id")
    if not isinstance(question_id, str) or isinstance(run_id, bool) or not isinstance(run_id, int):
        return None
    if checkpoint:
        checkpoint_id = record.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            return None
        return (question_id, run_id, checkpoint_id)
    return (question_id, run_id)


def _common_record_compatibility(
    record: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
) -> list[str]:
    errors = []
    expected = {
        "study_id": model_manifest["study_id"],
        "model_run_id": model_manifest["model_run_id"],
        "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
        "question_manifest_hash": model_manifest["question_manifest_hash"],
        "question_id": question["question_id"],
        "sample_index": question["sample_index"],
        "subject": question["subject"],
        "seed_algorithm_version": model_manifest["seed_algorithm_version"],
    }
    for field, value in expected.items():
        if field in record and record.get(field) != value:
            errors.append(f"{field} differs from authoritative manifest")
    return errors


def _natural_compatibility(
    record: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
) -> list[str]:
    errors = _common_record_compatibility(
        record, question=question, model_manifest=model_manifest
    )
    run_id = int(record["run_id"])
    expected_seed = derive_generation_seed(
        base_seed=model_manifest["base_generation_seed"],
        canonical_model_identity=model_manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=run_id,
        algorithm_version=model_manifest["seed_algorithm_version"],
    )
    if record["generation_seed"] != expected_seed:
        errors.append("generation seed differs from canonical derivation")
    if record["raw_record_id"] != natural_record_id(
        model_manifest["study_id"], model_manifest["model_run_id"], question["question_id"], run_id
    ):
        errors.append("natural record identity is invalid")
    expected_attempt = attempt_id(
        model_manifest["study_id"],
        model_manifest["model_run_id"],
        question["question_id"],
        run_id,
        int(record["terminal_attempt_number"]),
    )
    if record["terminal_attempt_id"] != expected_attempt:
        errors.append("natural terminal attempt identity is invalid")
    if record["prompt_hash"] != model_manifest["prompt_hash"]:
        errors.append("natural prompt hash differs from model-run manifest")
    if record["natural_execution_outcome"] == "complete" and tuple(
        record.get("checkpoint_ids") or ()
    ) != CHECKPOINT_IDS:
        errors.append("complete natural checkpoint identities differ from fixed eleven")
    components = record.get("component_versions", {})
    for field, manifest_field in (
        ("adapter", "adapter_version"),
        ("prompt", "prompt_version"),
        ("parser", "parser_version"),
    ):
        if components.get(field) != model_manifest[manifest_field]:
            errors.append(f"natural component {field} differs from model-run manifest")
    return errors


def _checkpoint_compatibility(
    record: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
) -> list[str]:
    errors = _common_record_compatibility(
        record, question=question, model_manifest=model_manifest
    )
    run_id = int(record["run_id"])
    checkpoint_id = str(record["checkpoint_id"])
    expected_seed = derive_generation_seed(
        base_seed=model_manifest["base_generation_seed"],
        canonical_model_identity=model_manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=run_id,
        algorithm_version=model_manifest["seed_algorithm_version"],
    )
    if record["natural_seed"] != expected_seed:
        errors.append("checkpoint natural seed differs from canonical derivation")
    if record["parent_raw_record_id"] != natural_record_id(
        model_manifest["study_id"], model_manifest["model_run_id"], question["question_id"], run_id
    ):
        errors.append("checkpoint parent natural identity is invalid")
    if record["checkpoint_record_id"] != checkpoint_record_id(
        model_manifest["study_id"],
        model_manifest["model_run_id"],
        question["question_id"],
        run_id,
        checkpoint_id,
    ):
        errors.append("checkpoint record identity is invalid")
    expected_attempt = attempt_id(
        model_manifest["study_id"],
        model_manifest["model_run_id"],
        question["question_id"],
        run_id,
        int(record["terminal_attempt_number"]),
        checkpoint_id=checkpoint_id,
    )
    if record["terminal_attempt_id"] != expected_attempt:
        errors.append("checkpoint terminal attempt identity is invalid")
    if checkpoint_id not in CHECKPOINT_IDS or (
        checkpoint_id in CHECKPOINT_IDS
        and record["requested_checkpoint_index"] != CHECKPOINT_IDS.index(checkpoint_id)
    ):
        errors.append("checkpoint identity differs from requested index")
    if record["inducer_version"] != model_manifest["inducer_version"]:
        errors.append("checkpoint inducer version differs from model-run manifest")
    if record["inducer_text"] != model_manifest["inducer_text"]:
        errors.append("checkpoint inducer text differs from model-run manifest")
    if record.get("token_convention") is not None and record["token_convention"] != model_manifest[
        "ad_token_convention"
    ]:
        errors.append("checkpoint A-D token convention differs from model-run manifest")
    if record.get("ad_token_ids") is not None and record["ad_token_ids"] != model_manifest[
        "ad_token_ids"
    ]:
        errors.append("checkpoint A-D token IDs differ from model-run manifest")
    components = record.get("component_versions", {})
    for field, manifest_field in (
        ("adapter", "adapter_version"),
        ("parser", "parser_version"),
        ("inducer", "inducer_version"),
    ):
        if components.get(field) != model_manifest[manifest_field]:
            errors.append(f"checkpoint component {field} differs from model-run manifest")
    return errors


def _collect_observations(
    records: Sequence[dict[str, Any]],
    *,
    checkpoint: bool,
    question: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
) -> tuple[dict[tuple[Any, ...], tuple[RecordObservation, ...]], list[str], int]:
    collected: dict[tuple[Any, ...], list[RecordObservation]] = {}
    errors: list[str] = []
    unassignable = 0
    schema_name = "checkpoint_terminal_result" if checkpoint else "natural_terminal_result"
    for raw in records:
        record = dict(raw)
        source = str(record.pop("__coverage_source"))
        key = _observation_key(record, checkpoint=checkpoint)
        if key is None:
            errors.append(f"{source} has no assignable logical key")
            unassignable += 1
            continue
        try:
            validate_instance(schema_name, record)
        except ValueError as exc:
            observation = RecordObservation.schema_incompatible(str(exc), source=source)
        else:
            if record["question_id"] != question["question_id"]:
                reasons = ["question ID is not assigned to this shard"]
            else:
                reasons = (
                    _checkpoint_compatibility(
                        record, question=question, model_manifest=model_manifest
                    )
                    if checkpoint
                    else _natural_compatibility(
                        record, question=question, model_manifest=model_manifest
                    )
                )
            observation = (
                RecordObservation.manifest_incompatible("; ".join(reasons), source=source)
                if reasons
                else RecordObservation.valid(record, source=source)
            )
        collected.setdefault(key, []).append(observation)
    return {key: tuple(value) for key, value in collected.items()}, errors, unassignable


def _validate_auxiliary_layout(shard_root: Path) -> list[str]:
    allowed_files = {
        ".shard-provenance.json",
        ".writer.guard",
        ".writer-lock-takeover-event.json",
        "natural_results.jsonl",
        "checkpoint_results.jsonl",
        "audit_events.jsonl",
        ".finalized",
    }
    allowed_directories = {"recovery_journal", "quarantine", ".lock_history"}
    forbidden_active_prefixes = (
        ".writer.lock",
        ".writer-lock-recovery.claim",
    )
    errors: list[str] = []
    for entry in shard_root.iterdir():
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode):
            errors.append(f"shard layout contains a symlink: {entry.name}")
            continue
        if entry.name.startswith(forbidden_active_prefixes):
            errors.append(f"finalized shard retains active lock state: {entry.name}")
            continue
        if stat.S_ISREG(mode):
            if entry.name not in allowed_files:
                errors.append(f"unexpected shard file: {entry.name}")
        elif stat.S_ISDIR(mode):
            if entry.name not in allowed_directories:
                errors.append(f"unexpected shard directory: {entry.name}")
                continue
            for descendant in entry.rglob("*"):
                descendant_mode = descendant.lstat().st_mode
                if stat.S_ISLNK(descendant_mode):
                    errors.append(
                        f"shard auxiliary directory contains a symlink: {descendant.relative_to(shard_root)}"
                    )
                elif not (
                    stat.S_ISREG(descendant_mode) or stat.S_ISDIR(descendant_mode)
                ):
                    errors.append(
                        f"shard auxiliary entry is unsafe: {descendant.relative_to(shard_root)}"
                    )
        else:
            errors.append(f"shard layout contains a non-file entry: {entry.name}")
    return errors


def scan_production_shard(
    *,
    repository_root: Path,
    shard_root: Path,
    shard_index: int,
    question: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
) -> ShardScan:
    """Scan one finalized one-question shard without mutation or repair."""

    repository_root = Path(os.path.abspath(repository_root))
    shard_root = Path(os.path.abspath(shard_root))
    shard_id = f"shard-{shard_index:03d}"
    errors: list[str] = []
    warnings: list[str] = []
    unassignable = 0
    if shard_root.name != shard_id:
        errors.append(f"shard directory name differs from expected {shard_id}")
    if not os.path.lexists(shard_root):
        errors.append(f"shard directory is missing: {shard_id}")
        return ShardScan({}, {}, frozenset(), frozenset(), tuple(errors), (), 0, ())
    mode = shard_root.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        errors.append(f"shard root is a symlink or non-directory: {shard_id}")
        return ShardScan({}, {}, frozenset(), frozenset(), tuple(errors), (), 0, ())
    errors.extend(_validate_auxiliary_layout(shard_root))

    source_specs = (
        (".shard-provenance.json", "shard_provenance", False),
        ("natural_results.jsonl", "natural_results", True),
        ("checkpoint_results.jsonl", "checkpoint_results", True),
        ("audit_events.jsonl", "audit_events", True),
        (".finalized", "finalization_marker", False),
    )
    source_files: list[dict[str, Any]] = []
    source_bytes: dict[str, bytes | None] = {}
    for filename, kind, allow_absent in source_specs:
        entry, data, error = _read_source(
            repository_root=repository_root,
            path=shard_root / filename,
            shard_id=shard_id,
            kind=kind,
            allow_absent=allow_absent,
        )
        source_files.append(entry)
        source_bytes[kind] = data
        if error:
            errors.append(error)
    canonical_source_names = {filename for filename, _kind, _allow_absent in source_specs}
    auxiliary_kinds = {
        ".writer.guard": "runtime_guard",
        ".writer-lock-takeover-event.json": "takeover_evidence",
        "recovery_journal": "recovery_evidence",
        "quarantine": "quarantined_bytes",
        ".lock_history": "lock_history",
    }
    for auxiliary in sorted(shard_root.rglob("*")):
        if auxiliary.parent == shard_root and auxiliary.name in canonical_source_names:
            continue
        if auxiliary.is_dir() and not auxiliary.is_symlink():
            continue
        top_level = auxiliary.relative_to(shard_root).parts[0]
        entry, _data, error = _read_source(
            repository_root=repository_root,
            path=auxiliary,
            shard_id=shard_id,
            kind=auxiliary_kinds.get(top_level, "unexpected_file"),
            allow_absent=False,
        )
        source_files.append(entry)
        if error:
            errors.append(error)

    header, header_error = _json_object(
        source_bytes["shard_provenance"], label=f"{shard_id} provenance"
    )
    expected_header = {
        "schema_name": "part1_shard_provenance",
        "schema_version": "1.0.0",
        "study_id": model_manifest["study_id"],
        "model_run_id": model_manifest["model_run_id"],
        "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
        "shard_id": shard_id,
    }
    if header_error:
        errors.append(header_error)
    elif header != expected_header:
        errors.append(f"{shard_id} provenance identity or model-run hash differs")

    marker, marker_error = _json_object(
        source_bytes["finalization_marker"], label=f"{shard_id} finalization marker"
    )
    if marker_error:
        errors.append(marker_error)
    elif marker is not None:
        expected_marker_fields = {
            "store_version": STORE_VERSION,
            "shard_id": shard_id,
            "study_id": model_manifest["study_id"],
            "model_run_id": model_manifest["model_run_id"],
        }
        if set(marker) != {*expected_marker_fields, "finalized_at"} or any(
            marker.get(field) != value for field, value in expected_marker_fields.items()
        ):
            errors.append(f"{shard_id} finalization marker is incompatible")
        else:
            try:
                timestamp = marker["finalized_at"]
                if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
                    raise ValueError("timestamp must use canonical UTC Z form")
                parsed_time = datetime.fromisoformat(
                    timestamp[:-1] + "+00:00"
                )
                if parsed_time.utcoffset() != UTC.utcoffset(parsed_time):
                    raise ValueError("timestamp is not UTC")
            except ValueError as exc:
                errors.append(f"{shard_id} finalization timestamp is invalid: {exc}")

    natural_records, natural_errors, natural_unassignable = _parse_jsonl(
        source_bytes["natural_results"], label=f"{shard_id}/natural_results.jsonl"
    )
    checkpoint_records, checkpoint_errors, checkpoint_unassignable = _parse_jsonl(
        source_bytes["checkpoint_results"], label=f"{shard_id}/checkpoint_results.jsonl"
    )
    audit_records, audit_errors, audit_unassignable = _parse_jsonl(
        source_bytes["audit_events"], label=f"{shard_id}/audit_events.jsonl"
    )
    errors.extend((*natural_errors, *checkpoint_errors, *audit_errors))
    unassignable += natural_unassignable + checkpoint_unassignable + audit_unassignable
    natural_observations, observation_errors, observation_unassignable = _collect_observations(
        natural_records,
        checkpoint=False,
        question=question,
        model_manifest=model_manifest,
    )
    errors.extend(observation_errors)
    unassignable += observation_unassignable
    checkpoint_observations, observation_errors, observation_unassignable = _collect_observations(
        checkpoint_records,
        checkpoint=True,
        question=question,
        model_manifest=model_manifest,
    )
    errors.extend(observation_errors)
    unassignable += observation_unassignable

    natural_lifecycle: set[NaturalLogicalKey] = set()
    checkpoint_lifecycle: set[CheckpointLogicalKey] = set()
    for raw in audit_records:
        event = dict(raw)
        source = str(event.pop("__coverage_source"))
        try:
            validate_instance("audit_event", event)
        except ValueError as exc:
            errors.append(f"{source} schema-incompatible audit event: {exc}")
            continue
        if event["study_id"] != model_manifest["study_id"] or event[
            "model_run_id"
        ] != model_manifest["model_run_id"] or event["shard_id"] != shard_id:
            errors.append(f"{source} audit provenance differs from shard/model-run")
            continue
        if event["event_scope"] == "attempt":
            key = _observation_key(event, checkpoint=event["checkpoint_id"] is not None)
            if key is None:
                errors.append(f"{source} attempt event has no assignable logical key")
                unassignable += 1
            elif event["question_id"] != question["question_id"]:
                errors.append(f"{source} audit question is not assigned to {shard_id}")
                unassignable += 1
            elif event["run_id"] not in range(10):
                errors.append(f"{source} audit run ID is outside the fixed 0-9 range")
                unassignable += 1
            elif event["checkpoint_id"] is not None and event[
                "checkpoint_id"
            ] not in CHECKPOINT_IDS:
                errors.append(f"{source} audit checkpoint ID is outside the fixed eleven")
                unassignable += 1
            elif event["checkpoint_id"] is None:
                natural_lifecycle.add(key)  # type: ignore[arg-type]
            else:
                checkpoint_lifecycle.add(key)  # type: ignore[arg-type]

    # The established store validator independently recomputes terminal/event
    # identities, hierarchy, aliases, and lifecycle.  It is read-only here.
    store = Part1ShardStore(
        shard_root,
        shard_id=shard_id,
        study_id=model_manifest["study_id"],
        model_run_id=model_manifest["model_run_id"],
        model_run_manifest_hash=model_manifest["model_run_manifest_hash"],
    )
    try:
        validation = store.validate_shard(
            artifact_kind="natural_shard",
            started_at="1970-01-01T00:00:00Z",
            completed_at="1970-01-01T00:00:00Z",
        )
    except Exception as exc:
        errors.append(f"{shard_id} store validation failed: {exc}")
    else:
        for check in validation["checks"]:
            if check["outcome"] == "failed":
                errors.append(
                    f"{shard_id} {check['name']} failed: {json.dumps(check['details'], sort_keys=True)}"
                )
            elif check["outcome"] == "warning":
                # A finalized production shard must have no reconciliation or
                # not-evaluated warning left behind.
                errors.append(
                    f"{shard_id} lifecycle/finalization warning in {check['name']}: "
                    f"{json.dumps(check['details'], sort_keys=True)}"
                )
        try:
            index = store.build_index()
        except Exception as exc:
            errors.append(f"{shard_id} index identity/lifecycle validation failed: {exc}")
        else:
            errors.extend(f"{shard_id} hierarchy defect: {item}" for item in index.hierarchy_errors)
            errors.extend(f"{shard_id} lifecycle defect: {item}" for item in index.lifecycle_errors)
            if index.missing_completion_record_ids:
                errors.append(
                    f"{shard_id} lifecycle has authoritative results missing completion: "
                    f"{sorted(index.missing_completion_record_ids)}"
                )
            if index.orphaned_attempt_ids or index.terminalization_required:
                errors.append(f"{shard_id} lifecycle contains incomplete attempts")

    # Detect a concurrent or post-scan source mutation before trusting hashes.
    for entry in source_files:
        path = repository_root / entry["relative_path"]
        if entry["state"] == "absent":
            if os.path.lexists(path):
                errors.append(f"source appeared during validation: {entry['relative_path']}")
        elif not path.is_file() or path.is_symlink():
            errors.append(f"source changed type during validation: {entry['relative_path']}")
        else:
            current = path.read_bytes()
            if len(current) != entry["byte_size"] or hashlib.sha256(current).hexdigest() != entry[
                "sha256"
            ]:
                errors.append(f"source bytes changed during validation: {entry['relative_path']}")

    return ShardScan(
        natural_observations=natural_observations,
        checkpoint_observations=checkpoint_observations,
        natural_lifecycle_keys=frozenset(natural_lifecycle),
        checkpoint_lifecycle_keys=frozenset(checkpoint_lifecycle),
        structural_errors=tuple(errors),
        structural_warnings=tuple(warnings),
        unassignable_physical_record_count=unassignable,
        source_files=tuple(source_files),
    )


def _load_regular_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not os.path.lexists(path):
        raise ValueError(f"{label} is missing: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a non-symlink regular file: {path}")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, data


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _global_snapshot_errors(
    *,
    repository_root: Path,
    source_files: Sequence[Mapping[str, Any]],
    expected_git_commit: str,
    expected_clean_tracked: bool,
) -> list[str]:
    """Revalidate every inventoried byte plus Git HEAD/cleanliness as one snapshot."""

    errors: list[str] = []
    for entry in source_files:
        relative_path = str(entry["relative_path"])
        path = repository_root / relative_path
        try:
            _require_safe_existing_components(path)
        except ValueError as exc:
            errors.append(f"source changed after inventory: {relative_path}: {exc}")
            continue
        if entry["state"] == "absent":
            if os.path.lexists(path):
                errors.append(
                    f"source changed after inventory: {relative_path} "
                    "(expected absent; observed present)"
                )
            continue
        if not os.path.lexists(path):
            errors.append(
                f"source changed after inventory: {relative_path} "
                "(expected regular file; observed absent)"
            )
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            observed_type = "symlink" if stat.S_ISLNK(mode) else "non-regular"
            errors.append(
                f"source changed after inventory: {relative_path} "
                f"(expected regular file; observed {observed_type})"
            )
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(
                f"source changed after inventory: {relative_path} "
                f"(could not reread: {exc})"
            )
            continue
        observed_hash = hashlib.sha256(data).hexdigest()
        if len(data) != entry["byte_size"] or observed_hash != entry["sha256"]:
            errors.append(
                f"source changed after inventory: {relative_path} "
                f"(expected {entry['byte_size']} bytes/{entry['sha256']}; "
                f"observed {len(data)} bytes/{observed_hash})"
            )

    try:
        observed_git_commit = _git(repository_root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"Git HEAD changed after inventory: unable to revalidate ({exc})")
    else:
        if observed_git_commit != expected_git_commit:
            errors.append(
                "Git HEAD changed after inventory "
                f"(expected {expected_git_commit}; observed {observed_git_commit})"
            )
    try:
        clean_output = _git(
            repository_root, "status", "--porcelain", "--untracked-files=no"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(
            f"tracked worktree state changed after inventory: unable to revalidate ({exc})"
        )
    else:
        observed_clean = clean_output == ""
        if observed_clean != expected_clean_tracked:
            errors.append(
                "tracked worktree state changed after inventory "
                f"(expected clean={expected_clean_tracked}; observed clean={observed_clean})"
            )
    return errors


def build_coverage_report(
    *,
    repository_root: Path,
    model_run_manifest_path: Path,
    validation_started_at: str | None = None,
    validation_completed_at: str | None = None,
) -> dict[str, Any]:
    """Validate fixed manifests/provenance and scan all observed raw shards."""

    repository_root = Path(os.path.abspath(repository_root))
    model_run_manifest_path = Path(model_run_manifest_path)
    if not model_run_manifest_path.is_absolute():
        model_run_manifest_path = repository_root / model_run_manifest_path
    started_at = validation_started_at or _now()
    completed_at = validation_completed_at or _now()
    manifest_root = repository_root / "manifests" / "part1"
    tracked_paths = (
        manifest_root / "questions.jsonl",
        manifest_root / "questions.manifest.json",
        manifest_root / "study_manifest.json",
    )
    tracked_initial_bytes: dict[Path, bytes] = {}
    for tracked_path in tracked_paths:
        _require_safe_existing_components(tracked_path)
        if not os.path.lexists(tracked_path):
            raise ValueError(f"tracked manifest source is missing: {tracked_path}")
        tracked_mode = tracked_path.lstat().st_mode
        if stat.S_ISLNK(tracked_mode) or not stat.S_ISREG(tracked_mode):
            raise ValueError(
                f"tracked manifest source must be a non-symlink regular file: {tracked_path}"
            )
        tracked_initial_bytes[tracked_path] = tracked_path.read_bytes()
    bundle = load_manifest_bundle(
        questions_path=manifest_root / "questions.jsonl",
        question_manifest_path=manifest_root / "questions.manifest.json",
        study_manifest_path=manifest_root / "study_manifest.json",
    )
    _require_safe_existing_components(model_run_manifest_path)
    model_manifest, model_manifest_initial_bytes = _load_regular_json(
        model_run_manifest_path, label="production model-run manifest"
    )
    validate_instance("model_run_manifest", model_manifest)
    validate_fixed_model_requested_contract(model_manifest)
    if model_manifest.get("production") is not True or model_manifest.get(
        "execution_scope"
    ) != "production" or model_manifest.get("schema_version") != "1.1.0":
        raise ValueError("coverage validation requires a production schema-1.1 model-run manifest")
    compatibility = validate_manifest_compatibility(bundle.study_manifest, model_manifest)
    expected_model_manifest_path = (
        repository_root
        / "results"
        / "part1"
        / model_manifest["model_run_id"]
        / "model_run_manifest.json"
    )
    if Path(os.path.abspath(model_run_manifest_path)) != Path(
        os.path.abspath(expected_model_manifest_path)
    ):
        raise ValueError("production model-run manifest path is not canonical")
    if model_manifest["model_run_id"] != model_run_id(model_manifest) or model_manifest[
        "model_run_manifest_hash"
    ] != model_run_manifest_hash(model_manifest):
        raise ValueError("production model-run identities do not recompute")

    structural_errors: list[str] = []
    structural_warnings: list[str] = []
    head = _git(repository_root, "rev-parse", "HEAD")
    clean_output = _git(repository_root, "status", "--porcelain", "--untracked-files=no")
    clean_tracked = clean_output == ""
    if head != model_manifest["final_production_git_commit"]:
        structural_errors.append("current Git commit differs from production manifest")
    if not clean_tracked:
        structural_errors.append("tracked worktree is not clean")
    uv_path = repository_root / "uv.lock"
    uv_entry, _uv_bytes, uv_error = _read_source(
        repository_root=repository_root,
        path=uv_path,
        shard_id=None,
        kind="dependency_lock",
        allow_absent=False,
    )
    if uv_error:
        structural_errors.append(uv_error)
    if uv_entry["sha256"] != model_manifest["dependency_lock_sha256"]:
        structural_errors.append("dependency lock hash differs from production manifest")

    expected_raw_relative = f"results/part1/{model_manifest['model_run_id']}/raw_shards"
    expected_validation_relative = f"results/part1/{model_manifest['model_run_id']}/validation"
    if model_manifest["output_paths"]["raw_shards"] != expected_raw_relative or model_manifest[
        "output_paths"
    ]["validation"] != expected_validation_relative:
        raise ValueError("production output paths are not canonical")
    raw_root = repository_root / expected_raw_relative
    observed_names: list[str] = []
    historical_layout: int | None = None
    try:
        _require_safe_existing_components(raw_root)
    except ValueError as exc:
        structural_errors.append(str(exc))
    if not os.path.lexists(raw_root):
        structural_errors.append("production raw shard root is missing")
    elif stat.S_ISLNK(raw_root.lstat().st_mode) or not stat.S_ISDIR(raw_root.lstat().st_mode):
        structural_errors.append("production raw shard root is a symlink or non-directory")
    else:
        for entry in raw_root.iterdir():
            if entry.is_symlink():
                structural_errors.append(f"raw shard root contains a symlink: {entry.name}")
                continue
            observed_names.append(entry.name)
        observed_names.sort()
        if len(observed_names) in {20, 200} and observed_names == [
            f"shard-{index:03d}" for index in range(len(observed_names))
        ]:
            historical_layout = len(observed_names)
            structural_errors.append(
                f"historical {historical_layout}-shard pilot layout is forbidden"
            )
        expected_names = {f"shard-{index:03d}" for index in range(500)}
        unexpected_names = sorted(set(observed_names).difference(expected_names))
        if unexpected_names:
            structural_errors.append(f"unexpected raw shard entries: {unexpected_names}")
        missing_count = len(expected_names.difference(observed_names))
        if missing_count:
            structural_errors.append(f"production raw shard layout is missing {missing_count} shards")

    natural_observations: dict[NaturalLogicalKey, list[RecordObservation]] = {}
    checkpoint_observations: dict[CheckpointLogicalKey, list[RecordObservation]] = {}
    natural_lifecycle: set[NaturalLogicalKey] = set()
    checkpoint_lifecycle: set[CheckpointLogicalKey] = set()
    source_files: list[dict[str, Any]] = []
    unassignable = 0
    if raw_root.is_dir() and not raw_root.is_symlink():
        expected_names = {f"shard-{index:03d}" for index in range(500)}
        for entry in sorted(raw_root.iterdir()):
            valid_expected_directory = (
                entry.name in expected_names and entry.is_dir() and not entry.is_symlink()
            )
            if valid_expected_directory:
                continue
            candidates = (
                [entry]
                if not entry.is_dir() or entry.is_symlink()
                else [candidate for candidate in entry.rglob("*") if not candidate.is_dir()]
            )
            for candidate in candidates:
                inventory_entry, _data, error = _read_source(
                    repository_root=repository_root,
                    path=candidate,
                    shard_id=None,
                    kind="unexpected_raw_entry",
                    allow_absent=False,
                )
                source_files.append(inventory_entry)
                if error:
                    structural_errors.append(error)
    question_by_index = {record["sample_index"]: record for record in bundle.records}
    for shard_index in range(500):
        shard_name = f"shard-{shard_index:03d}"
        if shard_name not in observed_names:
            continue
        scan = scan_production_shard(
            repository_root=repository_root,
            shard_root=raw_root / shard_name,
            shard_index=shard_index,
            question=question_by_index[shard_index],
            model_manifest=model_manifest,
        )
        for key, observations in scan.natural_observations.items():
            natural_observations.setdefault(key, []).extend(observations)
        for key, observations in scan.checkpoint_observations.items():
            checkpoint_observations.setdefault(key, []).extend(observations)
        natural_lifecycle.update(scan.natural_lifecycle_keys)
        checkpoint_lifecycle.update(scan.checkpoint_lifecycle_keys)
        structural_errors.extend(scan.structural_errors)
        structural_warnings.extend(scan.structural_warnings)
        unassignable += scan.unassignable_physical_record_count
        source_files.extend(scan.source_files)

    manifest_sources = (
        (manifest_root / "questions.jsonl", "questions"),
        (manifest_root / "questions.manifest.json", "question_manifest"),
        (manifest_root / "study_manifest.json", "study_manifest"),
        (model_run_manifest_path, "model_run_manifest"),
    )
    for path, kind in manifest_sources:
        entry, data, error = _read_source(
            repository_root=repository_root,
            path=path,
            shard_id=None,
            kind=kind,
            allow_absent=False,
        )
        source_files.append(entry)
        if error:
            structural_errors.append(error)
        initial = (
            model_manifest_initial_bytes
            if path == model_run_manifest_path
            else tracked_initial_bytes.get(path)
        )
        if data is not None and initial is not None and data != initial:
            structural_errors.append(
                f"manifest source bytes changed during validation: {entry['relative_path']}"
            )
    source_files.append(uv_entry)

    snapshot_errors = _global_snapshot_errors(
        repository_root=repository_root,
        source_files=source_files,
        expected_git_commit=head,
        expected_clean_tracked=clean_tracked,
    )
    structural_errors.extend(snapshot_errors)

    classification = classify_logical_coverage(
        question_ids=[record["question_id"] for record in bundle.records],
        natural_observations={key: tuple(value) for key, value in natural_observations.items()},
        checkpoint_observations={
            key: tuple(value) for key, value in checkpoint_observations.items()
        },
        natural_lifecycle_keys=natural_lifecycle,
        checkpoint_lifecycle_keys=checkpoint_lifecycle,
        structural_errors=structural_errors,
        structural_warnings=structural_warnings,
        unexpected_physical_record_count=unassignable,
    )
    observed_shards = sum(name.startswith("shard-") for name in observed_names)
    summary = {
        "question_manifest_hash": bundle.question_manifest["question_manifest_hash"],
        "study_manifest_hash": bundle.study_manifest["study_manifest_hash"],
        "final_production_git_commit": model_manifest["final_production_git_commit"],
        "observed_git_commit": head,
        "dependency_lock_sha256": uv_entry["sha256"],
        "clean_tracked_worktree": clean_tracked,
        "expected": {
            "questions": EXPECTED_QUESTION_COUNT,
            "shards": 500,
            "natural_logical_keys": EXPECTED_NATURAL_COUNT,
            "checkpoint_logical_keys": EXPECTED_CHECKPOINT_COUNT,
        },
        "observed": {
            "shards": observed_shards,
            "natural_physical_records": sum(len(value) for value in natural_observations.values()),
            "checkpoint_physical_records": sum(
                len(value) for value in checkpoint_observations.values()
            ),
            "source_files": sum(item["state"] == "regular_file" for item in source_files),
        },
        "historical_layout_detected": historical_layout,
        "natural_partition": classification["natural_partition"],
        "checkpoint_partition": classification["checkpoint_partition"],
        "outcome_counts": {
            "natural_execution_complete": classification[
                "natural_execution_complete_count"
            ],
            "natural_terminal_infrastructure_failure": classification[
                "natural_partition"
            ]["terminal_infrastructure_failure"],
            "checkpoint_execution_complete": classification[
                "checkpoint_execution_complete_count"
            ],
            "checkpoint_terminal_infrastructure_failure": classification[
                "checkpoint_partition"
            ]["terminal_infrastructure_failure"],
            "checkpoint_ineligible": classification["checkpoint_partition"]["ineligible"],
        },
        "unexpected_physical_record_count": classification[
            "unexpected_physical_record_count"
        ],
        "structural_error_count": len(classification["structural_errors"]),
        "structural_warning_count": len(classification["structural_warnings"]),
        "structural_errors": classification["structural_errors"],
        "structural_warnings": classification["structural_warnings"],
        "natural_model_output_matrix": classification["natural_model_output_matrix"],
        "checkpoint_model_output_matrix": classification[
            "checkpoint_model_output_matrix"
        ],
    }
    structure_error = next(
        (
            error
            for error in classification["structural_errors"]
            if "historical" in error
        ),
        classification["structural_errors"][0]
        if classification["structural_errors"]
        else None,
    )
    checks = [
        {
            "name": "provenance_paths_and_sources",
            "outcome": "passed" if classification["structurally_valid"] else "failed",
            "details": {"error": structure_error, "clean_tracked_worktree": clean_tracked},
        },
        {
            "name": "logical_coverage",
            "outcome": "passed" if classification["coverage_complete"] else "failed",
            "details": {
                "natural_partition": classification["natural_partition"],
                "checkpoint_partition": classification["checkpoint_partition"],
            },
        },
        {
            "name": "paper_analysis_readiness",
            "outcome": (
                "passed"
                if classification["paper_analysis_ready"]
                else "warning"
                if classification["coverage_complete"] and classification["structurally_valid"]
                else "failed"
            ),
            "details": {
                "terminal_natural_failures": classification["natural_partition"][
                    "terminal_infrastructure_failure"
                ],
                "terminal_checkpoint_failures": classification["checkpoint_partition"][
                    "terminal_infrastructure_failure"
                ],
            },
        },
    ]
    report: dict[str, Any] = {
        "schema_name": "part1_validation_report",
        "schema_version": "1.1.0",
        "validation_report_id": "",
        "study_id": compatibility["study_id"],
        "model_run_id": compatibility["model_run_id"],
        "model_run_manifest_hash": compatibility["model_run_manifest_hash"],
        "validated_artifact_kind": "production_coverage",
        "validated_artifact_identity": compatibility["model_run_id"],
        "validation_started_at": started_at,
        "validation_completed_at": completed_at,
        "validator_version": COVERAGE_VALIDATOR_VERSION,
        "is_valid": classification["structurally_valid"],
        "structurally_valid": classification["structurally_valid"],
        "coverage_complete": classification["coverage_complete"],
        "paper_analysis_ready": classification["paper_analysis_ready"],
        "checks": checks,
        "error_count": sum(check["outcome"] == "failed" for check in checks),
        "warning_count": sum(check["outcome"] == "warning" for check in checks),
        "summary": summary,
        "source_files": sorted(source_files, key=lambda item: item["relative_path"]),
    }
    report["validation_report_id"] = coverage_report_id(report)
    validate_instance("validation_report", report)
    validate_coverage_report_semantics(report)
    final_snapshot_errors = _global_snapshot_errors(
        repository_root=repository_root,
        source_files=report["source_files"],
        expected_git_commit=head,
        expected_clean_tracked=clean_tracked,
    )
    newly_observed_errors = [
        error for error in final_snapshot_errors if error not in snapshot_errors
    ]
    if newly_observed_errors:
        raise RuntimeError(
            "coverage inputs changed during final report construction: "
            + "; ".join(newly_observed_errors)
        )
    return report
