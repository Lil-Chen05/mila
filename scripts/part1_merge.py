"""Provenance-bound, deterministic, validate-before-publish Part 1 merge.

This module is login-safe.  It imports no model, tokenizer, dataset, torch, or
CUDA code and treats every raw source as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from part1_contract import (
    canonical_json_bytes,
    model_run_id,
    model_run_manifest_hash,
    validate_fixed_model_requested_contract,
    validate_instance,
)
from part1_coverage import (
    EXPECTED_CHECKPOINT_COUNT,
    EXPECTED_NATURAL_COUNT,
    build_coverage_report,
    coverage_report_id,
    validate_coverage_report_semantics,
)
from part1_manifests import load_manifest_bundle
from part1_runtime import validate_manifest_compatibility
from part1_store import Part1ShardStore


MERGE_FORMAT_VERSION = "part1-merge-v1"
MERGE_IDENTITY_VERSION = "part1-merge-identity-v1"
MERGE_MANIFEST_HASH_VERSION = "part1-merge-manifest-hash-v1"
PARQUET_WRITER_VERSION = "part1-pyarrow-parquet-v1"
RAW_SCHEMA_VERSION = "1.0.0"
ROW_GROUP_SIZE = 1024

PARQUET_WRITER_SETTINGS: dict[str, Any] = {
    "version": "2.6",
    "compression": "zstd",
    "compression_level": 9,
    "use_dictionary": False,
    "write_statistics": True,
    "data_page_version": "1.0",
    "row_group_size": ROW_GROUP_SIZE,
    "use_compliant_nested_type": True,
    "write_page_index": False,
    "write_page_checksum": False,
}

TABLE_FILENAMES = {
    "natural_results": "natural_results.parquet",
    "checkpoint_results": "checkpoint_results.parquet",
    "audit_events": "audit_events.parquet",
}
TABLE_SCHEMA_NAMES = {
    "natural_results": "natural_terminal_result",
    "checkpoint_results": "checkpoint_terminal_result",
    "audit_events": "audit_event",
}
TABLE_SORT_ORDERS = {
    "natural_results": ("sample_index", "run_id", "raw_record_id"),
    "checkpoint_results": (
        "sample_index",
        "run_id",
        "requested_checkpoint_index",
        "checkpoint_record_id",
    ),
    # Shard-scope events sort before attempt events in a shard.  Null sentinels
    # are -1 except checkpoint_id, where natural attempts use -2 and cp-N uses N.
    "audit_events": (
        "shard_index",
        "event_scope_rank",
        "question_id_or_empty",
        "run_id_or_minus_one",
        "checkpoint_index_or_natural_minus_two",
        "attempt_number_or_minus_one",
        "event_sequence",
        "event_type",
        "event_id",
    ),
}


def _s(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.string(), nullable


def _i(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.int64(), nullable


def _f(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.float64(), nullable


def _b(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.bool_(), nullable


def _li(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.list_(pa.field("element", pa.int64(), nullable=False)), nullable


def _lf(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.list_(pa.field("element", pa.float64(), nullable=False)), nullable


def _ls(nullable: bool = False) -> tuple[pa.DataType, bool]:
    return pa.list_(pa.field("element", pa.string(), nullable=False)), nullable


NATURAL_FIELD_SPECS = (
    ("schema_name", *_s()), ("schema_version", *_s()),
    ("raw_record_id", *_s()), ("study_id", *_s()), ("model_run_id", *_s()),
    ("model_run_manifest_hash", *_s()), ("question_manifest_hash", *_s()),
    ("question_id", *_s()), ("sample_index", *_i()), ("subject", *_s()),
    ("run_id", *_i()), ("generation_seed", *_i()),
    ("seed_algorithm_version", *_s()), ("terminal_attempt_number", *_i()),
    ("terminal_attempt_id", *_s()), ("infrastructure_failure_reference", *_s(True)),
    ("prompt_hash", *_s()), ("rendered_prompt", *_s(True)),
    ("prompt_token_ids", *_li(True)), ("generated_token_ids", *_li(True)),
    ("decoded_output", *_s(True)), ("reasoning_text", *_s(True)),
    ("reasoning_boundaries", *_s(True)), ("close_tag_information", *_s(True)),
    ("stop_reason", *_s()), ("generated_token_count", *_i(True)),
    ("reasoning_token_count", *_i(True)), ("per_token_entropy_nats", *_lf(True)),
    ("mean_reasoning_entropy_nats", *_f(True)),
    ("tail_reasoning_entropy_nats", *_f(True)),
    ("terminal_answer_block_text", *_s(True)),
    ("terminal_answer_block_span", *_s(True)), ("natural_answer", *_s(True)),
    ("raw_confidence_text", *_s(True)), ("raw_parsed_confidence", *_i(True)),
    ("normalized_confidence", *_f(True)), ("natural_correct", *_b(True)),
    ("diagnostic_answer_like_text", *_s(True)), ("checkpoint_eligible", *_b()),
    ("checkpoint_ids", *_ls(True)), ("natural_execution_outcome", *_s()),
    ("reasoning_status", *_s()), ("answer_parse_status", *_s()),
    ("confidence_parse_status", *_s()), ("component_versions", *_s()),
    ("terminal_error_details", *_s(True)),
)

CHECKPOINT_FIELD_SPECS = (
    ("schema_name", *_s()), ("schema_version", *_s()),
    ("checkpoint_record_id", *_s()), ("parent_raw_record_id", *_s()),
    ("study_id", *_s()), ("model_run_id", *_s()),
    ("model_run_manifest_hash", *_s()), ("question_manifest_hash", *_s()),
    ("question_id", *_s()), ("sample_index", *_i()), ("subject", *_s()),
    ("run_id", *_i()), ("checkpoint_id", *_s()), ("natural_seed", *_i()),
    ("terminal_attempt_number", *_i()), ("terminal_attempt_id", *_s()),
    ("infrastructure_failure_reference", *_s(True)),
    ("requested_checkpoint_index", *_i()), ("requested_fraction", *_f()),
    ("k_keep", *_i()), ("actual_fraction", *_f(True)), ("shared_probe_id", *_s()),
    ("is_alias", *_b()), ("alias_metadata", *_s()), ("prefix_hash", *_s()),
    ("inducer_version", *_s()), ("inducer_text", *_s()),
    ("forced_generated_token_ids", *_li(True)), ("decoded_forced_output", *_s(True)),
    ("terminal_answer_block_text", *_s(True)), ("forced_answer", *_s(True)),
    ("raw_confidence_text", *_s(True)), ("raw_parsed_confidence", *_i(True)),
    ("normalized_confidence", *_f(True)), ("checkpoint_local_correct", *_b(True)),
    ("answer_token_index", *_i(True)), ("answer_token_id", *_i(True)),
    ("token_convention", *_s(True)), ("ad_token_ids", *_li(True)),
    ("ad_logits_float32", *_lf(True)), ("ad_probabilities_float32", *_lf(True)),
    ("answer_entropy_nats", *_f(True)),
    ("full_vocabulary_answer_step_entropy_nats", *_f(True)),
    ("maximum_ad_probability", *_f(True)), ("agrees_with_natural_answer", *_b(True)),
    ("checkpoint_execution_outcome", *_s()), ("checkpoint_model_output_status", *_s()),
    ("answer_parse_status", *_s()), ("confidence_parse_status", *_s()),
    ("answer_token_status", *_s()), ("entropy_status", *_s()),
    ("component_versions", *_s()), ("terminal_error_details", *_s(True)),
)

AUDIT_FIELD_SPECS = (
    ("schema_name", *_s()), ("schema_version", *_s()), ("event_id", *_s()),
    ("event_scope", *_s()), ("study_id", *_s()), ("model_run_id", *_s()),
    ("shard_id", *_s(True)), ("question_id", *_s(True)), ("run_id", *_i(True)),
    ("checkpoint_id", *_s(True)), ("attempt_id", *_s(True)),
    ("attempt_number", *_i(True)), ("event_sequence", *_i()), ("event_type", *_s()),
    ("event_timestamp", *_s()), ("execution_context", *_s()),
    ("outcome_category", *_s(True)), ("error_details", *_s(True)),
    ("retry_classification", *_s(True)), ("retry_decision", *_s(True)),
    ("backoff_seconds", *_f(True)), ("related_lock_owner", *_s(True)),
    ("terminal_record_id", *_s(True)), ("operator_reason", *_s(True)),
)

TABLE_FIELD_SPECS = {
    "natural_results": NATURAL_FIELD_SPECS,
    "checkpoint_results": CHECKPOINT_FIELD_SPECS,
    "audit_events": AUDIT_FIELD_SPECS,
}
TABLE_COLUMN_ORDER = {
    kind: tuple(field[0] for field in fields) for kind, fields in TABLE_FIELD_SPECS.items()
}
ENCODED_OBJECT_FIELDS = {
    "natural_results": (
        "reasoning_boundaries", "close_tag_information", "terminal_answer_block_span",
        "component_versions", "terminal_error_details",
    ),
    "checkpoint_results": ("alias_metadata", "component_versions", "terminal_error_details"),
    "audit_events": ("execution_context", "error_details", "related_lock_owner"),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_document(value: Mapping[str, Any]) -> bytes:
    return _canonical_object_json(value).encode("utf-8") + b"\n"


def _canonical_object_json(value: Any) -> str:
    """Encode a raw object itself, without the identity serializer envelope."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _audit_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    shard_id = row.get("shard_id")
    if not isinstance(shard_id, str) or not shard_id.startswith("shard-"):
        raise ValueError("audit row has no canonical shard ID")
    try:
        shard_index = int(shard_id.removeprefix("shard-"))
    except ValueError as exc:
        raise ValueError("audit row has no canonical shard index") from exc
    checkpoint_id = row.get("checkpoint_id")
    if checkpoint_id is None:
        checkpoint_index = -2
    elif isinstance(checkpoint_id, str) and checkpoint_id.startswith("cp-"):
        try:
            checkpoint_index = int(checkpoint_id.removeprefix("cp-"))
        except ValueError as exc:
            raise ValueError("audit row checkpoint ID is not canonical") from exc
    else:
        raise ValueError("audit row checkpoint ID is not canonical")
    return (
        shard_index,
        0 if row["event_scope"] == "shard" else 1,
        row.get("question_id") or "",
        -1 if row.get("run_id") is None else row["run_id"],
        checkpoint_index,
        -1 if row.get("attempt_number") is None else row["attempt_number"],
        row["event_sequence"],
        row["event_type"],
        row["event_id"],
    )


def _sort_key(kind: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    if kind == "natural_results":
        return row["sample_index"], row["run_id"], row["raw_record_id"]
    if kind == "checkpoint_results":
        return (
            row["sample_index"], row["run_id"], row["requested_checkpoint_index"],
            row["checkpoint_record_id"],
        )
    if kind == "audit_events":
        return _audit_sort_key(row)
    raise ValueError(f"unsupported merged table kind: {kind}")


def _metadata(kind: str, provenance: Mapping[str, str], row_count: int) -> dict[bytes, bytes]:
    required = (
        "study_id", "study_manifest_hash", "question_manifest_hash", "model_run_id",
        "model_run_manifest_hash", "coverage_report_id",
    )
    if set(provenance) != set(required):
        raise ValueError("merged table provenance fields differ from the fixed contract")
    for key in required:
        value = provenance[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"merged table provenance {key} is not a SHA-256 identity")
    values: dict[str, str] = {
        **dict(provenance),
        "table_kind": kind,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "merge_format_version": MERGE_FORMAT_VERSION,
        "parquet_writer_version": PARQUET_WRITER_VERSION,
        "row_count": str(row_count),
        "encoded_object_fields": _canonical_object_json(
            list(ENCODED_OBJECT_FIELDS[kind])
        ),
        "column_order": _canonical_object_json(list(TABLE_COLUMN_ORDER[kind])),
        "sort_order": _canonical_object_json(list(TABLE_SORT_ORDERS[kind])),
    }
    return {key.encode("utf-8"): value.encode("utf-8") for key, value in values.items()}


def _schema(kind: str, provenance: Mapping[str, str], row_count: int) -> pa.Schema:
    try:
        specs = TABLE_FIELD_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported merged table kind: {kind}") from exc
    return pa.schema(
        [pa.field(name, datatype, nullable=nullable) for name, datatype, nullable in specs],
        metadata=_metadata(kind, provenance, row_count),
    )


def build_merge_table(
    kind: str,
    records: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, str],
) -> pa.Table:
    """Validate, losslessly encode, explicitly type, and deterministically sort rows."""

    schema_name = TABLE_SCHEMA_NAMES.get(kind)
    if schema_name is None:
        raise ValueError(f"unsupported merged table kind: {kind}")
    ordered = [copy.deepcopy(dict(record)) for record in records]
    for record in ordered:
        validate_instance(schema_name, record)
    ordered.sort(key=lambda row: _sort_key(kind, row))
    encoded_fields = frozenset(ENCODED_OBJECT_FIELDS[kind])
    schema = _schema(kind, provenance, len(ordered))
    arrays = []
    for field in schema:
        values = []
        for record in ordered:
            value = record[field.name]
            if field.name in encoded_fields and value is not None:
                value = _canonical_object_json(value)
            values.append(value)
        arrays.append(pa.array(values, type=field.type, from_pandas=False))
    table = pa.Table.from_arrays(arrays, schema=schema)
    if decode_merge_table(kind, table) != ordered:
        raise ValueError(f"{kind} Arrow encoding is not lossless")
    return table


def decode_merge_table(kind: str, table: pa.Table) -> list[dict[str, Any]]:
    encoded_fields = frozenset(ENCODED_OBJECT_FIELDS[kind])
    rows = table.to_pylist()
    for row in rows:
        for field in encoded_fields:
            value = row[field]
            if value is not None:
                row[field] = json.loads(value)
    return rows


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _schema_sha256(schema: pa.Schema) -> str:
    return _sha256(schema.serialize().to_pybytes())


def write_parquet_tables(
    directory: Path,
    natural_records: Sequence[Mapping[str, Any]],
    checkpoint_records: Sequence[Mapping[str, Any]],
    audit_events: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Write the three deterministic Parquet files and return manifest summaries."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    row_sets = {
        "natural_results": natural_records,
        "checkpoint_results": checkpoint_records,
        "audit_events": audit_events,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for kind in TABLE_FILENAMES:
        table = build_merge_table(kind, row_sets[kind], provenance=provenance)
        path = directory / TABLE_FILENAMES[kind]
        pq.write_table(
            table,
            path,
            version=PARQUET_WRITER_SETTINGS["version"],
            compression=PARQUET_WRITER_SETTINGS["compression"],
            compression_level=PARQUET_WRITER_SETTINGS["compression_level"],
            use_dictionary=PARQUET_WRITER_SETTINGS["use_dictionary"],
            write_statistics=PARQUET_WRITER_SETTINGS["write_statistics"],
            data_page_version=PARQUET_WRITER_SETTINGS["data_page_version"],
            row_group_size=PARQUET_WRITER_SETTINGS["row_group_size"],
            use_compliant_nested_type=PARQUET_WRITER_SETTINGS[
                "use_compliant_nested_type"
            ],
            write_page_index=PARQUET_WRITER_SETTINGS["write_page_index"],
            write_page_checksum=PARQUET_WRITER_SETTINGS["write_page_checksum"],
        )
        _fsync_file(path)
        data = path.read_bytes()
        metadata = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in table.schema.metadata.items()
        }
        outputs[kind] = {
            "relative_path": TABLE_FILENAMES[kind],
            "sha256": _sha256(data),
            "byte_size": len(data),
            "row_count": table.num_rows,
            "schema_sha256": _schema_sha256(table.schema),
            "embedded_metadata": metadata,
        }
    return outputs


def _without_location_paths(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("merge_id", None)
    payload.pop("merge_manifest_hash", None)
    coverage = payload.get("coverage_report")
    if isinstance(coverage, dict):
        coverage.pop("relative_path", None)
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        for item in outputs.values():
            if isinstance(item, dict):
                item.pop("relative_path", None)
    return payload


def merge_id(manifest: Mapping[str, Any]) -> str:
    payload = {
        "identity_type": "part1_merge",
        "identity_version": MERGE_IDENTITY_VERSION,
        "payload": _without_location_paths(manifest),
    }
    return _sha256(canonical_json_bytes(payload))


def merge_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = _without_location_paths(manifest)
    payload["merge_id"] = manifest.get("merge_id")
    wrapped = {
        "identity_type": "part1_merge_manifest",
        "identity_version": MERGE_MANIFEST_HASH_VERSION,
        "payload": payload,
    }
    return _sha256(canonical_json_bytes(wrapped))


def build_merge_manifest(
    *,
    provenance: Mapping[str, str],
    coverage_report_path: str,
    coverage_report_sha256: str,
    coverage_report_byte_size: int,
    source_files: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_name": "part1_merge_manifest",
        "schema_version": "1.0.0",
        "merge_id": "",
        "merge_manifest_hash": "",
        "merge_format_version": MERGE_FORMAT_VERSION,
        "parquet_writer_version": PARQUET_WRITER_VERSION,
        **dict(provenance),
        "coverage_report": {
            "relative_path": coverage_report_path,
            "sha256": coverage_report_sha256,
            "byte_size": coverage_report_byte_size,
        },
        "source_files": sorted(
            (dict(item) for item in source_files), key=lambda item: item["relative_path"]
        ),
        "sort_orders": {kind: list(order) for kind, order in TABLE_SORT_ORDERS.items()},
        "parquet_writer_settings": dict(PARQUET_WRITER_SETTINGS),
        "outputs": {kind: dict(item) for kind, item in outputs.items()},
    }
    manifest["merge_id"] = merge_id(manifest)
    manifest["merge_manifest_hash"] = merge_manifest_hash(manifest)
    return manifest


def validate_merge_manifest(manifest: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_name", "schema_version", "merge_id", "merge_manifest_hash",
        "merge_format_version", "parquet_writer_version", "study_id",
        "study_manifest_hash", "question_manifest_hash", "model_run_id",
        "model_run_manifest_hash", "coverage_report_id", "coverage_report",
        "source_files", "sort_orders", "parquet_writer_settings", "outputs",
    }
    if set(manifest) != expected_fields:
        raise ValueError("merge manifest fields differ from the fixed contract")
    if manifest["schema_name"] != "part1_merge_manifest" or manifest[
        "schema_version"
    ] != "1.0.0":
        raise ValueError("merge manifest schema identity differs")
    if manifest["merge_format_version"] != MERGE_FORMAT_VERSION or manifest[
        "parquet_writer_version"
    ] != PARQUET_WRITER_VERSION:
        raise ValueError("merge format/writer version differs")
    if manifest["sort_orders"] != {
        kind: list(order) for kind, order in TABLE_SORT_ORDERS.items()
    } or manifest["parquet_writer_settings"] != PARQUET_WRITER_SETTINGS:
        raise ValueError("merge deterministic sort/writer settings differ")
    for key in (
        "study_id", "study_manifest_hash", "question_manifest_hash", "model_run_id",
        "model_run_manifest_hash", "coverage_report_id",
    ):
        if not isinstance(manifest[key], str) or len(manifest[key]) != 64:
            raise ValueError(f"merge manifest {key} is not a SHA-256 identity")
    coverage = manifest["coverage_report"]
    if not isinstance(coverage, Mapping) or set(coverage) != {
        "relative_path", "sha256", "byte_size"
    }:
        raise ValueError("merge manifest coverage report fields differ")
    coverage_path = Path(coverage["relative_path"])
    if (
        not isinstance(coverage["relative_path"], str)
        or coverage_path.is_absolute()
        or ".." in coverage_path.parts
        or coverage_path.as_posix() != coverage["relative_path"]
        or not isinstance(coverage["sha256"], str)
        or len(coverage["sha256"]) != 64
        or isinstance(coverage["byte_size"], bool)
        or not isinstance(coverage["byte_size"], int)
        or coverage["byte_size"] < 0
    ):
        raise ValueError("merge manifest coverage report provenance is invalid")
    source_files = manifest["source_files"]
    if not isinstance(source_files, list):
        raise ValueError("merge manifest source inventory is not a list")
    source_paths: list[str] = []
    for item in source_files:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path", "shard_id", "kind", "state", "sha256", "byte_size"
        }:
            raise ValueError("merge manifest source inventory fields differ")
        relative_path = item["relative_path"]
        parsed = Path(relative_path) if isinstance(relative_path, str) else Path("/")
        if (
            not isinstance(relative_path, str)
            or parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != relative_path
            or item["state"] not in {"regular_file", "absent"}
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or isinstance(item["byte_size"], bool)
            or not isinstance(item["byte_size"], int)
            or item["byte_size"] < 0
        ):
            raise ValueError("merge manifest source inventory entry is invalid")
        source_paths.append(relative_path)
    if source_paths != sorted(source_paths) or len(source_paths) != len(set(source_paths)):
        raise ValueError("merge manifest source inventory order or uniqueness differs")
    outputs = manifest["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != set(TABLE_FILENAMES):
        raise ValueError("merge manifest outputs differ from the three-table contract")
    for kind, summary in outputs.items():
        if not isinstance(summary, Mapping) or set(summary) != {
            "relative_path", "sha256", "byte_size", "row_count", "schema_sha256",
            "embedded_metadata",
        }:
            raise ValueError(f"merge manifest output fields differ for {kind}")
        if (
            summary["relative_path"] != TABLE_FILENAMES[kind]
            or not isinstance(summary["sha256"], str)
            or len(summary["sha256"]) != 64
            or not isinstance(summary["schema_sha256"], str)
            or len(summary["schema_sha256"]) != 64
            or isinstance(summary["byte_size"], bool)
            or not isinstance(summary["byte_size"], int)
            or summary["byte_size"] < 0
            or isinstance(summary["row_count"], bool)
            or not isinstance(summary["row_count"], int)
            or summary["row_count"] < 0
            or not isinstance(summary["embedded_metadata"], Mapping)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in summary["embedded_metadata"].items()
            )
        ):
            raise ValueError(f"merge manifest output summary is invalid for {kind}")
    if manifest["merge_id"] != merge_id(manifest):
        raise ValueError("merge identity does not recompute")
    if manifest["merge_manifest_hash"] != merge_manifest_hash(manifest):
        raise ValueError("merge manifest hash does not recompute")


@dataclass
class MergeInputs:
    repository_root: Path
    model_manifest: dict[str, Any]
    coverage_report: dict[str, Any]
    coverage_report_path: Path
    coverage_report_bytes: bytes
    source_files: tuple[dict[str, Any], ...]
    natural_records: tuple[dict[str, Any], ...]
    checkpoint_records: tuple[dict[str, Any], ...]
    audit_events: tuple[dict[str, Any], ...]


def require_mergeable_coverage(report: Mapping[str, Any]) -> None:
    validate_instance("validation_report", report)
    if report.get("schema_version") != "1.1.0" or report.get(
        "validated_artifact_kind"
    ) != "production_coverage":
        raise ValueError("merge requires a production schema-1.1 coverage report")
    if report.get("structurally_valid") is not True:
        raise ValueError("coverage structurally_valid must be true")
    if report.get("coverage_complete") is not True:
        raise ValueError("coverage coverage_complete must be true")
    validate_coverage_report_semantics(report)
    if report["validation_report_id"] != coverage_report_id(report):
        raise ValueError("coverage report stable identity does not recompute")


def require_source_snapshot(
    repository_root: Path, source_files: Sequence[Mapping[str, Any]]
) -> None:
    """Restate every regular/absent inventory entry without mutating sources."""

    repository_root = Path(os.path.abspath(repository_root))
    errors: list[str] = []
    for entry in source_files:
        relative_path = str(entry["relative_path"])
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_path:
            errors.append(f"source path is unsafe: {relative_path}")
            continue
        path = repository_root / relative
        _require_no_symlink_components(path)
        if entry["state"] == "absent":
            if os.path.lexists(path):
                errors.append(f"source changed: expected absent but observed present: {relative_path}")
            continue
        if not os.path.lexists(path):
            errors.append(f"source changed: expected regular file but observed absent: {relative_path}")
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            errors.append(f"source changed type: {relative_path}")
            continue
        data = path.read_bytes()
        if len(data) != entry["byte_size"] or _sha256(data) != entry["sha256"]:
            errors.append(f"source bytes changed: {relative_path}")
    if errors:
        raise ValueError("source snapshot differs from coverage: " + "; ".join(errors[:5]))


def _require_no_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    components = [absolute]
    cursor = absolute
    while cursor != cursor.parent:
        cursor = cursor.parent
        components.append(cursor)
    for component in reversed(components):
        if os.path.lexists(component) and stat.S_ISLNK(component.lstat().st_mode):
            raise ValueError(f"path contains a symlink component: {component}")


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


def _canonical_relative(repository_root: Path, path: Path, *, label: str) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(
            Path(os.path.abspath(repository_root))
        ).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc


def revalidate_merge_inputs(inputs: MergeInputs) -> None:
    require_mergeable_coverage(inputs.coverage_report)
    path = inputs.coverage_report_path
    _require_no_symlink_components(path)
    if not os.path.lexists(path) or stat.S_ISLNK(path.lstat().st_mode) or not stat.S_ISREG(
        path.lstat().st_mode
    ):
        raise ValueError("coverage report changed type or disappeared")
    current_coverage = path.read_bytes()
    if current_coverage != inputs.coverage_report_bytes:
        raise ValueError("coverage report bytes changed after validation")
    require_source_snapshot(inputs.repository_root, inputs.source_files)
    from part1_coverage import _global_snapshot_errors

    snapshot_errors = _global_snapshot_errors(
        repository_root=inputs.repository_root,
        source_files=inputs.source_files,
        expected_git_commit=inputs.coverage_report["summary"]["observed_git_commit"],
        expected_clean_tracked=inputs.coverage_report["summary"]["clean_tracked_worktree"],
    )
    if snapshot_errors:
        raise ValueError("whole-run snapshot changed: " + "; ".join(snapshot_errors[:5]))


def load_validated_merge_inputs(
    *,
    repository_root: Path,
    model_run_manifest_path: Path,
    coverage_report_path: Path | None = None,
) -> MergeInputs:
    """Re-run the production coverage gate, then read exactly its named streams."""

    repository_root = Path(os.path.abspath(repository_root))
    model_run_manifest_path = Path(model_run_manifest_path)
    if not model_run_manifest_path.is_absolute():
        model_run_manifest_path = repository_root / model_run_manifest_path
    model_manifest, _manifest_bytes = _load_regular_json(
        model_run_manifest_path, label="production model-run manifest"
    )
    validate_instance("model_run_manifest", model_manifest)
    validate_fixed_model_requested_contract(model_manifest)
    if model_manifest.get("production") is not True or model_manifest.get(
        "execution_scope"
    ) != "production" or model_manifest.get("schema_version") != "1.1.0":
        raise ValueError("merge requires a production schema-1.1 model-run manifest")
    if model_manifest["model_run_id"] != model_run_id(model_manifest) or model_manifest[
        "model_run_manifest_hash"
    ] != model_run_manifest_hash(model_manifest):
        raise ValueError("production model-run identities do not recompute")
    expected_manifest = (
        repository_root / "results" / "part1" / model_manifest["model_run_id"]
        / "model_run_manifest.json"
    )
    if Path(os.path.abspath(model_run_manifest_path)) != Path(os.path.abspath(expected_manifest)):
        raise ValueError("production model-run manifest path is not canonical")
    expected_paths = {
        "raw_shards": f"results/part1/{model_manifest['model_run_id']}/raw_shards",
        "validation": f"results/part1/{model_manifest['model_run_id']}/validation",
        "merged": f"results/part1/{model_manifest['model_run_id']}/merged",
    }
    for key, value in expected_paths.items():
        if model_manifest["output_paths"].get(key) != value:
            raise ValueError(f"production {key} output path is not canonical")
    expected_coverage_path = repository_root / expected_paths["validation"] / "coverage_report.json"
    if coverage_report_path is None:
        coverage_report_path = expected_coverage_path
    elif not Path(coverage_report_path).is_absolute():
        coverage_report_path = repository_root / coverage_report_path
    coverage_report_path = Path(os.path.abspath(coverage_report_path))
    if coverage_report_path != Path(os.path.abspath(expected_coverage_path)):
        raise ValueError("coverage report path is not canonical")
    coverage_report, coverage_bytes = _load_regular_json(
        coverage_report_path, label="coverage report"
    )
    require_mergeable_coverage(coverage_report)
    if coverage_report["model_run_id"] != model_manifest["model_run_id"] or coverage_report[
        "model_run_manifest_hash"
    ] != model_manifest["model_run_manifest_hash"]:
        raise ValueError("coverage and production model-run identities differ")
    require_source_snapshot(repository_root, coverage_report["source_files"])

    rebuilt = build_coverage_report(
        repository_root=repository_root,
        model_run_manifest_path=model_run_manifest_path,
        validation_started_at=coverage_report["validation_started_at"],
        validation_completed_at=coverage_report["validation_completed_at"],
    )
    if rebuilt != coverage_report:
        raise ValueError("published coverage report does not equal current procedural revalidation")

    manifest_root = repository_root / "manifests" / "part1"
    bundle = load_manifest_bundle(
        questions_path=manifest_root / "questions.jsonl",
        question_manifest_path=manifest_root / "questions.manifest.json",
        study_manifest_path=manifest_root / "study_manifest.json",
    )
    validate_manifest_compatibility(bundle.study_manifest, model_manifest)
    question_by_index = {record["sample_index"]: record for record in bundle.records}
    if set(question_by_index) != set(range(500)):
        raise ValueError("tracked question bundle is not the fixed 500-question manifest")

    natural: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    natural_ids: set[str] = set()
    checkpoint_ids: set[str] = set()
    event_ids: set[str] = set()
    raw_root = repository_root / expected_paths["raw_shards"]
    for shard_index in range(500):
        shard_id = f"shard-{shard_index:03d}"
        store = Part1ShardStore(
            raw_root / shard_id,
            shard_id=shard_id,
            study_id=model_manifest["study_id"],
            model_run_id=model_manifest["model_run_id"],
            model_run_manifest_hash=model_manifest["model_run_manifest_hash"],
        )
        inspection = store.inspect()
        index = store.build_index()
        if (
            index.hierarchy_errors or index.lifecycle_errors
            or index.missing_completion_record_ids or index.missing_started_attempt_ids
            or index.inconsistent_completion_attempt_ids or index.orphaned_attempt_ids
            or index.pending_recovery_event_ids or index.terminalization_required
        ):
            raise ValueError(f"{shard_id} lifecycle/hierarchy is incomplete")
        question = question_by_index[shard_index]
        for row in inspection.natural_results:
            validate_instance("natural_terminal_result", row)
            if row["sample_index"] != shard_index or row["question_id"] != question["question_id"]:
                raise ValueError(f"natural row is assigned to the wrong shard: {shard_id}")
            if row["raw_record_id"] in natural_ids:
                raise ValueError("duplicate natural record ID across shards")
            natural_ids.add(row["raw_record_id"])
            natural.append(row)
        for row in inspection.checkpoint_results:
            validate_instance("checkpoint_terminal_result", row)
            if row["sample_index"] != shard_index or row["question_id"] != question["question_id"]:
                raise ValueError(f"checkpoint row is assigned to the wrong shard: {shard_id}")
            if row["checkpoint_record_id"] in checkpoint_ids:
                raise ValueError("duplicate checkpoint record ID across shards")
            checkpoint_ids.add(row["checkpoint_record_id"])
            checkpoints.append(row)
        for row in inspection.audit_events:
            validate_instance("audit_event", row)
            if row["shard_id"] != shard_id:
                raise ValueError(f"audit event is assigned to the wrong shard: {shard_id}")
            if row["event_id"] in event_ids:
                raise ValueError("duplicate audit event ID across shards")
            event_ids.add(row["event_id"])
            audit.append(row)

    natural_partition = coverage_report["summary"]["natural_partition"]
    checkpoint_partition = coverage_report["summary"]["checkpoint_partition"]
    expected_natural = natural_partition["complete"] + natural_partition[
        "terminal_infrastructure_failure"
    ]
    expected_checkpoints = checkpoint_partition["complete"] + checkpoint_partition[
        "terminal_infrastructure_failure"
    ]
    if len(natural) != expected_natural or len(natural) != coverage_report["summary"][
        "observed"
    ]["natural_physical_records"]:
        raise ValueError("merged natural physical count differs from coverage partitions")
    if len(checkpoints) != expected_checkpoints or len(checkpoints) != coverage_report[
        "summary"
    ]["observed"]["checkpoint_physical_records"]:
        raise ValueError("merged checkpoint physical count differs from coverage partitions")
    if len(natural) != EXPECTED_NATURAL_COUNT:
        raise ValueError("merged natural source inventory is partial")
    if len(checkpoints) > EXPECTED_CHECKPOINT_COUNT:
        raise ValueError("merged checkpoint source inventory exceeds the fixed workload")

    inputs = MergeInputs(
        repository_root=repository_root,
        model_manifest=model_manifest,
        coverage_report=coverage_report,
        coverage_report_path=coverage_report_path,
        coverage_report_bytes=coverage_bytes,
        source_files=tuple(dict(item) for item in coverage_report["source_files"]),
        natural_records=tuple(natural),
        checkpoint_records=tuple(checkpoints),
        audit_events=tuple(audit),
    )
    revalidate_merge_inputs(inputs)
    return inputs


def _provenance(inputs: MergeInputs) -> dict[str, str]:
    return {
        "study_id": inputs.model_manifest["study_id"],
        "study_manifest_hash": inputs.coverage_report["summary"]["study_manifest_hash"],
        "question_manifest_hash": inputs.coverage_report["summary"]["question_manifest_hash"],
        "model_run_id": inputs.model_manifest["model_run_id"],
        "model_run_manifest_hash": inputs.model_manifest["model_run_manifest_hash"],
        "coverage_report_id": inputs.coverage_report["validation_report_id"],
    }


def _safe_existing_directory(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        raise ValueError(f"{label} is missing: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} is a symlink: {path}")
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is not a directory: {path}")


def _ensure_safe_directory(path: Path) -> None:
    path = Path(os.path.abspath(path))
    _require_no_symlink_components(path)
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        cursor = cursor.parent
    for component in (cursor, *reversed(missing)):
        if component == cursor:
            _safe_existing_directory(component, label="publication path parent")
        else:
            component.mkdir()
            _fsync_directory(component.parent)
    _safe_existing_directory(path, label="publication directory")


def _expected_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return _canonical_document(manifest)


def validate_merge_directory(
    directory: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Strictly validate a staged or finalized four-file merge directory."""

    directory = Path(directory)
    _safe_existing_directory(directory, label="merged directory")
    expected_names = {*TABLE_FILENAMES.values(), "merge_manifest.json"}
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != expected_names or len(entries) != len(expected_names):
        raise ValueError("merged directory has missing or extra contents")
    for entry in entries:
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"merged directory entry is symlinked or nonregular: {entry.name}")
    manifest_path = directory / "merge_manifest.json"
    manifest, manifest_bytes = _load_regular_json(manifest_path, label="merge manifest")
    validate_merge_manifest(manifest)
    if manifest_bytes != _expected_manifest_bytes(manifest):
        raise ValueError("merge manifest bytes are not canonical deterministic JSON")
    if expected_manifest is not None and manifest != dict(expected_manifest):
        raise ValueError("finalized merge manifest differs from current validated inputs")
    if set(manifest["outputs"]) != set(TABLE_FILENAMES):
        raise ValueError("merge manifest does not describe exactly three Parquet outputs")
    provenance = {
        key: manifest[key]
        for key in (
            "study_id", "study_manifest_hash", "question_manifest_hash", "model_run_id",
            "model_run_manifest_hash", "coverage_report_id",
        )
    }
    for kind, filename in TABLE_FILENAMES.items():
        path = directory / filename
        data = path.read_bytes()
        summary = manifest["outputs"][kind]
        if summary["relative_path"] != filename or summary["sha256"] != _sha256(
            data
        ) or summary["byte_size"] != len(data):
            raise ValueError(f"{kind} output bytes differ from merge manifest")
        table = pq.read_table(path)
        expected_schema = _schema(kind, provenance, summary["row_count"])
        if table.schema != expected_schema or _schema_sha256(table.schema) != summary[
            "schema_sha256"
        ]:
            raise ValueError(f"{kind} explicit Arrow schema or metadata differs")
        metadata = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in table.schema.metadata.items()
        }
        if metadata != summary["embedded_metadata"] or table.num_rows != summary["row_count"]:
            raise ValueError(f"{kind} metadata or row count differs")
        decoded = decode_merge_table(kind, table)
        if decoded != sorted(decoded, key=lambda row: _sort_key(kind, row)):
            raise ValueError(f"{kind} row ordering differs from deterministic sort")
        schema_name = TABLE_SCHEMA_NAMES[kind]
        for row in decoded:
            validate_instance(schema_name, row)
        if expected_rows is not None:
            expected = [copy.deepcopy(dict(row)) for row in expected_rows[kind]]
            expected.sort(key=lambda row: _sort_key(kind, row))
            if decoded != expected:
                raise ValueError(f"{kind} Parquet recovery is not lossless")
    return manifest


def _remove_own_stage(stage: Path, parent: Path, prefix: str) -> None:
    if stage.parent != parent or not stage.name.startswith(prefix):
        raise RuntimeError("refusing to clean a directory not owned by this merge invocation")
    if os.path.lexists(stage):
        if stage.is_symlink():
            raise RuntimeError("merge staging directory was replaced by a symlink")
        shutil.rmtree(stage)


def publish_merge(
    inputs: MergeInputs,
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> Path:
    """Stage, reload, verify, and atomically publish one no-overwrite merge."""

    revalidate_merge_inputs(inputs)
    target_relative = inputs.model_manifest["output_paths"]["merged"]
    expected_relative = f"results/part1/{inputs.model_manifest['model_run_id']}/merged"
    if target_relative != expected_relative:
        raise ValueError("production merged output path is not canonical")
    target = inputs.repository_root / target_relative
    _ensure_safe_directory(target.parent)
    prefix = f".{target.name}.stage-"
    stage = Path(tempfile.mkdtemp(prefix=prefix, dir=target.parent))
    rows = {
        "natural_results": inputs.natural_records,
        "checkpoint_results": inputs.checkpoint_records,
        "audit_events": inputs.audit_events,
    }
    hook = fault_hook or (lambda _boundary: None)
    published = False
    try:
        hook("stage_created")
        provenance = _provenance(inputs)
        outputs = write_parquet_tables(
            stage,
            inputs.natural_records,
            inputs.checkpoint_records,
            inputs.audit_events,
            provenance=provenance,
        )
        hook("table_writes_complete")
        coverage_relative = _canonical_relative(
            inputs.repository_root, inputs.coverage_report_path, label="coverage report"
        )
        manifest = build_merge_manifest(
            provenance=provenance,
            coverage_report_path=coverage_relative,
            coverage_report_sha256=_sha256(inputs.coverage_report_bytes),
            coverage_report_byte_size=len(inputs.coverage_report_bytes),
            source_files=inputs.source_files,
            outputs=outputs,
        )
        manifest_path = stage / "merge_manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(_expected_manifest_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        hook("manifest_written")
        _fsync_directory(stage)
        validate_merge_directory(stage, expected_manifest=manifest, expected_rows=rows)
        hook("reload_complete")
        hook("before_rename")
        revalidate_merge_inputs(inputs)
        if os.path.lexists(target):
            validate_merge_directory(target, expected_manifest=manifest, expected_rows=rows)
            return target
        try:
            os.rename(stage, target)
            published = True
        except OSError:
            if not os.path.lexists(target):
                raise
            revalidate_merge_inputs(inputs)
            validate_merge_directory(target, expected_manifest=manifest, expected_rows=rows)
            return target
        _fsync_directory(target.parent)
        return target
    finally:
        if not published:
            _remove_own_stage(stage, target.parent, prefix)


def merge_part1_results(
    *,
    repository_root: Path,
    model_run_manifest_path: Path,
    coverage_report_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    inputs = load_validated_merge_inputs(
        repository_root=repository_root,
        model_run_manifest_path=model_run_manifest_path,
        coverage_report_path=coverage_report_path,
    )
    target = publish_merge(inputs)
    manifest, _bytes = _load_regular_json(target / "merge_manifest.json", label="merge manifest")
    return target, manifest
