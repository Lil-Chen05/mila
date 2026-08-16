"""Provenance-bound, deterministic, validate-before-publish Part 1 merge.

This module is login-safe.  It imports no model, tokenizer, dataset, torch, or
CUDA code and treats every raw source as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
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
    _authoritative_source_inventory_defects,
    _checkpoint_compatibility,
    _natural_compatibility,
    build_coverage_report,
    coverage_report_id,
    validate_coverage_report_semantics,
)
from part1_manifests import load_manifest_bundle
from part1_runtime import validate_manifest_compatibility
from part1_store import Part1ShardStore
from part1_prompt_hash_waiver import (
    require_complete_checkpoint_outcome,
    require_content_derived_prompt_hash,
    require_exact_failed_report,
    require_production_checkout_generation_state,
    validate_prompt_hash_waiver,
)


MERGE_FORMAT_VERSION = "part1-merge-v1"
MERGE_IDENTITY_VERSION = "part1-merge-identity-v1"
MERGE_MANIFEST_HASH_VERSION = "part1-merge-manifest-hash-v1"
PARQUET_WRITER_VERSION = "part1-pyarrow-parquet-v1"
RAW_SCHEMA_VERSION = "1.0.0"
ROW_GROUP_SIZE = 1024
LOSSLESS_RAW_ROW_FIELD = "raw_row_canonical_json"
PROJECTION_CONVERSION_VERSION = "part1-json-arrow-projection-v1"
LINUX_RENAME_NOREPLACE = 0x00000001
MACOS_RENAME_EXCL = 0x00000004
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CHECKPOINTS_PER_NATURAL = 11

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

RAW_FIELD_SPECS = {
    "natural_results": NATURAL_FIELD_SPECS,
    "checkpoint_results": CHECKPOINT_FIELD_SPECS,
    "audit_events": AUDIT_FIELD_SPECS,
}
TABLE_FIELD_SPECS = {
    kind: (*fields, (LOSSLESS_RAW_ROW_FIELD, pa.string(), False))
    for kind, fields in RAW_FIELD_SPECS.items()
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
        "lossless_raw_row_field": LOSSLESS_RAW_ROW_FIELD,
        "projection_conversion_version": PROJECTION_CONVERSION_VERSION,
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
            if field.name == LOSSLESS_RAW_ROW_FIELD:
                value = _canonical_object_json(record)
            else:
                value = record[field.name]
            if field.name in encoded_fields and value is not None:
                value = _canonical_object_json(value)
            values.append(value)
        arrays.append(pa.array(values, type=field.type, from_pandas=False))
    table = pa.Table.from_arrays(arrays, schema=schema)
    if [_canonical_object_json(row) for row in decode_merge_table(kind, table)] != [
        _canonical_object_json(row) for row in ordered
    ]:
        raise ValueError(f"{kind} Arrow encoding is not lossless")
    return table


def _project_raw_value(field: pa.Field, value: Any, *, encoded_object: bool) -> Any:
    if value is None:
        return None
    if encoded_object:
        return _canonical_object_json(value)
    datatype = field.type
    if pa.types.is_int64(datatype):
        return int(value)
    if pa.types.is_float64(datatype):
        return float(value)
    if pa.types.is_boolean(datatype) or pa.types.is_string(datatype):
        return value
    if pa.types.is_list(datatype):
        element_field = datatype.value_field
        return [
            _project_raw_value(element_field, item, encoded_object=False)
            for item in value
        ]
    raise ValueError(f"unsupported direct projection type: {datatype}")


def decode_merge_table(kind: str, table: pa.Table) -> list[dict[str, Any]]:
    encoded_fields = frozenset(ENCODED_OBJECT_FIELDS[kind])
    if LOSSLESS_RAW_ROW_FIELD not in table.column_names:
        raise ValueError(f"{kind} table has no lossless raw-row representation")
    projected_rows = table.to_pylist()
    raw_rows: list[dict[str, Any]] = []
    schema_name = TABLE_SCHEMA_NAMES[kind]
    for row_number, projected in enumerate(projected_rows):
        raw_text = projected[LOSSLESS_RAW_ROW_FIELD]
        try:
            raw = json.loads(raw_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{kind} row {row_number} raw JSON is invalid") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{kind} row {row_number} raw JSON is not an object")
        validate_instance(schema_name, raw)
        for field in table.schema:
            if field.name == LOSSLESS_RAW_ROW_FIELD:
                continue
            expected = _project_raw_value(
                field, raw[field.name], encoded_object=field.name in encoded_fields
            )
            if _canonical_object_json(projected[field.name]) != _canonical_object_json(
                expected
            ):
                raise ValueError(
                    f"{kind} row {row_number} projection differs for {field.name}"
                )
        raw_rows.append(raw)
    return raw_rows


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


class ExclusiveRenameUnavailable(RuntimeError):
    """The host has no verified native exclusive directory-rename primitive."""


class PublicationDurabilityError(RuntimeError):
    """Publication crossed rename but was safely rolled back before success."""


class PublicationStateIndeterminateError(RuntimeError):
    """Storage failure left publication durability genuinely indeterminate."""


def _exclusive_rename_at(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    """Rename one sibling directory without replacement, anchored to parent FD."""

    for label, name in (("source", source_name), ("destination", destination_name)):
        if not name or Path(name).name != name or "/" in name:
            raise ValueError(f"exclusive rename {label} must be one basename")
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        try:
            function = library.renameat2
        except AttributeError as exc:
            raise ExclusiveRenameUnavailable(
                "Linux renameat2(RENAME_NOREPLACE) is unavailable; failing closed"
            ) from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            LINUX_RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        try:
            function = library.renameatx_np
        except AttributeError as exc:
            raise ExclusiveRenameUnavailable(
                "macOS renameatx_np(RENAME_EXCL) is unavailable; failing closed"
            ) from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            MACOS_RENAME_EXCL,
        )
    else:
        raise ExclusiveRenameUnavailable(
            f"no verified exclusive rename primitive for platform {sys.platform!r}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


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
    prompt_hash_waiver_path: str | None = None,
    prompt_hash_waiver_id: str | None = None,
    prompt_hash_waiver_sha256: str | None = None,
    prompt_hash_waiver_byte_size: int | None = None,
    source_files: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_name": "part1_merge_manifest",
        "schema_version": "1.1.0" if prompt_hash_waiver_path is not None else "1.0.0",
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
    waiver_values = (
        prompt_hash_waiver_path,
        prompt_hash_waiver_id,
        prompt_hash_waiver_sha256,
        prompt_hash_waiver_byte_size,
    )
    if any(value is not None for value in waiver_values):
        if any(value is None for value in waiver_values):
            raise ValueError("merge prompt-hash waiver provenance is incomplete")
        manifest["prompt_hash_waiver"] = {
            "relative_path": prompt_hash_waiver_path,
            "waiver_id": prompt_hash_waiver_id,
            "sha256": prompt_hash_waiver_sha256,
            "byte_size": prompt_hash_waiver_byte_size,
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
    waiver_mode = manifest.get("schema_version") == "1.1.0"
    if waiver_mode:
        expected_fields.add("prompt_hash_waiver")
    if set(manifest) != expected_fields:
        raise ValueError("merge manifest fields differ from the fixed contract")
    if manifest["schema_name"] != "part1_merge_manifest" or manifest[
        "schema_version"
    ] not in {"1.0.0", "1.1.0"}:
        raise ValueError("merge manifest schema identity differs")
    if (manifest["schema_version"] == "1.0.0") != ("prompt_hash_waiver" not in manifest):
        raise ValueError("merge manifest schema/waiver mode differs")
    if manifest["merge_format_version"] != MERGE_FORMAT_VERSION or manifest[
        "parquet_writer_version"
    ] != PARQUET_WRITER_VERSION:
        raise ValueError("merge format/writer version differs")
    if canonical_json_bytes(manifest["sort_orders"]) != canonical_json_bytes(
        {kind: list(order) for kind, order in TABLE_SORT_ORDERS.items()}
    ) or canonical_json_bytes(manifest["parquet_writer_settings"]) != canonical_json_bytes(
        PARQUET_WRITER_SETTINGS
    ):
        raise ValueError("merge deterministic sort/writer settings differ")
    for key in (
        "study_id", "study_manifest_hash", "question_manifest_hash", "model_run_id",
        "model_run_manifest_hash", "coverage_report_id",
    ):
        if not isinstance(manifest[key], str) or SHA256_PATTERN.fullmatch(manifest[key]) is None:
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
        or coverage["relative_path"]
        != f"results/part1/{manifest['model_run_id']}/validation/coverage_report.json"
        or not isinstance(coverage["sha256"], str)
        or SHA256_PATTERN.fullmatch(coverage["sha256"]) is None
        or coverage["sha256"] == _sha256(b"")
        or isinstance(coverage["byte_size"], bool)
        or not isinstance(coverage["byte_size"], int)
        or coverage["byte_size"] <= 0
    ):
        raise ValueError("merge manifest coverage report provenance is invalid")
    if waiver_mode:
        waiver = manifest["prompt_hash_waiver"]
        expected_waiver_path = (
            f"results/part1/{manifest['model_run_id']}/validation/prompt_hash_waiver.json"
        )
        if not isinstance(waiver, Mapping) or set(waiver) != {
            "relative_path", "waiver_id", "sha256", "byte_size"
        } or (
            waiver["relative_path"] != expected_waiver_path
            or any(
                not isinstance(waiver[field], str)
                or SHA256_PATTERN.fullmatch(waiver[field]) is None
                for field in ("waiver_id", "sha256")
            )
            or isinstance(waiver["byte_size"], bool)
            or not isinstance(waiver["byte_size"], int)
            or waiver["byte_size"] <= 0
        ):
            raise ValueError("merge manifest prompt-hash waiver provenance is invalid")
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
            or SHA256_PATTERN.fullmatch(item["sha256"]) is None
            or isinstance(item["byte_size"], bool)
            or not isinstance(item["byte_size"], int)
            or item["byte_size"] < 0
            or not isinstance(item["kind"], str)
            or not (item["shard_id"] is None or isinstance(item["shard_id"], str))
        ):
            raise ValueError("merge manifest source inventory entry is invalid")
        if item["state"] == "absent":
            if item["sha256"] != _sha256(b"") or item["byte_size"] != 0:
                raise ValueError("absent merge source must have empty hash and zero bytes")
        elif item["byte_size"] <= 0 or item["sha256"] == _sha256(b""):
            raise ValueError("regular merge source must be nonempty")
        source_paths.append(relative_path)
    if source_paths != sorted(source_paths) or len(source_paths) != len(set(source_paths)):
        raise ValueError("merge manifest source inventory order or uniqueness differs")
    dependency_sources = [
        item for item in source_files if item["kind"] == "dependency_lock"
    ]
    if len(dependency_sources) != 1:
        raise ValueError("merge source inventory requires one dependency lock")
    inventory_defects = _authoritative_source_inventory_defects(
        {
            "model_run_id": manifest["model_run_id"],
            "source_files": source_files,
            "structurally_valid": True,
            "summary": {
                "dependency_lock_sha256": dependency_sources[0]["sha256"],
                "observed": {"shards": 500},
            },
        }
    )
    if inventory_defects:
        raise ValueError(
            "merge source inventory violates the fixed contract: "
            + "; ".join(inventory_defects[:5])
        )
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
            or SHA256_PATTERN.fullmatch(summary["sha256"]) is None
            or summary["sha256"] == _sha256(b"")
            or not isinstance(summary["schema_sha256"], str)
            or SHA256_PATTERN.fullmatch(summary["schema_sha256"]) is None
            or isinstance(summary["byte_size"], bool)
            or not isinstance(summary["byte_size"], int)
            or summary["byte_size"] <= 0
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
        provenance = {
            key: manifest[key]
            for key in (
                "study_id",
                "study_manifest_hash",
                "question_manifest_hash",
                "model_run_id",
                "model_run_manifest_hash",
                "coverage_report_id",
            )
        }
        expected_schema = _schema(kind, provenance, summary["row_count"])
        expected_metadata = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in expected_schema.metadata.items()
        }
        if (
            summary["schema_sha256"] != _schema_sha256(expected_schema)
            or dict(summary["embedded_metadata"]) != expected_metadata
        ):
            raise ValueError(
                f"merge manifest output schema or embedded metadata differs for {kind}"
            )
    natural_row_count = outputs["natural_results"]["row_count"]
    checkpoint_row_count = outputs["checkpoint_results"]["row_count"]
    if natural_row_count != EXPECTED_NATURAL_COUNT:
        raise ValueError("merge manifest natural row count differs from fixed workload")
    if (
        checkpoint_row_count > EXPECTED_CHECKPOINT_COUNT
        or checkpoint_row_count % CHECKPOINTS_PER_NATURAL != 0
    ):
        raise ValueError("merge manifest checkpoint row count violates fixed workload")
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
    prompt_hash_waiver: dict[str, Any] | None = None
    prompt_hash_waiver_path: Path | None = None
    prompt_hash_waiver_bytes: bytes | None = None


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


def _read_inventory_entry_at(
    repository_descriptor: int,
    entry: Mapping[str, Any],
    *,
    after_open: Callable[[], None] | None = None,
) -> bytes | None:
    """Read and hash one inventory entry through an ``O_NOFOLLOW`` fd walk.

    The bytes are read from the same final descriptor that was type-checked, so
    a concurrent path replacement cannot substitute bytes after validation.
    """

    relative_path = str(entry["relative_path"])
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ValueError(f"source path is unsafe: {relative_path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    current_descriptor = os.dup(repository_descriptor)
    try:
        for component in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=current_descriptor
                )
            except OSError as exc:
                if entry["state"] == "absent" and exc.errno == errno.ENOENT:
                    return None
                raise ValueError(
                    f"source path cannot be opened without following links: {relative_path}"
                ) from exc
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        try:
            file_descriptor = os.open(
                relative.parts[-1], file_flags, dir_fd=current_descriptor
            )
        except OSError as exc:
            if entry["state"] == "absent" and exc.errno == errno.ENOENT:
                return None
            raise ValueError(
                f"source cannot be opened as a non-symlink file: {relative_path}"
            ) from exc
        try:
            if entry["state"] == "absent":
                raise ValueError(
                    f"source changed: expected absent but observed present: {relative_path}"
                )
            file_status = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise ValueError(f"source changed type: {relative_path}")
            if after_open is not None:
                after_open()
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            byte_size = 0
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
                byte_size += len(chunk)
            if byte_size != entry["byte_size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"source bytes changed: {relative_path}")
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(current_descriptor)


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
    report_source_files = inputs.coverage_report.get("source_files")
    if not isinstance(report_source_files, list) or canonical_json_bytes(
        list(inputs.source_files)
    ) != canonical_json_bytes(report_source_files):
        raise ValueError(
            "validated source_files must be exactly equal to coverage report source_files"
        )
    if inputs.prompt_hash_waiver is None:
        require_mergeable_coverage(inputs.coverage_report)
        if inputs.prompt_hash_waiver_path is not None or inputs.prompt_hash_waiver_bytes is not None:
            raise ValueError("merge waiver provenance is inconsistent")
    else:
        if inputs.prompt_hash_waiver_path is None or inputs.prompt_hash_waiver_bytes is None:
            raise ValueError("merge waiver bytes/path are missing")
        validate_prompt_hash_waiver(inputs.prompt_hash_waiver)
        current_waiver, current_waiver_bytes = _load_regular_json(
            inputs.prompt_hash_waiver_path, label="prompt-hash waiver"
        )
        if current_waiver != inputs.prompt_hash_waiver or current_waiver_bytes != inputs.prompt_hash_waiver_bytes:
            raise ValueError("prompt-hash waiver changed after validation")
        require_exact_failed_report(
            inputs.coverage_report,
            report_bytes=inputs.coverage_report_bytes,
            model_manifest=inputs.model_manifest,
        )
        require_production_checkout_generation_state(
            inputs.repository_root,
            expected_generation_commit=inputs.model_manifest["final_production_git_commit"],
        )
        waiver_report = inputs.prompt_hash_waiver["coverage_report"]
        if (
            waiver_report["validation_report_id"] != inputs.coverage_report["validation_report_id"]
            or waiver_report["sha256"] != _sha256(inputs.coverage_report_bytes)
            or waiver_report["byte_size"] != len(inputs.coverage_report_bytes)
        ):
            raise ValueError("waiver and failed coverage report provenance differ")
        recovery_code_root = Path(__file__).resolve().parents[1]
        if inputs.prompt_hash_waiver["recovery_git_commit"] != subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=recovery_code_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip():
            raise ValueError("current Git commit differs from prompt-hash waiver recovery commit")
        if subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=recovery_code_root, check=True, capture_output=True, text=True,
        ).stdout.strip():
            raise ValueError("tracked worktree is not clean during prompt-hash waiver recovery")
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
    if inputs.prompt_hash_waiver is None:
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
    prompt_hash_waiver_path: Path | None = None,
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
    prompt_hash_waiver: dict[str, Any] | None = None
    prompt_hash_waiver_bytes: bytes | None = None
    resolved_waiver_path: Path | None = None
    if prompt_hash_waiver_path is None:
        require_mergeable_coverage(coverage_report)
    else:
        resolved_waiver_path = Path(prompt_hash_waiver_path)
        if not resolved_waiver_path.is_absolute():
            resolved_waiver_path = repository_root / resolved_waiver_path
        resolved_waiver_path = Path(os.path.abspath(resolved_waiver_path))
        expected_waiver_path = expected_coverage_path.with_name("prompt_hash_waiver.json")
        if resolved_waiver_path != Path(os.path.abspath(expected_waiver_path)):
            raise ValueError("prompt-hash waiver path is not canonical")
        prompt_hash_waiver, prompt_hash_waiver_bytes = _load_regular_json(
            resolved_waiver_path, label="prompt-hash waiver"
        )
        validate_prompt_hash_waiver(prompt_hash_waiver)
        require_exact_failed_report(
            coverage_report,
            report_bytes=coverage_bytes,
            model_manifest=model_manifest,
        )
        report_provenance = prompt_hash_waiver["coverage_report"]
        if (
            prompt_hash_waiver["study_id"] != model_manifest["study_id"]
            or prompt_hash_waiver["model_run_id"] != model_manifest["model_run_id"]
            or prompt_hash_waiver["model_run_manifest_hash"] != model_manifest["model_run_manifest_hash"]
            or prompt_hash_waiver["generation_git_commit"] != model_manifest["final_production_git_commit"]
            or report_provenance["relative_path"] != _canonical_relative(
                repository_root, coverage_report_path, label="coverage report"
            )
            or report_provenance["validation_report_id"] != coverage_report["validation_report_id"]
            or report_provenance["sha256"] != _sha256(coverage_bytes)
            or report_provenance["byte_size"] != len(coverage_bytes)
        ):
            raise ValueError("prompt-hash waiver provenance differs from production inputs")
    if coverage_report["model_run_id"] != model_manifest["model_run_id"] or coverage_report[
        "model_run_manifest_hash"
    ] != model_manifest["model_run_manifest_hash"]:
        raise ValueError("coverage and production model-run identities differ")
    require_source_snapshot(repository_root, coverage_report["source_files"])

    if prompt_hash_waiver is None:
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
    inventory_by_shard: dict[str, list[Mapping[str, Any]]] = {}
    for entry in coverage_report["source_files"]:
        if isinstance(entry.get("shard_id"), str):
            inventory_by_shard.setdefault(entry["shard_id"], []).append(entry)
    repository_descriptor = os.open(
        repository_root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for shard_index in range(500):
            shard_id = f"shard-{shard_index:03d}"
            entries = inventory_by_shard.get(shard_id, [])
            by_kind: dict[str, list[Mapping[str, Any]]] = {}
            for entry in entries:
                by_kind.setdefault(str(entry["kind"]), []).append(entry)
            required_kinds = {
                "shard_provenance",
                "natural_results",
                "checkpoint_results",
                "audit_events",
                "finalization_marker",
            }
            if any(len(by_kind.get(kind, ())) != 1 for kind in required_kinds):
                raise ValueError(f"{shard_id} source snapshot is incomplete")
            snapshot = {
                kind: _read_inventory_entry_at(repository_descriptor, by_kind[kind][0])
                for kind in required_kinds
            }
            recovery_snapshot: dict[str, bytes] = {}
            for recovery_entry in by_kind.get("recovery_evidence", ()):
                recovery_bytes = _read_inventory_entry_at(
                    repository_descriptor, recovery_entry
                )
                if recovery_bytes is None:
                    raise ValueError(f"{shard_id} recovery evidence became absent")
                recovery_snapshot[str(recovery_entry["relative_path"])] = recovery_bytes
            store = Part1ShardStore(
                raw_root / shard_id,
                shard_id=shard_id,
                study_id=model_manifest["study_id"],
                model_run_id=model_manifest["model_run_id"],
                model_run_manifest_hash=model_manifest["model_run_manifest_hash"],
            )
            provenance_bytes = snapshot["shard_provenance"]
            if provenance_bytes is None:
                raise ValueError(f"{shard_id} provenance snapshot is absent")
            inspection = store.inspect_from_snapshot(
                provenance_header_bytes=provenance_bytes,
                stream_bytes={
                    "natural_results": snapshot["natural_results"],
                    "checkpoint_results": snapshot["checkpoint_results"],
                    "audit_events": snapshot["audit_events"],
                },
            )
            recovery_events = store.recovery_journal_events_from_snapshot(
                recovery_snapshot
            )
            index = store.build_index_from_snapshot(
                inspection, recovery_journal_events=recovery_events
            )
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
                if prompt_hash_waiver is not None:
                    compatibility_errors = _natural_compatibility(
                        row, question=question, model_manifest=model_manifest
                    )
                    if compatibility_errors != [
                        "natural prompt hash differs from model-run manifest"
                    ]:
                        raise ValueError(
                            f"{shard_id} natural record has an unrelated manifest mismatch: "
                            + "; ".join(compatibility_errors)
                        )
                    require_content_derived_prompt_hash(row)
                natural_ids.add(row["raw_record_id"])
                natural.append(row)
            for row in inspection.checkpoint_results:
                validate_instance("checkpoint_terminal_result", row)
                if row["sample_index"] != shard_index or row["question_id"] != question["question_id"]:
                    raise ValueError(f"checkpoint row is assigned to the wrong shard: {shard_id}")
                if row["checkpoint_record_id"] in checkpoint_ids:
                    raise ValueError("duplicate checkpoint record ID across shards")
                if prompt_hash_waiver is not None:
                    compatibility_errors = _checkpoint_compatibility(
                        row, question=question, model_manifest=model_manifest
                    )
                    if compatibility_errors:
                        raise ValueError(
                            f"{shard_id} checkpoint record has an unrelated manifest mismatch: "
                            + "; ".join(compatibility_errors)
                        )
                    require_complete_checkpoint_outcome(row)
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
            del snapshot, recovery_snapshot, inspection, recovery_events
    finally:
        os.close(repository_descriptor)

    natural_partition = coverage_report["summary"]["natural_partition"]
    checkpoint_partition = coverage_report["summary"]["checkpoint_partition"]
    expected_natural = (
        EXPECTED_NATURAL_COUNT
        if prompt_hash_waiver is not None
        else natural_partition["complete"] + natural_partition["terminal_infrastructure_failure"]
    )
    expected_checkpoints = (
        coverage_report["summary"]["observed"]["checkpoint_physical_records"]
        if prompt_hash_waiver is not None
        else checkpoint_partition["complete"] + checkpoint_partition["terminal_infrastructure_failure"]
    )
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
        prompt_hash_waiver=prompt_hash_waiver,
        prompt_hash_waiver_path=resolved_waiver_path,
        prompt_hash_waiver_bytes=prompt_hash_waiver_bytes,
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


def _read_regular_file_at(directory_descriptor: int, name: str) -> bytes:
    if Path(name).name != name:
        raise ValueError(f"unsafe merged output name: {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_descriptor,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"merged directory entry is nonregular: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_merge_directory_descriptor(
    directory_descriptor: int,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    expected_names = {*TABLE_FILENAMES.values(), "merge_manifest.json"}
    entries = os.listdir(directory_descriptor)
    if set(entries) != expected_names or len(entries) != len(expected_names):
        raise ValueError("merged directory has missing or extra contents")
    manifest_bytes = _read_regular_file_at(directory_descriptor, "merge_manifest.json")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"merge manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("merge manifest is not a JSON object")
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
        data = _read_regular_file_at(directory_descriptor, filename)
        summary = manifest["outputs"][kind]
        if summary["relative_path"] != filename or summary["sha256"] != _sha256(
            data
        ) or summary["byte_size"] != len(data):
            raise ValueError(f"{kind} output bytes differ from merge manifest")
        table = pq.read_table(pa.BufferReader(data))
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
            if [_canonical_object_json(row) for row in decoded] != [
                _canonical_object_json(row) for row in expected
            ]:
                raise ValueError(f"{kind} Parquet recovery is not lossless")
    return manifest


def validate_merge_directory_at(
    parent_descriptor: int,
    directory_name: str,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Validate a child directory through an already anchored parent fd."""

    if Path(directory_name).name != directory_name:
        raise ValueError(f"unsafe merged directory name: {directory_name}")
    try:
        directory_descriptor = os.open(
            directory_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise ValueError(
            f"merged directory is missing, symlinked, or non-directory: {directory_name}"
        ) from exc
    try:
        return _validate_merge_directory_descriptor(
            directory_descriptor,
            expected_manifest=expected_manifest,
            expected_rows=expected_rows,
        )
    finally:
        os.close(directory_descriptor)


def validate_merge_directory(
    directory: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Strictly validate a staged or finalized four-file merge directory."""

    directory = Path(os.path.abspath(directory))
    parent_descriptor = os.open(
        directory.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return validate_merge_directory_at(
            parent_descriptor,
            directory.name,
            expected_manifest=expected_manifest,
            expected_rows=expected_rows,
        )
    finally:
        os.close(parent_descriptor)


def _remove_own_stage(
    stage: Path,
    parent: Path,
    prefix: str,
    *,
    parent_descriptor: int,
    stage_descriptor: int,
    fault_hook: Callable[[str], None],
) -> None:
    if stage.parent != parent or not stage.name.startswith(prefix):
        raise RuntimeError("refusing to clean a directory not owned by this merge invocation")
    expected = os.fstat(stage_descriptor)
    try:
        observed = os.stat(stage.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise RuntimeError("merge staging directory identity was replaced; refusing cleanup")
    fault_hook("before_stage_cleanup_identity_move")
    cleanup_name = f".{stage.name}.cleanup-{secrets.token_hex(16)}"
    _exclusive_rename_at(parent_descriptor, stage.name, cleanup_name)
    moved = os.stat(cleanup_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (moved.st_dev, moved.st_ino) != (expected.st_dev, expected.st_ino):
        try:
            _exclusive_rename_at(parent_descriptor, cleanup_name, stage.name)
            _fsync_directory_descriptor(parent_descriptor)
        except BaseException as restore_error:
            raise PublicationStateIndeterminateError(
                "stage cleanup identity changed during quarantine and safe restore failed: "
                f"stage={stage}; quarantine={cleanup_name}; error={restore_error}"
            ) from restore_error
        raise RuntimeError("stage identity changed during cleanup; replacement was restored")
    cleanup_path = parent / cleanup_name
    shutil.rmtree(cleanup_path)
    _fsync_directory_descriptor(parent_descriptor)


def publish_merge(
    inputs: MergeInputs,
    *,
    fault_hook: Callable[[str], None] | None = None,
    return_manifest: bool = False,
) -> Path | tuple[Path, dict[str, Any]]:
    """Stage, reload, verify, and atomically publish one no-overwrite merge."""

    revalidate_merge_inputs(inputs)
    target_relative = inputs.model_manifest["output_paths"]["merged"]
    expected_relative = f"results/part1/{inputs.model_manifest['model_run_id']}/merged"
    if target_relative != expected_relative:
        raise ValueError("production merged output path is not canonical")
    target = inputs.repository_root / target_relative
    _ensure_safe_directory(target.parent)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(target.parent, directory_flags)
    prefix = f".{target.name}.stage-"
    try:
        stage = Path(tempfile.mkdtemp(prefix=prefix, dir=target.parent))
    except BaseException:
        os.close(parent_descriptor)
        raise
    try:
        initial_stage_status = os.stat(
            stage.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
    except BaseException as identity_error:
        os.close(parent_descriptor)
        raise PublicationStateIndeterminateError(
            f"stage={stage}; identity could not be established after creation, "
            "so cleanup was refused and the parent descriptor was closed"
        ) from identity_error
    try:
        stage_descriptor = os.open(stage.name, directory_flags, dir_fd=parent_descriptor)
    except BaseException:
        try:
            current = os.stat(
                stage.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if (current.st_dev, current.st_ino) == (
                initial_stage_status.st_dev,
                initial_stage_status.st_ino,
            ):
                os.rmdir(stage.name, dir_fd=parent_descriptor)
                _fsync_directory_descriptor(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        raise
    rows = {
        "natural_results": inputs.natural_records,
        "checkpoint_results": inputs.checkpoint_records,
        "audit_events": inputs.audit_events,
    }
    hook = fault_hook or (lambda _boundary: None)
    published = False
    retain_stage = False
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
        waiver_arguments: dict[str, Any] = {}
        if inputs.prompt_hash_waiver is not None:
            assert inputs.prompt_hash_waiver_path is not None
            assert inputs.prompt_hash_waiver_bytes is not None
            waiver_arguments = {
                "prompt_hash_waiver_path": _canonical_relative(
                    inputs.repository_root,
                    inputs.prompt_hash_waiver_path,
                    label="prompt-hash waiver",
                ),
                "prompt_hash_waiver_id": inputs.prompt_hash_waiver["waiver_id"],
                "prompt_hash_waiver_sha256": _sha256(inputs.prompt_hash_waiver_bytes),
                "prompt_hash_waiver_byte_size": len(inputs.prompt_hash_waiver_bytes),
            }
        manifest = build_merge_manifest(
            provenance=provenance,
            coverage_report_path=coverage_relative,
            coverage_report_sha256=_sha256(inputs.coverage_report_bytes),
            coverage_report_byte_size=len(inputs.coverage_report_bytes),
            **waiver_arguments,
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
        validate_merge_directory_at(
            parent_descriptor, stage.name, expected_manifest=manifest, expected_rows=rows
        )
        hook("reload_complete")
        hook("before_rename")
        revalidate_merge_inputs(inputs)
        if os.path.lexists(target):
            validate_merge_directory_at(
                parent_descriptor,
                target.name,
                expected_manifest=manifest,
                expected_rows=rows,
            )
            return (target, manifest) if return_manifest else target
        hook("before_exclusive_rename")
        revalidate_merge_inputs(inputs)
        try:
            _exclusive_rename_at(parent_descriptor, stage.name, target.name)
        except FileExistsError:
            if not os.path.lexists(target):
                raise
            revalidate_merge_inputs(inputs)
            validate_merge_directory_at(
                parent_descriptor,
                target.name,
                expected_manifest=manifest,
                expected_rows=rows,
            )
            return (target, manifest) if return_manifest else target
        try:
            _fsync_directory_descriptor(parent_descriptor)
        except OSError as fsync_error:
            try:
                _exclusive_rename_at(parent_descriptor, target.name, stage.name)
            except BaseException as rollback_error:
                published = True
                raise PublicationStateIndeterminateError(
                    "post-rename parent fsync failed and exclusive rollback failed; "
                    f"final path may remain at {target}: fsync={fsync_error}; "
                    f"rollback={rollback_error}"
                ) from fsync_error
            try:
                _fsync_directory_descriptor(parent_descriptor)
            except OSError as rollback_fsync_error:
                retain_stage = True
                raise PublicationStateIndeterminateError(
                    "publication was renamed back from the final path but rollback "
                    f"durability is indeterminate; final path={target}; stage={stage}; "
                    f"fsync={rollback_fsync_error}"
                ) from fsync_error
            raise PublicationDurabilityError(
                f"post-rename parent fsync failed; publication at {target} was rolled back"
            ) from fsync_error
        published = True
        return (target, manifest) if return_manifest else target
    finally:
        try:
            if not published and not retain_stage:
                _remove_own_stage(
                    stage,
                    target.parent,
                    prefix,
                    parent_descriptor=parent_descriptor,
                    stage_descriptor=stage_descriptor,
                    fault_hook=hook,
                )
        finally:
            try:
                os.close(stage_descriptor)
            finally:
                os.close(parent_descriptor)


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
    result = publish_merge(inputs, return_manifest=True)
    if not isinstance(result, tuple):
        raise RuntimeError("manifest-returning publication returned no manifest")
    return result
