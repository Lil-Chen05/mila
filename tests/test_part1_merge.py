"""Deterministic, lossless table and manifest tests for the Part 1 merge."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def _rows(tmp_path: Path):
    from part1_store import Part1ShardStore
    from test_part1_coverage import _populate_shard, _production_fixture

    fixture = _production_fixture(tmp_path)
    shard_root = _populate_shard(fixture)
    store = Part1ShardStore(
        shard_root,
        shard_id="shard-000",
        study_id=fixture["manifest"]["study_id"],
        model_run_id=fixture["manifest"]["model_run_id"],
        model_run_manifest_hash=fixture["manifest"]["model_run_manifest_hash"],
    )
    inspection = store.inspect()
    return fixture, inspection


def _provenance(fixture: dict) -> dict[str, str]:
    manifest = fixture["manifest"]
    return {
        "study_id": manifest["study_id"],
        "study_manifest_hash": fixture["bundle"].study_manifest[
            "study_manifest_hash"
        ],
        "question_manifest_hash": manifest["question_manifest_hash"],
        "model_run_id": manifest["model_run_id"],
        "model_run_manifest_hash": manifest["model_run_manifest_hash"],
        "coverage_report_id": "c" * 64,
    }


def test_tables_sort_exactly_and_round_trip_every_raw_value(tmp_path: Path) -> None:
    from part1_merge import build_merge_table, decode_merge_table

    fixture, inspection = _rows(tmp_path)
    provenance = _provenance(fixture)
    natural = [copy.deepcopy(inspection.natural_results[0]) for _ in range(3)]
    for row, sample_index, run_id, record_digit in zip(
        natural, (2, 0, 0), (0, 1, 0), ("3", "2", "1"), strict=True
    ):
        row["sample_index"] = sample_index
        row["run_id"] = run_id
        row["raw_record_id"] = record_digit * 64
        row["reasoning_boundaries"] = {"z": [None, 2], "a": {"x": True}}
        row["component_versions"] = {"prompt": "v1", "adapter": "v2"}
    checkpoints = [copy.deepcopy(inspection.checkpoint_results[index]) for index in (1, 0)]
    audit = [copy.deepcopy(row) for row in reversed(inspection.audit_events)]

    natural_table = build_merge_table("natural_results", natural, provenance=provenance)
    checkpoint_table = build_merge_table(
        "checkpoint_results", checkpoints, provenance=provenance
    )
    audit_table = build_merge_table("audit_events", audit, provenance=provenance)

    assert [row["raw_record_id"] for row in decode_merge_table("natural_results", natural_table)] == [
        "1" * 64,
        "2" * 64,
        "3" * 64,
    ]
    assert [
        row["requested_checkpoint_index"]
        for row in decode_merge_table("checkpoint_results", checkpoint_table)
    ] == [0, 1]
    assert [
        (row["shard_id"], row["event_scope"], row["question_id"], row["run_id"],
         row["checkpoint_id"], row["attempt_number"], row["event_sequence"],
         row["event_type"], row["event_id"])
        for row in decode_merge_table("audit_events", audit_table)
    ] == [
        (row["shard_id"], row["event_scope"], row["question_id"], row["run_id"],
         row["checkpoint_id"], row["attempt_number"], row["event_sequence"],
         row["event_type"], row["event_id"])
        for row in sorted(audit, key=lambda item: (
            int(item["shard_id"].removeprefix("shard-")),
            0 if item["event_scope"] == "shard" else 1,
            item["question_id"] or "",
            -1 if item["run_id"] is None else item["run_id"],
            -2 if item["checkpoint_id"] is None else int(item["checkpoint_id"].removeprefix("cp-")),
            -1 if item["attempt_number"] is None else item["attempt_number"],
            item["event_sequence"], item["event_type"], item["event_id"],
        ))
    ]
    assert decode_merge_table("natural_results", natural_table) == sorted(
        natural, key=lambda row: (row["sample_index"], row["run_id"], row["raw_record_id"])
    )


def test_explicit_schema_handles_empty_checkpoint_and_null_object_list_values(
    tmp_path: Path,
) -> None:
    from part1_merge import (
        ENCODED_OBJECT_FIELDS,
        TABLE_COLUMN_ORDER,
        build_merge_table,
        decode_merge_table,
    )

    fixture, inspection = _rows(tmp_path)
    provenance = _provenance(fixture)
    natural = copy.deepcopy(inspection.natural_results[0])
    natural["diagnostic_answer_like_text"] = None
    natural["prompt_token_ids"] = []
    natural["reasoning_boundaries"] = {"end": None, "nested": [1, {"ok": True}]}

    table = build_merge_table("natural_results", [natural], provenance=provenance)
    empty = build_merge_table("checkpoint_results", [], provenance=provenance)

    assert table.column_names == list(TABLE_COLUMN_ORDER["natural_results"])
    assert empty.column_names == list(TABLE_COLUMN_ORDER["checkpoint_results"])
    assert empty.num_rows == 0
    assert decode_merge_table("natural_results", table) == [natural]
    assert decode_merge_table("checkpoint_results", empty) == []
    metadata = {key.decode(): value.decode() for key, value in table.schema.metadata.items()}
    assert json.loads(metadata["encoded_object_fields"]) == list(
        ENCODED_OBJECT_FIELDS["natural_results"]
    )
    assert metadata["row_count"] == "1"
    assert metadata["raw_schema_version"] == "1.0.0"
    assert all(metadata[key] == value for key, value in provenance.items())


def test_parquet_writer_is_byte_deterministic_with_pinned_metadata(tmp_path: Path) -> None:
    from part1_merge import write_parquet_tables

    fixture, inspection = _rows(tmp_path)
    provenance = _provenance(fixture)
    first = tmp_path / "first"
    second = tmp_path / "second"
    rows = (
        inspection.natural_results,
        inspection.checkpoint_results,
        inspection.audit_events,
    )
    outputs_first = write_parquet_tables(first, *rows, provenance=provenance)
    outputs_second = write_parquet_tables(second, *rows, provenance=provenance)

    assert outputs_first == outputs_second
    for filename in (
        "natural_results.parquet",
        "checkpoint_results.parquet",
        "audit_events.parquet",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
        parquet_metadata = pq.read_metadata(first / filename).metadata
        assert parquet_metadata[b"coverage_report_id"] == b"c" * 64
        assert parquet_metadata[b"merge_format_version"] == b"part1-merge-v1"


def test_merge_id_and_complete_hash_bind_content_but_exclude_locations() -> None:
    from part1_merge import build_merge_manifest, merge_id, merge_manifest_hash

    provenance = {
        "study_id": "1" * 64,
        "study_manifest_hash": "2" * 64,
        "question_manifest_hash": "3" * 64,
        "model_run_id": "4" * 64,
        "model_run_manifest_hash": "5" * 64,
        "coverage_report_id": "6" * 64,
    }
    source_files = [
        {
            "relative_path": "results/part1/source",
            "shard_id": "shard-000",
            "kind": "natural_results",
            "state": "regular_file",
            "sha256": "7" * 64,
            "byte_size": 12,
        }
    ]
    outputs = {
        "natural_results": {
            "relative_path": "natural_results.parquet",
            "sha256": "8" * 64,
            "byte_size": 20,
            "row_count": 1,
            "schema_sha256": "9" * 64,
            "embedded_metadata": provenance,
        }
    }
    first = build_merge_manifest(
        provenance=provenance,
        coverage_report_path="results/part1/validation/coverage_report.json",
        coverage_report_sha256="a" * 64,
        coverage_report_byte_size=99,
        source_files=source_files,
        outputs=outputs,
    )
    relocated = copy.deepcopy(first)
    relocated["coverage_report"]["relative_path"] = "elsewhere/report.json"
    relocated["outputs"]["natural_results"]["relative_path"] = "elsewhere.parquet"
    assert merge_id(relocated) == first["merge_id"]
    assert merge_manifest_hash(relocated) == first["merge_manifest_hash"]

    changed = copy.deepcopy(first)
    changed["source_files"][0]["sha256"] = "b" * 64
    assert merge_id(changed) != first["merge_id"]
    assert merge_manifest_hash(changed) != first["merge_manifest_hash"]

    self_changed = copy.deepcopy(first)
    self_changed["merge_id"] = "f" * 64
    self_changed["merge_manifest_hash"] = "e" * 64
    assert merge_id(self_changed) == first["merge_id"]
    assert merge_manifest_hash({**self_changed, "merge_id": first["merge_id"]}) == first[
        "merge_manifest_hash"
    ]


def test_merge_manifest_is_strictly_self_validating_below_the_top_level() -> None:
    from part1_merge import (
        build_merge_manifest,
        merge_id,
        merge_manifest_hash,
        validate_merge_manifest,
    )

    provenance = {
        "study_id": "1" * 64,
        "study_manifest_hash": "2" * 64,
        "question_manifest_hash": "3" * 64,
        "model_run_id": "4" * 64,
        "model_run_manifest_hash": "5" * 64,
        "coverage_report_id": "6" * 64,
    }
    source_files = [
        {
            "relative_path": "source",
            "shard_id": None,
            "kind": "dependency_lock",
            "state": "regular_file",
            "sha256": "7" * 64,
            "byte_size": 1,
        }
    ]
    outputs = {
        kind: {
            "relative_path": f"{kind}.parquet",
            "sha256": "8" * 64,
            "byte_size": 1,
            "row_count": 0,
            "schema_sha256": "9" * 64,
            "embedded_metadata": {},
        }
        for kind in ("natural_results", "checkpoint_results", "audit_events")
    }
    manifest = build_merge_manifest(
        provenance=provenance,
        coverage_report_path="coverage_report.json",
        coverage_report_sha256="a" * 64,
        coverage_report_byte_size=1,
        source_files=source_files,
        outputs=outputs,
    )
    validate_merge_manifest(manifest)

    malformed = copy.deepcopy(manifest)
    malformed["coverage_report"]["uncontracted"] = True
    malformed["merge_id"] = merge_id(malformed)
    malformed["merge_manifest_hash"] = merge_manifest_hash(malformed)
    with __import__("pytest").raises(ValueError, match="coverage report"):
        validate_merge_manifest(malformed)

    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["natural_results"]["byte_size"] = -1
    malformed["merge_id"] = merge_id(malformed)
    malformed["merge_manifest_hash"] = merge_manifest_hash(malformed)
    with __import__("pytest").raises(ValueError, match="output"):
        validate_merge_manifest(malformed)
