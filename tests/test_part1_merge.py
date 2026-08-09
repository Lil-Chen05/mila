"""Deterministic, lossless table and manifest tests for the Part 1 merge."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


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


def test_lossless_raw_row_preserves_json_numeric_types_and_checks_projections(
    tmp_path: Path,
) -> None:
    from part1_merge import build_merge_table, decode_merge_table

    fixture, inspection = _rows(tmp_path)
    provenance = _provenance(fixture)
    natural = copy.deepcopy(inspection.natural_results[0])
    natural["generated_token_count"] = 3.0
    natural["prompt_token_ids"] = [1.0, 2]
    natural["generated_token_ids"] = [1, 2.0, 3]
    natural["per_token_entropy_nats"] = [1, 2.0, 3]
    natural["raw_parsed_confidence"] = 0
    natural["normalized_confidence"] = 0

    table = build_merge_table("natural_results", [natural], provenance=provenance)
    assert "raw_row_canonical_json" in table.column_names
    raw = json.loads(table["raw_row_canonical_json"][0].as_py())
    assert type(raw["generated_token_count"]) is float
    assert [type(value) for value in raw["prompt_token_ids"]] == [float, int]
    assert [type(value) for value in raw["per_token_entropy_nats"]] == [int, float, int]
    assert type(raw["normalized_confidence"]) is int
    assert type(table["generated_token_count"][0].as_py()) is int
    assert type(table["normalized_confidence"][0].as_py()) is float
    recovered = decode_merge_table("natural_results", table)
    assert json.dumps(recovered[0], sort_keys=True, separators=(",", ":")) == json.dumps(
        natural, sort_keys=True, separators=(",", ":")
    )
    metadata = {key.decode(): value.decode() for key, value in table.schema.metadata.items()}
    assert metadata["lossless_raw_row_field"] == "raw_row_canonical_json"
    assert metadata["projection_conversion_version"] == "part1-json-arrow-projection-v1"

    column_index = table.column_names.index("generated_token_count")
    corrupted = table.set_column(
        column_index,
        table.schema.field(column_index),
        pa.array([999], type=pa.int64()),
    )
    with __import__("pytest").raises(ValueError, match="projection"):
        decode_merge_table("natural_results", corrupted)


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
        TABLE_FILENAMES,
        _schema,
        _schema_sha256,
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
    def regular(relative_path: str, kind: str, shard_id: str | None) -> dict:
        payload = relative_path.encode()
        return {
            "relative_path": relative_path,
            "shard_id": shard_id,
            "kind": kind,
            "state": "regular_file",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        }

    global_paths = {
        "questions": "manifests/part1/questions.jsonl",
        "question_manifest": "manifests/part1/questions.manifest.json",
        "study_manifest": "manifests/part1/study_manifest.json",
        "model_run_manifest": f"results/part1/{provenance['model_run_id']}/model_run_manifest.json",
        "dependency_lock": "uv.lock",
    }
    source_files = [
        regular(relative_path, kind, None)
        for kind, relative_path in global_paths.items()
    ]
    core_names = {
        "shard_provenance": ".shard-provenance.json",
        "natural_results": "natural_results.jsonl",
        "checkpoint_results": "checkpoint_results.jsonl",
        "audit_events": "audit_events.jsonl",
        "finalization_marker": ".finalized",
    }
    for shard_index in range(500):
        shard_id = f"shard-{shard_index:03d}"
        prefix = f"results/part1/{provenance['model_run_id']}/raw_shards/{shard_id}"
        for kind, filename in core_names.items():
            relative_path = f"{prefix}/{filename}"
            if kind == "checkpoint_results":
                source_files.append(
                    {
                        "relative_path": relative_path,
                        "shard_id": shard_id,
                        "kind": kind,
                        "state": "absent",
                        "sha256": hashlib.sha256(b"").hexdigest(),
                        "byte_size": 0,
                    }
                )
            else:
                source_files.append(regular(relative_path, kind, shard_id))
    source_files.sort(key=lambda item: item["relative_path"])
    outputs = {
        kind: {
            "relative_path": TABLE_FILENAMES[kind],
            "sha256": "8" * 64,
            "byte_size": 1,
            "row_count": 5000 if kind == "natural_results" else 0,
            "schema_sha256": _schema_sha256(
                _schema(kind, provenance, 5000 if kind == "natural_results" else 0)
            ),
            "embedded_metadata": {
                key.decode(): value.decode()
                for key, value in _schema(
                    kind, provenance, 5000 if kind == "natural_results" else 0
                ).metadata.items()
            },
        }
        for kind in ("natural_results", "checkpoint_results", "audit_events")
    }
    manifest = build_merge_manifest(
        provenance=provenance,
        coverage_report_path=(
            f"results/part1/{provenance['model_run_id']}/validation/coverage_report.json"
        ),
        coverage_report_sha256="a" * 64,
        coverage_report_byte_size=1,
        source_files=source_files,
        outputs=outputs,
    )
    validate_merge_manifest(manifest)

    def rehash(value: dict) -> dict:
        value["merge_id"] = merge_id(value)
        value["merge_manifest_hash"] = merge_manifest_hash(value)
        return value

    mutations = []
    malformed = copy.deepcopy(manifest)
    malformed["coverage_report"]["uncontracted"] = True
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["study_manifest_hash"] = "A" * 64
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["coverage_report"]["sha256"] = "g" * 64
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"][0]["shard_id"] = {"not": "a shard"}
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"][0]["kind"] = ["questions"]
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"][0]["relative_path"] = "wrong/global"
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"] = [
        item for item in malformed["source_files"] if item["kind"] != "questions"
    ]
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    duplicated_global = copy.deepcopy(
        next(item for item in malformed["source_files"] if item["kind"] == "questions")
    )
    malformed["source_files"].append(duplicated_global)
    malformed["source_files"].sort(key=lambda item: item["relative_path"])
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"] = [
        item for item in malformed["source_files"] if item["shard_id"] != "shard-499"
    ]
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"] = [
        item
        for item in malformed["source_files"]
        if not (
            item["shard_id"] == "shard-000" and item["kind"] == "audit_events"
        )
    ]
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    extra_path = (
        f"results/part1/{provenance['model_run_id']}/raw_shards/"
        "shard-500/natural_results.jsonl"
    )
    malformed["source_files"].append(
        regular(extra_path, "natural_results", "shard-500")
    )
    malformed["source_files"].sort(key=lambda item: item["relative_path"])
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    absent = next(item for item in malformed["source_files"] if item["state"] == "absent")
    absent["sha256"] = "7" * 64
    absent["byte_size"] = 1
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["source_files"].append(
        regular(
            f"results/part1/{provenance['model_run_id']}/raw_shards/shard-000/not-canonical",
            "recovery_evidence",
            "shard-000",
        )
    )
    malformed["source_files"].sort(key=lambda item: item["relative_path"])
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["natural_results"]["embedded_metadata"]["extra"] = "field"
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    del malformed["outputs"]["natural_results"]["embedded_metadata"][
        "lossless_raw_row_field"
    ]
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["natural_results"]["embedded_metadata"][
        "projection_conversion_version"
    ] = "changed"
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["natural_results"]["schema_sha256"] = "9" * 64
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["natural_results"]["byte_size"] = -1
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["parquet_writer_settings"]["row_group_size"] = 1024.0
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["natural_results"]["row_count"] = 4999
    malformed["outputs"]["natural_results"]["schema_sha256"] = _schema_sha256(
        _schema("natural_results", provenance, 4999)
    )
    malformed["outputs"]["natural_results"]["embedded_metadata"] = {
        key.decode(): value.decode()
        for key, value in _schema("natural_results", provenance, 4999).metadata.items()
    }
    mutations.append(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["outputs"]["checkpoint_results"]["row_count"] = 10
    malformed["outputs"]["checkpoint_results"]["schema_sha256"] = _schema_sha256(
        _schema("checkpoint_results", provenance, 10)
    )
    malformed["outputs"]["checkpoint_results"]["embedded_metadata"] = {
        key.decode(): value.decode()
        for key, value in _schema("checkpoint_results", provenance, 10).metadata.items()
    }
    mutations.append(malformed)

    for malformed in mutations:
        with __import__("pytest").raises(ValueError):
            validate_merge_manifest(rehash(malformed))
