"""Synthetic CPU-only tests for the Part 1 analysis artifact lifecycle."""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import pytest

from part1_contract import FIXED_CHECKPOINT_FRACTIONS, load_config


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fixture_source(repository_root: Path):
    from part1_analysis import AnalysisSource
    from test_part1_trajectories import _checkpoint, _natural

    naturals: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    question_frame: list[dict[str, str]] = []
    for sample_index in range(2):
        question_id = _hex(f"question-{sample_index}")
        question_frame.append(
            {"subject": "high_school_mathematics", "question_id": question_id}
        )
        for run_id, answer in enumerate(("A", "C")):
            natural = _natural(
                sample_index=sample_index,
                run_id=run_id,
                answer=answer,
                confidence=0.2 if answer == "A" else 0.8,
            )
            naturals.append(natural)
            checkpoints.extend(
                _checkpoint(natural, index, answer="C") for index in range(11)
            )

    model_run_id = naturals[0]["model_run_id"]
    model_manifest = {
        "study_id": naturals[0]["study_id"],
        "study_manifest_hash": "5" * 64,
        "question_manifest_hash": naturals[0]["question_manifest_hash"],
        "model_run_id": model_run_id,
        "model_run_manifest_hash": naturals[0]["model_run_manifest_hash"],
        "output_paths": {
            "analysis": f"results/part1/{model_run_id}/analysis",
        },
    }
    coverage = {
        "validation_report_id": "6" * 64,
        "paper_analysis_ready": True,
        "summary": {
            "natural_partition": {"terminal_infrastructure_failure": 0},
            "checkpoint_partition": {"terminal_infrastructure_failure": 0},
        },
    }
    merge = {
        "merge_id": "7" * 64,
        "merge_manifest_hash": "8" * 64,
        "study_id": model_manifest["study_id"],
        "study_manifest_hash": model_manifest["study_manifest_hash"],
        "question_manifest_hash": model_manifest["question_manifest_hash"],
        "model_run_id": model_run_id,
        "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
        "coverage_report_id": coverage["validation_report_id"],
    }
    revalidations: list[str] = []
    return AnalysisSource(
        repository_root=repository_root,
        model_manifest=model_manifest,
        merge_manifest=merge,
        coverage_report=coverage,
        analysis_config=load_config("analysis"),
        question_frame=tuple(question_frame),
        natural_rows=tuple(naturals),
        checkpoint_rows=tuple(checkpoints),
        small_fixture=True,
        revalidate_inputs=lambda: revalidations.append("checked"),
    ), revalidations


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return list(reader.fieldnames or ()), list(reader), data


def _write_plain_json(path: Path, value: Any) -> None:
    import part1_analysis as analysis

    path.write_bytes(analysis._plain_json_bytes(value))


def _rewrite_csv(
    path: Path, mutate: Any
) -> None:
    columns, rows, _data = _read_csv(path)
    mutate(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode("utf-8"))


def _rehash_analysis_directory(output: Path, *, sync_table_sidecars: bool) -> None:
    import part1_analysis as analysis

    manifest_path = output / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sync_table_sidecars:
        for table, sidecar in manifest["tables"].items():
            table_bytes = (output / table).read_bytes()
            metadata_path = output / sidecar
            metadata = json.loads(metadata_path.read_text())
            metadata["table_sha256"] = hashlib.sha256(table_bytes).hexdigest()
            metadata["table_byte_size"] = len(table_bytes)
            metadata["row_count"] = table_bytes.count(b"\n") - 1
            _write_plain_json(metadata_path, metadata)
    for name, entry in manifest["artifacts"].items():
        data = (output / name).read_bytes()
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["byte_size"] = len(data)
    manifest["analysis_manifest_hash"] = analysis._analysis_manifest_hash(manifest)
    _write_plain_json(manifest_path, manifest)


def _strict_loader_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Build tiny real files while replacing only upstream accepted validators."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import part1_analysis as analysis

    repository = tmp_path / "repository"
    model_run_id = "a" * 64
    run_root = repository / "results" / "part1" / model_run_id
    merged = run_root / "merged"
    validation = run_root / "validation"
    config_path = repository / "configs" / "part1" / "analysis.json"
    merged.mkdir(parents=True)
    validation.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "configs" / "part1" / "analysis.json",
        config_path,
    )

    table_rows = {
        "natural_results": ({"row_kind": "natural"},),
        "checkpoint_results": ({"row_kind": "checkpoint"},),
    }
    outputs: dict[str, dict[str, Any]] = {}
    for kind, filename in analysis.MERGE_TABLE_FILENAMES.items():
        if kind not in table_rows:
            continue
        table = pa.table({"fixture": [kind]})
        path = merged / filename
        pq.write_table(table, path)
        data = path.read_bytes()
        outputs[kind] = {
            "relative_path": path.relative_to(repository).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
            "row_count": 1,
        }

    model = {
        "schema_version": "1.1.0",
        "production": True,
        "execution_scope": "production",
        "clean_tracked_worktree": True,
        "study_id": "1" * 64,
        "study_manifest_hash": "2" * 64,
        "question_manifest_hash": "3" * 64,
        "model_run_id": model_run_id,
        "model_run_manifest_hash": "4" * 64,
        "output_paths": {
            "raw_shards": f"results/part1/{model_run_id}/raw_shards",
            "validation": f"results/part1/{model_run_id}/validation",
            "merged": f"results/part1/{model_run_id}/merged",
            "analysis": f"results/part1/{model_run_id}/analysis",
        },
    }
    model_path = run_root / "model_run_manifest.json"
    model_path.write_bytes(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    coverage = {
        "validation_report_id": "5" * 64,
        "paper_analysis_ready": True,
        "validation_started_at": "2026-01-01T00:00:00Z",
        "validation_completed_at": "2026-01-01T00:01:00Z",
        "source_files": [],
        "summary": {
            "observed_git_commit": "6" * 40,
            "clean_tracked_worktree": True,
            "natural_partition": {"terminal_infrastructure_failure": 0},
            "checkpoint_partition": {"terminal_infrastructure_failure": 0},
        },
    }
    coverage_path = validation / "coverage_report.json"

    def write_coverage() -> bytes:
        data = json.dumps(coverage, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        coverage_path.write_bytes(data)
        return data

    coverage_bytes = write_coverage()
    merge_manifest = {
        "study_id": model["study_id"],
        "study_manifest_hash": model["study_manifest_hash"],
        "question_manifest_hash": model["question_manifest_hash"],
        "model_run_id": model["model_run_id"],
        "model_run_manifest_hash": model["model_run_manifest_hash"],
        "merge_id": "7" * 64,
        "merge_manifest_hash": "8" * 64,
        "coverage_report_id": coverage["validation_report_id"],
        "coverage_report": {
            "relative_path": coverage_path.relative_to(repository).as_posix(),
            "sha256": hashlib.sha256(coverage_bytes).hexdigest(),
            "byte_size": len(coverage_bytes),
        },
        "outputs": outputs,
    }
    records = tuple(
        {
            "sample_index": index,
            "subject": "high_school_mathematics",
            "question_id": f"{index:064x}",
        }
        for index in range(500)
    )
    bundle = SimpleNamespace(records=records, study_manifest={"study_id": model["study_id"]})
    calls: dict[str, Any] = {
        "descriptor_reads": [],
        "merge_validations": 0,
        "compatibility": 0,
        "coverage_semantics": 0,
        "snapshot_errors": [],
    }

    monkeypatch.setattr(analysis, "validate_instance", lambda *_args: None)
    monkeypatch.setattr(analysis, "validate_fixed_model_requested_contract", lambda *_args: None)
    monkeypatch.setattr(analysis, "model_run_id", lambda value: value["model_run_id"])
    monkeypatch.setattr(
        analysis,
        "model_run_manifest_hash",
        lambda value: value["model_run_manifest_hash"],
    )

    def validate_merge(_path: Path, *, expected_manifest: Any = None):
        calls["merge_validations"] += 1
        if expected_manifest is not None:
            assert expected_manifest == merge_manifest
        return copy.deepcopy(merge_manifest)

    monkeypatch.setattr(analysis, "validate_merge_directory", validate_merge)
    monkeypatch.setattr(analysis, "validate_merge_directory_at", lambda *_args, **_kwargs: None)
    original_read_at = analysis._read_regular_file_at

    def read_at(descriptor: int, filename: str) -> bytes:
        calls["descriptor_reads"].append(filename)
        return original_read_at(descriptor, filename)

    monkeypatch.setattr(analysis, "_read_regular_file_at", read_at)
    monkeypatch.setattr(
        analysis,
        "decode_merge_table",
        lambda kind, _table: list(copy.deepcopy(table_rows[kind])),
    )
    monkeypatch.setattr(analysis, "coverage_report_id", lambda value: value["validation_report_id"])

    def validate_coverage(_value: Any) -> None:
        calls["coverage_semantics"] += 1

    monkeypatch.setattr(analysis, "validate_coverage_report_semantics", validate_coverage)
    monkeypatch.setattr(analysis, "build_coverage_report", lambda **_kwargs: copy.deepcopy(coverage))
    monkeypatch.setattr(analysis, "load_manifest_bundle", lambda **_kwargs: bundle)

    def compatible(_study: Any, _model: Any) -> None:
        calls["compatibility"] += 1

    monkeypatch.setattr(analysis, "validate_manifest_compatibility", compatible)
    monkeypatch.setattr(
        analysis,
        "_global_snapshot_errors",
        lambda **_kwargs: list(calls["snapshot_errors"]),
    )
    return {
        "analysis": analysis,
        "repository": repository,
        "model": model,
        "model_path": model_path,
        "merge": merge_manifest,
        "coverage": coverage,
        "coverage_path": coverage_path,
        "write_coverage": write_coverage,
        "bundle": bundle,
        "calls": calls,
    }


def _explicit_recovery_loader_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Extend the strict fixture with the explicit recovery authorization files."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    analysis = fixture["analysis"]
    repository = fixture["repository"]
    merged = repository / fixture["model"]["output_paths"]["merged"]
    validation = fixture["coverage_path"].parent
    tracked = repository / "manifests" / "part1"
    tracked.mkdir(parents=True)
    for name in ("questions.jsonl", "questions.manifest.json", "study_manifest.json"):
        (tracked / name).write_text("fixture", encoding="utf-8")
    (repository / "uv.lock").write_text("fixture", encoding="utf-8")
    fixture["model"]["final_production_git_commit"] = "a" * 40
    fixture["model_path"].write_bytes(
        json.dumps(fixture["model"], sort_keys=True, separators=(",", ":")).encode()
    )

    tables = {}
    for kind, output in fixture["merge"]["outputs"].items():
        output["relative_path"] = analysis.MERGE_TABLE_FILENAMES[kind]
        table_path = merged / analysis.MERGE_TABLE_FILENAMES[kind]
        tables[kind] = pq.read_table(table_path)
        output["schema_sha256"] = hashlib.sha256(
            tables[kind].schema.serialize().to_pybytes()
        ).hexdigest()
        output["embedded_metadata"] = {}
    audit_path = merged / analysis.MERGE_TABLE_FILENAMES["audit_events"]
    audit_table = pa.table({"fixture": ["audit"]})
    pq.write_table(audit_table, audit_path)
    audit_bytes = audit_path.read_bytes()
    tables["audit_events"] = audit_table
    fixture["merge"]["outputs"]["audit_events"] = {
        "relative_path": analysis.MERGE_TABLE_FILENAMES["audit_events"],
        "sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "byte_size": len(audit_bytes),
        "row_count": 1,
        "schema_sha256": hashlib.sha256(
            audit_table.schema.serialize().to_pybytes()
        ).hexdigest(),
        "embedded_metadata": {},
    }
    fixture["merge"]["schema_version"] = "1.1.0"
    fixture["merge"]["source_files"] = []

    waiver = {
        "waiver_id": "b" * 64,
        "recovery_git_commit": "c" * 40,
        "model_run_id": fixture["model"]["model_run_id"],
        "model_run_manifest_hash": fixture["model"]["model_run_manifest_hash"],
        "generation_git_commit": fixture["model"]["final_production_git_commit"],
        "coverage_report": {
            "validation_report_id": fixture["coverage"]["validation_report_id"],
            "sha256": hashlib.sha256(fixture["coverage_path"].read_bytes()).hexdigest(),
            "byte_size": len(fixture["coverage_path"].read_bytes()),
        },
    }
    waiver_path = validation / "prompt_hash_waiver.json"
    waiver_bytes = json.dumps(waiver, sort_keys=True, separators=(",", ":")).encode()
    waiver_path.write_bytes(waiver_bytes)
    fixture["merge"]["prompt_hash_waiver"] = {
        "relative_path": waiver_path.relative_to(repository).as_posix(),
        "waiver_id": waiver["waiver_id"],
        "sha256": hashlib.sha256(waiver_bytes).hexdigest(),
        "byte_size": len(waiver_bytes),
    }
    merge_path = merged / "merge_manifest.json"
    merge_bytes = json.dumps(
        fixture["merge"], sort_keys=True, separators=(",", ":")
    ).encode()
    merge_path.write_bytes(merge_bytes)
    sidecar = {
        "merge_stage_recovery_id": "d" * 64,
        "publication_recovery_commit": "e" * 40,
        "original_merge_recovery_commit": "f" * 40,
        "merge_manifest": {
            "merge_id": fixture["merge"]["merge_id"],
            "merge_manifest_hash": fixture["merge"]["merge_manifest_hash"],
            "sha256": hashlib.sha256(merge_bytes).hexdigest(),
            "byte_size": len(merge_bytes),
        },
        "outputs": copy.deepcopy(fixture["merge"]["outputs"]),
        "source_inventory_sha256": hashlib.sha256(
            analysis.canonical_json_bytes([])
        ).hexdigest(),
        "prompt_hash_waiver": {
            "waiver_id": waiver["waiver_id"],
            "sha256": hashlib.sha256(waiver_bytes).hexdigest(),
            "byte_size": len(waiver_bytes),
        },
        "coverage_report": {
            "validation_report_id": fixture["coverage"]["validation_report_id"],
            "sha256": hashlib.sha256(fixture["coverage_path"].read_bytes()).hexdigest(),
            "byte_size": len(fixture["coverage_path"].read_bytes()),
        },
    }
    sidecar_path = validation / "merge_stage_recovery.json"
    sidecar_bytes = json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode()
    sidecar_path.write_bytes(sidecar_bytes)
    receipt = {
        "schema_version": "part1-direct-analysis-recovery-receipt-v1",
        "direct_analysis_recovery_id": "1" * 64,
        "model_run_id": fixture["model"]["model_run_id"],
        "model_run_manifest_hash": fixture["model"]["model_run_manifest_hash"],
        "merge_stage_recovery_id": sidecar["merge_stage_recovery_id"],
        "merge_stage_recovery_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "merge_stage_recovery_byte_size": len(sidecar_bytes),
        "analysis_execution_commit": "2" * 40,
        "bootstrap_replicates": 5000,
        "no_preflight": True,
        "status": "submitted",
        "analysis_job_id": "12345",
    }
    receipt_path = validation / "direct_analysis_recovery_receipt.json"
    receipt_path.write_bytes(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    )

    monkeypatch.setattr(analysis, "validate_prompt_hash_waiver", lambda *_args: None)
    monkeypatch.setattr(analysis, "require_exact_failed_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analysis, "require_production_checkout_generation_state", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(analysis, "validate_merge_stage_recovery", lambda *_args: None)
    monkeypatch.setattr(analysis, "validate_merge_manifest", lambda *_args: None, raising=False)
    monkeypatch.setattr(analysis, "validate_direct_analysis_recovery_receipt", lambda *_args: None, raising=False)
    monkeypatch.setattr(analysis, "_current_git_state", lambda *_args: ("2" * 40, False), raising=False)
    monkeypatch.setattr(
        analysis,
        "_schema",
        lambda kind, _provenance, _row_count: tables[kind].schema,
        raising=False,
    )
    monkeypatch.setattr(
        analysis,
        "_schema_sha256",
        lambda schema: hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
        raising=False,
    )
    return {
        **fixture,
        "waiver_path": waiver_path,
        "sidecar_path": sidecar_path,
        "receipt_path": receipt_path,
    }


def test_analysis_config_uses_an_independent_exact_json_typed_oracle() -> None:
    from part1_analysis import analysis_config_hash, validate_analysis_config

    config = load_config("analysis")
    validate_analysis_config(config)
    first_hash = analysis_config_hash(config)
    assert len(first_hash) == 64

    for field, value in (
        ("bootstrap_seed", 42.0),
        ("final_bootstrap_replicates", 1000),
        ("primary_target", "checkpoint_local_correct"),
    ):
        drifted = copy.deepcopy(config)
        drifted[field] = value
        with pytest.raises(ValueError, match="analysis config"):
            validate_analysis_config(drifted)


def test_tracked_pretty_printed_analysis_config_loads_as_exact_json_typed_input() -> None:
    from part1_analysis import _read_analysis_config

    repository_root = Path(__file__).resolve().parents[1]
    config, data, path = _read_analysis_config(repository_root)
    assert config == load_config("analysis")
    assert data == path.read_bytes()
    assert data.startswith(b"{\n")


def test_strict_production_loader_reads_canonical_descriptor_bound_inputs_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    source = fixture["analysis"].load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
    )

    assert source.small_fixture is False
    assert source.natural_rows == ({"row_kind": "natural"},)
    assert source.checkpoint_rows == ({"row_kind": "checkpoint"},)
    assert source.question_frame[0]["question_id"] == f"{0:064x}"
    assert source.question_frame[-1]["question_id"] == f"{499:064x}"
    assert fixture["calls"]["descriptor_reads"] == [
        fixture["analysis"].MERGE_TABLE_FILENAMES["natural_results"],
        fixture["analysis"].MERGE_TABLE_FILENAMES["checkpoint_results"],
    ]
    assert fixture["calls"]["coverage_semantics"] == 1
    assert fixture["calls"]["compatibility"] == 1
    assert fixture["calls"]["merge_validations"] >= 3

    fixture["calls"]["snapshot_errors"].append("tracked source changed")
    with pytest.raises(ValueError, match="source/Git snapshot changed"):
        source.revalidate_inputs()


def test_explicit_recovery_loader_reads_each_merged_file_once_without_full_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _explicit_recovery_loader_fixture(tmp_path, monkeypatch)
    analysis = fixture["analysis"]
    forbidden: list[str] = []

    def forbid(label: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden.append(label)
            raise AssertionError(f"explicit recovery called forbidden {label}")
        return fail

    monkeypatch.setattr(analysis, "validate_merge_directory", forbid("full merge validator"))
    monkeypatch.setattr(analysis, "validate_merge_directory_at", forbid("descriptor full validator"))
    monkeypatch.setattr(analysis, "require_source_snapshot", forbid("raw shard scan"))
    monkeypatch.setattr(
        analysis.pq,
        "read_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("audit/full-table read_table path is forbidden")
        ),
    )
    decoded: list[str] = []
    original_decode = analysis.decode_merge_table

    def decode(kind: str, table: Any):
        decoded.append(kind)
        if kind == "audit_events":
            raise AssertionError("audit events must not be fully decoded")
        return original_decode(kind, table)

    monkeypatch.setattr(analysis, "decode_merge_table", decode)
    source = analysis.load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
        prompt_hash_waiver_path=fixture["waiver_path"],
        merge_stage_recovery_path=fixture["sidecar_path"],
        direct_analysis_recovery_receipt_path=fixture["receipt_path"],
    )

    assert source.merge_stage_recovery is not None
    assert source.merge_stage_recovery["analysis_execution_commit"] == "2" * 40
    assert decoded == ["natural_results", "checkpoint_results"]
    assert forbidden == []
    assert fixture["calls"]["descriptor_reads"] == [
        "merge_manifest.json",
        analysis.MERGE_TABLE_FILENAMES["natural_results"],
        analysis.MERGE_TABLE_FILENAMES["checkpoint_results"],
        analysis.MERGE_TABLE_FILENAMES["audit_events"],
    ]
    source.revalidate_inputs()
    assert forbidden == []


def test_direct_analysis_recovery_launcher_submits_one_no_preflight_job() -> None:
    submitter = Path("scripts/submit_part1_direct_analysis_recovery.py").read_text(
        encoding="utf-8"
    )
    job = Path("jobs/part1_direct_analysis_recovery.sh").read_text(encoding="utf-8")
    assert "_exclusive_create_json" in submitter
    assert "direct_analysis_recovery_receipt.json" in submitter
    assert submitter.count("jobs/part1_direct_analysis_recovery.sh") == 1
    assert "part1_validate" not in submitter
    assert "jobs/part1_merge" not in submitter
    assert "preflight" not in submitter.lower().replace("no_preflight", "")
    assert "--direct-analysis-recovery-receipt" in job
    assert "--merge-stage-recovery" in job
    assert "--prompt-hash-waiver" in job
    assert "srun --cpu-bind=none" in job


def test_strict_production_loader_rejects_merged_table_hash_drift_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    natural_path = (
        fixture["repository"]
        / fixture["model"]["output_paths"]["merged"]
        / fixture["analysis"].MERGE_TABLE_FILENAMES["natural_results"]
    )
    natural_path.write_bytes(natural_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="natural_results bytes differ"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production", False),
        ("execution_scope", "smoke"),
        ("schema_version", "1.0.0"),
    ],
)
def test_strict_production_loader_rejects_smoke_or_legacy_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    fixture["model"][field] = value
    fixture["model_path"].write_bytes(
        json.dumps(fixture["model"], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    with pytest.raises(ValueError, match="canonical production"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_rejects_noncanonical_path_and_mixed_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    alternate = fixture["repository"] / "alternate-model-run-manifest.json"
    alternate.write_bytes(fixture["model_path"].read_bytes())
    with pytest.raises(ValueError, match="path is not canonical"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"], model_run_manifest_path=alternate
        )

    fixture["merge"]["question_manifest_hash"] = "9" * 64
    with pytest.raises(ValueError, match="model/merge provenance differs"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_rejects_coverage_identity_readiness_and_rebuild_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    fixture["coverage"]["paper_analysis_ready"] = False
    data = fixture["write_coverage"]()
    fixture["merge"]["coverage_report"]["sha256"] = hashlib.sha256(data).hexdigest()
    fixture["merge"]["coverage_report"]["byte_size"] = len(data)
    with pytest.raises(ValueError, match="paper-final analysis requires"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )

    fixture = _strict_loader_fixture(tmp_path / "identity", monkeypatch)
    fixture["merge"]["coverage_report_id"] = "f" * 64
    with pytest.raises(ValueError, match="coverage bytes or identity"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )

    fixture = _strict_loader_fixture(tmp_path / "rebuild", monkeypatch)
    monkeypatch.setattr(
        fixture["analysis"],
        "build_coverage_report",
        lambda **_kwargs: {"different": True},
    )
    with pytest.raises(ValueError, match="current immutable source/Git snapshot"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_rejects_tracked_bundle_or_dependency_incompatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture["analysis"],
        "load_manifest_bundle",
        lambda **_kwargs: SimpleNamespace(
            records=fixture["bundle"].records[:-1],
            study_manifest=fixture["bundle"].study_manifest,
        ),
    )
    with pytest.raises(ValueError, match="fixed ordered 500-question frame"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )

    fixture = _strict_loader_fixture(tmp_path / "dependency", monkeypatch)

    def incompatible(*_args: Any) -> None:
        raise ValueError("dependency compatibility drift")

    monkeypatch.setattr(fixture["analysis"], "validate_manifest_compatibility", incompatible)
    with pytest.raises(ValueError, match="dependency compatibility drift"):
        fixture["analysis"].load_production_analysis_source(
            repository_root=fixture["repository"],
            model_run_manifest_path=fixture["model_path"],
        )


def test_strict_production_loader_revalidation_detects_config_or_coverage_byte_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    source = fixture["analysis"].load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
    )
    fixture["coverage"]["validation_completed_at"] = "2026-01-01T00:02:00Z"
    fixture["coverage_path"].write_bytes(
        json.dumps(
            fixture["coverage"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    with pytest.raises(ValueError, match="coverage report changed"):
        source.revalidate_inputs()

    fixture = _strict_loader_fixture(tmp_path / "config", monkeypatch)
    source = fixture["analysis"].load_production_analysis_source(
        repository_root=fixture["repository"],
        model_run_manifest_path=fixture["model_path"],
    )
    config_path = fixture["repository"] / "configs" / "part1" / "analysis.json"
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="analysis config changed"):
        source.revalidate_inputs()


@pytest.mark.parametrize(
    "ancestor",
    [
        "repository_root",
        "results",
        "part1",
        "run_root",
        "merge",
        "coverage",
        "tracked_manifests",
        "analysis_config",
        "dependency_lock",
        "coverage_source",
    ],
)
def test_production_loader_explicitly_guards_every_symlinked_input_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ancestor: str
) -> None:
    fixture = _strict_loader_fixture(tmp_path, monkeypatch)
    analysis = fixture["analysis"]
    repository = fixture["repository"]
    run_root = fixture["model_path"].parent
    model_path = fixture["model_path"]

    def replace_with_symlink(path: Path, label: str) -> None:
        held = path.with_name(f"{path.name}-{label}-real")
        path.rename(held)
        path.symlink_to(held, target_is_directory=True)

    def replace_file_with_symlink(path: Path, label: str) -> None:
        held = path.with_name(f"{path.name}-{label}-real")
        path.rename(held)
        path.symlink_to(held)

    if ancestor == "repository_root":
        alias = tmp_path / "repository-alias"
        alias.symlink_to(repository, target_is_directory=True)
        repository = alias
        model_path = alias / fixture["model_path"].relative_to(fixture["repository"])
        guarded_input = repository
    elif ancestor == "results":
        replace_with_symlink(repository / "results", ancestor)
        guarded_input = model_path
    elif ancestor == "part1":
        replace_with_symlink(repository / "results" / "part1", ancestor)
        guarded_input = model_path
    elif ancestor == "run_root":
        replace_with_symlink(run_root, ancestor)
        guarded_input = model_path
    elif ancestor == "merge":
        merge_path = repository / fixture["model"]["output_paths"]["merged"]
        replace_with_symlink(merge_path, ancestor)
        guarded_input = merge_path
    elif ancestor == "coverage":
        replace_with_symlink(fixture["coverage_path"].parent, ancestor)
        guarded_input = fixture["coverage_path"]
    elif ancestor == "tracked_manifests":
        manifest_root = repository / "manifests" / "part1"
        manifest_root.mkdir(parents=True)
        replace_with_symlink(repository / "manifests", ancestor)
        guarded_input = manifest_root / "questions.jsonl"
    elif ancestor == "analysis_config":
        config_path = repository / "configs" / "part1" / "analysis.json"
        replace_with_symlink(repository / "configs", ancestor)
        guarded_input = config_path
    elif ancestor == "dependency_lock":
        lock_path = repository / "uv.lock"
        lock_path.write_text("fixture", encoding="utf-8")
        replace_file_with_symlink(lock_path, ancestor)
        guarded_input = lock_path
    else:
        source_path = repository / "tracked" / "source.txt"
        source_path.parent.mkdir()
        source_path.write_text("fixture", encoding="utf-8")
        replace_with_symlink(source_path.parent, ancestor)
        fixture["coverage"]["source_files"] = [
            {"relative_path": "tracked/source.txt"}
        ]
        coverage_bytes = fixture["write_coverage"]()
        fixture["merge"]["coverage_report"]["sha256"] = hashlib.sha256(
            coverage_bytes
        ).hexdigest()
        fixture["merge"]["coverage_report"]["byte_size"] = len(coverage_bytes)
        guarded_input = source_path

    guarded: list[Path] = []
    original_guard = analysis._require_no_symlink_components

    def record_guard(path: Path) -> None:
        guarded.append(Path(path))
        original_guard(path)

    monkeypatch.setattr(analysis, "_require_no_symlink_components", record_guard)
    if ancestor in {"dependency_lock", "coverage_source"}:
        original_build = analysis.build_coverage_report

        def build_after_guard(**kwargs: Any):
            assert guarded_input in guarded
            return original_build(**kwargs)

        monkeypatch.setattr(analysis, "build_coverage_report", build_after_guard)
    with pytest.raises(ValueError, match="symlink"):
        analysis.load_production_analysis_source(
            repository_root=repository,
            model_run_manifest_path=model_path,
        )
    assert guarded_input in guarded


def test_one_trajectory_build_and_one_shared_plan_feed_every_fixed_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    original_builder = analysis.build_trajectory_rows
    builder_calls = 0
    plan_ids: list[int] = []

    def build_once(*args: Any, **kwargs: Any):
        nonlocal builder_calls
        builder_calls += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(analysis, "build_trajectory_rows", build_once)
    for name in (
        "primary_auroc_analysis",
        "natural_calibration_analysis",
        "checkpoint_calibration_analysis",
        "secondary_checkpoint_auroc_analysis",
        "within_question_analysis",
    ):
        original = getattr(analysis, name)

        def wrapper(rows: Any, plan: Any, *args: Any, _original=original, **kwargs: Any):
            plan_ids.append(id(plan))
            return _original(rows, plan, *args, **kwargs)

        monkeypatch.setattr(analysis, name, wrapper)

    computation = analysis.compute_analysis(source, bootstrap_replicates=3)
    assert builder_calls == 1
    assert len(set(plan_ids)) == 1
    assert len(computation.trajectory_rows) == len(source.natural_rows)
    assert computation.summary["repetition_filter_applied"] is False
    assert computation.summary["successful_abnormal_output_policy"] == "preserved"
    checkpoint = computation.analyses["checkpoint_calibration"]
    assert checkpoint["predictors"] == [
        "checkpoint_normalized_confidence",
        "checkpoint_maximum_ad_probability",
    ]
    assert {
        row["requested_fraction"] for row in checkpoint["metric_rows"]
    } == set(FIXED_CHECKPOINT_FRACTIONS)
    assert {
        row["requested_fraction"]
        for row in checkpoint["metric_rows"]
        if row["is_main_checkpoint"]
    } == {0.0, 0.5, 1.0}
    assert all(row["target"] == "natural_correct" for row in computation.tables["primary_auroc"])
    assert all(
        row["target"] == "checkpoint_local_correct"
        for row in computation.tables["secondary_checkpoint_auroc"]
    )
    assert all("entropy" not in row["predictor"] for row in computation.tables["calibration_metrics"])


def test_switching_stabilization_summary_preserves_unavailable_and_not_applicable(
    tmp_path: Path,
) -> None:
    from part1_analysis import compute_analysis

    source, _ = _fixture_source(tmp_path)
    computation = compute_analysis(source, bootstrap_replicates=2)
    pooled = computation.tables["trajectory_events"][0]
    assert pooled["grouping"] == "pooled"
    assert pooled["trajectory_count"] == 4
    assert pooled["switch_count_available"] == 4
    assert pooled["switch_count_unavailable"] == 0
    assert pooled["first_appearance_found"] == 2
    assert pooled["first_appearance_not_found"] == 2
    assert pooled["later_recovery_not_applicable"] == 4
    assert pooled["endpoint_agreement_true"] == 2
    assert pooled["endpoint_agreement_false"] == 2
    assert pooled["stabilization_computed"] == 4
    assert pooled["stabilization_unavailable"] == 0


def test_deterministic_artifacts_sidecars_summary_and_plot_series(tmp_path: Path) -> None:
    from part1_analysis import EXPECTED_ARTIFACT_NAMES, publish_analysis

    source, revalidations = _fixture_source(tmp_path)
    output, manifest = publish_analysis(source, bootstrap_replicates=3)
    assert output.name == "development-r3"
    assert set(path.name for path in output.iterdir()) == set(EXPECTED_ARTIFACT_NAMES)
    assert revalidations == ["checked", "checked"]
    assert manifest == json.loads((output / "analysis_manifest.json").read_text())
    summary = json.loads((output / "analysis_summary.json").read_text())
    assert summary["analysis_id"] == manifest["analysis_id"]
    assert summary["paper_analysis_ready"] is True
    assert summary["terminal_infrastructure_failure_count"] == 0
    assert summary["all_checkpoint_fractions_table"] == "calibration_metrics.csv"
    assert len(summary["plot_series"]["checkpoint_ece"]) == 22
    assert {
        row["requested_fraction"]
        for row in summary["plot_series"]["checkpoint_ece"]
        if row["is_main_checkpoint"]
    } == {0.0, 0.5, 1.0}

    for table_name, metadata_name in manifest["tables"].items():
        columns, rows, data = _read_csv(output / table_name)
        assert not data.startswith(b"#")
        assert data.count(b"\n") == len(rows) + 1
        metadata = json.loads((output / metadata_name).read_text())
        assert metadata["ordered_columns"] == columns
        assert metadata["row_count"] == len(rows)
        assert metadata["table_sha256"] == hashlib.sha256(data).hexdigest()
        assert metadata["analysis_config_hash"] == manifest["analysis_config_hash"]
        assert metadata["source_provenance"]["merge_manifest_hash"] == "8" * 64
    _, trajectory_rows, _ = _read_csv(output / "trajectory_features.csv")
    assert isinstance(json.loads(trajectory_rows[0]["checkpoint_calibration"]), list)
    assert isinstance(json.loads(trajectory_rows[0]["feature_missing_reasons"]), dict)
    for plot in manifest["plots"]:
        payload = (output / plot).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) > 100


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("schema", lambda metadata: metadata.__setitem__("schema_name", "drift")),
        (
            "source_provenance",
            lambda metadata: metadata["source_provenance"].__setitem__(
                "merge_manifest_hash", "f" * 64
            ),
        ),
        (
            "analysis_contract",
            lambda metadata: metadata.__setitem__(
                "analysis_contract_version", "drift"
            ),
        ),
        (
            "bootstrap",
            lambda metadata: metadata.__setitem__("bootstrap_seed", 42.0),
        ),
        (
            "nested_columns",
            lambda metadata: metadata.__setitem__("nested_json_columns", []),
        ),
    ],
)
def test_rehashed_sidecar_provenance_groups_are_exactly_validated(
    tmp_path: Path, case: str, mutate: Any
) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path / case)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    metadata_path = output / "trajectory_features.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    mutate(metadata)
    _write_plain_json(metadata_path, metadata)
    _rehash_analysis_directory(output, sync_table_sidecars=False)
    with pytest.raises(ValueError):
        validate_analysis_directory(output)


def test_sidecars_bind_an_exact_versioned_column_type_contract(tmp_path: Path) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    metadata_path = output / "primary_auroc.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["column_type_contract"]["version"].startswith(
        "part1-analysis-csv-types-"
    )
    metadata["column_type_contract"]["columns"][0]["type"] = "integer"
    _write_plain_json(metadata_path, metadata)
    _rehash_analysis_directory(output, sync_table_sidecars=False)
    with pytest.raises(ValueError, match="column|sidecar"):
        validate_analysis_directory(output)


def test_sidecar_numeric_json_types_and_metric_status_enums_are_exact(
    tmp_path: Path,
) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path / "sidecar")
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    metadata_path = output / "trajectory_features.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["table_byte_size"] = float(metadata["table_byte_size"])
    _write_plain_json(metadata_path, metadata)
    _rehash_analysis_directory(output, sync_table_sidecars=False)
    with pytest.raises(ValueError, match="sidecar"):
        validate_analysis_directory(output)

    source, _ = _fixture_source(tmp_path / "enum")
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    _rewrite_csv(
        output / "secondary_checkpoint_auroc.csv",
        lambda rows: rows[0].__setitem__("point_estimate_status", "unknown"),
    )
    _rehash_analysis_directory(output, sync_table_sidecars=True)
    with pytest.raises(ValueError, match="status|enum"):
        validate_analysis_directory(output)


@pytest.mark.parametrize(
    ("case", "filename", "mutate"),
    [
        (
            "primary_target",
            "primary_auroc.csv",
            lambda rows: rows[0].__setitem__("target", "checkpoint_local_correct"),
        ),
        (
            "interval_arithmetic",
            "primary_auroc.csv",
            lambda rows: rows[0].__setitem__("valid_replicates", "1"),
        ),
        (
            "checkpoint_fraction_key",
            "calibration_metrics.csv",
            lambda rows: [
                row.__setitem__("requested_fraction", "0.0")
                for row in rows
                if row["calibration_family"] == "checkpoint_confidence"
                and row["requested_fraction"] == "0.1"
            ],
        ),
        (
            "negative_bin_count",
            "reliability_bins.csv",
            lambda rows: rows[0].__setitem__("count", "-1"),
        ),
        (
            "duplicate_distribution_key",
            "within_question_distribution.csv",
            lambda rows: rows[1].__setitem__("question_id", rows[0]["question_id"]),
        ),
        (
            "wrong_integer_type",
            "trajectory_features.csv",
            lambda rows: rows[0].__setitem__("sample_index", "1.5"),
        ),
        (
            "trajectory_provenance",
            "trajectory_features.csv",
            lambda rows: rows[0].__setitem__("study_id", "f" * 64),
        ),
        (
            "trajectory_subject",
            "trajectory_features.csv",
            lambda rows: rows[0].__setitem__("subject", "unknown_subject"),
        ),
        (
            "natural_answer_enum",
            "trajectory_features.csv",
            lambda rows: rows[0].__setitem__("natural_answer", "Z"),
        ),
        (
            "reliability_contribution",
            "reliability_bins.csv",
            lambda rows: next(
                row for row in rows if int(row["count"]) > 0
            ).__setitem__("weighted_ece_contribution", "0.123"),
        ),
        (
            "within_qualifying_count",
            "within_question_summary.csv",
            lambda rows: rows[0].__setitem__("qualifying_question_count", "999"),
        ),
        (
            "event_source_count",
            "trajectory_events.csv",
            lambda rows: [
                rows[0].__setitem__(field, str(int(rows[0][field]) + 1))
                for field in (
                    "trajectory_count",
                    "switch_count_unavailable",
                    "first_appearance_unavailable",
                    "left_correct_unavailable",
                    "later_recovery_unavailable",
                    "endpoint_agreement_unavailable",
                    "stabilization_unavailable",
                )
            ],
        ),
    ],
)
def test_rehashed_scientific_csv_drift_is_rejected_semantically(
    tmp_path: Path, case: str, filename: str, mutate: Any
) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path / case)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    _rewrite_csv(output / filename, mutate)
    _rehash_analysis_directory(output, sync_table_sidecars=True)
    with pytest.raises(ValueError):
        validate_analysis_directory(output)


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "primary_rows",
            lambda summary: summary["primary_main_rows"][0].__setitem__(
                "point_estimate", 0.123
            ),
        ),
        (
            "natural_rows",
            lambda summary: summary["natural_calibration_main_rows"][0].__setitem__(
                "point_estimate", 0.123
            ),
        ),
        (
            "checkpoint_rows",
            lambda summary: summary["checkpoint_calibration_main_rows"][0].__setitem__(
                "point_estimate", 0.123
            ),
        ),
        (
            "within_rows",
            lambda summary: summary["within_question_summaries"][0].__setitem__(
                "mean_paired_difference", 0.123
            ),
        ),
        (
            "event_rows",
            lambda summary: summary["switching_stabilization_summaries"][0].__setitem__(
                "trajectory_count", 999
            ),
        ),
        (
            "output_path",
            lambda summary: summary["output_tables"].__setitem__(
                "primary_auroc", "elsewhere.csv"
            ),
        ),
        (
            "policy",
            lambda summary: summary.__setitem__("repetition_filter_applied", True),
        ),
        (
            "count",
            lambda summary: summary.__setitem__("trajectory_row_count", 999),
        ),
        (
            "primary_plot",
            lambda summary: summary["plot_series"]["primary_auroc"][0].__setitem__(
                "point_estimate", 0.123
            ),
        ),
        (
            "checkpoint_family",
            lambda summary: [
                row.__setitem__("calibration_family", "checkpoint_confidence")
                for row in summary["plot_series"]["checkpoint_ece"]
                if row["calibration_family"] == "maximum_ad_probability"
            ],
        ),
        (
            "family_main_marker",
            lambda summary: next(
                row
                for row in summary["plot_series"]["checkpoint_ece"]
                if row["calibration_family"] == "checkpoint_confidence"
                and row["requested_fraction"] == 0.0
            ).__setitem__("is_main_checkpoint", False),
        ),
    ],
)
def test_rehashed_summary_and_plot_series_drift_is_cross_checked_to_tables(
    tmp_path: Path, case: str, mutate: Any
) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path / case)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    summary_path = output / "analysis_summary.json"
    summary = json.loads(summary_path.read_text())
    mutate(summary)
    _write_plain_json(summary_path, summary)
    _rehash_analysis_directory(output, sync_table_sidecars=False)
    with pytest.raises(ValueError):
        validate_analysis_directory(output)


@pytest.mark.parametrize("case", ["trajectory_event_count", "within_count"])
def test_jointly_rehashed_summary_and_table_counts_still_bind_source_tables(
    tmp_path: Path, case: str
) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path / case)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    summary_path = output / "analysis_summary.json"
    summary = json.loads(summary_path.read_text())
    if case == "trajectory_event_count":
        fields = (
            "trajectory_count",
            "switch_count_unavailable",
            "first_appearance_unavailable",
            "left_correct_unavailable",
            "later_recovery_unavailable",
            "endpoint_agreement_unavailable",
            "stabilization_unavailable",
        )

        def mutate(rows: list[dict[str, str]]) -> None:
            for field in fields:
                rows[0][field] = str(int(rows[0][field]) + 1)

        _rewrite_csv(output / "trajectory_events.csv", mutate)
        for field in fields:
            summary["switching_stabilization_summaries"][0][field] += 1
            summary["plot_series"]["switching_stabilization"][0][field] += 1
    else:
        _rewrite_csv(
            output / "within_question_summary.csv",
            lambda rows: rows[0].__setitem__("qualifying_question_count", "999"),
        )
        summary["within_question_summaries"][0]["qualifying_question_count"] = 999
    _write_plain_json(summary_path, summary)
    _rehash_analysis_directory(output, sync_table_sidecars=True)
    with pytest.raises(ValueError, match="source|distribution|trajectory"):
        validate_analysis_directory(output)


@pytest.mark.parametrize(
    "case",
    [
        "event_switch",
        "event_appearance",
        "event_recovery",
        "event_stabilization",
        "within_mean_median",
        "checkpoint_present",
        "checkpoint_index",
        "checkpoint_fraction",
        "checkpoint_present_type",
        "source_checkpoint_count",
    ],
)
def test_coordinated_rehash_cannot_change_recomputable_source_semantics(
    tmp_path: Path, case: str
) -> None:
    from part1_analysis import publish_analysis, validate_analysis_directory

    source, _ = _fixture_source(tmp_path / case)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    summary_path = output / "analysis_summary.json"
    summary = json.loads(summary_path.read_text())

    if case.startswith("event_"):
        table_row = summary["switching_stabilization_summaries"][0]
        plot_row = summary["plot_series"]["switching_stabilization"][0]

        def mutate_event(rows: list[dict[str, str]]) -> None:
            row = rows[0]
            if case == "event_switch":
                row["switch_count_sum"] = str(int(row["switch_count_sum"]) + 1)
                row["switch_count_mean"] = json.dumps(
                    int(row["switch_count_sum"]) / int(row["switch_count_available"])
                )
            elif case == "event_appearance":
                row["first_appearance_found"] = str(
                    int(row["first_appearance_found"]) - 1
                )
                row["first_appearance_not_found"] = str(
                    int(row["first_appearance_not_found"]) + 1
                )
            elif case == "event_recovery":
                row["later_recovery_true"] = str(
                    int(row["later_recovery_true"]) + 1
                )
                row["later_recovery_not_applicable"] = str(
                    int(row["later_recovery_not_applicable"]) - 1
                )
            else:
                row["stabilization_mean_fraction"] = json.dumps(
                    float(row["stabilization_mean_fraction"]) + 0.1
                )

        _rewrite_csv(output / "trajectory_events.csv", mutate_event)
        if case == "event_switch":
            table_row["switch_count_sum"] += 1
            table_row["switch_count_mean"] = (
                table_row["switch_count_sum"] / table_row["switch_count_available"]
            )
            plot_row["switch_count_sum"] = table_row["switch_count_sum"]
            plot_row["switch_count_mean"] = table_row["switch_count_mean"]
        elif case == "event_appearance":
            for row in (table_row, plot_row):
                row["first_appearance_found"] -= 1
                row["first_appearance_not_found"] += 1
        elif case == "event_recovery":
            for row in (table_row, plot_row):
                row["later_recovery_true"] += 1
                row["later_recovery_not_applicable"] -= 1
        else:
            for row in (table_row, plot_row):
                row["stabilization_mean_fraction"] += 0.1
    elif case == "within_mean_median":
        _rewrite_csv(
            output / "within_question_summary.csv",
            lambda rows: [
                rows[0].__setitem__(field, json.dumps(float(rows[0][field]) + 0.1))
                for field in ("mean_paired_difference", "median_paired_difference")
            ],
        )
        for field in ("mean_paired_difference", "median_paired_difference"):
            summary["within_question_summaries"][0][field] += 0.1
            summary["plot_series"]["within_question"][0][field] += 0.1
    elif case == "source_checkpoint_count":
        summary["source_checkpoint_row_count"] += 1
    else:
        def mutate_slots(rows: list[dict[str, str]]) -> None:
            slots = json.loads(rows[0]["checkpoint_calibration"])
            if case == "checkpoint_present":
                slots[0]["present"] = False
            elif case == "checkpoint_index":
                slots[0]["requested_checkpoint_index"] = 1
            elif case == "checkpoint_fraction":
                slots[0]["requested_fraction"] = 0.2
            else:
                slots[0]["present"] = 1
            rows[0]["checkpoint_calibration"] = json.dumps(
                slots, sort_keys=True, separators=(",", ":")
            )

        _rewrite_csv(output / "trajectory_features.csv", mutate_slots)
        if case == "checkpoint_present":
            summary["source_checkpoint_row_count"] -= 1

    _write_plain_json(summary_path, summary)
    _rehash_analysis_directory(output, sync_table_sidecars=True)
    with pytest.raises(ValueError, match="checkpoint|trajectory|within|event|source"):
        validate_analysis_directory(output)


def test_identity_is_relocation_invariant_and_sensitive_to_bootstrap_or_source(
    tmp_path: Path,
) -> None:
    from part1_analysis import compute_analysis

    first, _ = _fixture_source(tmp_path / "first")
    second, _ = _fixture_source(tmp_path / "second")
    one = compute_analysis(first, bootstrap_replicates=2)
    relocated = compute_analysis(second, bootstrap_replicates=2)
    changed_replicates = compute_analysis(first, bootstrap_replicates=3)
    changed_source = copy.deepcopy(first.merge_manifest)
    changed_source["merge_manifest_hash"] = "9" * 64
    altered = copy.copy(first)
    object.__setattr__(altered, "merge_manifest", changed_source)
    changed_merge = compute_analysis(altered, bootstrap_replicates=2)
    assert one.analysis_id == relocated.analysis_id
    assert one.analysis_id != changed_replicates.analysis_id
    assert one.analysis_id != changed_merge.analysis_id


def test_identical_rerun_keeps_inode_and_malformed_collision_fails(tmp_path: Path) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    inode = output.stat().st_ino
    same, _ = publish_analysis(source, bootstrap_replicates=2)
    assert same.stat().st_ino == inode
    (output / "unexpected.txt").write_text("collision")
    with pytest.raises(ValueError, match="missing or extra"):
        publish_analysis(source, bootstrap_replicates=2)


def test_explicit_recovery_mode_publishes_with_cooperative_claim_on_gpfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    source = replace(
        source,
        publication_mode="cooperative_claim_same_parent_atomic_rename_v1",
    )
    monkeypatch.setattr(
        analysis,
        "_exclusive_rename_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("renameat2-EINVAL")),
    )
    output, manifest = analysis.publish_analysis(source, bootstrap_replicates=2)
    assert analysis.validate_analysis_directory(output) == manifest
    assert not output.with_name(".development-r2.publish-claim").exists()


def test_recovery_mode_competing_claim_and_target_race_fail_without_overwrite(
    tmp_path: Path
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    source = replace(source, publication_mode="cooperative_claim_same_parent_atomic_rename_v1")
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    root.mkdir(parents=True)
    claim = root / ".development-r2.publish-claim"
    claim.mkdir()
    with pytest.raises(FileExistsError, match="claim"):
        analysis.publish_analysis(source, bootstrap_replicates=2)
    claim.rmdir()

    target = root / "development-r2"
    def race(boundary: str) -> None:
        if boundary == "after_stage_identity_check":
            target.mkdir()
            (target / "winner").write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="appeared"):
        analysis.publish_analysis(source, bootstrap_replicates=2, fault_hook=race)
    assert (target / "winner").read_bytes() == b"do-not-overwrite"


def test_analysis_cleanup_error_never_masks_active_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    source = replace(source, publication_mode="cooperative_claim_same_parent_atomic_rename_v1")
    monkeypatch.setattr(
        analysis,
        "_remove_own_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup-einval")),
    )

    def fail(boundary: str) -> None:
        if boundary == "artifacts_written":
            raise RuntimeError("primary-analysis-failure")

    with pytest.raises(RuntimeError, match="primary-analysis-failure") as captured:
        analysis.publish_analysis(source, bootstrap_replicates=2, fault_hook=fail)
    assert any("cleanup-einval" in note for note in captured.value.__notes__)


def test_merge_stage_recovery_provenance_enters_all_analysis_artifact_layers(
    tmp_path: Path
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    waiver = {"waiver_id": _hex("waiver")}
    source.merge_manifest["schema_version"] = "1.1.0"
    source.merge_manifest["prompt_hash_waiver"] = {
        "relative_path": f"results/part1/{source.model_manifest['model_run_id']}/validation/prompt_hash_waiver.json",
        "waiver_id": waiver["waiver_id"],
        "sha256": _hex("waiver-bytes"),
        "byte_size": 100,
    }
    source.merge_manifest["coverage_report"] = {
        "sha256": _hex("coverage-bytes"), "byte_size": 200,
    }
    provenance = {
        "merge_stage_recovery_id": _hex("stage-recovery"),
        "relative_path": f"results/part1/{source.model_manifest['model_run_id']}/validation/merge_stage_recovery.json",
        "sha256": _hex("stage-recovery-bytes"),
        "byte_size": 1234,
        "original_merge_recovery_commit": "1" * 40,
        "publication_recovery_commit": "2" * 40,
    }
    source = replace(
        source,
        prompt_hash_waiver=waiver,
        merge_stage_recovery=provenance,
        publication_mode="cooperative_claim_same_parent_atomic_rename_v1",
    )
    output, manifest = analysis.publish_analysis(source, bootstrap_replicates=2)
    assert manifest["merge_stage_recovery"] == provenance
    summary = json.loads((output / "analysis_summary.json").read_text())
    assert summary["merge_stage_recovery"] == provenance
    metadata = json.loads((output / "primary_auroc.metadata.json").read_text())
    assert metadata["source_provenance"]["merge_stage_recovery"] == provenance


def test_recovery_mode_first_post_rename_lookup_failure_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    source = replace(source, publication_mode="cooperative_claim_same_parent_atomic_rename_v1")
    original = analysis._stat_directory_name_at

    def fail(descriptor: int, name: str, *, label: str):
        if label == "published analysis":
            raise OSError("post-rename-lookup")
        return original(descriptor, name, label=label)

    monkeypatch.setattr(analysis, "_stat_directory_name_at", fail)
    with pytest.raises(analysis.PublicationStateIndeterminateError, match="indeterminate"):
        analysis.publish_analysis(source, bootstrap_replicates=2)
    target = tmp_path / source.model_manifest["output_paths"]["analysis"] / "development-r2"
    assert target.is_dir()


def test_manifest_standalone_identity_and_json_types_reject_rehashed_tampering(
    tmp_path: Path,
) -> None:
    from part1_analysis import (
        _analysis_manifest_hash,
        publish_analysis,
        validate_analysis_manifest,
    )

    source, _ = _fixture_source(tmp_path)
    _output, manifest = publish_analysis(source, bootstrap_replicates=2)
    validate_analysis_manifest(manifest)
    for field, value in (
        ("merge_manifest_hash", "9" * 64),
        ("bootstrap_seed", 42.0),
        ("bootstrap_replicates", 2.0),
    ):
        tampered = copy.deepcopy(manifest)
        tampered[field] = value
        tampered["analysis_manifest_hash"] = _analysis_manifest_hash(tampered)
        with pytest.raises(ValueError):
            validate_analysis_manifest(tampered)


@pytest.mark.parametrize(
    "boundary",
    [
        "stage_created",
        "artifacts_written",
        "stage_fsynced",
        "reload_complete",
        "before_input_revalidation",
        "before_exclusive_rename",
    ],
)
def test_prepublication_faults_leave_no_final_and_clean_own_stage(
    tmp_path: Path, boundary: str
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)

    def fail(observed: str) -> None:
        if observed == boundary:
            raise RuntimeError(boundary)

    with pytest.raises(RuntimeError, match=boundary):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=fail)
    analysis_root = (
        tmp_path / source.model_manifest["output_paths"]["analysis"]
    )
    assert not (analysis_root / "development-r2").exists()
    assert not list(analysis_root.glob(".development-r2.stage-*"))


def test_postrename_fault_preserves_final_as_indeterminate_without_rollback(
    tmp_path: Path,
) -> None:
    from part1_analysis import PublicationStateIndeterminateError, publish_analysis

    source, _ = _fixture_source(tmp_path)

    def fail(boundary: str) -> None:
        if boundary == "after_exclusive_rename":
            raise OSError("injected post-rename durability failure")

    with pytest.raises(PublicationStateIndeterminateError, match="preserv|indeterminate"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=fail)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    assert (root / "development-r2").is_dir()
    assert not list(root.glob(".development-r2.stage-*"))


def test_stage_cleanup_quarantines_by_open_identity_and_restores_a_substitution(
    tmp_path: Path,
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    analysis_root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def fault(boundary: str) -> None:
        nonlocal held
        if boundary == "artifacts_written":
            raise RuntimeError("enter cleanup")
        if boundary == "before_stage_cleanup_identity_move":
            stage = next(analysis_root.glob(".development-r2.stage-*"))
            held = stage.with_name(f"{stage.name}.held")
            stage.rename(held)
            stage.mkdir()
            (stage / "replacement-marker").write_text("unrelated", encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity changed"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=fault)
    assert held is not None and held.is_dir()
    replacement = next(
        path
        for path in analysis_root.glob(".development-r2.stage-*")
        if path != held
    )
    assert (replacement / "replacement-marker").read_text() == "unrelated"
    assert not (analysis_root / "development-r2").exists()


def _replace_directory_with_copy(path: Path, suffix: str) -> Path:
    held = path.with_name(f"{path.name}.{suffix}")
    path.rename(held)
    shutil.copytree(held, path)
    return held


def test_stage_substitution_after_reload_is_rejected_before_rename(
    tmp_path: Path,
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal held
        if boundary == "before_stage_identity_check":
            stage = next(root.glob(".development-r2.stage-*"))
            held = _replace_directory_with_copy(stage, "original-held")

    with pytest.raises(RuntimeError, match="identity|indeterminate"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=substitute)
    assert held is not None and held.is_dir()
    replacement = next(path for path in root.glob(".development-r2.stage-*") if path != held)
    assert replacement.is_dir()
    assert not (root / "development-r2").exists()


def test_stage_substitution_after_prerename_check_is_caught_postrename(
    tmp_path: Path,
) -> None:
    from part1_analysis import PublicationStateIndeterminateError, publish_analysis

    source, _ = _fixture_source(tmp_path)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal held
        if boundary == "after_stage_identity_check":
            stage = next(root.glob(".development-r2.stage-*"))
            held = _replace_directory_with_copy(stage, "original-held")

    with pytest.raises(PublicationStateIndeterminateError, match="original|inode"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=substitute)
    assert held is not None and held.is_dir()
    assert (root / "development-r2").is_dir()
    assert not [
        path for path in root.glob(".development-r2.stage-*") if path != held
    ]


def test_final_substitution_after_rename_is_never_reported_as_success(
    tmp_path: Path,
) -> None:
    from part1_analysis import PublicationStateIndeterminateError, publish_analysis

    source, _ = _fixture_source(tmp_path)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal held
        if boundary == "after_exclusive_rename":
            held = _replace_directory_with_copy(
                root / "development-r2", "original-held"
            )

    with pytest.raises(PublicationStateIndeterminateError, match="original|inode"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=substitute)
    assert held is not None and held.is_dir()
    assert (root / "development-r2").is_dir()


def test_final_substitution_after_descriptor_validation_is_rechecked_before_success(
    tmp_path: Path,
) -> None:
    from part1_analysis import PublicationStateIndeterminateError, publish_analysis

    source, _ = _fixture_source(tmp_path)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal held
        if boundary == "published_final_validated":
            held = _replace_directory_with_copy(
                root / "development-r2", "validated-held"
            )

    with pytest.raises(PublicationStateIndeterminateError, match="original|inode"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=substitute)
    assert held is not None and held.is_dir()
    assert (root / "development-r2").is_dir()


def test_final_substitution_on_postrename_failure_is_preserved_without_rollback(
    tmp_path: Path,
) -> None:
    from part1_analysis import PublicationStateIndeterminateError, publish_analysis

    source, _ = _fixture_source(tmp_path)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None

    def substitute_and_fail(boundary: str) -> None:
        nonlocal held
        if boundary == "after_exclusive_rename":
            held = _replace_directory_with_copy(
                root / "development-r2", "original-held"
            )
            raise OSError("force post-rename durability failure")

    with pytest.raises(PublicationStateIndeterminateError, match="rollback|inode"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=substitute_and_fail)
    assert held is not None and held.is_dir()
    assert (root / "development-r2").is_dir()
    assert not list(root.glob(".development-r2.stage-*"))


def test_durability_failure_never_attempts_name_based_rollback_and_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    held: Path | None = None
    durability_armed = False
    rename_calls = 0
    original_fsync = analysis._fsync_directory_descriptor
    original_rename = analysis._exclusive_rename_at

    def hook(boundary: str) -> None:
        nonlocal durability_armed, held
        if boundary == "published_final_validated":
            durability_armed = True
        if boundary == "durability_failure_preserved":
            held = _replace_directory_with_copy(
                root / "development-r2", "original-held"
            )

    def fail_durability(descriptor: int) -> None:
        if durability_armed:
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    def count_renames(*args: Any, **kwargs: Any) -> None:
        nonlocal rename_calls
        rename_calls += 1
        original_rename(*args, **kwargs)

    monkeypatch.setattr(analysis, "_fsync_directory_descriptor", fail_durability)
    monkeypatch.setattr(analysis, "_exclusive_rename_at", count_renames)
    with pytest.raises(
        analysis.PublicationStateIndeterminateError,
        match="preserv|durability|indeterminate",
    ):
        analysis.publish_analysis(source, bootstrap_replicates=2, fault_hook=hook)
    assert rename_calls == 1
    assert held is not None and held.is_dir()
    assert (root / "development-r2").is_dir()
    assert not list(root.glob(".development-r2.stage-*"))


def test_identical_existing_final_must_still_name_the_validated_open_inode(
    tmp_path: Path,
) -> None:
    from part1_analysis import PublicationStateIndeterminateError, publish_analysis

    source, _ = _fixture_source(tmp_path)
    target, _ = publish_analysis(source, bootstrap_replicates=2)
    held: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal held
        if boundary == "existing_final_validated":
            held = _replace_directory_with_copy(target, "validated-held")

    with pytest.raises(PublicationStateIndeterminateError, match="existing|inode"):
        publish_analysis(source, bootstrap_replicates=2, fault_hook=substitute)
    assert held is not None and held.is_dir()
    assert target.is_dir()


def test_two_publishers_converge_on_one_identical_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_analysis as analysis

    source, _ = _fixture_source(tmp_path)
    (tmp_path / source.model_manifest["output_paths"]["analysis"]).mkdir(
        parents=True
    )
    barrier = Barrier(2)
    render_lock = Lock()
    original_render = analysis._render_plots

    # Matplotlib's process-global state is not thread-safe. Serialize only plot
    # rendering so the test isolates the publication race it is intended to cover.
    def render(*args: Any, **kwargs: Any) -> None:
        with render_lock:
            original_render(*args, **kwargs)

    monkeypatch.setattr(analysis, "_render_plots", render)

    def publish():
        def hook(boundary: str) -> None:
            if boundary == "before_exclusive_rename":
                barrier.wait(timeout=30)

        return analysis.publish_analysis(source, bootstrap_replicates=2, fault_hook=hook)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result() for future in (executor.submit(publish), executor.submit(publish))]
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]
    assert not list(results[0][0].parent.glob(".development-r2.stage-*"))


@pytest.mark.parametrize("mutation", ["missing", "content", "symlink"])
def test_existing_partial_divergent_or_symlinked_artifact_is_incompatible(
    tmp_path: Path, mutation: str
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    output, _ = publish_analysis(source, bootstrap_replicates=2)
    target = output / "primary_auroc.csv"
    if mutation == "missing":
        target.unlink()
    elif mutation == "content":
        target.write_bytes(target.read_bytes() + b"corrupt\n")
    else:
        target.unlink()
        target.symlink_to(output / "analysis_summary.json")
    with pytest.raises(ValueError):
        publish_analysis(source, bootstrap_replicates=2)


def test_false_paper_readiness_or_any_terminal_failure_blocks_publication(
    tmp_path: Path,
) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    for mutator in (
        lambda coverage: coverage.update(paper_analysis_ready=False),
        lambda coverage: coverage["summary"]["natural_partition"].update(
            terminal_infrastructure_failure=1
        ),
        lambda coverage: coverage["summary"]["checkpoint_partition"].update(
            terminal_infrastructure_failure=1
        ),
    ):
        changed = copy.deepcopy(source.coverage_report)
        mutator(changed)
        invalid = copy.copy(source)
        object.__setattr__(invalid, "coverage_report", changed)
        with pytest.raises(ValueError, match="paper-final analysis"):
            publish_analysis(invalid, bootstrap_replicates=2)


def test_analysis_publication_rejects_symlinked_final_directory(tmp_path: Path) -> None:
    from part1_analysis import publish_analysis

    source, _ = _fixture_source(tmp_path)
    analysis_root = tmp_path / source.model_manifest["output_paths"]["analysis"]
    analysis_root.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (analysis_root / "development-r2").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked|non-directory"):
        publish_analysis(source, bootstrap_replicates=2)


def test_fixture_evidence_is_explicitly_limited(tmp_path: Path) -> None:
    from part1_analysis import FIXTURE_EVIDENCE_LIMITS, compute_analysis

    source, _ = _fixture_source(tmp_path)
    result = compute_analysis(source, bootstrap_replicates=1)
    assert result.summary["evidence_scope"] == FIXTURE_EVIDENCE_LIMITS
    assert "real SmolLM3" in FIXTURE_EVIDENCE_LIMITS
