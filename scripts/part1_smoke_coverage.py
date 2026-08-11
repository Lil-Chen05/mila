"""Fail-closed, read-only coverage validation for bounded Part 1 smoke shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

from part1_checkpoints import build_checkpoint_probe_plans
from part1_contract import (
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
from run_part1_shard import select_shard_work
from run_part1_smoke import select_smoke_work


SMOKE_SCOPES = frozenset({"smoke_a", "smoke_b", "phase3_smoke"})
CHECKPOINT_IDS = tuple(f"cp-{index:02d}" for index in range(11))


def _regular_bytes(path: Path, *, label: str) -> bytes:
    if not os.path.lexists(path):
        raise ValueError(f"{label} is missing: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a non-symlink regular file: {path}")
    return path.read_bytes()


def _json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = _regular_bytes(path, label=label)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, data


def _safe_directory(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        raise ValueError(f"{label} is missing: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a non-symlink directory: {path}")


def _safe_components(repository_root: Path, path: Path) -> None:
    """Reject an otherwise regular target reached through a symlink component."""

    try:
        relative = path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("validation input must remain inside repository root") from exc
    current = repository_root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"validation input has a symlink component: {current}")


def _snapshot(paths: list[Path], repository_root: Path) -> tuple[dict[str, tuple[int, str]], str]:
    state: dict[str, tuple[int, str]] = {}
    for path in paths:
        data = _regular_bytes(path, label="validation input")
        state[path.relative_to(repository_root).as_posix()] = (len(data), hashlib.sha256(data).hexdigest())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, text=True, capture_output=True).stdout.strip()
    return state, head


def _verify_snapshot(paths: list[Path], repository_root: Path, before: dict[str, tuple[int, str]], head: str) -> None:
    after, current_head = _snapshot(paths, repository_root)
    if after != before or current_head != head:
        raise RuntimeError("validation inputs changed during validation")


def _canonical_paths(repository_root: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    scope = str(manifest["execution_scope"])
    manifest_path = repository_root / "results/part1-smoke/model-runs" / scope / "model_run_manifest.json"
    if scope == "phase3_smoke":
        shard_root = repository_root / "results/part1-smoke/phase3_smoke" / str(manifest["model_run_id"]) / "raw_shards/shard-000"
    else:
        shard_root = repository_root / "results/part1-smoke" / scope / str(manifest["model_run_id"]) / "shard-000"
    return manifest_path, shard_root


def _source_files(repository_root: Path, shard_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    allowed_top = {".shard-provenance.json", "natural_results.jsonl", "checkpoint_results.jsonl", "audit_events.jsonl", ".finalized", ".writer.guard", ".writer-lock-takeover-event.json", "recovery_journal", "quarantine", ".lock_history"}
    sources: list[dict[str, Any]] = []
    paths: list[Path] = []
    for path in sorted(shard_root.rglob("*"), key=lambda item: item.relative_to(repository_root).as_posix()):
        relative_shard = path.relative_to(shard_root)
        if relative_shard.parts[0] not in allowed_top:
            raise ValueError(f"unexpected shard entry: {relative_shard.as_posix()}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"shard source is a symlink: {relative_shard.as_posix()}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"shard source is not regular: {relative_shard.as_posix()}")
        data = path.read_bytes()
        paths.append(path)
        sources.append({"relative_path": path.relative_to(repository_root).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data)})
    required = {".shard-provenance.json", "natural_results.jsonl", "checkpoint_results.jsonl", "audit_events.jsonl", ".finalized"}
    present = {path.relative_to(shard_root).as_posix() for path in paths}
    missing = sorted(required.difference(present))
    if missing:
        raise ValueError(f"shard sources are missing: {missing}")
    return sources, paths


def _partition(keys: set[tuple[Any, ...]], records: Mapping[tuple[Any, ...], list[Mapping[str, Any]]], *, outcome_name: str, eligible: set[tuple[Any, ...]] | None = None) -> tuple[dict[str, int], list[str], dict[tuple[Any, ...], Mapping[str, Any]]]:
    names = ("complete", "terminal_infrastructure_failure", "ineligible", "missing", "duplicate") if eligible is not None else ("complete", "terminal_infrastructure_failure", "missing", "duplicate")
    result = {name: 0 for name in names}
    errors: list[str] = []
    terminal: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for key in sorted(keys):
        values = records.get(key, [])
        if eligible is not None and key not in eligible:
            result["ineligible"] += 1
            if values:
                errors.append(f"ineligible checkpoint has physical data: {key!r}")
            continue
        if len(values) == 0:
            result["missing"] += 1
        elif len(values) != 1:
            result["duplicate"] += 1
            errors.append(f"duplicate terminal result: {key!r}")
        else:
            outcome = values[0].get(outcome_name)
            if outcome not in {"complete", "terminal_infrastructure_failure"}:
                result["missing"] += 1
                errors.append(f"nonterminal result: {key!r}")
            else:
                result[str(outcome)] += 1
                terminal[key] = values[0]
    return result, errors, terminal


def build_smoke_coverage_report(*, repository_root: Path, model_run_manifest_path: Path, shard_root: Path) -> dict[str, Any]:
    """Inspect one canonical finalized smoke shard without acquiring a writer lock."""

    repository_root = Path(os.path.abspath(repository_root))
    _safe_directory(repository_root, label="repository root")
    manifest_root = repository_root / "manifests/part1"
    tracked = [manifest_root / "questions.jsonl", manifest_root / "questions.manifest.json", manifest_root / "study_manifest.json"]
    for path in tracked:
        _safe_components(repository_root, path)
        _regular_bytes(path, label="tracked manifest")
    bundle = load_manifest_bundle(questions_path=tracked[0], question_manifest_path=tracked[1], study_manifest_path=tracked[2])
    supplied_manifest = Path(model_run_manifest_path)
    if not supplied_manifest.is_absolute():
        supplied_manifest = repository_root / supplied_manifest
    _safe_components(repository_root, supplied_manifest)
    manifest, _manifest_bytes = _json_object(supplied_manifest, label="smoke model-run manifest")
    validate_instance("model_run_manifest", manifest)
    validate_fixed_model_requested_contract(manifest)
    scope = manifest.get("execution_scope")
    if scope not in SMOKE_SCOPES or manifest.get("production") is not False or manifest.get("schema_version") != "1.0.0":
        raise ValueError("validator supports only canonical non-production smoke scopes")
    validate_manifest_compatibility(bundle.study_manifest, manifest)
    if manifest["model_run_id"] != model_run_id(manifest) or manifest["model_run_manifest_hash"] != model_run_manifest_hash(manifest):
        raise ValueError("smoke model-run identities do not recompute")
    base_commit = manifest.get("smoke_git_provenance", {}).get("base_commit")
    try:
        subprocess.run(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=repository_root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError("recorded smoke base commit is unavailable") from exc
    canonical_manifest, canonical_shard = _canonical_paths(repository_root, manifest)
    if Path(os.path.abspath(supplied_manifest)) != Path(os.path.abspath(canonical_manifest)):
        raise ValueError("smoke model-run manifest path is not canonical")
    supplied_shard = Path(shard_root)
    if not supplied_shard.is_absolute():
        supplied_shard = repository_root / supplied_shard
    if Path(os.path.abspath(supplied_shard)) != Path(os.path.abspath(canonical_shard)):
        raise ValueError("smoke shard root is not canonical")
    _safe_components(repository_root, canonical_shard)
    _safe_directory(canonical_shard, label="smoke shard root")
    if any((canonical_shard / name).exists() for name in (".writer.lock", ".writer-lock-recovery.claim")):
        raise ValueError("finalized smoke shard retains active lock or pending takeover")
    sources, shard_paths = _source_files(repository_root, canonical_shard)
    snapshot_paths = [*tracked, supplied_manifest, *shard_paths]
    before, head = _snapshot(snapshot_paths, repository_root)

    selection = select_shard_work(bundle.records, shard_index=0, shard_count=500) if scope == "phase3_smoke" else select_smoke_work(bundle.records, execution_scope=str(scope))
    expected_natural = {(str(question["question_id"]), int(run_id)) for question, run_id in selection}
    question_by_id = {str(question["question_id"]): question for question, _run_id in selection}
    store = Part1ShardStore(canonical_shard, shard_id="shard-000", study_id=manifest["study_id"], model_run_id=manifest["model_run_id"], model_run_manifest_hash=manifest["model_run_manifest_hash"])
    marker, _ = _json_object(store.finalization_path, label="smoke finalization marker")
    if marker.get("store_version") != STORE_VERSION or marker.get("shard_id") != "shard-000" or marker.get("study_id") != manifest["study_id"] or marker.get("model_run_id") != manifest["model_run_id"]:
        raise ValueError("smoke shard finalization marker is incompatible")
    validation = store.validate_shard(artifact_kind="natural_shard", started_at="1970-01-01T00:00:00Z", completed_at="1970-01-01T00:00:00Z")
    failed_checks = [check["name"] for check in validation["checks"] if check["outcome"] != "passed"]
    if failed_checks:
        raise ValueError("smoke shard validation failed: " + ", ".join(failed_checks))
    inspection = store.inspect()
    index = store.build_index()
    if index.hierarchy_errors or index.lifecycle_errors or index.orphaned_attempt_ids or index.terminalization_required or index.missing_completion_record_ids:
        raise ValueError("smoke shard has hierarchy or lifecycle failure")

    errors: list[str] = []
    naturals: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    checkpoints: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in inspection.natural_results:
        try:
            validate_instance("natural_terminal_result", record)
            key = (str(record["question_id"]), int(record["run_id"]))
            question = question_by_id.get(key[0])
            if key not in expected_natural or question is None:
                raise ValueError("unexpected natural key")
            seed = derive_generation_seed(base_seed=manifest["base_generation_seed"], canonical_model_identity=manifest["canonical_model_identity"], question_id=key[0], run_id=key[1])
            if record["generation_seed"] != seed or record["raw_record_id"] != natural_record_id(manifest["study_id"], manifest["model_run_id"], key[0], key[1]):
                raise ValueError("natural canonical identity differs")
            if any(record[field] != expected for field, expected in {"study_id": manifest["study_id"], "model_run_id": manifest["model_run_id"], "model_run_manifest_hash": manifest["model_run_manifest_hash"], "question_manifest_hash": manifest["question_manifest_hash"], "sample_index": question["sample_index"], "subject": question["subject"]}.items()):
                raise ValueError("natural manifest compatibility differs")
            naturals.setdefault(key, []).append(record)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid natural record: {exc}")
    natural_partition, natural_errors, natural_terminal = _partition(expected_natural, naturals, outcome_name="natural_execution_outcome")
    errors.extend(natural_errors)
    if set(naturals).difference(expected_natural):
        errors.append("unexpected natural key")

    eligible_checkpoint_keys: set[tuple[Any, ...]] = set()
    checkpoint_plans: dict[tuple[Any, ...], Any] = {}
    all_checkpoint_keys = {(question_id, run_id, checkpoint_id) for question_id, run_id in expected_natural for checkpoint_id in CHECKPOINT_IDS}
    for (question_id, run_id), parent in natural_terminal.items():
        if parent["natural_execution_outcome"] == "complete":
            plans = build_checkpoint_probe_plans(parent, inducer_token_ids=manifest["inducer_token_ids"], inducer_version=manifest["inducer_version"])
            for plan in plans:
                key = (question_id, run_id, plan.checkpoint_id)
                eligible_checkpoint_keys.add(key)
                checkpoint_plans[key] = plan
    for record in inspection.checkpoint_results:
        try:
            validate_instance("checkpoint_terminal_result", record)
            key = (str(record["question_id"]), int(record["run_id"]), str(record["checkpoint_id"]))
            if key not in all_checkpoint_keys:
                raise ValueError("unexpected checkpoint key")
            if record["checkpoint_record_id"] != checkpoint_record_id(manifest["study_id"], manifest["model_run_id"], *key):
                raise ValueError("checkpoint canonical identity differs")
            plan = checkpoint_plans.get(key)
            if plan is not None and any(
                record[field] != expected
                for field, expected in {
                    "requested_checkpoint_index": plan.requested_checkpoint_index,
                    "requested_fraction": plan.requested_fraction,
                    "k_keep": plan.k_keep,
                    "actual_fraction": plan.actual_fraction,
                    "is_alias": plan.is_alias,
                    "alias_metadata": plan.alias_metadata,
                    "prefix_hash": plan.prefix_hash,
                    "shared_probe_id": plan.shared_probe_id,
                    "inducer_version": plan.inducer_version,
                }.items()
            ):
                raise ValueError("checkpoint probe identity differs from canonical plan")
            checkpoints.setdefault(key, []).append(record)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid checkpoint record: {exc}")
    checkpoint_partition, checkpoint_errors, _checkpoint_terminal = _partition(all_checkpoint_keys, checkpoints, outcome_name="checkpoint_execution_outcome", eligible=eligible_checkpoint_keys)
    errors.extend(checkpoint_errors)
    if set(checkpoints).difference(all_checkpoint_keys):
        errors.append("unexpected checkpoint key")
    if errors:
        raise ValueError("smoke coverage validation failed: " + "; ".join(errors[:5]))
    coverage_complete = natural_partition["missing"] == natural_partition["duplicate"] == checkpoint_partition["missing"] == checkpoint_partition["duplicate"] == 0
    structurally_valid = coverage_complete
    paper_analysis_ready = structurally_valid and natural_partition["terminal_infrastructure_failure"] == checkpoint_partition["terminal_infrastructure_failure"] == 0
    _verify_snapshot(snapshot_paths, repository_root, before, head)
    return {
        "is_valid": structurally_valid,
        "structurally_valid": structurally_valid,
        "coverage_complete": coverage_complete,
        "paper_analysis_ready": paper_analysis_ready,
        "mutation_performed": False,
        "execution_scope": scope,
        "study_id": manifest["study_id"],
        "study_manifest_hash": manifest["study_manifest_hash"],
        "question_manifest_hash": manifest["question_manifest_hash"],
        "model_run_id": manifest["model_run_id"],
        "model_run_manifest_hash": manifest["model_run_manifest_hash"],
        "summary": {
            "natural_partition": natural_partition,
            "checkpoint_partition": checkpoint_partition,
            "natural_run_ids": sorted({run_id for _question_id, run_id in expected_natural}),
            "checkpoint_indices": list(range(11)),
            "audit_event_count": len(inspection.audit_events),
            "natural_record_count": len(inspection.natural_results),
            "checkpoint_record_count": len(inspection.checkpoint_results),
            "stable_source_hashes": sorted(sources, key=lambda item: item["relative_path"]),
        },
    }
