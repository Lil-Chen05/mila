"""Fail-closed, read-only coverage validation for bounded Part 1 smoke shards."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping

from part1_checkpoints import build_checkpoint_probe_plans
from part1_contract import (
    attempt_id,
    checkpoint_record_id,
    derive_generation_seed,
    model_run_id,
    model_run_manifest_hash,
    natural_record_id,
    validate_fixed_model_requested_contract,
    validate_instance,
)
from part1_manifests import ManifestBundle, validate_manifest_bundle
from part1_runtime import validate_manifest_compatibility
from part1_store import Part1ShardStore, STORE_VERSION
from run_part1_shard import select_shard_work
from run_part1_smoke import select_smoke_work


SMOKE_SCOPES = frozenset({"smoke_a", "smoke_b", "phase3_smoke"})
CHECKPOINT_IDS = tuple(f"cp-{index:02d}" for index in range(11))
CORE_FILES = frozenset(
    {
        ".shard-provenance.json",
        "natural_results.jsonl",
        "checkpoint_results.jsonl",
        "audit_events.jsonl",
        ".finalized",
    }
)
OPTIONAL_FILES = frozenset({".writer.guard", ".writer-lock-takeover-event.json"})
OPTIONAL_DIRECTORIES = frozenset({"recovery_journal", "quarantine", ".lock_history"})
_FINALIZED_AT = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
_RECOVERY_NAME = re.compile(r"^[0-9a-f]{64}\.json$")
_QUARANTINE_NAME = re.compile(
    r"^(?:natural_results|checkpoint_results|audit_events)\.[0-9a-f]{64}\.trailing-bytes\.bin$"
)
_LOCK_HISTORY_NAME = re.compile(
    r"^[0-9a-f]{32}\.(?:claim\.json|event\.json|pending-quarantine)$"
)


def _safe_directory(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        raise ValueError(f"{label} is missing: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a non-symlink directory: {path}")


def _safe_components(repository_root: Path, path: Path) -> None:
    """Reject a validation source reached through a symlink component."""

    try:
        relative = path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("validation input must remain inside repository root") from exc
    current = repository_root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"validation input has a symlink component: {current}")


def _regular_bytes(path: Path, *, label: str) -> bytes:
    if not os.path.lexists(path):
        raise ValueError(f"{label} is missing: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a non-symlink regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read {label}: {path}: {exc}") from exc


def _json_object_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _manifest_bundle_from_bytes(
    *, questions: bytes, question_manifest: bytes, study_manifest: bytes
) -> ManifestBundle:
    if questions and not questions.endswith(b"\n"):
        raise ValueError("questions JSONL must end with a newline")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(questions.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid questions JSONL line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"questions JSONL line {line_number} must be an object")
        records.append(value)
    bundle = ManifestBundle(
        records=tuple(records),
        question_manifest=_json_object_bytes(
            question_manifest, label="question manifest snapshot"
        ),
        study_manifest=_json_object_bytes(study_manifest, label="study manifest snapshot"),
    )
    validate_manifest_bundle(bundle)
    return bundle


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _canonical_paths(
    repository_root: Path, manifest: Mapping[str, Any]
) -> tuple[Path, Path]:
    scope = str(manifest["execution_scope"])
    manifest_path = (
        repository_root
        / "results/part1-smoke/model-runs"
        / scope
        / "model_run_manifest.json"
    )
    if scope == "phase3_smoke":
        shard_root = (
            repository_root
            / "results/part1-smoke/phase3_smoke"
            / str(manifest["model_run_id"])
            / "raw_shards/shard-000"
        )
    else:
        shard_root = (
            repository_root
            / "results/part1-smoke"
            / scope
            / str(manifest["model_run_id"])
            / "shard-000"
        )
    return manifest_path, shard_root


def _walk_tree(root: Path) -> list[Path]:
    paths = [root]
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise ValueError(f"could not inventory shard directory {directory}: {exc}") from exc
        for child in children:
            paths.append(child)
            mode = child.lstat().st_mode
            if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                pending.append(child)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _capture_shard_tree(
    *, repository_root: Path, shard_root: Path
) -> tuple[dict[str, tuple[str, int | None, str | None]], dict[str, bytes]]:
    """Capture one validated tree and exact file bytes without following symlinks."""

    _safe_components(repository_root, shard_root)
    _safe_directory(shard_root, label="smoke shard root")
    tree: dict[str, tuple[str, int | None, str | None]] = {}
    files: dict[str, bytes] = {}
    for path in _walk_tree(shard_root):
        relative = "." if path == shard_root else path.relative_to(shard_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"shard source is a symlink: {relative}")
        if stat.S_ISDIR(mode):
            tree[relative] = ("directory", None, None)
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"shard source is not regular: {relative}")
        data = _regular_bytes(path, label=f"shard source {relative}")
        files[relative] = data
        tree[relative] = ("file", len(data), hashlib.sha256(data).hexdigest())

    top_level = {name.split("/", 1)[0] for name in tree if name != "."}
    active = sorted(
        name
        for name in top_level
        if name.startswith(".writer.lock")
        or name.startswith(".writer-lock-recovery.claim")
    )
    if active:
        raise ValueError(f"finalized smoke shard retains active lock state: {active}")
    allowed = CORE_FILES | OPTIONAL_FILES | OPTIONAL_DIRECTORIES
    unexpected = sorted(top_level.difference(allowed))
    if unexpected:
        raise ValueError(f"unexpected shard entry: {unexpected}")

    missing = sorted(CORE_FILES.difference(top_level))
    if missing:
        raise ValueError(f"shard sources are missing: {missing}")
    for name in CORE_FILES | OPTIONAL_FILES:
        if name in top_level and tree.get(name, (None, None, None))[0] != "file":
            raise ValueError(f"canonical optional/core source must be a file: {name}")
    for name in OPTIONAL_DIRECTORIES:
        if name in top_level and tree.get(name, (None, None, None))[0] != "directory":
            raise ValueError(f"canonical auxiliary source must be a directory: {name}")

    recovery_prefix = "recovery_journal/"
    for relative, entry in tree.items():
        if not relative.startswith(recovery_prefix):
            continue
        remainder = relative.removeprefix(recovery_prefix)
        if "/" in remainder or entry[0] != "file" or _RECOVERY_NAME.fullmatch(remainder) is None:
            raise ValueError(
                f"recovery journal entry has noncanonical name or layout: {relative}"
            )
    for directory, pattern in (
        ("quarantine", _QUARANTINE_NAME),
        (".lock_history", _LOCK_HISTORY_NAME),
    ):
        prefix = f"{directory}/"
        for relative, entry in tree.items():
            if not relative.startswith(prefix):
                continue
            remainder = relative.removeprefix(prefix)
            if "/" in remainder or entry[0] != "file" or pattern.fullmatch(remainder) is None:
                raise ValueError(
                    f"{directory} entry has noncanonical name or layout: {relative}"
                )
    return tree, files


def _stable_source_hashes(
    *, repository_root: Path, shard_root: Path, files: Mapping[str, bytes]
) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": (shard_root / relative).relative_to(repository_root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        }
        for relative, data in sorted(files.items())
    ]


def _verify_finalization_marker(marker_bytes: bytes, manifest: Mapping[str, Any]) -> None:
    marker = _json_object_bytes(marker_bytes, label="smoke finalization marker")
    expected = {
        "store_version": STORE_VERSION,
        "shard_id": "shard-000",
        "study_id": manifest["study_id"],
        "model_run_id": manifest["model_run_id"],
    }
    if set(marker) != {*expected, "finalized_at"} or any(
        marker.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("smoke shard finalization marker is incompatible")
    timestamp = marker["finalized_at"]
    try:
        if not isinstance(timestamp, str) or _FINALIZED_AT.fullmatch(timestamp) is None:
            raise ValueError("timestamp must use canonical UTC Z form")
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
        if parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("timestamp is not UTC")
    except ValueError as exc:
        raise ValueError(f"smoke shard finalization timestamp is invalid: {exc}") from exc


def _verify_recovery_semantics(
    *,
    recovery_events: tuple[dict[str, Any], ...],
    files: Mapping[str, bytes],
    audit_events: tuple[dict[str, Any], ...],
) -> None:
    audit_by_id = {event["event_id"]: event for event in audit_events}
    referenced_quarantine: set[str] = set()
    for event in recovery_events:
        if audit_by_id.get(event["event_id"]) != event:
            raise ValueError("recovery journal event is absent from the authoritative audit stream")
        details = event["error_details"]
        expected_detail_fields = {
            "stream",
            "recovered_byte_count",
            "recovered_bytes_sha256",
            "quarantine_artifact",
            "original_size",
            "valid_prefix_size",
            "valid_prefix_sha256",
        }
        if set(details) != expected_detail_fields:
            raise ValueError("recovery journal details have noncanonical fields")
        stream = details.get("stream")
        stream_name = {
            "natural_results": "natural_results.jsonl",
            "checkpoint_results": "checkpoint_results.jsonl",
            "audit_events": "audit_events.jsonl",
        }.get(stream)
        if stream_name is None:
            raise ValueError("recovery journal names an unsupported stream")
        stream_bytes = files[stream_name]
        prefix_size = details.get("valid_prefix_size")
        prefix_hash = details.get("valid_prefix_sha256")
        if not isinstance(prefix_size, int) or isinstance(prefix_size, bool) or prefix_size < 0:
            raise ValueError("recovery journal valid prefix size is invalid")
        recovered_count = details.get("recovered_byte_count")
        recovered_hash = details.get("recovered_bytes_sha256")
        original_size = details.get("original_size")
        if (
            not isinstance(recovered_count, int)
            or isinstance(recovered_count, bool)
            or recovered_count < 0
            or not isinstance(original_size, int)
            or isinstance(original_size, bool)
            or original_size < 0
            or not isinstance(recovered_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", recovered_hash) is None
            or not isinstance(prefix_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", prefix_hash) is None
        ):
            raise ValueError("recovery journal byte evidence is invalid")
        if event["outcome_category"] == "invalid_final_line":
            prefix = stream_bytes[:prefix_size]
            valid_state = len(prefix) == prefix_size and hashlib.sha256(
                prefix
            ).hexdigest() == prefix_hash
            artifact = details.get("quarantine_artifact")
            quarantine = files.get(f"quarantine/{artifact}") if isinstance(artifact, str) else None
            if (
                not valid_state
                or recovered_count < 1
                or original_size != prefix_size + recovered_count
                or artifact != f"{stream}.{recovered_hash}.trailing-bytes.bin"
                or quarantine is None
                or len(quarantine) != recovered_count
                or hashlib.sha256(quarantine).hexdigest() != recovered_hash
            ):
                raise ValueError("recovery quarantine evidence differs from recovered stream state")
            referenced_quarantine.add(str(artifact))
        elif event["outcome_category"] == "valid_record_missing_newline":
            prefix = stream_bytes[:prefix_size]
            valid_state = (
                len(prefix) == prefix_size
                and hashlib.sha256(prefix).hexdigest() == prefix_hash
                and stream_bytes[prefix_size : prefix_size + 1] == b"\n"
                and details.get("quarantine_artifact") is None
                and recovered_count == 0
                and recovered_hash == hashlib.sha256(b"").hexdigest()
                and original_size == prefix_size
            )
            if not valid_state:
                raise ValueError("newline recovery evidence differs from recovered stream state")
        else:
            raise ValueError("recovery journal has an unsupported recovery kind")
    observed_quarantine = {
        relative.removeprefix("quarantine/")
        for relative in files
        if relative.startswith("quarantine/")
    }
    if observed_quarantine != referenced_quarantine:
        raise ValueError("retained quarantine files differ from recovery-journal evidence")


def _verify_takeover_evidence(
    *, store: Part1ShardStore, files: Mapping[str, bytes], audit_events: tuple[dict[str, Any], ...]
) -> None:
    data = files.get(".writer-lock-takeover-event.json")
    if data is None:
        return
    event = _json_object_bytes(data, label="writer-lock takeover evidence")
    validate_instance("audit_event", event)
    store._assert_provenance(event)
    store._verify_event_identity(event)
    if event["event_scope"] != "shard" or event["event_type"] not in {
        "stale_lock_recovered",
        "operator_unlock",
    }:
        raise ValueError("writer-lock takeover evidence has incompatible semantics")
    if not any(candidate == event for candidate in audit_events):
        raise ValueError("writer-lock takeover evidence is absent from the audit stream")


def _partition(
    keys: set[tuple[Any, ...]],
    records: Mapping[tuple[Any, ...], list[Mapping[str, Any]]],
    *,
    outcome_name: str,
    eligible: set[tuple[Any, ...]] | None = None,
) -> tuple[dict[str, int], list[str], dict[tuple[Any, ...], Mapping[str, Any]]]:
    names = (
        ("complete", "terminal_infrastructure_failure", "ineligible", "missing", "duplicate")
        if eligible is not None
        else ("complete", "terminal_infrastructure_failure", "missing", "duplicate")
    )
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


def _common_compatibility(
    record: Mapping[str, Any], *, question: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    expected = {
        "study_id": manifest["study_id"],
        "model_run_id": manifest["model_run_id"],
        "model_run_manifest_hash": manifest["model_run_manifest_hash"],
        "question_manifest_hash": manifest["question_manifest_hash"],
        "question_id": question["question_id"],
        "sample_index": question["sample_index"],
        "subject": question["subject"],
        "seed_algorithm_version": manifest["seed_algorithm_version"],
    }
    for field, expected_value in expected.items():
        if field in record and record.get(field) != expected_value:
            errors.append(f"{field} differs from authoritative manifest")
    return errors


def _natural_compatibility(
    record: Mapping[str, Any], *, question: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    errors = _common_compatibility(record, question=question, manifest=manifest)
    run_id = int(record["run_id"])
    seed = derive_generation_seed(
        base_seed=manifest["base_generation_seed"],
        canonical_model_identity=manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=run_id,
        algorithm_version=manifest["seed_algorithm_version"],
    )
    expected = {
        "generation_seed": seed,
        "raw_record_id": natural_record_id(
            manifest["study_id"], manifest["model_run_id"], question["question_id"], run_id
        ),
        "terminal_attempt_id": attempt_id(
            manifest["study_id"], manifest["model_run_id"], question["question_id"], run_id,
            int(record["terminal_attempt_number"]),
        ),
        "prompt_hash": manifest["prompt_hash"],
    }
    for field, expected_value in expected.items():
        if record[field] != expected_value:
            errors.append(f"natural {field} differs from model-run manifest")
    if record["natural_execution_outcome"] == "complete" and tuple(
        record.get("checkpoint_ids") or ()
    ) != CHECKPOINT_IDS:
        errors.append("complete natural checkpoint identities differ from fixed eleven")
    components = record.get("component_versions", {})
    for component, manifest_field in (
        ("adapter", "adapter_version"),
        ("prompt", "prompt_version"),
        ("parser", "parser_version"),
    ):
        if components.get(component) != manifest[manifest_field]:
            errors.append(f"natural component {component} differs from model-run manifest")
    return errors


def _checkpoint_compatibility(
    record: Mapping[str, Any], *, question: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    errors = _common_compatibility(record, question=question, manifest=manifest)
    run_id = int(record["run_id"])
    checkpoint_id = str(record["checkpoint_id"])
    seed = derive_generation_seed(
        base_seed=manifest["base_generation_seed"],
        canonical_model_identity=manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=run_id,
        algorithm_version=manifest["seed_algorithm_version"],
    )
    expected = {
        "natural_seed": seed,
        "parent_raw_record_id": natural_record_id(
            manifest["study_id"], manifest["model_run_id"], question["question_id"], run_id
        ),
        "checkpoint_record_id": checkpoint_record_id(
            manifest["study_id"], manifest["model_run_id"], question["question_id"], run_id,
            checkpoint_id,
        ),
        "terminal_attempt_id": attempt_id(
            manifest["study_id"], manifest["model_run_id"], question["question_id"], run_id,
            int(record["terminal_attempt_number"]), checkpoint_id=checkpoint_id,
        ),
        "inducer_version": manifest["inducer_version"],
        "inducer_text": manifest["inducer_text"],
    }
    for field, expected_value in expected.items():
        if record[field] != expected_value:
            errors.append(f"checkpoint {field} differs from model-run manifest")
    if checkpoint_id not in CHECKPOINT_IDS or record["requested_checkpoint_index"] != (
        CHECKPOINT_IDS.index(checkpoint_id) if checkpoint_id in CHECKPOINT_IDS else -1
    ):
        errors.append("checkpoint identity differs from requested index")
    if record.get("token_convention") is not None and record["token_convention"] != manifest[
        "ad_token_convention"
    ]:
        errors.append("checkpoint A-D token convention differs from model-run manifest")
    if record.get("ad_token_ids") is not None and record["ad_token_ids"] != manifest[
        "ad_token_ids"
    ]:
        errors.append("checkpoint A-D token IDs differ from model-run manifest")
    components = record.get("component_versions", {})
    for component, manifest_field in (
        ("adapter", "adapter_version"),
        ("parser", "parser_version"),
        ("inducer", "inducer_version"),
    ):
        if components.get(component) != manifest[manifest_field]:
            errors.append(f"checkpoint component {component} differs from model-run manifest")
    return errors


def _verify_unchanged(
    *,
    repository_root: Path,
    manifest_sources: Mapping[Path, bytes],
    shard_root: Path,
    shard_tree: Mapping[str, tuple[str, int | None, str | None]],
    shard_files: Mapping[str, bytes],
    git_head: str,
) -> None:
    errors: list[str] = []
    for path, expected in manifest_sources.items():
        try:
            _safe_components(repository_root, path)
            observed = _regular_bytes(path, label="manifest source")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if observed != expected:
                errors.append(f"manifest source bytes changed: {path.relative_to(repository_root)}")
    try:
        current_tree, current_files = _capture_shard_tree(
            repository_root=repository_root, shard_root=shard_root
        )
    except ValueError as exc:
        errors.append(f"shard tree changed: {exc}")
    else:
        if current_tree != shard_tree or current_files != shard_files:
            errors.append("shard tree or source bytes changed")
    try:
        current_head = _git(repository_root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"Git HEAD could not be revalidated: {exc}")
    else:
        if current_head != git_head:
            errors.append(f"Git HEAD changed from {git_head} to {current_head}")
    if errors:
        raise RuntimeError("validation inputs changed during validation: " + "; ".join(errors))


def build_smoke_coverage_report(
    *, repository_root: Path, model_run_manifest_path: Path, shard_root: Path
) -> dict[str, Any]:
    """Inspect one canonical finalized smoke shard without acquiring a writer lock."""

    repository_root = Path(os.path.abspath(repository_root))
    _safe_directory(repository_root, label="repository root")
    supplied_manifest = Path(model_run_manifest_path)
    if not supplied_manifest.is_absolute():
        supplied_manifest = repository_root / supplied_manifest
    supplied_manifest = Path(os.path.abspath(supplied_manifest))
    supplied_shard = Path(shard_root)
    if not supplied_shard.is_absolute():
        supplied_shard = repository_root / supplied_shard
    supplied_shard = Path(os.path.abspath(supplied_shard))

    manifest_root = repository_root / "manifests/part1"
    tracked_paths = (
        manifest_root / "questions.jsonl",
        manifest_root / "questions.manifest.json",
        manifest_root / "study_manifest.json",
    )
    manifest_sources: dict[Path, bytes] = {}
    for path in (*tracked_paths, supplied_manifest):
        _safe_components(repository_root, path)
        manifest_sources[path] = _regular_bytes(path, label="manifest source")
    bundle = _manifest_bundle_from_bytes(
        questions=manifest_sources[tracked_paths[0]],
        question_manifest=manifest_sources[tracked_paths[1]],
        study_manifest=manifest_sources[tracked_paths[2]],
    )
    manifest = _json_object_bytes(
        manifest_sources[supplied_manifest], label="smoke model-run manifest snapshot"
    )
    validate_instance("model_run_manifest", manifest)
    validate_fixed_model_requested_contract(manifest)
    scope = manifest.get("execution_scope")
    if (
        scope not in SMOKE_SCOPES
        or manifest.get("production") is not False
        or manifest.get("schema_version") != "1.0.0"
    ):
        raise ValueError("validator supports only canonical non-production smoke scopes")
    validate_manifest_compatibility(bundle.study_manifest, manifest)
    if manifest["model_run_id"] != model_run_id(manifest) or manifest[
        "model_run_manifest_hash"
    ] != model_run_manifest_hash(manifest):
        raise ValueError("smoke model-run identities do not recompute")
    base_commit = manifest.get("smoke_git_provenance", {}).get("base_commit")
    try:
        _git(repository_root, "cat-file", "-e", f"{base_commit}^{{commit}}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("recorded smoke base commit is unavailable") from exc

    canonical_manifest, canonical_shard = _canonical_paths(repository_root, manifest)
    if supplied_manifest != Path(os.path.abspath(canonical_manifest)):
        raise ValueError("smoke model-run manifest path is not canonical")
    if supplied_shard != Path(os.path.abspath(canonical_shard)):
        raise ValueError("smoke shard root is not canonical")
    git_head = _git(repository_root, "rev-parse", "HEAD")
    shard_tree, shard_files = _capture_shard_tree(
        repository_root=repository_root, shard_root=canonical_shard
    )
    _verify_finalization_marker(shard_files[".finalized"], manifest)

    store = Part1ShardStore(
        canonical_shard,
        shard_id="shard-000",
        study_id=manifest["study_id"],
        model_run_id=manifest["model_run_id"],
        model_run_manifest_hash=manifest["model_run_manifest_hash"],
    )
    inspection = store.inspect_from_snapshot(
        provenance_header_bytes=shard_files[".shard-provenance.json"],
        stream_bytes={
            "natural_results": shard_files["natural_results.jsonl"],
            "checkpoint_results": shard_files["checkpoint_results.jsonl"],
            "audit_events": shard_files["audit_events.jsonl"],
        },
    )
    recovery_snapshot = {
        relative: data
        for relative, data in shard_files.items()
        if relative.startswith("recovery_journal/")
    }
    recovery_events = store.recovery_journal_events_from_snapshot(recovery_snapshot)
    index = store.build_index_from_snapshot(
        inspection, recovery_journal_events=recovery_events
    )
    _verify_recovery_semantics(
        recovery_events=recovery_events,
        files=shard_files,
        audit_events=inspection.audit_events,
    )
    _verify_takeover_evidence(store=store, files=shard_files, audit_events=inspection.audit_events)
    for record in (*inspection.natural_results, *inspection.checkpoint_results):
        store._validate_scientific_alignment(record)
    if (
        index.hierarchy_errors
        or index.lifecycle_errors
        or index.orphaned_attempt_ids
        or index.missing_started_attempt_ids
        or index.inconsistent_completion_attempt_ids
        or index.terminalization_required
        or index.missing_completion_record_ids
        or index.pending_recovery_event_ids
    ):
        raise ValueError("smoke shard has hierarchy or lifecycle failure")

    selection = (
        select_shard_work(bundle.records, shard_index=0, shard_count=500)
        if scope == "phase3_smoke"
        else select_smoke_work(bundle.records, execution_scope=str(scope))
    )
    expected_natural = {
        (str(question["question_id"]), int(run_id)) for question, run_id in selection
    }
    question_by_id = {
        str(question["question_id"]): question for question, _run_id in selection
    }
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
            compatibility = _natural_compatibility(
                record, question=question, manifest=manifest
            )
            if compatibility:
                raise ValueError("; ".join(compatibility))
            naturals.setdefault(key, []).append(record)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid natural record: {exc}")
    natural_partition, natural_errors, natural_terminal = _partition(
        expected_natural, naturals, outcome_name="natural_execution_outcome"
    )
    errors.extend(natural_errors)
    if set(naturals).difference(expected_natural):
        errors.append("unexpected natural key")

    eligible_checkpoint_keys: set[tuple[Any, ...]] = set()
    checkpoint_plans: dict[tuple[Any, ...], Any] = {}
    all_checkpoint_keys = {
        (question_id, run_id, checkpoint_id)
        for question_id, run_id in expected_natural
        for checkpoint_id in CHECKPOINT_IDS
    }
    for (question_id, run_id), parent in natural_terminal.items():
        if parent["natural_execution_outcome"] != "complete":
            continue
        plans = build_checkpoint_probe_plans(
            parent,
            inducer_token_ids=manifest["inducer_token_ids"],
            inducer_version=manifest["inducer_version"],
        )
        for plan in plans:
            key = (question_id, run_id, plan.checkpoint_id)
            eligible_checkpoint_keys.add(key)
            checkpoint_plans[key] = plan
    for record in inspection.checkpoint_results:
        try:
            validate_instance("checkpoint_terminal_result", record)
            key = (
                str(record["question_id"]),
                int(record["run_id"]),
                str(record["checkpoint_id"]),
            )
            question = question_by_id.get(key[0])
            if key not in all_checkpoint_keys or question is None:
                raise ValueError("unexpected checkpoint key")
            compatibility = _checkpoint_compatibility(
                record, question=question, manifest=manifest
            )
            if compatibility:
                raise ValueError("; ".join(compatibility))
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
    checkpoint_partition, checkpoint_errors, _checkpoint_terminal = _partition(
        all_checkpoint_keys,
        checkpoints,
        outcome_name="checkpoint_execution_outcome",
        eligible=eligible_checkpoint_keys,
    )
    errors.extend(checkpoint_errors)
    if set(checkpoints).difference(all_checkpoint_keys):
        errors.append("unexpected checkpoint key")
    if errors:
        raise ValueError("smoke coverage validation failed: " + "; ".join(errors[:5]))

    _verify_unchanged(
        repository_root=repository_root,
        manifest_sources=manifest_sources,
        shard_root=canonical_shard,
        shard_tree=shard_tree,
        shard_files=shard_files,
        git_head=git_head,
    )
    # Keep the established public validation gate as an independent final
    # structural check. Report assembly above consumes only captured bytes.
    validation = store.validate_shard(
        artifact_kind="natural_shard",
        started_at="1970-01-01T00:00:00Z",
        completed_at="1970-01-01T00:00:00Z",
    )
    failed_checks = [
        check["name"] for check in validation["checks"] if check["outcome"] != "passed"
    ]
    if failed_checks:
        raise ValueError("smoke shard validation failed: " + ", ".join(failed_checks))

    coverage_complete = (
        natural_partition["missing"]
        == natural_partition["duplicate"]
        == checkpoint_partition["missing"]
        == checkpoint_partition["duplicate"]
        == 0
    )
    structurally_valid = coverage_complete
    paper_analysis_ready = (
        structurally_valid
        and natural_partition["terminal_infrastructure_failure"]
        == checkpoint_partition["terminal_infrastructure_failure"]
        == 0
    )
    _verify_unchanged(
        repository_root=repository_root,
        manifest_sources=manifest_sources,
        shard_root=canonical_shard,
        shard_tree=shard_tree,
        shard_files=shard_files,
        git_head=git_head,
    )
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
            "natural_run_ids": sorted(
                {run_id for _question_id, run_id in expected_natural}
            ),
            "checkpoint_indices": list(range(11)),
            "audit_event_count": len(inspection.audit_events),
            "natural_record_count": len(inspection.natural_results),
            "checkpoint_record_count": len(inspection.checkpoint_results),
            "stable_source_hashes": _stable_source_hashes(
                repository_root=repository_root,
                shard_root=canonical_shard,
                files=shard_files,
            ),
        },
    }
