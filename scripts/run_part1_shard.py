#!/usr/bin/env python3
"""Run one manifest-bound Part 1 production or Phase 3 smoke shard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
import uuid

from part1_checkpoints import (
    build_alias_checkpoint_terminal_result,
    build_checkpoint_infrastructure_failure_result,
    build_checkpoint_probe_plans,
)
from part1_contract import FIXED_SUBJECTS, derive_generation_seed
from part1_generation import CHECKPOINT_IDS, build_natural_infrastructure_failure_result
from part1_manifests import load_manifest_bundle
from part1_model_run import validate_preflight_model_run_compatibility
from part1_runtime import (
    LockHeldError,
    LockMetadata,
    LockedShardSession,
    WorkSpec,
)
from part1_smollm3_adapter import (
    load_model_and_tokenizer,
    preflight_tokenizer_contract,
)
from part1_storage_estimate import assess_free_space, estimate_part1_storage
from part1_store import Part1ShardStore
from run_part1_smoke import (
    _execute_checkpoint,
    _execute_natural,
    _run_work_lifecycle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "part1"
DEFAULT_PREFLIGHT = REPOSITORY_ROOT / "results" / "part1-smoke" / "preflight" / (
    "preflight.json"
)


def select_shard_work(
    records: Sequence[Mapping[str, Any]], *, shard_index: int, shard_count: int
) -> list[tuple[Mapping[str, Any], int]]:
    """Partition fixed questions by stable sample index, then expand ten runs."""

    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValueError("shard_index must identify one shard in shard_count")
    if len(records) != 500:
        raise ValueError("sharding requires the validated 500-question manifest")
    selected: list[tuple[Mapping[str, Any], int]] = []
    for sample_index, record in enumerate(records):
        if record.get("sample_index") != sample_index:
            raise ValueError("question manifest sample_index order is invalid")
        if record.get("subject") != FIXED_SUBJECTS[sample_index // 100]:
            raise ValueError("question manifest subject block order is invalid")
        if sample_index % shard_count == shard_index:
            selected.extend((record, run_id) for run_id in range(10))
    return selected


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"dependency lock is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_clean_recorded_commit(
    repository_root: Path, model_manifest: Mapping[str, Any]
) -> None:
    expected = model_manifest.get("final_production_git_commit")
    if model_manifest.get("production") is not True:
        expected = model_manifest.get("smoke_git_provenance", {}).get("base_commit")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected:
        raise ValueError("current Git commit differs from model-run provenance")
    tracked = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repository_root,
        check=False,
    )
    if tracked.returncode != 0:
        if tracked.returncode == 1:
            raise ValueError("shard execution requires a clean tracked worktree")
        raise RuntimeError("could not inspect tracked Git state")


def _resolve_raw_shards_root(
    *,
    repository_root: Path,
    model_manifest: Mapping[str, Any],
    execution_scope: str,
    output_root: Path | None,
) -> Path:
    production_root = (repository_root / "results" / "part1").resolve()
    if execution_scope == "production":
        expected_relative = Path("results/part1") / str(model_manifest["model_run_id"]) / (
            "raw_shards"
        )
        recorded = model_manifest.get("output_paths", {}).get("raw_shards")
        if recorded != expected_relative.as_posix():
            raise ValueError("production manifest raw_shards output root is not canonical")
        expected = (repository_root / expected_relative).resolve()
        if output_root is not None and Path(output_root).resolve() != expected:
            raise ValueError("requested output root differs from manifest raw_shards path")
        return expected
    if execution_scope != "phase3_smoke":
        raise ValueError(f"unsupported shard execution scope: {execution_scope}")
    default = (
        repository_root
        / "results"
        / "part1-smoke"
        / "phase3_smoke"
        / str(model_manifest["model_run_id"])
        / "raw_shards"
    ).resolve()
    selected = default if output_root is None else Path(output_root).resolve()
    smoke_base = (repository_root / "results" / "part1-smoke" / "phase3_smoke").resolve()
    if selected == production_root or production_root in selected.parents:
        raise ValueError("phase3 smoke output must remain separate from production paths")
    if selected != smoke_base and smoke_base not in selected.parents:
        raise ValueError("phase3 smoke output root must use the dedicated smoke root")
    return selected


def _owner(model_manifest: Mapping[str, Any], shard_id: str) -> LockMetadata:
    return LockMetadata(
        lock_id=uuid.uuid4().hex,
        study_id=str(model_manifest["study_id"]),
        model_run_id=str(model_manifest["model_run_id"]),
        shard_id=shard_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        acquired_at=_now(),
    )


def _expected_natural_keys(
    model_manifest: Mapping[str, Any],
    selected: Sequence[tuple[Mapping[str, Any], int]],
) -> set[tuple[str, str, str, int]]:
    return {
        (
            str(model_manifest["study_id"]),
            str(model_manifest["model_run_id"]),
            str(question["question_id"]),
            run_id,
        )
        for question, run_id in selected
    }


def _require_complete_shard_coverage(
    store: Part1ShardStore,
    *,
    model_manifest: Mapping[str, Any],
    selected: Sequence[tuple[Mapping[str, Any], int]],
) -> None:
    index = store.build_index()
    expected_natural = _expected_natural_keys(model_manifest, selected)
    actual_natural = set(index.natural_terminal_by_key)
    if actual_natural != expected_natural:
        raise RuntimeError("shard finalization coverage is incomplete or contains extra naturals")
    expected_checkpoints: set[tuple[str, str, str, int, str]] = set()
    for key, parent in index.natural_terminal_by_key.items():
        outcome = parent["natural_execution_outcome"]
        child_keys = {
            checkpoint_key
            for checkpoint_key in index.checkpoint_terminal_by_key
            if checkpoint_key[:4] == key
        }
        if outcome == "complete":
            if not parent.get("checkpoint_eligible"):
                raise RuntimeError("complete natural is not checkpoint eligible")
            expected_ids = tuple(parent.get("checkpoint_ids") or ())
            if expected_ids != CHECKPOINT_IDS:
                raise RuntimeError("complete natural does not request the fixed checkpoints")
            required = {(*key, checkpoint_id) for checkpoint_id in CHECKPOINT_IDS}
            expected_checkpoints.update(required)
            if child_keys != required:
                raise RuntimeError("complete natural is missing terminal checkpoint coverage")
        elif outcome == "terminal_infrastructure_failure":
            if parent.get("checkpoint_eligible") or child_keys:
                raise RuntimeError("failed natural has checkpoint work")
        else:
            raise RuntimeError("natural result has a nonterminal execution outcome")
    if set(index.checkpoint_terminal_by_key) != expected_checkpoints:
        raise RuntimeError("shard checkpoint coverage contains unexpected keys")
    if (
        index.hierarchy_errors
        or index.lifecycle_errors
        or index.missing_started_attempt_ids
        or index.inconsistent_completion_attempt_ids
        or index.orphaned_attempt_ids
        or index.terminalization_required
        or index.missing_completion_record_ids
    ):
        raise RuntimeError("shard lifecycle or hierarchy is invalid")
    report = store.validate_shard(
        artifact_kind="natural_shard", started_at=_now(), completed_at=_now()
    )
    if not report["is_valid"] or report["warning_count"]:
        raise RuntimeError("shard validation did not pass cleanly")


def _acquire_or_recover(
    shard_root: Path,
    *,
    owner: LockMetadata,
    model_run_manifest_hash: str,
) -> LockedShardSession:
    try:
        return LockedShardSession.acquire(
            shard_root,
            owner=owner,
            model_run_manifest_hash=model_run_manifest_hash,
        )
    except LockHeldError:
        claim = shard_root / ".writer-lock-recovery.claim"
        if claim.exists():
            return LockedShardSession.finish_pending_takeover(shard_root)
        return LockedShardSession.recover_stale(
            shard_root,
            owner=owner,
            model_run_manifest_hash=model_run_manifest_hash,
            event_timestamp=_now(),
            execution_context={
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
        )


def run_part1_shard(
    *,
    execution_scope: str,
    shard_index: int,
    shard_count: int,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_root: Path = DEFAULT_MANIFEST_ROOT,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    model_run_manifest_path: Path,
    output_root: Path | None = None,
    load_runtime: Callable[..., tuple[Any, Any]] = load_model_and_tokenizer,
    inspect_tokenizer: Callable[[Any], Mapping[str, Any]] = preflight_tokenizer_contract,
    execute_natural: Callable[..., Mapping[str, Any]] = _execute_natural,
    execute_checkpoint: Callable[..., Mapping[str, Any]] = _execute_checkpoint,
) -> dict[str, Any]:
    """Validate, resume, and finalize one deterministic manifest-bound shard."""

    if shard_count != 500:
        raise ValueError("Part 1 readiness execution requires shard_count=500")
    if execution_scope == "phase3_smoke" and shard_index != 0:
        raise ValueError("Phase 3 smoke is bounded to shard 0")
    repository_root = Path(repository_root)
    bundle = load_manifest_bundle(
        questions_path=Path(manifest_root) / "questions.jsonl",
        question_manifest_path=Path(manifest_root) / "questions.manifest.json",
        study_manifest_path=Path(manifest_root) / "study_manifest.json",
    )
    preflight = _load_json(Path(preflight_path))
    model_manifest = _load_json(Path(model_run_manifest_path))
    if model_manifest.get("execution_scope") != execution_scope:
        raise ValueError("model-run manifest execution scope differs from requested scope")
    if execution_scope == "production":
        if model_manifest.get("production") is not True or model_manifest.get(
            "schema_version"
        ) != "1.1.0":
            raise ValueError("production execution requires a production schema-1.1.0 manifest")
    elif model_manifest.get("production") is not False:
        raise ValueError("Phase 3 smoke requires a non-production manifest")
    validate_preflight_model_run_compatibility(
        preflight_report=preflight,
        model_manifest=model_manifest,
        study_manifest=bundle.study_manifest,
        question_manifest=bundle.question_manifest,
    )
    _require_clean_recorded_commit(repository_root, model_manifest)
    lock_hash = _sha256_regular_file(repository_root / "uv.lock")
    if preflight.get("environment_versions", {}).get("uv_lock_sha256") != lock_hash:
        raise ValueError("preflight dependency lock hash differs from current uv.lock")
    if execution_scope == "production" and model_manifest.get(
        "dependency_lock_sha256"
    ) != lock_hash:
        raise ValueError("production dependency lock hash differs from current uv.lock")
    raw_shards_root = _resolve_raw_shards_root(
        repository_root=repository_root,
        model_manifest=model_manifest,
        execution_scope=execution_scope,
        output_root=output_root,
    )
    selected = select_shard_work(
        bundle.records, shard_index=shard_index, shard_count=shard_count
    )
    if execution_scope == "phase3_smoke" and len(selected) != 10:
        raise ValueError("Phase 3 smoke budget is exactly one question by ten runs")
    raw_shards_root.mkdir(parents=True, exist_ok=True)
    estimated_questions = 500 if execution_scope == "production" else 1
    estimate = estimate_part1_storage(
        question_count=estimated_questions,
        natural_runs_per_question=10,
        checkpoints_per_natural=11,
    )
    free_space = assess_free_space(
        estimate, free_bytes=shutil.disk_usage(raw_shards_root).free
    )
    if free_space["status"] == "insufficient":
        raise OSError(free_space["warning"])

    shard_id = f"shard-{shard_index:03d}"
    shard_root = raw_shards_root / shard_id
    read_store = Part1ShardStore(
        shard_root,
        shard_id=shard_id,
        study_id=model_manifest["study_id"],
        model_run_id=model_manifest["model_run_id"],
        model_run_manifest_hash=model_manifest["model_run_manifest_hash"],
    )
    if read_store.finalization_path.exists():
        _require_complete_shard_coverage(
            read_store, model_manifest=model_manifest, selected=selected
        )
        return {
            "status": "already_finalized",
            "execution_scope": execution_scope,
            "model_run_id": model_manifest["model_run_id"],
            "shard_root": str(shard_root),
            "new_natural_results": 0,
            "new_checkpoint_results": 0,
            "storage_estimate": estimate,
            "free_space_assessment": free_space,
        }

    completed_natural = 0
    completed_checkpoints = 0
    owner = _owner(model_manifest, shard_id)
    with _acquire_or_recover(
        shard_root,
        owner=owner,
        model_run_manifest_hash=model_manifest["model_run_manifest_hash"],
    ) as session:
        try:
            _require_complete_shard_coverage(
                session.store, model_manifest=model_manifest, selected=selected
            )
        except RuntimeError:
            pass
        else:
            session.store.finalize()
            return {
                "status": "completed",
                "execution_scope": execution_scope,
                "model_run_id": model_manifest["model_run_id"],
                "shard_root": str(shard_root),
                "new_natural_results": 0,
                "new_checkpoint_results": 0,
                "storage_estimate": estimate,
                "free_space_assessment": free_space,
            }

        model, tokenizer = load_runtime(
            model_revision=model_manifest["model_revision"],
            tokenizer_revision=model_manifest["tokenizer_revision"],
        )
        token_contract = preflight["token_contract"]
        if dict(inspect_tokenizer(tokenizer)) != token_contract:
            raise ValueError("runtime tokenizer contract differs from GPU preflight")

        for question, run_id in selected:
            seed = derive_generation_seed(
                base_seed=model_manifest["base_generation_seed"],
                canonical_model_identity=model_manifest["canonical_model_identity"],
                question_id=question["question_id"],
                run_id=run_id,
            )
            natural_work = WorkSpec.natural(
                model_manifest["study_id"],
                model_manifest["model_run_id"],
                model_manifest["model_run_manifest_hash"],
                question["question_id"],
                run_id,
                seed=seed,
            )
            identity = {
                "study_id": model_manifest["study_id"],
                "model_run_id": model_manifest["model_run_id"],
                "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
                "question_manifest_hash": model_manifest["question_manifest_hash"],
                "question_id": question["question_id"],
                "sample_index": question["sample_index"],
                "subject": question["subject"],
                "gold_letter": question["gold_letter"],
            }

            def execute_natural_attempt(attempt_number: int) -> Mapping[str, Any]:
                return execute_natural(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    run_id=run_id,
                    seed=seed,
                    attempt_number=attempt_number,
                    model_manifest=model_manifest,
                    token_contract=token_contract,
                )

            def build_natural_failure(
                attempt_number: int,
                category: str,
                reference: str,
                details: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                return build_natural_infrastructure_failure_result(
                    identity=identity,
                    run_id=run_id,
                    generation_seed=seed,
                    terminal_attempt_number=attempt_number,
                    prompt_hash=model_manifest["prompt_hash"],
                    failure_category=category,
                    infrastructure_failure_reference=reference,
                    error_details=details,
                )

            natural_status, natural_published = _run_work_lifecycle(
                session,
                work=natural_work,
                execute_attempt=execute_natural_attempt,
                build_terminal_failure=build_natural_failure,
            )
            if natural_published:
                completed_natural += 1
            if natural_status != "completed":
                continue

            parent_key = (
                model_manifest["study_id"],
                model_manifest["model_run_id"],
                question["question_id"],
                run_id,
            )
            parent = session.store.build_index().natural_terminal_by_key[parent_key]
            plans = build_checkpoint_probe_plans(
                parent,
                inducer_token_ids=token_contract["inducer_token_ids"],
                inducer_version=model_manifest["inducer_version"],
            )
            for plan in plans:
                checkpoint_work = WorkSpec.checkpoint(
                    model_manifest["study_id"],
                    model_manifest["model_run_id"],
                    model_manifest["model_run_manifest_hash"],
                    question["question_id"],
                    run_id,
                    plan.checkpoint_id,
                    seed=seed,
                )

                def execute_checkpoint_attempt(
                    attempt_number: int,
                ) -> Mapping[str, Any]:
                    if plan.is_alias:
                        owner_id = plan.alias_metadata["owner_checkpoint_id"]
                        owner_record = session.store.build_index().checkpoint_terminal_by_key.get(
                            (*parent_key, owner_id)
                        )
                        if owner_record is None:
                            raise RuntimeError("alias owner checkpoint is not durable")
                        return build_alias_checkpoint_terminal_result(
                            parent=parent,
                            owner_record=owner_record,
                            alias_plan=plan,
                            terminal_attempt_number=attempt_number,
                        )
                    return execute_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        parent=parent,
                        plan=plan,
                        token_contract=token_contract,
                        gold_letter=question["gold_letter"],
                        attempt_number=attempt_number,
                    )

                def build_checkpoint_failure(
                    attempt_number: int,
                    category: str,
                    reference: str,
                    details: Mapping[str, Any],
                ) -> Mapping[str, Any]:
                    return build_checkpoint_infrastructure_failure_result(
                        parent=parent,
                        plan=plan,
                        terminal_attempt_number=attempt_number,
                        failure_category=category,
                        infrastructure_failure_reference=reference,
                        error_details=details,
                        inducer_text=token_contract.get(
                            "inducer_text", "</think>\nAnswer:"
                        ),
                    )

                _checkpoint_status, checkpoint_published = _run_work_lifecycle(
                    session,
                    work=checkpoint_work,
                    execute_attempt=execute_checkpoint_attempt,
                    build_terminal_failure=build_checkpoint_failure,
                )
                if checkpoint_published:
                    completed_checkpoints += 1

        _require_complete_shard_coverage(
            session.store, model_manifest=model_manifest, selected=selected
        )
        session.store.finalize()

    return {
        "status": "completed",
        "execution_scope": execution_scope,
        "model_run_id": model_manifest["model_run_id"],
        "shard_root": str(shard_root),
        "new_natural_results": completed_natural,
        "new_checkpoint_results": completed_checkpoints,
        "storage_estimate": estimate,
        "free_space_assessment": free_space,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-scope", choices=("production", "phase3_smoke"), required=True
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=500)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_part1_shard(
            execution_scope=args.execution_scope,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            repository_root=args.repository_root,
            manifest_root=args.manifest_root,
            preflight_path=args.preflight,
            model_run_manifest_path=args.model_run_manifest,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
