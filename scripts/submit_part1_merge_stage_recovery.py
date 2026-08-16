#!/usr/bin/env python3
"""Submit exact preserved-stage finalize -> final-analysis recovery jobs.

The sole stage is `.merged.stage-ri97qy41`; its remaining evidence is imported
from the fail-closed recovery contract rather than repeated here.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from part1_merge import _load_regular_json
from part1_merge_stage_recovery import (
    AUTHORIZED_GENERATION_GIT_COMMIT,
    AUTHORIZED_MERGE_ID,
    AUTHORIZED_MERGE_MANIFEST_HASH,
    AUTHORIZED_MODEL_RUN_ID,
    AUTHORIZED_ORIGINAL_RECOVERY_COMMIT,
    AUTHORIZED_STAGE_BASENAME,
    AUTHORIZED_STAGE_MANIFEST_BYTE_SIZE,
    AUTHORIZED_STAGE_MANIFEST_SHA256,
)
from submit_part1_production_chain import (
    _atomic_replace_json,
    _exclusive_create_json,
    _parse_job_id,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
FAILED_MERGE_JOB = {"job_id": "10383206", "state": "FAILED", "exit_code": "2:0", "elapsed": "03:54:36"}
CANCELLED_ANALYSIS_JOB = {"job_id": "10383207", "state": "CANCELLED", "exit_code": "0:0"}


def submit_recovery(*, production_repository_root: Path) -> dict[str, Any]:
    production_root = production_repository_root.resolve()
    model_run_id = AUTHORIZED_MODEL_RUN_ID
    manifest_path = production_root / "results" / "part1" / model_run_id / "model_run_manifest.json"
    model, _ = _load_regular_json(manifest_path, label="production model-run manifest")
    if (
        model.get("model_run_id") != model_run_id
        or model.get("final_production_git_commit") != AUTHORIZED_GENERATION_GIT_COMMIT
    ):
        raise ValueError("production manifest differs from authorized recovery target")
    recovery_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise ValueError("recovery checkout must be tracked-clean")
    queue = subprocess.run(
        ["squeue", "--noheader", "--user", os.environ["USER"], "--format=%i|%T|%k"],
        check=True, capture_output=True, text=True,
    ).stdout
    if any(model_run_id in line for line in queue.splitlines()):
        raise ValueError("a job for the authorized model run is already queued or running")

    receipt_path = manifest_path.parent / "validation" / "merge_stage_recovery_submission_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "part1-merge-stage-recovery-submission-receipt-v1",
        "status": "submitting",
        "submitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_run_id": model_run_id,
        "generation_git_commit": AUTHORIZED_GENERATION_GIT_COMMIT,
        "original_merge_recovery_commit": AUTHORIZED_ORIGINAL_RECOVERY_COMMIT,
        "publication_recovery_commit": recovery_commit,
        "production_repository_root": production_root.as_posix(),
        "bootstrap_replicates": 5000,
        "historical_jobs": {
            "failed_merge": FAILED_MERGE_JOB,
            "cancelled_analysis": CANCELLED_ANALYSIS_JOB,
        },
        "preserved_stage": {
            "basename": AUTHORIZED_STAGE_BASENAME,
            "manifest_sha256": AUTHORIZED_STAGE_MANIFEST_SHA256,
            "manifest_byte_size": AUTHORIZED_STAGE_MANIFEST_BYTE_SIZE,
            "merge_id": AUTHORIZED_MERGE_ID,
            "merge_manifest_hash": AUTHORIZED_MERGE_MANIFEST_HASH,
        },
        "jobs": {},
        "commands": {},
    }
    _exclusive_create_json(receipt_path, receipt)
    export = f"ALL,MODEL_RUN_ID={model_run_id},PRODUCTION_REPOSITORY_ROOT={production_root.as_posix()}"
    jobs: dict[str, str] = receipt["jobs"]
    stages = (
        ("finalize", ["sbatch", "--parsable", f"--export={export}", "jobs/part1_recover_merge_stage.sh"]),
        ("analysis", lambda: [
            "sbatch", "--parsable", f"--dependency=afterok:{jobs['finalize']}",
            f"--export={export},BOOTSTRAP_REPLICATES=5000",
            "jobs/part1_analyze_merge_stage_recovery.sh",
        ]),
    )
    for stage, command_or_factory in stages:
        command = command_or_factory() if callable(command_or_factory) else command_or_factory
        command[2:2] = [
            f"--job-name=part1-stage-{stage}-{model_run_id[:12]}",
            f"--comment=part1:{model_run_id}:merge_stage_recovery:{stage}",
        ]
        receipt["commands"][stage] = list(command)
        receipt["in_flight"] = {"stage": stage, "arguments": list(command)}
        _atomic_replace_json(receipt_path, receipt)
        try:
            completed = subprocess.run(
                command, cwd=CODE_ROOT, check=True, capture_output=True, text=True
            )
            jobs[stage] = _parse_job_id(completed.stdout)
            receipt.pop("in_flight", None)
            _atomic_replace_json(receipt_path, receipt)
        except Exception:
            receipt["status"] = "submission_failed"
            receipt["failed_stage"] = stage
            _atomic_replace_json(receipt_path, receipt)
            raise
    receipt["status"] = "submitted"
    _atomic_replace_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-repository-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = submit_recovery(production_repository_root=args.production_repository_root)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
