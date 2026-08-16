#!/usr/bin/env python3
"""Submit the authorized waiver prepare -> merge -> final-analysis recovery chain."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from part1_merge import _load_regular_json
from submit_part1_production_chain import (
    _atomic_replace_json,
    _exclusive_create_json,
    _parse_job_id,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


def submit_recovery(*, production_repository_root: Path, model_run_id: str) -> dict[str, Any]:
    production_root = production_repository_root.resolve()
    manifest_path = production_root / "results" / "part1" / model_run_id / "model_run_manifest.json"
    model, _ = _load_regular_json(manifest_path, label="production model-run manifest")
    if model.get("model_run_id") != model_run_id:
        raise ValueError("requested model-run ID differs from production manifest")
    recovery_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise ValueError("recovery code worktree must be clean")
    receipt_path = manifest_path.parent / "validation" / "prompt_hash_waiver_recovery_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "part1-prompt-hash-waiver-recovery-receipt-v1",
        "status": "submitting",
        "submitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_run_id": model_run_id,
        "generation_git_commit": model["final_production_git_commit"],
        "recovery_git_commit": recovery_commit,
        "production_repository_root": production_root.as_posix(),
        "bootstrap_replicates": 5000,
        "jobs": {},
    }
    _exclusive_create_json(receipt_path, receipt)
    export = (
        f"ALL,MODEL_RUN_ID={model_run_id},"
        f"PRODUCTION_REPOSITORY_ROOT={production_root.as_posix()}"
    )
    stages = (
        (
            "prepare",
            ["sbatch", "--parsable", f"--export={export}", "jobs/part1_prepare_prompt_hash_waiver.sh"],
        ),
        (
            "merge",
            lambda jobs: [
                "sbatch", "--parsable", f"--dependency=afterok:{jobs['prepare']}",
                f"--export={export}", "jobs/part1_merge_prompt_hash_waiver.sh",
            ],
        ),
        (
            "analysis",
            lambda jobs: [
                "sbatch", "--parsable", f"--dependency=afterok:{jobs['merge']}",
                f"--export={export},BOOTSTRAP_REPLICATES=5000",
                "jobs/part1_analyze_prompt_hash_waiver.sh",
            ],
        ),
    )
    for stage, arguments_or_factory in stages:
        arguments = (
            arguments_or_factory(receipt["jobs"])
            if callable(arguments_or_factory)
            else arguments_or_factory
        )
        arguments[2:2] = [
            f"--job-name=part1-waiver-{stage}-{model_run_id[:12]}",
            f"--comment=part1:{model_run_id}:prompt_hash_waiver:{stage}",
        ]
        receipt["in_flight"] = {"stage": stage, "arguments": arguments}
        receipt.setdefault("commands", {})[stage] = list(arguments)
        _atomic_replace_json(receipt_path, receipt)
        try:
            completed = subprocess.run(
                arguments, cwd=CODE_ROOT, check=True, capture_output=True, text=True
            )
            receipt["jobs"][stage] = _parse_job_id(completed.stdout)
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
    parser.add_argument("--model-run-id", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = submit_recovery(
            production_repository_root=args.production_repository_root,
            model_run_id=args.model_run_id,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
