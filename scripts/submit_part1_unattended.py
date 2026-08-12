#!/usr/bin/env python3
"""Submit focused CPU readiness and its fail-closed afterok production gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

from create_part1_model_run_manifest import _ensure_regular_directory_chain, _git_state
from submit_part1_production_chain import (
    _atomic_replace_json,
    _exclusive_create_json,
    _parse_job_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def submit_unattended(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    submitted_at: str | None = None,
) -> dict[str, Any]:
    repository_root = Path(repository_root)
    commit, blocking_paths, unrelated_paths = _git_state(repository_root)
    if blocking_paths:
        raise RuntimeError(
            "unattended submission requires a clean tracked worktree and no "
            "execution-relevant untracked paths: " + ", ".join(blocking_paths)
        )

    receipt_relative = (
        Path("results/part1-submission") / commit / "bootstrap_receipt.json"
    )
    receipt_parent = _ensure_regular_directory_chain(
        repository_root, receipt_relative.parent
    )
    receipt_path = receipt_parent / receipt_relative.name
    receipt: dict[str, Any] = {
        "schema_version": "part1-submission-bootstrap-v2",
        "acceptance_mode": "focused_readiness_v1",
        "status": "submitting",
        "submitted_at": submitted_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "final_production_git_commit": commit,
        "gate_dependency": None,
        "jobs": {},
        "unrelated_untracked_paths": list(unrelated_paths),
    }
    _exclusive_create_json(receipt_path, receipt)

    stages = (
        (
            "acceptance",
            [
                "sbatch",
                "--parsable",
                f"--job-name=part1-launch-readiness-{commit[:12]}",
                f"--comment=part1:{commit}:focused_readiness",
                "jobs/part1_launch_readiness.sh",
            ],
        ),
        ("production_gate", None),
    )
    for stage, static_arguments in stages:
        if stage == "production_gate":
            acceptance_job_id = receipt["jobs"]["acceptance"]
            receipt["gate_dependency"] = f"afterok:{acceptance_job_id}"
            arguments = [
                "sbatch",
                "--parsable",
                f"--job-name=part1-production-gate-{commit[:12]}",
                f"--comment=part1:{commit}:production_gate",
                f"--dependency=afterok:{acceptance_job_id}",
                "--export=ALL,"
                f"ACCEPTANCE_JOB_ID={acceptance_job_id},"
                f"BOOTSTRAP_RECEIPT={receipt_relative.as_posix()}",
                "jobs/part1_production_gate.sh",
            ]
        else:
            assert static_arguments is not None
            arguments = static_arguments
        receipt["in_flight"] = {
            "stage": stage,
            "job_name": next(
                value.split("=", 1)[1]
                for value in arguments
                if value.startswith("--job-name=")
            ),
            "arguments": arguments,
        }
        _atomic_replace_json(receipt_path, receipt)
        try:
            completed = run_command(
                arguments,
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
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
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    try:
        receipt = submit_unattended(repository_root=args.repository_root)
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
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
