#!/usr/bin/env python3
"""Submit the single explicitly authorized production analysis job."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from part1_direct_analysis_recovery import direct_analysis_recovery_id
from part1_merge import _load_regular_json
from part1_merge_stage_recovery import (
    AUTHORIZED_GENERATION_GIT_COMMIT,
    AUTHORIZED_MODEL_RUN_ID,
    validate_merge_stage_recovery,
)
from submit_part1_production_chain import (
    _atomic_replace_json,
    _exclusive_create_json,
    _parse_job_id,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


def submit_direct_analysis(*, production_repository_root: Path) -> dict[str, Any]:
    production_root = production_repository_root.resolve()
    run_root = production_root / "results" / "part1" / AUTHORIZED_MODEL_RUN_ID
    model, _ = _load_regular_json(
        run_root / "model_run_manifest.json", label="production model-run manifest"
    )
    if (
        model.get("model_run_id") != AUTHORIZED_MODEL_RUN_ID
        or model.get("final_production_git_commit") != AUTHORIZED_GENERATION_GIT_COMMIT
    ):
        raise ValueError("production manifest differs from authorized direct analysis")
    sidecar_path = run_root / "validation" / "merge_stage_recovery.json"
    sidecar, sidecar_bytes = _load_regular_json(
        sidecar_path, label="merge-stage recovery sidecar"
    )
    validate_merge_stage_recovery(sidecar)
    execution_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise ValueError("direct-analysis recovery checkout must be tracked-clean")
    export = (
        f"ALL,MODEL_RUN_ID={AUTHORIZED_MODEL_RUN_ID},"
        f"PRODUCTION_REPOSITORY_ROOT={production_root.as_posix()}"
    )
    command = [
        "sbatch", "--parsable",
        f"--job-name=part1-direct-analysis-{AUTHORIZED_MODEL_RUN_ID[:12]}",
        f"--comment=part1:{AUTHORIZED_MODEL_RUN_ID}:direct_analysis",
        f"--export={export}",
        "jobs/part1_direct_analysis_recovery.sh",
    ]
    receipt_path = run_root / "validation" / "direct_analysis_recovery_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "part1-direct-analysis-recovery-receipt-v1",
        "direct_analysis_recovery_id": "",
        "model_run_id": AUTHORIZED_MODEL_RUN_ID,
        "model_run_manifest_hash": model["model_run_manifest_hash"],
        "merge_stage_recovery_id": sidecar["merge_stage_recovery_id"],
        "merge_stage_recovery_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "merge_stage_recovery_byte_size": len(sidecar_bytes),
        "analysis_execution_commit": execution_commit,
        "bootstrap_replicates": 5000,
        "no_preflight": True,
        "command": command,
        "status": "submitting",
        "analysis_job_id": None,
        "submitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    receipt["direct_analysis_recovery_id"] = direct_analysis_recovery_id(receipt)
    _exclusive_create_json(receipt_path, receipt)
    try:
        completed = subprocess.run(
            command, cwd=CODE_ROOT, check=True, capture_output=True, text=True
        )
        receipt["analysis_job_id"] = _parse_job_id(completed.stdout)
        receipt["status"] = "submitted"
        _atomic_replace_json(receipt_path, receipt)
    except Exception:
        receipt["status"] = "submission_failed"
        _atomic_replace_json(receipt_path, receipt)
        raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-repository-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = submit_direct_analysis(
            production_repository_root=args.production_repository_root
        )
    except Exception as exc:
        print(json.dumps({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        }, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
