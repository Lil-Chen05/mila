#!/usr/bin/env python3
"""Submit and durably record the fail-closed Part 1 production job chain."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

from part1_launch_plan import build_launch_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_JOB_ID = re.compile(r"^[0-9]+$")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"submission receipt parent is not a regular directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_create_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"submission receipt parent is not a regular directory: {path.parent}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"submission receipt already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _parse_job_id(output: str) -> str:
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9._-]+)?\n?", output)
    if match is None:
        raise RuntimeError(f"sbatch returned an invalid job ID: {output!r}")
    return match.group(1)


def submit_production_chain(
    *,
    model_manifest: Mapping[str, Any],
    acceptance_job_id: str,
    gate_job_id: str,
    bootstrap_receipt_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    submitted_at: str | None = None,
) -> dict[str, Any]:
    """Submit generation then dependent CPU jobs, persisting every job ID."""

    repository_root = Path(repository_root)
    bootstrap_receipt_path = Path(bootstrap_receipt_path)
    if not bootstrap_receipt_path.is_absolute():
        bootstrap_receipt_path = repository_root / bootstrap_receipt_path
    expected_bootstrap_path = (
        repository_root
        / "results/part1-submission"
        / str(model_manifest["final_production_git_commit"])
        / "bootstrap_receipt.json"
    )
    if bootstrap_receipt_path != expected_bootstrap_path:
        raise RuntimeError("submission bootstrap receipt path is not canonical")
    if bootstrap_receipt_path.is_symlink() or not bootstrap_receipt_path.is_file():
        raise RuntimeError("submission bootstrap receipt is not a regular file")
    bootstrap = _load_json(bootstrap_receipt_path)
    if (
        bootstrap.get("schema_version") != "part1-submission-bootstrap-v2"
        or bootstrap.get("acceptance_mode") != "focused_readiness_v1"
        or bootstrap.get("status") != "submitted"
        or bootstrap.get("jobs", {}).get("acceptance") != acceptance_job_id
        or bootstrap.get("jobs", {}).get("production_gate") != gate_job_id
        or bootstrap.get("gate_dependency") != f"afterok:{acceptance_job_id}"
        or bootstrap.get("final_production_git_commit")
        != model_manifest.get("final_production_git_commit")
    ):
        raise RuntimeError("submission bootstrap receipt is incompatible")
    plan = build_launch_plan(
        model_manifest=model_manifest, repository_root=repository_root
    )
    model_run_id = str(model_manifest["model_run_id"])
    for label, value in (
        ("acceptance_job_id", acceptance_job_id),
        ("gate_job_id", gate_job_id),
    ):
        if _JOB_ID.fullmatch(value) is None:
            raise ValueError(f"{label} must be a numeric SLURM job ID")
    if (
        plan.get("is_valid") is not True
        or plan.get("command_executed") is not False
        or plan.get("model_run_id") != model_run_id
        or plan.get("resources", {}).get("initial_concurrency") != 16
    ):
        raise RuntimeError("production launch plan is not the approved unrun %16 plan")

    receipt_path = (
        repository_root
        / "results/part1"
        / model_run_id
        / "submission_receipt.json"
    )
    if not receipt_path.parent.is_dir():
        raise RuntimeError(
            f"production model-run directory does not exist: {receipt_path.parent}"
        )

    receipt: dict[str, Any] = {
        "schema_version": "part1-submission-receipt-v1",
        "status": "submitting",
        "submitted_at": submitted_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_run_id": model_run_id,
        "final_production_git_commit": model_manifest["final_production_git_commit"],
        "array": "0-499%16",
        "bootstrap_replicates": 5_000,
        "acceptance_mode": "focused_readiness_v1",
        "jobs": {
            "acceptance": acceptance_job_id,
            "production_gate": gate_job_id,
        },
    }
    _exclusive_create_json(receipt_path, receipt)

    stages = (
        (
            "generation",
            lambda jobs: [
                "sbatch",
                "--parsable",
                f"--export=ALL,MODEL_RUN_ID={model_run_id}",
                "--array=0-499%16",
                "jobs/part1_generate_array.sh",
            ],
        ),
        (
            "validation",
            lambda jobs: [
                "sbatch",
                "--parsable",
                f"--dependency=afterany:{jobs['generation']}",
                f"--export=ALL,MODEL_RUN_ID={model_run_id}",
                "jobs/part1_validate.sh",
            ],
        ),
        (
            "merge",
            lambda jobs: [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{jobs['validation']}",
                f"--export=ALL,MODEL_RUN_ID={model_run_id}",
                "jobs/part1_merge.sh",
            ],
        ),
        (
            "analysis",
            lambda jobs: [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{jobs['merge']}",
                f"--export=ALL,MODEL_RUN_ID={model_run_id},BOOTSTRAP_REPLICATES=5000",
                "jobs/part1_analyze.sh",
            ],
        ),
    )

    for stage, arguments_for in stages:
        short_id = model_run_id[:12]
        job_name = f"part1-{stage}-{short_id}"
        comment = f"part1:{model_run_id}:{stage}"
        arguments = arguments_for(receipt["jobs"])
        arguments[2:2] = [f"--job-name={job_name}", f"--comment={comment}"]
        receipt["in_flight"] = {
            "stage": stage,
            "job_name": job_name,
            "comment": comment,
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-job-id", required=True)
    parser.add_argument("--gate-job-id", required=True)
    parser.add_argument("--bootstrap-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = submit_production_chain(
            model_manifest=_load_json(args.model_run_manifest),
            repository_root=args.repository_root,
            acceptance_job_id=args.acceptance_job_id,
            gate_job_id=args.gate_job_id,
            bootstrap_receipt_path=args.bootstrap_receipt,
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
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
