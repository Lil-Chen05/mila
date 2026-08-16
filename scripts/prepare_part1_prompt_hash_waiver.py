#!/usr/bin/env python3
"""Prepare the authorized, content-addressed production prompt-hash waiver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from part1_contract import validate_fixed_model_requested_contract, validate_instance
from part1_merge import _load_regular_json
from part1_prompt_hash_waiver import (
    build_prompt_hash_waiver,
    canonical_waiver_bytes,
    require_production_checkout_generation_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.repository_root.resolve()
        manifest_path = args.model_run_manifest
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        model, _ = _load_regular_json(manifest_path, label="production model-run manifest")
        validate_instance("model_run_manifest", model)
        validate_fixed_model_requested_contract(model)
        expected_manifest = root / "results" / "part1" / model["model_run_id"] / "model_run_manifest.json"
        if manifest_path.resolve() != expected_manifest:
            raise ValueError("production model-run manifest path is not canonical")
        require_production_checkout_generation_state(
            root,
            expected_generation_commit=model["final_production_git_commit"],
        )
        recovery_code_root = REPOSITORY_ROOT
        recovery_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=recovery_code_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=recovery_code_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip():
            raise ValueError("tracked worktree must be clean before preparing waiver")
        report_path = expected_manifest.parent / "validation" / "coverage_report.json"
        report, report_bytes = _load_regular_json(report_path, label="failed coverage report")
        report_relative = report_path.relative_to(root).as_posix()
        waiver = build_prompt_hash_waiver(
            report=report,
            report_bytes=report_bytes,
            report_relative_path=report_relative,
            model_manifest=model,
            recovery_git_commit=recovery_commit,
        )
        data = canonical_waiver_bytes(waiver)
        target = report_path.with_name("prompt_hash_waiver.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != data:
                raise ValueError("existing prompt-hash waiver is incompatible")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".prompt_hash_waiver.json.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, target)
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
        print(json.dumps({"status": "prepared", "waiver_id": waiver["waiver_id"], "path": str(target)}, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
