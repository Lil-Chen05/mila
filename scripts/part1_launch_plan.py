#!/usr/bin/env python3
"""Validate and print the exact unsubmitted Part 1 production launch plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from part1_manifests import load_manifest_bundle
from part1_runtime import validate_manifest_compatibility
from part1_storage_estimate import estimate_part1_storage
from run_part1_shard import _require_clean_recorded_commit, _sha256_regular_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INITIAL_CONCURRENCY = 16
SUBMISSION_COMMAND_TEMPLATE = (
    "sbatch --export=ALL,MODEL_RUN_ID={model_run_id} "
    f"--array=0-499%{INITIAL_CONCURRENCY} jobs/part1_generate_array.sh"
)


def build_launch_plan(
    *, model_manifest: Mapping[str, Any], repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    repository_root = Path(repository_root)
    bundle = load_manifest_bundle(
        questions_path=repository_root / "manifests/part1/questions.jsonl",
        question_manifest_path=repository_root / "manifests/part1/questions.manifest.json",
        study_manifest_path=repository_root / "manifests/part1/study_manifest.json",
    )
    validate_manifest_compatibility(bundle.study_manifest, model_manifest)
    if (
        model_manifest.get("production") is not True
        or model_manifest.get("execution_scope") != "production"
        or model_manifest.get("schema_version") != "1.1.0"
    ):
        raise ValueError("launch plan requires a production schema-1.1.0 manifest")
    _require_clean_recorded_commit(repository_root, model_manifest)
    lock_hash = _sha256_regular_file(repository_root / "uv.lock")
    if model_manifest.get("dependency_lock_sha256") != lock_hash:
        raise ValueError("production dependency lock hash differs from current uv.lock")
    expected_raw = (
        Path("results/part1") / str(model_manifest["model_run_id"]) / "raw_shards"
    ).as_posix()
    if model_manifest.get("output_paths", {}).get("raw_shards") != expected_raw:
        raise ValueError("production raw_shards path is not canonical")
    return {
        "is_valid": True,
        "command_executed": False,
        "model_run_id": model_manifest["model_run_id"],
        "final_production_git_commit": model_manifest["final_production_git_commit"],
        "dependency_lock_sha256": lock_hash,
        "workload": {
            "questions": 500,
            "natural_runs": 5_000,
            "checkpoint_keys": 55_000,
            "shards": 500,
            "questions_per_shard": 1,
            "natural_runs_per_task": 10,
        },
        "resources": {
            "gpu": "l40s:1",
            "wall_time": "12:00:00",
            "initial_concurrency": INITIAL_CONCURRENCY,
        },
        "storage_estimates": {
            "expected_2048_tokens": estimate_part1_storage(
                expected_generated_tokens=2048
            ),
            "cap_8192_tokens": estimate_part1_storage(expected_generated_tokens=8192),
        },
        "submission_command": SUBMISSION_COMMAND_TEMPLATE.format(
            model_run_id=model_manifest["model_run_id"]
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_launch_plan(
            model_manifest=_load_json(args.model_run_manifest),
            repository_root=args.repository_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"is_valid": False, "error_type": type(exc).__name__, "error": str(exc)},
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
