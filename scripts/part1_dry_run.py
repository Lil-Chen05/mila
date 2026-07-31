#!/usr/bin/env python3
"""Validate Phase 1 templates and persisted metadata without model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from part1_runtime import WorkSpec, run_dry_run


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_work_specs(path: Path | None) -> list[WorkSpec]:
    if path is None:
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [WorkSpec.from_mapping(item) for item in value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--study-manifest", type=Path)
    parser.add_argument("--model-run-manifest", type=Path)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--work-specs", type=Path)
    parser.add_argument("--retry-request", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_dry_run(
            mode=args.mode,
            persistent_root=args.persistent_root,
            allow_root_override=args.persistent_root is not None,
            study_manifest=_load_json(args.study_manifest),
            model_run_manifest=_load_json(args.model_run_manifest),
            shard_root=args.shard_root,
            work_items=_load_work_specs(args.work_specs),
            retry_request=_load_json(args.retry_request),
        )
    except Exception as exc:
        report = {
            "is_valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "mutation_performed": False,
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["is_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
