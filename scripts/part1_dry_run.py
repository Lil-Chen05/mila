#!/usr/bin/env python3
"""Validate Phase 1 templates and persisted metadata without model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from part1_runtime import run_dry_run


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--study-manifest", type=Path)
    parser.add_argument("--model-run-manifest", type=Path)
    parser.add_argument("--shard-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_dry_run(
        mode=args.mode,
        persistent_root=args.persistent_root,
        study_manifest=_load_json(args.study_manifest),
        model_run_manifest=_load_json(args.model_run_manifest),
        shard_root=args.shard_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
