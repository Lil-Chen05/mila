#!/usr/bin/env python3
"""Publish the one authorized preserved Part 1 production merge stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from part1_merge_stage_recovery import recover_authorized_merge_stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        target, sidecar_path, sidecar = recover_authorized_merge_stage(
            repository_root=args.repository_root,
            model_run_manifest_path=args.model_run_manifest,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        }, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "published_preserved_stage",
        "merged_directory": target.as_posix(),
        "merge_stage_recovery_sidecar": sidecar_path.as_posix(),
        "merge_stage_recovery_id": sidecar["merge_stage_recovery_id"],
        "merge_id": sidecar["merge_manifest"]["merge_id"],
        "merge_manifest_hash": sidecar["merge_manifest"]["merge_manifest_hash"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
