"""Print a read-only bounded-smoke coverage report as compact JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from part1_smoke_coverage import build_smoke_coverage_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_smoke_coverage_report(repository_root=args.repository_root, model_run_manifest_path=args.model_run_manifest, shard_root=args.shard_root)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "mutation_performed": False}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["is_valid"] and report["coverage_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
