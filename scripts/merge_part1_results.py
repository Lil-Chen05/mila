"""Validate and atomically publish deterministic Part 1 merged Parquet tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from part1_merge import (
    load_validated_merge_inputs,
    publish_merge,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repository_root = args.repository_root.resolve()
        inputs = load_validated_merge_inputs(
            repository_root=repository_root,
            model_run_manifest_path=args.model_run_manifest,
            coverage_report_path=args.coverage_report,
        )
        publication = publish_merge(inputs, return_manifest=True)
        if not isinstance(publication, tuple):
            raise RuntimeError("manifest-returning publication returned no manifest")
        target, manifest = publication
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
    paper_ready = inputs.coverage_report["paper_analysis_ready"]
    print(
        json.dumps(
            {
                "status": "merged" if paper_ready else "merged_diagnostic",
                "merged_directory": str(target),
                "merge_id": manifest["merge_id"],
                "merge_manifest_hash": manifest["merge_manifest_hash"],
                "coverage_report_id": inputs.coverage_report["validation_report_id"],
                "paper_analysis_ready": paper_ready,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
