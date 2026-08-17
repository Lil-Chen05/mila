#!/usr/bin/env python3
"""Production-only CLI for the manifest-driven Part 1 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from part1_analysis import analyze_production


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _JSONErrorParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def build_parser(*, json_errors: bool = False) -> argparse.ArgumentParser:
    parser_class = _JSONErrorParser if json_errors else argparse.ArgumentParser
    parser = parser_class(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    parser.add_argument("--prompt-hash-waiver", type=Path)
    parser.add_argument("--merge-stage-recovery", type=Path)
    parser.add_argument("--direct-analysis-recovery-receipt", type=Path)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        choices=(1000, 5000),
        default=5000,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser(json_errors=True).parse_args(argv)
        directory, manifest = analyze_production(
            repository_root=arguments.repository_root,
            model_run_manifest_path=arguments.model_run_manifest,
            bootstrap_replicates=arguments.bootstrap_replicates,
            prompt_hash_waiver_path=arguments.prompt_hash_waiver,
            merge_stage_recovery_path=arguments.merge_stage_recovery,
            direct_analysis_recovery_receipt_path=(
                arguments.direct_analysis_recovery_receipt
            ),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "published",
                "mode": manifest["bootstrap_mode"],
                "analysis_directory": directory.as_posix(),
                "analysis_id": manifest["analysis_id"],
                "analysis_manifest_hash": manifest["analysis_manifest_hash"],
                "merge_id": manifest["merge_id"],
                "merge_manifest_hash": manifest["merge_manifest_hash"],
                "coverage_report_id": manifest["coverage_report_id"],
                "bootstrap_replicates": manifest["bootstrap_replicates"],
                "paper_analysis_ready": manifest["paper_analysis_ready"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
