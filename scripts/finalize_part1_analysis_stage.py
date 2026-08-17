#!/usr/bin/env python3
"""Validate and publish a preserved Part 1 analysis stage without recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from part1_analysis_stage_recovery import finalize_preserved_analysis_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--expected-model-run-id", required=True)
    parser.add_argument("--expected-analysis-id", required=True)
    parser.add_argument("--expected-bootstrap-replicates", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        manifest = finalize_preserved_analysis_stage(
            stage=arguments.stage,
            target_name=arguments.target_name,
            expected_model_run_id=arguments.expected_model_run_id,
            expected_analysis_id=arguments.expected_analysis_id,
            expected_bootstrap_replicates=arguments.expected_bootstrap_replicates,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "status": "error",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    target = Path(arguments.stage).resolve().parent / arguments.target_name
    print(
        json.dumps(
            {
                "analysis_id": manifest["analysis_id"],
                "analysis_manifest_hash": manifest["analysis_manifest_hash"],
                "bootstrap_replicates": manifest["bootstrap_replicates"],
                "paper_analysis_ready": manifest["paper_analysis_ready"],
                "status": "published",
                "target": str(target),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
