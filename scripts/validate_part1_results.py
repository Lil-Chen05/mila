"""Validate and atomically publish complete Part 1 production coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from part1_coverage import build_coverage_report, publish_coverage_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-run-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repository_root = args.repository_root.resolve()
        manifest_path = args.model_run_manifest
        if not manifest_path.is_absolute():
            manifest_path = repository_root / manifest_path
        manifest = _load_json(manifest_path)
        output_paths = manifest.get("output_paths")
        if not isinstance(output_paths, dict) or not isinstance(
            output_paths.get("validation"), str
        ):
            raise ValueError("production manifest has no canonical validation output path")
        relative_validation = Path(output_paths["validation"])
        if relative_validation.is_absolute() or ".." in relative_validation.parts:
            raise ValueError("production validation output path must be safe and relative")
        report = build_coverage_report(
            repository_root=repository_root,
            model_run_manifest_path=manifest_path,
        )
        target = repository_root / relative_validation / "coverage_report.json"
        publish_coverage_report(report, target, repository_root=repository_root)
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
    print(
        json.dumps(
            {
                "status": "ready" if report["paper_analysis_ready"] else "not_ready",
                "coverage_report": str(target),
                "validation_report_id": report["validation_report_id"],
                "structurally_valid": report["structurally_valid"],
                "coverage_complete": report["coverage_complete"],
                "paper_analysis_ready": report["paper_analysis_ready"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["paper_analysis_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
