#!/usr/bin/env python3
"""Recompute and validate returned Part 1 question and study manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Callable

from part1_contract import FIXED_SUBJECTS, canonical_json_bytes
from part1_manifests import load_manifest_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "part1"


def _load_hugging_face_cache(path: Path) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    return [dict(row) for row in load_from_disk(str(path))]


def validate_manifest_paths(
    *,
    questions_path: Path,
    question_manifest_path: Path,
    study_manifest_path: Path,
    dataset_cache: Path | None = None,
    cache_loader: Callable[[Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    bundle = load_manifest_bundle(
        questions_path=questions_path,
        question_manifest_path=question_manifest_path,
        study_manifest_path=study_manifest_path,
    )
    counts = Counter(record["subject"] for record in bundle.records)
    cache_status = "not_checked"
    if dataset_cache is not None:
        loader = cache_loader or _load_hugging_face_cache
        cached_records = loader(dataset_cache)
        if canonical_json_bytes(cached_records) != canonical_json_bytes(list(bundle.records)):
            raise ValueError("saved dataset cache differs from authoritative question manifest")
        cache_status = "matches_authoritative_manifest"
    return {
        "is_valid": True,
        "total_count": len(bundle.records),
        "subject_counts": {subject: counts[subject] for subject in FIXED_SUBJECTS},
        "source_revision": bundle.question_manifest["source_revision"],
        "question_manifest_hash": bundle.question_manifest["question_manifest_hash"],
        "study_id": bundle.study_manifest["study_id"],
        "study_manifest_hash": bundle.study_manifest["study_manifest_hash"],
        "dataset_cache": cache_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT / "questions.jsonl",
    )
    parser.add_argument(
        "--question-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT / "questions.manifest.json",
    )
    parser.add_argument(
        "--study-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT / "study_manifest.json",
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="optional saved Dataset path; importing datasets is deferred unless provided",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = validate_manifest_paths(
            questions_path=args.questions,
            question_manifest_path=args.question_manifest,
            study_manifest_path=args.study_manifest,
            dataset_cache=args.dataset_cache,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "is_valid": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
