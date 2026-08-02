#!/usr/bin/env python3
"""Estimate Part 1 JSONL storage before any model execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return converted


def estimate_part1_storage(
    *,
    question_count: int = 500,
    natural_runs_per_question: int = 10,
    checkpoints_per_natural: int = 11,
    expected_generated_tokens: int = 2048,
    expected_decoded_utf8_bytes: int = 12000,
    expected_natural_record_overhead_bytes: int = 4000,
    expected_checkpoint_record_bytes: int = 2400,
    json_integer_bytes: float = 8.0,
    json_number_bytes: float = 20.0,
) -> dict[str, Any]:
    values = {
        "question_count": question_count,
        "natural_runs_per_question": natural_runs_per_question,
        "checkpoints_per_natural": checkpoints_per_natural,
        "expected_generated_tokens": expected_generated_tokens,
        "expected_decoded_utf8_bytes": expected_decoded_utf8_bytes,
        "expected_natural_record_overhead_bytes": expected_natural_record_overhead_bytes,
        "expected_checkpoint_record_bytes": expected_checkpoint_record_bytes,
    }
    checked = {name: _nonnegative_int(name, value) for name, value in values.items()}
    integer_bytes = _finite_nonnegative("json_integer_bytes", json_integer_bytes)
    number_bytes = _finite_nonnegative("json_number_bytes", json_number_bytes)
    natural_count = checked["question_count"] * checked["natural_runs_per_question"]
    checkpoint_count = natural_count * checked["checkpoints_per_natural"]
    decoded_text_bytes = natural_count * checked["expected_decoded_utf8_bytes"]
    token_id_array_bytes = int(
        natural_count * checked["expected_generated_tokens"] * integer_bytes
    )
    entropy_array_bytes = int(
        natural_count * checked["expected_generated_tokens"] * number_bytes
    )
    natural_overhead_bytes = natural_count * checked[
        "expected_natural_record_overhead_bytes"
    ]
    checkpoint_record_bytes = checkpoint_count * checked[
        "expected_checkpoint_record_bytes"
    ]
    total = (
        decoded_text_bytes
        + token_id_array_bytes
        + entropy_array_bytes
        + natural_overhead_bytes
        + checkpoint_record_bytes
    )
    return {
        "natural_run_count": natural_count,
        "checkpoint_count": checkpoint_count,
        "decoded_text_bytes": decoded_text_bytes,
        "token_id_array_bytes": token_id_array_bytes,
        "entropy_array_bytes": entropy_array_bytes,
        "natural_record_overhead_bytes": natural_overhead_bytes,
        "checkpoint_record_bytes": checkpoint_record_bytes,
        "total_estimated_bytes": total,
        "stores_full_vocabulary_logits": False,
    }


def assess_free_space(
    estimate: dict[str, Any], *, free_bytes: int, near_multiplier: float = 1.25
) -> dict[str, Any]:
    free = _nonnegative_int("free_bytes", free_bytes)
    multiplier = _finite_nonnegative("near_multiplier", near_multiplier)
    if multiplier < 1.0:
        raise ValueError("near_multiplier must be at least 1.0")
    required = _nonnegative_int(
        "total_estimated_bytes", estimate.get("total_estimated_bytes")
    )
    near_threshold = math.ceil(required * multiplier)
    if free < required:
        status = "insufficient"
        warning = f"free space is {required - free} bytes below the estimate"
    elif free < near_threshold:
        status = "near_threshold"
        warning = f"free space is below the configured {multiplier:g}x safety threshold"
    else:
        status = "sufficient"
        warning = None
    return {
        "status": status,
        "warning": warning,
        "free_bytes": free,
        "required_bytes": required,
        "near_threshold_bytes": near_threshold,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-count", type=int, default=500)
    parser.add_argument("--expected-generated-tokens", type=int, default=2048)
    parser.add_argument("--expected-decoded-utf8-bytes", type=int, default=12000)
    parser.add_argument(
        "--check-free-space-at",
        type=Path,
        help="optional persistent root whose current free bytes should be assessed",
    )
    parser.add_argument("--near-multiplier", type=float, default=1.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    estimate = estimate_part1_storage(
        question_count=args.question_count,
        expected_generated_tokens=args.expected_generated_tokens,
        expected_decoded_utf8_bytes=args.expected_decoded_utf8_bytes,
    )
    report: dict[str, Any] = {"estimate": estimate}
    if args.check_free_space_at is not None:
        free = shutil.disk_usage(args.check_free_space_at).free
        report["free_space"] = assess_free_space(
            estimate, free_bytes=free, near_multiplier=args.near_multiplier
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if report.get("free_space", {}).get("status") == "insufficient":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
