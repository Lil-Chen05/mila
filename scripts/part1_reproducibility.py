#!/usr/bin/env python3
"""GPU-only same-environment deterministic regeneration check."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from part1_contract import derive_generation_seed
from part1_manifests import load_manifest_bundle
from part1_model_run import validate_preflight_model_run_compatibility
from part1_smollm3_adapter import (
    load_model_and_tokenizer,
    parse_natural_output,
    preflight_tokenizer_contract,
)
from run_part1_smoke import _execute_natural


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "part1"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "part1-smoke"


def compare_natural_results(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    entropy_abs_tolerance: float,
) -> dict[str, Any]:
    if not math.isfinite(entropy_abs_tolerance) or entropy_abs_tolerance < 0:
        raise ValueError("entropy_abs_tolerance must be finite and nonnegative")
    left_entropy = first["per_token_entropy_nats"]
    right_entropy = second["per_token_entropy_nats"]
    entropy_equal = len(left_entropy) == len(right_entropy) and all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=entropy_abs_tolerance)
        for left, right in zip(left_entropy, right_entropy, strict=False)
    )
    return {
        "exact_generated_token_equality": first["generated_token_ids"]
        == second["generated_token_ids"],
        "exact_parsed_output_equality": parse_natural_output(
            first["decoded_output"]
        )
        == parse_natural_output(second["decoded_output"]),
        "entropy_array_equal_within_tolerance": entropy_equal,
        "entropy_abs_tolerance": entropy_abs_tolerance,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing reproducibility report is incompatible: {path}")
        return
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def run_reproducibility(
    *,
    manifest_root: Path,
    preflight_path: Path,
    model_run_manifest_path: Path,
    output_root: Path,
    entropy_abs_tolerance: float,
) -> dict[str, Any]:
    bundle = load_manifest_bundle(
        questions_path=manifest_root / "questions.jsonl",
        question_manifest_path=manifest_root / "questions.manifest.json",
        study_manifest_path=manifest_root / "study_manifest.json",
    )
    preflight = _load_json(preflight_path)
    model_manifest = _load_json(model_run_manifest_path)
    if model_manifest.get("execution_scope") != "reproducibility":
        raise ValueError("reproducibility requires its separate model-run identity")
    if model_manifest.get("production") is not False:
        raise ValueError("reproducibility runner refuses production manifests")
    validate_preflight_model_run_compatibility(
        preflight_report=preflight,
        model_manifest=model_manifest,
        study_manifest=bundle.study_manifest,
        question_manifest=bundle.question_manifest,
    )
    model, tokenizer = load_model_and_tokenizer(
        model_revision=model_manifest["model_revision"],
        tokenizer_revision=model_manifest["tokenizer_revision"],
    )
    token_contract = preflight_tokenizer_contract(tokenizer)
    if token_contract != preflight["token_contract"]:
        raise ValueError("runtime tokenizer contract differs from GPU preflight")
    question = bundle.records[0]
    seed = derive_generation_seed(
        base_seed=model_manifest["base_generation_seed"],
        canonical_model_identity=model_manifest["canonical_model_identity"],
        question_id=question["question_id"],
        run_id=0,
    )
    first = _execute_natural(
        model=model,
        tokenizer=tokenizer,
        question=question,
        run_id=0,
        seed=seed,
        attempt_number=1,
        model_manifest=model_manifest,
        token_contract=token_contract,
    )
    second = _execute_natural(
        model=model,
        tokenizer=tokenizer,
        question=question,
        run_id=0,
        seed=seed,
        attempt_number=1,
        model_manifest=model_manifest,
        token_contract=token_contract,
    )
    comparison = compare_natural_results(
        first,
        second,
        entropy_abs_tolerance=entropy_abs_tolerance,
    )
    passed = all(
        comparison[field]
        for field in (
            "exact_generated_token_equality",
            "exact_parsed_output_equality",
            "entropy_array_equal_within_tolerance",
        )
    )
    report = {
        "status": "passed" if passed else "failed",
        "scope": "same model/tokenizer/software/GPU/configuration/seed only",
        "model_run_id": model_manifest["model_run_id"],
        "question_id": question["question_id"],
        "run_id": 0,
        "generation_seed": seed,
        **comparison,
    }
    report_path = (
        output_root
        / "reproducibility"
        / model_manifest["model_run_id"]
        / "reproducibility_report.json"
    )
    _atomic_write(report_path, report)
    return {**report, "report_path": str(report_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "preflight" / "preflight.json",
    )
    parser.add_argument(
        "--model-run-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "model-runs"
        / "reproducibility"
        / "model_run_manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--entropy-abs-tolerance", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_reproducibility(
            manifest_root=args.manifest_root,
            preflight_path=args.preflight,
            model_run_manifest_path=args.model_run_manifest,
            output_root=args.output_root,
            entropy_abs_tolerance=args.entropy_abs_tolerance,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
