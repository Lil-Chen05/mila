#!/usr/bin/env python3
"""Create the clean-provenance, ignored Phase 3 smoke model-run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

from create_part1_model_run_manifest import (
    _ensure_regular_directory_chain,
    _fsync_directory,
    _git_state,
    _manifest_bytes,
    _sha256_regular_file,
)
from part1_manifests import load_manifest_bundle
from part1_model_run import (
    build_smoke_model_run_manifest,
    validate_preflight_model_run_compatibility,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("results/part1-smoke/model-runs/phase3_smoke")


def publish_phase3_smoke_manifest(
    *,
    study_manifest: Mapping[str, Any],
    question_manifest: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    repository_root = Path(repository_root)
    output_root = Path(output_root)
    if output_root != OUTPUT_ROOT:
        raise ValueError("Phase 3 smoke manifest must use its dedicated ignored root")
    head, blocking, _unrelated = _git_state(repository_root)
    if blocking:
        raise RuntimeError(
            "Phase 3 smoke manifest requires a clean tracked worktree: "
            + ", ".join(blocking)
        )
    lock_hash = _sha256_regular_file(repository_root / "uv.lock")
    if preflight_report.get("environment_versions", {}).get("uv_lock_sha256") != lock_hash:
        raise RuntimeError("preflight dependency lock hash differs from current uv.lock")
    manifest = build_smoke_model_run_manifest(
        study_manifest=study_manifest,
        preflight_report=preflight_report,
        execution_scope="phase3_smoke",
        base_git_commit=head,
        diff_hash=hashlib.sha256(b"").hexdigest(),
    )
    validate_preflight_model_run_compatibility(
        preflight_report=preflight_report,
        model_manifest=manifest,
        study_manifest=study_manifest,
        question_manifest=question_manifest,
    )
    content = _manifest_bytes(manifest)
    parent = _ensure_regular_directory_chain(repository_root, output_root)
    target = parent / "model_run_manifest.json"
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise RuntimeError(f"existing Phase 3 smoke manifest is incompatible: {target}")
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".model_run_manifest.json.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        head_again, blocking_again, _ = _git_state(repository_root)
        if head_again != head or blocking_again:
            raise RuntimeError("Phase 3 smoke manifest requires a clean tracked worktree")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                raise RuntimeError(
                    f"Phase 3 smoke manifest changed during publication: {target}"
                )
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args(argv)
    manifest_root = (
        args.manifest_root
        if args.manifest_root is not None
        else args.repository_root / "manifests/part1"
    )
    preflight_path = (
        args.preflight
        if args.preflight is not None
        else args.repository_root / "results/part1-smoke/preflight/preflight.json"
    )
    try:
        bundle = load_manifest_bundle(
            questions_path=manifest_root / "questions.jsonl",
            question_manifest_path=manifest_root / "questions.manifest.json",
            study_manifest_path=manifest_root / "study_manifest.json",
        )
        path = publish_phase3_smoke_manifest(
            study_manifest=bundle.study_manifest,
            question_manifest=bundle.question_manifest,
            preflight_report=_load_json(preflight_path),
            repository_root=args.repository_root,
        )
        manifest = _load_json(path)
    except Exception as exc:
        print(
            json.dumps(
                {"is_valid": False, "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "is_valid": True,
                "manifest_path": str(path),
                "model_run_id": manifest["model_run_id"],
                "execution_scope": "phase3_smoke",
                "natural_run_budget": 10,
                "checkpoint_budget": 110,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
