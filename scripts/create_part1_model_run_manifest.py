#!/usr/bin/env python3
"""Create the ignored Part 1 production model-run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from part1_model_run import build_production_model_run_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_RELEVANT_ROOTS = {"scripts", "jobs", "configs", "schemas", "manifests"}


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_state(repository_root: Path) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    head = _git_output(repository_root, "rev-parse", "HEAD").strip()
    tracked = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repository_root,
        check=False,
    )
    if tracked.returncode not in {0, 1}:
        raise RuntimeError("could not inspect tracked Git worktree state")
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    paths = tuple(
        sorted(
            os.fsdecode(item)
            for item in untracked_output.split(b"\0")
            if item
        )
    )
    scoped = tuple(
        path for path in paths if Path(path).parts and Path(path).parts[0] in EXECUTION_RELEVANT_ROOTS
    )
    unrelated = tuple(path for path in paths if path not in scoped)
    if tracked.returncode == 1:
        scoped = ("<tracked-worktree-or-index-change>", *scoped)
    return head, scoped, unrelated


def _require_clean_git_state(repository_root: Path, expected_head: str) -> tuple[str, ...]:
    head, blocking_paths, unrelated_paths = _git_state(repository_root)
    if head != expected_head:
        raise RuntimeError(
            f"HEAD {head} differs from recorded final production commit {expected_head}"
        )
    if blocking_paths:
        raise RuntimeError(
            "production manifest requires a clean tracked worktree and no "
            "execution-relevant untracked paths: " + ", ".join(blocking_paths)
        )
    return unrelated_paths


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"dependency lock is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_regular_directory_chain(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("production output path must be repository-relative")
    current = root
    for part in relative.parts:
        child = current / part
        if os.path.lexists(child):
            if child.is_symlink() or not child.is_dir():
                raise RuntimeError(
                    f"production output path component is not a regular directory: {child}"
                )
        else:
            child.mkdir()
            _fsync_directory(current)
        current = child
    return current


def _remove_new_empty_run_directory(path: Path) -> None:
    """Remove only an empty run directory created by the current attempt."""

    try:
        path.rmdir()
    except OSError:
        return
    _fsync_directory(path.parent)


def publish_production_model_run_manifest(
    *,
    study_manifest: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    final_git_commit: str,
    output_root: Path = Path("results/part1"),
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Build and atomically publish an identical-only production manifest."""

    repository_root = Path(repository_root)
    output_root = Path(output_root)
    _require_clean_git_state(repository_root, final_git_commit)
    lock_hash = _sha256_regular_file(repository_root / "uv.lock")
    preflight_lock_hash = preflight_report.get("environment_versions", {}).get(
        "uv_lock_sha256"
    )
    if preflight_lock_hash != lock_hash:
        raise RuntimeError("preflight dependency lock hash differs from current uv.lock bytes")

    manifest = build_production_model_run_manifest(
        study_manifest=study_manifest,
        preflight_report=preflight_report,
        final_git_commit=final_git_commit,
        output_root=output_root,
    )
    if manifest["dependency_lock_sha256"] != lock_hash:
        raise RuntimeError("production dependency lock hash differs from current uv.lock bytes")
    content = _manifest_bytes(manifest)
    run_relative = output_root / manifest["model_run_id"]
    run_root_existed = os.path.lexists(repository_root / run_relative)
    target_parent = _ensure_regular_directory_chain(repository_root, run_relative)
    target = target_parent / "model_run_manifest.json"
    if run_root_existed and not os.path.lexists(target):
        raise RuntimeError(
            f"existing production model-run directory is partial: {target_parent}"
        )
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise RuntimeError(f"existing production model-run manifest is incompatible: {target}")
        _require_clean_git_state(repository_root, final_git_commit)
        return target

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _require_clean_git_state(repository_root, final_git_commit)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
                raise RuntimeError(
                    f"production model-run manifest changed during publication: {target}"
                )
        temporary.unlink()
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
        if not run_root_existed and not os.path.lexists(target):
            _remove_new_empty_run_directory(target_parent)
    return target


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--study-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "manifests" / "part1" / "study_manifest.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "part1-smoke" / "preflight" / "preflight.json",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/part1"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        head, blocking_paths, unrelated_paths = _git_state(args.repository_root)
        if blocking_paths:
            raise RuntimeError(
                "production manifest requires a clean tracked worktree and no "
                "execution-relevant untracked paths: " + ", ".join(blocking_paths)
            )
        path = publish_production_model_run_manifest(
            study_manifest=_load_json_object(args.study_manifest),
            preflight_report=_load_json_object(args.preflight),
            final_git_commit=head,
            output_root=args.output_root,
            repository_root=args.repository_root,
        )
        manifest = _load_json_object(path)
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
    print(
        json.dumps(
            {
                "is_valid": True,
                "manifest_path": str(path),
                "model_run_id": manifest["model_run_id"],
                "model_run_manifest_hash": manifest["model_run_manifest_hash"],
                "final_production_git_commit": manifest[
                    "final_production_git_commit"
                ],
                "unrelated_untracked_paths": list(unrelated_paths),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
