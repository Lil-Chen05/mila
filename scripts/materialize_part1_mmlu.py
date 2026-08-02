#!/usr/bin/env python3
"""Materialize the fixed Part 1 MMLU selection inside a CPU SLURM job."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Protocol

from part1_contract import canonical_json_bytes, load_config
from part1_manifests import (
    DATASET_REPOSITORY,
    DATASET_SPLIT,
    QUESTION_QUOTA_PER_SUBJECT,
    QUESTION_SAMPLING_SEED,
    build_manifest_bundle,
    load_manifest_bundle,
    preflight_manifest_publication,
    publish_manifest_bundle,
    require_immutable_revision,
    write_staged_manifest_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "part1" / "dataset_materialization.json"
DEFAULT_MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "part1"
DEFAULT_CACHE_ROOT = REPOSITORY_ROOT / "data" / "part1"


class DatasetBackend(Protocol):
    def resolve_revision(self, repository: str, requested_revision: str) -> str: ...

    def test_split_size(self, repository: str, subject: str, revision: str) -> int: ...

    def select_subject_rows(
        self,
        *,
        repository: str,
        subject: str,
        split: str,
        revision: str,
        seed: int,
        buffer_size: int,
        quota: int,
    ) -> list[dict[str, Any]]: ...

    def save_cache(self, path: Path, records: list[dict[str, Any]]) -> None: ...

    def load_cache(self, path: Path) -> list[dict[str, Any]]: ...


def _attach_source_row_index(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {**row, "_source_row_index": index}


class HuggingFaceDatasetBackend:
    """Thin lazy-import adapter; importing this module never loads a dataset."""

    def resolve_revision(self, repository: str, requested_revision: str) -> str:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id=repository, revision=requested_revision)
        revision = getattr(info, "sha", None)
        return require_immutable_revision(revision)

    def test_split_size(self, repository: str, subject: str, revision: str) -> int:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(repository, name=subject, revision=revision)
        split_info = builder.info.splits.get(DATASET_SPLIT)
        count = getattr(split_info, "num_examples", None) if split_info is not None else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"could not verify full {subject}/{DATASET_SPLIT} split size at {revision}"
            )
        return count

    def select_subject_rows(
        self,
        *,
        repository: str,
        subject: str,
        split: str,
        revision: str,
        seed: int,
        buffer_size: int,
        quota: int,
    ) -> list[dict[str, Any]]:
        from datasets import load_dataset

        stream = load_dataset(
            repository,
            name=subject,
            split=split,
            revision=revision,
            streaming=True,
        )
        indexed = stream.map(_attach_source_row_index, with_indices=True)
        shuffled = indexed.shuffle(seed=seed, buffer_size=buffer_size)
        return [dict(row) for row in shuffled.take(quota)]

    def save_cache(self, path: Path, records: list[dict[str, Any]]) -> None:
        from datasets import Dataset

        Dataset.from_list(records).save_to_disk(str(path))

    def load_cache(self, path: Path) -> list[dict[str, Any]]:
        from datasets import load_from_disk

        saved = load_from_disk(str(path))
        return [dict(row) for row in saved]


def _load_materialization_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    expected = load_config("dataset_materialization")
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(
            "dataset materialization config differs from the tracked fixed Part 1 contract"
        )
    return value


def _records_equal(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    return canonical_json_bytes(list(left)) == canonical_json_bytes(list(right))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preflight_cache(
    *, backend: DatasetBackend, staged_cache: Path, final_cache: Path, records: list[dict[str, Any]]
) -> str:
    staged_records = backend.load_cache(staged_cache)
    if not _records_equal(staged_records, records):
        raise ValueError("staged dataset cache differs from the finalized 500 question records")
    if os.path.lexists(final_cache):
        if not final_cache.is_dir() or final_cache.is_symlink():
            raise RuntimeError(f"existing finalized dataset cache is incompatible: {final_cache}")
        existing_records = backend.load_cache(final_cache)
        if not _records_equal(existing_records, records):
            raise RuntimeError(f"existing finalized dataset cache is incompatible: {final_cache}")
        return "identical_existing"
    return "missing"


def _publish_cache(staged_cache: Path, final_cache: Path, state: str) -> str:
    if state == "identical_existing":
        return state
    if state != "missing":
        raise ValueError(f"unsupported cache publication state: {state}")
    if staged_cache.stat().st_dev != final_cache.parent.stat().st_dev:
        raise OSError("staged and final dataset cache paths are on different filesystems")
    os.replace(staged_cache, final_cache)
    _fsync_directory(final_cache.parent)
    return "published"


def run_materialization(
    *,
    config_path: Path,
    manifest_root: Path,
    cache_root: Path,
    backend: DatasetBackend | None = None,
) -> dict[str, Any]:
    """Run the validated temp-first bootstrap and return a JSON-safe report."""

    config = _load_materialization_config(config_path)
    selected_backend = backend or HuggingFaceDatasetBackend()
    resolved_revision = require_immutable_revision(
        selected_backend.resolve_revision(
            config["source_repository"], config["source_revision"]
        )
    )

    selected: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, int] = {}
    for subject in config["source_configs"]:
        split_count = selected_backend.test_split_size(
            config["source_repository"], subject, resolved_revision
        )
        if isinstance(split_count, bool) or not isinstance(split_count, int) or split_count < 0:
            raise ValueError(f"could not verify full test split size for {subject}")
        if split_count < config["quota_per_subject"]:
            raise ValueError(
                f"{subject} has fewer than 100 verified test questions at {resolved_revision}"
            )
        split_counts[subject] = split_count
        rows = selected_backend.select_subject_rows(
            repository=config["source_repository"],
            subject=subject,
            split=config["source_split"],
            revision=resolved_revision,
            seed=config["question_sampling_seed"],
            buffer_size=split_count,
            quota=config["quota_per_subject"],
        )
        if len(rows) != QUESTION_QUOTA_PER_SUBJECT:
            raise ValueError(f"{subject} streaming take returned fewer than 100 rows")
        selected[subject] = rows

    bundle = build_manifest_bundle(selected, resolved_revision=resolved_revision)
    records = list(bundle.records)
    manifest_root.parent.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_temp = Path(
        tempfile.mkdtemp(prefix=".part1.materialize-", dir=manifest_root.parent)
    )
    cache_temp = Path(tempfile.mkdtemp(prefix=".materialize-", dir=cache_root))
    try:
        staged_manifests = write_staged_manifest_bundle(manifest_temp / "part1", bundle)
        staged_cache = cache_temp / "dataset"
        selected_backend.save_cache(staged_cache, records)

        # Reload both staged representations before checking or mutating final paths.
        reloaded_bundle = load_manifest_bundle(
            questions_path=staged_manifests["questions"],
            question_manifest_path=staged_manifests["question_manifest"],
            study_manifest_path=staged_manifests["study_manifest"],
        )
        if reloaded_bundle != bundle:
            raise ValueError("staged manifest bundle differs after reload")

        final_manifests = {
            "questions": manifest_root / "questions.jsonl",
            "question_manifest": manifest_root / "questions.manifest.json",
            "study_manifest": manifest_root / "study_manifest.json",
        }
        final_cache = cache_root / f"mmlu-{bundle.question_manifest['question_manifest_hash']}"
        cache_state = _preflight_cache(
            backend=selected_backend,
            staged_cache=staged_cache,
            final_cache=final_cache,
            records=records,
        )
        preflight_manifest_publication(staged_manifests, final_manifests)

        # The ignored cache is published first. Final manifest publication has
        # already been all-target preflighted and remains authoritative.
        cache_publication = _publish_cache(staged_cache, final_cache, cache_state)
        manifest_publication = publish_manifest_bundle(staged_manifests, final_manifests)
    finally:
        shutil.rmtree(manifest_temp, ignore_errors=True)
        shutil.rmtree(cache_temp, ignore_errors=True)

    return {
        "status": "published",
        "source_repository": DATASET_REPOSITORY,
        "resolved_revision": resolved_revision,
        "subject_split_counts": split_counts,
        "total_count": len(bundle.records),
        "question_manifest_hash": bundle.question_manifest["question_manifest_hash"],
        "study_id": bundle.study_manifest["study_id"],
        "study_manifest_hash": bundle.study_manifest["study_manifest_hash"],
        "manifest_paths": {key: str(path) for key, path in final_manifests.items()},
        "cache_path": str(final_cache),
        "manifest_publication": manifest_publication,
        "cache_publication": cache_publication,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_materialization(
            config_path=args.config,
            manifest_root=args.manifest_root,
            cache_root=args.cache_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
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
