"""CPU bootstrap orchestration tests with no real dataset access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from materialize_part1_mmlu import run_materialization
from part1_contract import FIXED_SUBJECTS
from part1_manifests import load_manifest_bundle


REVISION = "d" * 40


class FakeDatasetBackend:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.split_counts = {subject: 150 + index for index, subject in enumerate(FIXED_SUBJECTS)}
        self.fail_cache_reload = False

    def resolve_revision(self, repository: str, requested_revision: str) -> str:
        self.calls.append(("resolve", repository, requested_revision))
        return REVISION

    def test_split_size(self, repository: str, subject: str, revision: str) -> int:
        self.calls.append(("split_size", repository, subject, revision))
        return self.split_counts[subject]

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
    ) -> list[dict]:
        self.calls.append(
            (
                "select",
                repository,
                subject,
                split,
                revision,
                seed,
                buffer_size,
                quota,
            )
        )
        subject_index = FIXED_SUBJECTS.index(subject)
        return [
            {
                "question": f"{subject} {selection_index}?",
                "choices": [f"choice {choice}" for choice in range(4)],
                "answer": (subject_index + selection_index) % 4,
                "_source_row_index": subject_index * 1000 + selection_index,
            }
            for selection_index in range(quota)
        ]

    def save_cache(self, path: Path, records: list[dict]) -> None:
        self.calls.append(("save_cache", path, len(records)))
        path.mkdir(parents=True)
        (path / "records.json").write_text(json.dumps(records), encoding="utf-8")

    def load_cache(self, path: Path) -> list[dict]:
        self.calls.append(("load_cache", path))
        if self.fail_cache_reload:
            raise ValueError("synthetic cache reload failure")
        return json.loads((path / "records.json").read_text(encoding="utf-8"))


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": "part1_dataset_materialization_config",
                "config_version": "1.1.0",
                "source_repository": "cais/mmlu",
                "source_revision": "main",
                "source_config_strategy": "per_subject",
                "source_configs": FIXED_SUBJECTS,
                "source_split": "test",
                "streaming": True,
                "bounded_take_required": True,
                "question_sampling_seed": 42,
                "subjects": FIXED_SUBJECTS,
                "quota_per_subject": 100,
                "source_revision_required_before_materialization": True,
            }
        ),
        encoding="utf-8",
    )


def test_run_materialization_stages_validates_and_publishes_fixed_bundle(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.json"
    write_config(config_path)
    backend = FakeDatasetBackend()

    report = run_materialization(
        config_path=config_path,
        manifest_root=tmp_path / "manifests" / "part1",
        cache_root=tmp_path / "data" / "part1",
        backend=backend,
    )

    assert backend.calls[0] == ("resolve", "cais/mmlu", "main")
    selection_calls = [call for call in backend.calls if call[0] == "select"]
    assert [call[2] for call in selection_calls] == FIXED_SUBJECTS
    for subject, call in zip(FIXED_SUBJECTS, selection_calls, strict=True):
        assert call[3:] == (
            "test",
            REVISION,
            42,
            backend.split_counts[subject],
            100,
        )

    assert report["resolved_revision"] == REVISION
    assert report["total_count"] == 500
    assert report["manifest_publication"] == {
        "questions": "published",
        "question_manifest": "published",
        "study_manifest": "published",
    }
    assert report["cache_publication"] == "published"
    final = tmp_path / "manifests" / "part1"
    bundle = load_manifest_bundle(
        questions_path=final / "questions.jsonl",
        question_manifest_path=final / "questions.manifest.json",
        study_manifest_path=final / "study_manifest.json",
    )
    cache_path = Path(report["cache_path"])
    assert cache_path.name == f"mmlu-{bundle.question_manifest['question_manifest_hash']}"
    assert backend.load_cache(cache_path) == list(bundle.records)


def test_run_materialization_rejects_subject_smaller_than_quota_before_staging(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dataset.json"
    write_config(config_path)
    backend = FakeDatasetBackend()
    backend.split_counts[FIXED_SUBJECTS[2]] = 99
    manifest_root = tmp_path / "manifests" / "part1"
    cache_root = tmp_path / "data" / "part1"

    with pytest.raises(ValueError, match="fewer than 100"):
        run_materialization(
            config_path=config_path,
            manifest_root=manifest_root,
            cache_root=cache_root,
            backend=backend,
        )
    assert not (manifest_root / "questions.jsonl").exists()
    assert not (manifest_root / "questions.manifest.json").exists()
    assert not (manifest_root / "study_manifest.json").exists()
    assert not any(path.name.startswith("mmlu-") for path in cache_root.glob("mmlu-*"))


def test_cache_reload_failure_publishes_nothing(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.json"
    write_config(config_path)
    backend = FakeDatasetBackend()
    backend.fail_cache_reload = True
    manifest_root = tmp_path / "manifests" / "part1"
    cache_root = tmp_path / "data" / "part1"

    with pytest.raises(ValueError, match="cache reload failure"):
        run_materialization(
            config_path=config_path,
            manifest_root=manifest_root,
            cache_root=cache_root,
            backend=backend,
        )
    assert not any((manifest_root / name).exists() for name in (
        "questions.jsonl",
        "questions.manifest.json",
        "study_manifest.json",
    ))
    assert not list(cache_root.glob("mmlu-*"))


def test_incompatible_existing_manifest_prevents_cache_or_other_manifest_publication(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dataset.json"
    write_config(config_path)
    backend = FakeDatasetBackend()
    manifest_root = tmp_path / "manifests" / "part1"
    manifest_root.mkdir(parents=True)
    existing_study = manifest_root / "study_manifest.json"
    existing_study.write_text('{"different":true}\n', encoding="utf-8")
    cache_root = tmp_path / "data" / "part1"

    with pytest.raises(RuntimeError, match="study_manifest"):
        run_materialization(
            config_path=config_path,
            manifest_root=manifest_root,
            cache_root=cache_root,
            backend=backend,
        )
    assert existing_study.read_text(encoding="utf-8") == '{"different":true}\n'
    assert not (manifest_root / "questions.jsonl").exists()
    assert not (manifest_root / "questions.manifest.json").exists()
    assert not list(cache_root.glob("mmlu-*"))


def test_identical_rerun_retains_finalized_manifests_and_cache(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.json"
    write_config(config_path)
    manifest_root = tmp_path / "manifests" / "part1"
    cache_root = tmp_path / "data" / "part1"
    first = run_materialization(
        config_path=config_path,
        manifest_root=manifest_root,
        cache_root=cache_root,
        backend=FakeDatasetBackend(),
    )
    inode = (manifest_root / "questions.jsonl").stat().st_ino

    second = run_materialization(
        config_path=config_path,
        manifest_root=manifest_root,
        cache_root=cache_root,
        backend=FakeDatasetBackend(),
    )

    assert second["question_manifest_hash"] == first["question_manifest_hash"]
    assert second["manifest_publication"] == {
        "questions": "identical_existing",
        "question_manifest": "identical_existing",
        "study_manifest": "identical_existing",
    }
    assert second["cache_publication"] == "identical_existing"
    assert (manifest_root / "questions.jsonl").stat().st_ino == inode
