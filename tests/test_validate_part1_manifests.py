"""Tests for the post-Mila manifest validation command."""

from __future__ import annotations

from pathlib import Path

import pytest

from part1_contract import FIXED_SUBJECTS
from part1_manifests import build_manifest_bundle, write_staged_manifest_bundle
from validate_part1_manifests import validate_manifest_paths


def _bundle():
    rows = {
        subject: [
            {
                "question": f"{subject}-{index}",
                "choices": ["a", "b", "c", "d"],
                "answer": index % 4,
                "_source_row_index": index,
            }
            for index in range(100)
        ]
        for subject in FIXED_SUBJECTS
    }
    return build_manifest_bundle(rows, resolved_revision="a" * 40)


def test_validate_manifest_paths_reports_recomputed_authoritative_identity(tmp_path: Path) -> None:
    bundle = _bundle()
    paths = write_staged_manifest_bundle(tmp_path / "bundle", bundle)

    report = validate_manifest_paths(
        questions_path=paths["questions"],
        question_manifest_path=paths["question_manifest"],
        study_manifest_path=paths["study_manifest"],
    )

    assert report == {
        "is_valid": True,
        "total_count": 500,
        "subject_counts": {subject: 100 for subject in FIXED_SUBJECTS},
        "source_revision": "a" * 40,
        "question_manifest_hash": bundle.question_manifest["question_manifest_hash"],
        "study_id": bundle.study_manifest["study_id"],
        "study_manifest_hash": bundle.study_manifest["study_manifest_hash"],
        "dataset_cache": "not_checked",
    }


def test_validate_manifest_paths_rejects_tampered_returned_jsonl(tmp_path: Path) -> None:
    bundle = _bundle()
    paths = write_staged_manifest_bundle(tmp_path / "bundle", bundle)
    questions = paths["questions"].read_text(encoding="utf-8")
    paths["questions"].write_text(questions.replace("high_school_mathematics-0", "changed", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash|question ID|manifest hash"):
        validate_manifest_paths(
            questions_path=paths["questions"],
            question_manifest_path=paths["question_manifest"],
            study_manifest_path=paths["study_manifest"],
        )
