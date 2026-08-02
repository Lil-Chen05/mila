"""Login-safe tests for fixed Part 1 question/study materialization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from part1_contract import FIXED_STUDY_CONTRACT, FIXED_SUBJECTS
from part1_manifests import (
    ManifestCompatibilityError,
    build_manifest_bundle,
    load_manifest_bundle,
    manifest_bytes,
    preflight_manifest_publication,
    publish_manifest_bundle,
    validate_manifest_bundle,
    write_staged_manifest_bundle,
)


REVISION = "a" * 40


def selected_rows() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for subject_index, subject in enumerate(FIXED_SUBJECTS):
        rows[subject] = [
            {
                "question": f"{subject} question {selection_index}?",
                "choices": [
                    f"choice-{choice_index}-{selection_index}"
                    for choice_index in range(4)
                ],
                "answer": (selection_index + subject_index) % 4,
                "_source_row_index": 1000 * subject_index + selection_index,
            }
            for selection_index in range(100)
        ]
    return rows


def test_build_manifest_bundle_preserves_fixed_subject_and_seeded_selection_order() -> None:
    source = selected_rows()
    bundle = build_manifest_bundle(source, resolved_revision=REVISION)

    assert len(bundle.records) == 500
    assert [record["sample_index"] for record in bundle.records] == list(range(500))
    for subject_index, subject in enumerate(FIXED_SUBJECTS):
        block = bundle.records[subject_index * 100 : (subject_index + 1) * 100]
        assert [record["subject"] for record in block] == [subject] * 100
        assert [record["subject_selection_index"] for record in block] == list(range(100))
        assert [record["source_row_identity"]["row_index"] for record in block] == [
            row["_source_row_index"] for row in source[subject]
        ]
        assert [record["source_config"] for record in block] == [subject] * 100

    question_manifest = bundle.question_manifest
    assert question_manifest["schema_version"] == "1.1.0"
    assert question_manifest["source_revision"] == REVISION
    assert question_manifest["source_config_strategy"] == "per_subject"
    assert question_manifest["source_configs"] == FIXED_SUBJECTS
    assert question_manifest["total_count"] == 500
    assert len(question_manifest["question_manifest_hash"]) == 64

    study = bundle.study_manifest
    assert study["schema_version"] == "1.1.0"
    assert study["question_source_revision"] == REVISION
    assert study["question_manifest_hash"] == question_manifest["question_manifest_hash"]
    assert study["study_id"] != study["study_manifest_hash"]
    for field, value in FIXED_STUDY_CONTRACT.items():
        assert study[field] == value


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[FIXED_SUBJECTS[0]].pop(), "exactly 100"),
        (
            lambda rows: rows[FIXED_SUBJECTS[0]][0].update(choices=["A", "B", "C"]),
            "four choices",
        ),
        (lambda rows: rows[FIXED_SUBJECTS[0]][0].update(answer=4), "gold index"),
        (
            lambda rows: rows[FIXED_SUBJECTS[0]][1].update(
                rows[FIXED_SUBJECTS[0]][0]
            ),
            "source row identity|duplicate",
        ),
    ],
)
def test_build_manifest_bundle_fails_closed_on_invalid_selection(mutation, message: str) -> None:
    rows = selected_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        build_manifest_bundle(rows, resolved_revision=REVISION)


@pytest.mark.parametrize("revision", ["main", "a" * 39, "G" * 40, "a" * 64])
def test_build_manifest_bundle_requires_verified_hugging_face_commit_sha(revision: str) -> None:
    with pytest.raises(ValueError, match="immutable.*40-character|revision"):
        build_manifest_bundle(selected_rows(), resolved_revision=revision)


def test_bundle_hash_validation_detects_record_and_study_tampering() -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    validate_manifest_bundle(bundle)

    tampered_records = list(copy.deepcopy(bundle.records))
    tampered_records[0]["question"] = "changed"
    with pytest.raises(ValueError, match="content hash|question ID|manifest hash"):
        validate_manifest_bundle(
            type(bundle)(
                records=tuple(tampered_records),
                question_manifest=bundle.question_manifest,
                study_manifest=bundle.study_manifest,
            )
        )

    tampered_study = {**bundle.study_manifest, "question_source_revision": "b" * 40}
    with pytest.raises(ValueError, match="revision|study"):
        validate_manifest_bundle(
            type(bundle)(
                records=bundle.records,
                question_manifest=bundle.question_manifest,
                study_manifest=tampered_study,
            )
        )


def test_staged_bundle_round_trip_reloads_exact_finalized_bytes(tmp_path: Path) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)

    loaded = load_manifest_bundle(
        questions_path=staged["questions"],
        question_manifest_path=staged["question_manifest"],
        study_manifest_path=staged["study_manifest"],
    )
    assert loaded == bundle
    assert staged["questions"].read_bytes() == manifest_bytes(bundle)["questions"]
    assert staged["question_manifest"].read_bytes() == manifest_bytes(bundle)[
        "question_manifest"
    ]
    assert staged["study_manifest"].read_bytes() == manifest_bytes(bundle)[
        "study_manifest"
    ]


def final_paths(root: Path) -> dict[str, Path]:
    return {
        "questions": root / "questions.jsonl",
        "question_manifest": root / "questions.manifest.json",
        "study_manifest": root / "study_manifest.json",
    }


def test_publication_preflights_all_targets_before_creating_any_final_file(
    tmp_path: Path,
) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)
    finals = final_paths(tmp_path / "final")
    finals["study_manifest"].parent.mkdir(parents=True)
    finals["study_manifest"].write_text('{"incompatible":true}\n', encoding="utf-8")

    with pytest.raises(ManifestCompatibilityError, match="study_manifest"):
        preflight_manifest_publication(staged, finals)
    assert not finals["questions"].exists()
    assert not finals["question_manifest"].exists()

    with pytest.raises(ManifestCompatibilityError, match="study_manifest"):
        publish_manifest_bundle(staged, finals)
    assert not finals["questions"].exists()
    assert not finals["question_manifest"].exists()
    assert finals["study_manifest"].read_text(encoding="utf-8") == '{"incompatible":true}\n'


def test_publication_rejects_partial_existing_bundle_without_filling_missing_targets(
    tmp_path: Path,
) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)
    finals = final_paths(tmp_path / "final")
    finals["questions"].parent.mkdir(parents=True)
    finals["questions"].write_bytes(staged["questions"].read_bytes())
    before_stat = finals["questions"].stat()

    with pytest.raises(ManifestCompatibilityError, match="partial|complete bundle"):
        publish_manifest_bundle(staged, finals)

    assert finals["questions"].stat().st_ino == before_stat.st_ino
    assert not finals["question_manifest"].exists()
    assert not finals["study_manifest"].exists()


def test_publication_rejects_identical_named_files_when_final_directory_has_extras(
    tmp_path: Path,
) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)
    finals = final_paths(tmp_path / "final")
    finals["questions"].parent.mkdir(parents=True)
    for key, destination in finals.items():
        destination.write_bytes(staged[key].read_bytes())
    extra = finals["questions"].parent / "unexpected.txt"
    extra.write_text("must make the finalized bundle incompatible\n", encoding="utf-8")

    with pytest.raises(ManifestCompatibilityError, match="extra|complete bundle"):
        publish_manifest_bundle(staged, finals)

    assert extra.read_text(encoding="utf-8") == (
        "must make the finalized bundle incompatible\n"
    )


def test_publication_renames_the_complete_staged_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)
    finals = final_paths(tmp_path / "final")
    real_replace = __import__("os").replace
    replacements: list[tuple[Path, Path]] = []

    def capture_replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("part1_manifests.os.replace", capture_replace)
    publication = publish_manifest_bundle(staged, finals)

    assert publication == {key: "published" for key in finals}
    assert replacements == [(tmp_path / "stage", tmp_path / "final")]
    loaded = load_manifest_bundle(
        questions_path=finals["questions"],
        question_manifest_path=finals["question_manifest"],
        study_manifest_path=finals["study_manifest"],
    )
    assert loaded == bundle


def test_directory_publish_failure_leaves_no_final_manifest_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)
    finals = final_paths(tmp_path / "final")

    def fail_replace(source, destination) -> None:
        raise OSError("synthetic directory publication failure")

    monkeypatch.setattr("part1_manifests.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic directory publication failure"):
        publish_manifest_bundle(staged, finals)

    assert not finals["questions"].exists()
    assert not finals["question_manifest"].exists()
    assert not finals["study_manifest"].exists()


def test_staged_validation_failure_never_reaches_final_paths(tmp_path: Path) -> None:
    bundle = build_manifest_bundle(selected_rows(), resolved_revision=REVISION)
    staged = write_staged_manifest_bundle(tmp_path / "stage", bundle)
    finals = final_paths(tmp_path / "final")
    question_manifest = json.loads(staged["question_manifest"].read_text(encoding="utf-8"))
    question_manifest["total_count"] = 499
    staged["question_manifest"].write_text(json.dumps(question_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="total_count|manifest"):
        load_manifest_bundle(
            questions_path=staged["questions"],
            question_manifest_path=staged["question_manifest"],
            study_manifest_path=staged["study_manifest"],
        )
    assert not any(path.exists() for path in finals.values())
