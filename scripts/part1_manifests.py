"""Login-safe construction and publication of fixed Part 1 manifests.

This module imports no dataset, model, tokenizer, or torch code. Cluster entry
points supply the already bounded, seeded per-subject rows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from part1_contract import (
    CANONICAL_JSON_VERSION,
    FIXED_STUDY_CONTRACT,
    FIXED_SUBJECTS,
    question_content_hash,
    question_id,
    question_manifest_hash,
    study_id,
    study_manifest_hash,
    validate_fixed_study_contract,
    validate_instance,
)


DATASET_REPOSITORY = "cais/mmlu"
DATASET_SPLIT = "test"
QUESTION_QUOTA_PER_SUBJECT = 100
QUESTION_TOTAL = 500
QUESTION_SAMPLING_SEED = 42
SELECTION_ALGORITHM_VERSION = "part1-per-subject-full-buffer-shuffle-v1"
ORDERED_RECORD_AGGREGATION = "canonical-record-bytes-in-manifest-order-v1"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_KEYS = ("questions", "question_manifest", "study_manifest")


class ManifestCompatibilityError(RuntimeError):
    """An existing finalized manifest differs from the prepared bundle."""


@dataclass(frozen=True)
class ManifestBundle:
    records: tuple[dict[str, Any], ...]
    question_manifest: dict[str, Any]
    study_manifest: dict[str, Any]


def require_immutable_revision(value: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise ValueError(
            "resolved dataset revision must be an immutable lowercase 40-character commit SHA"
        )
    return value


def _require_real_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _normalize_row(
    row: Mapping[str, Any],
    *,
    subject: str,
    subject_selection_index: int,
    sample_index: int,
    resolved_revision: str,
) -> dict[str, Any]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{subject} row {subject_selection_index} has an empty question")
    choices = row.get("choices")
    if (
        not isinstance(choices, (list, tuple))
        or len(choices) != 4
        or any(not isinstance(choice, str) for choice in choices)
    ):
        raise ValueError(f"{subject} row {subject_selection_index} must have exactly four choices")
    gold_index = _require_real_int(
        row.get("answer"), field=f"{subject} row {subject_selection_index} gold index"
    )
    if gold_index > 3:
        raise ValueError(f"{subject} row {subject_selection_index} gold index must be 0 through 3")
    source_row_index = _require_real_int(
        row.get("_source_row_index"),
        field=f"{subject} row {subject_selection_index} source row index",
    )
    record: dict[str, Any] = {
        "schema_name": "part1_question_record",
        "schema_version": "1.0.0",
        "question_id": "",
        "question_content_hash": "",
        "sample_index": sample_index,
        "subject": subject,
        "subject_selection_index": subject_selection_index,
        "source_repository": DATASET_REPOSITORY,
        "source_revision": resolved_revision,
        "source_config": subject,
        "source_split": DATASET_SPLIT,
        "source_row_identity": {
            "config": subject,
            "split": DATASET_SPLIT,
            "row_index": source_row_index,
        },
        "question": question,
        "choices": list(choices),
        "gold_index": gold_index,
        "gold_letter": "ABCD"[gold_index],
    }
    record["question_content_hash"] = question_content_hash(record)
    record["question_id"] = question_id(record)
    validate_instance("question_record", record)
    return record


def _build_records(
    selected_by_subject: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    resolved_revision: str,
) -> tuple[dict[str, Any], ...]:
    unexpected = set(selected_by_subject).difference(FIXED_SUBJECTS)
    missing = set(FIXED_SUBJECTS).difference(selected_by_subject)
    if unexpected or missing:
        details = []
        if missing:
            details.append(f"missing subjects: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected subjects: {', '.join(sorted(unexpected))}")
        raise ValueError("; ".join(details))

    records: list[dict[str, Any]] = []
    for subject in FIXED_SUBJECTS:
        rows = selected_by_subject[subject]
        if len(rows) != QUESTION_QUOTA_PER_SUBJECT:
            raise ValueError(f"{subject} must contribute exactly 100 selected rows")
        for subject_selection_index, row in enumerate(rows):
            records.append(
                _normalize_row(
                    row,
                    subject=subject,
                    subject_selection_index=subject_selection_index,
                    sample_index=len(records),
                    resolved_revision=resolved_revision,
                )
            )
    if len(records) != QUESTION_TOTAL:
        raise ValueError("materialized selection must contain exactly 500 questions")
    return tuple(records)


def _build_question_manifest(
    records: Sequence[Mapping[str, Any]], *, resolved_revision: str
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_name": "part1_question_manifest",
        "schema_version": "1.1.0",
        "question_manifest_hash": "",
        "manifest_format_version": "jsonl-v1",
        "source_repository": DATASET_REPOSITORY,
        "source_revision": resolved_revision,
        "source_config_strategy": "per_subject",
        "source_configs": list(FIXED_SUBJECTS),
        "source_split": DATASET_SPLIT,
        "subjects": list(FIXED_SUBJECTS),
        "quota_per_subject": QUESTION_QUOTA_PER_SUBJECT,
        "total_count": QUESTION_TOTAL,
        "question_sampling_seed": QUESTION_SAMPLING_SEED,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "canonicalization_version": CANONICAL_JSON_VERSION,
        "ordered_record_aggregation": ORDERED_RECORD_AGGREGATION,
        "logical_filename": "questions.jsonl",
    }
    manifest["question_manifest_hash"] = question_manifest_hash(manifest, records)
    validate_instance("question_manifest", manifest)
    return manifest


def _build_study_manifest(question_manifest: Mapping[str, Any]) -> dict[str, Any]:
    study: dict[str, Any] = {
        "schema_name": "part1_study_manifest",
        "schema_version": "1.1.0",
        "study_id": "",
        "study_manifest_hash": "",
        "question_source_repository": question_manifest["source_repository"],
        "question_source_revision": question_manifest["source_revision"],
        "question_manifest_hash": question_manifest["question_manifest_hash"],
        **copy.deepcopy(FIXED_STUDY_CONTRACT),
    }
    study["study_id"] = study_id(study)
    study["study_manifest_hash"] = study_manifest_hash(study)
    validate_instance("study_manifest", study)
    validate_fixed_study_contract(study)
    return study


def build_manifest_bundle(
    selected_by_subject: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    resolved_revision: str,
) -> ManifestBundle:
    """Build and fully validate the final in-memory manifest content."""

    revision = require_immutable_revision(resolved_revision)
    records = _build_records(selected_by_subject, resolved_revision=revision)
    question_manifest = _build_question_manifest(records, resolved_revision=revision)
    # The study is derived only after the complete question bundle has its final
    # hash and has passed record/sidecar validation.
    provisional = ManifestBundle(records, question_manifest, {})
    _validate_question_bundle(provisional)
    study_manifest = _build_study_manifest(question_manifest)
    bundle = ManifestBundle(records, question_manifest, study_manifest)
    validate_manifest_bundle(bundle)
    return bundle


def _validate_question_bundle(bundle: ManifestBundle) -> None:
    records = bundle.records
    manifest = bundle.question_manifest
    if len(records) != QUESTION_TOTAL:
        raise ValueError("question manifest must contain exactly 500 records")
    validate_instance("question_manifest", manifest)
    if manifest["source_revision"] != require_immutable_revision(manifest["source_revision"]):
        raise ValueError("question manifest revision is not immutable")

    ids: set[str] = set()
    row_identities: set[tuple[str, str, int]] = set()
    for sample_index, record in enumerate(records):
        validate_instance("question_record", record)
        subject_index, selection_index = divmod(sample_index, QUESTION_QUOTA_PER_SUBJECT)
        expected_subject = FIXED_SUBJECTS[subject_index]
        if record["sample_index"] != sample_index:
            raise ValueError("question sample_index order is not contiguous 0 through 499")
        if record["subject"] != expected_subject or record["source_config"] != expected_subject:
            raise ValueError("question records do not preserve the fixed subject block order")
        if record["subject_selection_index"] != selection_index:
            raise ValueError("question records do not preserve seeded selection order")
        if record["source_revision"] != manifest["source_revision"]:
            raise ValueError("question record revision differs from question manifest revision")
        expected_content_hash = question_content_hash(record)
        if record["question_content_hash"] != expected_content_hash:
            raise ValueError(f"record {sample_index} question content hash is invalid")
        expected_question_id = question_id(record)
        if record["question_id"] != expected_question_id:
            raise ValueError(f"record {sample_index} question ID is invalid")
        if record["question_id"] in ids:
            raise ValueError(f"duplicate stable question ID at sample_index {sample_index}")
        ids.add(record["question_id"])
        source_identity = record["source_row_identity"]
        expected_identity = (
            record["source_config"],
            record["source_split"],
            source_identity.get("row_index"),
        )
        if source_identity != {
            "config": expected_identity[0],
            "split": expected_identity[1],
            "row_index": expected_identity[2],
        }:
            raise ValueError("source row identity must contain exact config/split/row_index")
        if expected_identity in row_identities:
            raise ValueError(f"duplicate source row identity at sample_index {sample_index}")
        row_identities.add(expected_identity)

    recomputed = question_manifest_hash(manifest, records)
    if manifest["question_manifest_hash"] != recomputed:
        raise ValueError("question manifest hash differs from finalized records")


def validate_manifest_bundle(bundle: ManifestBundle) -> None:
    _validate_question_bundle(bundle)
    study = bundle.study_manifest
    validate_instance("study_manifest", study)
    validate_fixed_study_contract(study)
    if study["question_source_repository"] != bundle.question_manifest["source_repository"]:
        raise ValueError("study question source repository differs from question manifest")
    if study["question_source_revision"] != bundle.question_manifest["source_revision"]:
        raise ValueError("study question source revision differs from question manifest")
    if study["question_manifest_hash"] != bundle.question_manifest["question_manifest_hash"]:
        raise ValueError("study question manifest hash differs from finalized question manifest")
    if study["study_id"] != study_id(study):
        raise ValueError("study ID differs from recomputed identity")
    if study["study_manifest_hash"] != study_manifest_hash(study):
        raise ValueError("study manifest hash differs from recomputed complete hash")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def manifest_bytes(bundle: ManifestBundle) -> dict[str, bytes]:
    validate_manifest_bundle(bundle)
    questions = b"".join(_json_bytes(record) for record in bundle.records)
    return {
        "questions": questions,
        "question_manifest": _json_bytes(bundle.question_manifest),
        "study_manifest": _json_bytes(bundle.study_manifest),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def write_staged_manifest_bundle(directory: Path, bundle: ManifestBundle) -> dict[str, Path]:
    payloads = manifest_bytes(bundle)
    paths = {
        "questions": directory / "questions.jsonl",
        "question_manifest": directory / "questions.manifest.json",
        "study_manifest": directory / "study_manifest.json",
    }
    directory.mkdir(parents=True, exist_ok=False)
    _fsync_directory(directory.parent)
    for key in _MANIFEST_KEYS:
        _durable_write_new(paths[key], payloads[key])
    return paths


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def load_manifest_bundle(
    *,
    questions_path: Path,
    question_manifest_path: Path,
    study_manifest_path: Path,
) -> ManifestBundle:
    raw = questions_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("questions JSONL must end with a newline")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid questions JSONL line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"questions JSONL line {line_number} must be an object")
        records.append(value)
    bundle = ManifestBundle(
        records=tuple(records),
        question_manifest=_load_json_object(question_manifest_path),
        study_manifest=_load_json_object(study_manifest_path),
    )
    validate_manifest_bundle(bundle)
    return bundle


def _validate_path_maps(staged: Mapping[str, Path], final: Mapping[str, Path]) -> None:
    expected = set(_MANIFEST_KEYS)
    if set(staged) != expected or set(final) != expected:
        raise ValueError("manifest path maps must contain questions, question_manifest, study_manifest")


def preflight_manifest_publication(
    staged: Mapping[str, Path], final: Mapping[str, Path]
) -> dict[str, str]:
    """Check the complete directory bundle before publication; mutate nothing."""

    _validate_path_maps(staged, final)
    staged_parents = {Path(path).parent for path in staged.values()}
    final_parents = {Path(path).parent for path in final.values()}
    if len(staged_parents) != 1 or len(final_parents) != 1:
        raise ValueError("all three manifest files must share one staged and final directory")
    staged_parent = next(iter(staged_parents))
    final_parent = next(iter(final_parents))
    expected_names = {Path(path).name for path in staged.values()}
    if expected_names != {Path(path).name for path in final.values()}:
        raise ValueError("staged and final manifest filenames must match exactly")
    if not staged_parent.is_dir() or staged_parent.is_symlink():
        raise ValueError(f"staged manifest bundle is not a regular directory: {staged_parent}")
    if {path.name for path in staged_parent.iterdir()} != expected_names:
        raise ValueError("staged manifest directory must contain exactly the three final files")
    if os.path.lexists(final_parent) and (
        not final_parent.is_dir() or final_parent.is_symlink()
    ):
        raise ManifestCompatibilityError(
            f"existing finalized manifest bundle is not a regular directory: {final_parent}"
        )
    if os.path.lexists(final_parent):
        final_names = {path.name for path in final_parent.iterdir()}
        extra_names = sorted(final_names.difference(expected_names))
        if extra_names:
            raise ManifestCompatibilityError(
                "existing finalized manifest directory contains extra entries: "
                + ", ".join(extra_names)
            )

    states: dict[str, str] = {}
    for key in _MANIFEST_KEYS:
        source = Path(staged[key])
        destination = Path(final[key])
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"staged {key} is not a regular file: {source}")
        if os.path.lexists(destination):
            if not destination.is_file() or destination.is_symlink():
                raise ManifestCompatibilityError(
                    f"existing finalized {key} is not a regular file: {destination}"
                )
            if destination.read_bytes() != source.read_bytes():
                raise ManifestCompatibilityError(
                    f"existing finalized {key} is incompatible with staged content: {destination}"
                )
            states[key] = "identical_existing"
        else:
            states[key] = "missing"
    existing = [key for key, state in states.items() if state == "identical_existing"]
    if existing and len(existing) != len(_MANIFEST_KEYS):
        raise ManifestCompatibilityError(
            "existing finalized manifest directory contains a partial complete bundle"
        )
    if not existing and os.path.lexists(final_parent):
        raise ManifestCompatibilityError(
            "existing finalized manifest directory does not contain the complete bundle"
        )
    return states


def publish_manifest_bundle(
    staged: Mapping[str, Path], final: Mapping[str, Path]
) -> dict[str, str]:
    """Publish all three files with one same-filesystem directory rename."""

    states = preflight_manifest_publication(staged, final)
    if all(state == "identical_existing" for state in states.values()):
        return states

    source_parent = Path(staged["questions"]).parent
    destination_parent = Path(final["questions"]).parent
    destination_parent.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination_parent):
        raise ManifestCompatibilityError(
            f"finalized manifest bundle appeared after preflight: {destination_parent}"
        )
    if source_parent.stat().st_dev != destination_parent.parent.stat().st_dev:
        raise OSError("staged and final manifest directories are on different filesystems")
    os.replace(source_parent, destination_parent)
    _fsync_directory(destination_parent.parent)
    for key in _MANIFEST_KEYS:
        states[key] = "published"
    return states
