"""Identical-only Phase 3 smoke-manifest publication tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from test_run_part1_shard import production_fixture


def test_phase3_smoke_manifest_is_clean_bounded_and_identical_only(
    tmp_path: Path,
) -> None:
    from create_part1_phase3_smoke_manifest import publish_phase3_smoke_manifest

    fixture = production_fixture(tmp_path)
    first = publish_phase3_smoke_manifest(
        study_manifest=fixture["bundle"].study_manifest,
        question_manifest=fixture["bundle"].question_manifest,
        preflight_report=fixture["preflight"],
        repository_root=fixture["repository"],
    )
    inode = first.stat().st_ino
    second = publish_phase3_smoke_manifest(
        study_manifest=fixture["bundle"].study_manifest,
        question_manifest=fixture["bundle"].question_manifest,
        preflight_report=fixture["preflight"],
        repository_root=fixture["repository"],
    )
    manifest = json.loads(first.read_text(encoding="utf-8"))

    assert first == fixture["repository"] / (
        "results/part1-smoke/model-runs/phase3_smoke/model_run_manifest.json"
    )
    assert second.stat().st_ino == inode
    assert manifest["production"] is False
    assert manifest["execution_scope"] == "phase3_smoke"
    assert manifest["smoke_git_provenance"]["base_commit"] == fixture["manifest"][
        "final_production_git_commit"
    ]
    assert manifest["smoke_git_provenance"]["diff_hash"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    first.write_text('{"divergent":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        publish_phase3_smoke_manifest(
            study_manifest=fixture["bundle"].study_manifest,
            question_manifest=fixture["bundle"].question_manifest,
            preflight_report=fixture["preflight"],
            repository_root=fixture["repository"],
        )


def test_phase3_smoke_manifest_refuses_dirty_tracked_state(tmp_path: Path) -> None:
    from create_part1_phase3_smoke_manifest import publish_phase3_smoke_manifest

    fixture = production_fixture(tmp_path)
    tracked = fixture["repository"] / "manifests" / "part1" / "study_manifest.json"
    tracked.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        publish_phase3_smoke_manifest(
            study_manifest=fixture["bundle"].study_manifest,
            question_manifest=fixture["bundle"].question_manifest,
            preflight_report=fixture["preflight"],
            repository_root=fixture["repository"],
        )


@pytest.mark.parametrize(
    "field",
    ["study_id", "study_manifest_hash", "question_manifest_hash"],
)
def test_phase3_smoke_manifest_refuses_incompatible_preflight_without_partial_output(
    tmp_path: Path, field: str
) -> None:
    from create_part1_phase3_smoke_manifest import publish_phase3_smoke_manifest

    fixture = production_fixture(tmp_path)
    preflight = copy.deepcopy(fixture["preflight"])
    preflight[field] = "0" * 64
    target = fixture["repository"] / (
        "results/part1-smoke/model-runs/phase3_smoke/model_run_manifest.json"
    )

    with pytest.raises(ValueError, match=field):
        publish_phase3_smoke_manifest(
            study_manifest=fixture["bundle"].study_manifest,
            question_manifest=fixture["bundle"].question_manifest,
            preflight_report=preflight,
            repository_root=fixture["repository"],
        )

    assert not target.exists()
    if target.parent.exists():
        assert list(target.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("case", ["default", "relocated", "explicit"])
def test_cli_repository_root_relocates_only_unspecified_input_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    import create_part1_phase3_smoke_manifest as creator

    fixture = production_fixture(tmp_path)
    selected_repository = (
        creator.REPOSITORY_ROOT if case == "default" else fixture["repository"]
    )
    expected_manifest_root = (
        tmp_path / "explicit-manifests"
        if case == "explicit"
        else selected_repository / "manifests/part1"
    )
    expected_preflight = (
        tmp_path / "explicit-preflight.json"
        if case == "explicit"
        else selected_repository / "results/part1-smoke/preflight/preflight.json"
    )
    output = fixture["repository"] / "synthetic-model-run-manifest.json"
    output.write_text('{"model_run_id":"synthetic"}\n', encoding="utf-8")
    seen: dict[str, Path] = {}
    original_load_json = creator._load_json

    def capture_bundle(*, questions_path, question_manifest_path, study_manifest_path):
        seen["questions"] = questions_path
        seen["question_manifest"] = question_manifest_path
        seen["study_manifest"] = study_manifest_path
        return fixture["bundle"]

    def capture_json(path):
        if "preflight" not in seen:
            seen["preflight"] = Path(path)
            return fixture["preflight"]
        return original_load_json(path)

    monkeypatch.setattr(creator, "load_manifest_bundle", capture_bundle)
    monkeypatch.setattr(creator, "_load_json", capture_json)
    monkeypatch.setattr(
        creator, "publish_phase3_smoke_manifest", lambda **_kwargs: output
    )

    arguments = (
        []
        if case == "default"
        else ["--repository-root", str(fixture["repository"])]
    )
    if case == "explicit":
        arguments.extend(
            [
                "--manifest-root",
                str(expected_manifest_root),
                "--preflight",
                str(expected_preflight),
            ]
        )
    result = creator.main(arguments)

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["is_valid"] is True
    assert seen == {
        "questions": expected_manifest_root / "questions.jsonl",
        "question_manifest": expected_manifest_root / "questions.manifest.json",
        "study_manifest": expected_manifest_root / "study_manifest.json",
        "preflight": expected_preflight,
    }
