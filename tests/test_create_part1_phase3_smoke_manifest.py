"""Identical-only Phase 3 smoke-manifest publication tests."""

from __future__ import annotations

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
        preflight_report=fixture["preflight"],
        repository_root=fixture["repository"],
    )
    inode = first.stat().st_ino
    second = publish_phase3_smoke_manifest(
        study_manifest=fixture["bundle"].study_manifest,
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
            preflight_report=fixture["preflight"],
            repository_root=fixture["repository"],
        )
