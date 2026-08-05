"""Production model-run publication and clean-worktree gate tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test_part1_model_run import preflight, study


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def initialized_repository(tmp_path: Path) -> tuple[Path, str, dict]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Part 1 Test")
    _git(repository, "config", "user.email", "part1-test@example.invalid")
    (repository / "scripts").mkdir()
    (repository / "scripts" / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    lock_bytes = b"synthetic uv lock\n"
    (repository / "uv.lock").write_bytes(lock_bytes)
    (repository / ".gitignore").write_text("results/part1/\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "scripts/tracked.py", "uv.lock")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    report = copy.deepcopy(preflight())
    report["environment_versions"]["uv_lock_sha256"] = hashlib.sha256(
        lock_bytes
    ).hexdigest()
    return repository, head, report


def test_publisher_creates_valid_manifest_beneath_canonical_ignored_root(
    tmp_path: Path,
) -> None:
    from create_part1_model_run_manifest import publish_production_model_run_manifest

    repository, head, report = initialized_repository(tmp_path)

    path = publish_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=report,
        final_git_commit=head,
        output_root=Path("results/part1"),
        repository_root=repository,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert path == repository / "results" / "part1" / manifest["model_run_id"] / (
        "model_run_manifest.json"
    )
    assert path.read_bytes().endswith(b"\n")
    assert manifest["final_production_git_commit"] == head
    assert manifest["dependency_lock_sha256"] == hashlib.sha256(
        (repository / "uv.lock").read_bytes()
    ).hexdigest()


def test_publisher_rejects_symlinked_output_root(tmp_path: Path) -> None:
    from create_part1_model_run_manifest import publish_production_model_run_manifest

    repository, head, report = initialized_repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "results").mkdir()
    (repository / "results" / "part1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink|regular directory"):
        publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=report,
            final_git_commit=head,
            output_root=Path("results/part1"),
            repository_root=repository,
        )

    assert list(outside.iterdir()) == []


def test_publisher_rejects_partial_existing_run_directory(tmp_path: Path) -> None:
    from create_part1_model_run_manifest import publish_production_model_run_manifest
    from part1_model_run import build_production_model_run_manifest

    repository, head, report = initialized_repository(tmp_path)
    manifest = build_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=report,
        final_git_commit=head,
        output_root=Path("results/part1"),
    )
    run_root = repository / "results" / "part1" / manifest["model_run_id"]
    run_root.mkdir(parents=True)
    marker = run_root / "partial-state.json"
    marker.write_text("do not modify\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial"):
        publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=report,
            final_git_commit=head,
            output_root=Path("results/part1"),
            repository_root=repository,
        )

    assert marker.read_text(encoding="utf-8") == "do not modify\n"
    assert not (run_root / "model_run_manifest.json").exists()


def test_cli_reports_unrelated_untracked_paths_without_modifying_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from create_part1_model_run_manifest import main

    repository, head, report = initialized_repository(tmp_path)
    study_path = repository / "study_manifest.json"
    preflight_path = repository / "preflight.json"
    study_path.write_text(json.dumps(study()), encoding="utf-8")
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    unrelated = repository / "operator-notes.txt"
    unrelated.write_text("leave me alone\n", encoding="utf-8")

    exit_code = main(
        [
            "--repository-root",
            str(repository),
            "--study-manifest",
            str(study_path),
            "--preflight",
            str(preflight_path),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["unrelated_untracked_paths"] == [
        "operator-notes.txt",
        "preflight.json",
        "study_manifest.json",
    ]
    assert unrelated.read_text(encoding="utf-8") == "leave me alone\n"
    assert _git(repository, "rev-parse", "HEAD").stdout.strip() == head
    assert _git(repository, "rev-list", "--count", "HEAD").stdout.strip() == "1"


@pytest.mark.parametrize("state", ["unstaged", "staged", "scoped_untracked"])
def test_publisher_rejects_dirty_or_execution_relevant_git_state(
    tmp_path: Path, state: str
) -> None:
    from create_part1_model_run_manifest import publish_production_model_run_manifest

    repository, head, report = initialized_repository(tmp_path)
    if state in {"unstaged", "staged"}:
        (repository / "scripts" / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
        if state == "staged":
            _git(repository, "add", "scripts/tracked.py")
    else:
        (repository / "jobs").mkdir()
        (repository / "jobs" / "new.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean tracked worktree|execution-relevant"):
        publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=report,
            final_git_commit=head,
            repository_root=repository,
        )


def test_publisher_rejects_non_head_commit_and_dependency_lock_mismatch(
    tmp_path: Path,
) -> None:
    from create_part1_model_run_manifest import publish_production_model_run_manifest

    repository, head, report = initialized_repository(tmp_path)
    with pytest.raises(RuntimeError, match="differs from recorded"):
        publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=report,
            final_git_commit="0" * 40,
            repository_root=repository,
        )

    mismatched = copy.deepcopy(report)
    mismatched["environment_versions"]["uv_lock_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="dependency lock hash"):
        publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=mismatched,
            final_git_commit=head,
            repository_root=repository,
        )


def test_publication_is_identical_only_and_preserves_existing_inode(tmp_path: Path) -> None:
    from create_part1_model_run_manifest import publish_production_model_run_manifest

    repository, head, report = initialized_repository(tmp_path)
    first = publish_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=report,
        final_git_commit=head,
        repository_root=repository,
    )
    inode = first.stat().st_ino
    second = publish_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=report,
        final_git_commit=head,
        repository_root=repository,
    )
    assert second == first
    assert second.stat().st_ino == inode

    first.write_text('{"divergent":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=report,
            final_git_commit=head,
            repository_root=repository,
        )


def test_publisher_rechecks_git_immediately_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import create_part1_model_run_manifest as module

    repository, head, report = initialized_repository(tmp_path)
    real_check = module._require_clean_git_state
    calls = 0

    def dirty_after_first_check(root: Path, expected_head: str) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        result = real_check(root, expected_head)
        if calls == 1:
            (repository / "scripts" / "tracked.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(module, "_require_clean_git_state", dirty_after_first_check)
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        module.publish_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=report,
            final_git_commit=head,
            repository_root=repository,
        )
    assert calls == 2
    assert list((repository / "results" / "part1").rglob("*.tmp")) == []
    assert list((repository / "results" / "part1").rglob("model_run_manifest.json")) == []
