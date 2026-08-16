"""Focused regressions for the preserved production merge-stage recovery."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import pytest


def _manifest_with_runtime_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    import part1_merge
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    _fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    _target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    assert isinstance(manifest, dict)
    model_run_id = manifest["model_run_id"]
    for index in range(500):
        shard_id = f"shard-{index:03d}"
        manifest["source_files"].append(
            {
                "relative_path": (
                    f"results/part1/{model_run_id}/raw_shards/{shard_id}/.writer.guard"
                ),
                "shard_id": shard_id,
                "kind": "runtime_guard",
                "state": "regular_file",
                "sha256": hashlib.sha256(b"").hexdigest(),
                "byte_size": 0,
            }
        )
    manifest["source_files"].sort(key=lambda item: item["relative_path"])
    manifest["merge_id"] = part1_merge.merge_id(manifest)
    manifest["merge_manifest_hash"] = part1_merge.merge_manifest_hash(manifest)
    return manifest


def test_merge_manifest_allows_only_runtime_guard_regular_files_to_be_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge

    manifest = _manifest_with_runtime_guards(tmp_path, monkeypatch)
    part1_merge.validate_merge_manifest(manifest)

    changed = copy.deepcopy(manifest)
    source = next(item for item in changed["source_files"] if item["kind"] == "natural_results")
    source["sha256"] = hashlib.sha256(b"").hexdigest()
    source["byte_size"] = 0
    changed["merge_id"] = part1_merge.merge_id(changed)
    changed["merge_manifest_hash"] = part1_merge.merge_manifest_hash(changed)
    with pytest.raises(ValueError, match="nonempty"):
        part1_merge.validate_merge_manifest(changed)

    for byte_size, sha256 in ((1, hashlib.sha256(b"").hexdigest()), (0, "f" * 64)):
        changed = copy.deepcopy(manifest)
        guard = next(item for item in changed["source_files"] if item["kind"] == "runtime_guard")
        guard["byte_size"] = byte_size
        guard["sha256"] = sha256
        changed["merge_id"] = part1_merge.merge_id(changed)
        changed["merge_manifest_hash"] = part1_merge.merge_manifest_hash(changed)
        with pytest.raises(ValueError, match="runtime_guard"):
            part1_merge.validate_merge_manifest(changed)


def test_cleanup_failure_never_masks_active_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    _fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)

    def fail_primary(boundary: str) -> None:
        if boundary == "table_writes_complete":
            raise RuntimeError("primary-validation-failure")

    monkeypatch.setattr(
        part1_merge,
        "_remove_own_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup-einval")),
    )
    with pytest.raises(RuntimeError, match="primary-validation-failure") as captured:
        part1_merge.publish_merge(inputs, fault_hook=fail_primary)
    assert any("cleanup-einval" in note for note in captured.value.__notes__)
    assert list(
        (
            inputs.repository_root
            / inputs.model_manifest["output_paths"]["merged"]
        ).parent.glob(".merged.stage-*")
    )


def test_claimed_stage_publish_is_atomic_no_overwrite_and_validates_exact_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    from part1_merge_stage_recovery import publish_claimed_validated_stage
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    stage = target.with_name(".merged.stage-preserved")
    target.rename(stage)
    row_counts = {
        kind: summary["row_count"] for kind, summary in manifest["outputs"].items()
    }

    published = publish_claimed_validated_stage(
        stage=stage,
        target=target,
        expected_manifest=manifest,
        expected_row_counts=row_counts,
        revalidate=lambda: None,
    )
    assert published == target
    assert not stage.exists()
    assert part1_merge.validate_merge_directory(target) == manifest

    replacement = target.with_name(".merged.stage-preserved")
    replacement.mkdir()
    (replacement / "marker").write_text("incompatible", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(ValueError, match="stage|contents"):
        publish_claimed_validated_stage(
            stage=replacement,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
        )
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_claimed_stage_publish_rejects_competing_claim_and_target_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    from part1_merge_stage_recovery import publish_claimed_validated_stage
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    stage = target.with_name(".merged.stage-preserved")
    target.rename(stage)
    row_counts = {
        kind: summary["row_count"] for kind, summary in manifest["outputs"].items()
    }
    claim = target.with_name(".merged.publish-claim")
    claim.mkdir()
    with pytest.raises(FileExistsError, match="claim"):
        publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
        )
    assert stage.is_dir()
    claim.rmdir()

    def create_racer() -> None:
        target.mkdir()
        (target / "winner").write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="target"):
        publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
            before_final_absence_check=create_racer,
        )
    assert (target / "winner").read_bytes() == b"do-not-overwrite"
    assert stage.is_dir()


def test_claimed_publish_keeps_validated_stage_fd_across_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    from part1_merge_stage_recovery import publish_claimed_validated_stage
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    _fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    stage = target.with_name(".merged.stage-preserved")
    target.rename(stage)
    held = stage.with_name("held-validated-stage")
    row_counts = {kind: item["row_count"] for kind, item in manifest["outputs"].items()}

    def substitute() -> None:
        stage.rename(held)
        stage.mkdir()
        (stage / "replacement").write_bytes(b"unvalidated")

    with pytest.raises(RuntimeError, match="identity"):
        publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
            after_stage_validation=substitute,
        )
    assert not target.exists()
    assert (stage / "replacement").read_bytes() == b"unvalidated"
    assert held.is_dir()


def test_post_rename_failure_is_indeterminate_and_never_name_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    import part1_merge_stage_recovery as recovery
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    _fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    stage = target.with_name(".merged.stage-preserved")
    target.rename(stage)
    row_counts = {kind: item["row_count"] for kind, item in manifest["outputs"].items()}
    monkeypatch.setattr(
        recovery,
        "_fsync_directory_descriptor",
        lambda _descriptor: (_ for _ in ()).throw(OSError("gpfs-fsync")),
    )
    with pytest.raises(part1_merge.PublicationStateIndeterminateError, match="indeterminate"):
        recovery.publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
        )
    assert target.is_dir()
    assert not stage.exists()


def test_first_post_rename_lookup_failure_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    import part1_merge_stage_recovery as recovery
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    _fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    stage = target.with_name(".merged.stage-preserved")
    target.rename(stage)
    row_counts = {kind: item["row_count"] for kind, item in manifest["outputs"].items()}
    real_stat = recovery.os.stat
    real_rename = recovery.os.rename
    renamed = False

    def track_rename(*args, **kwargs):
        nonlocal renamed
        result = real_rename(*args, **kwargs)
        renamed = True
        return result

    def fail_first_target_stat(path, *args, **kwargs):
        if renamed and path == target.name:
            raise OSError("post-rename-lookup")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "rename", track_rename)
    monkeypatch.setattr(recovery.os, "stat", fail_first_target_stat)
    with pytest.raises(part1_merge.PublicationStateIndeterminateError, match="indeterminate"):
        recovery.publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
        )
    assert target.is_dir()
    assert not stage.exists()


def test_rename_error_with_ambiguous_paths_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import part1_merge
    import part1_merge_stage_recovery as recovery
    from test_merge_part1_results import _disable_snapshot_recheck, _prepared

    _fixture, _shard, inputs, _raw = _prepared(tmp_path)
    _disable_snapshot_recheck(monkeypatch)
    target, manifest = part1_merge.publish_merge(inputs, return_manifest=True)
    stage = target.with_name(".merged.stage-preserved")
    target.rename(stage)
    row_counts = {kind: item["row_count"] for kind, item in manifest["outputs"].items()}
    real_rename = recovery.os.rename

    def ambiguous_rename(src, dst, *, src_dir_fd, dst_dir_fd):
        real_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        raise OSError("gpfs-ambiguous-rename")

    monkeypatch.setattr(recovery.os, "rename", ambiguous_rename)
    with pytest.raises(part1_merge.PublicationStateIndeterminateError, match="indeterminate"):
        recovery.publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=row_counts,
            revalidate=lambda: None,
        )
    assert target.is_dir()
    assert not stage.exists()


def test_stage_recovery_launcher_is_exact_two_job_afterok_chain() -> None:
    text = Path("scripts/submit_part1_merge_stage_recovery.py").read_text(
        encoding="utf-8"
    )
    assert "jobs/part1_recover_merge_stage.sh" in text
    assert "jobs/part1_analyze_merge_stage_recovery.sh" in text
    assert "afterok:{jobs['finalize']}" in text
    assert "10383206" in text and "10383207" in text
    assert ".merged.stage-ri97qy41" in text
    assert "part1_generate_array" not in text
    assert "part1_merge_prompt_hash_waiver" not in text
