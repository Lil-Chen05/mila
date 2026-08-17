"""Validate and atomically publish one preserved Part 1 analysis stage."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

from part1_analysis import (
    PublicationStateIndeterminateError,
    _fsync_directory_descriptor,
    _same_inode,
    _stat_directory_name_at,
    _validate_analysis_directory_descriptor,
)


def finalize_preserved_analysis_stage(
    *,
    stage: Path,
    target_name: str,
    expected_model_run_id: str,
    expected_analysis_id: str,
    expected_bootstrap_replicates: int,
) -> dict[str, Any]:
    """Publish a fully written stage without recomputing its analysis."""

    stage = Path(os.path.abspath(stage))
    if (
        not target_name
        or target_name in {".", ".."}
        or "/" in target_name
        or stage.name.startswith(f".{target_name}.stage-") is False
    ):
        raise ValueError("analysis stage and target names are not canonical")
    root = stage.parent
    parent_descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    stage_descriptor = -1
    claim_name = f".{target_name}.publish-claim"
    claim_identity: tuple[int, int] | None = None
    renamed = False
    primary: BaseException | None = None
    try:
        stage_descriptor = os.open(
            stage.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        original_stage = os.fstat(stage_descriptor)
        manifest = _validate_analysis_directory_descriptor(stage_descriptor)
        if (
            manifest.get("model_run_id") != expected_model_run_id
            or manifest.get("analysis_id") != expected_analysis_id
            or manifest.get("bootstrap_replicates")
            != expected_bootstrap_replicates
            or manifest.get("paper_analysis_ready") is not True
        ):
            raise ValueError("preserved analysis stage identity differs")
        observed_stage = _stat_directory_name_at(
            parent_descriptor, stage.name, label="preserved analysis stage"
        )
        if not _same_inode(original_stage, observed_stage):
            raise PublicationStateIndeterminateError(
                "preserved analysis stage pathname changed during validation"
            )
        if os.path.lexists(root / target_name):
            raise FileExistsError(
                f"final analysis target already exists: {root / target_name}"
            )
        os.mkdir(claim_name, 0o700, dir_fd=parent_descriptor)
        claim_status = os.stat(
            claim_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISDIR(claim_status.st_mode):
            raise RuntimeError("analysis publication claim is not a directory")
        claim_identity = (claim_status.st_dev, claim_status.st_ino)
        if os.path.lexists(root / target_name):
            raise FileExistsError(
                f"final analysis target already exists: {root / target_name}"
            )
        _validate_analysis_directory_descriptor(
            stage_descriptor, expected_manifest=manifest
        )
        claimed_stage = _stat_directory_name_at(
            parent_descriptor, stage.name, label="claimed analysis stage"
        )
        if not _same_inode(original_stage, claimed_stage):
            raise PublicationStateIndeterminateError(
                "preserved analysis stage changed while publication claim was held"
            )
        os.rename(
            stage.name,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        renamed = True
        published = _stat_directory_name_at(
            parent_descriptor, target_name, label="published analysis"
        )
        if not _same_inode(original_stage, published):
            raise PublicationStateIndeterminateError(
                "published analysis does not name the validated stage inode"
            )
        _validate_analysis_directory_descriptor(
            stage_descriptor, expected_manifest=manifest
        )
        durable = _stat_directory_name_at(
            parent_descriptor, target_name, label="validated published analysis"
        )
        if not _same_inode(original_stage, durable):
            raise PublicationStateIndeterminateError(
                "published analysis pathname changed after final validation"
            )
        _fsync_directory_descriptor(parent_descriptor)
        return manifest
    except BaseException as exc:
        primary = exc
        if renamed and not isinstance(exc, PublicationStateIndeterminateError):
            raise PublicationStateIndeterminateError(
                "analysis stage recovery failed after atomic rename; published paths "
                "were preserved for inspection"
            ) from exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        if claim_identity is not None:
            try:
                current = os.stat(
                    claim_name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != claim_identity:
                    raise RuntimeError("analysis publication claim identity changed")
                os.rmdir(claim_name, dir_fd=parent_descriptor)
                _fsync_directory_descriptor(parent_descriptor)
            except BaseException as exc:
                cleanup_error = exc
        if stage_descriptor >= 0:
            os.close(stage_descriptor)
        os.close(parent_descriptor)
        if cleanup_error is not None:
            if primary is not None:
                primary.add_note(
                    "analysis publication claim cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise cleanup_error


__all__ = ["finalize_preserved_analysis_stage"]
