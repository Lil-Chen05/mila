"""Validate and cooperatively publish an already complete merge stage.

The claim coordinates every recovery publisher in this codebase.  It is not a
kernel-enforced no-replace primitive against an unrelated non-cooperating
writer; the recovery launcher separately requires no competing merge job.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from part1_contract import (
    canonical_json_bytes,
    model_run_id,
    model_run_manifest_hash,
    validate_fixed_model_requested_contract,
    validate_instance,
)
from part1_merge import (
    PublicationStateIndeterminateError,
    _load_regular_json,
    _validate_merge_directory_descriptor,
    _fsync_directory_descriptor,
    validate_merge_directory,
    validate_merge_directory_at,
)
from part1_prompt_hash_waiver import (
    AUTHORIZED_GENERATION_GIT_COMMIT,
    AUTHORIZED_MODEL_RUN_ID,
    canonical_waiver_bytes,
    require_exact_failed_report,
    require_production_checkout_generation_state,
    validate_prompt_hash_waiver,
)


AUTHORIZED_ORIGINAL_RECOVERY_COMMIT = "1a4b6039758cd8fd84b68f74c0828e6c5f382dae"
AUTHORIZED_STAGE_BASENAME = ".merged.stage-ri97qy41"
AUTHORIZED_ROW_COUNTS = {
    "natural_results": 5000,
    "checkpoint_results": 55000,
    "audit_events": 120003,
}
AUTHORIZED_STAGE_MANIFEST_SHA256 = "16ad6ae082af664a3f3afecee83568f340e839ad2220117e3f03391c4bd10509"
AUTHORIZED_STAGE_MANIFEST_BYTE_SIZE = 921804
AUTHORIZED_MERGE_ID = "447cfc9125349369f24b3e0e6865c254b516ceb84c70263d9f0a0e36801938e6"
AUTHORIZED_MERGE_MANIFEST_HASH = "a2f47af9378a6906c64f4f0ea9ae76d9d2f41c67be913b7c9cca5fe63dcbce03"
AUTHORIZED_OUTPUTS = {
    "audit_events": {
        "sha256": "ac72306de70170ba6a9c8e32e83b6008903e009f9ed91ca1fcdde5cc029d5cd2",
        "byte_size": 21515936,
        "row_count": 120003,
    },
    "checkpoint_results": {
        "sha256": "16dc510a2ea214f6225f3efec48b927bebaee279788cbd61e7d41b12db63f4f6",
        "byte_size": 26980587,
        "row_count": 55000,
    },
    "natural_results": {
        "sha256": "25f1a61104b17fa085fc16c2eb13df67cb01e6477d59d5417d0146c3127986d3",
        "byte_size": 113580101,
        "row_count": 5000,
    },
}
RECOVERY_SIDECAR_NAME = "merge_stage_recovery.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def merge_stage_recovery_id(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload["merge_stage_recovery_id"] = ""
    return _sha256(
        canonical_json_bytes(
            {
                "identity_type": "part1_merge_stage_recovery",
                "identity_version": "part1-merge-stage-recovery-v1",
                "payload": payload,
            }
        )
    )


def validate_merge_stage_recovery(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_name", "schema_version", "merge_stage_recovery_id",
        "publication_mode", "model_run_id", "generation_git_commit",
        "original_merge_recovery_commit", "publication_recovery_commit",
        "prompt_hash_waiver", "coverage_report", "preserved_stage_relative_path",
        "published_directory_relative_path", "merge_manifest", "outputs",
        "source_inventory_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("merge-stage recovery sidecar fields differ")
    if (
        value["schema_name"] != "part1_merge_stage_recovery"
        or value["schema_version"] != "1.0.0"
        or value["publication_mode"] != "cooperative_claim_same_parent_atomic_rename_v1"
        or value["model_run_id"] != AUTHORIZED_MODEL_RUN_ID
        or value["generation_git_commit"] != AUTHORIZED_GENERATION_GIT_COMMIT
        or value["original_merge_recovery_commit"] != AUTHORIZED_ORIGINAL_RECOVERY_COMMIT
        or value["preserved_stage_relative_path"]
        != f"results/part1/{AUTHORIZED_MODEL_RUN_ID}/{AUTHORIZED_STAGE_BASENAME}"
        or value["published_directory_relative_path"]
        != f"results/part1/{AUTHORIZED_MODEL_RUN_ID}/merged"
        or value["merge_stage_recovery_id"] != merge_stage_recovery_id(value)
    ):
        raise ValueError("merge-stage recovery sidecar fixed identity differs")
    waiver = value["prompt_hash_waiver"]
    coverage = value["coverage_report"]
    merge = value["merge_manifest"]
    if (
        not isinstance(waiver, Mapping)
        or set(waiver) != {"waiver_id", "relative_path", "sha256", "byte_size"}
        or not isinstance(coverage, Mapping)
        or set(coverage) != {"validation_report_id", "relative_path", "sha256", "byte_size"}
        or not isinstance(merge, Mapping)
        or set(merge) != {"merge_id", "merge_manifest_hash", "relative_path", "sha256", "byte_size"}
    ):
        raise ValueError("merge-stage recovery nested provenance fields differ")
    if (
        coverage["validation_report_id"]
        != "2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db"
        or coverage["relative_path"]
        != f"results/part1/{AUTHORIZED_MODEL_RUN_ID}/validation/coverage_report.json"
        or waiver["relative_path"]
        != f"results/part1/{AUTHORIZED_MODEL_RUN_ID}/validation/prompt_hash_waiver.json"
        or merge["relative_path"]
        != f"results/part1/{AUTHORIZED_MODEL_RUN_ID}/merged/merge_manifest.json"
        or merge["merge_id"] != AUTHORIZED_MERGE_ID
        or merge["merge_manifest_hash"] != AUTHORIZED_MERGE_MANIFEST_HASH
        or merge["sha256"] != AUTHORIZED_STAGE_MANIFEST_SHA256
        or merge["byte_size"] != AUTHORIZED_STAGE_MANIFEST_BYTE_SIZE
    ):
        raise ValueError("merge-stage recovery exact artifact identity differs")
    sha_values = (
        value["merge_stage_recovery_id"], value["source_inventory_sha256"],
        waiver["waiver_id"], waiver["sha256"], coverage["sha256"], merge["sha256"],
    )
    if any(
        not isinstance(item, str) or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in sha_values
    ):
        raise ValueError("merge-stage recovery SHA-256 value is invalid")
    if any(
        type(item["byte_size"]) is not int or item["byte_size"] <= 0
        for item in (waiver, coverage, merge)
    ):
        raise ValueError("merge-stage recovery artifact byte size is invalid")
    if (
        not isinstance(value["publication_recovery_commit"], str)
        or len(value["publication_recovery_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in value["publication_recovery_commit"])
    ):
        raise ValueError("merge-stage publication Git commit is invalid")
    if not isinstance(value["outputs"], Mapping) or set(value["outputs"]) != {
        "natural_results", "checkpoint_results", "audit_events"
    }:
        raise ValueError("merge-stage recovery output inventory differs")
    if {
        kind: item.get("row_count") for kind, item in value["outputs"].items()
    } != AUTHORIZED_ROW_COUNTS:
        raise ValueError("merge-stage recovery output counts differ")
    for kind, exact in AUTHORIZED_OUTPUTS.items():
        item = value["outputs"][kind]
        if not isinstance(item, Mapping) or any(
            item.get(field) != expected for field, expected in exact.items()
        ):
            raise ValueError(f"merge-stage recovery {kind} exact evidence differs")


def _write_sidecar(path: Path, value: Mapping[str, Any]) -> None:
    validate_merge_stage_recovery(value)
    data = _json_bytes(value)
    if os.path.lexists(path):
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("existing merge-stage recovery sidecar is not regular")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            if b"".join(chunks) != data:
                raise ValueError("existing merge-stage recovery sidecar is incompatible")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ValueError("existing merge-stage recovery sidecar is incompatible")
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _row_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {
        kind: int(summary["row_count"])
        for kind, summary in manifest["outputs"].items()
    }


def _git_state(path: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path,
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    return head, dirty


def recover_authorized_merge_stage(
    *, repository_root: Path, model_run_manifest_path: Path
) -> tuple[Path, Path, dict[str, Any]]:
    """Publish only the one preserved production stage authorized by the user."""

    repository_root = Path(os.path.abspath(repository_root))
    manifest_path = Path(model_run_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = repository_root / manifest_path
    manifest_path = Path(os.path.abspath(manifest_path))
    model, model_bytes = _load_regular_json(
        manifest_path, label="production model-run manifest"
    )
    validate_instance("model_run_manifest", model)
    validate_fixed_model_requested_contract(model)
    if (
        model.get("schema_version") != "1.1.0"
        or model.get("production") is not True
        or model.get("execution_scope") != "production"
        or model.get("model_run_id") != AUTHORIZED_MODEL_RUN_ID
        or model.get("model_run_id") != model_run_id(model)
        or model.get("model_run_manifest_hash") != model_run_manifest_hash(model)
        or model.get("final_production_git_commit") != AUTHORIZED_GENERATION_GIT_COMMIT
    ):
        raise ValueError("model manifest differs from the authorized production run")
    expected_manifest_path = (
        repository_root / "results" / "part1" / AUTHORIZED_MODEL_RUN_ID
        / "model_run_manifest.json"
    )
    if manifest_path != expected_manifest_path:
        raise ValueError("model manifest path is not canonical")
    require_production_checkout_generation_state(
        repository_root,
        expected_generation_commit=AUTHORIZED_GENERATION_GIT_COMMIT,
    )

    validation_directory = manifest_path.parent / "validation"
    waiver_path = validation_directory / "prompt_hash_waiver.json"
    report_path = validation_directory / "coverage_report.json"
    waiver, waiver_bytes = _load_regular_json(waiver_path, label="prompt-hash waiver")
    report, report_bytes = _load_regular_json(report_path, label="coverage report")
    validate_prompt_hash_waiver(waiver)
    require_exact_failed_report(report, report_bytes=report_bytes, model_manifest=model)
    if (
        waiver.get("recovery_git_commit") != AUTHORIZED_ORIGINAL_RECOVERY_COMMIT
        or waiver.get("model_run_id") != model.get("model_run_id")
        or waiver.get("model_run_manifest_hash") != model.get("model_run_manifest_hash")
        or waiver.get("generation_git_commit") != model.get("final_production_git_commit")
        or waiver.get("coverage_report", {}).get("validation_report_id")
        != report.get("validation_report_id")
        or waiver.get("coverage_report", {}).get("sha256") != _sha256(report_bytes)
        or waiver.get("coverage_report", {}).get("byte_size") != len(report_bytes)
    ):
        raise ValueError("waiver does not bind the original merge recovery commit")

    target = repository_root / model["output_paths"]["merged"]
    stage = target.with_name(AUTHORIZED_STAGE_BASENAME)
    stage_exists = os.path.lexists(stage)
    target_exists = os.path.lexists(target)
    if stage_exists and target_exists:
        raise ValueError("both preserved stage and canonical target exist")
    if not stage_exists and not target_exists:
        raise ValueError("preserved stage and canonical target are both absent")
    merge_directory = target if target_exists else stage
    manifest, merge_manifest_bytes = _load_regular_json(
        merge_directory / "merge_manifest.json", label="preserved merge manifest"
    )
    if (
        _sha256(merge_manifest_bytes) != AUTHORIZED_STAGE_MANIFEST_SHA256
        or len(merge_manifest_bytes) != AUTHORIZED_STAGE_MANIFEST_BYTE_SIZE
        or manifest.get("merge_id") != AUTHORIZED_MERGE_ID
        or manifest.get("merge_manifest_hash") != AUTHORIZED_MERGE_MANIFEST_HASH
    ):
        raise ValueError("preserved merge manifest differs from exact observed evidence")
    if target_exists:
        validate_merge_directory(target, expected_manifest=manifest)
    if manifest.get("model_run_id") != AUTHORIZED_MODEL_RUN_ID:
        raise ValueError("preserved merge stage model-run identity differs")
    for field in (
        "study_id", "study_manifest_hash", "question_manifest_hash",
        "model_run_id", "model_run_manifest_hash",
    ):
        if manifest.get(field) != model.get(field):
            raise ValueError(f"preserved merge/model provenance differs for {field}")
    if canonical_json_bytes(manifest.get("source_files")) != canonical_json_bytes(
        report.get("source_files")
    ):
        raise ValueError("preserved merge source inventory differs from coverage report")
    if (
        manifest.get("coverage_report_id") != report.get("validation_report_id")
        or manifest.get("coverage_report", {}).get("sha256") != _sha256(report_bytes)
        or manifest.get("coverage_report", {}).get("byte_size") != len(report_bytes)
        or manifest.get("prompt_hash_waiver", {}).get("waiver_id") != waiver.get("waiver_id")
        or manifest.get("prompt_hash_waiver", {}).get("sha256") != _sha256(waiver_bytes)
        or manifest.get("prompt_hash_waiver", {}).get("byte_size") != len(waiver_bytes)
    ):
        raise ValueError("preserved merge report/waiver provenance differs")
    for kind, exact in AUTHORIZED_OUTPUTS.items():
        observed = manifest["outputs"][kind]
        if any(observed.get(field) != value for field, value in exact.items()):
            raise ValueError(f"preserved {kind} evidence differs")

    code_root = Path(__file__).resolve().parents[1]
    publication_commit, dirty = _git_state(code_root)
    if dirty:
        raise ValueError("publication recovery checkout must be tracked-clean")

    immutable_bytes = {
        "model": model_bytes,
        "waiver": waiver_bytes,
        "report": report_bytes,
    }

    def revalidate() -> None:
        current_model, current_model_bytes = _load_regular_json(
            manifest_path, label="production model-run manifest"
        )
        current_waiver, current_waiver_bytes = _load_regular_json(
            waiver_path, label="prompt-hash waiver"
        )
        current_report, current_report_bytes = _load_regular_json(
            report_path, label="coverage report"
        )
        if (
            current_model_bytes != immutable_bytes["model"]
            or current_waiver_bytes != immutable_bytes["waiver"]
            or current_report_bytes != immutable_bytes["report"]
            or canonical_json_bytes(current_model) != canonical_json_bytes(model)
            or canonical_json_bytes(current_waiver) != canonical_json_bytes(waiver)
            or canonical_json_bytes(current_report) != canonical_json_bytes(report)
        ):
            raise ValueError("recovery inputs changed before publication")
        require_production_checkout_generation_state(
            repository_root,
            expected_generation_commit=AUTHORIZED_GENERATION_GIT_COMMIT,
        )
        current_commit, current_dirty = _git_state(code_root)
        if current_commit != publication_commit or current_dirty:
            raise ValueError("publication recovery Git state changed")

    if stage_exists:
        publish_claimed_validated_stage(
            stage=stage,
            target=target,
            expected_manifest=manifest,
            expected_row_counts=AUTHORIZED_ROW_COUNTS,
            revalidate=revalidate,
        )
    sidecar: dict[str, Any] = {
        "schema_name": "part1_merge_stage_recovery",
        "schema_version": "1.0.0",
        "merge_stage_recovery_id": "",
        "publication_mode": "cooperative_claim_same_parent_atomic_rename_v1",
        "model_run_id": AUTHORIZED_MODEL_RUN_ID,
        "generation_git_commit": AUTHORIZED_GENERATION_GIT_COMMIT,
        "original_merge_recovery_commit": AUTHORIZED_ORIGINAL_RECOVERY_COMMIT,
        "publication_recovery_commit": publication_commit,
        "prompt_hash_waiver": {
            "waiver_id": waiver["waiver_id"],
            "relative_path": waiver_path.relative_to(repository_root).as_posix(),
            "sha256": _sha256(waiver_bytes),
            "byte_size": len(waiver_bytes),
        },
        "coverage_report": {
            "validation_report_id": report["validation_report_id"],
            "relative_path": report_path.relative_to(repository_root).as_posix(),
            "sha256": _sha256(report_bytes),
            "byte_size": len(report_bytes),
        },
        "preserved_stage_relative_path": stage.relative_to(repository_root).as_posix(),
        "published_directory_relative_path": target.relative_to(repository_root).as_posix(),
        "merge_manifest": {
            "merge_id": manifest["merge_id"],
            "merge_manifest_hash": manifest["merge_manifest_hash"],
            "relative_path": (target / "merge_manifest.json").relative_to(repository_root).as_posix(),
            "sha256": _sha256(merge_manifest_bytes),
            "byte_size": len(merge_manifest_bytes),
        },
        "outputs": copy.deepcopy(manifest["outputs"]),
        "source_inventory_sha256": _sha256(canonical_json_bytes(manifest["source_files"])),
    }
    sidecar["merge_stage_recovery_id"] = merge_stage_recovery_id(sidecar)
    sidecar_path = validation_directory / RECOVERY_SIDECAR_NAME
    _write_sidecar(sidecar_path, sidecar)
    return target, sidecar_path, sidecar


def publish_claimed_validated_stage(
    *,
    stage: Path,
    target: Path,
    expected_manifest: Mapping[str, Any],
    expected_row_counts: Mapping[str, int],
    revalidate: Callable[[], None],
    before_final_absence_check: Callable[[], None] | None = None,
    after_stage_validation: Callable[[], None] | None = None,
) -> Path:
    """Publish by same-parent atomic rename under an exclusive cooperative claim."""

    stage = Path(os.path.abspath(stage))
    target = Path(os.path.abspath(target))
    if stage.parent != target.parent or stage.name == target.name:
        raise ValueError("recovery stage and target must be distinct same-parent paths")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(stage.parent, parent_flags)
    stage_descriptor: int | None = None
    claim_name = f".{target.name}.publish-claim"
    claim_owned = False
    claim_identity: tuple[int, int] | None = None
    primary: BaseException | None = None
    try:
        if not os.path.lexists(stage):
            if os.path.lexists(target):
                validate_merge_directory_at(
                    parent_descriptor, target.name, expected_manifest=expected_manifest
                )
                return target
            raise ValueError(f"recovery stage is missing: {stage}")
        stage_descriptor = os.open(
            stage.name, parent_flags, dir_fd=parent_descriptor
        )
        stage_status = os.fstat(stage_descriptor)
        manifest = _validate_merge_directory_descriptor(
            stage_descriptor, expected_manifest=expected_manifest
        )
        if _row_counts(manifest) != dict(expected_row_counts):
            raise ValueError("recovery stage row counts differ from the exact contract")
        if after_stage_validation is not None:
            after_stage_validation()
        current_stage = os.stat(
            stage.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (current_stage.st_dev, current_stage.st_ino) != (
            stage_status.st_dev,
            stage_status.st_ino,
        ):
            raise RuntimeError("recovery stage path identity changed after validation")
        if os.path.lexists(target):
            validate_merge_directory_at(
                parent_descriptor, target.name, expected_manifest=expected_manifest
            )
            return target
        try:
            os.mkdir(claim_name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise FileExistsError(
                f"cooperative publication claim already exists: {stage.parent / claim_name}"
            ) from exc
        claim_owned = True
        claim_status = os.stat(
            claim_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISDIR(claim_status.st_mode):
            raise RuntimeError("publication claim is not a directory")
        claim_identity = (claim_status.st_dev, claim_status.st_ino)
        revalidate()
        if before_final_absence_check is not None:
            before_final_absence_check()
        if os.path.lexists(target):
            raise FileExistsError(f"merge target appeared while claim was held: {target}")
        current_stage = os.stat(
            stage.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (current_stage.st_dev, current_stage.st_ino) != (
            stage_status.st_dev,
            stage_status.st_ino,
        ):
            raise RuntimeError("recovery stage identity changed while claim was held")
        # Plain rename is atomic within one parent.  The cooperative claim and
        # final absence check provide no-overwrite coordination for recovery
        # publishers on GPFS, where renameat2(RENAME_NOREPLACE) is unsupported.
        expected_identity = (stage_status.st_dev, stage_status.st_ino)
        try:
            os.rename(
                stage.name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except BaseException as rename_error:
            try:
                stage_after = os.stat(
                    stage.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                stage_still_original = (
                    stage_after.st_dev, stage_after.st_ino
                ) == expected_identity
                target_absent = not os.path.lexists(target)
            except BaseException as evidence_error:
                raise PublicationStateIndeterminateError(
                    "rename failed and publication state is indeterminate because it "
                    "could not be classified; "
                    f"stage={stage}; target={target}; expected_inode={expected_identity}; "
                    f"rename_error={rename_error}; evidence_error={evidence_error}"
                ) from rename_error
            if stage_still_original and target_absent:
                raise
            raise PublicationStateIndeterminateError(
                "rename reported failure but paths indicate publication may have occurred; "
                f"stage={stage}; target={target}; expected_inode={expected_identity}; "
                f"stage_still_original={stage_still_original}; "
                f"target_absent={target_absent}; rename_error={rename_error}"
            ) from rename_error
        try:
            published_status = os.stat(
                target.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            observed_identity = (published_status.st_dev, published_status.st_ino)
            if observed_identity != expected_identity:
                raise RuntimeError(
                    "post-rename target identity differs from validated stage: "
                    f"expected={expected_identity}; observed={observed_identity}; target={target}"
                )
            _fsync_directory_descriptor(parent_descriptor)
            _validate_merge_directory_descriptor(
                stage_descriptor, expected_manifest=expected_manifest
            )
            current_target = os.stat(
                target.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if (current_target.st_dev, current_target.st_ino) != expected_identity:
                raise RuntimeError("published target path no longer names validated inode")
        except BaseException as post_rename_error:
            raise PublicationStateIndeterminateError(
                "post-rename publication state is indeterminate; paths were preserved without "
                f"name-based rollback: target={target}; expected_inode={expected_identity}; "
                f"error={post_rename_error}"
            ) from post_rename_error
        return target
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        if claim_owned and claim_identity is not None:
            try:
                current = os.stat(
                    claim_name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != claim_identity:
                    raise RuntimeError("publication claim identity was replaced")
                os.rmdir(claim_name, dir_fd=parent_descriptor)
                _fsync_directory_descriptor(parent_descriptor)
            except BaseException as exc:
                cleanup_error = exc
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        os.close(parent_descriptor)
        if cleanup_error is not None:
            if primary is not None:
                primary.add_note(
                    "publication claim cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise cleanup_error


__all__ = [
    "merge_stage_recovery_id",
    "validate_merge_stage_recovery",
    "recover_authorized_merge_stage",
    "publish_claimed_validated_stage",
]
