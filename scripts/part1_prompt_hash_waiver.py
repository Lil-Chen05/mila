"""Fail-closed recovery contract for the production prompt-hash validator bug.

This module does not weaken the normal validator.  It describes one explicit,
content-addressed waiver and the replacement check used while the already
generated production shards are read for merge.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from part1_contract import canonical_json_bytes, validate_instance
from part1_coverage import coverage_report_id, validate_coverage_report_semantics


WAIVER_SCOPE = "production_prompt_hash_validator_bug_only"
WAIVER_AUTHORIZATION = (
    "User approved the narrow prompt-hash recovery waiver on 2026-08-15."
)
WAIVED_CHECK = "global_prompt_contract_hash_compared_to_rendered_prompt_hash"
REPLACEMENT_CHECKS = (
    "source_inventory_bytes",
    "schema_lifecycle_hierarchy_duplicates_counts",
    "nonwaived_natural_checkpoint_manifest_compatibility",
    "all_natural_checkpoint_executions_complete",
    "content_derived_natural_prompt_hash",
    "generation_and_recovery_git_state",
)
AUTHORIZED_MODEL_RUN_ID = "6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c"
AUTHORIZED_GENERATION_GIT_COMMIT = "ffa998a7ee1f156e150c8da33b258165ee53e032"
AUTHORIZED_VALIDATION_REPORT_ID = "2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_NATURAL_FAILURE = re.compile(r"natural \(.+\) is manifest_incompatible")
_CHECKPOINT_CASCADE = re.compile(
    r"checkpoint \(.+\) has physical data beneath a nonterminal or incompatible parent"
)
_UNEXPECTED_COUNT = re.compile(r"unexpected physical record count is ([0-9]+)")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _without_location(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["waiver_id"] = ""
    report = payload.get("coverage_report")
    if isinstance(report, dict):
        report.pop("relative_path", None)
    return payload


def prompt_hash_waiver_id(value: Mapping[str, Any]) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                "identity_type": "part1_prompt_hash_waiver",
                "identity_version": "part1-prompt-hash-waiver-v1",
                "payload": _without_location(value),
            }
        )
    )


def validate_prompt_hash_waiver(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_name", "schema_version", "waiver_id", "scope",
        "user_authorization", "study_id", "model_run_id",
        "model_run_manifest_hash", "generation_git_commit",
        "recovery_git_commit", "coverage_report",
        "observed_failure_fingerprint", "waived_check", "replacement_checks",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("prompt-hash waiver fields differ from the fixed contract")
    if (
        value["schema_name"] != "part1_prompt_hash_waiver"
        or value["schema_version"] != "1.0.0"
        or value["scope"] != WAIVER_SCOPE
        or value["user_authorization"] != WAIVER_AUTHORIZATION
        or value["waived_check"] != WAIVED_CHECK
        or value["replacement_checks"] != list(REPLACEMENT_CHECKS)
    ):
        raise ValueError("prompt-hash waiver fixed semantics differ")
    for field in ("waiver_id", "study_id", "model_run_id", "model_run_manifest_hash"):
        if not isinstance(value[field], str) or _SHA256.fullmatch(value[field]) is None:
            raise ValueError(f"prompt-hash waiver {field} is not a SHA-256 identity")
    for field in ("generation_git_commit", "recovery_git_commit"):
        if not isinstance(value[field], str) or _GIT_SHA.fullmatch(value[field]) is None:
            raise ValueError(f"prompt-hash waiver {field} is not a Git commit")
    report = value["coverage_report"]
    if not isinstance(report, Mapping) or set(report) != {
        "relative_path", "validation_report_id", "sha256", "byte_size"
    }:
        raise ValueError("prompt-hash waiver report provenance fields differ")
    expected_path = f"results/part1/{value['model_run_id']}/validation/coverage_report.json"
    if (
        report["relative_path"] != expected_path
        or any(
            not isinstance(report[field], str) or _SHA256.fullmatch(report[field]) is None
            for field in ("validation_report_id", "sha256")
        )
        or isinstance(report["byte_size"], bool)
        or not isinstance(report["byte_size"], int)
        or report["byte_size"] <= 0
    ):
        raise ValueError("prompt-hash waiver report provenance is invalid")
    fingerprint = value["observed_failure_fingerprint"]
    expected_fingerprint_fields = {
        "shards", "natural_physical_records", "checkpoint_physical_records",
        "natural_complete", "natural_terminal_infrastructure_failure",
        "natural_retryable_incomplete", "natural_missing", "natural_duplicate",
        "natural_schema_incompatible", "natural_manifest_incompatible",
        "checkpoint_complete", "checkpoint_terminal_infrastructure_failure",
        "checkpoint_retryable_incomplete", "checkpoint_ineligible",
        "checkpoint_missing", "checkpoint_duplicate",
        "unexpected_physical_record_count", "structural_error_count", "warning_count",
    }
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != expected_fingerprint_fields:
        raise ValueError("prompt-hash waiver failure fingerprint fields differ")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in fingerprint.values()):
        raise ValueError("prompt-hash waiver failure fingerprint values are invalid")
    if value["waiver_id"] != prompt_hash_waiver_id(value):
        raise ValueError("prompt-hash waiver identity does not recompute")


def require_only_prompt_hash_failure_messages(
    errors: Sequence[str],
    *,
    expected_natural: int,
    expected_checkpoints: int,
    expected_unexpected: int,
) -> None:
    natural = checkpoint = count_messages = 0
    for error in errors:
        if _NATURAL_FAILURE.fullmatch(error):
            natural += 1
        elif _CHECKPOINT_CASCADE.fullmatch(error):
            checkpoint += 1
        else:
            match = _UNEXPECTED_COUNT.fullmatch(error)
            if match is not None and int(match.group(1)) == expected_unexpected:
                count_messages += 1
            else:
                raise ValueError(f"coverage report contains an unrelated defect: {error}")
    if (natural, checkpoint, count_messages) != (
        expected_natural,
        expected_checkpoints,
        1,
    ):
        raise ValueError(
            "coverage report prompt-hash/cascade failure counts differ: "
            f"natural={natural}, checkpoint={checkpoint}, count_messages={count_messages}"
        )


def observed_failure_fingerprint(report: Mapping[str, Any]) -> dict[str, int]:
    summary = report["summary"]
    natural = summary["natural_partition"]
    checkpoint = summary["checkpoint_partition"]
    return {
        "shards": summary["observed"]["shards"],
        "natural_physical_records": summary["observed"]["natural_physical_records"],
        "checkpoint_physical_records": summary["observed"]["checkpoint_physical_records"],
        **{f"natural_{key}": natural[key] for key in (
            "complete", "terminal_infrastructure_failure", "retryable_incomplete",
            "missing", "duplicate", "schema_incompatible", "manifest_incompatible",
        )},
        **{f"checkpoint_{key}": checkpoint[key] for key in (
            "complete", "terminal_infrastructure_failure", "retryable_incomplete",
            "ineligible", "missing", "duplicate",
        )},
        "unexpected_physical_record_count": summary["unexpected_physical_record_count"],
        "structural_error_count": summary["structural_error_count"],
        "warning_count": report["warning_count"],
    }


def require_authorized_production_target(
    *, model_manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    if (
        model_manifest.get("model_run_id") != AUTHORIZED_MODEL_RUN_ID
        or model_manifest.get("final_production_git_commit")
        != AUTHORIZED_GENERATION_GIT_COMMIT
        or report.get("validation_report_id") != AUTHORIZED_VALIDATION_REPORT_ID
    ):
        raise ValueError(
            "prompt-hash recovery waiver is not authorized for this model run, "
            "generation commit, or validation report"
        )


def require_exact_failed_report(
    report: Mapping[str, Any],
    *,
    report_bytes: bytes,
    model_manifest: Mapping[str, Any],
) -> dict[str, int]:
    validate_instance("validation_report", report)
    validate_coverage_report_semantics(report)
    require_authorized_production_target(
        model_manifest=model_manifest, report=report
    )
    if report.get("validation_report_id") != coverage_report_id(report):
        raise ValueError("failed coverage report identity does not recompute")
    if (
        report.get("schema_version") != "1.1.0"
        or report.get("validated_artifact_kind") != "production_coverage"
        or report.get("model_run_id") != model_manifest.get("model_run_id")
        or report.get("model_run_manifest_hash") != model_manifest.get("model_run_manifest_hash")
        or report.get("study_id") != model_manifest.get("study_id")
        or report.get("paper_analysis_ready") is not False
        or report.get("coverage_complete") is not False
        or report.get("structurally_valid") is not False
        or report.get("warning_count") != 0
        or report.get("summary", {}).get("structural_warning_count") != 0
        or report.get("summary", {}).get("structural_warnings") != []
    ):
        raise ValueError("coverage report does not match the waived failed-report state")
    fingerprint = observed_failure_fingerprint(report)
    expected = {
        "shards": 500,
        "natural_physical_records": 5000,
        "checkpoint_physical_records": 55000,
        "natural_complete": 0,
        "natural_terminal_infrastructure_failure": 0,
        "natural_retryable_incomplete": 0,
        "natural_missing": 0,
        "natural_duplicate": 0,
        "natural_schema_incompatible": 0,
        "natural_manifest_incompatible": 5000,
        "checkpoint_complete": 0,
        "checkpoint_terminal_infrastructure_failure": 0,
        "checkpoint_retryable_incomplete": 55000,
        "checkpoint_ineligible": 0,
        "checkpoint_missing": 0,
        "checkpoint_duplicate": 0,
        "unexpected_physical_record_count": 55000,
        "structural_error_count": 60001,
        "warning_count": 0,
    }
    if fingerprint != expected:
        raise ValueError(f"coverage report failure fingerprint differs: {fingerprint}")
    expected_natural_partition = {
        "complete": 0,
        "terminal_infrastructure_failure": 0,
        "retryable_incomplete": 0,
        "missing": 0,
        "duplicate": 0,
        "schema_incompatible": 0,
        "manifest_incompatible": 5000,
    }
    expected_checkpoint_partition = {
        "complete": 0,
        "terminal_infrastructure_failure": 0,
        "retryable_incomplete": 55000,
        "ineligible": 0,
        "missing": 0,
        "duplicate": 0,
    }
    if (
        report["summary"]["natural_partition"] != expected_natural_partition
        or report["summary"]["checkpoint_partition"] != expected_checkpoint_partition
    ):
        raise ValueError("coverage report logical partitions differ from exact cascade")
    require_only_prompt_hash_failure_messages(
        report["summary"]["structural_errors"],
        expected_natural=5000,
        expected_checkpoints=55000,
        expected_unexpected=55000,
    )
    if not report_bytes or _sha256(report_bytes) == _sha256(b""):
        raise ValueError("failed coverage report bytes are empty")
    return fingerprint


def require_content_derived_prompt_hash(record: Mapping[str, Any]) -> None:
    if record.get("natural_execution_outcome") != "complete":
        raise ValueError("prompt-hash waiver requires every natural row to be a complete natural execution")
    prompt_version = record.get("component_versions", {}).get("prompt")
    prompt = record.get("rendered_prompt")
    token_ids = record.get("prompt_token_ids")
    if not isinstance(prompt_version, str) or not isinstance(prompt, str) or not isinstance(token_ids, list):
        raise ValueError("successful natural record lacks prompt-hash source content")
    expected = _sha256(
        canonical_json_bytes(
            {
                "prompt_version": prompt_version,
                "rendered_prompt": prompt,
                "prompt_token_ids": token_ids,
            }
        )
    )
    if record.get("prompt_hash") != expected:
        raise ValueError("natural prompt hash differs from content-derived prompt hash")


def require_complete_checkpoint_outcome(record: Mapping[str, Any]) -> None:
    if record.get("checkpoint_execution_outcome") != "complete":
        raise ValueError("prompt-hash waiver requires every checkpoint row to be a complete checkpoint execution")


def require_production_checkout_generation_state(
    repository_root: Path,
    *,
    expected_generation_commit: str,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    observed = run_command(
        ["git", "rev-parse", "HEAD"], cwd=Path(repository_root), check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if observed != expected_generation_commit:
        raise ValueError(
            "production checkout differs from generation commit: "
            f"expected {expected_generation_commit}, observed {observed}"
        )
    dirty = run_command(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=Path(repository_root), check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("production checkout tracked worktree is not clean")


def build_prompt_hash_waiver(
    *,
    report: Mapping[str, Any],
    report_bytes: bytes,
    report_relative_path: str,
    model_manifest: Mapping[str, Any],
    recovery_git_commit: str,
) -> dict[str, Any]:
    fingerprint = require_exact_failed_report(
        report, report_bytes=report_bytes, model_manifest=model_manifest
    )
    value: dict[str, Any] = {
        "schema_name": "part1_prompt_hash_waiver",
        "schema_version": "1.0.0",
        "waiver_id": "",
        "scope": WAIVER_SCOPE,
        "user_authorization": WAIVER_AUTHORIZATION,
        "study_id": model_manifest["study_id"],
        "model_run_id": model_manifest["model_run_id"],
        "model_run_manifest_hash": model_manifest["model_run_manifest_hash"],
        "generation_git_commit": model_manifest["final_production_git_commit"],
        "recovery_git_commit": recovery_git_commit,
        "coverage_report": {
            "relative_path": report_relative_path,
            "validation_report_id": report["validation_report_id"],
            "sha256": _sha256(report_bytes),
            "byte_size": len(report_bytes),
        },
        "observed_failure_fingerprint": fingerprint,
        "waived_check": WAIVED_CHECK,
        "replacement_checks": list(REPLACEMENT_CHECKS),
    }
    value["waiver_id"] = prompt_hash_waiver_id(value)
    validate_prompt_hash_waiver(value)
    return value


def canonical_waiver_bytes(value: Mapping[str, Any]) -> bytes:
    validate_prompt_hash_waiver(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


__all__ = [
    "build_prompt_hash_waiver", "canonical_waiver_bytes", "observed_failure_fingerprint",
    "prompt_hash_waiver_id", "require_content_derived_prompt_hash",
    "require_authorized_production_target",
    "require_complete_checkpoint_outcome", "require_production_checkout_generation_state",
    "require_exact_failed_report", "require_only_prompt_hash_failure_messages",
    "validate_prompt_hash_waiver",
]
