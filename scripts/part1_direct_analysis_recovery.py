"""Immutable identity contract for the explicit no-preflight analysis recovery."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from part1_contract import canonical_json_bytes


SCHEMA_VERSION = "part1-direct-analysis-recovery-receipt-v1"


def direct_analysis_recovery_id(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload["direct_analysis_recovery_id"] = ""
    for mutable in ("status", "analysis_job_id", "submitted_at"):
        payload.pop(mutable, None)
    return hashlib.sha256(canonical_json_bytes({
        "identity_type": "part1_direct_analysis_recovery",
        "identity_version": "part1-direct-analysis-recovery-v1",
        "payload": payload,
    })).hexdigest()


def validate_direct_analysis_recovery_receipt(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "direct_analysis_recovery_id", "model_run_id",
        "model_run_manifest_hash", "merge_stage_recovery_id",
        "merge_stage_recovery_sha256", "merge_stage_recovery_byte_size",
        "analysis_execution_commit", "bootstrap_replicates", "no_preflight",
        "command", "status", "analysis_job_id", "submitted_at",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("direct-analysis recovery receipt fields differ")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["bootstrap_replicates"] != 5000
        or value["no_preflight"] is not True
        or value["status"] not in {"submitting", "submitted", "submission_failed"}
        or value["direct_analysis_recovery_id"] != direct_analysis_recovery_id(value)
    ):
        raise ValueError("direct-analysis recovery receipt fixed semantics differ")
    for field in (
        "direct_analysis_recovery_id", "model_run_id", "model_run_manifest_hash",
        "merge_stage_recovery_id", "merge_stage_recovery_sha256",
    ):
        item = value[field]
        if not isinstance(item, str) or len(item) != 64 or any(
            character not in "0123456789abcdef" for character in item
        ):
            raise ValueError(f"direct-analysis recovery {field} is invalid")
    commit = value["analysis_execution_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("direct-analysis recovery commit is invalid")
    if type(value["merge_stage_recovery_byte_size"]) is not int or value[
        "merge_stage_recovery_byte_size"
    ] <= 0:
        raise ValueError("direct-analysis recovery sidecar byte size is invalid")
    if not isinstance(value["command"], list) or not value["command"] or any(
        not isinstance(item, str) or not item for item in value["command"]
    ):
        raise ValueError("direct-analysis recovery command is invalid")
    if not isinstance(value["submitted_at"], str) or not value["submitted_at"]:
        raise ValueError("direct-analysis recovery timestamp is invalid")
    if value["analysis_job_id"] is not None and (
        not isinstance(value["analysis_job_id"], str) or not value["analysis_job_id"].isdigit()
    ):
        raise ValueError("direct-analysis recovery job ID is invalid")


__all__ = [
    "SCHEMA_VERSION",
    "direct_analysis_recovery_id",
    "validate_direct_analysis_recovery_receipt",
]
