"""Focused contract tests for the one-run prompt-hash validation waiver."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def _waiver() -> dict:
    from part1_prompt_hash_waiver import prompt_hash_waiver_id

    value = {
        "schema_name": "part1_prompt_hash_waiver",
        "schema_version": "1.0.0",
        "waiver_id": "",
        "scope": "production_prompt_hash_validator_bug_only",
        "user_authorization": "User approved the narrow prompt-hash recovery waiver on 2026-08-15.",
        "study_id": "1" * 64,
        "model_run_id": "2" * 64,
        "model_run_manifest_hash": "3" * 64,
        "generation_git_commit": "4" * 40,
        "recovery_git_commit": "5" * 40,
        "coverage_report": {
            "relative_path": "results/part1/" + "2" * 64 + "/validation/coverage_report.json",
            "validation_report_id": "6" * 64,
            "sha256": "7" * 64,
            "byte_size": 123,
        },
        "observed_failure_fingerprint": {
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
        },
        "waived_check": "global_prompt_contract_hash_compared_to_rendered_prompt_hash",
        "replacement_checks": [
            "source_inventory_bytes",
            "schema_lifecycle_hierarchy_duplicates_counts",
            "nonwaived_natural_checkpoint_manifest_compatibility",
            "all_natural_checkpoint_executions_complete",
            "content_derived_natural_prompt_hash",
            "generation_and_recovery_git_state",
        ],
    }
    value["waiver_id"] = prompt_hash_waiver_id(value)
    return value


def test_waiver_identity_binds_every_nonlocation_field() -> None:
    from part1_prompt_hash_waiver import prompt_hash_waiver_id, validate_prompt_hash_waiver

    waiver = _waiver()
    validate_prompt_hash_waiver(waiver)
    moved = copy.deepcopy(waiver)
    moved["coverage_report"]["relative_path"] = "elsewhere/coverage.json"
    assert prompt_hash_waiver_id(moved) == waiver["waiver_id"]
    changed = copy.deepcopy(waiver)
    changed["coverage_report"]["sha256"] = "8" * 64
    assert prompt_hash_waiver_id(changed) != waiver["waiver_id"]
    with pytest.raises(ValueError, match="identity"):
        validate_prompt_hash_waiver(changed)


def test_failure_messages_reject_any_unrelated_validation_defect() -> None:
    from part1_prompt_hash_waiver import require_only_prompt_hash_failure_messages

    require_only_prompt_hash_failure_messages(
        [
            "natural ('question-a', 0) is manifest_incompatible",
            "checkpoint ('question-a', 0, 'cp-00') has physical data beneath a nonterminal or incompatible parent",
            "unexpected physical record count is 11",
        ],
        expected_natural=1,
        expected_checkpoints=1,
        expected_unexpected=11,
    )
    with pytest.raises(ValueError, match="unrelated"):
        require_only_prompt_hash_failure_messages(
            [
                "natural ('question-a', 0) is manifest_incompatible",
                "checkpoint ('question-a', 0, 'cp-00') has physical data beneath a nonterminal or incompatible parent",
                "shard-000 lifecycle defect: missing completion",
                "unexpected physical record count is 11",
            ],
            expected_natural=1,
            expected_checkpoints=1,
            expected_unexpected=11,
        )


def test_successful_natural_prompt_hash_is_recomputed_from_saved_content() -> None:
    from part1_contract import canonical_json_bytes
    from part1_prompt_hash_waiver import require_content_derived_prompt_hash

    record = {
        "natural_execution_outcome": "complete",
        "prompt_hash": hashlib.sha256(
            canonical_json_bytes(
                {
                    "prompt_version": "smollm3-thinking-mmlu-v1",
                    "rendered_prompt": "Question?",
                    "prompt_token_ids": [1, 2, 3],
                }
            )
        ).hexdigest(),
        "rendered_prompt": "Question?",
        "prompt_token_ids": [1, 2, 3],
        "component_versions": {"prompt": "smollm3-thinking-mmlu-v1"},
    }
    require_content_derived_prompt_hash(record)
    record["prompt_token_ids"] = [1, 2, 4]
    with pytest.raises(ValueError, match="content-derived"):
        require_content_derived_prompt_hash(record)


def test_waiver_rejects_terminal_failure_rows_even_when_schema_valid() -> None:
    from part1_prompt_hash_waiver import (
        require_complete_checkpoint_outcome,
        require_content_derived_prompt_hash,
    )

    with pytest.raises(ValueError, match="complete natural"):
        require_content_derived_prompt_hash(
            {"natural_execution_outcome": "terminal_infrastructure_failure"}
        )
    with pytest.raises(ValueError, match="complete checkpoint"):
        require_complete_checkpoint_outcome(
            {"checkpoint_execution_outcome": "terminal_infrastructure_failure"}
        )


def test_production_checkout_must_remain_at_clean_generation_commit(tmp_path: Path) -> None:
    from subprocess import CompletedProcess
    from part1_prompt_hash_waiver import require_production_checkout_generation_state

    calls = []

    def clean(arguments, **kwargs):
        calls.append((arguments, kwargs["cwd"]))
        output = "4" * 40 + "\n" if arguments[-2:] == ["rev-parse", "HEAD"] else ""
        return CompletedProcess(arguments, 0, stdout=output, stderr="")

    require_production_checkout_generation_state(
        tmp_path, expected_generation_commit="4" * 40, run_command=clean
    )
    assert len(calls) == 2

    def advanced(arguments, **kwargs):
        output = "5" * 40 + "\n" if arguments[-2:] == ["rev-parse", "HEAD"] else ""
        return CompletedProcess(arguments, 0, stdout=output, stderr="")

    with pytest.raises(ValueError, match="generation commit"):
        require_production_checkout_generation_state(
            tmp_path, expected_generation_commit="4" * 40, run_command=advanced
        )


def test_waiver_authorization_is_scoped_to_exact_run_commit_and_report() -> None:
    from part1_prompt_hash_waiver import require_authorized_production_target

    model = {
        "model_run_id": "6b29188239ffa494ddb5f409f6dba2db510bcb9c3b6f8f7ada81c524af4d1e7c",
        "final_production_git_commit": "ffa998a7ee1f156e150c8da33b258165ee53e032",
    }
    report = {
        "validation_report_id": "2bfe7cd6908351e3f1d6c9a2eec4f41c9dfa97f124f9da2f70925365490f23db"
    }
    require_authorized_production_target(model_manifest=model, report=report)
    for field, replacement in (
        ("model_run_id", "0" * 64),
        ("final_production_git_commit", "0" * 40),
    ):
        changed = dict(model)
        changed[field] = replacement
        with pytest.raises(ValueError, match="not authorized"):
            require_authorized_production_target(model_manifest=changed, report=report)
    with pytest.raises(ValueError, match="not authorized"):
        require_authorized_production_target(
            model_manifest=model,
            report={"validation_report_id": "0" * 64},
        )


def test_merge_manifest_v11_records_exact_waiver_bytes() -> None:
    from part1_merge import build_merge_manifest, validate_merge_manifest

    waiver = _waiver()
    waiver_bytes = json.dumps(waiver, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    provenance = {
        "study_id": "1" * 64,
        "study_manifest_hash": "9" * 64,
        "question_manifest_hash": "a" * 64,
        "model_run_id": "2" * 64,
        "model_run_manifest_hash": "3" * 64,
        "coverage_report_id": "6" * 64,
    }
    outputs = {}
    manifest = build_merge_manifest(
        provenance=provenance,
        coverage_report_path=f"results/part1/{provenance['model_run_id']}/validation/coverage_report.json",
        coverage_report_sha256="7" * 64,
        coverage_report_byte_size=123,
        prompt_hash_waiver_path=f"results/part1/{provenance['model_run_id']}/validation/prompt_hash_waiver.json",
        prompt_hash_waiver_id=waiver["waiver_id"],
        prompt_hash_waiver_sha256=hashlib.sha256(waiver_bytes).hexdigest(),
        prompt_hash_waiver_byte_size=len(waiver_bytes),
        source_files=[],
        outputs=outputs,
    )
    # The validator reaches output validation only after accepting the v1.1
    # waiver schema; this small unit test does not construct Parquet summaries.
    assert manifest["schema_version"] == "1.1.0"
    assert manifest["prompt_hash_waiver"]["waiver_id"] == waiver["waiver_id"]
    with pytest.raises(ValueError, match="source inventory|outputs"):
        validate_merge_manifest(manifest)


def test_recovery_jobs_are_cpu_bind_safe_and_require_explicit_waiver() -> None:
    merge = Path("jobs/part1_merge_prompt_hash_waiver.sh").read_text(encoding="utf-8")
    analysis = Path("jobs/part1_analyze_prompt_hash_waiver.sh").read_text(encoding="utf-8")
    prepare = Path("jobs/part1_prepare_prompt_hash_waiver.sh").read_text(encoding="utf-8")
    for text in (prepare, merge, analysis):
        assert "#SBATCH --gpus" not in text
        assert "srun --cpu-bind=none" in text
        assert '"${PRODUCTION_REPOSITORY_ROOT:?PRODUCTION_REPOSITORY_ROOT' in text
        assert '--repository-root "$PRODUCTION_REPOSITORY_ROOT"' in text
    assert "--prompt-hash-waiver" in merge
    assert "--prompt-hash-waiver" in analysis
    assert "scripts/prepare_part1_prompt_hash_waiver.py" in prepare


def test_recovery_launcher_submits_only_prepare_merge_analysis_afterok_chain() -> None:
    text = Path("scripts/submit_part1_prompt_hash_waiver_recovery.py").read_text(
        encoding="utf-8"
    )
    assert "jobs/part1_prepare_prompt_hash_waiver.sh" in text
    assert "jobs/part1_merge_prompt_hash_waiver.sh" in text
    assert "jobs/part1_analyze_prompt_hash_waiver.sh" in text
    assert "afterok:{jobs['prepare']}" in text
    assert "afterok:{jobs['merge']}" in text
    assert "part1_generate_array" not in text
