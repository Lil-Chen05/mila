"""Non-production smoke model-run manifest tests."""

from __future__ import annotations

import copy

import pytest

from part1_contract import (
    FIXED_STUDY_CONTRACT,
    study_id,
    study_manifest_hash,
    validate_instance,
)
from part1_model_run import build_smoke_model_run_manifest


def study() -> dict:
    value = {
        "schema_name": "part1_study_manifest",
        "schema_version": "1.1.0",
        "study_id": "",
        "study_manifest_hash": "",
        "question_source_repository": "cais/mmlu",
        "question_source_revision": "a" * 40,
        "question_manifest_hash": "b" * 64,
        **FIXED_STUDY_CONTRACT,
    }
    value["study_id"] = study_id(value)
    value["study_manifest_hash"] = study_manifest_hash(value)
    return value


def preflight() -> dict:
    study_manifest = study()
    return {
        "study_id": study_manifest["study_id"],
        "study_manifest_hash": study_manifest["study_manifest_hash"],
        "question_manifest_hash": study_manifest["question_manifest_hash"],
        "model_repository": "HuggingFaceTB/SmolLM3-3B",
        "model_revision": "c" * 40,
        "tokenizer_repository": "HuggingFaceTB/SmolLM3-3B",
        "tokenizer_revision": "d" * 40,
        "token_contract": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": None,
            "reasoning_open_tag": "<think>",
            "reasoning_open_token_ids": [10],
            "reasoning_close_tag": "</think>",
            "reasoning_close_token_ids": [11],
            "inducer_text": "</think>\nAnswer:",
            "inducer_token_ids": [11, 12],
            "ad_token_convention": "inducer_boundary_space_uppercase_single_token",
            "ad_raw_token_sequences": {"A": [20], "B": [21], "C": [22], "D": [23]},
            "ad_token_ids": [20, 21, 22, 23],
        },
        "effective_natural_generation": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 8192,
            "return_dict_in_generate": True,
            "output_logits": True,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": None,
        },
        "effective_checkpoint_generation": {
            "do_sample": False,
            "max_new_tokens": 32,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": None,
        },
        "environment_versions": {
            "python": "3.12.0",
            "transformers": "test",
            "torch": "test",
            "cuda": "test",
            "gpu_model": "synthetic",
            "uv_lock_sha256": "e" * 64,
            "model_config_sha256": "f" * 64,
            "tokenizer_files_sha256": {"tokenizer.json": "1" * 64},
            "model_context_window": 65536,
        },
        "component_versions": {
            "adapter": "part1-smollm3-adapter-v1",
            "prompt": "part1-smollm3-mcq-v1",
            "parser": "part1-terminal-block-v1",
            "inducer": "part1-smollm3-forced-close-v1",
        },
    }


def question_manifest() -> dict:
    return {
        "schema_name": "part1_question_manifest",
        "schema_version": "1.1.0",
        "question_manifest_hash": "b" * 64,
        "manifest_format_version": "jsonl-v1",
        "source_repository": "cais/mmlu",
        "source_revision": "a" * 40,
        "source_config_strategy": "per_subject",
        "source_configs": list(FIXED_STUDY_CONTRACT["subjects"]),
        "source_split": "test",
        "subjects": list(FIXED_STUDY_CONTRACT["subjects"]),
        "quota_per_subject": 100,
        "total_count": 500,
        "question_sampling_seed": 42,
        "selection_algorithm_version": "part1-per-subject-full-buffer-shuffle-v1",
        "canonicalization_version": "part1-canonical-json-v1",
        "ordered_record_aggregation": "canonical-record-bytes-in-manifest-order-v1",
        "logical_filename": "questions.jsonl",
    }


def test_smoke_a_and_b_manifests_are_valid_nonproduction_and_have_distinct_ids() -> None:
    smoke_a = build_smoke_model_run_manifest(
        study_manifest=study(),
        preflight_report=preflight(),
        execution_scope="smoke_a",
        base_git_commit="1" * 40,
        diff_hash="2" * 64,
    )
    smoke_b = build_smoke_model_run_manifest(
        study_manifest=study(),
        preflight_report=preflight(),
        execution_scope="smoke_b",
        base_git_commit="1" * 40,
        diff_hash="2" * 64,
    )
    validate_instance("model_run_manifest", smoke_a)
    validate_instance("model_run_manifest", smoke_b)
    assert smoke_a["production"] is False
    assert smoke_a["final_production_git_commit"] is None
    assert smoke_a["smoke_git_provenance"]["production_eligible"] is False
    assert smoke_a["execution_scope"] == "smoke_a"
    assert smoke_b["execution_scope"] == "smoke_b"
    assert smoke_a["model_run_id"] != smoke_b["model_run_id"]
    assert smoke_a["model_run_manifest_hash"] != smoke_b["model_run_manifest_hash"]
    assert smoke_a["environment_versions"] == preflight()["environment_versions"]


def test_cross_artifact_validator_accepts_exact_preflight_model_study_question_bundle() -> None:
    from part1_model_run import validate_preflight_model_run_compatibility

    study_manifest = study()
    preflight_report = preflight()
    model_manifest = build_smoke_model_run_manifest(
        study_manifest=study_manifest,
        preflight_report=preflight_report,
        execution_scope="smoke_a",
        base_git_commit="1" * 40,
        diff_hash="2" * 64,
    )
    validate_preflight_model_run_compatibility(
        preflight_report=preflight_report,
        model_manifest=model_manifest,
        study_manifest=study_manifest,
        question_manifest=question_manifest(),
    )


@pytest.mark.parametrize(
    ("artifact", "field_path", "value", "error_match"),
    [
        ("preflight", ("study_id",), "9" * 64, "study_id"),
        ("preflight", ("question_manifest_hash",), "8" * 64, "question_manifest_hash"),
        ("preflight", ("model_repository",), "wrong/model", "model_repository"),
        (
            "preflight",
            ("token_contract", "reasoning_open_token_ids"),
            [999],
            "reasoning_open_token_ids",
        ),
        (
            "preflight",
            ("effective_natural_generation", "temperature"),
            0.7,
            "effective_natural_generation",
        ),
        (
            "preflight",
            ("environment_versions", "uv_lock_sha256"),
            "7" * 64,
            "environment_versions",
        ),
        (
            "preflight",
            ("component_versions", "parser"),
            "wrong-parser",
            "parser_version",
        ),
        (
            "question",
            ("source_revision",),
            "6" * 40,
            "source_revision",
        ),
    ],
)
def test_cross_artifact_validator_rejects_bound_field_mismatches(
    artifact: str,
    field_path: tuple[str, ...],
    value,
    error_match: str,
) -> None:
    from part1_model_run import validate_preflight_model_run_compatibility

    study_manifest = study()
    preflight_report = preflight()
    model_manifest = build_smoke_model_run_manifest(
        study_manifest=study_manifest,
        preflight_report=preflight_report,
        execution_scope="smoke_a",
        base_git_commit="1" * 40,
        diff_hash="2" * 64,
    )
    question = question_manifest()
    target = preflight_report if artifact == "preflight" else question
    target = copy.deepcopy(target)
    cursor = target
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = value
    if artifact == "preflight":
        preflight_report = target
    else:
        question = target

    with pytest.raises((ValueError, RuntimeError), match=error_match):
        validate_preflight_model_run_compatibility(
            preflight_report=preflight_report,
            model_manifest=model_manifest,
            study_manifest=study_manifest,
            question_manifest=question,
        )
