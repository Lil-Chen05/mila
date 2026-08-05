"""Non-production smoke model-run manifest tests."""

from __future__ import annotations

import copy
from pathlib import Path

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
            "bos_token_id": None,
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
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": None,
        },
        "effective_checkpoint_generation": {
            "do_sample": False,
            "max_new_tokens": 32,
            "bos_token_id": None,
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
            "adapter": "part1-smollm3-adapter-v2",
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
    assert (smoke_a["model_run_id"], smoke_a["model_run_manifest_hash"]) == (
        "2bb6d6c8d3dbe2800baf1b5ad8a24a047d825919f24f525f89206f445e6e7363",
        "867b45e96447bdce5f80303d71ad1bb9619f2abc3d5b2271fb1757703e72774c",
    )
    assert (smoke_b["model_run_id"], smoke_b["model_run_manifest_hash"]) == (
        "1a11c9055343318778f17284d3bf4039773f802dd7d565d5ef441a18c4ef86de",
        "993e84ad70d3ba315a4361d6f4bc532bef2696f02603eecf7eef125d1b723893",
    )


def test_phase3_smoke_is_a_valid_nonproduction_scope() -> None:
    manifest = build_smoke_model_run_manifest(
        study_manifest=study(),
        preflight_report=preflight(),
        execution_scope="phase3_smoke",
        base_git_commit="1" * 40,
        diff_hash="2" * 64,
    )

    validate_instance("model_run_manifest", manifest)
    assert manifest["production"] is False
    assert manifest["execution_scope"] == "phase3_smoke"


def test_build_production_manifest_copies_explicit_preflight_provenance() -> None:
    from part1_model_run import build_production_model_run_manifest

    manifest = build_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=preflight(),
        final_git_commit="1" * 40,
        output_root=Path("results/part1"),
    )

    assert manifest["schema_version"] == "1.1.0"
    assert manifest["production"] is True
    assert manifest["execution_scope"] == "production"
    assert manifest["bos_token_id"] is None
    assert manifest["eos_token_id"] == 2
    assert manifest["pad_token_id"] is None
    assert manifest["model_context_window"] == 65536
    assert manifest["dependency_lock_sha256"] == "e" * 64
    assert manifest["clean_tracked_worktree"] is True
    assert manifest["output_paths"]["raw_shards"].startswith(
        f"results/part1/{manifest['model_run_id']}/"
    )
    assert manifest["output_paths"] == {
        key: f"results/part1/{manifest['model_run_id']}/{suffix}"
        for key, suffix in {
            "raw_shards": "raw_shards",
            "validation": "validation",
            "merged": "merged",
            "analysis": "analysis",
        }.items()
    }
    validate_instance("model_run_manifest", manifest)


def test_build_production_manifest_rejects_preflight_study_mismatch() -> None:
    from part1_model_run import build_production_model_run_manifest

    mismatched_preflight = preflight()
    mismatched_preflight["study_manifest_hash"] = "9" * 64

    with pytest.raises(ValueError, match="study_manifest_hash"):
        build_production_model_run_manifest(
            study_manifest=study(),
            preflight_report=mismatched_preflight,
            final_git_commit="1" * 40,
            output_root=Path("results/part1"),
        )


def test_production_schema_rejects_missing_and_invalid_explicit_fields() -> None:
    from part1_model_run import build_production_model_run_manifest

    manifest = build_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=preflight(),
        final_git_commit="1" * 40,
        output_root=Path("results/part1"),
    )
    for field in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "model_context_window",
        "dependency_lock_sha256",
        "clean_tracked_worktree",
        "output_paths",
    ):
        changed = {key: value for key, value in manifest.items() if key != field}
        with pytest.raises(ValueError, match=field):
            validate_instance("model_run_manifest", changed)

    for field, invalid in (
        ("eos_token_id", -1),
        ("model_context_window", 0),
        ("dependency_lock_sha256", "not-a-hash"),
        ("clean_tracked_worktree", False),
    ):
        with pytest.raises(ValueError, match=field):
            validate_instance("model_run_manifest", {**manifest, field: invalid})

    invalid_paths = copy.deepcopy(manifest)
    invalid_paths["output_paths"]["analysis"] = "/absolute/analysis"
    with pytest.raises(ValueError, match="output_paths.analysis"):
        validate_instance("model_run_manifest", invalid_paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(dependency_lock_sha256="9" * 64), "dependency"),
        (lambda value: value.update(model_context_window=32768), "context"),
        (lambda value: value.update(eos_token_id=3), "eos_token_id"),
        (
            lambda value: value["output_paths"].update(
                analysis=f"results/part1/{'9' * 64}/analysis"
            ),
            "output_paths.analysis",
        ),
    ],
)
def test_production_schema_rejects_cross_field_provenance_mismatch(
    mutation, message: str
) -> None:
    from part1_model_run import build_production_model_run_manifest

    manifest = build_production_model_run_manifest(
        study_manifest=study(),
        preflight_report=preflight(),
        final_git_commit="1" * 40,
        output_root=Path("results/part1"),
    )
    mutation(manifest)
    with pytest.raises(ValueError, match=message):
        validate_instance("model_run_manifest", manifest)


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
