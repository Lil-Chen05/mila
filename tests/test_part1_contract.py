"""Pure Phase 1 contract tests; safe on a login node."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess

import pytest

from part1_contract import (
    AUDIT_EVENT_TYPES,
    CANONICAL_JSON_VERSION,
    CONFIG_NAMES,
    PYTORCH_SEED_MAX,
    SCHEMA_NAMES,
    audit_event_id,
    attempt_id,
    canonical_json_bytes,
    checkpoint_record_id,
    derive_generation_seed,
    load_config,
    load_schema,
    model_run_id,
    model_run_manifest_hash,
    natural_record_id,
    question_content_hash,
    question_id,
    question_manifest_hash,
    shared_probe_id,
    study_id,
    study_manifest_hash,
    validate_instance,
    validate_phase1_config,
)


HEX_64 = "a" * 64
HEX_64_B = "b" * 64
HEX_64_C = "c" * 64


def question() -> dict:
    return {
        "schema_name": "part1_question_record",
        "schema_version": "1.0.0",
        "question_id": HEX_64,
        "question_content_hash": HEX_64_B,
        "sample_index": 0,
        "subject": "high_school_mathematics",
        "subject_selection_index": 0,
        "source_repository": "cais/mmlu",
        "source_revision": "immutable-revision",
        "source_config": "high_school_mathematics",
        "source_split": "test",
        "source_row_identity": {"row_key": "m-17"},
        "question": "What is 2 + 2?",
        "choices": ["1", "2", "4", "5"],
        "gold_index": 2,
        "gold_letter": "C",
    }


def natural(*, outcome: str = "complete") -> dict:
    complete = outcome == "complete"
    return {
        "schema_name": "part1_natural_terminal_result",
        "schema_version": "1.0.0",
        "raw_record_id": HEX_64,
        "study_id": HEX_64_B,
        "model_run_id": HEX_64_C,
        "model_run_manifest_hash": "d" * 64,
        "question_manifest_hash": "e" * 64,
        "question_id": "f" * 64,
        "sample_index": 0,
        "subject": "high_school_mathematics",
        "run_id": 0,
        "generation_seed": 123,
        "seed_algorithm_version": "part1-seed-v1",
        "terminal_attempt_number": 1,
        "terminal_attempt_id": "4" * 64,
        "infrastructure_failure_reference": None if complete else "audit-event:failure",
        "prompt_hash": "1" * 64,
        "rendered_prompt": "Prompt" if complete else None,
        "prompt_token_ids": [1, 2] if complete else None,
        "generated_token_ids": [10, 11] if complete else None,
        "decoded_output": "<think>x</think> Answer: C" if complete else None,
        "reasoning_text": "x" if complete else None,
        "reasoning_boundaries": {"start": 1, "end": 2} if complete else None,
        "close_tag_information": {"found": True, "token_start": 2, "token_end": 3}
        if complete
        else None,
        "stop_reason": "eos" if complete else "error",
        "generated_token_count": 2 if complete else None,
        "reasoning_token_count": 1 if complete else None,
        "per_token_entropy_nats": [1.123456789, 0.987654321] if complete else None,
        "mean_reasoning_entropy_nats": 1.123456789 if complete else None,
        "tail_reasoning_entropy_nats": 1.123456789 if complete else None,
        "terminal_answer_block_text": "Answer: C" if complete else None,
        "terminal_answer_block_span": {"start": 20, "end": 29} if complete else None,
        "natural_answer": "C" if complete else None,
        "raw_confidence_text": "80" if complete else None,
        "raw_parsed_confidence": 80 if complete else None,
        "normalized_confidence": 0.8 if complete else None,
        "natural_correct": True if complete else None,
        "diagnostic_answer_like_text": None,
        "checkpoint_eligible": complete,
        "checkpoint_ids": [f"cp-{index:02d}" for index in range(11)] if complete else None,
        "natural_execution_outcome": outcome,
        "reasoning_status": "closed" if complete else "malformed",
        "answer_parse_status": "parsed" if complete else "missing",
        "confidence_parse_status": "parsed" if complete else "missing",
        "component_versions": {"adapter": "smollm3-v1", "parser": "v1"},
        "terminal_error_details": None
        if complete
        else {"category": "temporary_filesystem_failure", "message": "disk busy"},
    }


def checkpoint(*, outcome: str = "complete", output_status: str = "valid") -> dict:
    complete = outcome == "complete"
    valid = complete and output_status == "valid"
    return {
        "schema_name": "part1_checkpoint_terminal_result",
        "schema_version": "1.0.0",
        "checkpoint_record_id": HEX_64,
        "parent_raw_record_id": HEX_64_B,
        "study_id": HEX_64_C,
        "model_run_id": "d" * 64,
        "model_run_manifest_hash": "e" * 64,
        "question_manifest_hash": "f" * 64,
        "question_id": "1" * 64,
        "sample_index": 0,
        "subject": "high_school_mathematics",
        "run_id": 0,
        "checkpoint_id": "cp-05",
        "natural_seed": 123,
        "terminal_attempt_number": 1,
        "terminal_attempt_id": "4" * 64,
        "infrastructure_failure_reference": None if complete else "audit-event:failure",
        "requested_checkpoint_index": 5,
        "requested_fraction": 0.5,
        "k_keep": 1,
        "actual_fraction": 0.5,
        "shared_probe_id": "2" * 64,
        "is_alias": False,
        "alias_metadata": {"owner_checkpoint_id": "cp-05", "members": ["cp-05"]},
        "prefix_hash": "3" * 64,
        "inducer_version": "smollm3-forced-close-v1",
        "inducer_text": "</think>\nAnswer:",
        "forced_generated_token_ids": [20, 21] if complete else None,
        "decoded_forced_output": " C Confidence: 80" if complete else None,
        "terminal_answer_block_text": "Answer: C\nConfidence: 80" if valid else None,
        "forced_answer": "C" if valid else None,
        "raw_confidence_text": "80" if valid else None,
        "raw_parsed_confidence": 80 if valid else None,
        "normalized_confidence": 0.8 if valid else None,
        "checkpoint_local_correct": True if valid else None,
        "answer_token_index": 0 if valid else None,
        "answer_token_id": 67 if valid else None,
        "token_convention": "leading_space_uppercase_single_token" if valid else None,
        "ad_token_ids": [65, 66, 67, 68] if valid else None,
        "ad_logits_float32": [0.1, 0.2, 0.7, 0.0] if valid else None,
        "ad_probabilities_float32": [0.2, 0.2, 0.5, 0.1] if valid else None,
        "answer_entropy_nats": 1.2206072645530175 if valid else None,
        "full_vocabulary_answer_step_entropy_nats": 4.123456789 if valid else None,
        "maximum_ad_probability": 0.5 if valid else None,
        "agrees_with_natural_answer": True if valid else None,
        "checkpoint_execution_outcome": outcome,
        "checkpoint_model_output_status": output_status if complete else "invalid",
        "answer_parse_status": "parsed" if valid else "missing",
        "confidence_parse_status": "parsed" if valid else "missing",
        "answer_token_status": "located" if valid else "unsupported",
        "entropy_status": "computed" if valid else "unavailable",
        "component_versions": {"adapter": "smollm3-v1", "inducer": "v1"},
        "terminal_error_details": None
        if complete
        else {"category": "transient_worker_failure", "message": "worker lost"},
    }


def test_canonical_json_contract() -> None:
    left = {"z": "café\r\nline", "a": [2, 1], "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "a": [2, 1], "z": "café\nline"}
    encoded = canonical_json_bytes(left)
    assert encoded == canonical_json_bytes(right)
    assert encoded.startswith(b'{"serialization_version":"part1-canonical-json-v1","value":')
    assert b" " not in encoded
    assert "café".encode("utf-8") in encoded
    assert b"\\u00e9" not in encoded
    assert b"\\r" not in encoded
    assert encoded != canonical_json_bytes({**right, "a": [1, 2]})
    assert CANONICAL_JSON_VERSION == "part1-canonical-json-v1"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})


def test_identity_payloads_are_domain_separated_and_ignore_mutable_metadata() -> None:
    q = question()
    content_hash = question_content_hash(q)
    stable_id = question_id(q)
    mutated = {**q, "question_id": "0" * 64, "question_content_hash": "0" * 64}
    mutated.update(
        created_at="later",
        output_path="/different/path",
        status="changed",
        validation_state="failed",
        operator_notes="mutable",
    )
    assert question_content_hash(mutated) == content_hash
    assert question_id(mutated) == stable_id
    changed = copy.deepcopy(q)
    changed["question"] = "What is 3 + 3?"
    assert question_content_hash(changed) != content_hash
    assert question_id(changed) != stable_id
    duplicate_content_other_source_row = copy.deepcopy(q)
    duplicate_content_other_source_row["source_row_identity"] = {"row_key": "m-18"}
    # The content identity intentionally includes immutable source-row identity:
    # duplicate text in distinct source rows must never collapse into one ID.
    assert question_content_hash(duplicate_content_other_source_row) != content_hash
    assert question_id(duplicate_content_other_source_row) != stable_id
    assert content_hash != stable_id


def test_manifest_and_study_identity_hashes_use_explicit_immutable_fields() -> None:
    q = question()
    sidecar = {
        "schema_name": "part1_question_manifest",
        "schema_version": "1.0.0",
        "manifest_format_version": "jsonl-v1",
        "source_repository": "cais/mmlu",
        "source_revision": "immutable-revision",
        "source_config": "all",
        "source_split": "test",
        "subjects": ["high_school_mathematics"],
        "quota_per_subject": 1,
        "total_count": 1,
        "question_sampling_seed": 42,
        "selection_algorithm_version": "part1-balanced-sample-v1",
        "canonicalization_version": CANONICAL_JSON_VERSION,
        "ordered_record_aggregation": "canonical-record-bytes-in-manifest-order-v1",
        "logical_filename": "questions.jsonl",
    }
    mh = question_manifest_hash(sidecar, [q])
    assert mh == question_manifest_hash(
        {**sidecar, "question_manifest_hash": "0" * 64, "created_at": "now", "path": "/tmp/x"},
        [{**q, "validation_state": "ok"}],
    )
    assert mh != question_manifest_hash(sidecar, [{**q, "question": "changed"}])

    study = {
        "schema_name": "part1_study_manifest",
        "schema_version": "1.0.0",
        "question_manifest_hash": mh,
        "subjects": ["high_school_mathematics"],
        "subject_quotas": {"high_school_mathematics": 1},
        "question_sampling_seed": 42,
        "scientific_protocol_version": "part1-science-v1",
        "checkpoint_fractions": [0.0, 0.5, 1.0],
        "checkpoint_placement_contract": {"version": "ties-even-v1"},
        "entropy_contract": {"version": "entropy-v1"},
        "natural_answer_validity_rule": {"version": "post-close-v1"},
        "status_contract_version": "part1-status-v1",
        "calibration_contract": {"version": "calibration-v1"},
        "bootstrap_contract": {"version": "bootstrap-v1"},
        "primary_auroc_feature_registry": ["negative_mean_reasoning_entropy"],
        "within_question_analysis": {"version": "paired-v1"},
        "switching_stabilization_contract": {"version": "trajectory-v1"},
        "repetition_policy": "preserve-successful-output",
        "compatible_raw_record_schema_versions": ["1.0.0"],
        "analysis_contract_version": "part1-analysis-v1",
    }
    sid = study_id(study)
    shash = study_manifest_hash(study)
    mutable = {
        **study,
        "study_id": "0" * 64,
        "study_manifest_hash": "0" * 64,
        "created_at": "later",
        "validation": {"ok": False},
        "notes": "operator",
    }
    assert study_id(mutable) == sid
    assert study_manifest_hash(mutable) == shash
    assert sid != shash


def test_model_run_and_record_identity_key_boundaries() -> None:
    manifest = {
        "schema_name": "part1_model_run_manifest",
        "schema_version": "1.0.0",
        "study_id": HEX_64,
        "study_manifest_hash": HEX_64_B,
        "question_manifest_hash": HEX_64_C,
        "model_repository": "HuggingFaceTB/SmolLM3-3B",
        "model_revision": "model-commit",
        "tokenizer_repository": "HuggingFaceTB/SmolLM3-3B",
        "tokenizer_revision": "tokenizer-commit",
        "canonical_model_identity": "hf:HuggingFaceTB/SmolLM3-3B@model-commit",
        "adapter_version": "smollm3-v1",
        "prompt_version": "part1-prompt-v1",
        "prompt_hash": "d" * 64,
        "parser_version": "part1-parser-v1",
        "inducer_version": "part1-inducer-v1",
        "inducer_text": "</think>\nAnswer:",
        "inducer_token_ids": [1, 2],
        "reasoning_open_tag": "<think>",
        "reasoning_open_token_ids": [3],
        "reasoning_close_tag": "</think>",
        "reasoning_close_token_ids": [4],
        "requested_natural_generation": {"do_sample": True},
        "effective_natural_generation": {"do_sample": True},
        "requested_checkpoint_generation": {"do_sample": False},
        "effective_checkpoint_generation": {"do_sample": False},
        "ad_token_convention": "single-token",
        "ad_raw_token_sequences": {"A": [65], "B": [66], "C": [67], "D": [68]},
        "ad_token_ids": [65, 66, 67, 68],
        "seed_algorithm_version": "part1-seed-v1",
        "base_generation_seed": 42,
        "environment_versions": {"python": "3.12"},
        "final_production_git_commit": None,
        "production": False,
        "smoke_git_provenance": {"base_commit": "e" * 40, "diff_hash": "1" * 64},
    }
    rid = model_run_id(manifest)
    rhash = model_run_manifest_hash(manifest)
    validate_instance(
        "model_run_manifest",
        {**manifest, "model_run_id": rid, "model_run_manifest_hash": rhash},
    )
    with pytest.raises(ValueError, match="final_production_git_commit"):
        validate_instance(
            "model_run_manifest",
            {
                **manifest,
                "model_run_id": rid,
                "model_run_manifest_hash": rhash,
                "final_production_git_commit": "f" * 40,
            },
        )
    production_manifest = {
        **manifest,
        "model_run_id": rid,
        "model_run_manifest_hash": rhash,
        "production": True,
        "final_production_git_commit": "f" * 40,
        "smoke_git_provenance": None,
    }
    validate_instance("model_run_manifest", production_manifest)
    with pytest.raises(ValueError, match="smoke_git_provenance"):
        validate_instance(
            "model_run_manifest",
            {**production_manifest, "smoke_git_provenance": {"base_commit": "e" * 40}},
        )
    assert model_run_id({**manifest, "model_run_id": "0" * 64, "output_path": "/new"}) == rid
    assert model_run_manifest_hash(
        {**manifest, "model_run_manifest_hash": "0" * 64, "runtime_status": "complete"}
    ) == rhash
    assert rid != rhash

    natural_id = natural_record_id(HEX_64, rid, HEX_64_B, 0)
    assert natural_id == natural_record_id(HEX_64, rid, HEX_64_B, 0)
    assert natural_id != natural_record_id(HEX_64, rid, HEX_64_B, 1)
    checkpoint_id = checkpoint_record_id(HEX_64, rid, HEX_64_B, 0, "cp-00")
    assert checkpoint_id != natural_id
    assert checkpoint_id != checkpoint_record_id(HEX_64, rid, HEX_64_B, 0, "cp-01")

    probe_id = shared_probe_id(
        HEX_64,
        rid,
        HEX_64_B,
        0,
        prefix_hash="2" * 64,
        inducer_version="part1-inducer-v1",
    )
    assert probe_id == shared_probe_id(
        HEX_64,
        rid,
        HEX_64_B,
        0,
        prefix_hash="2" * 64,
        inducer_version="part1-inducer-v1",
    )
    assert probe_id != shared_probe_id(
        HEX_64,
        rid,
        HEX_64_B,
        0,
        prefix_hash="3" * 64,
        inducer_version="part1-inducer-v1",
    )
    # Requested checkpoint identity and alias owner are intentionally absent:
    # all aliases for the same physical prefix/probe share this ID.
    assert probe_id not in {checkpoint_id, natural_id}
    alias_checkpoint_id = checkpoint_record_id(HEX_64, rid, HEX_64_B, 0, "cp-04")
    assert alias_checkpoint_id != checkpoint_id
    assert probe_id == shared_probe_id(
        HEX_64,
        rid,
        HEX_64_B,
        0,
        prefix_hash="2" * 64,
        inducer_version="part1-inducer-v1",
    )

    nat_attempt = attempt_id(HEX_64, rid, HEX_64_B, 0, 1)
    cp_attempt = attempt_id(HEX_64, rid, HEX_64_B, 0, 1, checkpoint_id="cp-00")
    assert nat_attempt != cp_attempt
    event = audit_event_id(nat_attempt, "attempt_started", 0)
    assert event != audit_event_id(nat_attempt, "attempt_completed", 0)
    assert event != audit_event_id(nat_attempt, "attempt_started", 1)


def test_seed_algorithm_is_deterministic_bounded_and_versioned() -> None:
    kwargs = dict(
        base_seed=42,
        canonical_model_identity="hf:HuggingFaceTB/SmolLM3-3B@immutable",
        question_id=HEX_64,
        run_id=0,
    )
    seed = derive_generation_seed(**kwargs)
    assert seed == 7276464104989940744
    assert derive_generation_seed(**{**kwargs, "run_id": 9}) == 7450580410574659730
    assert seed == derive_generation_seed(**kwargs)
    assert 0 <= seed <= PYTORCH_SEED_MAX == 2**63 - 1
    assert seed != derive_generation_seed(**{**kwargs, "run_id": 1})
    assert seed != derive_generation_seed(**{**kwargs, "question_id": HEX_64_B})
    assert seed != derive_generation_seed(**{**kwargs, "canonical_model_identity": "hf:other@immutable"})
    assert seed != derive_generation_seed(**kwargs, algorithm_version="part1-seed-v2-test")


def test_all_public_identity_functions_have_fixed_golden_vectors() -> None:
    q = question()
    sidecar = {
        "schema_name": "part1_question_manifest",
        "schema_version": "1.0.0",
        "manifest_format_version": "jsonl-v1",
        "source_repository": "cais/mmlu",
        "source_revision": "immutable-revision",
        "source_config": "all",
        "source_split": "test",
        "subjects": ["high_school_mathematics"],
        "quota_per_subject": 1,
        "total_count": 1,
        "question_sampling_seed": 42,
        "selection_algorithm_version": "part1-balanced-sample-v1",
        "canonicalization_version": CANONICAL_JSON_VERSION,
        "ordered_record_aggregation": "canonical-record-bytes-in-manifest-order-v1",
        "logical_filename": "questions.jsonl",
    }
    qmh = question_manifest_hash(sidecar, [q])
    study = {
        "schema_name": "part1_study_manifest",
        "schema_version": "1.0.0",
        "question_manifest_hash": qmh,
        "subjects": ["high_school_mathematics"],
        "subject_quotas": {"high_school_mathematics": 1},
        "question_sampling_seed": 42,
        "scientific_protocol_version": "part1-science-v1",
        "checkpoint_fractions": [0.0, 0.5, 1.0],
        "checkpoint_placement_contract": {"version": "ties-even-v1"},
        "entropy_contract": {"version": "entropy-v1"},
        "natural_answer_validity_rule": {"version": "post-close-v1"},
        "status_contract_version": "part1-status-v1",
        "calibration_contract": {"version": "calibration-v1"},
        "bootstrap_contract": {"version": "bootstrap-v1"},
        "primary_auroc_feature_registry": ["negative_mean_reasoning_entropy"],
        "within_question_analysis": {"version": "paired-v1"},
        "switching_stabilization_contract": {"version": "trajectory-v1"},
        "repetition_policy": "preserve-successful-output",
        "compatible_raw_record_schema_versions": ["1.0.0"],
        "analysis_contract_version": "part1-analysis-v1",
    }
    sid = study_id(study)
    model = {
        "schema_name": "part1_model_run_manifest",
        "schema_version": "1.0.0",
        "study_id": sid,
        "study_manifest_hash": study_manifest_hash(study),
        "question_manifest_hash": qmh,
        "model_repository": "HuggingFaceTB/SmolLM3-3B",
        "model_revision": "model-commit",
        "tokenizer_repository": "HuggingFaceTB/SmolLM3-3B",
        "tokenizer_revision": "tokenizer-commit",
        "canonical_model_identity": "hf:HuggingFaceTB/SmolLM3-3B@model-commit",
        "adapter_version": "smollm3-v1",
        "prompt_version": "part1-prompt-v1",
        "prompt_hash": "d" * 64,
        "parser_version": "part1-parser-v1",
        "inducer_version": "part1-inducer-v1",
        "inducer_text": "</think>\nAnswer:",
        "inducer_token_ids": [1, 2],
        "reasoning_open_tag": "<think>",
        "reasoning_open_token_ids": [3],
        "reasoning_close_tag": "</think>",
        "reasoning_close_token_ids": [4],
        "requested_natural_generation": {"do_sample": True},
        "effective_natural_generation": {"do_sample": True},
        "requested_checkpoint_generation": {"do_sample": False},
        "effective_checkpoint_generation": {"do_sample": False},
        "ad_token_convention": "single-token",
        "ad_raw_token_sequences": {"A": [65], "B": [66], "C": [67], "D": [68]},
        "ad_token_ids": [65, 66, 67, 68],
        "seed_algorithm_version": "part1-seed-v1",
        "base_generation_seed": 42,
        "environment_versions": {"python": "3.12"},
        "final_production_git_commit": None,
        "production": False,
        "smoke_git_provenance": {"base_commit": "e" * 40, "diff_hash": "1" * 64},
    }
    mrid = model_run_id(model)
    natural_id = natural_record_id(sid, mrid, q["question_id"], 0)
    checkpoint_id = checkpoint_record_id(sid, mrid, q["question_id"], 0, "cp-05")
    probe_id = shared_probe_id(
        sid,
        mrid,
        q["question_id"],
        0,
        prefix_hash="2" * 64,
        inducer_version="part1-inducer-v1",
    )
    aid = attempt_id(sid, mrid, q["question_id"], 0, 1)
    actual = {
        "question_content_hash": question_content_hash(q),
        "question_id": question_id(q),
        "question_manifest_hash": qmh,
        "study_id": sid,
        "study_manifest_hash": study_manifest_hash(study),
        "model_run_id": mrid,
        "model_run_manifest_hash": model_run_manifest_hash(model),
        "natural_record_id": natural_id,
        "checkpoint_record_id": checkpoint_id,
        "shared_probe_id": probe_id,
        "attempt_id": aid,
        "audit_event_id": audit_event_id(aid, "attempt_started", 0),
    }
    assert actual == {
        "question_content_hash": "c57e912ef72fee82c4920d43a9b284a5e79657f10ef6b72f136a2318824c8f30",
        "question_id": "71a3c45f7d81db3c6acd133e67e277b08f7b492ba1db8fb3ae3531123520548b",
        "question_manifest_hash": "c81a3382357488a06aa6ddaea3a999779f3dada7787dc42c27d8142914d57133",
        "study_id": "a6b8a3b2d52bbc692dc65704ddddfd843f09c823e8ff6f8007469fee1cac011e",
        "study_manifest_hash": "0b559ef0867d875da4c7bdba30f44ba72bf034284cf67f0a8fea3c2020155bdd",
        "model_run_id": "c5805e34bfd4f3661e9ad8289d8cc54785dc530731e5cb701b64e16df9e18246",
        "model_run_manifest_hash": "4502d6cefb35b3c819ea59c2bdf50874ea6cf274d831df73642de7ee7a81863b",
        "natural_record_id": "1e251f76fb293bae1b6228bcfc14c50c49bbd9f8971857fb99cc1d564eea879b",
        "checkpoint_record_id": "ba39257c156e76b1d0100107af50d6987b73f327e7e16eccebef7ea753357dd2",
        "shared_probe_id": "13582938eb479996fb846ca6f940c918f841ab22c6465865b36a8dcfe03599d1",
        "attempt_id": "05c4cdf98323649b43357c2e3446bb55ada994405ee4f76bf3d80dd0623b2ef3",
        "audit_event_id": "6460e30564fc371f5b83c4e4695a8fab8c3b980b710f24e7e7c57060fe1f2165",
    }


def test_all_machine_readable_schemas_and_configs_are_present() -> None:
    assert SCHEMA_NAMES == {
        "question_record",
        "question_manifest",
        "study_manifest",
        "model_run_manifest",
        "natural_terminal_result",
        "checkpoint_terminal_result",
        "audit_event",
        "validation_report",
    }
    assert CONFIG_NAMES == {
        "study_protocol",
        "model_run_execution",
        "dataset_materialization",
        "storage",
        "retries",
        "analysis",
    }
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].endswith(f"/{name}.schema.json")
    for name in CONFIG_NAMES:
        config = load_config(name)
        assert config["config_version"] == "1.0.0"


def test_question_and_manifest_schemas_validate_and_reject_extra_fields() -> None:
    validate_instance("question_record", question())
    with pytest.raises(ValueError, match="gold_letter"):
        validate_instance("question_record", {**question(), "gold_letter": "E"})
    with pytest.raises(ValueError, match="selected_token_log_probabilities"):
        validate_instance(
            "question_record", {**question(), "selected_token_log_probabilities": [0.1]}
        )


def test_natural_schema_enforces_terminal_and_parse_nullability() -> None:
    validate_instance("natural_terminal_result", natural())
    validate_instance("natural_terminal_result", natural(outcome="terminal_infrastructure_failure"))

    bad_failure = natural(outcome="terminal_infrastructure_failure")
    bad_failure["decoded_output"] = "partial"
    with pytest.raises(ValueError, match="decoded_output"):
        validate_instance("natural_terminal_result", bad_failure)

    bad_answer = natural()
    bad_answer["answer_parse_status"] = "missing"
    with pytest.raises(ValueError, match="natural_answer"):
        validate_instance("natural_terminal_result", bad_answer)

    missing_close = natural()
    missing_close.update(
        reasoning_status="missing_close",
        answer_parse_status="missing",
        confidence_parse_status="missing",
        terminal_answer_block_text=None,
        terminal_answer_block_span=None,
        natural_answer=None,
        natural_correct=None,
        raw_confidence_text=None,
        raw_parsed_confidence=None,
        normalized_confidence=None,
    )
    validate_instance("natural_terminal_result", missing_close)

    no_reasoning = natural()
    no_reasoning.update(
        reasoning_status="no_reasoning",
        reasoning_token_count=0,
        reasoning_text="",
        mean_reasoning_entropy_nats=None,
        tail_reasoning_entropy_nats=None,
    )
    validate_instance("natural_terminal_result", no_reasoning)

    misaligned = natural()
    misaligned["per_token_entropy_nats"] = [1.0]
    with pytest.raises(ValueError, match="aligned"):
        validate_instance("natural_terminal_result", misaligned)

    wrong_normalization = natural()
    wrong_normalization["normalized_confidence"] = 0.7
    with pytest.raises(ValueError, match="raw_parsed_confidence / 100"):
        validate_instance("natural_terminal_result", wrong_normalization)


def test_checkpoint_schema_enforces_outcome_and_measurement_nullability() -> None:
    validate_instance("checkpoint_terminal_result", checkpoint())
    validate_instance(
        "checkpoint_terminal_result", checkpoint(outcome="terminal_infrastructure_failure")
    )

    invalid = checkpoint(output_status="invalid")
    validate_instance("checkpoint_terminal_result", invalid)
    invalid["answer_entropy_nats"] = 0.3
    with pytest.raises(ValueError, match="answer_entropy_nats"):
        validate_instance("checkpoint_terminal_result", invalid)

    out_of_range = checkpoint(output_status="invalid")
    out_of_range.update(
        confidence_parse_status="out_of_range",
        raw_confidence_text="250",
        raw_parsed_confidence=250,
        normalized_confidence=None,
    )
    validate_instance("checkpoint_terminal_result", out_of_range)

    selected_logprob = {**checkpoint(), "selected_token_log_probabilities": [-0.1]}
    with pytest.raises(ValueError, match="selected_token_log_probabilities"):
        validate_instance("checkpoint_terminal_result", selected_logprob)

    out_of_range_index = checkpoint()
    out_of_range_index["requested_checkpoint_index"] = 11
    with pytest.raises(ValueError, match="requested_checkpoint_index"):
        validate_instance("checkpoint_terminal_result", out_of_range_index)

    mismatched_fraction = checkpoint()
    mismatched_fraction["requested_fraction"] = 0.4
    with pytest.raises(ValueError, match="requested fraction"):
        validate_instance("checkpoint_terminal_result", mismatched_fraction)

    wrong_normalization = checkpoint()
    wrong_normalization["normalized_confidence"] = 0.7
    with pytest.raises(ValueError, match="raw_parsed_confidence / 100"):
        validate_instance("checkpoint_terminal_result", wrong_normalization)

    contradictory_invalid = checkpoint()
    contradictory_invalid["checkpoint_model_output_status"] = "invalid"
    with pytest.raises(ValueError, match="valid triad|should not be valid"):
        validate_instance("checkpoint_terminal_result", contradictory_invalid)

    contradictory_valid = checkpoint(output_status="invalid")
    contradictory_valid["checkpoint_model_output_status"] = "valid"
    with pytest.raises(ValueError, match="answer_parse_status"):
        validate_instance("checkpoint_terminal_result", contradictory_valid)


def test_audit_event_taxonomy_is_exact() -> None:
    assert AUDIT_EVENT_TYPES == {
        "attempt_started",
        "attempt_failed",
        "attempt_interrupted",
        "attempt_completed",
        "terminal_result_recovered",
        "stale_lock_recovered",
        "trailing_line_recovered",
        "operator_unlock",
    }
    event = {
        "schema_name": "part1_audit_event",
        "schema_version": "1.0.0",
        "event_id": HEX_64,
        "study_id": HEX_64_B,
        "model_run_id": HEX_64_C,
        "question_id": "d" * 64,
        "run_id": 0,
        "checkpoint_id": None,
        "attempt_id": "e" * 64,
        "attempt_number": 1,
        "event_sequence": 0,
        "event_type": "attempt_started",
        "event_scope": "attempt",
        "shard_id": None,
        "event_timestamp": "2026-07-31T00:00:00Z",
        "execution_context": {"hostname": "node", "pid": 123},
        "outcome_category": None,
        "error_details": None,
        "retry_classification": None,
        "retry_decision": None,
        "backoff_seconds": None,
        "related_lock_owner": None,
        "terminal_record_id": None,
        "operator_reason": None,
    }
    validate_instance("audit_event", event)
    with pytest.raises(ValueError, match="event_type"):
        validate_instance("audit_event", {**event, "event_type": "lock_acquired"})

    shard_event_id = audit_event_id(
        None,
        "operator_unlock",
        4,
        study_id_value=HEX_64_B,
        model_run_id_value=HEX_64_C,
        shard_id="shard-007",
    )
    shard_event = {
        **event,
        "event_id": shard_event_id,
        "event_type": "operator_unlock",
        "event_scope": "shard",
        "shard_id": "shard-007",
        "question_id": None,
        "run_id": None,
        "checkpoint_id": None,
        "attempt_id": None,
        "attempt_number": None,
        "event_sequence": 4,
        "operator_reason": "verified worker and SLURM job are gone",
    }
    validate_instance("audit_event", shard_event)
    assert shard_event_id == "b56cca4e3208cf934bde2c299f498f77b96086fcb1193833a85e47ddabe5e4b8"

    with pytest.raises(ValueError, match="attempt_id|event_type"):
        validate_instance(
            "audit_event",
            {**shard_event, "event_scope": "attempt", "event_type": "operator_unlock"},
        )
    with pytest.raises(ValueError, match="attempt_id"):
        validate_instance("audit_event", {**shard_event, "attempt_id": "e" * 64})
    with pytest.raises(ValueError, match="operator_reason"):
        validate_instance(
            "audit_event",
            {**shard_event, "event_type": "stale_lock_recovered"},
        )
    with pytest.raises(ValueError, match="shard identity"):
        audit_event_id(
            None,
            "stale_lock_recovered",
            0,
            study_id_value=HEX_64_B,
            model_run_id_value=HEX_64_C,
            shard_id=None,
        )


def test_manifest_and_validation_report_schemas_accept_complete_examples() -> None:
    qmh = HEX_64
    fixed_subjects = [
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "high_school_psychology",
    ]
    qmanifest = {
        "schema_name": "part1_question_manifest",
        "schema_version": "1.0.0",
        "question_manifest_hash": qmh,
        "manifest_format_version": "jsonl-v1",
        "source_repository": "cais/mmlu",
        "source_revision": "immutable",
        "source_config": "all",
        "source_split": "test",
        "subjects": fixed_subjects,
        "quota_per_subject": 100,
        "total_count": 500,
        "question_sampling_seed": 42,
        "selection_algorithm_version": "part1-balanced-sample-v1",
        "canonicalization_version": CANONICAL_JSON_VERSION,
        "ordered_record_aggregation": "canonical-record-bytes-in-manifest-order-v1",
        "logical_filename": "questions.jsonl",
    }
    validate_instance("question_manifest", qmanifest)
    with pytest.raises(ValueError, match="total_count"):
        validate_instance("question_manifest", {**qmanifest, "total_count": 499})

    study = {
        "schema_name": "part1_study_manifest",
        "schema_version": "1.0.0",
        "study_id": HEX_64_B,
        "study_manifest_hash": HEX_64_C,
        "question_manifest_hash": qmh,
        "subjects": fixed_subjects,
        "subject_quotas": {subject: 100 for subject in fixed_subjects},
        "question_sampling_seed": 42,
        "scientific_protocol_version": "part1-science-v1",
        "checkpoint_fractions": [i / 10 for i in range(11)],
        "checkpoint_placement_contract": {"version": "ties-even-v1"},
        "entropy_contract": {"version": "entropy-v1"},
        "natural_answer_validity_rule": {"version": "post-close-v1"},
        "status_contract_version": "part1-status-v1",
        "calibration_contract": {"version": "calibration-v1"},
        "bootstrap_contract": {"version": "bootstrap-v1"},
        "primary_auroc_feature_registry": ["negative_mean_reasoning_entropy"],
        "within_question_analysis": {"version": "paired-v1"},
        "switching_stabilization_contract": {"version": "trajectory-v1"},
        "repetition_policy": "preserve-successful-output",
        "compatible_raw_record_schema_versions": ["1.0.0"],
        "analysis_contract_version": "part1-analysis-v1",
    }
    validate_instance("study_manifest", study)
    with pytest.raises(ValueError, match="checkpoint_fractions"):
        validate_instance("study_manifest", {**study, "checkpoint_fractions": [0.0, 1.0]})

    report = {
        "schema_name": "part1_validation_report",
        "schema_version": "1.0.0",
        "validation_report_id": HEX_64,
        "study_id": HEX_64_B,
        "model_run_id": HEX_64_C,
        "model_run_manifest_hash": "d" * 64,
        "validated_artifact_kind": "natural_shard",
        "validated_artifact_identity": "shard-000",
        "validation_started_at": "2026-07-31T00:00:00Z",
        "validation_completed_at": "2026-07-31T00:00:01Z",
        "validator_version": "part1-validator-v1",
        "is_valid": True,
        "checks": [{"name": "schema", "outcome": "passed", "details": {}}],
        "error_count": 0,
        "warning_count": 0,
        "summary": {"records": 1},
    }
    validate_instance("validation_report", report)


def test_config_templates_encode_fixed_science_and_phase1_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {name: load_config(name) for name in CONFIG_NAMES}
    protocol = configs["study_protocol"]
    assert protocol["subjects"] == [
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "high_school_psychology",
    ]
    assert protocol["quota_per_subject"] == 100
    assert protocol["natural_runs_per_question"] == 10
    assert protocol["checkpoint_fractions"] == [i / 10 for i in range(11)]
    execution = configs["model_run_execution"]
    assert execution["natural_generation"] == {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 8192,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    assert execution["checkpoint_generation"] == {"do_sample": False, "max_new_tokens": 32}
    retries = configs["retries"]
    assert retries["max_total_attempts"] == 3
    assert retries["retryable_categories"] == [
        "interrupted_process",
        "temporary_filesystem_failure",
        "transient_worker_failure",
        "transient_cuda_runtime_failure",
    ]
    assert retries["cuda_retry_requires_fresh_process"] is True
    assert retries["preserve_seed_and_logical_identity"] is True

    root = Path(configs["storage"]["smoke_root"])
    summary = validate_phase1_config(configs, mode="smoke", persistent_root=root)
    assert summary["mode"] == "smoke"
    assert summary["production_allowed"] is False
    assert summary["persistent_root"] == str(root.resolve())
    with pytest.raises(ValueError, match="production"):
        validate_phase1_config(configs, mode="production", persistent_root=root)
    with pytest.raises(ValueError, match="separate"):
        aliased = copy.deepcopy(configs)
        aliased["storage"]["production_root"] = aliased["storage"]["smoke_root"]
        validate_phase1_config(aliased, mode="smoke", persistent_root=root)

    with pytest.raises(ValueError, match="configured smoke root"):
        validate_phase1_config(
            configs,
            mode="smoke",
            persistent_root=tmp_path / "different-root",
        )

    with pytest.raises(ValueError, match="ephemeral"):
        ephemeral = copy.deepcopy(configs)
        ephemeral["storage"]["smoke_root"] = "/tmp/part1-smoke"
        validate_phase1_config(
            ephemeral,
            mode="smoke",
            persistent_root=Path("/private/tmp/../tmp/part1-smoke"),
        )

    with pytest.raises(ValueError, match="explicitly configured"):
        validate_phase1_config(configs, mode="smoke", persistent_root=None)

    with pytest.raises(ValueError, match="ephemeral"):
        slurm_tmp = copy.deepcopy(configs)
        slurm_tmp["storage"]["smoke_root"] = "$SLURM_TMPDIR/part1-smoke"
        validate_phase1_config(
            slurm_tmp,
            mode="smoke",
            persistent_root=Path("$SLURM_TMPDIR/part1-smoke"),
        )

    slurm_tmp_root = tmp_path / "slurm-job-tmp"
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm_tmp_root))
    with pytest.raises(ValueError, match="ephemeral"):
        expanded_slurm_tmp = copy.deepcopy(configs)
        expanded_slurm_tmp["storage"]["smoke_root"] = str(slurm_tmp_root / "part1-smoke")
        validate_phase1_config(
            expanded_slurm_tmp,
            mode="smoke",
            persistent_root=slurm_tmp_root / "part1-smoke",
        )

    shared_target = tmp_path / "shared-target"
    shared_target.mkdir()
    smoke_alias = tmp_path / "smoke-alias"
    production_alias = tmp_path / "production-alias"
    smoke_alias.symlink_to(shared_target, target_is_directory=True)
    production_alias.symlink_to(shared_target, target_is_directory=True)
    aliased = copy.deepcopy(configs)
    aliased["storage"]["smoke_root"] = str(smoke_alias)
    aliased["storage"]["production_root"] = str(production_alias)
    with pytest.raises(ValueError, match="separate"):
        validate_phase1_config(aliased, mode="smoke", persistent_root=smoke_alias)


@pytest.mark.parametrize(
    "config_name, field, bad_value",
    [
        ("study_protocol", "base_generation_seed", 43),
        ("study_protocol", "checkpoint_fractions", [0.0, 1.0]),
        ("model_run_execution", "natural_generation", {"do_sample": False}),
        ("retries", "retryable_categories", ["everything"]),
        ("analysis", "primary_target", "checkpoint_correct"),
    ],
)
def test_config_validation_rejects_scientific_or_retry_drift(
    tmp_path: Path, config_name: str, field: str, bad_value: object
) -> None:
    configs = {name: load_config(name) for name in CONFIG_NAMES}
    configs[config_name][field] = bad_value
    with pytest.raises(ValueError, match="fixed Part 1"):
        validate_phase1_config(
            configs,
            mode="smoke",
            persistent_root=Path(configs["storage"]["smoke_root"]),
        )


@pytest.mark.parametrize(
    "generated_path",
    [
        "results/part1-smoke/smoke-run/model_run_manifest.json",
        "results/part1-smoke/smoke-run/natural_results.jsonl",
        "results/part1/model-run/checkpoint_results.jsonl",
        "results/part1/model-run/audit_events.jsonl",
    ],
)
def test_configured_generated_result_roots_are_narrowly_ignored(generated_path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", generated_path],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert result.returncode == 0


def test_historical_result_roots_are_not_newly_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "results/20q/new-untracked-summary.json"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert result.returncode == 1
