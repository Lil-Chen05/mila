"""Versioned, login-safe contracts for the Part 1 experiment.

This module deliberately imports no model, tokenizer, dataset, or torch code.
It owns canonical bytes, stable identities, deterministic generation seeds,
and loading/validation of tracked Phase 1 schemas and configuration templates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


_RFC3339_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _is_rfc3339_date_time(value: object) -> bool:
    if not isinstance(value, str) or _RFC3339_DATE_TIME.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


FORMAT_CHECKER = FormatChecker()
FORMAT_CHECKER.checkers["date-time"] = (_is_rfc3339_date_time, ())


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIRECTORY = REPOSITORY_ROOT / "schemas" / "part1"
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs" / "part1"

CANONICAL_JSON_VERSION = "part1-canonical-json-v1"
IDENTITY_VERSION = "part1-identity-v1"
SEED_ALGORITHM_VERSION = "part1-seed-v1"
PYTORCH_SEED_MAX = 2**63 - 1

SCHEMA_NAMES = {
    "question_record",
    "question_manifest",
    "study_manifest",
    "model_run_manifest",
    "natural_terminal_result",
    "checkpoint_terminal_result",
    "audit_event",
    "validation_report",
}
CONFIG_NAMES = {
    "study_protocol",
    "model_run_execution",
    "dataset_materialization",
    "storage",
    "retries",
    "analysis",
}
AUDIT_EVENT_TYPES = {
    "attempt_started",
    "attempt_failed",
    "attempt_interrupted",
    "attempt_completed",
    "terminal_result_recovered",
    "stale_lock_recovered",
    "trailing_line_recovered",
    "operator_unlock",
}
ATTEMPT_AUDIT_EVENT_TYPES = {
    "attempt_started",
    "attempt_failed",
    "attempt_interrupted",
    "attempt_completed",
    "terminal_result_recovered",
}
SHARD_AUDIT_EVENT_TYPES = {
    "stale_lock_recovered",
    "trailing_line_recovered",
    "operator_unlock",
}
FIXED_SUBJECTS = [
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "high_school_psychology",
]
FIXED_CHECKPOINT_FRACTIONS = [index / 10 for index in range(11)]
FIXED_NATURAL_GENERATION = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 8192,
    "return_dict_in_generate": True,
    "output_logits": True,
}
FIXED_RETRYABLE_CATEGORIES = [
    "interrupted_process",
    "temporary_filesystem_failure",
    "transient_worker_failure",
    "transient_cuda_runtime_failure",
]
FIXED_PRIMARY_AUROC_FEATURE_REGISTRY = [
    "negative_mean_reasoning_entropy",
    "negative_tail_reasoning_entropy",
    "negative_answer_entropy_fraction_0.0",
    "negative_answer_entropy_fraction_0.5",
    "negative_answer_entropy_fraction_1.0",
    "natural_verbalized_confidence",
    "maximum_ad_probability_fraction_0.0",
    "maximum_ad_probability_fraction_0.5",
    "maximum_ad_probability_fraction_1.0",
    "negative_answer_switch_count",
    "negative_stabilization_fraction",
]
FIXED_STUDY_CONTRACT = {
    "subjects": FIXED_SUBJECTS,
    "subject_quotas": {subject: 100 for subject in FIXED_SUBJECTS},
    "question_sampling_seed": 42,
    "scientific_protocol_version": "part1-science-v1",
    "checkpoint_fractions": FIXED_CHECKPOINT_FRACTIONS,
    "checkpoint_placement_contract": {
        "version": "ties-even-v1",
        "formula": "clamp(round(requested_fraction*n_reasoning),0,n_reasoning)",
        "actual_fraction_zero_reasoning": None,
    },
    "entropy_contract": {
        "version": "entropy-v1",
        "natural": "raw_full_vocabulary_reasoning_token_entropy_nats",
        "checkpoint": "renormalized_ad_answer_entropy_nats",
        "tail": "last_20_percent_reasoning_tokens",
    },
    "natural_answer_validity_rule": {
        "version": "post-close-v1",
        "region": "after_first_valid_reasoning_close",
        "domain": ["A", "B", "C", "D"],
    },
    "status_contract_version": "part1-status-v1",
    "calibration_contract": {
        "version": "calibration-v1",
        "ece_bins": 10,
        "main_checkpoint_fractions": [0.0, 0.5, 1.0],
        "all_checkpoint_fractions": FIXED_CHECKPOINT_FRACTIONS,
    },
    "bootstrap_contract": {
        "version": "bootstrap-v1",
        "seed": 42,
        "development_replicates": 1000,
        "final_replicates": 5000,
        "confidence_interval_percent": 95,
        "minimum_valid_fraction": 0.95,
        "unit": "subject_stratified_question",
    },
    "primary_auroc_feature_registry": FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
    "within_question_analysis": {
        "version": "paired-v1",
        "eligibility": "at_least_one_correct_and_one_incorrect_natural_run",
        "question_weighting": "equal",
    },
    "switching_stabilization_contract": {
        "version": "trajectory-v1",
        "aliases_are_transitions": False,
        "missing_breaks_adjacency": True,
        "stabilization_reference": "final_valid_forced_checkpoint_answer",
    },
    "repetition_policy": "preserve-successful-output",
    "compatible_raw_record_schema_versions": ["1.0.0"],
    "analysis_contract_version": "part1-analysis-v1",
}
FIXED_MODEL_REQUESTED_CONTRACT = {
    "model_repository": "HuggingFaceTB/SmolLM3-3B",
    "tokenizer_repository": "HuggingFaceTB/SmolLM3-3B",
    "requested_natural_generation": FIXED_NATURAL_GENERATION,
    "requested_checkpoint_generation": {"do_sample": False, "max_new_tokens": 32},
    "seed_algorithm_version": SEED_ALGORITHM_VERSION,
    "base_generation_seed": 42,
}
_FIXED_PHASE1_CONFIGS = {
    "study_protocol": {
        "schema_name": "part1_study_protocol_config",
        "config_version": "1.0.0",
        "scientific_protocol_version": "part1-science-v1",
        "question_sampling_seed": 42,
        "base_generation_seed": 42,
        "bootstrap_seed": 42,
        "subjects": FIXED_SUBJECTS,
        "quota_per_subject": 100,
        "natural_runs_per_question": 10,
        "run_ids": list(range(10)),
        "checkpoint_fractions": FIXED_CHECKPOINT_FRACTIONS,
        "primary_target": "natural_correct",
    },
    "model_run_execution": {
        "schema_name": "part1_model_run_execution_config",
        "config_version": "1.0.0",
        "mode": "smoke",
        "production": False,
        "model_repository": "HuggingFaceTB/SmolLM3-3B",
        "model_revision": None,
        "tokenizer_revision": None,
        "thinking_mode": True,
        "dtype": "bfloat16",
        "batch_size": 1,
        "natural_generation": FIXED_NATURAL_GENERATION,
        "checkpoint_generation": {"do_sample": False, "max_new_tokens": 32},
        "immutable_revision_required_before_execution": True,
    },
    "dataset_materialization": {
        "schema_name": "part1_dataset_materialization_config",
        "config_version": "1.0.0",
        "source_repository": "cais/mmlu",
        "source_revision": None,
        "source_config": "all",
        "source_split": "test",
        "streaming": True,
        "bounded_take_required": True,
        "question_sampling_seed": 42,
        "subjects": FIXED_SUBJECTS,
        "quota_per_subject": 100,
        "source_revision_required_before_materialization": True,
    },
    "storage": {
        "schema_name": "part1_storage_config",
        "config_version": "1.0.0",
        "phase1_production_allowed": False,
        "persistent_root_required": True,
        "smoke_root": "results/part1-smoke",
        "production_root": "results/part1",
        "natural_shard_filename": "natural_results.jsonl",
        "checkpoint_shard_filename": "checkpoint_results.jsonl",
        "audit_filename": "audit_events.jsonl",
        "validation_report_directory": "validation_reports",
        "active_shards_append_only": True,
        "immutable_after_finalization": True,
        "ephemeral_roots_forbidden": ["$SLURM_TMPDIR", "/tmp", "/private/tmp"],
    },
    "retries": {
        "schema_name": "part1_retries_config",
        "config_version": "1.0.0",
        "max_total_attempts": 3,
        "attempt_numbers": [1, 2, 3],
        "retryable_categories": FIXED_RETRYABLE_CATEGORIES,
        "terminal_categories": [
            "invalid_configuration",
            "schema_incompatibility",
            "manifest_incompatibility",
            "tokenizer_preflight_incompatibility",
            "deterministic_context_overflow",
            "reproducible_cuda_oom",
            "unsupported_model_or_tokenizer_behaviour",
            "corrupt_immutable_manifest",
        ],
        "cuda_retry_requires_fresh_process": True,
        "preserve_seed_and_logical_identity": True,
        "backoff_seconds": [0, 30, 120],
    },
    "analysis": {
        "schema_name": "part1_analysis_config",
        "config_version": "1.0.0",
        "analysis_contract_version": "part1-analysis-v1",
        "primary_target": "natural_correct",
        "primary_auroc_feature_registry": FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
        "bootstrap_seed": 42,
        "development_bootstrap_replicates": 1000,
        "final_bootstrap_replicates": 5000,
        "confidence_interval_percent": 95,
        "minimum_valid_bootstrap_fraction": 0.95,
        "ece_bins": 10,
        "main_checkpoint_fractions": [0.0, 0.5, 1.0],
        "all_checkpoint_fractions": FIXED_CHECKPOINT_FRACTIONS,
    },
}

QUESTION_CONTENT_FIELDS = (
    "source_repository",
    "source_revision",
    "source_config",
    "source_split",
    "source_row_identity",
    "question",
    "choices",
    "gold_index",
    "gold_letter",
)
QUESTION_ID_FIELDS = (
    "source_repository",
    "source_revision",
    "source_config",
    "source_split",
    "source_row_identity",
    "question",
    "choices",
    "gold_index",
    "gold_letter",
)
QUESTION_RECORD_IMMUTABLE_FIELDS = (
    "schema_name",
    "schema_version",
    "question_id",
    "question_content_hash",
    "sample_index",
    "subject",
    "subject_selection_index",
    *QUESTION_CONTENT_FIELDS,
)
QUESTION_MANIFEST_FIELDS = (
    "schema_name",
    "schema_version",
    "manifest_format_version",
    "source_repository",
    "source_revision",
    "source_config",
    "source_split",
    "subjects",
    "quota_per_subject",
    "total_count",
    "question_sampling_seed",
    "selection_algorithm_version",
    "canonicalization_version",
    "ordered_record_aggregation",
    "logical_filename",
)
STUDY_FIELDS = (
    "schema_name",
    "schema_version",
    "question_manifest_hash",
    "subjects",
    "subject_quotas",
    "question_sampling_seed",
    "scientific_protocol_version",
    "checkpoint_fractions",
    "checkpoint_placement_contract",
    "entropy_contract",
    "natural_answer_validity_rule",
    "status_contract_version",
    "calibration_contract",
    "bootstrap_contract",
    "primary_auroc_feature_registry",
    "within_question_analysis",
    "switching_stabilization_contract",
    "repetition_policy",
    "compatible_raw_record_schema_versions",
    "analysis_contract_version",
)
STUDY_ID_FIELDS = (
    "question_manifest_hash",
    "subjects",
    "subject_quotas",
    "question_sampling_seed",
    "scientific_protocol_version",
    "checkpoint_fractions",
    "checkpoint_placement_contract",
    "entropy_contract",
    "natural_answer_validity_rule",
    "status_contract_version",
    "calibration_contract",
    "bootstrap_contract",
    "primary_auroc_feature_registry",
    "within_question_analysis",
    "switching_stabilization_contract",
    "repetition_policy",
    "analysis_contract_version",
)
MODEL_RUN_FIELDS = (
    "schema_name",
    "schema_version",
    "study_id",
    "study_manifest_hash",
    "question_manifest_hash",
    "model_repository",
    "model_revision",
    "tokenizer_repository",
    "tokenizer_revision",
    "canonical_model_identity",
    "adapter_version",
    "prompt_version",
    "prompt_hash",
    "parser_version",
    "inducer_version",
    "inducer_text",
    "inducer_token_ids",
    "reasoning_open_tag",
    "reasoning_open_token_ids",
    "reasoning_close_tag",
    "reasoning_close_token_ids",
    "requested_natural_generation",
    "effective_natural_generation",
    "requested_checkpoint_generation",
    "effective_checkpoint_generation",
    "ad_token_convention",
    "ad_raw_token_sequences",
    "ad_token_ids",
    "seed_algorithm_version",
    "base_generation_seed",
    "environment_versions",
    "final_production_git_commit",
    "production",
    "smoke_git_provenance",
)
MODEL_RUN_ID_FIELDS = (
    "study_id",
    "study_manifest_hash",
    "question_manifest_hash",
    "model_repository",
    "model_revision",
    "tokenizer_repository",
    "tokenizer_revision",
    "canonical_model_identity",
    "adapter_version",
    "prompt_version",
    "prompt_hash",
    "parser_version",
    "inducer_version",
    "inducer_text",
    "inducer_token_ids",
    "reasoning_open_tag",
    "reasoning_open_token_ids",
    "reasoning_close_tag",
    "reasoning_close_token_ids",
    "requested_natural_generation",
    "effective_natural_generation",
    "requested_checkpoint_generation",
    "effective_checkpoint_generation",
    "ad_token_convention",
    "ad_raw_token_sequences",
    "ad_token_ids",
    "seed_algorithm_version",
    "base_generation_seed",
)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON requires finite floating-point values")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized_key = _normalize(key)
            if normalized_key in normalized:
                raise ValueError("line-ending normalization produced a duplicate object key")
            normalized[normalized_key] = _normalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: Any, *, serialization_version: str = CANONICAL_JSON_VERSION) -> bytes:
    """Return canonical UTF-8 JSON bytes in a versioned envelope.

    Object keys are recursively sorted, list order is preserved, line endings
    in all strings are normalized to LF, Unicode is emitted directly, and
    non-finite numbers are rejected. The returned bytes have no trailing LF.
    """

    envelope = {
        "serialization_version": _normalize(serialization_version),
        "value": _normalize(value),
    }
    try:
        rendered = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except ValueError as exc:
        raise ValueError("canonical JSON requires finite numbers") from exc
    return rendered.encode("utf-8")


def _select(source: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    missing = [field for field in fields if field not in source]
    if missing:
        raise ValueError(f"identity payload missing required fields: {', '.join(missing)}")
    return {field: source[field] for field in fields}


def _identity_hash(identity_type: str, payload: Mapping[str, Any], *, version: str = IDENTITY_VERSION) -> str:
    envelope = {
        "identity_type": identity_type,
        "identity_version": version,
        "payload": payload,
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def question_content_hash(record: Mapping[str, Any]) -> str:
    return _identity_hash("question_content_hash", _select(record, QUESTION_CONTENT_FIELDS))


def question_id(record: Mapping[str, Any]) -> str:
    return _identity_hash("question_id", _select(record, QUESTION_ID_FIELDS))


def question_manifest_hash(manifest: Mapping[str, Any], ordered_records: Sequence[Mapping[str, Any]]) -> str:
    records = [_select(record, QUESTION_RECORD_IMMUTABLE_FIELDS) for record in ordered_records]
    payload = {"manifest": _select(manifest, QUESTION_MANIFEST_FIELDS), "ordered_records": records}
    return _identity_hash("question_manifest_hash", payload)


def study_id(manifest: Mapping[str, Any]) -> str:
    return _identity_hash("study_id", _select(manifest, STUDY_ID_FIELDS))


def study_manifest_hash(manifest: Mapping[str, Any]) -> str:
    return _identity_hash("study_manifest_hash", _select(manifest, STUDY_FIELDS))


def model_run_id(manifest: Mapping[str, Any]) -> str:
    return _identity_hash("model_run_id", _select(manifest, MODEL_RUN_ID_FIELDS))


def model_run_manifest_hash(manifest: Mapping[str, Any]) -> str:
    return _identity_hash("model_run_manifest_hash", _select(manifest, MODEL_RUN_FIELDS))


def natural_record_id(study_id_value: str, model_run_id_value: str, question_id_value: str, run_id: int) -> str:
    return _identity_hash(
        "natural_record_id",
        {
            "study_id": study_id_value,
            "model_run_id": model_run_id_value,
            "question_id": question_id_value,
            "run_id": run_id,
        },
    )


def checkpoint_record_id(
    study_id_value: str,
    model_run_id_value: str,
    question_id_value: str,
    run_id: int,
    checkpoint_id: str,
) -> str:
    return _identity_hash(
        "checkpoint_record_id",
        {
            "study_id": study_id_value,
            "model_run_id": model_run_id_value,
            "question_id": question_id_value,
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
        },
    )


def shared_probe_id(
    study_id_value: str,
    model_run_id_value: str,
    question_id_value: str,
    run_id: int,
    *,
    prefix_hash: str,
    inducer_version: str,
) -> str:
    """Identify one physical checkpoint probe independently of alias owners."""

    return _identity_hash(
        "shared_probe_id",
        {
            "study_id": study_id_value,
            "model_run_id": model_run_id_value,
            "question_id": question_id_value,
            "run_id": run_id,
            "prefix_hash": prefix_hash,
            "inducer_version": inducer_version,
        },
    )


def attempt_id(
    study_id_value: str,
    model_run_id_value: str,
    question_id_value: str,
    run_id: int,
    attempt_number: int,
    *,
    checkpoint_id: str | None = None,
) -> str:
    if attempt_number < 1:
        raise ValueError("attempt_number must start at 1")
    payload: dict[str, Any] = {
        "study_id": study_id_value,
        "model_run_id": model_run_id_value,
        "question_id": question_id_value,
        "run_id": run_id,
        "work_kind": "checkpoint" if checkpoint_id is not None else "natural",
        "attempt_number": attempt_number,
    }
    if checkpoint_id is not None:
        payload["checkpoint_id"] = checkpoint_id
    return _identity_hash("attempt_id", payload)


def audit_event_id(
    attempt_id_value: str | None,
    event_type: str,
    event_sequence: int,
    *,
    study_id_value: str | None = None,
    model_run_id_value: str | None = None,
    shard_id: str | None = None,
) -> str:
    if event_type not in AUDIT_EVENT_TYPES:
        raise ValueError(f"unsupported audit event type: {event_type}")
    if event_sequence < 0:
        raise ValueError("event_sequence must be nonnegative")
    if event_type in ATTEMPT_AUDIT_EVENT_TYPES:
        if attempt_id_value is None:
            raise ValueError("attempt-scoped event requires attempt identity")
        payload = {
            "attempt_id": attempt_id_value,
            "event_type": event_type,
            "event_sequence": event_sequence,
        }
    elif event_type in SHARD_AUDIT_EVENT_TYPES:
        if attempt_id_value is not None:
            raise ValueError("shard-scoped event must not use attempt identity")
        if not study_id_value or not model_run_id_value or not shard_id:
            raise ValueError("shard-scoped event requires complete shard identity")
        payload = {
            "study_id": study_id_value,
            "model_run_id": model_run_id_value,
            "shard_id": shard_id,
            "event_type": event_type,
            "event_sequence": event_sequence,
        }
    else:  # AUDIT_EVENT_TYPES is intentionally partitioned by the two sets.
        raise ValueError(f"audit event type has no scope: {event_type}")
    return _identity_hash(
        "audit_event_id",
        payload,
    )


def derive_generation_seed(
    *,
    base_seed: int,
    canonical_model_identity: str,
    question_id: str,
    run_id: int,
    algorithm_version: str = SEED_ALGORITHM_VERSION,
) -> int:
    """Derive a stable seed in PyTorch's nonnegative signed-64-bit range."""

    if base_seed < 0 or run_id < 0:
        raise ValueError("base_seed and run_id must be nonnegative")
    payload = {
        "seed_algorithm_version": algorithm_version,
        "base_seed": base_seed,
        "canonical_model_identity": canonical_model_identity,
        "question_id": question_id,
        "run_id": run_id,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & PYTORCH_SEED_MAX


def load_schema(name: str) -> dict[str, Any]:
    if name not in SCHEMA_NAMES:
        raise ValueError(f"unknown Part 1 schema: {name}")
    with (SCHEMA_DIRECTORY / f"{name}.schema.json").open(encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def load_config(name: str) -> dict[str, Any]:
    if name not in CONFIG_NAMES:
        raise ValueError(f"unknown Part 1 config: {name}")
    with (CONFIG_DIRECTORY / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_instance(schema_name: str, instance: Mapping[str, Any]) -> None:
    """Validate an instance and raise one concise, field-oriented ValueError."""

    if schema_name in {"natural_terminal_result", "checkpoint_terminal_result"} and (
        instance.get("confidence_parse_status") == "out_of_range"
        and isinstance(instance.get("raw_parsed_confidence"), int)
        and 0 <= instance["raw_parsed_confidence"] <= 100
    ):
        raise ValueError(
            "out-of-range confidence requires raw_parsed_confidence <= -1 or >= 101"
        )
    validator = Draft202012Validator(load_schema(schema_name), format_checker=FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ValueError(f"{path}: {error.message}")

    if schema_name in {"natural_terminal_result", "checkpoint_terminal_result"}:
        if instance["confidence_parse_status"] == "parsed":
            expected_confidence = instance["raw_parsed_confidence"] / 100
            if not math.isclose(
                instance["normalized_confidence"],
                expected_confidence,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "normalized_confidence must equal raw_parsed_confidence / 100"
                )

        confidence_status = instance["confidence_parse_status"]
        if confidence_status == "missing" and (
            instance["raw_confidence_text"] is not None
            or instance["raw_parsed_confidence"] is not None
            or instance["normalized_confidence"] is not None
        ):
            raise ValueError("missing confidence requires all confidence fields to be null")
        if confidence_status == "malformed" and (
            instance["raw_parsed_confidence"] is not None
            or instance["normalized_confidence"] is not None
        ):
            raise ValueError("malformed confidence cannot retain a parsed integer")

    if schema_name == "question_record":
        expected_letter = "ABCD"[instance["gold_index"]]
        if instance["gold_letter"] != expected_letter:
            raise ValueError("gold_letter must agree with gold_index")
    elif schema_name == "natural_terminal_result":
        if instance["natural_execution_outcome"] == "terminal_infrastructure_failure":
            if instance["diagnostic_answer_like_text"] is not None:
                raise ValueError("infrastructure failure diagnostic output must be null")
        else:
            assert instance["generated_token_ids"] is not None
            if len(instance["generated_token_ids"]) != len(instance["per_token_entropy_nats"]):
                raise ValueError("generated_token_ids and per_token_entropy_nats must be aligned")
            if instance["generated_token_count"] != len(instance["generated_token_ids"]):
                raise ValueError("generated_token_count must equal generated_token_ids length")
            if instance["reasoning_status"] == "no_reasoning":
                if instance["reasoning_token_count"] != 0 or any(
                    instance[field] is not None
                    for field in (
                        "mean_reasoning_entropy_nats",
                        "tail_reasoning_entropy_nats",
                    )
                ):
                    raise ValueError("no_reasoning requires zero count and null summaries")
            elif (
                instance["reasoning_token_count"] is None
                or instance["reasoning_token_count"] < 1
                or instance["mean_reasoning_entropy_nats"] is None
                or instance["tail_reasoning_entropy_nats"] is None
            ):
                raise ValueError(
                    "nonempty complete reasoning requires count and entropy summaries"
                )
            if (
                instance["reasoning_status"] != "missing_close"
                and instance["diagnostic_answer_like_text"] is not None
            ):
                raise ValueError("diagnostic answer-like text is only valid for missing_close")
    elif schema_name == "checkpoint_terminal_result":
        if instance["checkpoint_model_output_status"] == "invalid" and (
            instance["answer_parse_status"] == "parsed"
            and instance["answer_token_status"] == "located"
            and instance["entropy_status"] == "computed"
        ):
            raise ValueError(
                "invalid checkpoint output cannot retain the complete valid triad"
            )
        expected_fraction = instance["requested_checkpoint_index"] / 10
        if instance["requested_fraction"] != expected_fraction:
            raise ValueError("requested fraction must match requested_checkpoint_index / 10")
        if instance["ad_token_ids"] is not None:
            for field in ("ad_token_ids", "ad_logits_float32", "ad_probabilities_float32"):
                if len(instance[field]) != 4:
                    raise ValueError(f"{field} must contain exactly four A-D values")


def _fixed_value_matches(actual: Any, expected: Any) -> bool:
    try:
        return canonical_json_bytes({"value": actual}) == canonical_json_bytes(
            {"value": expected}
        )
    except (TypeError, ValueError):
        return False


def validate_fixed_study_contract(study_manifest: Mapping[str, Any]) -> None:
    for field, expected in FIXED_STUDY_CONTRACT.items():
        if not _fixed_value_matches(study_manifest.get(field), expected):
            if field == "compatible_raw_record_schema_versions":
                raise ValueError(
                    "study_manifest.compatible_raw_record_schema_versions differs from "
                    "the fixed Part 1 raw record schema contract"
                )
            raise ValueError(f"study_manifest.{field} differs from the fixed Part 1 contract")


def validate_fixed_model_requested_contract(model_manifest: Mapping[str, Any]) -> None:
    for field, expected in FIXED_MODEL_REQUESTED_CONTRACT.items():
        if not _fixed_value_matches(model_manifest.get(field), expected):
            raise ValueError(f"model_run_manifest.{field} differs from the fixed Part 1 requested contract")
    for field, required in (
        ("effective_natural_generation", FIXED_NATURAL_GENERATION),
        (
            "effective_checkpoint_generation",
            FIXED_MODEL_REQUESTED_CONTRACT["requested_checkpoint_generation"],
        ),
    ):
        effective = model_manifest.get(field)
        if not isinstance(effective, Mapping):
            raise ValueError(f"model_run_manifest.{field} must be an object")
        for setting, expected in required.items():
            if setting not in effective:
                raise ValueError(
                    f"model_run_manifest.{field} is missing required setting {setting}"
                )
            if not _fixed_value_matches(effective[setting], expected):
                raise ValueError(
                    f"model_run_manifest.{field} required setting {setting} differs"
                )
        try:
            canonical_json_bytes(effective)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model_run_manifest.{field} contains a non-serializable resolved setting"
            ) from exc


def validate_phase1_config(
    configs: Mapping[str, Mapping[str, Any]], *, mode: str, persistent_root: Path | None
) -> dict[str, Any]:
    """Validate the tracked templates and enforce the Phase 1 no-production gate."""

    missing = CONFIG_NAMES.difference(configs)
    if missing:
        raise ValueError(f"missing configuration templates: {', '.join(sorted(missing))}")
    for name in CONFIG_NAMES:
        config = configs[name]
        if config.get("schema_name") != f"part1_{name}_config" or config.get("config_version") != "1.0.0":
            raise ValueError(f"invalid schema_name/config_version for {name}")
        expected_config = _FIXED_PHASE1_CONFIGS[name]
        ignored_fields = {"smoke_root", "production_root"} if name == "storage" else set()
        if set(config).difference(ignored_fields) != set(expected_config).difference(
            ignored_fields
        ):
            raise ValueError(f"{name}.fields differ from the fixed Part 1 contract")
        for field, expected_value in expected_config.items():
            if field not in ignored_fields and not _fixed_value_matches(
                config.get(field), expected_value
            ):
                raise ValueError(f"{name}.{field} differs from the fixed Part 1 contract")
    protocol = configs["study_protocol"]
    fixed_protocol = {
        "question_sampling_seed": 42,
        "base_generation_seed": 42,
        "bootstrap_seed": 42,
        "subjects": FIXED_SUBJECTS,
        "quota_per_subject": 100,
        "natural_runs_per_question": 10,
        "run_ids": list(range(10)),
        "checkpoint_fractions": FIXED_CHECKPOINT_FRACTIONS,
        "primary_target": "natural_correct",
    }
    execution = configs["model_run_execution"]
    fixed_execution = {
        "production": False,
        "model_repository": "HuggingFaceTB/SmolLM3-3B",
        "thinking_mode": True,
        "dtype": "bfloat16",
        "batch_size": 1,
        "natural_generation": FIXED_NATURAL_GENERATION,
        "checkpoint_generation": {"do_sample": False, "max_new_tokens": 32},
    }
    for name, config, expected in (
        ("study_protocol", protocol, fixed_protocol),
        ("model_run_execution", execution, fixed_execution),
    ):
        for field, expected_value in expected.items():
            if config.get(field) != expected_value:
                raise ValueError(f"{name}.{field} differs from the fixed Part 1 contract")
    if configs["retries"].get("retryable_categories") != FIXED_RETRYABLE_CATEGORIES:
        raise ValueError("retries.retryable_categories differs from the fixed Part 1 contract")
    if configs["analysis"].get("primary_target") != "natural_correct":
        raise ValueError("analysis.primary_target differs from the fixed Part 1 contract")
    if mode not in {"smoke", "production"}:
        raise ValueError("mode must be smoke or production")
    if mode == "production" or configs["storage"].get("phase1_production_allowed") is not False:
        raise ValueError("production execution is forbidden during Phase 1")
    if persistent_root is None:
        raise ValueError("persistent_root must be explicitly configured")
    storage = configs["storage"]
    resolved = Path(persistent_root).expanduser().resolve()
    configured_smoke_root = Path(storage["smoke_root"]).expanduser().resolve()
    production_root = Path(storage["production_root"]).expanduser().resolve()
    if configured_smoke_root == production_root:
        raise ValueError("smoke and production roots must be separate")
    raw_root_values = (str(persistent_root), str(storage["smoke_root"]))
    for forbidden in storage["ephemeral_roots_forbidden"]:
        if forbidden.startswith("$"):
            if any(forbidden in value for value in raw_root_values):
                raise ValueError(f"ephemeral root {forbidden} is forbidden")
            environment_root = os.environ.get(forbidden[1:])
            if environment_root:
                expanded_forbidden_root = Path(environment_root).expanduser().resolve()
                if resolved == expanded_forbidden_root or resolved.is_relative_to(
                    expanded_forbidden_root
                ):
                    raise ValueError(
                        f"ephemeral root alias {expanded_forbidden_root} is forbidden"
                    )
            continue
        forbidden_root = Path(forbidden).expanduser().resolve()
        if resolved == forbidden_root or resolved.is_relative_to(forbidden_root):
            raise ValueError(f"ephemeral root {forbidden_root} is forbidden")
    if resolved != configured_smoke_root:
        raise ValueError("persistent_root must equal the configured smoke root")
    if configs["retries"].get("max_total_attempts") != 3:
        raise ValueError("retry policy differs from the fixed Part 1 three-attempt contract")
    return {
        "mode": mode,
        "production_allowed": False,
        "persistent_root": str(resolved),
        "canonicalization_version": CANONICAL_JSON_VERSION,
        "identity_version": IDENTITY_VERSION,
        "seed_algorithm_version": SEED_ALGORITHM_VERSION,
        "max_total_attempts": 3,
    }
