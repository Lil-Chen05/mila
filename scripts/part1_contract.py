"""Versioned, login-safe contracts for the Part 1 experiment.

This module deliberately imports no model, tokenizer, dataset, or torch code.
It owns canonical bytes, stable identities, deterministic generation seeds,
and loading/validation of tracked Phase 1 schemas and configuration templates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


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


def audit_event_id(attempt_id_value: str, event_type: str, event_sequence: int) -> str:
    if event_type not in AUDIT_EVENT_TYPES:
        raise ValueError(f"unsupported audit event type: {event_type}")
    if event_sequence < 0:
        raise ValueError("event_sequence must be nonnegative")
    return _identity_hash(
        "audit_event_id",
        {
            "attempt_id": attempt_id_value,
            "event_type": event_type,
            "event_sequence": event_sequence,
        },
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

    validator = Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ValueError(f"{path}: {error.message}")

    if schema_name == "question_record":
        expected_letter = "ABCD"[instance["gold_index"]]
        if instance["gold_letter"] != expected_letter:
            raise ValueError("gold_letter must agree with gold_index")
    elif schema_name == "natural_terminal_result" and instance["generated_token_ids"] is not None:
        if len(instance["generated_token_ids"]) != len(instance["per_token_entropy_nats"]):
            raise ValueError("generated_token_ids and per_token_entropy_nats must be aligned")
        if instance["generated_token_count"] != len(instance["generated_token_ids"]):
            raise ValueError("generated_token_count must equal generated_token_ids length")
    elif schema_name == "checkpoint_terminal_result":
        expected_fraction = instance["requested_checkpoint_index"] / 10
        if instance["requested_fraction"] != expected_fraction:
            raise ValueError("requested fraction must match requested_checkpoint_index / 10")
        if instance["ad_token_ids"] is not None:
            for field in ("ad_token_ids", "ad_logits_float32", "ad_probabilities_float32"):
                if len(instance[field]) != 4:
                    raise ValueError(f"{field} must contain exactly four A-D values")


def validate_phase1_config(
    configs: Mapping[str, Mapping[str, Any]], *, mode: str, persistent_root: Path
) -> dict[str, Any]:
    """Validate the tracked templates and enforce the Phase 1 no-production gate."""

    missing = CONFIG_NAMES.difference(configs)
    if missing:
        raise ValueError(f"missing configuration templates: {', '.join(sorted(missing))}")
    for name in CONFIG_NAMES:
        config = configs[name]
        if config.get("schema_name") != f"part1_{name}_config" or config.get("config_version") != "1.0.0":
            raise ValueError(f"invalid schema_name/config_version for {name}")
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
    resolved = Path(persistent_root).expanduser().resolve()
    production_root = Path(configs["storage"]["production_root"]).expanduser().resolve()
    if resolved == production_root:
        raise ValueError("smoke and production roots must be separate")
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
