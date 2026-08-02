"""Build separate non-production model-run manifests after GPU preflight."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Mapping

from part1_checkpoints import CHECKPOINT_GENERATION_SETTINGS
from part1_contract import (
    FIXED_MODEL_REQUESTED_CONTRACT,
    SEED_ALGORITHM_VERSION,
    canonical_json_bytes,
    model_run_id,
    model_run_manifest_hash,
    validate_fixed_model_requested_contract,
    validate_instance,
)
from part1_generation import NATURAL_GENERATION_SETTINGS
from part1_runtime import validate_manifest_compatibility
from part1_smollm3_adapter import (
    ADAPTER_VERSION,
    INDUCER_VERSION,
    MODEL_REPOSITORY,
    PARSER_VERSION,
    PROMPT_VERSION,
    TOKENIZER_REPOSITORY,
    require_model_commit_sha,
)


_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SMOKE_SCOPES = {"smoke_a", "smoke_b", "reproducibility"}


def _exact_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _require_exact(label: str, *values: Any) -> None:
    if not values or any(not _exact_equal(values[0], value) for value in values[1:]):
        raise ValueError(f"cross-artifact {label} differs")


def _prompt_contract_hash() -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "prompt_version": PROMPT_VERSION,
                "question_format": "Question plus ordered A-D choices",
                "thinking_mode": True,
                "terminal_block": "Answer: <A|B|C|D>\\nConfidence: <integer 0-100>",
            }
        )
    ).hexdigest()


def validate_preflight_model_run_compatibility(
    *,
    preflight_report: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    question_manifest: Mapping[str, Any],
) -> None:
    """Fail closed unless all immutable preflight/model/study/question fields bind."""

    validate_instance("question_manifest", question_manifest)
    validate_manifest_compatibility(study_manifest, model_manifest)
    for field in ("study_id", "study_manifest_hash"):
        _require_exact(
            field,
            study_manifest.get(field),
            model_manifest.get(field),
            preflight_report.get(field),
        )
    _require_exact(
        "question_manifest_hash",
        question_manifest.get("question_manifest_hash"),
        study_manifest.get("question_manifest_hash"),
        model_manifest.get("question_manifest_hash"),
        preflight_report.get("question_manifest_hash"),
    )
    _require_exact(
        "question source_repository",
        question_manifest.get("source_repository"),
        study_manifest.get("question_source_repository"),
    )
    _require_exact(
        "question source_revision",
        question_manifest.get("source_revision"),
        study_manifest.get("question_source_revision"),
    )
    for field, expected in (
        ("model_repository", MODEL_REPOSITORY),
        ("tokenizer_repository", TOKENIZER_REPOSITORY),
    ):
        _require_exact(
            field,
            expected,
            model_manifest.get(field),
            preflight_report.get(field),
        )
    for field in ("model_revision", "tokenizer_revision"):
        _require_exact(
            field,
            model_manifest.get(field),
            preflight_report.get(field),
        )
    _require_exact(
        "canonical_model_identity",
        model_manifest.get("canonical_model_identity"),
        f"hf:{MODEL_REPOSITORY}@{model_manifest.get('model_revision')}",
    )

    token_contract = preflight_report.get("token_contract")
    if not isinstance(token_contract, Mapping):
        raise ValueError("cross-artifact token_contract is missing")
    for field in (
        "inducer_text",
        "inducer_token_ids",
        "reasoning_open_tag",
        "reasoning_open_token_ids",
        "reasoning_close_tag",
        "reasoning_close_token_ids",
        "ad_token_convention",
        "ad_raw_token_sequences",
        "ad_token_ids",
    ):
        _require_exact(field, model_manifest.get(field), token_contract.get(field))
    for field in (
        "effective_natural_generation",
        "effective_checkpoint_generation",
        "environment_versions",
    ):
        _require_exact(
            field,
            model_manifest.get(field),
            preflight_report.get(field),
        )

    component_versions = preflight_report.get("component_versions")
    if not isinstance(component_versions, Mapping):
        raise ValueError("cross-artifact component_versions is missing")
    for preflight_field, model_field, expected in (
        ("adapter", "adapter_version", ADAPTER_VERSION),
        ("prompt", "prompt_version", PROMPT_VERSION),
        ("parser", "parser_version", PARSER_VERSION),
        ("inducer", "inducer_version", INDUCER_VERSION),
    ):
        _require_exact(
            model_field,
            expected,
            model_manifest.get(model_field),
            component_versions.get(preflight_field),
        )


def build_smoke_model_run_manifest(
    *,
    study_manifest: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    execution_scope: str,
    base_git_commit: str,
    diff_hash: str,
) -> dict[str, Any]:
    if execution_scope not in SMOKE_SCOPES:
        raise ValueError(f"unsupported non-production execution scope: {execution_scope}")
    if _HEX_40.fullmatch(base_git_commit) is None:
        raise ValueError("base_git_commit must be a lowercase 40-character Git SHA")
    if _HEX_64.fullmatch(diff_hash) is None:
        raise ValueError("diff_hash must be a lowercase SHA-256 digest")
    validate_instance("study_manifest", study_manifest)
    model_revision = require_model_commit_sha(
        preflight_report["model_revision"], label="model_revision"
    )
    tokenizer_revision = require_model_commit_sha(
        preflight_report["tokenizer_revision"], label="tokenizer_revision"
    )
    token_contract = preflight_report["token_contract"]
    manifest: dict[str, Any] = {
        "schema_name": "part1_model_run_manifest",
        "schema_version": "1.0.0",
        "model_run_id": "",
        "model_run_manifest_hash": "",
        "execution_scope": execution_scope,
        "study_id": study_manifest["study_id"],
        "study_manifest_hash": study_manifest["study_manifest_hash"],
        "question_manifest_hash": study_manifest["question_manifest_hash"],
        "model_repository": MODEL_REPOSITORY,
        "model_revision": model_revision,
        "tokenizer_repository": TOKENIZER_REPOSITORY,
        "tokenizer_revision": tokenizer_revision,
        "canonical_model_identity": f"hf:{MODEL_REPOSITORY}@{model_revision}",
        "adapter_version": ADAPTER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _prompt_contract_hash(),
        "parser_version": PARSER_VERSION,
        "inducer_version": INDUCER_VERSION,
        "inducer_text": token_contract["inducer_text"],
        "inducer_token_ids": list(token_contract["inducer_token_ids"]),
        "reasoning_open_tag": token_contract["reasoning_open_tag"],
        "reasoning_open_token_ids": list(token_contract["reasoning_open_token_ids"]),
        "reasoning_close_tag": token_contract["reasoning_close_tag"],
        "reasoning_close_token_ids": list(token_contract["reasoning_close_token_ids"]),
        "requested_natural_generation": copy.deepcopy(NATURAL_GENERATION_SETTINGS),
        "effective_natural_generation": copy.deepcopy(
            preflight_report["effective_natural_generation"]
        ),
        "requested_checkpoint_generation": copy.deepcopy(
            CHECKPOINT_GENERATION_SETTINGS
        ),
        "effective_checkpoint_generation": copy.deepcopy(
            preflight_report["effective_checkpoint_generation"]
        ),
        "ad_token_convention": token_contract["ad_token_convention"],
        "ad_raw_token_sequences": copy.deepcopy(token_contract["ad_raw_token_sequences"]),
        "ad_token_ids": list(token_contract["ad_token_ids"]),
        "seed_algorithm_version": SEED_ALGORITHM_VERSION,
        "base_generation_seed": FIXED_MODEL_REQUESTED_CONTRACT["base_generation_seed"],
        "environment_versions": copy.deepcopy(preflight_report["environment_versions"]),
        "final_production_git_commit": None,
        "production": False,
        "smoke_git_provenance": {
            "base_commit": base_git_commit,
            "diff_hash": diff_hash,
            "production_eligible": False,
            "execution_scope": execution_scope,
        },
    }
    manifest["model_run_id"] = model_run_id(manifest)
    manifest["model_run_manifest_hash"] = model_run_manifest_hash(manifest)
    validate_instance("model_run_manifest", manifest)
    validate_fixed_model_requested_contract(manifest)
    return manifest
