"""Login-safe Part 1 checkpoint placement, aliases, and answer-step metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from part1_contract import (
    attempt_id,
    canonical_json_bytes,
    checkpoint_record_id,
    shared_probe_id,
    validate_instance,
)
from part1_failure_policy import classify_failure
from part1_generation import CHECKPOINT_IDS, entropy_from_logits
from part1_smollm3_adapter import (
    ADAPTER_VERSION,
    INDUCER_VERSION,
    PARSER_VERSION,
    parse_forced_output,
)


CHECKPOINT_GENERATION_SETTINGS = {"do_sample": False, "max_new_tokens": 32}


@dataclass(frozen=True)
class CheckpointPlacement:
    requested_checkpoint_index: int
    checkpoint_id: str
    requested_fraction: float
    k_keep: int
    actual_fraction: float | None
    is_alias: bool
    alias_metadata: dict[str, Any]


@dataclass(frozen=True)
class CheckpointProbePlan:
    requested_checkpoint_index: int
    checkpoint_id: str
    requested_fraction: float
    k_keep: int
    actual_fraction: float | None
    is_alias: bool
    alias_metadata: dict[str, Any]
    prefix_token_ids: tuple[int, ...]
    model_input_token_ids: tuple[int, ...]
    prefix_hash: str
    shared_probe_id: str
    inducer_version: str


@dataclass(frozen=True)
class CheckpointGenerationCapture:
    forced_generated_token_ids: tuple[int, ...]
    decoded_forced_output: str
    raw_prewarper_logits: tuple[tuple[float, ...], ...]
    answer_step_raw_logits: tuple[float, ...] | None = None


def checkpoint_placements(n_reasoning: int) -> tuple[CheckpointPlacement, ...]:
    if isinstance(n_reasoning, bool) or not isinstance(n_reasoning, int) or n_reasoning < 0:
        raise ValueError("n_reasoning must be a nonnegative integer")
    placements = [
        min(max(int(round((index / 10) * n_reasoning)), 0), n_reasoning)
        for index in range(11)
    ]
    groups: dict[int, list[str]] = {}
    for checkpoint_id_value, k_keep in zip(CHECKPOINT_IDS, placements, strict=True):
        groups.setdefault(k_keep, []).append(checkpoint_id_value)
    return tuple(
        CheckpointPlacement(
            requested_checkpoint_index=index,
            checkpoint_id=CHECKPOINT_IDS[index],
            requested_fraction=index / 10,
            k_keep=k_keep,
            actual_fraction=(k_keep / n_reasoning if n_reasoning > 0 else None),
            is_alias=CHECKPOINT_IDS[index] != groups[k_keep][0],
            alias_metadata={
                "owner_checkpoint_id": groups[k_keep][0],
                "members": list(groups[k_keep]),
            },
        )
        for index, k_keep in enumerate(placements)
    )


def _prefix_hash(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "prefix_version": "part1-token-id-prefix-v1",
                "token_ids": list(token_ids),
            }
        )
    ).hexdigest()


def build_checkpoint_probe_plans(
    parent: Mapping[str, Any],
    *,
    inducer_token_ids: Sequence[int],
    inducer_version: str,
) -> tuple[CheckpointProbePlan, ...]:
    if parent.get("natural_execution_outcome") != "complete" or not parent.get(
        "checkpoint_eligible"
    ):
        raise ValueError("checkpoint planning requires a complete eligible natural parent")
    if list(parent.get("checkpoint_ids") or []) != list(CHECKPOINT_IDS):
        raise ValueError("natural parent checkpoint IDs differ from the fixed eleven IDs")
    generated = tuple(parent["generated_token_ids"])
    prompt = tuple(parent["prompt_token_ids"])
    boundaries = parent["reasoning_boundaries"]
    reasoning_start = boundaries.get("generated_start")
    reasoning_end = boundaries.get("generated_end_exclusive")
    if (
        isinstance(reasoning_start, bool)
        or not isinstance(reasoning_start, int)
        or isinstance(reasoning_end, bool)
        or not isinstance(reasoning_end, int)
        or not 0 <= reasoning_start <= reasoning_end <= len(generated)
    ):
        raise ValueError("natural parent has invalid reasoning token boundaries")
    n_reasoning = reasoning_end - reasoning_start
    if n_reasoning != parent["reasoning_token_count"]:
        raise ValueError("natural parent reasoning count differs from token boundaries")
    inducer = tuple(int(token_id) for token_id in inducer_token_ids)
    if not inducer or any(token_id < 0 for token_id in inducer):
        raise ValueError("inducer token IDs must be nonempty and nonnegative")

    plans: list[CheckpointProbePlan] = []
    for placement in checkpoint_placements(n_reasoning):
        natural_prefix = generated[: reasoning_start + placement.k_keep]
        prefix = (*prompt, *natural_prefix)
        prefix_hash_value = _prefix_hash(prefix)
        plans.append(
            CheckpointProbePlan(
                requested_checkpoint_index=placement.requested_checkpoint_index,
                checkpoint_id=placement.checkpoint_id,
                requested_fraction=placement.requested_fraction,
                k_keep=placement.k_keep,
                actual_fraction=placement.actual_fraction,
                is_alias=placement.is_alias,
                alias_metadata=placement.alias_metadata,
                prefix_token_ids=prefix,
                model_input_token_ids=(*prefix, *inducer),
                prefix_hash=prefix_hash_value,
                shared_probe_id=shared_probe_id(
                    parent["study_id"],
                    parent["model_run_id"],
                    parent["question_id"],
                    parent["run_id"],
                    prefix_hash=prefix_hash_value,
                    inducer_version=inducer_version,
                ),
                inducer_version=inducer_version,
            )
        )
    return tuple(plans)


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    normalizer = sum(weights)
    return [weight / normalizer for weight in weights]


def choice_answer_metrics(
    full_vocabulary_logits: Sequence[float], *, ad_token_ids: Sequence[int]
) -> dict[str, Any]:
    if len(ad_token_ids) != 4 or len(set(ad_token_ids)) != 4:
        raise ValueError("A-D token IDs must contain exactly four distinct values")
    full = [float(value) for value in full_vocabulary_logits]
    if not full or any(not math.isfinite(value) for value in full):
        raise ValueError("full vocabulary logits must be nonempty and finite")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= len(full)
        for token_id in ad_token_ids
    ):
        raise ValueError("A-D token ID is outside the full vocabulary logits")
    ad_logits = [full[token_id] for token_id in ad_token_ids]
    probabilities = _softmax(ad_logits)
    answer_entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )
    return {
        "ad_logits_float32": ad_logits,
        "ad_probabilities_float32": probabilities,
        "answer_entropy_nats": answer_entropy,
        "full_vocabulary_answer_step_entropy_nats": entropy_from_logits(full),
        "maximum_ad_probability": max(probabilities),
    }


def build_checkpoint_terminal_result(
    *,
    parent: Mapping[str, Any],
    plan: CheckpointProbePlan,
    capture: CheckpointGenerationCapture,
    token_contract: Mapping[str, Any],
    gold_letter: str,
    terminal_attempt_number: int,
) -> dict[str, Any]:
    if capture.answer_step_raw_logits is None:
        if len(capture.raw_prewarper_logits) != len(capture.forced_generated_token_ids):
            raise ValueError("checkpoint raw pre-warper logits and generated tokens must be aligned")
        answer_step_logits = (
            capture.raw_prewarper_logits[0]
            if capture.raw_prewarper_logits
            else None
        )
    else:
        if capture.raw_prewarper_logits:
            raise ValueError(
                "checkpoint capture must not retain a vocabulary trace with answer-step logits"
            )
        if not capture.forced_generated_token_ids:
            raise ValueError("checkpoint answer-step logits require a generated token")
        answer_step_logits = capture.answer_step_raw_logits
    parsed = parse_forced_output(capture.decoded_forced_output)
    generated = list(capture.forced_generated_token_ids)
    ad_token_ids = list(token_contract["ad_token_ids"])
    parsed_answer = parsed.answer if parsed.answer_parse_status == "parsed" else None

    answer_token_status = "missing"
    answer_token_index: int | None = None
    answer_token_id: int | None = None
    metrics: dict[str, Any] | None = None
    if generated and generated[0] in ad_token_ids:
        candidate_letter = "ABCD"[ad_token_ids.index(generated[0])]
        if parsed_answer is not None and candidate_letter != parsed_answer:
            answer_token_status = "ambiguous"
        else:
            answer_token_status = "located"
            answer_token_index = 0
            answer_token_id = generated[0]
            if answer_step_logits is None:
                raise ValueError("located answer token is missing raw answer-step logits")
            metrics = choice_answer_metrics(
                answer_step_logits, ad_token_ids=ad_token_ids
            )

    entropy_status = "computed" if metrics is not None else "unavailable"
    output_valid = (
        parsed_answer is not None
        and answer_token_status == "located"
        and answer_token_id == ad_token_ids["ABCD".index(parsed_answer)]
        and entropy_status == "computed"
    )
    terminal_attempt = attempt_id(
        parent["study_id"],
        parent["model_run_id"],
        parent["question_id"],
        parent["run_id"],
        terminal_attempt_number,
        checkpoint_id=plan.checkpoint_id,
    )
    result: dict[str, Any] = {
        "schema_name": "part1_checkpoint_terminal_result",
        "schema_version": "1.0.0",
        "checkpoint_record_id": checkpoint_record_id(
            parent["study_id"],
            parent["model_run_id"],
            parent["question_id"],
            parent["run_id"],
            plan.checkpoint_id,
        ),
        "parent_raw_record_id": parent["raw_record_id"],
        "study_id": parent["study_id"],
        "model_run_id": parent["model_run_id"],
        "model_run_manifest_hash": parent["model_run_manifest_hash"],
        "question_manifest_hash": parent["question_manifest_hash"],
        "question_id": parent["question_id"],
        "sample_index": parent["sample_index"],
        "subject": parent["subject"],
        "run_id": parent["run_id"],
        "checkpoint_id": plan.checkpoint_id,
        "natural_seed": parent["generation_seed"],
        "terminal_attempt_number": terminal_attempt_number,
        "terminal_attempt_id": terminal_attempt,
        "infrastructure_failure_reference": None,
        "requested_checkpoint_index": plan.requested_checkpoint_index,
        "requested_fraction": plan.requested_fraction,
        "k_keep": plan.k_keep,
        "actual_fraction": plan.actual_fraction,
        "shared_probe_id": plan.shared_probe_id,
        "is_alias": plan.is_alias,
        "alias_metadata": plan.alias_metadata,
        "prefix_hash": plan.prefix_hash,
        "inducer_version": plan.inducer_version,
        "inducer_text": token_contract.get("inducer_text", "</think>\nAnswer:"),
        "forced_generated_token_ids": generated,
        "decoded_forced_output": capture.decoded_forced_output,
        "terminal_answer_block_text": parsed.terminal_answer_block_text,
        "forced_answer": parsed_answer,
        "raw_confidence_text": parsed.raw_confidence_text,
        "raw_parsed_confidence": parsed.raw_parsed_confidence,
        "normalized_confidence": parsed.normalized_confidence,
        "checkpoint_local_correct": (
            parsed_answer == gold_letter if parsed_answer is not None else None
        ),
        "answer_token_index": answer_token_index,
        "answer_token_id": answer_token_id,
        "token_convention": (
            token_contract["ad_token_convention"]
            if answer_token_status == "located"
            else None
        ),
        "ad_token_ids": ad_token_ids if answer_token_status == "located" else None,
        "ad_logits_float32": metrics["ad_logits_float32"] if metrics else None,
        "ad_probabilities_float32": (
            metrics["ad_probabilities_float32"] if metrics else None
        ),
        "answer_entropy_nats": metrics["answer_entropy_nats"] if metrics else None,
        "full_vocabulary_answer_step_entropy_nats": (
            metrics["full_vocabulary_answer_step_entropy_nats"] if metrics else None
        ),
        "maximum_ad_probability": metrics["maximum_ad_probability"] if metrics else None,
        "agrees_with_natural_answer": (
            parsed_answer == parent["natural_answer"]
            if parsed_answer is not None and parent.get("natural_answer") is not None
            else None
        ),
        "checkpoint_execution_outcome": "complete",
        "checkpoint_model_output_status": "valid" if output_valid else "invalid",
        "answer_parse_status": parsed.answer_parse_status,
        "confidence_parse_status": parsed.confidence_parse_status,
        "answer_token_status": answer_token_status,
        "entropy_status": entropy_status,
        "component_versions": {
            "adapter": ADAPTER_VERSION,
            "parser": PARSER_VERSION,
            "inducer": plan.inducer_version or INDUCER_VERSION,
            "checkpoint_metrics": "part1-answer-step-v1",
        },
        "terminal_error_details": None,
    }
    validate_instance("checkpoint_terminal_result", result)
    return result


def build_checkpoint_infrastructure_failure_result(
    *,
    parent: Mapping[str, Any],
    plan: CheckpointProbePlan,
    terminal_attempt_number: int,
    failure_category: str,
    infrastructure_failure_reference: str,
    error_details: Mapping[str, Any],
    inducer_text: str = "</think>\nAnswer:",
) -> dict[str, Any]:
    """Build the authoritative terminal record for failed checkpoint execution."""

    policy = classify_failure(failure_category, terminal_attempt_number)
    if policy.retry_decision == "retry":
        raise ValueError("retry-authorized checkpoint failure must not publish a terminal result")
    details = dict(error_details)
    if details.get("category") != failure_category:
        raise ValueError("checkpoint failure details category differs from failure category")
    terminal_attempt = attempt_id(
        parent["study_id"],
        parent["model_run_id"],
        parent["question_id"],
        parent["run_id"],
        terminal_attempt_number,
        checkpoint_id=plan.checkpoint_id,
    )
    result: dict[str, Any] = {
        "schema_name": "part1_checkpoint_terminal_result",
        "schema_version": "1.0.0",
        "checkpoint_record_id": checkpoint_record_id(
            parent["study_id"],
            parent["model_run_id"],
            parent["question_id"],
            parent["run_id"],
            plan.checkpoint_id,
        ),
        "parent_raw_record_id": parent["raw_record_id"],
        "study_id": parent["study_id"],
        "model_run_id": parent["model_run_id"],
        "model_run_manifest_hash": parent["model_run_manifest_hash"],
        "question_manifest_hash": parent["question_manifest_hash"],
        "question_id": parent["question_id"],
        "sample_index": parent["sample_index"],
        "subject": parent["subject"],
        "run_id": parent["run_id"],
        "checkpoint_id": plan.checkpoint_id,
        "natural_seed": parent["generation_seed"],
        "terminal_attempt_number": terminal_attempt_number,
        "terminal_attempt_id": terminal_attempt,
        "infrastructure_failure_reference": infrastructure_failure_reference,
        "requested_checkpoint_index": plan.requested_checkpoint_index,
        "requested_fraction": plan.requested_fraction,
        "k_keep": plan.k_keep,
        "actual_fraction": plan.actual_fraction,
        "shared_probe_id": plan.shared_probe_id,
        "is_alias": plan.is_alias,
        "alias_metadata": plan.alias_metadata,
        "prefix_hash": plan.prefix_hash,
        "inducer_version": plan.inducer_version,
        "inducer_text": inducer_text,
        "forced_generated_token_ids": None,
        "decoded_forced_output": None,
        "terminal_answer_block_text": None,
        "forced_answer": None,
        "raw_confidence_text": None,
        "raw_parsed_confidence": None,
        "normalized_confidence": None,
        "checkpoint_local_correct": None,
        "answer_token_index": None,
        "answer_token_id": None,
        "token_convention": None,
        "ad_token_ids": None,
        "ad_logits_float32": None,
        "ad_probabilities_float32": None,
        "answer_entropy_nats": None,
        "full_vocabulary_answer_step_entropy_nats": None,
        "maximum_ad_probability": None,
        "agrees_with_natural_answer": None,
        "checkpoint_execution_outcome": "terminal_infrastructure_failure",
        "checkpoint_model_output_status": "invalid",
        "answer_parse_status": "missing",
        "confidence_parse_status": "missing",
        "answer_token_status": "unsupported",
        "entropy_status": "unavailable",
        "component_versions": {
            "adapter": ADAPTER_VERSION,
            "parser": PARSER_VERSION,
            "inducer": plan.inducer_version or INDUCER_VERSION,
            "checkpoint_metrics": "part1-answer-step-v1",
        },
        "terminal_error_details": details,
    }
    validate_instance("checkpoint_terminal_result", result)
    return result


def build_alias_checkpoint_terminal_result(
    *,
    parent: Mapping[str, Any],
    owner_record: Mapping[str, Any],
    alias_plan: CheckpointProbePlan,
    terminal_attempt_number: int,
) -> dict[str, Any]:
    """Recover an alias record from its durable physical owner without a probe."""

    validate_instance("checkpoint_terminal_result", owner_record)
    if not alias_plan.is_alias:
        raise ValueError("alias recovery requires a non-owner checkpoint plan")
    if owner_record["checkpoint_id"] != alias_plan.alias_metadata["owner_checkpoint_id"]:
        raise ValueError("durable checkpoint is not the owner of the requested alias group")
    for field, expected in (
        ("parent_raw_record_id", parent["raw_record_id"]),
        ("shared_probe_id", alias_plan.shared_probe_id),
        ("prefix_hash", alias_plan.prefix_hash),
        ("k_keep", alias_plan.k_keep),
    ):
        if owner_record[field] != expected:
            raise ValueError(f"alias owner {field} differs from the requested physical probe")
    alias = dict(owner_record)
    alias.update(
        checkpoint_record_id=checkpoint_record_id(
            parent["study_id"],
            parent["model_run_id"],
            parent["question_id"],
            parent["run_id"],
            alias_plan.checkpoint_id,
        ),
        checkpoint_id=alias_plan.checkpoint_id,
        terminal_attempt_number=terminal_attempt_number,
        terminal_attempt_id=attempt_id(
            parent["study_id"],
            parent["model_run_id"],
            parent["question_id"],
            parent["run_id"],
            terminal_attempt_number,
            checkpoint_id=alias_plan.checkpoint_id,
        ),
        requested_checkpoint_index=alias_plan.requested_checkpoint_index,
        requested_fraction=alias_plan.requested_fraction,
        k_keep=alias_plan.k_keep,
        actual_fraction=alias_plan.actual_fraction,
        shared_probe_id=alias_plan.shared_probe_id,
        is_alias=True,
        alias_metadata=alias_plan.alias_metadata,
        prefix_hash=alias_plan.prefix_hash,
        inducer_version=alias_plan.inducer_version,
    )
    validate_instance("checkpoint_terminal_result", alias)
    return alias
