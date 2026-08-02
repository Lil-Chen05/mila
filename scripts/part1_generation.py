"""Login-safe Part 1 natural-generation records and entropy science.

GPU execution supplies raw `generate(..., output_logits=True).logits` captured
before sampling warpers. This module validates and converts those captures; it
does not load a model, tokenizer, dataset, or torch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

from part1_contract import (
    SEED_ALGORITHM_VERSION,
    attempt_id,
    canonical_json_bytes,
    derive_generation_seed,
    natural_record_id,
    validate_instance,
)
from part1_failure_policy import classify_failure
from part1_smollm3_adapter import (
    ADAPTER_VERSION,
    PARSER_VERSION,
    PROMPT_VERSION,
    locate_reasoning_tokens,
    parse_natural_output,
)


NATURAL_GENERATION_SETTINGS = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 8192,
    "return_dict_in_generate": True,
    "output_logits": True,
}
CHECKPOINT_IDS = tuple(f"cp-{index:02d}" for index in range(11))


@dataclass(frozen=True)
class NaturalGenerationCapture:
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    decoded_output: str
    raw_prewarper_logits: tuple[tuple[float, ...], ...]
    stop_reason: str
    precomputed_entropy_nats: tuple[float, ...] | None = None


def entropy_from_logits(logits: Sequence[float]) -> float:
    if not logits:
        raise ValueError("entropy requires at least one vocabulary logit")
    values = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("entropy logits must be finite")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    normalizer = sum(weights)
    probabilities = [weight / normalizer for weight in weights]
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def entropy_trace_from_raw_logits(
    raw_prewarper_logits: Sequence[Sequence[float]], *, expected_tokens: int
) -> list[float]:
    if isinstance(expected_tokens, bool) or not isinstance(expected_tokens, int) or expected_tokens < 0:
        raise ValueError("expected_tokens must be a nonnegative integer")
    if len(raw_prewarper_logits) != expected_tokens:
        raise ValueError("raw pre-warper logits and generated tokens must be aligned")
    return [entropy_from_logits(step) for step in raw_prewarper_logits]


def _capture_entropy_trace(capture: NaturalGenerationCapture) -> list[float]:
    if capture.precomputed_entropy_nats is None:
        return entropy_trace_from_raw_logits(
            capture.raw_prewarper_logits,
            expected_tokens=len(capture.generated_token_ids),
        )
    values = [float(value) for value in capture.precomputed_entropy_nats]
    if len(values) != len(capture.generated_token_ids):
        raise ValueError("precomputed entropy and generated tokens must be aligned")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("precomputed entropy must be finite and nonnegative")
    if capture.raw_prewarper_logits:
        raise ValueError("capture must not retain raw logits alongside precomputed entropy")
    return values


def summarize_reasoning_entropy(
    per_token_entropy_nats: Sequence[float], *, reasoning_indices: Sequence[int]
) -> dict[str, int | float | None]:
    indices = tuple(reasoning_indices)
    if len(set(indices)) != len(indices) or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(per_token_entropy_nats)
        for index in indices
    ):
        raise ValueError("reasoning entropy indices must be unique valid token positions")
    values = [float(per_token_entropy_nats[index]) for index in indices]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("reasoning entropy values must be finite and nonnegative")
    if not values:
        return {
            "reasoning_token_count": 0,
            "mean_reasoning_entropy_nats": None,
            "tail_reasoning_entropy_nats": None,
            "tail_token_count": 0,
        }
    tail_count = max(1, math.ceil(0.10 * len(values)))
    return {
        "reasoning_token_count": len(values),
        "mean_reasoning_entropy_nats": sum(values) / len(values),
        "tail_reasoning_entropy_nats": sum(values[-tail_count:]) / tail_count,
        "tail_token_count": tail_count,
    }


def plan_ten_generation_seeds(
    *, canonical_model_identity: str, question_id: str, base_seed: int = 42
) -> dict[int, int]:
    seeds = {
        run_id: derive_generation_seed(
            base_seed=base_seed,
            canonical_model_identity=canonical_model_identity,
            question_id=question_id,
            run_id=run_id,
        )
        for run_id in range(10)
    }
    if len(set(seeds.values())) != 10:
        raise ValueError("derived run seeds unexpectedly collide within one question")
    return seeds


def _prompt_hash(prompt: str, prompt_token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "prompt_version": PROMPT_VERSION,
                "rendered_prompt": prompt,
                "prompt_token_ids": list(prompt_token_ids),
            }
        )
    ).hexdigest()


def build_natural_terminal_result(
    *,
    identity: Mapping[str, Any],
    run_id: int,
    generation_seed: int,
    terminal_attempt_number: int,
    capture: NaturalGenerationCapture,
    token_contract: Mapping[str, Any],
    decode_reasoning: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    """Convert one successful model execution into an authoritative result."""

    required_identity = (
        "study_id",
        "model_run_id",
        "model_run_manifest_hash",
        "question_manifest_hash",
        "question_id",
        "sample_index",
        "subject",
        "gold_letter",
    )
    missing = [field for field in required_identity if field not in identity]
    if missing:
        raise ValueError(f"natural result identity missing: {', '.join(missing)}")
    if run_id not in range(10):
        raise ValueError("run_id must be 0 through 9")
    generated = list(capture.generated_token_ids)
    prompt_ids = list(capture.prompt_token_ids)
    entropy_trace = _capture_entropy_trace(capture)
    location = locate_reasoning_tokens(
        generated_token_ids=generated,
        prompt_token_ids=prompt_ids,
        open_token_ids=token_contract["reasoning_open_token_ids"],
        close_token_ids=token_contract["reasoning_close_token_ids"],
    )
    entropy_summary = summarize_reasoning_entropy(
        entropy_trace, reasoning_indices=location.reasoning_indices
    )
    parsed = parse_natural_output(capture.decoded_output)
    token_close_found = bool(location.close_tag_information["found"])
    if token_close_found != parsed.reasoning_close_found:
        raise ValueError("decoded and token-ID reasoning close detection disagree")
    reasoning_text = decode_reasoning(location.reasoning_token_ids)
    if not isinstance(reasoning_text, str):
        raise ValueError("decode_reasoning must return a string")
    answer = parsed.answer if parsed.answer_parse_status == "parsed" else None
    correct = answer == identity["gold_letter"] if answer is not None else None
    terminal_attempt = attempt_id(
        identity["study_id"],
        identity["model_run_id"],
        identity["question_id"],
        run_id,
        terminal_attempt_number,
    )
    result: dict[str, Any] = {
        "schema_name": "part1_natural_terminal_result",
        "schema_version": "1.0.0",
        "raw_record_id": natural_record_id(
            identity["study_id"],
            identity["model_run_id"],
            identity["question_id"],
            run_id,
        ),
        "study_id": identity["study_id"],
        "model_run_id": identity["model_run_id"],
        "model_run_manifest_hash": identity["model_run_manifest_hash"],
        "question_manifest_hash": identity["question_manifest_hash"],
        "question_id": identity["question_id"],
        "sample_index": identity["sample_index"],
        "subject": identity["subject"],
        "run_id": run_id,
        "generation_seed": generation_seed,
        "seed_algorithm_version": SEED_ALGORITHM_VERSION,
        "terminal_attempt_number": terminal_attempt_number,
        "terminal_attempt_id": terminal_attempt,
        "infrastructure_failure_reference": None,
        "prompt_hash": _prompt_hash(capture.rendered_prompt, prompt_ids),
        "rendered_prompt": capture.rendered_prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "decoded_output": capture.decoded_output,
        "reasoning_text": reasoning_text,
        "reasoning_boundaries": location.reasoning_boundaries,
        "close_tag_information": location.close_tag_information,
        "stop_reason": capture.stop_reason,
        "generated_token_count": len(generated),
        "reasoning_token_count": entropy_summary["reasoning_token_count"],
        "per_token_entropy_nats": entropy_trace,
        "mean_reasoning_entropy_nats": entropy_summary[
            "mean_reasoning_entropy_nats"
        ],
        "tail_reasoning_entropy_nats": entropy_summary[
            "tail_reasoning_entropy_nats"
        ],
        "terminal_answer_block_text": parsed.terminal_answer_block_text,
        "terminal_answer_block_span": parsed.terminal_answer_block_span,
        "natural_answer": answer,
        "raw_confidence_text": parsed.raw_confidence_text,
        "raw_parsed_confidence": parsed.raw_parsed_confidence,
        "normalized_confidence": parsed.normalized_confidence,
        "natural_correct": correct,
        "diagnostic_answer_like_text": parsed.diagnostic_answer_like_text,
        "checkpoint_eligible": True,
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "natural_execution_outcome": "complete",
        "reasoning_status": location.reasoning_status,
        "answer_parse_status": parsed.answer_parse_status,
        "confidence_parse_status": parsed.confidence_parse_status,
        "component_versions": {
            "adapter": ADAPTER_VERSION,
            "prompt": PROMPT_VERSION,
            "parser": PARSER_VERSION,
            "entropy": "part1-raw-prewarper-entropy-v1",
        },
        "terminal_error_details": None,
    }
    validate_instance("natural_terminal_result", result)
    return result


def build_natural_infrastructure_failure_result(
    *,
    identity: Mapping[str, Any],
    run_id: int,
    generation_seed: int,
    terminal_attempt_number: int,
    prompt_hash: str,
    failure_category: str,
    infrastructure_failure_reference: str,
    error_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the authoritative terminal record for failed natural execution."""

    required_identity = (
        "study_id",
        "model_run_id",
        "model_run_manifest_hash",
        "question_manifest_hash",
        "question_id",
        "sample_index",
        "subject",
    )
    missing = [field for field in required_identity if field not in identity]
    if missing:
        raise ValueError(f"natural result identity missing: {', '.join(missing)}")
    policy = classify_failure(failure_category, terminal_attempt_number)
    if policy.retry_decision == "retry":
        raise ValueError("retry-authorized natural failure must not publish a terminal result")
    details = dict(error_details)
    if details.get("category") != failure_category:
        raise ValueError("natural failure details category differs from failure category")
    terminal_attempt = attempt_id(
        identity["study_id"],
        identity["model_run_id"],
        identity["question_id"],
        run_id,
        terminal_attempt_number,
    )
    result: dict[str, Any] = {
        "schema_name": "part1_natural_terminal_result",
        "schema_version": "1.0.0",
        "raw_record_id": natural_record_id(
            identity["study_id"],
            identity["model_run_id"],
            identity["question_id"],
            run_id,
        ),
        "study_id": identity["study_id"],
        "model_run_id": identity["model_run_id"],
        "model_run_manifest_hash": identity["model_run_manifest_hash"],
        "question_manifest_hash": identity["question_manifest_hash"],
        "question_id": identity["question_id"],
        "sample_index": identity["sample_index"],
        "subject": identity["subject"],
        "run_id": run_id,
        "generation_seed": generation_seed,
        "seed_algorithm_version": SEED_ALGORITHM_VERSION,
        "terminal_attempt_number": terminal_attempt_number,
        "terminal_attempt_id": terminal_attempt,
        "infrastructure_failure_reference": infrastructure_failure_reference,
        "prompt_hash": prompt_hash,
        "rendered_prompt": None,
        "prompt_token_ids": None,
        "generated_token_ids": None,
        "decoded_output": None,
        "reasoning_text": None,
        "reasoning_boundaries": None,
        "close_tag_information": None,
        "stop_reason": "error",
        "generated_token_count": None,
        "reasoning_token_count": None,
        "per_token_entropy_nats": None,
        "mean_reasoning_entropy_nats": None,
        "tail_reasoning_entropy_nats": None,
        "terminal_answer_block_text": None,
        "terminal_answer_block_span": None,
        "natural_answer": None,
        "raw_confidence_text": None,
        "raw_parsed_confidence": None,
        "normalized_confidence": None,
        "natural_correct": None,
        "diagnostic_answer_like_text": None,
        "checkpoint_eligible": False,
        "checkpoint_ids": None,
        "natural_execution_outcome": "terminal_infrastructure_failure",
        "reasoning_status": "malformed",
        "answer_parse_status": "missing",
        "confidence_parse_status": "missing",
        "component_versions": {
            "adapter": ADAPTER_VERSION,
            "prompt": PROMPT_VERSION,
            "parser": PARSER_VERSION,
            "entropy": "part1-raw-prewarper-entropy-v1",
        },
        "terminal_error_details": details,
    }
    validate_instance("natural_terminal_result", result)
    return result


def compare_reproducibility(
    first: NaturalGenerationCapture,
    second: NaturalGenerationCapture,
    *,
    entropy_abs_tolerance: float,
) -> dict[str, bool | float]:
    if not math.isfinite(entropy_abs_tolerance) or entropy_abs_tolerance < 0:
        raise ValueError("entropy_abs_tolerance must be finite and nonnegative")
    first_entropy = _capture_entropy_trace(first)
    second_entropy = _capture_entropy_trace(second)
    entropy_equal = len(first_entropy) == len(second_entropy) and all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=entropy_abs_tolerance)
        for left, right in zip(first_entropy, second_entropy, strict=False)
    )
    return {
        "exact_generated_token_equality": first.generated_token_ids
        == second.generated_token_ids,
        "exact_parsed_output_equality": parse_natural_output(first.decoded_output)
        == parse_natural_output(second.decoded_output),
        "entropy_array_equal_within_tolerance": entropy_equal,
        "entropy_abs_tolerance": entropy_abs_tolerance,
    }
