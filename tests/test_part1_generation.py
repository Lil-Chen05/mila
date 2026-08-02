"""Pure natural-generation science/control tests; no real model is loaded."""

from __future__ import annotations

import math

import pytest

from part1_contract import derive_generation_seed, validate_instance
from part1_generation import (
    NaturalGenerationCapture,
    build_natural_terminal_result,
    compare_reproducibility,
    entropy_from_logits,
    entropy_trace_from_raw_logits,
    plan_ten_generation_seeds,
    summarize_reasoning_entropy,
)


TOKEN_CONTRACT = {
    "reasoning_open_token_ids": [10],
    "reasoning_close_token_ids": [11],
}
IDENTITY = {
    "study_id": "a" * 64,
    "model_run_id": "b" * 64,
    "model_run_manifest_hash": "c" * 64,
    "question_manifest_hash": "d" * 64,
    "question_id": "e" * 64,
    "sample_index": 0,
    "subject": "high_school_mathematics",
    "gold_letter": "C",
}


def logits_for(tokens: int) -> list[list[float]]:
    return [[float(index), -float(index), 0.25] for index in range(1, tokens + 1)]


def capture(
    *,
    generated: list[int] | None = None,
    decoded: str = "<think>one two</think>\nAnswer: C\nConfidence: 80",
    stop_reason: str = "eos",
) -> NaturalGenerationCapture:
    generated_ids = generated or [10, 100, 101, 11, 20, 21]
    return NaturalGenerationCapture(
        rendered_prompt="prompt ending in assistant generation marker",
        prompt_token_ids=(1, 2),
        generated_token_ids=tuple(generated_ids),
        decoded_output=decoded,
        raw_prewarper_logits=tuple(tuple(step) for step in logits_for(len(generated_ids))),
        stop_reason=stop_reason,
    )


def test_entropy_uses_full_vocabulary_natural_log_and_preserves_precision() -> None:
    assert entropy_from_logits([0.0, 0.0, 0.0, 0.0]) == pytest.approx(math.log(4))
    value = entropy_from_logits([1000.0, 999.0, 998.0])
    assert math.isfinite(value)
    assert value > 0
    assert value != round(value, 6)


def test_entropy_trace_requires_one_raw_prewarper_step_per_generated_token() -> None:
    trace = entropy_trace_from_raw_logits([[0.0, 0.0], [1.0, -1.0]], expected_tokens=2)
    assert len(trace) == 2
    with pytest.raises(ValueError, match="aligned"):
        entropy_trace_from_raw_logits([[0.0, 0.0]], expected_tokens=2)
    with pytest.raises(ValueError, match="vocabulary"):
        entropy_trace_from_raw_logits([[]], expected_tokens=1)


def test_natural_record_accepts_gpu_precomputed_raw_logit_entropy_without_vocab_storage() -> None:
    base = capture()
    precomputed = tuple(
        entropy_trace_from_raw_logits(
            base.raw_prewarper_logits,
            expected_tokens=len(base.generated_token_ids),
        )
    )
    summarized = NaturalGenerationCapture(
        rendered_prompt=base.rendered_prompt,
        prompt_token_ids=base.prompt_token_ids,
        generated_token_ids=base.generated_token_ids,
        decoded_output=base.decoded_output,
        raw_prewarper_logits=(),
        stop_reason=base.stop_reason,
        precomputed_entropy_nats=precomputed,
    )
    result = build_natural_terminal_result(
        identity=IDENTITY,
        run_id=0,
        generation_seed=123,
        terminal_attempt_number=1,
        capture=summarized,
        token_contract=TOKEN_CONTRACT,
        decode_reasoning=lambda token_ids: "one two" if token_ids else "",
    )
    assert tuple(result["per_token_entropy_nats"]) == precomputed


def test_reasoning_entropy_summary_uses_exact_final_ceil_ten_percent() -> None:
    entropies = [float(index) for index in range(20)]
    summary = summarize_reasoning_entropy(entropies, reasoning_indices=tuple(range(2, 13)))
    assert summary == {
        "reasoning_token_count": 11,
        "mean_reasoning_entropy_nats": sum(range(2, 13)) / 11,
        "tail_reasoning_entropy_nats": (11.0 + 12.0) / 2,
        "tail_token_count": 2,
    }
    assert summarize_reasoning_entropy(entropies, reasoning_indices=()) == {
        "reasoning_token_count": 0,
        "mean_reasoning_entropy_nats": None,
        "tail_reasoning_entropy_nats": None,
        "tail_token_count": 0,
    }


def test_ten_run_seeds_are_deterministic_unique_and_match_contract_function() -> None:
    seeds = plan_ten_generation_seeds(
        canonical_model_identity="hf:model@" + "f" * 40,
        question_id=IDENTITY["question_id"],
        base_seed=42,
    )
    assert list(seeds) == list(range(10))
    assert len(set(seeds.values())) == 10
    assert seeds[0] == derive_generation_seed(
        base_seed=42,
        canonical_model_identity="hf:model@" + "f" * 40,
        question_id=IDENTITY["question_id"],
        run_id=0,
    )


def test_complete_natural_result_is_schema_valid_and_keeps_full_precision_trace() -> None:
    result = build_natural_terminal_result(
        identity=IDENTITY,
        run_id=0,
        generation_seed=123,
        terminal_attempt_number=1,
        capture=capture(),
        token_contract=TOKEN_CONTRACT,
        decode_reasoning=lambda token_ids: "one two" if token_ids else "",
    )
    validate_instance("natural_terminal_result", result)
    assert result["natural_answer"] == "C"
    assert result["natural_correct"] is True
    assert result["raw_parsed_confidence"] == 80
    assert result["reasoning_token_count"] == 2
    assert result["checkpoint_eligible"] is True
    assert result["checkpoint_ids"] == [f"cp-{index:02d}" for index in range(11)]
    assert len(result["generated_token_ids"]) == len(result["per_token_entropy_nats"])
    assert result["per_token_entropy_nats"][0] != round(
        result["per_token_entropy_nats"][0], 6
    )


def test_natural_infrastructure_failure_builder_is_schema_valid_and_ineligible() -> None:
    from part1_generation import build_natural_infrastructure_failure_result

    result = build_natural_infrastructure_failure_result(
        identity=IDENTITY,
        run_id=0,
        generation_seed=123,
        terminal_attempt_number=3,
        prompt_hash="f" * 64,
        failure_category="transient_worker_failure",
        infrastructure_failure_reference="audit-event-id",
        error_details={
            "category": "transient_worker_failure",
            "exception_type": "RuntimeError",
            "message": "worker stopped",
        },
    )

    validate_instance("natural_terminal_result", result)
    assert result["natural_execution_outcome"] == "terminal_infrastructure_failure"
    assert result["checkpoint_eligible"] is False
    assert result["checkpoint_ids"] is None
    assert result["generated_token_ids"] is None
    assert result["terminal_error_details"]["category"] == "transient_worker_failure"


@pytest.mark.parametrize(
    ("generated", "decoded", "reasoning_status", "answer_status"),
    [
        (
            [10, 100, 101],
            "<think>unfinished Answer: A\nConfidence: 90",
            "missing_close",
            "missing",
        ),
        ([10, 11, 20], "<think></think>\nAnswer: C\nConfidence: 50", "no_reasoning", "parsed"),
    ],
)
def test_abnormal_but_executed_outputs_remain_checkpoint_eligible(
    generated: list[int], decoded: str, reasoning_status: str, answer_status: str
) -> None:
    result = build_natural_terminal_result(
        identity=IDENTITY,
        run_id=1,
        generation_seed=124,
        terminal_attempt_number=1,
        capture=capture(generated=generated, decoded=decoded, stop_reason="max_new_tokens"),
        token_contract=TOKEN_CONTRACT,
        decode_reasoning=lambda token_ids: "reasoning" if token_ids else "",
    )
    validate_instance("natural_terminal_result", result)
    assert result["reasoning_status"] == reasoning_status
    assert result["answer_parse_status"] == answer_status
    assert result["checkpoint_eligible"] is True
    assert result["stop_reason"] == "max_new_tokens"
    if reasoning_status == "missing_close":
        assert result["natural_answer"] is None
        assert result["normalized_confidence"] is None
        assert result["diagnostic_answer_like_text"] == "Answer: A"
    else:
        assert result["reasoning_token_count"] == 0
        assert result["mean_reasoning_entropy_nats"] is None
        assert result["tail_reasoning_entropy_nats"] is None


def test_reproducibility_comparison_separates_token_parse_and_entropy_evidence() -> None:
    first = capture()
    equal = compare_reproducibility(first, capture(), entropy_abs_tolerance=0.0)
    assert equal == {
        "exact_generated_token_equality": True,
        "exact_parsed_output_equality": True,
        "entropy_array_equal_within_tolerance": True,
        "entropy_abs_tolerance": 0.0,
    }
    changed = capture(decoded="<think>one two</think>\nAnswer: B\nConfidence: 80")
    comparison = compare_reproducibility(first, changed, entropy_abs_tolerance=1e-6)
    assert comparison["exact_generated_token_equality"] is True
    assert comparison["exact_parsed_output_equality"] is False
