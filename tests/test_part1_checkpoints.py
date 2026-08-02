"""Pure checkpoint placement/metric tests; no real model is loaded."""

from __future__ import annotations

import math

import pytest

from part1_checkpoints import (
    CheckpointGenerationCapture,
    build_alias_checkpoint_terminal_result,
    build_checkpoint_probe_plans,
    build_checkpoint_terminal_result,
    checkpoint_placements,
    choice_answer_metrics,
)
from part1_contract import validate_instance
from part1_generation import NaturalGenerationCapture, build_natural_terminal_result


TOKEN_CONTRACT = {
    "reasoning_open_token_ids": [10],
    "reasoning_close_token_ids": [11],
    "ad_token_ids": [20, 21, 22, 23],
    "ad_token_convention": "inducer_boundary_space_uppercase_single_token",
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


def natural(*, reasoning_tokens: int = 2) -> dict:
    generated = [10, *range(100, 100 + reasoning_tokens), 11, 22]
    capture = NaturalGenerationCapture(
        rendered_prompt="prompt",
        prompt_token_ids=(1, 2),
        generated_token_ids=tuple(generated),
        decoded_output=f"<think>{'x ' * reasoning_tokens}</think>\nAnswer: C\nConfidence: 80",
        raw_prewarper_logits=tuple((0.0, 1.0, 2.0) for _ in generated),
        stop_reason="eos",
    )
    return build_natural_terminal_result(
        identity=IDENTITY,
        run_id=0,
        generation_seed=123,
        terminal_attempt_number=1,
        capture=capture,
        token_contract=TOKEN_CONTRACT,
        decode_reasoning=lambda ids: "x " * len(ids),
    )


def test_checkpoint_placement_uses_python_ties_even_and_preserves_aliases() -> None:
    placements = checkpoint_placements(2)
    assert [placement.k_keep for placement in placements] == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    assert [placement.actual_fraction for placement in placements] == [
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
    ]
    assert placements[0].alias_metadata == {
        "owner_checkpoint_id": "cp-00",
        "members": ["cp-00", "cp-01", "cp-02"],
    }
    assert placements[0].is_alias is False
    assert placements[1].is_alias is True

    zero = checkpoint_placements(0)
    assert {placement.k_keep for placement in zero} == {0}
    assert {placement.actual_fraction for placement in zero} == {None}
    assert zero[0].alias_metadata["members"] == [f"cp-{i:02d}" for i in range(11)]


def test_probe_plans_use_token_id_prefixes_and_share_physical_alias_identity() -> None:
    parent = natural(reasoning_tokens=2)
    plans = build_checkpoint_probe_plans(
        parent,
        inducer_token_ids=[11, 30],
        inducer_version="inducer-v1",
    )
    assert len(plans) == 11
    assert plans[0].prefix_token_ids == (1, 2, 10)
    assert plans[0].model_input_token_ids == (1, 2, 10, 11, 30)
    assert plans[3].prefix_token_ids == (1, 2, 10, 100)
    assert plans[0].prefix_hash == plans[1].prefix_hash == plans[2].prefix_hash
    assert plans[0].shared_probe_id == plans[1].shared_probe_id == plans[2].shared_probe_id
    assert plans[3].prefix_hash != plans[0].prefix_hash


def test_choice_answer_metrics_store_only_four_logits_and_summary_values() -> None:
    full_logits = [0.0] * 30
    full_logits[20:24] = [1.0, 2.0, 3.0, 4.0]
    metrics = choice_answer_metrics(full_logits, ad_token_ids=[20, 21, 22, 23])
    assert metrics["ad_logits_float32"] == [1.0, 2.0, 3.0, 4.0]
    assert len(metrics["ad_probabilities_float32"]) == 4
    assert sum(metrics["ad_probabilities_float32"]) == pytest.approx(1.0)
    assert metrics["maximum_ad_probability"] == max(metrics["ad_probabilities_float32"])
    assert metrics["answer_entropy_nats"] > 0
    assert metrics["full_vocabulary_answer_step_entropy_nats"] > 0
    assert "full_vocabulary_logits" not in metrics
    assert metrics["answer_entropy_nats"] != round(metrics["answer_entropy_nats"], 6)


def test_checkpoint_terminal_result_is_schema_valid_and_aliases_reuse_probe() -> None:
    parent = natural(reasoning_tokens=1)
    plans = build_checkpoint_probe_plans(
        parent,
        inducer_token_ids=[11, 30],
        inducer_version="inducer-v1",
    )
    full_logits = [0.0] * 30
    full_logits[20:24] = [0.1, 0.2, 0.7, 0.0]
    capture = CheckpointGenerationCapture(
        forced_generated_token_ids=(22, 40),
        decoded_forced_output=" C\nConfidence: 70",
        raw_prewarper_logits=(tuple(full_logits), tuple([0.0] * 30)),
    )
    owner = build_checkpoint_terminal_result(
        parent=parent,
        plan=plans[0],
        capture=capture,
        token_contract=TOKEN_CONTRACT,
        gold_letter="C",
        terminal_attempt_number=1,
    )
    alias = build_checkpoint_terminal_result(
        parent=parent,
        plan=plans[1],
        capture=capture,
        token_contract=TOKEN_CONTRACT,
        gold_letter="C",
        terminal_attempt_number=1,
    )
    validate_instance("checkpoint_terminal_result", owner)
    validate_instance("checkpoint_terminal_result", alias)
    assert owner["checkpoint_model_output_status"] == "valid"
    assert owner["forced_answer"] == "C"
    assert owner["checkpoint_local_correct"] is True
    assert owner["answer_token_index"] == 0
    assert owner["answer_token_id"] == 22
    assert owner["agrees_with_natural_answer"] is True
    assert owner["is_alias"] is False and alias["is_alias"] is True
    for field in ("prefix_hash", "shared_probe_id", "ad_logits_float32", "answer_entropy_nats"):
        assert owner[field] == alias[field]

    recovered_alias = build_alias_checkpoint_terminal_result(
        parent=parent,
        owner_record=owner,
        alias_plan=plans[1],
        terminal_attempt_number=1,
    )
    validate_instance("checkpoint_terminal_result", recovered_alias)
    assert recovered_alias == alias


def test_checkpoint_infrastructure_failure_builder_is_schema_valid() -> None:
    from part1_checkpoints import build_checkpoint_infrastructure_failure_result

    parent = natural(reasoning_tokens=1)
    plan = build_checkpoint_probe_plans(
        parent,
        inducer_token_ids=[11, 30],
        inducer_version="inducer-v1",
    )[0]
    result = build_checkpoint_infrastructure_failure_result(
        parent=parent,
        plan=plan,
        terminal_attempt_number=1,
        failure_category="unsupported_model_or_tokenizer_behaviour",
        infrastructure_failure_reference="audit-event-id",
        error_details={
            "category": "unsupported_model_or_tokenizer_behaviour",
            "exception_type": "ValueError",
            "message": "missing checkpoint logits",
        },
    )

    validate_instance("checkpoint_terminal_result", result)
    assert result["checkpoint_execution_outcome"] == "terminal_infrastructure_failure"
    assert result["checkpoint_model_output_status"] == "invalid"
    assert result["forced_generated_token_ids"] is None
    assert result["answer_token_status"] == "unsupported"
    assert result["terminal_error_details"]["category"] == (
        "unsupported_model_or_tokenizer_behaviour"
    )


def test_missing_choice_token_is_preserved_as_invalid_model_output_not_retry() -> None:
    parent = natural(reasoning_tokens=1)
    plan = build_checkpoint_probe_plans(
        parent,
        inducer_token_ids=[11, 30],
        inducer_version="inducer-v1",
    )[0]
    capture = CheckpointGenerationCapture(
        forced_generated_token_ids=(99,),
        decoded_forced_output=" C\nConfidence: 70",
        raw_prewarper_logits=(tuple([0.0] * 100),),
    )
    result = build_checkpoint_terminal_result(
        parent=parent,
        plan=plan,
        capture=capture,
        token_contract=TOKEN_CONTRACT,
        gold_letter="C",
        terminal_attempt_number=1,
    )
    validate_instance("checkpoint_terminal_result", result)
    assert result["checkpoint_execution_outcome"] == "complete"
    assert result["checkpoint_model_output_status"] == "invalid"
    assert result["answer_token_status"] == "missing"
    assert result["entropy_status"] == "unavailable"
    assert result["answer_token_id"] is None
    assert result["ad_logits_float32"] is None


def test_checkpoint_record_accepts_gpu_answer_step_logits_without_vocab_trace_storage() -> None:
    parent = natural(reasoning_tokens=1)
    plan = build_checkpoint_probe_plans(
        parent,
        inducer_token_ids=[11, 30],
        inducer_version="inducer-v1",
    )[0]
    full_logits = [0.0] * 30
    full_logits[20:24] = [0.1, 0.2, 0.7, 0.0]
    capture = CheckpointGenerationCapture(
        forced_generated_token_ids=(22, 40),
        decoded_forced_output=" C\nConfidence: 70",
        raw_prewarper_logits=(),
        answer_step_raw_logits=tuple(full_logits),
    )
    result = build_checkpoint_terminal_result(
        parent=parent,
        plan=plan,
        capture=capture,
        token_contract=TOKEN_CONTRACT,
        gold_letter="C",
        terminal_attempt_number=1,
    )
    assert result["entropy_status"] == "computed"
    assert result["ad_logits_float32"] == [0.1, 0.2, 0.7, 0.0]


def test_choice_metric_rejects_duplicate_or_out_of_vocabulary_token_ids() -> None:
    with pytest.raises(ValueError, match="distinct"):
        choice_answer_metrics([0.0] * 10, ad_token_ids=[1, 2, 2, 3])
    with pytest.raises(ValueError, match="vocabulary"):
        choice_answer_metrics([0.0] * 10, ad_token_ids=[1, 2, 3, 10])
    with pytest.raises(ValueError, match="finite"):
        choice_answer_metrics([0.0, math.inf, 0.0, 0.0], ad_token_ids=[0, 1, 2, 3])
