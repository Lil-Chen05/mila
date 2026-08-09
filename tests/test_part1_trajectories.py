"""Pure trajectory feature extraction tests; safe on a login node."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import pytest

from part1_contract import FIXED_PRIMARY_AUROC_FEATURE_REGISTRY, validate_instance


LETTERS = "ABCD"


def _hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _natural(
    *,
    sample_index: int = 0,
    run_id: int = 0,
    outcome: str = "complete",
    answer: str | None = "C",
    confidence: float | None = 0.8,
    reasoning_count: int = 2,
) -> dict[str, Any]:
    complete = outcome == "complete"
    parsed = complete and isinstance(answer, str) and answer in LETTERS
    confidence_parsed = complete and confidence is not None
    raw_id = _hex(f"natural-{sample_index}-{run_id}")
    checkpoint_ids = [_hex(f"{raw_id}-checkpoint-{index}") for index in range(11)]
    row = {
        "schema_name": "part1_natural_terminal_result",
        "schema_version": "1.0.0",
        "raw_record_id": raw_id,
        "study_id": "1" * 64,
        "model_run_id": "2" * 64,
        "model_run_manifest_hash": "3" * 64,
        "question_manifest_hash": "4" * 64,
        "question_id": _hex(f"question-{sample_index}"),
        "sample_index": sample_index,
        "subject": "high_school_mathematics",
        "run_id": run_id,
        "generation_seed": 42 + run_id,
        "seed_algorithm_version": "part1-seed-v1",
        "terminal_attempt_number": 1,
        "terminal_attempt_id": _hex(f"natural-attempt-{sample_index}-{run_id}"),
        "infrastructure_failure_reference": None if complete else "audit:failure",
        "prompt_hash": _hex(f"prompt-{sample_index}"),
        "rendered_prompt": "prompt" if complete else None,
        "prompt_token_ids": [1, 2] if complete else None,
        "generated_token_ids": [10, 11] if complete else None,
        "decoded_output": "<think>x</think> Answer: C" if complete else None,
        "reasoning_text": "x" if complete else None,
        "reasoning_boundaries": {"start": 1, "end": 2} if complete else None,
        "close_tag_information": {"found": True} if complete else None,
        "stop_reason": "eos" if complete else "error",
        "generated_token_count": 2 if complete else None,
        "reasoning_token_count": reasoning_count if complete else None,
        "per_token_entropy_nats": [1.2, 1.4] if complete else None,
        "mean_reasoning_entropy_nats": 1.2 if complete and reasoning_count else None,
        "tail_reasoning_entropy_nats": 1.4 if complete and reasoning_count else None,
        "terminal_answer_block_text": "Answer: C" if parsed else None,
        "terminal_answer_block_span": {"start": 20, "end": 29} if parsed else None,
        "natural_answer": answer if parsed else None,
        "raw_confidence_text": str(round(confidence * 100)) if confidence_parsed else None,
        "raw_parsed_confidence": round(confidence * 100) if confidence_parsed else None,
        "normalized_confidence": confidence if confidence_parsed else None,
        "natural_correct": answer == "C" if parsed else None,
        "diagnostic_answer_like_text": None,
        "checkpoint_eligible": complete,
        "checkpoint_ids": checkpoint_ids if complete else None,
        "natural_execution_outcome": outcome,
        "reasoning_status": (
            "no_reasoning" if complete and reasoning_count == 0 else "closed" if complete else "malformed"
        ),
        "answer_parse_status": "parsed" if parsed else "missing",
        "confidence_parse_status": "parsed" if confidence_parsed else "missing",
        "component_versions": {"adapter": "smollm3-v1", "parser": "v1"},
        "terminal_error_details": None if complete else {"category": "transient_worker_failure"},
    }
    return row


def _checkpoint(
    natural: dict[str, Any],
    index: int,
    *,
    answer: str | None = "C",
    is_alias: bool = False,
    outcome: str = "complete",
    output_status: str | None = None,
    entropy: float = 0.4,
    maximum: float = 0.7,
) -> dict[str, Any]:
    complete = outcome == "complete"
    valid = (
        complete
        and isinstance(answer, str)
        and answer in LETTERS
        and output_status != "invalid"
    )
    status = "valid" if valid else "invalid"
    probabilities = [0.1, 0.1, 0.1, 0.7] if valid else None
    checkpoint_id = natural["checkpoint_ids"][index]
    return {
        "schema_name": "part1_checkpoint_terminal_result",
        "schema_version": "1.0.0",
        "checkpoint_record_id": _hex(f"record-{natural['raw_record_id']}-{index}"),
        "parent_raw_record_id": natural["raw_record_id"],
        "study_id": natural["study_id"],
        "model_run_id": natural["model_run_id"],
        "model_run_manifest_hash": natural["model_run_manifest_hash"],
        "question_manifest_hash": natural["question_manifest_hash"],
        "question_id": natural["question_id"],
        "sample_index": natural["sample_index"],
        "subject": natural["subject"],
        "run_id": natural["run_id"],
        "checkpoint_id": checkpoint_id,
        "natural_seed": natural["generation_seed"],
        "terminal_attempt_number": 1,
        "terminal_attempt_id": _hex(f"attempt-{natural['raw_record_id']}-{index}"),
        "infrastructure_failure_reference": None if complete else "audit:failure",
        "requested_checkpoint_index": index,
        "requested_fraction": index / 10,
        "k_keep": index,
        "actual_fraction": index / 10,
        "shared_probe_id": _hex(f"probe-{natural['raw_record_id']}-{index}"),
        "is_alias": is_alias,
        "alias_metadata": {"owner_checkpoint_id": checkpoint_id, "members": [checkpoint_id]},
        "prefix_hash": _hex(f"prefix-{natural['raw_record_id']}-{index}"),
        "inducer_version": "smollm3-inducer-v1",
        "inducer_text": "</think>\nAnswer:",
        "forced_generated_token_ids": (
            [10 + LETTERS.index(answer)] if valid else [10] if complete else None
        ),
        "decoded_forced_output": f" {answer}" if complete else None,
        "terminal_answer_block_text": f"Answer: {answer}" if valid else None,
        "forced_answer": answer if valid else None,
        "raw_confidence_text": "70" if valid else None,
        "raw_parsed_confidence": 70 if valid else None,
        "normalized_confidence": 0.7 if valid else None,
        "checkpoint_local_correct": answer == "C" if valid else None,
        "answer_token_index": 0 if valid else None,
        "answer_token_id": 10 + LETTERS.index(answer) if valid else None,
        "token_convention": "bare" if valid else None,
        "ad_token_ids": [10, 11, 12, 13] if valid else None,
        "ad_logits_float32": [0.0, 0.0, 0.0, 1.0] if valid else None,
        "ad_probabilities_float32": probabilities,
        "answer_entropy_nats": entropy if valid else None,
        "full_vocabulary_answer_step_entropy_nats": 2.0 if valid else None,
        "maximum_ad_probability": maximum if valid else None,
        "agrees_with_natural_answer": answer == natural["natural_answer"] if valid and natural["natural_answer"] else None,
        "checkpoint_execution_outcome": outcome,
        "checkpoint_model_output_status": status,
        "answer_parse_status": "parsed" if valid else "missing",
        "confidence_parse_status": "parsed" if valid else "missing",
        "answer_token_status": "located" if valid else "unsupported",
        "entropy_status": "computed" if valid else "unavailable",
        "component_versions": {"adapter": "smollm3-v1", "parser": "v1"},
        "terminal_error_details": None if complete else {"category": "transient_worker_failure"},
    }


def _checkpoints(
    natural: dict[str, Any],
    *,
    answers: dict[int, str | None] | None = None,
    aliases: set[int] | None = None,
    invalid: set[int] | None = None,
) -> list[dict[str, Any]]:
    answers = answers or {}
    aliases = aliases or set()
    invalid = invalid or set()
    return [
        _checkpoint(
            natural,
            index,
            answer=answers.get(index, "C"),
            is_alias=index in aliases,
            output_status="invalid" if index in invalid else None,
        )
        for index in range(11)
    ]


def _extract(natural: dict[str, Any], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    from part1_trajectories import extract_trajectory_features

    return extract_trajectory_features(natural, checkpoints)


def test_base_fixtures_are_schema_shaped() -> None:
    natural = _natural()
    validate_instance("natural_terminal_result", natural)
    for checkpoint in _checkpoints(natural):
        validate_instance("checkpoint_terminal_result", checkpoint)


def test_exact_primary_registry_orientation_order_and_no_input_mutation() -> None:
    from part1_trajectories import PRIMARY_FEATURE_REGISTRY, build_trajectory_rows

    first = _natural(sample_index=1, run_id=1)
    second = _natural(sample_index=0, run_id=2)
    first_checkpoints = _checkpoints(first)
    second_checkpoints = _checkpoints(second)
    second_checkpoints[0]["answer_entropy_nats"] = 0.25
    second_checkpoints[5]["answer_entropy_nats"] = 0.5
    second_checkpoints[10]["answer_entropy_nats"] = 0.75
    second_checkpoints[0]["maximum_ad_probability"] = 0.9
    second_checkpoints[0]["ad_probabilities_float32"] = [0.9, 0.05, 0.03, 0.02]
    before = copy.deepcopy(([first, second], first_checkpoints + second_checkpoints))

    rows = build_trajectory_rows(
        [first, second], first_checkpoints + second_checkpoints
    )

    assert tuple(PRIMARY_FEATURE_REGISTRY) == tuple(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)
    assert [(row["sample_index"], row["run_id"]) for row in rows] == [(0, 2), (1, 1)]
    row = rows[0]
    assert [name for name in FIXED_PRIMARY_AUROC_FEATURE_REGISTRY if name in row] == list(
        FIXED_PRIMARY_AUROC_FEATURE_REGISTRY
    )
    assert {name: row[name] for name in FIXED_PRIMARY_AUROC_FEATURE_REGISTRY} == {
        "negative_mean_reasoning_entropy": -1.2,
        "negative_tail_reasoning_entropy": -1.4,
        "negative_answer_entropy_fraction_0.0": -0.25,
        "negative_answer_entropy_fraction_0.5": -0.5,
        "negative_answer_entropy_fraction_1.0": -0.75,
        "natural_verbalized_confidence": 0.8,
        "maximum_ad_probability_fraction_0.0": 0.9,
        "maximum_ad_probability_fraction_0.5": 0.7,
        "maximum_ad_probability_fraction_1.0": 0.7,
        "negative_answer_switch_count": 0,
        "negative_stabilization_fraction": 0.0,
    }
    assert row["answer_switch_count"] == 0
    assert row["stabilization_fraction"] == 0.0
    assert list(row["feature_missing_reasons"]) == list(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)
    assert set(row["feature_missing_reasons"].values()) == {None}
    assert ([first, second], first_checkpoints + second_checkpoints) == before


@pytest.mark.parametrize(
    ("aliases", "invalid", "absent", "answers", "switches", "transitions"),
    [
        (set(), set(), set(), {0: "A", 1: "A", **{i: "B" for i in range(2, 11)}}, 1, 10),
        ({1}, set(), set(), {0: "A", 1: "B", **{i: "A" for i in range(2, 11)}}, 0, 9),
        (set(), {1}, set(), {0: "A", 1: None, **{i: "B" for i in range(2, 11)}}, 0, 8),
        (set(), set(), {1}, {0: "A", **{i: "B" for i in range(2, 11)}}, 0, 8),
    ],
)
def test_physical_transitions_skip_aliases_and_never_bridge_gaps(
    aliases: set[int],
    invalid: set[int],
    absent: set[int],
    answers: dict[int, str | None],
    switches: int,
    transitions: int,
) -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural, answers=answers, aliases=aliases, invalid=invalid)
    checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["requested_checkpoint_index"] not in absent
    ]

    row = _extract(natural, checkpoints)

    assert row["answer_switch_count"] == switches
    assert row["valid_transition_count"] == transitions
    assert row["transition_evaluability_status"] == "evaluated"
    assert row["transition_evaluability_reason"] is None


def test_alias_supplies_logical_metrics_and_appearance_without_a_physical_switch() -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural, aliases={5})
    checkpoints[5]["answer_entropy_nats"] = 0.25
    checkpoints[5]["maximum_ad_probability"] = 0.9
    checkpoints[5]["ad_probabilities_float32"] = [0.9, 0.05, 0.03, 0.02]

    row = _extract(natural, checkpoints)

    assert row["negative_answer_entropy_fraction_0.5"] == -0.25
    assert row["maximum_ad_probability_fraction_0.5"] == 0.9
    assert row["first_natural_answer_appearance_fraction"] == 0.0
    assert row["answer_switch_count"] == 0
    assert row["valid_transition_count"] == 9


def test_first_appearance_reports_natural_missing_and_never_appeared() -> None:
    natural_missing = _natural(answer=None, confidence=None)
    missing_row = _extract(natural_missing, _checkpoints(natural_missing))
    assert missing_row["first_natural_answer_appearance_fraction"] is None
    assert missing_row["first_natural_answer_appearance_status"] == "unavailable"
    assert missing_row["first_natural_answer_appearance_reason"] == "natural_answer_invalid_or_missing"

    natural = _natural(answer="A")
    never_row = _extract(natural, _checkpoints(natural))
    assert never_row["first_natural_answer_appearance_fraction"] is None
    assert never_row["first_natural_answer_appearance_status"] == "not_found"
    assert never_row["first_natural_answer_appearance_reason"] == "natural_answer_never_appeared"


def test_leave_recover_and_no_leave_semantics() -> None:
    natural = _natural()
    answers = {0: "C", 1: "A", 2: None, 3: "C"}
    recovered = _extract(natural, _checkpoints(natural, answers=answers, invalid={2}))
    assert recovered["left_correct_answer"] is True
    assert recovered["left_correct_answer_status"] == "evaluated"
    assert recovered["later_recovered_correct_answer"] is True
    assert recovered["later_recovered_correct_answer_status"] == "evaluated"

    not_recovered_answers = {0: "C", **{index: "A" for index in range(1, 11)}}
    not_recovered = _extract(natural, _checkpoints(natural, answers=not_recovered_answers))
    assert not_recovered["left_correct_answer"] is True
    assert not_recovered["later_recovered_correct_answer"] is False
    assert not_recovered["later_recovered_correct_answer_reason"] == "no_later_recovery"

    no_leave = _extract(natural, _checkpoints(natural))
    assert no_leave["left_correct_answer"] is False
    assert no_leave["later_recovered_correct_answer"] is False
    assert no_leave["later_recovered_correct_answer_status"] == "not_applicable"
    assert no_leave["later_recovered_correct_answer_reason"] == "not_applicable_no_leave"


@pytest.mark.parametrize(
    ("index", "answer", "local_correct"),
    [(0, "C", None), (1, None, True)],
)
def test_checkpoint_local_correctness_must_agree_with_answer_validity(
    index: int, answer: str | None, local_correct: bool | None
) -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural, answers={index: answer}, invalid={index} if answer is None else set())
    checkpoints[index]["checkpoint_local_correct"] = local_correct
    with pytest.raises(ValueError, match="checkpoint_local_correct"):
        _extract(natural, checkpoints)


def test_endpoint_agreement_is_recomputed_and_has_explicit_missingness() -> None:
    natural = _natural(answer="C")
    agreeing = _checkpoints(natural)
    agreeing[10]["agrees_with_natural_answer"] = False
    assert _extract(natural, agreeing)["forced_endpoint_agrees_with_natural"] is True

    disagreeing = _checkpoints(natural, answers={10: "A"})
    assert _extract(natural, disagreeing)["forced_endpoint_agrees_with_natural"] is False

    invalid = _checkpoints(natural, answers={10: None}, invalid={10})
    row = _extract(natural, invalid)
    assert row["forced_endpoint_agrees_with_natural"] is None
    assert row["forced_endpoint_agrees_with_natural_reason"] == "checkpoint_1.0_answer_invalid"


@pytest.mark.parametrize(
    ("answers", "invalid", "expected"),
    [
        ({}, set(), 0.0),
        ({0: "A", 1: "A", 2: "B", 3: "B"}, set(), 0.4),
        ({index: "A" for index in range(10)}, set(), 1.0),
        ({}, {10}, None),
    ],
)
def test_stabilization_at_zero_interior_one_and_invalid_endpoint(
    answers: dict[int, str], invalid: set[int], expected: float | None
) -> None:
    natural = _natural()
    row = _extract(natural, _checkpoints(natural, answers=answers, invalid=invalid))
    assert row["stabilization_fraction"] == expected
    assert row["negative_stabilization_fraction"] == (-expected if expected is not None else None)


def test_stabilization_suffix_starts_after_gaps_but_never_bridges_them() -> None:
    natural = _natural()
    after_early_gap = _checkpoints(natural, invalid={3})
    row = _extract(natural, after_early_gap)
    assert row["stabilization_fraction"] == 0.4

    gap_inside_earlier_suffix = _checkpoints(natural, invalid={8})
    gap_inside_earlier_suffix[8]["answer_parse_status"] = "malformed"
    row = _extract(natural, gap_inside_earlier_suffix)
    assert row["stabilization_fraction"] == 0.9

    missing_nine = [
        checkpoint
        for checkpoint in _checkpoints(natural)
        if checkpoint["requested_checkpoint_index"] != 9
    ]
    assert _extract(natural, missing_nine)["stabilization_fraction"] == 1.0


def test_zero_reasoning_alias_collapse_can_stabilize_at_owner_zero() -> None:
    natural = _natural(reasoning_count=0)
    checkpoints = _checkpoints(natural, aliases=set(range(1, 11)))
    row = _extract(natural, checkpoints)
    assert row["stabilization_fraction"] == 0.0
    assert row["negative_mean_reasoning_entropy"] is None
    assert row["feature_missing_reasons"]["negative_mean_reasoning_entropy"] == "no_reasoning_tokens"


def test_short_and_abnormal_complete_reasoning_entropy_remains_data() -> None:
    short = _natural(reasoning_count=1)
    short["generated_token_ids"] = [10]
    short["per_token_entropy_nats"] = [1.2]
    short["generated_token_count"] = 1
    assert _extract(short, _checkpoints(short))["negative_tail_reasoning_entropy"] == -1.4

    abnormal = _natural(answer=None, confidence=None, reasoning_count=2)
    abnormal["reasoning_status"] = "missing_close"
    abnormal["diagnostic_answer_like_text"] = "Answer: C"
    row = _extract(abnormal, _checkpoints(abnormal))
    assert row["negative_mean_reasoning_entropy"] == -1.2
    assert row["natural_verbalized_confidence"] is None
    assert row["answer_switch_count"] == 0


def test_infrastructure_failures_are_explicit_and_checkpoint_failures_are_missing_data() -> None:
    failed = _natural(outcome="terminal_infrastructure_failure", answer=None, confidence=None)
    row = _extract(failed, [])
    assert all(row[name] is None for name in FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)
    assert set(row["feature_missing_reasons"].values()) == {"natural_terminal_infrastructure_failure"}
    assert row["answer_switch_count"] is None
    assert row["transition_evaluability_status"] == "unavailable"

    natural = _natural()
    checkpoints = _checkpoints(natural)
    checkpoints[5] = _checkpoint(natural, 5, answer=None, outcome="terminal_infrastructure_failure")
    row = _extract(natural, checkpoints)
    assert row["negative_answer_entropy_fraction_0.5"] is None
    assert row["feature_missing_reasons"]["negative_answer_entropy_fraction_0.5"] == "checkpoint_entropy_not_computed"


def test_absent_logical_fraction_has_feature_missingness_without_inference() -> None:
    natural = _natural()
    checkpoints = [
        checkpoint
        for checkpoint in _checkpoints(natural)
        if checkpoint["requested_checkpoint_index"] != 5
    ]
    row = _extract(natural, checkpoints)
    assert row["negative_answer_entropy_fraction_0.5"] is None
    assert row["maximum_ad_probability_fraction_0.5"] is None
    assert row["feature_missing_reasons"]["negative_answer_entropy_fraction_0.5"] == "checkpoint_fraction_missing"


def test_collection_rejects_duplicate_natural_keys_and_record_ids() -> None:
    from part1_trajectories import build_trajectory_rows

    natural = _natural()
    duplicate_key = copy.deepcopy(natural)
    duplicate_key["raw_record_id"] = _hex("other-natural-record")
    with pytest.raises(ValueError, match="duplicate natural logical key"):
        build_trajectory_rows([natural, duplicate_key], [])

    other = _natural(sample_index=1)
    other["raw_record_id"] = natural["raw_record_id"]
    with pytest.raises(ValueError, match="duplicate natural raw_record_id"):
        build_trajectory_rows([natural, other], [])

    mixed = _natural(sample_index=1)
    mixed["study_id"] = "9" * 64
    with pytest.raises(ValueError, match="mixed natural provenance"):
        build_trajectory_rows([natural, mixed], [])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cp, natural: cp.__setitem__("parent_raw_record_id", _hex("absent")), "without a parent"),
        (lambda cp, natural: cp.__setitem__("study_id", "9" * 64), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("model_run_id", "9" * 64), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("model_run_manifest_hash", "9" * 64), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("question_manifest_hash", "9" * 64), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("question_id", "9" * 64), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("sample_index", 9), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("subject", "high_school_physics"), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("run_id", 9), "mixed checkpoint provenance"),
        (lambda cp, natural: cp.__setitem__("requested_fraction", 0.6), "noncanonical requested checkpoint"),
        (lambda cp, natural: cp.__setitem__("checkpoint_id", _hex("wrong-checkpoint")), "checkpoint_id does not match parent"),
    ],
)
def test_checkpoint_parent_and_full_immutable_hierarchy_are_enforced(
    mutate: Callable[[dict[str, Any], dict[str, Any]], None], message: str
) -> None:
    from part1_trajectories import build_trajectory_rows

    natural = _natural()
    checkpoint = _checkpoint(natural, 0)
    mutate(checkpoint, natural)
    with pytest.raises(ValueError, match=message):
        build_trajectory_rows([natural], [checkpoint])


def test_checkpoint_duplicates_and_input_order_are_rejected() -> None:
    from part1_trajectories import build_trajectory_rows

    natural = _natural()
    checkpoints = _checkpoints(natural)
    duplicate_index = copy.deepcopy(checkpoints[0])
    with pytest.raises(ValueError, match="duplicate checkpoint requested index"):
        build_trajectory_rows([natural], [checkpoints[0], duplicate_index])

    duplicate_id = copy.deepcopy(checkpoints[1])
    duplicate_id["checkpoint_id"] = checkpoints[0]["checkpoint_id"]
    with pytest.raises(ValueError, match="duplicate checkpoint_id"):
        build_trajectory_rows([natural], [checkpoints[0], duplicate_id])

    duplicate_record = copy.deepcopy(checkpoints[1])
    duplicate_record["checkpoint_record_id"] = checkpoints[0]["checkpoint_record_id"]
    with pytest.raises(ValueError, match="duplicate checkpoint_record_id"):
        build_trajectory_rows([natural], [checkpoints[0], duplicate_record])

    with pytest.raises(ValueError, match="out-of-order checkpoints"):
        build_trajectory_rows([natural], [checkpoints[1], checkpoints[0]])


def test_checkpoint_under_natural_infrastructure_failure_is_rejected() -> None:
    from part1_trajectories import build_trajectory_rows

    failed = _natural(outcome="terminal_infrastructure_failure", answer=None, confidence=None)
    complete = _natural(sample_index=1)
    checkpoint = _checkpoint(complete, 0)
    checkpoint["parent_raw_record_id"] = failed["raw_record_id"]
    for field in ("question_id", "sample_index", "run_id"):
        checkpoint[field] = failed[field]
    with pytest.raises(ValueError, match="natural terminal infrastructure failure"):
        build_trajectory_rows([failed], [checkpoint])


@pytest.mark.parametrize(
    ("target", "mutate", "message"),
    [
        ("natural", lambda row: row.__setitem__("mean_reasoning_entropy_nats", float("nan")), "mean_reasoning_entropy_nats"),
        ("natural", lambda row: row.__setitem__("tail_reasoning_entropy_nats", float("inf")), "tail_reasoning_entropy_nats"),
        ("natural", lambda row: row.__setitem__("mean_reasoning_entropy_nats", -0.1), "mean_reasoning_entropy_nats"),
        ("natural", lambda row: (row.__setitem__("confidence_parse_status", "missing"), row.__setitem__("normalized_confidence", 0.8)), "normalized_confidence"),
        ("natural", lambda row: row.__setitem__("normalized_confidence", float("inf")), "normalized_confidence"),
        ("checkpoint", lambda row: (row.__setitem__("entropy_status", "unavailable"), row.__setitem__("answer_entropy_nats", 0.4)), "answer_entropy_nats"),
        ("checkpoint", lambda row: row.__setitem__("answer_entropy_nats", float("nan")), "answer_entropy_nats"),
        ("checkpoint", lambda row: row.__setitem__("maximum_ad_probability", float("inf")), "maximum_ad_probability"),
        ("checkpoint", lambda row: row.__setitem__("maximum_ad_probability", 1.1), "maximum_ad_probability"),
        ("checkpoint", lambda row: row.__setitem__("ad_probabilities_float32", [0.1, 0.2, float("nan"), 0.7]), "ad_probabilities_float32"),
        ("checkpoint", lambda row: row.__setitem__("maximum_ad_probability", 0.6), "maximum_ad_probability"),
    ],
)
def test_metric_status_value_contradictions_and_nonfinite_values_are_rejected(
    target: str, mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    mutate(natural if target == "natural" else checkpoints[0])
    with pytest.raises(ValueError, match=message):
        _extract(natural, checkpoints)


def test_computed_checkpoint_metrics_require_complete_execution() -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    checkpoint = checkpoints[0]
    checkpoint.update(
        {
            "checkpoint_execution_outcome": "terminal_infrastructure_failure",
            "checkpoint_model_output_status": "invalid",
            "answer_parse_status": "missing",
            "forced_answer": None,
            "checkpoint_local_correct": None,
            "entropy_status": "computed",
            "answer_entropy_nats": None,
            "maximum_ad_probability": None,
            "ad_probabilities_float32": None,
        }
    )
    with pytest.raises(ValueError, match="computed checkpoint metrics require complete execution"):
        _extract(natural, checkpoints)


def test_ad_probability_vector_must_remain_renormalized() -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    checkpoints[0]["ad_probabilities_float32"] = [0.2, 0.2, 0.2, 0.2]
    checkpoints[0]["maximum_ad_probability"] = 0.2
    with pytest.raises(ValueError, match="sum to one"):
        _extract(natural, checkpoints)


def _make_parsed_aggregate_invalid(checkpoint: dict[str, Any]) -> None:
    """Keep parsed answer/confidence while making answer-step metrics unavailable."""

    checkpoint.update(
        {
            "checkpoint_model_output_status": "invalid",
            "answer_token_status": "missing",
            "answer_token_index": None,
            "answer_token_id": None,
            "token_convention": None,
            "ad_token_ids": None,
            "ad_logits_float32": None,
            "ad_probabilities_float32": None,
            "answer_entropy_nats": None,
            "full_vocabulary_answer_step_entropy_nats": None,
            "maximum_ad_probability": None,
            "entropy_status": "unavailable",
        }
    )


def test_parsed_answer_survives_invalid_aggregate_output_status() -> None:
    natural = _natural(answer="C")
    checkpoints = _checkpoints(natural)
    _make_parsed_aggregate_invalid(checkpoints[0])
    validate_instance("checkpoint_terminal_result", checkpoints[0])

    row = _extract(natural, checkpoints)

    assert row["first_natural_answer_appearance_fraction"] == 0.0
    assert row["valid_transition_count"] == 10
    assert row["answer_switch_count"] == 0
    assert row["stabilization_fraction"] == 0.0
    slot = row["checkpoint_calibration"][0]
    assert slot["answer_valid"] is True
    assert slot["checkpoint_local_correct"] is True
    assert slot["confidence_available"] is True
    assert slot["normalized_confidence"] == 0.7
    assert slot["maximum_ad_probability_available"] is False
    assert slot["maximum_ad_probability"] is None
    assert slot["maximum_ad_probability_missing_reason"] == "checkpoint_ad_probability_unavailable"


def test_all_eleven_checkpoint_calibration_slots_are_logical_and_deterministic() -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural, aliases={5}, invalid={4})
    checkpoints[1].update(
        {
            "confidence_parse_status": "missing",
            "raw_confidence_text": None,
            "raw_parsed_confidence": None,
            "normalized_confidence": None,
        }
    )
    checkpoints[2].update(
        {
            "confidence_parse_status": "out_of_range",
            "raw_confidence_text": "250",
            "raw_parsed_confidence": 250,
            "normalized_confidence": None,
        }
    )
    _make_parsed_aggregate_invalid(checkpoints[3])
    checkpoints[4].update(
        {
            "confidence_parse_status": "parsed",
            "raw_confidence_text": "70",
            "raw_parsed_confidence": 70,
            "normalized_confidence": 0.7,
        }
    )
    checkpoints[6] = _checkpoint(
        natural, 6, answer=None, outcome="terminal_infrastructure_failure"
    )
    checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["requested_checkpoint_index"] != 7
    ]
    before = copy.deepcopy((natural, checkpoints))

    row = _extract(natural, checkpoints)
    slots = row["checkpoint_calibration"]

    assert len(slots) == 11
    assert [slot["requested_checkpoint_index"] for slot in slots] == list(range(11))
    assert [slot["requested_fraction"] for slot in slots] == [i / 10 for i in range(11)]
    assert all(type(slot["requested_fraction"]) is float for slot in slots)
    assert set(slots[0]) == {
        "requested_checkpoint_index",
        "requested_fraction",
        "checkpoint_id",
        "present",
        "eligibility_status",
        "is_alias",
        "checkpoint_execution_outcome",
        "checkpoint_model_output_status",
        "answer_parse_status",
        "forced_answer",
        "answer_valid",
        "checkpoint_local_correct",
        "confidence_parse_status",
        "normalized_confidence",
        "confidence_available",
        "confidence_missing_reason",
        "entropy_status",
        "maximum_ad_probability",
        "maximum_ad_probability_available",
        "maximum_ad_probability_missing_reason",
    }
    assert slots[0]["confidence_available"] is True
    assert slots[0]["confidence_missing_reason"] is None
    assert slots[0]["maximum_ad_probability_available"] is True
    assert slots[0]["maximum_ad_probability_missing_reason"] is None
    assert slots[1]["confidence_available"] is False
    assert slots[1]["confidence_missing_reason"] == "checkpoint_confidence_not_parsed"
    assert slots[2]["confidence_available"] is False
    assert slots[2]["confidence_missing_reason"] == "checkpoint_confidence_not_parsed"
    assert slots[3]["answer_valid"] is True
    assert slots[3]["confidence_available"] is True
    assert slots[3]["maximum_ad_probability_available"] is False
    assert slots[3]["maximum_ad_probability_missing_reason"] == "checkpoint_ad_probability_unavailable"
    assert slots[4]["answer_valid"] is False
    assert slots[4]["confidence_available"] is False
    assert slots[4]["confidence_missing_reason"] == "checkpoint_local_correctness_unavailable"
    assert slots[4]["maximum_ad_probability_available"] is False
    assert slots[4]["maximum_ad_probability_missing_reason"] == "checkpoint_local_correctness_unavailable"
    assert slots[5]["present"] is True
    assert slots[5]["is_alias"] is True
    assert slots[6]["eligibility_status"] == "eligible"
    assert slots[6]["confidence_missing_reason"] == "checkpoint_terminal_infrastructure_failure"
    assert slots[6]["maximum_ad_probability_missing_reason"] == "checkpoint_terminal_infrastructure_failure"
    assert slots[7] == {
        "requested_checkpoint_index": 7,
        "requested_fraction": 0.7,
        "checkpoint_id": natural["checkpoint_ids"][7],
        "present": False,
        "eligibility_status": "missing",
        "is_alias": None,
        "checkpoint_execution_outcome": None,
        "checkpoint_model_output_status": None,
        "answer_parse_status": None,
        "forced_answer": None,
        "answer_valid": False,
        "checkpoint_local_correct": None,
        "confidence_parse_status": None,
        "normalized_confidence": None,
        "confidence_available": False,
        "confidence_missing_reason": "checkpoint_missing",
        "entropy_status": None,
        "maximum_ad_probability": None,
        "maximum_ad_probability_available": False,
        "maximum_ad_probability_missing_reason": "checkpoint_missing",
    }
    assert (natural, checkpoints) == before


def test_natural_failure_has_eleven_explicit_ineligible_calibration_slots() -> None:
    natural = _natural(
        outcome="terminal_infrastructure_failure", answer=None, confidence=None
    )

    slots = _extract(natural, [])["checkpoint_calibration"]

    assert len(slots) == 11
    assert [slot["requested_checkpoint_index"] for slot in slots] == list(range(11))
    assert all(slot["requested_fraction"] == index / 10 for index, slot in enumerate(slots))
    assert all(type(slot["requested_fraction"]) is float for slot in slots)
    assert all(slot["checkpoint_id"] is None for slot in slots)
    assert all(slot["present"] is False for slot in slots)
    assert all(slot["eligibility_status"] == "ineligible_natural_failure" for slot in slots)
    assert all(slot["confidence_available"] is False for slot in slots)
    assert all(slot["confidence_missing_reason"] == "ineligible_natural_failure" for slot in slots)
    assert all(slot["maximum_ad_probability_available"] is False for slot in slots)
    assert all(
        slot["maximum_ad_probability_missing_reason"] == "ineligible_natural_failure"
        for slot in slots
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.__setitem__("checkpoint_eligible", False), "complete natural checkpoint eligibility"),
        (lambda row: row.__setitem__("checkpoint_ids", None), "complete natural checkpoint_ids"),
        (lambda row: row.__setitem__("checkpoint_ids", row["checkpoint_ids"][:10]), "complete natural checkpoint_ids"),
        (lambda row: row.__setitem__("checkpoint_ids", ["duplicate"] * 11), "complete natural checkpoint_ids"),
        (lambda row: row.__setitem__("checkpoint_ids", [*row["checkpoint_ids"][:10], 11]), "complete natural checkpoint_ids"),
        (lambda row: row.__setitem__("infrastructure_failure_reference", "audit:failure"), "complete natural failure state"),
        (lambda row: row.__setitem__("terminal_error_details", {"category": "failure"}), "complete natural failure state"),
        (lambda row: row.__setitem__("rendered_prompt", None), "complete natural generation field"),
        (lambda row: row.__setitem__("generated_token_ids", None), "complete natural generation field"),
    ],
)
def test_complete_natural_bundle_is_validated_even_without_child_rows(
    mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    natural = _natural()
    mutate(natural)
    with pytest.raises(ValueError, match=message):
        _extract(natural, [])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.__setitem__("checkpoint_eligible", True), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("checkpoint_ids", [f"cp-{i}" for i in range(11)]), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("stop_reason", "eos"), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("reasoning_status", "closed"), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("answer_parse_status", "malformed"), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("confidence_parse_status", "malformed"), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("infrastructure_failure_reference", None), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("terminal_error_details", None), "natural terminal infrastructure failure"),
        (lambda row: row.__setitem__("decoded_output", "stale"), "natural terminal infrastructure failure"),
    ],
)
def test_natural_terminal_failure_bundle_rejects_contradictions(
    mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    natural = _natural(
        outcome="terminal_infrastructure_failure", answer=None, confidence=None
    )
    mutate(natural)
    with pytest.raises(ValueError, match=message):
        _extract(natural, [])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.__setitem__("infrastructure_failure_reference", "audit:failure"), "complete checkpoint failure state"),
        (lambda row: row.__setitem__("terminal_error_details", {"category": "failure"}), "complete checkpoint failure state"),
        (lambda row: row.__setitem__("forced_generated_token_ids", None), "complete checkpoint generation field"),
        (lambda row: row.__setitem__("decoded_forced_output", None), "complete checkpoint generation field"),
        (lambda row: row.__setitem__("confidence_parse_status", "missing"), "checkpoint confidence"),
        (lambda row: row.__setitem__("normalized_confidence", float("inf")), "checkpoint confidence"),
    ],
)
def test_complete_checkpoint_bundle_rejects_contradictions(
    mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    mutate(checkpoints[0])
    with pytest.raises(ValueError, match=message):
        _extract(natural, checkpoints)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("checkpoint_model_output_status", "valid"),
        lambda row: row.__setitem__("answer_parse_status", "malformed"),
        lambda row: row.__setitem__("confidence_parse_status", "malformed"),
        lambda row: row.__setitem__("answer_token_status", "missing"),
        lambda row: row.__setitem__("entropy_status", "invalid"),
        lambda row: row.__setitem__("infrastructure_failure_reference", None),
        lambda row: row.__setitem__("terminal_error_details", None),
        lambda row: row.__setitem__("decoded_forced_output", "stale"),
        lambda row: row.__setitem__("normalized_confidence", 0.7),
    ],
)
def test_checkpoint_terminal_failure_bundle_rejects_contradictions(
    mutate: Callable[[dict[str, Any]], Any]
) -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    checkpoints[0] = _checkpoint(
        natural, 0, answer=None, outcome="terminal_infrastructure_failure"
    )
    mutate(checkpoints[0])
    with pytest.raises(ValueError, match="checkpoint terminal infrastructure failure"):
        _extract(natural, checkpoints)


def test_aggregate_model_output_status_requires_exact_valid_triad() -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    checkpoints[0]["checkpoint_model_output_status"] = "invalid"
    with pytest.raises(ValueError, match="aggregate invalid checkpoint"):
        _extract(natural, checkpoints)

    checkpoints = _checkpoints(natural)
    checkpoints[0]["entropy_status"] = "unavailable"
    checkpoints[0]["ad_logits_float32"] = None
    checkpoints[0]["ad_probabilities_float32"] = None
    checkpoints[0]["answer_entropy_nats"] = None
    checkpoints[0]["full_vocabulary_answer_step_entropy_nats"] = None
    checkpoints[0]["maximum_ad_probability"] = None
    with pytest.raises(ValueError, match="aggregate valid checkpoint"):
        _extract(natural, checkpoints)


@pytest.mark.parametrize(("index", "value"), [(0, 0), (10, 1)])
def test_requested_fraction_requires_canonical_float(index: int, value: int) -> None:
    natural = _natural()
    checkpoints = _checkpoints(natural)
    checkpoints[index]["requested_fraction"] = value
    with pytest.raises(ValueError, match="canonical float"):
        _extract(natural, checkpoints)
