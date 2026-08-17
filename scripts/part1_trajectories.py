"""Pure, deterministic trajectory feature extraction for Part 1 merged rows."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from part1_contract import (
    FIXED_CHECKPOINT_FRACTIONS,
    FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
)


PRIMARY_FEATURE_REGISTRY = tuple(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)
assert PRIMARY_FEATURE_REGISTRY == tuple(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY)

_ANSWERS = frozenset("ABCD")
_PROVENANCE_FIELDS = (
    "study_id",
    "model_run_id",
    "model_run_manifest_hash",
    "question_manifest_hash",
)
_GROUP_FIELDS = _PROVENANCE_FIELDS + (
    "question_id",
    "sample_index",
    "subject",
    "run_id",
)
_OUTPUT_NATURAL_FIELDS = _GROUP_FIELDS + (
    "raw_record_id",
    "natural_execution_outcome",
    "stop_reason",
    "reasoning_status",
    "answer_parse_status",
    "confidence_parse_status",
    "checkpoint_eligible",
    "natural_answer",
    "natural_correct",
)
_MAIN_FRACTION_INDICES = {"0.0": 0, "0.5": 5, "1.0": 10}
_NATURAL_COMPLETE_GENERATION_FIELDS = (
    "rendered_prompt",
    "prompt_token_ids",
    "generated_token_ids",
    "decoded_output",
    "reasoning_text",
    "reasoning_boundaries",
    "close_tag_information",
    "generated_token_count",
    "reasoning_token_count",
    "per_token_entropy_nats",
)
_NATURAL_FAILURE_NULL_FIELDS = _NATURAL_COMPLETE_GENERATION_FIELDS + (
    "mean_reasoning_entropy_nats",
    "tail_reasoning_entropy_nats",
    "terminal_answer_block_text",
    "terminal_answer_block_span",
    "natural_answer",
    "raw_confidence_text",
    "raw_parsed_confidence",
    "normalized_confidence",
    "natural_correct",
    "diagnostic_answer_like_text",
    "checkpoint_ids",
)
_CHECKPOINT_FAILURE_NULL_FIELDS = (
    "forced_generated_token_ids",
    "decoded_forced_output",
    "terminal_answer_block_text",
    "forced_answer",
    "raw_confidence_text",
    "raw_parsed_confidence",
    "normalized_confidence",
    "checkpoint_local_correct",
    "answer_token_index",
    "answer_token_id",
    "token_convention",
    "ad_token_ids",
    "ad_logits_float32",
    "ad_probabilities_float32",
    "answer_entropy_nats",
    "full_vocabulary_answer_step_entropy_nats",
    "maximum_ad_probability",
    "agrees_with_natural_answer",
)


def _same_json_scalar(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _finite_number(value: Any, *, minimum: float, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value) or value < minimum:
        return False
    return maximum is None or value <= maximum


def _natural_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in _GROUP_FIELDS)


def _natural_answer_is_valid(row: Mapping[str, Any]) -> bool:
    return (
        row.get("natural_execution_outcome") == "complete"
        and row.get("answer_parse_status") == "parsed"
        and row.get("natural_answer") in _ANSWERS
    )


def _checkpoint_answer_is_valid(row: Mapping[str, Any]) -> bool:
    return (
        row.get("checkpoint_execution_outcome") == "complete"
        and row.get("answer_parse_status") == "parsed"
        and row.get("forced_answer") in _ANSWERS
    )


def _validate_natural(row: Mapping[str, Any]) -> None:
    outcome = row.get("natural_execution_outcome")
    if outcome not in {"complete", "terminal_infrastructure_failure"}:
        raise ValueError("unsupported natural_execution_outcome")

    if outcome == "terminal_infrastructure_failure":
        if (
            row.get("checkpoint_eligible") is not False
            or row.get("stop_reason") != "error"
            or row.get("reasoning_status") != "malformed"
            or row.get("answer_parse_status") != "missing"
            or row.get("confidence_parse_status") != "missing"
            or not isinstance(row.get("infrastructure_failure_reference"), str)
            or not row["infrastructure_failure_reference"]
            or not isinstance(row.get("terminal_error_details"), Mapping)
            or not row["terminal_error_details"]
        ):
            raise ValueError("natural terminal infrastructure failure has an invalid status bundle")
        for field in _NATURAL_FAILURE_NULL_FIELDS:
            if row.get(field) is not None:
                raise ValueError(
                    f"natural terminal infrastructure failure requires null {field}"
                )
        return

    if row.get("checkpoint_eligible") is not True:
        raise ValueError("complete natural checkpoint eligibility must be true")
    checkpoint_ids = row.get("checkpoint_ids")
    if (
        not isinstance(checkpoint_ids, list)
        or len(checkpoint_ids) != 11
        or any(not isinstance(value, str) or not value for value in checkpoint_ids)
        or len(set(checkpoint_ids)) != 11
    ):
        raise ValueError("complete natural checkpoint_ids must be eleven unique strings")
    if (
        row.get("infrastructure_failure_reference") is not None
        or row.get("terminal_error_details") is not None
    ):
        raise ValueError("complete natural failure state must be null")
    for field in _NATURAL_COMPLETE_GENERATION_FIELDS:
        if row.get(field) is None:
            raise ValueError(f"complete natural generation field {field} must be non-null")

    count = row.get("reasoning_token_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("reasoning_token_count must be a nonnegative integer")
    mean = row.get("mean_reasoning_entropy_nats")
    tail = row.get("tail_reasoning_entropy_nats")
    if count == 0:
        if row.get("reasoning_status") != "no_reasoning":
            raise ValueError("zero reasoning_token_count requires no_reasoning status")
        if mean is not None or tail is not None:
            raise ValueError("zero reasoning requires null reasoning entropy summaries")
    else:
        if row.get("reasoning_status") == "no_reasoning":
            raise ValueError("no_reasoning status requires zero reasoning_token_count")
        for field, value in (
            ("mean_reasoning_entropy_nats", mean),
            ("tail_reasoning_entropy_nats", tail),
        ):
            if not _finite_number(value, minimum=0.0):
                raise ValueError(f"{field} must be finite and nonnegative")

    answer_valid = _natural_answer_is_valid(row)
    if row.get("answer_parse_status") == "parsed":
        if not answer_valid:
            raise ValueError("parsed natural answer must be one of A-D")
        if type(row.get("natural_correct")) is not bool:
            raise ValueError("parsed natural answer requires boolean natural_correct")
    elif row.get("natural_answer") is not None or row.get("natural_correct") is not None:
        raise ValueError("unparsed natural answer requires null answer and correctness")

    confidence = row.get("normalized_confidence")
    if row.get("confidence_parse_status") == "parsed":
        if not _finite_number(confidence, minimum=0.0, maximum=1.0):
            raise ValueError("parsed normalized_confidence must be finite in [0,1]")
    elif confidence is not None:
        raise ValueError("unparsed normalized_confidence must be null")


def _validate_checkpoint_metrics(row: Mapping[str, Any]) -> None:
    execution_complete = row.get("checkpoint_execution_outcome") == "complete"
    computed = row.get("entropy_status") == "computed"
    entropy = row.get("answer_entropy_nats")
    full_entropy = row.get("full_vocabulary_answer_step_entropy_nats")
    maximum = row.get("maximum_ad_probability")
    logits = row.get("ad_logits_float32")
    probabilities = row.get("ad_probabilities_float32")

    if computed:
        if not execution_complete:
            raise ValueError("computed checkpoint metrics require complete execution")
        if not _finite_number(entropy, minimum=0.0):
            raise ValueError("computed answer_entropy_nats must be finite and nonnegative")
        if not _finite_number(full_entropy, minimum=0.0):
            raise ValueError(
                "computed full_vocabulary_answer_step_entropy_nats must be finite and nonnegative"
            )
        if not _finite_number(maximum, minimum=0.0, maximum=1.0):
            raise ValueError("computed maximum_ad_probability must be finite in [0,1]")
        if not isinstance(logits, (list, tuple)) or len(logits) != 4:
            raise ValueError("computed ad_logits_float32 must contain four values")
        if not all(_finite_number(logit, minimum=-math.inf) for logit in logits):
            raise ValueError("ad_logits_float32 must be finite")
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 4:
            raise ValueError("computed ad_probabilities_float32 must contain four values")
        if not all(
            _finite_number(probability, minimum=0.0, maximum=1.0)
            for probability in probabilities
        ):
            raise ValueError("ad_probabilities_float32 must be finite in [0,1]")
        if not math.isclose(
            sum(float(probability) for probability in probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("ad_probabilities_float32 must sum to one")
        if not math.isclose(
            float(maximum), max(float(probability) for probability in probabilities),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("maximum_ad_probability must equal the A-D probability maximum")
    else:
        for field, value in (
            ("answer_entropy_nats", entropy),
            ("full_vocabulary_answer_step_entropy_nats", full_entropy),
            ("maximum_ad_probability", maximum),
            ("ad_logits_float32", logits),
            ("ad_probabilities_float32", probabilities),
        ):
            if value is not None:
                raise ValueError(f"{field} must be null when checkpoint entropy is unavailable")


def _validate_checkpoint_bundle(row: Mapping[str, Any]) -> None:
    outcome = row.get("checkpoint_execution_outcome")
    if outcome not in {"complete", "terminal_infrastructure_failure"}:
        raise ValueError("unsupported checkpoint_execution_outcome")

    if outcome == "terminal_infrastructure_failure":
        if (
            row.get("checkpoint_model_output_status") != "invalid"
            or row.get("answer_parse_status") != "missing"
            or row.get("confidence_parse_status") != "missing"
            or row.get("answer_token_status") != "unsupported"
            or row.get("entropy_status") != "unavailable"
            or not isinstance(row.get("infrastructure_failure_reference"), str)
            or not row["infrastructure_failure_reference"]
            or not isinstance(row.get("terminal_error_details"), Mapping)
            or not row["terminal_error_details"]
            or any(row.get(field) is not None for field in _CHECKPOINT_FAILURE_NULL_FIELDS)
        ):
            raise ValueError(
                "checkpoint terminal infrastructure failure has an invalid status/output bundle"
            )
        return

    if (
        row.get("infrastructure_failure_reference") is not None
        or row.get("terminal_error_details") is not None
    ):
        raise ValueError("complete checkpoint failure state must be null")
    if not isinstance(row.get("forced_generated_token_ids"), list) or not isinstance(
        row.get("decoded_forced_output"), str
    ):
        raise ValueError("complete checkpoint generation fields must be non-null")

    output_status = row.get("checkpoint_model_output_status")
    answer_status = row.get("answer_parse_status")
    confidence_status = row.get("confidence_parse_status")
    answer_token_status = row.get("answer_token_status")
    entropy_status = row.get("entropy_status")
    if output_status not in {"valid", "invalid"}:
        raise ValueError("unsupported checkpoint_model_output_status")
    if answer_status not in {"parsed", "missing", "malformed", "out_of_domain"}:
        raise ValueError("unsupported checkpoint answer_parse_status")
    if confidence_status not in {"parsed", "missing", "malformed", "out_of_range"}:
        raise ValueError("unsupported checkpoint confidence_parse_status")
    if answer_token_status not in {"located", "missing", "ambiguous", "unsupported"}:
        raise ValueError("unsupported checkpoint answer_token_status")
    if entropy_status not in {"computed", "unavailable", "invalid"}:
        raise ValueError("unsupported checkpoint entropy_status")

    answer_valid = _checkpoint_answer_is_valid(row)
    if answer_status == "parsed":
        if not answer_valid or type(row.get("checkpoint_local_correct")) is not bool:
            raise ValueError(
                "parsed checkpoint answer requires A-D and boolean checkpoint_local_correct"
            )
    elif row.get("forced_answer") is not None or row.get("checkpoint_local_correct") is not None:
        raise ValueError(
            "unparsed checkpoint answer requires null forced_answer and checkpoint_local_correct"
        )

    confidence = row.get("normalized_confidence")
    if confidence_status == "parsed":
        if not _finite_number(confidence, minimum=0.0, maximum=1.0):
            raise ValueError("parsed checkpoint confidence must be finite in [0,1]")
    elif confidence is not None:
        raise ValueError("unparsed checkpoint confidence must be null")

    if answer_token_status == "located":
        ad_token_ids = row.get("ad_token_ids")
        if (
            type(row.get("answer_token_index")) is not int
            or type(row.get("answer_token_id")) is not int
            or not isinstance(row.get("token_convention"), str)
            or not row["token_convention"]
            or not isinstance(ad_token_ids, list)
            or len(ad_token_ids) != 4
            or any(type(token_id) is not int for token_id in ad_token_ids)
        ):
            raise ValueError("located checkpoint answer token has an invalid token bundle")
    elif any(
        row.get(field) is not None
        for field in ("answer_token_index", "answer_token_id", "token_convention", "ad_token_ids")
    ):
        raise ValueError("unlocated checkpoint answer token requires null token fields")

    if entropy_status == "computed" and answer_token_status != "located":
        raise ValueError("computed checkpoint entropy requires a located answer token")

    ad_token_ids = row.get("ad_token_ids")
    answer_id_matches = (
        answer_valid
        and isinstance(ad_token_ids, list)
        and row.get("answer_token_id") == ad_token_ids["ABCD".index(row["forced_answer"])]
    )
    complete_valid_triad = (
        answer_valid
        and answer_token_status == "located"
        and entropy_status == "computed"
        and answer_id_matches
    )
    if output_status == "valid" and not complete_valid_triad:
        raise ValueError("aggregate valid checkpoint requires the complete valid triad")
    if output_status == "invalid" and complete_valid_triad:
        raise ValueError("aggregate invalid checkpoint cannot retain the complete valid triad")


def _validate_checkpoint(row: Mapping[str, Any], natural: Mapping[str, Any]) -> int:
    for field in _GROUP_FIELDS:
        if not _same_json_scalar(row.get(field), natural.get(field)):
            raise ValueError(f"mixed checkpoint provenance at {field}")
    if not _same_json_scalar(row.get("parent_raw_record_id"), natural.get("raw_record_id")):
        raise ValueError("checkpoint parent_raw_record_id does not match natural row")

    index = row.get("requested_checkpoint_index")
    fraction = row.get("requested_fraction")
    if type(index) is not int or not 0 <= index < len(FIXED_CHECKPOINT_FRACTIONS):
        raise ValueError("noncanonical requested checkpoint index")
    expected_fraction = FIXED_CHECKPOINT_FRACTIONS[index]
    if type(fraction) is not float or not math.isfinite(fraction) or fraction != expected_fraction:
        raise ValueError(
            "noncanonical requested checkpoint: requested_fraction must be canonical float"
        )

    expected_ids = natural.get("checkpoint_ids")
    if not isinstance(expected_ids, (list, tuple)) or len(expected_ids) != 11:
        raise ValueError("complete natural row requires eleven checkpoint_ids")
    if not _same_json_scalar(row.get("checkpoint_id"), expected_ids[index]):
        raise ValueError("checkpoint_id does not match parent requested index")
    if type(row.get("is_alias")) is not bool:
        raise ValueError("is_alias must be boolean")

    _validate_checkpoint_metrics(row)
    _validate_checkpoint_bundle(row)
    return index


def _index_checkpoints(
    natural: Mapping[str, Any], checkpoint_rows: Sequence[Mapping[str, Any]]
) -> dict[int, Mapping[str, Any]]:
    if natural.get("natural_execution_outcome") == "terminal_infrastructure_failure":
        if checkpoint_rows:
            raise ValueError("checkpoint exists under natural terminal infrastructure failure")
        return {}

    by_index: dict[int, Mapping[str, Any]] = {}
    checkpoint_ids: set[Any] = set()
    record_ids: set[Any] = set()
    previous_index = -1
    for checkpoint in checkpoint_rows:
        checkpoint_id = checkpoint.get("checkpoint_id")
        if isinstance(checkpoint_id, str) and checkpoint_id in checkpoint_ids:
            raise ValueError("duplicate checkpoint_id")
        index = _validate_checkpoint(checkpoint, natural)
        if index < previous_index:
            raise ValueError("out-of-order checkpoints for natural parent")
        previous_index = index
        if index in by_index:
            raise ValueError("duplicate checkpoint requested index")
        checkpoint_ids.add(checkpoint_id)
        record_id = checkpoint.get("checkpoint_record_id")
        if record_id in record_ids:
            raise ValueError("duplicate checkpoint_record_id")
        record_ids.add(record_id)
        by_index[index] = checkpoint
    return by_index


def _checkpoint_calibration_slot(
    *,
    index: int,
    expected_checkpoint_id: str | None,
    checkpoint: Mapping[str, Any] | None,
    natural_failure: bool,
) -> dict[str, Any]:
    fraction = FIXED_CHECKPOINT_FRACTIONS[index]
    if natural_failure:
        return {
            "requested_checkpoint_index": index,
            "requested_fraction": fraction,
            "checkpoint_id": None,
            "present": False,
            "eligibility_status": "ineligible_natural_failure",
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
            "confidence_missing_reason": "ineligible_natural_failure",
            "entropy_status": None,
            "maximum_ad_probability": None,
            "maximum_ad_probability_available": False,
            "maximum_ad_probability_missing_reason": "ineligible_natural_failure",
        }
    if checkpoint is None:
        return {
            "requested_checkpoint_index": index,
            "requested_fraction": fraction,
            "checkpoint_id": expected_checkpoint_id,
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

    execution_complete = checkpoint["checkpoint_execution_outcome"] == "complete"
    target_available = type(checkpoint["checkpoint_local_correct"]) is bool
    confidence_available = (
        execution_complete
        and checkpoint["confidence_parse_status"] == "parsed"
        and target_available
    )
    maximum_available = (
        execution_complete
        and checkpoint["entropy_status"] == "computed"
        and target_available
    )
    if not execution_complete:
        confidence_reason = "checkpoint_terminal_infrastructure_failure"
        maximum_reason = "checkpoint_terminal_infrastructure_failure"
    elif not target_available:
        confidence_reason = "checkpoint_local_correctness_unavailable"
        maximum_reason = "checkpoint_local_correctness_unavailable"
    else:
        confidence_reason = (
            None if confidence_available else "checkpoint_confidence_not_parsed"
        )
        maximum_reason = (
            None if maximum_available else "checkpoint_ad_probability_unavailable"
        )

    return {
        "requested_checkpoint_index": index,
        "requested_fraction": fraction,
        "checkpoint_id": expected_checkpoint_id,
        "present": True,
        "eligibility_status": "eligible",
        "is_alias": checkpoint["is_alias"],
        "checkpoint_execution_outcome": checkpoint["checkpoint_execution_outcome"],
        "checkpoint_model_output_status": checkpoint["checkpoint_model_output_status"],
        "answer_parse_status": checkpoint["answer_parse_status"],
        "forced_answer": checkpoint["forced_answer"],
        "answer_valid": _checkpoint_answer_is_valid(checkpoint),
        "checkpoint_local_correct": checkpoint["checkpoint_local_correct"],
        "confidence_parse_status": checkpoint["confidence_parse_status"],
        "normalized_confidence": (
            checkpoint["normalized_confidence"] if confidence_available else None
        ),
        "confidence_available": confidence_available,
        "confidence_missing_reason": confidence_reason,
        "entropy_status": checkpoint["entropy_status"],
        "maximum_ad_probability": (
            checkpoint["maximum_ad_probability"] if maximum_available else None
        ),
        "maximum_ad_probability_available": maximum_available,
        "maximum_ad_probability_missing_reason": maximum_reason,
    }


def _build_checkpoint_calibration(
    natural: Mapping[str, Any], checkpoints: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    natural_failure = (
        natural.get("natural_execution_outcome") == "terminal_infrastructure_failure"
    )
    expected_ids = natural.get("checkpoint_ids")
    return [
        _checkpoint_calibration_slot(
            index=index,
            expected_checkpoint_id=(
                None if natural_failure else expected_ids[index]
            ),
            checkpoint=checkpoints.get(index),
            natural_failure=natural_failure,
        )
        for index in range(11)
    ]


def _infrastructure_failure_row(natural: Mapping[str, Any]) -> dict[str, Any]:
    reason = "natural_terminal_infrastructure_failure"
    row = {field: natural.get(field) for field in _OUTPUT_NATURAL_FIELDS}
    primary = {feature: None for feature in PRIMARY_FEATURE_REGISTRY}
    row.update(primary)
    row.update(
        {
            "answer_switch_count": None,
            "valid_transition_count": None,
            "transition_evaluability_status": "unavailable",
            "transition_evaluability_reason": reason,
            "first_natural_answer_appearance_fraction": None,
            "first_natural_answer_appearance_status": "unavailable",
            "first_natural_answer_appearance_reason": reason,
            "left_correct_answer": None,
            "left_correct_answer_status": "unavailable",
            "left_correct_answer_reason": reason,
            "later_recovered_correct_answer": None,
            "later_recovered_correct_answer_status": "unavailable",
            "later_recovered_correct_answer_reason": reason,
            "forced_endpoint_agrees_with_natural": None,
            "forced_endpoint_agrees_with_natural_status": "unavailable",
            "forced_endpoint_agrees_with_natural_reason": reason,
            "stabilization_fraction": None,
            "stabilization_status": "unavailable",
            "stabilization_reason": reason,
            "checkpoint_calibration": _build_checkpoint_calibration(natural, {}),
            "feature_missing_reasons": {
                feature: reason for feature in PRIMARY_FEATURE_REGISTRY
            },
        }
    )
    return row


def extract_trajectory_features(
    natural_row: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive one immutable analysis row from one natural terminal result."""

    _validate_natural(natural_row)
    checkpoints = _index_checkpoints(natural_row, checkpoint_rows)
    if natural_row.get("natural_execution_outcome") == "terminal_infrastructure_failure":
        return _infrastructure_failure_row(natural_row)

    row = {field: natural_row.get(field) for field in _OUTPUT_NATURAL_FIELDS}
    primary: dict[str, float | int | None] = {}
    reasons: dict[str, str | None] = {}

    reasoning_count = natural_row["reasoning_token_count"]
    for output_name, input_name in (
        ("negative_mean_reasoning_entropy", "mean_reasoning_entropy_nats"),
        ("negative_tail_reasoning_entropy", "tail_reasoning_entropy_nats"),
    ):
        if reasoning_count == 0:
            primary[output_name] = None
            reasons[output_name] = "no_reasoning_tokens"
        else:
            primary[output_name] = -natural_row[input_name]
            reasons[output_name] = None

    for fraction_label, index in _MAIN_FRACTION_INDICES.items():
        checkpoint = checkpoints.get(index)
        entropy_name = f"negative_answer_entropy_fraction_{fraction_label}"
        maximum_name = f"maximum_ad_probability_fraction_{fraction_label}"
        if checkpoint is None:
            primary[entropy_name] = None
            reasons[entropy_name] = "checkpoint_fraction_missing"
            primary[maximum_name] = None
            reasons[maximum_name] = "checkpoint_fraction_missing"
        elif checkpoint.get("entropy_status") != "computed" or checkpoint.get(
            "checkpoint_execution_outcome"
        ) != "complete":
            primary[entropy_name] = None
            reasons[entropy_name] = "checkpoint_entropy_not_computed"
            primary[maximum_name] = None
            reasons[maximum_name] = "checkpoint_ad_probability_unavailable"
        else:
            primary[entropy_name] = -checkpoint["answer_entropy_nats"]
            reasons[entropy_name] = None
            primary[maximum_name] = checkpoint["maximum_ad_probability"]
            reasons[maximum_name] = None

    confidence = natural_row.get("normalized_confidence")
    if natural_row.get("confidence_parse_status") == "parsed":
        primary["natural_verbalized_confidence"] = confidence
        reasons["natural_verbalized_confidence"] = None
    else:
        primary["natural_verbalized_confidence"] = None
        reasons["natural_verbalized_confidence"] = "natural_confidence_not_parsed"

    natural_answer_valid = _natural_answer_is_valid(natural_row)
    natural_answer = natural_row.get("natural_answer")
    first_appearance: float | None = None
    if natural_answer_valid:
        for index in range(11):
            checkpoint = checkpoints.get(index)
            if (
                checkpoint is not None
                and _checkpoint_answer_is_valid(checkpoint)
                and checkpoint["forced_answer"] == natural_answer
            ):
                first_appearance = FIXED_CHECKPOINT_FRACTIONS[index]
                break
        if first_appearance is None:
            appearance_status = "not_found"
            appearance_reason = "natural_answer_never_appeared"
        else:
            appearance_status = "found"
            appearance_reason = None
    else:
        appearance_status = "unavailable"
        appearance_reason = "natural_answer_invalid_or_missing"

    switches = 0
    transitions = 0
    previous: Mapping[str, Any] | None = None
    left = False
    recovered = False
    for index in range(11):
        checkpoint = checkpoints.get(index)
        if checkpoint is None:
            previous = None
            continue
        if checkpoint["is_alias"]:
            continue
        if not _checkpoint_answer_is_valid(checkpoint):
            previous = None
            continue
        if left and checkpoint["checkpoint_local_correct"] is True:
            recovered = True
        if previous is not None:
            transitions += 1
            changed = checkpoint["forced_answer"] != previous["forced_answer"]
            if changed:
                switches += 1
            if (
                changed
                and previous["checkpoint_local_correct"] is True
                and checkpoint["checkpoint_local_correct"] is False
            ):
                left = True
        previous = checkpoint

    if left:
        recovery_status = "evaluated"
        recovery_reason = None if recovered else "no_later_recovery"
    else:
        recovery_status = "not_applicable"
        recovery_reason = "not_applicable_no_leave"

    endpoint = checkpoints.get(10)
    endpoint_valid = endpoint is not None and _checkpoint_answer_is_valid(endpoint)
    if not natural_answer_valid:
        endpoint_agreement = None
        endpoint_status = "unavailable"
        endpoint_reason = "natural_answer_invalid_or_missing"
    elif endpoint is None:
        endpoint_agreement = None
        endpoint_status = "unavailable"
        endpoint_reason = "checkpoint_1.0_missing"
    elif not endpoint_valid:
        endpoint_agreement = None
        endpoint_status = "unavailable"
        endpoint_reason = "checkpoint_1.0_answer_invalid"
    else:
        endpoint_agreement = endpoint["forced_answer"] == natural_answer
        endpoint_status = "evaluated"
        endpoint_reason = None

    stabilization: float | None = None
    if endpoint is None:
        stabilization_status = "unavailable"
        stabilization_reason = "checkpoint_1.0_missing"
    elif not endpoint_valid:
        stabilization_status = "unavailable"
        stabilization_reason = "checkpoint_1.0_answer_invalid"
    else:
        reference = endpoint["forced_answer"]
        suffix_is_complete_and_equal = True
        for index in range(10, -1, -1):
            checkpoint = checkpoints.get(index)
            if checkpoint is None:
                suffix_is_complete_and_equal = False
            elif checkpoint["is_alias"]:
                continue
            elif (
                _checkpoint_answer_is_valid(checkpoint)
                and checkpoint["forced_answer"] == reference
            ):
                if suffix_is_complete_and_equal:
                    stabilization = FIXED_CHECKPOINT_FRACTIONS[index]
            else:
                suffix_is_complete_and_equal = False
        if stabilization is None:
            stabilization_status = "unavailable"
            stabilization_reason = "no_complete_stable_physical_suffix"
        else:
            stabilization_status = "computed"
            stabilization_reason = None

    primary["negative_answer_switch_count"] = -switches
    reasons["negative_answer_switch_count"] = None
    primary["negative_stabilization_fraction"] = (
        -stabilization if stabilization is not None else None
    )
    reasons["negative_stabilization_fraction"] = stabilization_reason

    if set(primary) != set(PRIMARY_FEATURE_REGISTRY):
        raise AssertionError("produced primary feature registry differs from fixed contract")
    ordered_primary = {feature: primary[feature] for feature in PRIMARY_FEATURE_REGISTRY}
    ordered_reasons = {feature: reasons[feature] for feature in PRIMARY_FEATURE_REGISTRY}
    row.update(ordered_primary)
    row.update(
        {
            "answer_switch_count": switches,
            "valid_transition_count": transitions,
            "transition_evaluability_status": "evaluated",
            "transition_evaluability_reason": None,
            "first_natural_answer_appearance_fraction": first_appearance,
            "first_natural_answer_appearance_status": appearance_status,
            "first_natural_answer_appearance_reason": appearance_reason,
            "left_correct_answer": left,
            "left_correct_answer_status": "evaluated",
            "left_correct_answer_reason": None if left else "no_leave_event",
            "later_recovered_correct_answer": recovered,
            "later_recovered_correct_answer_status": recovery_status,
            "later_recovered_correct_answer_reason": recovery_reason,
            "forced_endpoint_agrees_with_natural": endpoint_agreement,
            "forced_endpoint_agrees_with_natural_status": endpoint_status,
            "forced_endpoint_agrees_with_natural_reason": endpoint_reason,
            "stabilization_fraction": stabilization,
            "stabilization_status": stabilization_status,
            "stabilization_reason": stabilization_reason,
            "checkpoint_calibration": _build_checkpoint_calibration(
                natural_row, checkpoints
            ),
            "feature_missing_reasons": ordered_reasons,
        }
    )
    return row


def build_trajectory_rows(
    natural_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build exactly one deterministically ordered analysis row per natural row."""

    naturals_by_id: dict[Any, Mapping[str, Any]] = {}
    natural_keys: set[tuple[Any, ...]] = set()
    common_provenance: tuple[Any, ...] | None = None
    for natural in natural_rows:
        raw_record_id = natural.get("raw_record_id")
        if raw_record_id in naturals_by_id:
            raise ValueError("duplicate natural raw_record_id")
        logical_key = _natural_key(natural)
        if logical_key in natural_keys:
            raise ValueError("duplicate natural logical key")
        provenance = tuple(natural.get(field) for field in _PROVENANCE_FIELDS)
        if common_provenance is None:
            common_provenance = provenance
        elif any(
            not _same_json_scalar(actual, expected)
            for actual, expected in zip(provenance, common_provenance, strict=True)
        ):
            raise ValueError("mixed natural provenance")
        natural_keys.add(logical_key)
        naturals_by_id[raw_record_id] = natural

    grouped: dict[Any, list[Mapping[str, Any]]] = {
        raw_record_id: [] for raw_record_id in naturals_by_id
    }
    checkpoint_indices_by_parent: dict[Any, set[Any]] = {
        raw_record_id: set() for raw_record_id in naturals_by_id
    }
    checkpoint_record_ids: set[Any] = set()
    for checkpoint in checkpoint_rows:
        parent_id = checkpoint.get("parent_raw_record_id")
        natural = naturals_by_id.get(parent_id)
        if natural is None:
            raise ValueError("checkpoint without a parent natural row")
        if natural.get("natural_execution_outcome") == "terminal_infrastructure_failure":
            raise ValueError("checkpoint exists under natural terminal infrastructure failure")
        requested_index = checkpoint.get("requested_checkpoint_index")
        if requested_index in checkpoint_indices_by_parent[parent_id]:
            raise ValueError("duplicate checkpoint requested index")
        checkpoint_indices_by_parent[parent_id].add(requested_index)
        record_id = checkpoint.get("checkpoint_record_id")
        if record_id in checkpoint_record_ids:
            raise ValueError("duplicate checkpoint_record_id")
        checkpoint_record_ids.add(record_id)
        grouped[parent_id].append(checkpoint)

    rows = [
        extract_trajectory_features(natural, grouped[raw_record_id])
        for raw_record_id, natural in naturals_by_id.items()
    ]
    return sorted(
        rows,
        key=lambda row: (row["sample_index"], row["run_id"], row["raw_record_id"]),
    )
