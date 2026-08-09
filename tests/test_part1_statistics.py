"""Pure fixed-statistics tests; safe on a login node."""

from __future__ import annotations

import json
from typing import Any

import pytest

from part1_contract import (
    FIXED_CHECKPOINT_FRACTIONS,
    FIXED_PRIMARY_AUROC_FEATURE_REGISTRY,
    FIXED_SUBJECTS,
)


def _row(
    subject: str,
    question_id: str,
    run_id: int,
    correct: bool | None,
    value: float | None,
    *,
    checkpoint_correct: bool | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "study_id": "study",
        "model_run_id": "model",
        "subject": subject,
        "question_id": question_id,
        "run_id": run_id,
        "natural_correct": correct,
    }
    for offset, feature in enumerate(FIXED_PRIMARY_AUROC_FEATURE_REGISTRY):
        row[feature] = None if value is None else float(value) + offset / 100.0
    row["natural_verbalized_confidence"] = confidence if confidence is not None else value
    local_target = correct if checkpoint_correct is None else checkpoint_correct
    row["checkpoint_calibration"] = [
        {
            "requested_checkpoint_index": index,
            "requested_fraction": fraction,
            "checkpoint_local_correct": local_target,
            "normalized_confidence": confidence if confidence is not None else value,
            "maximum_ad_probability": (
                None
                if value is None
                else min(1.0, max(0.0, float(value) + index / 100.0))
            ),
        }
        for index, fraction in enumerate(FIXED_CHECKPOINT_FRACTIONS)
    ]
    return row


def _balanced_rows() -> list[dict[str, Any]]:
    return [
        _row(subject, f"{subject}-q", run_id, correct, value)
        for subject in FIXED_SUBJECTS
        for run_id, (correct, value) in enumerate(((False, 0.1), (True, 0.9)))
    ]


def _one_draw_each_subject() -> list[dict[str, Any]]:
    from part1_bootstrap import build_question_draw_plan

    return build_question_draw_plan(
        [
            {"subject": subject, "question_id": f"{subject}-q"}
            for subject in FIXED_SUBJECTS
        ],
        replicates=3,
    )


@pytest.mark.parametrize(
    ("targets", "scores", "expected"),
    [
        ([False, False, True, True], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([False, False, True, True], [0.9, 0.8, 0.2, 0.1], 0.0),
        ([False, True], [0.5, 0.5], 0.5),
        ([False, False, True, True], [0.0, 1.0, 1.0, 1.0], 0.75),
        ([True, True], [0.1, 0.2], None),
    ],
)
def test_rank_auroc_exact_average_rank_values(
    targets: list[bool], scores: list[float], expected: float | None
) -> None:
    from part1_statistics import rank_auroc

    assert rank_auroc(targets, scores) == expected


@pytest.mark.parametrize(
    ("targets", "scores"),
    [([1, False], [0.1, 0.2]), (["true", False], [0.1, 0.2]),
     ([True, False], [True, 0.2]), ([True, False], [float("nan"), 0.2]),
     ([True], [0.1, 0.2])],
)
def test_rank_auroc_rejects_nonboolean_targets_and_nonfinite_scores(
    targets: list[Any], scores: list[Any]
) -> None:
    from part1_statistics import rank_auroc

    with pytest.raises(ValueError):
        rank_auroc(targets, scores)


def test_reliability_boundaries_empty_bins_and_weighted_ece() -> None:
    from part1_statistics import reliability_ece

    result = reliability_ece([False, True, False, True], [0.0, 0.1, 0.9, 1.0])

    assert result["ece"] == pytest.approx(0.45)
    assert result["sample_size"] == 4
    assert len(result["bins"]) == 10
    assert [bin_row["count"] for bin_row in result["bins"]] == [1, 1] + [0] * 7 + [2]
    assert result["bins"][0] == {
        "bin_index": 0,
        "bin_lower": 0.0,
        "bin_upper": 0.1,
        "upper_inclusive": False,
        "count": 1,
        "mean_confidence": 0.0,
        "empirical_accuracy": 0.0,
        "absolute_gap": 0.0,
        "weighted_ece_contribution": 0.0,
    }
    assert result["bins"][2]["mean_confidence"] is None
    assert result["bins"][2]["empirical_accuracy"] is None
    assert result["bins"][2]["absolute_gap"] is None
    assert result["bins"][2]["weighted_ece_contribution"] == 0.0
    assert result["bins"][9]["upper_inclusive"] is True
    assert result["bins"][9]["mean_confidence"] == pytest.approx(0.95)


def test_reliability_rejects_out_of_range_and_has_explicit_empty_result() -> None:
    from part1_statistics import reliability_ece

    empty = reliability_ece([], [])
    assert empty["ece"] is None
    assert empty["sample_size"] == 0
    assert len(empty["bins"]) == 10
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        reliability_ece([True], [1.01])


def test_primary_auroc_exact_registry_groups_counts_and_json_safety() -> None:
    from part1_statistics import primary_auroc_analysis

    rows = _balanced_rows()
    rows.append(_row(FIXED_SUBJECTS[0], "missing-target", 0, None, 0.4))
    rows.append(_row(FIXED_SUBJECTS[0], "missing-predictor", 0, True, None))
    rows[-1]["reasoning_status"] = "malformed"
    rows[-1]["stop_reason"] = "length"
    result = primary_auroc_analysis(rows, _one_draw_each_subject())

    assert result["analysis_label"] == "primary_auroc"
    assert result["target"] == "natural_correct"
    assert result["feature_registry"] == FIXED_PRIMARY_AUROC_FEATURE_REGISTRY
    assert len(result["metric_rows"]) == 11 * 7
    first_feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]
    pooled = next(
        row for row in result["metric_rows"]
        if row["feature"] == first_feature and row["grouping"] == "pooled"
    )
    assert pooled["analysis_label"] == "primary_auroc"
    assert pooled["predictor"] == first_feature
    assert pooled["cohort_definition"] == "boolean_target_and_finite_predictor"
    assert pooled["total_candidate_rows"] == 12
    assert pooled["target_missing_count"] == 1
    assert pooled["predictor_missing_count"] == 1
    assert pooled["sample_size"] == 10
    assert pooled["positive_count"] == 5
    assert pooled["negative_count"] == 5
    assert pooled["point_estimate"] == 1.0
    assert pooled["subject"] is None
    assert pooled["requested_replicates"] == 3
    macro = next(
        row for row in result["metric_rows"]
        if row["feature"] == first_feature and row["grouping"] == "macro"
    )
    assert macro["point_estimate"] == 1.0
    assert macro["subject"] is None
    json.dumps(result, allow_nan=False)


def test_primary_auroc_rejects_registry_target_and_malformed_predictors() -> None:
    from part1_statistics import primary_auroc_analysis

    rows = _balanced_rows()
    plan = _one_draw_each_subject()
    with pytest.raises(ValueError, match="registry"):
        primary_auroc_analysis(rows, plan, feature_registry=FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[:-1])
    with pytest.raises(ValueError, match="natural_correct"):
        primary_auroc_analysis(rows, plan, target="checkpoint_local_correct")
    rows[0][FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]] = True
    with pytest.raises(ValueError, match="finite real"):
        primary_auroc_analysis(rows, plan)


def test_primary_bootstrap_one_class_subject_invalidates_macro_only() -> None:
    from part1_statistics import primary_auroc_analysis

    feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]
    rows: list[dict[str, Any]] = []
    draw_plan: list[dict[str, Any]] = []
    for subject_index, subject in enumerate(FIXED_SUBJECTS):
        if subject_index == 0:
            rows.extend([
                _row(subject, "false-q", 0, False, 0.1),
                _row(subject, "true-q", 0, True, 0.9),
            ])
            selected = "false-q"
        else:
            question_id = f"{subject}-mixed-q"
            rows.extend([
                _row(subject, question_id, 0, False, 0.1),
                _row(subject, question_id, 1, True, 0.9),
            ])
            selected = question_id
        draw_plan.append({
            "replicate_id": 0,
            "subject": subject,
            "draw_index": 0,
            "draw_id": f"draw-{subject_index}",
            "question_id": selected,
        })

    result = primary_auroc_analysis(rows, draw_plan)
    math_bootstrap = next(
        row for row in result["bootstrap_rows"]
        if row["feature"] == feature and row["grouping"] == "subject"
        and row["subject"] == FIXED_SUBJECTS[0]
    )
    macro_bootstrap = next(
        row for row in result["bootstrap_rows"]
        if row["feature"] == feature and row["grouping"] == "macro"
    )
    pooled_bootstrap = next(
        row for row in result["bootstrap_rows"]
        if row["feature"] == feature and row["grouping"] == "pooled"
    )
    assert math_bootstrap["point_estimate"] is None
    assert math_bootstrap["invalid_reason"] == "single_target_class"
    assert macro_bootstrap["point_estimate"] is None
    assert macro_bootstrap["invalid_reason"] == "incomplete_subject_macro"
    assert pooled_bootstrap["point_estimate"] == 1.0


def test_natural_calibration_uses_only_natural_confidence_and_subject_macro() -> None:
    from part1_statistics import natural_calibration_analysis

    rows = _balanced_rows()
    result = natural_calibration_analysis(rows, _one_draw_each_subject())

    assert result["analysis_label"] == "natural_calibration"
    assert {row["predictor"] for row in result["metric_rows"]} == {
        "natural_verbalized_confidence"
    }
    assert {row["target"] for row in result["metric_rows"]} == {"natural_correct"}
    assert len(result["metric_rows"]) == 7
    assert len(result["reliability_rows"]) == 6 * 10
    subject_values = [
        row["point_estimate"] for row in result["metric_rows"]
        if row["grouping"] == "subject"
    ]
    macro = next(row for row in result["metric_rows"] if row["grouping"] == "macro")
    assert macro["point_estimate"] == pytest.approx(sum(subject_values) / 5)
    assert all("entropy" not in row["predictor"] for row in result["metric_rows"])


def test_checkpoint_calibration_keeps_fractions_pairing_main_markers_and_bins() -> None:
    from part1_statistics import checkpoint_calibration_analysis

    rows = _balanced_rows()
    # Prove local correctness, rather than natural correctness, is paired to confidence.
    for row in rows:
        for slot in row["checkpoint_calibration"]:
            slot["checkpoint_local_correct"] = not row["natural_correct"]
            slot["normalized_confidence"] = 0.9 if slot["checkpoint_local_correct"] else 0.1
            slot["maximum_ad_probability"] = 0.8 if slot["checkpoint_local_correct"] else 0.2
    result = checkpoint_calibration_analysis(rows, _one_draw_each_subject())

    assert result["analysis_label"] == "checkpoint_calibration"
    assert len(result["metric_rows"]) == 2 * 11 * 7
    assert len(result["reliability_rows"]) == 2 * 11 * 6 * 10
    assert {row["requested_fraction"] for row in result["metric_rows"]} == set(
        FIXED_CHECKPOINT_FRACTIONS
    )
    assert {
        row["requested_fraction"] for row in result["metric_rows"] if row["is_main_checkpoint"]
    } == {0.0, 0.5, 1.0}
    assert {row["target"] for row in result["metric_rows"]} == {
        "checkpoint_local_correct"
    }
    assert {row["predictor"] for row in result["metric_rows"]} == {
        "checkpoint_normalized_confidence",
        "checkpoint_maximum_ad_probability",
    }
    pooled_confidence = next(
        row for row in result["metric_rows"]
        if row["grouping"] == "pooled"
        and row["predictor"] == "checkpoint_normalized_confidence"
        and row["requested_fraction"] == 0.0
    )
    assert pooled_confidence["point_estimate"] == pytest.approx(0.1)


def test_checkpoint_calibration_rejects_entropy_family_and_bad_slots() -> None:
    from part1_statistics import checkpoint_calibration_analysis

    rows = _balanced_rows()
    plan = _one_draw_each_subject()
    with pytest.raises(ValueError, match="calibration predictors"):
        checkpoint_calibration_analysis(rows, plan, predictors=("answer_entropy_nats",))
    rows[0]["checkpoint_calibration"] = rows[0]["checkpoint_calibration"][:-1]
    with pytest.raises(ValueError, match="eleven checkpoint calibration slots"):
        checkpoint_calibration_analysis(rows, plan)


def test_secondary_checkpoint_auroc_is_separate_and_local() -> None:
    from part1_statistics import secondary_checkpoint_auroc_analysis

    result = secondary_checkpoint_auroc_analysis(_balanced_rows(), _one_draw_each_subject())
    assert result["analysis_label"] == "secondary_checkpoint_local_auroc"
    assert {row["analysis_label"] for row in result["metric_rows"]} == {
        "secondary_checkpoint_local_auroc"
    }
    assert {row["target"] for row in result["metric_rows"]} == {
        "checkpoint_local_correct"
    }
    assert len(result["metric_rows"]) == 2 * 11 * 7
    assert all(row["analysis_label"] != "primary_auroc" for row in result["metric_rows"])


def test_within_question_exact_means_equal_weights_missing_sides_and_distribution() -> None:
    from part1_bootstrap import build_question_draw_plan
    from part1_statistics import within_question_analysis

    subject = FIXED_SUBJECTS[0]
    rows = [
        _row(subject, "q1", 0, True, 5.0),
        _row(subject, "q1", 1, True, 7.0),
        _row(subject, "q1", 2, True, None),
        _row(subject, "q1", 3, False, 1.0),
        _row(subject, "q2", 0, True, 4.0),
        _row(subject, "q2", 1, False, 0.0),
        _row(subject, "q2", 2, False, 2.0),
        _row(subject, "q3", 0, True, 9.0),
        _row(subject, "q3", 1, None, 1.0),
    ]
    missing_feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[1]
    for row in rows:
        if row["question_id"] == "q2" and row["natural_correct"] is False:
            row[missing_feature] = None
    plan = build_question_draw_plan(
        [{"subject": subject, "question_id": question_id} for question_id in ("q1", "q2", "q3")],
        replicates=3,
        small_fixture=True,
    )
    result = within_question_analysis(rows, plan)

    feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]
    distribution = [row for row in result["distribution_rows"] if row["feature"] == feature]
    assert [(row["question_id"], row["correct_run_count"], row["incorrect_run_count"])
            for row in distribution] == [("q1", 2, 1), ("q2", 1, 2)]
    assert distribution[0]["correct_run_mean"] == 6.0
    assert distribution[0]["incorrect_run_mean"] == 1.0
    assert distribution[0]["paired_difference"] == 5.0
    assert distribution[1]["paired_difference"] == 3.0
    summary = next(row for row in result["summary_rows"] if row["feature"] == feature)
    assert summary["qualifying_question_count"] == 2
    assert summary["mean_paired_difference"] == 4.0
    assert summary["median_paired_difference"] == 4.0
    missing_summary = next(
        row for row in result["summary_rows"] if row["feature"] == missing_feature
    )
    assert missing_summary["qualifying_question_count"] == 1


def test_within_question_bootstrap_preserves_repeated_draws_in_mean() -> None:
    from part1_statistics import within_question_analysis

    subject = FIXED_SUBJECTS[0]
    rows = [
        _row(subject, "q1", 0, True, 6.0),
        _row(subject, "q1", 1, False, 1.0),
        _row(subject, "q2", 0, True, 4.0),
        _row(subject, "q2", 1, False, 1.0),
    ]
    plan = []
    for replicate_id, questions in enumerate((("q1", "q1", "q1", "q2"), ("q1", "q2", "q2", "q2"))):
        for draw_index, question_id in enumerate(questions):
            plan.append({
                "replicate_id": replicate_id,
                "subject": subject,
                "draw_index": draw_index,
                "draw_id": f"r{replicate_id}-d{draw_index}",
                "question_id": question_id,
            })
    result = within_question_analysis(rows, plan)
    feature = FIXED_PRIMARY_AUROC_FEATURE_REGISTRY[0]
    bootstrap = [row for row in result["bootstrap_rows"] if row["feature"] == feature]
    assert [row["point_estimate"] for row in bootstrap] == [4.5, 3.5]
    draw_rows = [
        row for row in result["bootstrap_draw_rows"] if row["feature"] == feature
    ]
    assert [row["draw_id"] for row in draw_rows] == [
        "r0-d0", "r0-d1", "r0-d2", "r0-d3", "r1-d0", "r1-d1", "r1-d2", "r1-d3"
    ]
    assert [row["question_id"] for row in draw_rows[:4]] == ["q1", "q1", "q1", "q2"]
    summary = next(row for row in result["summary_rows"] if row["feature"] == feature)
    assert summary["lower"] == pytest.approx(3.525)
    assert summary["upper"] == pytest.approx(4.475)
    json.dumps(result, allow_nan=False)


def test_within_question_replicate_without_qualifying_draw_is_invalid() -> None:
    from part1_statistics import within_question_analysis

    subject = FIXED_SUBJECTS[0]
    rows = [
        _row(subject, "mixed", 0, True, 2.0),
        _row(subject, "mixed", 1, False, 1.0),
        _row(subject, "one-class", 0, True, 3.0),
    ]
    plan = [{
        "replicate_id": 0,
        "subject": subject,
        "draw_index": 0,
        "draw_id": "draw-0",
        "question_id": "one-class",
    }]
    result = within_question_analysis(rows, plan)
    assert all(row["point_estimate"] is None for row in result["bootstrap_rows"])
    assert {row["invalid_reason"] for row in result["bootstrap_rows"]} == {
        "no_qualifying_drawn_questions"
    }
