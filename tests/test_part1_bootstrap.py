"""Pure bootstrap tests; safe on a login node."""

from __future__ import annotations

import json

import pytest

from part1_contract import FIXED_SUBJECTS


def _questions(subjects: tuple[str, ...] = ("s1", "s2")) -> list[dict[str, str]]:
    return [
        {"subject": subject, "question_id": f"{subject}-q{index}"}
        for subject in subjects
        for index in range(3)
    ]


def test_draw_plan_is_seed_stable_sensitive_and_preserves_subject_sizes() -> None:
    from part1_bootstrap import build_question_draw_plan

    questions = _questions()
    first = build_question_draw_plan(
        questions, replicates=4, seed=42, small_fixture=True
    )
    second = build_question_draw_plan(
        questions, replicates=4, seed=42, small_fixture=True
    )
    different = build_question_draw_plan(
        questions, replicates=4, seed=43, small_fixture=True
    )

    assert first == second
    assert first != different
    assert len(first) == 4 * 2 * 3
    assert len({row["draw_id"] for row in first}) == len(first)
    assert all(type(row["replicate_id"]) is int for row in first)
    assert all(type(row["draw_index"]) is int for row in first)
    assert [row["replicate_id"] for row in first] == sorted(
        row["replicate_id"] for row in first
    )
    for replicate_id in range(4):
        for subject in ("s1", "s2"):
            rows = [
                row
                for row in first
                if row["replicate_id"] == replicate_id and row["subject"] == subject
            ]
            assert [row["draw_index"] for row in rows] == [0, 1, 2]


def test_default_draw_plan_requires_exact_fixed_subject_presence_and_order() -> None:
    from part1_bootstrap import build_question_draw_plan

    production_fixture = [
        {"subject": subject, "question_id": f"{subject}-q0"}
        for subject in FIXED_SUBJECTS
    ]
    plan = build_question_draw_plan(production_fixture, replicates=1)
    assert [row["subject"] for row in plan] == FIXED_SUBJECTS

    with pytest.raises(ValueError, match="fixed subject presence and order"):
        build_question_draw_plan(_questions(), replicates=1)
    with pytest.raises(ValueError, match="fixed subject presence and order"):
        build_question_draw_plan(list(reversed(production_fixture)), replicates=1)


def test_draw_plan_rejects_duplicate_questions_and_invalid_controls() -> None:
    from part1_bootstrap import build_question_draw_plan

    questions = _questions(("s1",))
    with pytest.raises(ValueError, match="unique subject/question"):
        build_question_draw_plan(questions + [dict(questions[0])], small_fixture=True)
    with pytest.raises(ValueError, match="replicates"):
        build_question_draw_plan(questions, replicates=True, small_fixture=True)
    with pytest.raises(ValueError, match="seed"):
        build_question_draw_plan(questions, seed=True, small_fixture=True)


def test_expansion_preserves_draw_multiplicity_runs_and_nested_checkpoints() -> None:
    from part1_bootstrap import expand_question_draws

    draw_plan = [
        {
            "replicate_id": 0,
            "subject": "s1",
            "draw_index": index,
            "draw_id": f"draw-{index}",
            "question_id": "q1",
        }
        for index in range(3)
    ]
    rows = [
        {
            "subject": "s1",
            "question_id": "q1",
            "run_id": run_id,
            "checkpoint_calibration": [{"requested_fraction": 0.0, "value": run_id}],
        }
        for run_id in range(2)
    ]

    expanded = expand_question_draws(draw_plan, rows)

    assert len(expanded) == 3 * 2
    assert [row["draw_id"] for row in expanded] == [
        "draw-0",
        "draw-0",
        "draw-1",
        "draw-1",
        "draw-2",
        "draw-2",
    ]
    assert [row["run_id"] for row in expanded] == [0, 1, 0, 1, 0, 1]
    expanded[0]["checkpoint_calibration"][0]["value"] = 99
    assert rows[0]["checkpoint_calibration"][0]["value"] == 0


def test_expansion_validates_subject_question_consistency_and_draw_ids() -> None:
    from part1_bootstrap import expand_question_draws

    plan = [
        {
            "replicate_id": 0,
            "subject": "s1",
            "draw_index": 0,
            "draw_id": "draw-0",
            "question_id": "q1",
        }
    ]
    with pytest.raises(ValueError, match="subject/question consistency"):
        expand_question_draws(
            plan,
            [
                {"subject": "s1", "question_id": "q1", "run_id": 0},
                {"subject": "s2", "question_id": "q1", "run_id": 1},
            ],
        )
    with pytest.raises(ValueError, match="unique draw_id"):
        expand_question_draws(plan + [dict(plan[0])], [])
    with pytest.raises(ValueError, match="reserved bootstrap field"):
        expand_question_draws(plan, [{"subject": "s1", "question_id": "q1", "draw_id": "x"}])


def test_percentile_interval_exact_threshold_and_linear_percentiles() -> None:
    from part1_bootstrap import percentile_interval

    estimates = [float(index) for index in range(95)] + [None] * 5
    interval = percentile_interval(estimates)

    assert interval == {
        "requested_replicates": 100,
        "valid_replicates": 95,
        "invalid_replicates": 5,
        "valid_fraction": 0.95,
        "confidence_level": 0.95,
        "percentile_method": "linear",
        "lower": pytest.approx(2.35),
        "upper": pytest.approx(91.65),
        "interval_valid": True,
        "interval_reason": None,
        "warning": None,
    }


def test_percentile_interval_suppresses_below_threshold_and_counts_missing_slots() -> None:
    from part1_bootstrap import percentile_interval

    interval = percentile_interval([1.0] * 94 + [None] * 5, requested_replicates=100)

    assert interval["requested_replicates"] == 100
    assert interval["valid_replicates"] == 94
    assert interval["invalid_replicates"] == 6
    assert interval["valid_fraction"] == 0.94
    assert interval["lower"] is None
    assert interval["upper"] is None
    assert interval["interval_valid"] is False
    assert interval["interval_reason"] == "insufficient_valid_bootstrap_replicates"
    assert interval["warning"] == "insufficient_valid_bootstrap_replicates"


def test_percentile_interval_is_json_safe_and_rejects_malformed_valid_values() -> None:
    from part1_bootstrap import percentile_interval

    interval = percentile_interval([1, 2, 3], minimum_valid_fraction=0.0)
    json.dumps(interval, allow_nan=False)
    assert all(type(interval[key]) is int for key in (
        "requested_replicates", "valid_replicates", "invalid_replicates"
    ))

    with pytest.raises(ValueError, match="finite real number or None"):
        percentile_interval([1.0, True])
    with pytest.raises(ValueError, match="finite real number or None"):
        percentile_interval([float("nan")])
