"""Pure compact-bootstrap tests; safe on a login node."""

from __future__ import annotations

import json

import numpy as np
import pytest

from part1_contract import FIXED_SUBJECTS


def _questions(subjects: tuple[str, ...] = ("s1", "s2"), count: int = 3) -> list[dict[str, str]]:
    return [
        {"subject": subject, "question_id": f"{subject}-q{index}"}
        for subject in subjects
        for index in range(count)
    ]


def test_compact_plan_seed_stability_loop_order_multiplicity_and_sensitivity() -> None:
    from part1_bootstrap import QuestionDrawPlan, build_question_draw_plan

    questions = _questions()
    first = build_question_draw_plan(questions, replicates=2, seed=42, small_fixture=True)
    second = build_question_draw_plan(questions, replicates=2, seed=42, small_fixture=True)
    different = build_question_draw_plan(questions, replicates=2, seed=43, small_fixture=True)

    assert isinstance(first, QuestionDrawPlan)
    assert first.subjects == ("s1", "s2")
    assert first.question_ids_by_subject == (
        ("s1-q0", "s1-q1", "s1-q2"),
        ("s2-q0", "s2-q1", "s2-q2"),
    )
    assert first.replicates == 2
    assert first.seed == 42
    assert first.selected_indices.tolist() == [
        [0, 2, 1, 1, 1, 2],
        [0, 2, 0, 0, 1, 2],
    ]
    assert np.array_equal(first.selected_indices, second.selected_indices)
    assert not np.array_equal(first.selected_indices, different.selected_indices)
    assert first.logical_draw_count == 12
    assert first.selected_index_cell_count == 12
    assert first.question_multiplicities(0) == {
        ("s1", "s1-q0"): 1,
        ("s1", "s1-q1"): 1,
        ("s1", "s1-q2"): 1,
        ("s2", "s2-q0"): 0,
        ("s2", "s2-q1"): 2,
        ("s2", "s2-q2"): 1,
    }
    assert [row["question_id"] for row in first.iter_draw_rows(0)] == [
        "s1-q0", "s1-q2", "s1-q1", "s2-q1", "s2-q1", "s2-q2"
    ]
    assert [row["draw_index"] for row in first.iter_draw_rows(0)] == [0, 1, 2, 0, 1, 2]
    assert len({row["draw_id"] for row in first.iter_draw_rows(0)}) == 6


def test_compact_arrays_are_immutable_and_defensively_owned() -> None:
    from part1_bootstrap import question_draw_plan_from_indices

    selected = np.array([[0, 1, 2, 0, 1, 2]], dtype=np.int64)
    plan = question_draw_plan_from_indices(
        _questions(), selected, seed=7, small_fixture=True
    )
    selected[0, 0] = 2
    assert plan.selected_indices[0, 0] == 0
    assert plan.selected_indices.flags.writeable is False
    with pytest.raises(ValueError):
        plan.selected_indices[0, 0] = 1
    with pytest.raises(ValueError):
        plan.selected_indices.setflags(write=True)


def test_default_plan_requires_fixed_subjects_and_rejects_duplicate_questions() -> None:
    from part1_bootstrap import build_question_draw_plan

    production_fixture = [
        {"subject": subject, "question_id": f"{subject}-q0"}
        for subject in FIXED_SUBJECTS
    ]
    plan = build_question_draw_plan(production_fixture, replicates=1)
    assert plan.subjects == tuple(FIXED_SUBJECTS)
    with pytest.raises(ValueError, match="fixed subject presence and order"):
        build_question_draw_plan(_questions(), replicates=1)
    with pytest.raises(ValueError, match="fixed subject presence and order"):
        build_question_draw_plan(list(reversed(production_fixture)), replicates=1)
    questions = _questions(("s1",))
    with pytest.raises(ValueError, match="unique subject/question"):
        build_question_draw_plan(questions + [dict(questions[0])], small_fixture=True)
    cross_subject = [
        {"subject": "s1", "question_id": "q"},
        {"subject": "s2", "question_id": "q"},
    ]
    with pytest.raises(ValueError, match="one subject"):
        build_question_draw_plan(cross_subject, small_fixture=True)


def test_selected_index_constructor_rejects_shape_range_and_controls() -> None:
    from part1_bootstrap import build_question_draw_plan, question_draw_plan_from_indices

    questions = _questions()
    with pytest.raises(ValueError, match="shape"):
        question_draw_plan_from_indices(
            questions, np.zeros((1, 5), dtype=int), small_fixture=True
        )
    invalid = np.zeros((1, 6), dtype=int)
    invalid[0, 3] = 3
    with pytest.raises(ValueError, match="selected index out of range"):
        question_draw_plan_from_indices(questions, invalid, small_fixture=True)
    with pytest.raises(ValueError, match="integer array"):
        question_draw_plan_from_indices(
            questions, np.zeros((1, 6), dtype=float), small_fixture=True
        )
    with pytest.raises(ValueError, match="replicates"):
        build_question_draw_plan(questions, replicates=True, small_fixture=True)


def test_legacy_constructor_enforces_contiguity_order_counts_and_unique_ids() -> None:
    from part1_bootstrap import question_draw_plan_from_rows

    questions = _questions(count=2)
    good_rows = [
        {
            "replicate_id": replicate_id,
            "subject": subject,
            "draw_index": draw_index,
            "draw_id": f"r{replicate_id}-s{subject_index}-d{draw_index}",
            "question_id": f"{subject}-q{draw_index}",
        }
        for replicate_id in range(2)
        for subject_index, subject in enumerate(("s1", "s2"))
        for draw_index in range(2)
    ]
    plan = question_draw_plan_from_rows(
        questions, good_rows, seed=42, small_fixture=True
    )
    assert plan.selected_indices.tolist() == [[0, 1, 0, 1], [0, 1, 0, 1]]

    mutations = []
    duplicate_index = [dict(row) for row in good_rows]
    duplicate_index[1]["draw_index"] = 0
    mutations.append((duplicate_index, "contiguous draw indices"))
    reordered = [dict(row) for row in good_rows]
    reordered[0], reordered[2] = reordered[2], reordered[0]
    mutations.append((reordered, "replicate/subject/draw order"))
    missing = [dict(row) for row in good_rows[:-1]]
    mutations.append((missing, "consistent per-subject draw counts"))
    duplicate_id = [dict(row) for row in good_rows]
    duplicate_id[1]["draw_id"] = duplicate_id[0]["draw_id"]
    mutations.append((duplicate_id, "unique draw_id"))
    noncontiguous_replicate = [dict(row) for row in good_rows]
    for row in noncontiguous_replicate[4:]:
        row["replicate_id"] = 2
    mutations.append((noncontiguous_replicate, "contiguous replicate IDs"))
    for rows, message in mutations:
        with pytest.raises(ValueError, match=message):
            question_draw_plan_from_rows(
                questions, rows, seed=42, small_fixture=True
            )


def test_bounded_audit_materialization_and_expansion_preserve_draw_ids_without_deepcopy() -> None:
    from part1_bootstrap import build_question_draw_plan, expand_question_draws

    plan = build_question_draw_plan(_questions(count=2), replicates=2, small_fixture=True)
    with pytest.raises(ValueError, match="audit materialization limit"):
        plan.materialize_draw_rows(max_rows=7)
    materialized = plan.materialize_draw_rows(max_rows=8)
    assert len(materialized) == 8
    rows = [
        {
            "subject": "s1",
            "question_id": "s1-q0",
            "run_id": run_id,
            "checkpoint_calibration": [{"value": run_id}],
        }
        for run_id in range(2)
    ]
    expanded = expand_question_draws(
        plan, rows, replicate_id=0, max_rows=4
    )
    expected = plan.question_multiplicities(0)[("s1", "s1-q0")] * 2
    assert len(expanded) == expected
    assert all(row["replicate_id"] == 0 for row in expanded)
    assert len({row["draw_id"] for row in expanded}) == expected // 2
    assert expanded[0]["checkpoint_calibration"] is rows[0]["checkpoint_calibration"]


def test_storage_estimate_is_numeric_cells_not_python_draw_dictionaries() -> None:
    from part1_bootstrap import QuestionDrawPlan, build_question_draw_plan

    plan = build_question_draw_plan(_questions(count=4), replicates=100, small_fixture=True)
    assert plan.selected_index_cell_count == 100 * 8
    assert plan.estimated_storage_bytes <= plan.selected_index_cell_count * 4
    assert QuestionDrawPlan.estimated_selected_index_bytes(5_000, 500) <= 10_000_000
    assert not hasattr(plan, "draw_rows")


def test_percentile_interval_threshold_linear_json_and_malformed_values() -> None:
    from part1_bootstrap import percentile_interval

    exact = percentile_interval([float(index) for index in range(95)] + [None] * 5)
    assert exact["lower"] == pytest.approx(2.35)
    assert exact["upper"] == pytest.approx(91.65)
    assert exact["valid_replicates"] == 95
    assert exact["invalid_replicates"] == 5
    assert exact["interval_valid"] is True
    below = percentile_interval([1.0] * 94 + [None] * 5, requested_replicates=100)
    assert below["valid_replicates"] == 94
    assert below["invalid_replicates"] == 6
    assert below["lower"] is None and below["upper"] is None
    assert below["interval_reason"] == "insufficient_valid_bootstrap_replicates"
    json.dumps(exact, allow_nan=False)
    with pytest.raises(ValueError, match="finite real number or None"):
        percentile_interval([True])
    with pytest.raises(ValueError, match="finite real number or None"):
        percentile_interval([float("nan")])
