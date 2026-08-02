"""Guard the separation and evidence scope of Part 1 synthetic scenarios."""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

from part1_contract import (
    attempt_id,
    checkpoint_record_id,
    natural_record_id,
    validate_instance,
)
from part1_store_fixtures import (
    MODEL_RUN_ID,
    QUESTION_ID,
    STUDY_ID,
    checkpoint_result,
    natural_result,
)


REQUIRED_SCENARIOS = {
    "capped_output",
    "missing_close_output",
    "no_reasoning_output",
    "short_reasoning_aliases",
    "malformed_answer",
    "out_of_range_confidence",
    "missing_answer_token",
    "retry_exhaustion",
    "crash_boundaries",
    "stale_locks",
    "duplicate_records",
    "auroc_class_balance",
    "invalid_bootstrap_replicates",
    "bootstrap_draw_multiplicity",
    "macro_replicate_invalidity",
    "within_question_mixed_correctness",
    "switching_across_missing_checkpoints",
}

ANALYSIS_EDGE_CASE_SCENARIOS = {
    "auroc_class_balance",
    "invalid_bootstrap_replicates",
    "bootstrap_draw_multiplicity",
    "macro_replicate_invalidity",
    "within_question_mixed_correctness",
    "switching_across_missing_checkpoints",
}


def _natural_case(
    *, question_index: int, run_id: int, subject: str, correct: bool
) -> dict:
    record = copy.deepcopy(natural_result())
    question_id = f"{question_index + 1:064x}"
    record.update(
        question_id=question_id,
        sample_index=question_index,
        subject=subject,
        run_id=run_id,
        generation_seed=1000 + question_index * 10 + run_id,
        raw_record_id=natural_record_id(STUDY_ID, MODEL_RUN_ID, question_id, run_id),
        terminal_attempt_id=attempt_id(
            STUDY_ID, MODEL_RUN_ID, question_id, run_id, 1
        ),
        natural_answer="C" if correct else "B",
        natural_correct=correct,
        terminal_answer_block_text="Answer: C" if correct else "Answer: B",
    )
    validate_instance("natural_terminal_result", record)
    return record


def _checkpoint_case(*, checkpoint_index: int, forced_answer: str) -> dict:
    record = copy.deepcopy(checkpoint_result())
    checkpoint_id = f"cp-{checkpoint_index:02d}"
    answer_token_id = {"A": 65, "B": 66, "C": 67, "D": 68}[forced_answer]
    record.update(
        checkpoint_id=checkpoint_id,
        checkpoint_record_id=checkpoint_record_id(
            STUDY_ID, MODEL_RUN_ID, QUESTION_ID, 0, checkpoint_id
        ),
        terminal_attempt_id=attempt_id(
            STUDY_ID,
            MODEL_RUN_ID,
            QUESTION_ID,
            0,
            1,
            checkpoint_id=checkpoint_id,
        ),
        requested_checkpoint_index=checkpoint_index,
        requested_fraction=checkpoint_index / 10,
        forced_answer=forced_answer,
        terminal_answer_block_text=f"Answer: {forced_answer}\nConfidence: 80",
        checkpoint_local_correct=forced_answer == "C",
        answer_token_id=answer_token_id,
        agrees_with_natural_answer=forced_answer == "C",
    )
    validate_instance("checkpoint_terminal_result", record)
    return record


def test_synthetic_fixture_catalog_is_complete_and_cannot_claim_real_model_evidence() -> None:
    path = Path(__file__).parent / "fixtures" / "part1_synthetic" / "catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))

    assert catalog["catalog_version"] == "part1-synthetic-evidence-v1"
    assert catalog["real_model_evidence"] is False
    scenarios = catalog["scenarios"]
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    assert len(by_id) == len(scenarios)
    assert set(by_id) == REQUIRED_SCENARIOS

    for scenario in scenarios:
        assert scenario["status"] == "implemented"
        assert scenario["claim"].strip()
        assert scenario["evidence"].startswith("tests/test_")


def test_six_analysis_edge_case_families_are_cataloged_as_synthetic_raw_inputs() -> None:
    path = Path(__file__).parent / "fixtures" / "part1_synthetic" / "catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    by_id = {scenario["id"]: scenario for scenario in catalog["scenarios"]}

    for scenario_id in ANALYSIS_EDGE_CASE_SCENARIOS:
        assert by_id[scenario_id]["status"] == "implemented"
        assert by_id[scenario_id]["evidence"].startswith(
            "tests/test_part1_synthetic_fixture_catalog.py::test_"
        )
        assert "synthetic" in by_id[scenario_id]["claim"].lower()


def test_auroc_class_balance_fixture_has_schema_valid_both_and_one_class_inputs() -> None:
    both_class = [
        _natural_case(
            question_index=index,
            run_id=0,
            subject="high_school_mathematics",
            correct=correct,
        )
        for index, correct in enumerate((True, False))
    ]
    one_class = [
        _natural_case(
            question_index=index + 2,
            run_id=0,
            subject="high_school_mathematics",
            correct=True,
        )
        for index in range(2)
    ]

    assert {record["natural_correct"] for record in both_class} == {True, False}
    assert {record["natural_correct"] for record in one_class} == {True}


def test_invalid_bootstrap_replicate_fixture_is_schema_valid_and_one_class() -> None:
    raw_inputs = [
        _natural_case(
            question_index=index,
            run_id=0,
            subject="high_school_physics",
            correct=index < 2,
        )
        for index in range(3)
    ]
    synthetic_draw = [raw_inputs[0], raw_inputs[1], raw_inputs[0]]

    assert {record["natural_correct"] for record in raw_inputs} == {True, False}
    assert {record["natural_correct"] for record in synthetic_draw} == {True}


def test_bootstrap_draw_multiplicity_fixture_preserves_repeated_question_ids() -> None:
    first = _natural_case(
        question_index=0,
        run_id=0,
        subject="high_school_chemistry",
        correct=True,
    )
    second = _natural_case(
        question_index=1,
        run_id=0,
        subject="high_school_chemistry",
        correct=False,
    )
    synthetic_draw = [first, first, second, first]

    multiplicity = Counter(record["question_id"] for record in synthetic_draw)
    assert multiplicity[first["question_id"]] == 3
    assert multiplicity[second["question_id"]] == 1


def test_macro_replicate_fixture_has_one_invalid_subject_among_valid_subjects() -> None:
    subjects = (
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "high_school_psychology",
    )
    records = [
        _natural_case(
            question_index=subject_index * 2 + correctness_index,
            run_id=0,
            subject=subject,
            correct=(True if subject_index == 4 else correctness_index == 0),
        )
        for subject_index, subject in enumerate(subjects)
        for correctness_index in range(2)
    ]
    classes_by_subject = {
        subject: {
            record["natural_correct"]
            for record in records
            if record["subject"] == subject
        }
        for subject in subjects
    }

    assert classes_by_subject["high_school_psychology"] == {True}
    assert all(
        classes_by_subject[subject] == {True, False}
        for subject in subjects[:-1]
    )


def test_within_question_fixture_has_mixed_correctness_across_schema_valid_runs() -> None:
    records = [
        _natural_case(
            question_index=0,
            run_id=run_id,
            subject="high_school_biology",
            correct=correct,
        )
        for run_id, correct in enumerate((True, False, True))
    ]

    assert len({record["question_id"] for record in records}) == 1
    assert {record["run_id"] for record in records} == {0, 1, 2}
    assert {record["natural_correct"] for record in records} == {True, False}


def test_switching_fixture_has_answer_changes_across_missing_schema_valid_checkpoints() -> None:
    records = [
        _checkpoint_case(checkpoint_index=index, forced_answer=answer)
        for index, answer in ((0, "A"), (3, "B"), (10, "A"))
    ]
    observed_ids = {record["checkpoint_id"] for record in records}

    assert observed_ids < {f"cp-{index:02d}" for index in range(11)}
    assert [record["forced_answer"] for record in records] == ["A", "B", "A"]
