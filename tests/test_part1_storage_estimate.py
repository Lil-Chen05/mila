"""Output-size safeguard tests."""

from __future__ import annotations

import pytest

from part1_storage_estimate import assess_free_space, estimate_part1_storage


def test_production_estimate_counts_runs_checkpoints_and_array_components() -> None:
    estimate = estimate_part1_storage(
        question_count=500,
        natural_runs_per_question=10,
        checkpoints_per_natural=11,
        expected_generated_tokens=2048,
        expected_decoded_utf8_bytes=12000,
        expected_checkpoint_record_bytes=2400,
    )
    assert estimate["natural_run_count"] == 5000
    assert estimate["checkpoint_count"] == 55000
    assert estimate["decoded_text_bytes"] == 60_000_000
    assert estimate["token_id_array_bytes"] > 0
    assert estimate["entropy_array_bytes"] > 0
    assert estimate["checkpoint_record_bytes"] == 132_000_000
    assert estimate["total_estimated_bytes"] == sum(
        estimate[field]
        for field in (
            "decoded_text_bytes",
            "token_id_array_bytes",
            "entropy_array_bytes",
            "natural_record_overhead_bytes",
            "checkpoint_record_bytes",
        )
    )
    assert estimate["stores_full_vocabulary_logits"] is False


def test_storage_estimate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="question_count"):
        estimate_part1_storage(question_count=-1)
    with pytest.raises(ValueError, match="finite"):
        estimate_part1_storage(json_number_bytes=float("inf"))


def test_free_space_assessment_warns_near_threshold_and_fails_insufficient() -> None:
    estimate = estimate_part1_storage(question_count=1)
    required = estimate["total_estimated_bytes"]
    assert assess_free_space(estimate, free_bytes=required - 1)["status"] == "insufficient"
    near = assess_free_space(estimate, free_bytes=int(required * 1.1), near_multiplier=1.25)
    assert near["status"] == "near_threshold"
    assert near["warning"] is not None
    assert assess_free_space(estimate, free_bytes=required * 2)["status"] == "sufficient"
