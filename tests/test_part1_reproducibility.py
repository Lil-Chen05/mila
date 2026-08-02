"""Pure reproducibility report comparison tests."""

from __future__ import annotations

from pathlib import Path

from part1_reproducibility import compare_natural_results


def result() -> dict:
    return {
        "generated_token_ids": [1, 2],
        "decoded_output": "<think>x</think>\nAnswer: C\nConfidence: 80",
        "natural_answer": "C",
        "raw_parsed_confidence": 80,
        "answer_parse_status": "parsed",
        "confidence_parse_status": "parsed",
        "reasoning_status": "closed",
        "per_token_entropy_nats": [0.1, 0.2],
    }


def test_reproducibility_report_separates_exact_and_tolerance_checks() -> None:
    assert compare_natural_results(result(), result(), entropy_abs_tolerance=0.0) == {
        "exact_generated_token_equality": True,
        "exact_parsed_output_equality": True,
        "entropy_array_equal_within_tolerance": True,
        "entropy_abs_tolerance": 0.0,
    }
    changed = result()
    changed["natural_answer"] = "B"
    changed["decoded_output"] = "<think>x</think>\nAnswer: B\nConfidence: 80"
    changed["per_token_entropy_nats"] = [0.1000001, 0.2]
    report = compare_natural_results(result(), changed, entropy_abs_tolerance=1e-6)
    assert report["exact_generated_token_equality"] is True
    assert report["exact_parsed_output_equality"] is False
    assert report["entropy_array_equal_within_tolerance"] is True


def test_reproducibility_compares_complete_parser_result_not_selected_fields() -> None:
    changed_only_in_raw_parser_field = result()
    changed_only_in_raw_parser_field["decoded_output"] = (
        "<think>x</think>\nAnswer: C\nConfidence: 080"
    )

    report = compare_natural_results(
        result(),
        changed_only_in_raw_parser_field,
        entropy_abs_tolerance=0.0,
    )
    assert changed_only_in_raw_parser_field["raw_parsed_confidence"] == 80
    assert report["exact_parsed_output_equality"] is False


def test_reproducibility_report_publication_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch
) -> None:
    import part1_reproducibility

    synced = []
    monkeypatch.setattr(
        part1_reproducibility,
        "_fsync_directory",
        lambda path: synced.append(Path(path)),
    )
    report_path = tmp_path / "nested" / "report.json"
    part1_reproducibility._atomic_write(report_path, {"status": "synthetic"})

    assert synced == [report_path.parent]
